"""Hermetic GetOrganelle service tests using a synthetic environment.

Tests cover success, non-zero exit, missing output, malformed FASTA/GFA,
pre-start rejection, failure stage recording, evidence hashes, atomic
publication, reuse, and tamper rejection.  No network or real subprocess
execution is required.
"""

from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Mapping
from pathlib import Path

import pytest

from organelleverse.assembly.contracts import (
    AssemblyRequest,
    GetOrganelleParameters,
    effective_backend_parameters,
)
from organelleverse.assembly.environment_specs import (
    GETORGANELLE_ENVIRONMENT,
    ManagedDatabaseSpec,
)
from organelleverse.assembly.environments import (
    EnvironmentManager,
)
from organelleverse.assembly.execution import CommandOutcome, CommandRunner
from organelleverse.assembly.manifests import (
    AssemblyRunManifest,
)
from organelleverse.assembly.service import execute_assembly
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import (
    OrganelleExecutionError,
)
from organelleverse.runtime import managed_run_path

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _read_file(tmp_path: Path, name: str = "reads.fastq") -> Path:
    path = tmp_path / "inputs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"@r1\nACGTACGT\n+\nIIIIIIII\n")
    return path


def _pe_data(read1_path: Path, read2_path: Path) -> OrganelleData:
    r1 = ArtifactRef.from_path(
        read1_path, kind="short_read", format="fastq", media_type="application/x-fastq"
    )
    r2 = ArtifactRef.from_path(
        read2_path, kind="short_read", format="fastq", media_type="application/x-fastq"
    )
    return OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {"read1": r1, "read2": r2},
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "short_libraries": [
                    {
                        "technology": "illumina",
                        "layout": "paired_end",
                        "read1_artifact": "read1",
                        "read2_artifact": "read2",
                        "read_length": 150,
                    }
                ],
            },
        }
    )


def _request(
    tmp_path: Path,
    *,
    organelle: str = "plastid",
    read1_path: Path | None = None,
    read2_path: Path | None = None,
) -> AssemblyRequest:
    if read1_path is None:
        read1_path = _read_file(tmp_path, "reads_1.fastq")
    if read2_path is None:
        read2_path = _read_file(tmp_path, "reads_2.fastq")
    return AssemblyRequest(
        data=_pe_data(read1_path, read2_path),
        organelle=organelle,  # type: ignore[arg-type]
        method="getorganelle",  # type: ignore[arg-type]
        threads=2,
    )


def _expected_destination(request: AssemblyRequest) -> Path:
    backend = request.method if request.method != "auto" else "getorganelle"
    return managed_run_path(
        "assembly.assemble", f"sha256-{request.resolved_semantic_hash(backend)}"
    )


# ---------------------------------------------------------------------------
# Synthetic environment manager (network-free)
# ---------------------------------------------------------------------------


class _RecordingCarrier:
    def __init__(self) -> None:
        self.create_count = 0

    def create_prefix(self, explicit_lock: str, destination: Path) -> None:
        self.create_count += 1
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "bin").mkdir(parents=True, exist_ok=True)
        for name in (
            "get_organelle_from_reads.py",
            "get_organelle_config.py",
            "blastn",
            "bowtie2",
            "spades.py",
        ):
            exe = destination / "bin" / name
            if name == "get_organelle_config.py":
                exe.write_text(
                    "#!/usr/bin/env python3\n"
                    "import pathlib, sys\n"
                    "targets = sys.argv[sys.argv.index('-a') + 1].split(',')\n"
                    "root = pathlib.Path(sys.argv[sys.argv.index('--config-dir') + 1])\n"
                    "for directory in ('SeedDatabase', 'LabelDatabase'):\n"
                    "    path = root / directory\n"
                    "    path.mkdir(parents=True, exist_ok=True)\n"
                    "    for target in targets:\n"
                    "        (path / f'{target}.fasta').write_text('>seed\\nACGT\\n')\n"
                )
            else:
                exe.write_text("#!/bin/sh\necho ok\n")
            exe.chmod(0o755)
        meta = destination / "conda-meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "getorganelle-1.7.7.1-pyhdfd78af_0.json").write_text(
            json.dumps({"name": "getorganelle", "version": "1.7.7.1", "build": "pyhdfd78af_0"})
        )

    def probe_version(self, executable: Path, argv: tuple[str, ...]) -> str:
        return "GetOrganelle v1.7.7.1"

    def probe_package_identity(self, prefix: Path) -> tuple[tuple[str, str, str], ...]:
        return (("getorganelle", "1.7.7.1", "pyhdfd78af_0"),)


class _SyntheticEnvironmentManager(EnvironmentManager):
    def materialize_database_files(self, database: ManagedDatabaseSpec) -> Path:
        root = self.cache_root / "synthetic-getorganelledb"
        root.mkdir(parents=True, exist_ok=True)
        archive_path = root / database.files[0].name
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            content = b">embplant_pt\nACGT\n"
            info = tarfile.TarInfo(
                f"GetOrganelleDB-{database.source_commit}/{database.version}/"
                "SeedDatabase/embplant_pt.fasta"
            )
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        archive_path.write_bytes(stream.getvalue())
        return root


def _synthetic_environment_manager(cache_root: Path) -> EnvironmentManager:
    return _SyntheticEnvironmentManager(
        cache_root=cache_root,
        carrier=_RecordingCarrier(),
    )


# ---------------------------------------------------------------------------
# Synthetic command runner (creates GetOrganelle-like output)
# ---------------------------------------------------------------------------


class _SuccessRunner(CommandRunner):
    """Creates valid GetOrganelle output without actually running the tool."""

    def __init__(
        self,
        *,
        fasta_text: str = ">seq1 circular\nACGTACGTACGT\n",
        gfa_text: str = "S\tseq1\tACGTACGTACGT\nL\tseq1\t+\tseq1\t+\t0M\n",
    ) -> None:
        super().__init__()
        self.calls: list[tuple[str, ...]] = []
        self.fasta_text = fasta_text
        self.gfa_text = gfa_text

    def run(
        self,
        argv: tuple[str, ...],
        *,
        stage: str,
        cwd: Path,
        timeout_seconds: float,
        stdout_path: Path,
        stderr_path: Path,
        env: Mapping[str, str] | None = None,
        stdin_path: Path | None = None,
    ) -> CommandOutcome:
        self.calls.append(argv)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("")
        stderr_path.write_text("")
        # Parse output dir from argv
        argv_list = list(argv)
        try:
            o_idx = argv_list.index("-o")
            output_dir = Path(argv_list[o_idx + 1])
        except (ValueError, IndexError):
            output_dir = cwd / "output"

        # Create simulated GetOrganelle output
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "embplant_pt.K115.path_sequence.fasta").write_text(self.fasta_text)
        (output_dir / "embplant_pt.K115.selected_graph.gfa").write_text(self.gfa_text)
        # Write log
        (output_dir / "get_org.log.txt").write_text("GetOrganelle log\n")

        from datetime import UTC, datetime

        now = datetime.now(UTC)
        started = datetime.fromtimestamp(now.timestamp() - 1.0, tz=UTC)
        return CommandOutcome(
            stage=stage,
            argv=argv,
            cwd=str(cwd),
            started=True,
            started_at=started,
            finished_at=now,
            duration_seconds=1.0,
            termination="exit",
            returncode=0,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )


class _FailureRunner(CommandRunner):
    """Returns non-zero exit code."""

    def run(
        self,
        argv: tuple[str, ...],
        *,
        stage: str,
        cwd: Path,
        timeout_seconds: float,
        stdout_path: Path,
        stderr_path: Path,
        env: Mapping[str, str] | None = None,
        stdin_path: Path | None = None,
    ) -> CommandOutcome:
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        started = datetime.fromtimestamp(now.timestamp() - 0.1, tz=UTC)
        return CommandOutcome(
            stage=stage,
            argv=argv,
            cwd=str(cwd),
            started=True,
            started_at=started,
            finished_at=now,
            duration_seconds=0.1,
            termination="exit",
            returncode=1,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetOrganelleServiceSuccess:
    def test_success_path_produces_ok_result(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"

        env_manager = _synthetic_environment_manager(cache)
        runner = _SuccessRunner()
        request = _request(tmp_path)
        output_dir = _expected_destination(request)

        result = execute_assembly(
            request,
            environment_manager=env_manager,
            runner=runner,
        )
        assert result.status == "ok"
        assert result.scope == "plastid"
        assert result.provenance is not None
        assert result.provenance.actual_backend == "getorganelle"

        # Check evidence files exist
        assert (output_dir / "result.json").is_file()
        assert (output_dir / "assembly_run_manifest.json").is_file()
        # Normalized outputs are inside workspace-then-published dirs
        assert any(f.suffix == ".fasta" for f in output_dir.rglob("*") if f.is_file()), (
            f"no FASTA found in {list(output_dir.iterdir())}"
        )
        assert any(f.suffix == ".gfa" for f in output_dir.rglob("*") if f.is_file()), (
            f"no GFA found in {list(output_dir.iterdir())}"
        )

    def test_result_contains_manifest_and_observations(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"

        env_manager = _synthetic_environment_manager(cache)
        runner = _SuccessRunner()
        request = _request(tmp_path)
        output_dir = _expected_destination(request)

        result = execute_assembly(
            request,
            environment_manager=env_manager,
            runner=runner,
        )
        assert result.status == "ok"

        manifest_path = output_dir / "assembly_run_record.json"
        assert manifest_path.is_file()
        manifest = AssemblyRunManifest.model_validate_json(manifest_path.read_bytes())
        assert manifest.selected_backend == "getorganelle"
        assert manifest.organelle == "plastid"
        assert len(manifest.outputs) > 0

    def test_manifest_records_exact_parameters_and_hashes(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        env_manager = _synthetic_environment_manager(cache)
        parameters = GetOrganelleParameters(max_reads=42, word_size=21.0)
        request = _request(tmp_path).model_copy(update={"backend_parameters": parameters})
        output_dir = _expected_destination(request)

        result = execute_assembly(
            request,
            environment_manager=env_manager,
            runner=_SuccessRunner(),
        )

        assert result.status == "ok"
        assert result.provenance is not None
        manifest = AssemblyRunManifest.model_validate_json(
            (output_dir / "assembly_run_record.json").read_bytes()
        )
        expected_parameters = effective_backend_parameters(
            "getorganelle",
            parameters,
            payload=None,
            organelle="plastid",
            taxon_group="plant",
        )
        assert dict(manifest.parameters.backend_parameters.items()) == expected_parameters
        manifest_inputs = {item.role: item.artifact for item in manifest.input_artifacts}
        assert set(manifest_inputs) == {"getorganelledb_config", "read1", "read2"}
        for role, artifact in request.data.artifacts.items():
            assert manifest_inputs[role].sha256 == artifact.sha256
        assert len(manifest_inputs["getorganelledb_config"].sha256) == 64
        assert manifest.environment.digest == env_manager.expected_environment_digest(
            GETORGANELLE_ENVIRONMENT
        )
        assert result.provenance.input_artifact_hashes == tuple(
            sorted(artifact.sha256 for artifact in request.data.artifacts.values())
        )
        assert dict(result.provenance.database_hashes.items()) == {
            "getorganelledb_archive": (
                "a522e42f7127d38bb2ce3eb9224213f5997fdd12aae1fa782c57b29ed3754315"
            ),
            "getorganelledb_identity": (
                "e09e134e7f0b01ef72eec5a35a549cc154678026de73e3ef388d3be12d621526"
            ),
        }

    def test_reuse_publishes_consistent_evidence(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"

        env_manager = _synthetic_environment_manager(cache)
        runner = _SuccessRunner()
        request = _request(tmp_path)
        output_dir = _expected_destination(request)

        result1 = execute_assembly(
            request,
            environment_manager=env_manager,
            runner=runner,
        )
        assert result1.status == "ok"

        result2 = execute_assembly(
            request,
            environment_manager=env_manager,
            runner=runner,
        )
        assert result2.object_id == result1.object_id
        assert len(runner.calls) == 1

        # Verify output has key evidence files
        assert (output_dir / "result.json").is_file()
        assert (output_dir / "assembly_run_manifest.json").is_file()

    def test_tampered_artifact_rejected(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"

        env_manager = _synthetic_environment_manager(cache)
        runner = _SuccessRunner()
        read1 = _read_file(tmp_path, "reads_1.fastq")
        read2 = _read_file(tmp_path, "reads_2.fastq")

        # Create data with artifact hashes from current file contents
        data = _pe_data(read1, read2)
        request = AssemblyRequest(
            data=data,
            organelle="plastid",  # type: ignore[arg-type]
            method="getorganelle",  # type: ignore[arg-type]
            threads=2,
        )

        # Tamper with a file AFTER the artifact hash was captured
        read1.write_bytes(b"@tampered\nGGGG\n+\nIIII\n")

        # Execution should detect the hash mismatch
        with pytest.raises(OrganelleExecutionError, match="hash"):
            execute_assembly(
                request,
                environment_manager=env_manager,
                runner=runner,
            )


class TestGetOrganelleServiceFailure:
    def test_nonzero_exit_produces_failed_result(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"

        env_manager = _synthetic_environment_manager(cache)
        runner = _FailureRunner()
        request = _request(tmp_path)

        result = execute_assembly(
            request,
            environment_manager=env_manager,
            runner=runner,
        )

        assert result.status == "failed"
        assert len(result.errors) > 0
        assert result.errors[0].code == "assembly.execution_failed"
        assert result.errors[0].suggested_action["failed_stage"] == "execute_backend"

    def test_failed_result_published_with_evidence(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"

        env_manager = _synthetic_environment_manager(cache)
        runner = _FailureRunner()
        request = _request(tmp_path)
        output_dir = _expected_destination(request)

        result = execute_assembly(
            request,
            environment_manager=env_manager,
            runner=runner,
        )
        assert result.status == "failed"

        # Failed results should still publish evidence
        assert (output_dir / "result.json").is_file()
        assert (output_dir / "assembly_run_manifest.json").is_file()

    def test_missing_output_after_ok_exit_is_recorded(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"

        class _EmptyRunner(CommandRunner):
            def run(
                self,
                argv: tuple[str, ...],
                *,
                stage: str,
                cwd: Path,
                timeout_seconds: float,
                stdout_path: Path,
                stderr_path: Path,
                env: Mapping[str, str] | None = None,
                stdin_path: Path | None = None,
            ) -> CommandOutcome:
                from datetime import UTC, datetime

                now = datetime.now(UTC)
                started = datetime.fromtimestamp(now.timestamp() - 0.1, tz=UTC)
                return CommandOutcome(
                    stage=stage,
                    argv=argv,
                    cwd=str(cwd),
                    started=True,
                    started_at=started,
                    finished_at=now,
                    duration_seconds=0.1,
                    termination="exit",
                    returncode=0,
                    stdout_path=str(stdout_path),
                    stderr_path=str(stderr_path),
                )

        env_manager = _synthetic_environment_manager(cache)
        request = _request(tmp_path)

        result = execute_assembly(
            request,
            environment_manager=env_manager,
            runner=_EmptyRunner(),
        )

        assert result.status == "failed"
        assert any(e.code == "assembly.output_incomplete" for e in result.errors)

    @pytest.mark.parametrize(
        ("fasta_text", "gfa_text"),
        [
            ("ACGT\n", "S\tseq1\tACGTACGTACGT\n"),
            (">seq1\nACGTACGTACGT\n", "not-gfa\n"),
        ],
    )
    def test_malformed_outputs_are_recorded_at_normalization(
        self,
        tmp_path: Path,
        fasta_text: str,
        gfa_text: str,
    ) -> None:
        result = execute_assembly(
            _request(tmp_path),
            environment_manager=_synthetic_environment_manager(tmp_path / "cache"),
            runner=_SuccessRunner(fasta_text=fasta_text, gfa_text=gfa_text),
        )

        assert result.status == "failed"
        assert result.errors[0].suggested_action["failed_stage"] == "normalize_outputs"

    def test_pre_start_rejection_raises_before_side_effects(self, tmp_path: Path) -> None:
        # Request with wrong parameter discriminator should raise pre-start
        from organelleverse.assembly.contracts import OatkParameters

        cache = tmp_path / "cache"
        env_manager = _synthetic_environment_manager(cache)

        request = _request(tmp_path)

        output_dir = _expected_destination(request)
        request = request.model_copy(
            update={"method": "auto", "backend_parameters": OatkParameters()}
        )

        # When routing resolves to getorganelle, the backend_parameters mismatch
        # will be caught before any execution
        runner = _SuccessRunner()
        with pytest.raises((OrganelleExecutionError, ValueError)):
            execute_assembly(
                request,
                environment_manager=env_manager,
                runner=runner,
            )
        assert runner.calls == []
        assert not output_dir.exists()


class TestGetOrganelleServiceCatalog:
    def test_getorganelle_in_released_runtimes(self) -> None:
        from organelleverse.assembly.backends.runtime import RUNTIMES

        assert "getorganelle" in RUNTIMES
        runtime = RUNTIMES["getorganelle"]
        assert runtime.backend_id == "getorganelle"
        assert runtime.environment_spec.backend_id == "getorganelle"

    def test_pmat_is_released(self) -> None:
        from organelleverse.assembly.backends.runtime import RUNTIMES

        assert "pmat" in RUNTIMES
        from organelleverse.assembly.backends.runtime import pmat_runtime

        pmat = pmat_runtime()
        assert pmat is RUNTIMES["pmat"]

    def test_released_backend_parameters_include_getorganelle(self) -> None:

        # Check that GetOrganelleParameters is part of the released union
        go = GetOrganelleParameters()
        assert go.backend == "getorganelle"


class TestGetOrganelleServiceAtomicPublication:
    def test_publication_is_atomic(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"

        env_manager = _synthetic_environment_manager(cache)
        runner = _SuccessRunner()
        request = _request(tmp_path)
        output_dir = _expected_destination(request)

        result = execute_assembly(
            request,
            environment_manager=env_manager,
            runner=runner,
        )

        assert result.status == "ok"
        # output_dir should exist with all evidence
        assert output_dir.is_dir()
        # Verify no .tmp sibling remains
        siblings = list(output_dir.parent.glob("*.tmp*"))
        assert len(siblings) == 0
