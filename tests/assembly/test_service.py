from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import pytest

import organelleverse.assembly.service as assembly_service
from organelleverse.assembly.api import write as write_assembly
from organelleverse.assembly.backends import (
    AdapterContext,
    AssemblyCommand,
    BackendOutput,
    BackendRuntime,
    ExpectedBackendResources,
    NormalizedAssemblyOutputs,
    PreparedBackendResources,
    RawAssemblyOutputs,
)
from organelleverse.assembly.backends.runtime import (
    AssemblyResourceProvider,
    AssemblyRuntimeRegistry,
)
from organelleverse.assembly.contracts import (
    AssemblyInputPayload,
    AssemblyRequest,
    HimtParameters,
)
from organelleverse.assembly.data_contract import validate_oatk_assembly_data
from organelleverse.assembly.environment_specs import (
    HIMT_ENVIRONMENT,
    AssemblyEnvironmentSpec,
    CondaPlatformSpec,
)
from organelleverse.assembly.environments import (
    EnvironmentManager,
    PreparedEnvironment,
    PreparedExecutable,
    PreparedProfile,
)
from organelleverse.assembly.execution import CommandOutcome, CommandRunner
from organelleverse.assembly.manifests import (
    AssemblyComponentIdentity,
    AssemblyRunManifest,
    AssemblyRunObservations,
    ManifestArtifact,
    assembly_run_id_from_artifact,
)
from organelleverse.assembly.service import execute_assembly
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import (
    OrganelleDependencyError,
    OrganelleExecutionError,
    OrganelleInputError,
)
from organelleverse.core.serialization import load_genome, load_result, save_contract
from organelleverse.runtime import managed_run_path, managed_runs_root

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _read_file(tmp_path: Path, name: str = "reads.fastq") -> Path:
    path = tmp_path / "inputs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"@r1\nACGTACGT\n+\nIIIIIIII\n")
    return path


def _profile_file(tmp_path: Path) -> Path:
    path = tmp_path / "profiles" / "embryophyta_mito.fam"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"profile-content")
    return path


def _load_run_record(output_dir: Path) -> AssemblyRunManifest:
    return AssemblyRunManifest.model_validate_json(
        (output_dir / "assembly_run_record.json").read_bytes()
    )


def _expected_destination(request: AssemblyRequest) -> Path:
    """The managed run path execute_assembly derives for one request."""
    backend = request.method if request.method != "auto" else "oatk"
    return managed_run_path(
        "assembly.assemble", f"sha256-{request.resolved_semantic_hash(backend)}"
    )


def _no_assembly_run_committed() -> bool:
    """True when no managed assembly run directory has been published."""
    root = managed_runs_root() / "assembly.assemble"
    return not root.exists() or not any(root.iterdir())


def _hifi_data(read_path: Path, profile_path: Path) -> OrganelleData:
    read_artifact = ArtifactRef.from_path(
        read_path, kind="long_read", format="fastq", media_type="application/x-fastq"
    )
    profile_artifact = ArtifactRef.from_path(
        profile_path, kind="hmm_profile", format="fam", media_type="text/plain"
    )
    return OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {"long_0_reads": read_artifact, "hmm_profiles": profile_artifact},
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "long_libraries": [
                    {
                        "technology": "pacbio_hifi",
                        "quality_state": "ccs",
                        "reads_artifact": "long_0_reads",
                    }
                ],
                "auxiliary": {"hmm_profiles_artifact": "hmm_profiles"},
            },
        }
    )


def _request(
    tmp_path: Path,
    *,
    organelle: str = "mitochondrion",
    read_path: Path | None = None,
    profile_path: Path | None = None,
    environment_source: str = "auto",
    method: str = "oatk",
) -> AssemblyRequest:
    if read_path is None:
        read_path = _read_file(tmp_path)
    if profile_path is None:
        profile_path = _profile_file(tmp_path)
    return AssemblyRequest(
        data=_hifi_data(read_path, profile_path),
        organelle=organelle,  # type: ignore[arg-type]
        method=method,  # type: ignore[arg-type]
        environment_source=environment_source,  # type: ignore[arg-type]
        threads=2,
    )


def _fake_environment_manager(cache_root: Path) -> EnvironmentManager:
    """Return a deterministic network-free environment manager."""
    return _SyntheticEnvManager(cache_root)


class _SyntheticEnvManager(EnvironmentManager):
    """EnvironmentManager backed by a synthetic spec whose profile hashes match a
    deterministic downloader, so prepare/prepare_profile work without network."""

    def __init__(self, cache_root: Path) -> None:
        self._carrier = _RecordingCarrier()
        self._downloader = _SyntheticDownloader()
        self._profile_compiler = _SyntheticProfileCompiler()
        super().__init__(
            cache_root=cache_root,
            carrier=self._carrier,
            downloader=self._downloader,
            profile_compiler=self._profile_compiler,
        )

    def prepare(self, spec, *, policy, platform=None):  # type: ignore[override]
        return super().prepare(_SYNTHETIC_SPEC, policy=policy, platform=platform or "linux-64")

    def prepare_profile(
        self,
        spec: AssemblyEnvironmentSpec,
        *,
        organelle: Literal["mitochondrion", "plastid"],
        policy: Literal["ensure", "require"],
        override: ArtifactRef | None = None,
        environment: PreparedEnvironment | None = None,
    ) -> PreparedProfile:
        return super().prepare_profile(
            _SYNTHETIC_SPEC,
            organelle=organelle,
            policy=policy,
            override=override,
            environment=environment,
        )

    def expected_environment_digest(self, spec, *, platform=None):  # type: ignore[override]
        return super().expected_environment_digest(_SYNTHETIC_SPEC, platform=platform or "linux-64")

    def expected_profile_identity(self, spec, *, organelle, override=None):  # type: ignore[override]
        return super().expected_profile_identity(
            _SYNTHETIC_SPEC, organelle=organelle, override=override
        )


class _RecordingCarrier:
    def __init__(self) -> None:
        self.create_count = 0

    def create_prefix(self, explicit_lock: str, destination: Path) -> None:
        self.create_count += 1
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "bin").mkdir(parents=True, exist_ok=True)
        for name in ("oatk", "nhmmscan", "hmmpress"):
            exe = destination / "bin" / name
            exe.write_text("#!/bin/sh\necho ok\n")
            exe.chmod(0o755)
        (destination / "oatk.version").write_text("oatk 1.0")
        meta = destination / "conda-meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "oatk-1.0-h577a1d6_1.json").write_text(
            json.dumps({"name": "oatk", "version": "1.0", "build": "h577a1d6_1"})
        )

    def probe_version(self, executable: Path, argv: tuple[str, ...]) -> str:
        return (executable.parent.parent / "oatk.version").read_text()

    def probe_package_identity(self, prefix: Path) -> tuple[tuple[str, str, str], ...]:
        return (("oatk", "1.0", "h577a1d6_1"),)


class _SyntheticDownloader:
    def download(self, url: str, destination: Path, expected_sha256: str) -> bytes:
        content = (url.encode("utf-8") + b"\n") * 16
        destination.write_bytes(content)
        return content


class _SyntheticProfileCompiler:
    def press(self, executable: Path, profile: Path) -> None:
        for suffix in ("h3f", "h3i", "h3m", "h3p"):
            profile.with_name(f"{profile.name}.{suffix}").write_bytes(
                f"compiled-{suffix}\n".encode()
            )


def _synthetic_spec():
    from organelleverse.assembly.environment_specs import (
        AssemblyEnvironmentSpec,
        CondaPlatformSpec,
        ManagedDatabaseSpec,
        ManagedFileSpec,
    )

    base = "https://example.org/oatkdb/v1/"
    names = (
        "embryophyta_mito.fam",
        "embryophyta_mito.fam.h3f",
        "embryophyta_mito.fam.h3i",
        "embryophyta_mito.fam.h3m",
        "embryophyta_mito.fam.h3p",
        "embryophyta_pltd.fam",
        "embryophyta_pltd.fam.h3f",
        "embryophyta_pltd.fam.h3i",
        "embryophyta_pltd.fam.h3m",
        "embryophyta_pltd.fam.h3p",
    )

    def _file(name: str) -> ManagedFileSpec:
        content = (base + name).encode("utf-8") + b"\n"
        content = content * 16
        return ManagedFileSpec(
            name=name,
            url=base + name,
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        )

    files = tuple(_file(n) for n in names)
    return AssemblyEnvironmentSpec(
        backend_id="oatk",
        contract_version="synthetic.v1",
        platforms=(
            CondaPlatformSpec(
                platform="linux-64",
                package="oatk=1.0=h577a1d6_1",
                lock_resource="organelleverse.assembly.resources.environments.oatk/linux-64.explicit.txt",
                executable_names=("oatk", "nhmmscan", "hmmpress"),
                version_argv=("--version",),
            ),
        ),
        database=ManagedDatabaseSpec(
            database_id="oatkdb",
            version="v20230921",
            source_commit="75e8db0ac4a7d508a9a518d900876003ceb70737",
            license="MIT",
            files=files,
        ),
    )


_SYNTHETIC_SPEC = _synthetic_spec()


class ForbiddenRunner(CommandRunner):
    """A runner that fails the test if Oatk is actually started."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, ...]] = []

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
        raise AssertionError("ForbiddenRunner was invoked; reuse should have avoided Oatk")


def test_execution_environment_activates_managed_prefix_and_blocks_python_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "managed-prefix"
    monkeypatch.setenv("CONDA_PREFIX", "/host/conda")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "host")
    monkeypatch.setenv("VIRTUAL_ENV", "/host/venv")
    monkeypatch.setenv("PYTHONHOME", "/host/python")
    monkeypatch.setenv("PYTHONPATH", "/host/modules")
    environment = PreparedEnvironment(
        backend_id="himt",
        carrier="conda",
        platform="linux-64",
        digest="sha256:" + "a" * 64,
        prefix=prefix,
        executables=(),
        version="HiMT 1.1.3",
    )

    resolved = assembly_service._execution_environment(  # pyright: ignore[reportPrivateUsage]
        environment
    )

    assert resolved["PATH"].split(os.pathsep)[0] == str(prefix / "bin")
    assert resolved["CONDA_PREFIX"] == str(prefix)
    assert resolved["CONDA_DEFAULT_ENV"] == str(prefix)
    assert resolved["CONDA_SHLVL"] == "1"
    assert "VIRTUAL_ENV" not in resolved
    assert "PYTHONHOME" not in resolved
    assert "PYTHONPATH" not in resolved


def test_execution_environment_strips_conda_sysconfigdata_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The execution environment must not leak an active outer conda
    # environment's sysconfigdata identity into the managed environment.
    monkeypatch.setenv(
        "_CONDA_PYTHON_SYSCONFIGDATA_NAME", "_sysconfigdata_x86_64_conda_cos7_linux_gnu"
    )
    monkeypatch.setenv("_PYTHON_SYSCONFIGDATA_NAME", "_sysconfigdata_x86_64_conda_cos7_linux_gnu")
    environment = PreparedEnvironment(
        backend_id="himt",
        carrier="conda",
        platform="linux-64",
        digest="sha256:" + "a" * 64,
        prefix=tmp_path / "himt-prefix",
        executables=(),
        version="HiMT 1.1.3",
    )

    resolved = assembly_service._execution_environment(  # pyright: ignore[reportPrivateUsage]
        environment
    )

    assert "_CONDA_PYTHON_SYSCONFIGDATA_NAME" not in resolved
    assert "_PYTHON_SYSCONFIGDATA_NAME" not in resolved


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


def test_success_publishes_a_loadable_acyclic_evidence_chain(tmp_path: Path) -> None:
    read_path = _read_file(tmp_path)
    request = _request(tmp_path, read_path=read_path)
    result = execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_successful_fake_oatk_runner(tmp_path),
    )

    assert result.status == "ok"
    assert result.provenance is not None
    assert result.provenance.requested_backend == "oatk"
    assert result.provenance.actual_backend == "oatk"
    roles = {artifact.kind: artifact for artifact in result.artifacts}
    assert "assembly_run_manifest" in roles
    genome = load_genome(_expected_destination(request) / "primary_genome.json")
    restored = load_result(_expected_destination(request) / "result.json")
    assert restored == result
    assert genome.sequence is not None
    assert genome.source_manifests[0].sha256 == roles["assembly_run_manifest"].sha256
    manifest = _load_run_record(_expected_destination(request))
    canonical_path = _expected_destination(request) / "assembly_run_manifest.json"
    assert canonical_path.read_bytes() == manifest.canonical_bytes()
    assert genome.source_manifests == (roles["assembly_run_manifest"],)
    assert genome.source_manifests[0].resolve() == canonical_path.resolve()
    assert assembly_run_id_from_artifact(genome.source_manifests[0]) == manifest.run_manifest_id
    assert result.provenance.run_manifest_id == manifest.run_manifest_id
    assert "assembly_run_record" in roles
    # The validated input payload is manifest-authoritative for QC.
    assert manifest.input_payload is not None
    (long_library,) = manifest.input_payload.long_libraries
    assert long_library.technology == "pacbio_hifi"
    assert long_library.quality_state == "ccs"
    assert long_library.reads_artifact == "long_0_reads"
    assert manifest.input_payload.auxiliary.hmm_profiles_artifact == "hmm_profiles"
    # Managed HMM resource is a declared manifest input but not a read library.
    assert "hmm_profiles" in {item.role for item in manifest.input_artifacts}


def test_success_artifacts_resolve_after_atomic_publication(tmp_path: Path) -> None:
    request = _request(tmp_path)

    result = execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_successful_fake_oatk_runner(tmp_path),
    )

    assert result.artifacts
    assert all(artifact.resolve().is_file() for artifact in result.artifacts)
    genome = load_genome(_expected_destination(request) / "primary_genome.json")
    assert genome.sequence is not None
    assert genome.sequence.resolve().is_file()
    assert all(artifact.resolve().is_file() for artifact in genome.source_manifests)
    manifest = _load_run_record(_expected_destination(request))
    assert all(item.artifact.resolve().is_file() for item in manifest.outputs)


def test_user_profile_override_is_recorded_as_profile_component(tmp_path: Path) -> None:
    request = _request(tmp_path)
    declared = request.data.artifacts["hmm_profiles"]
    manager = _fake_environment_manager(tmp_path / "cache")
    expected = manager.expected_profile_identity(
        _SYNTHETIC_SPEC,
        organelle="mitochondrion",
        override=declared,
    )

    execute_assembly(
        request,
        environment_manager=manager,
        runner=_successful_fake_oatk_runner(tmp_path),
    )

    manifest = _load_run_record(_expected_destination(request))
    profile_components = tuple(
        component for component in manifest.components if component.category == "profile"
    )
    assert profile_components == (expected,)
    assert profile_components[0].locator == "user-supplied"


def test_success_records_all_fifteen_stages(tmp_path: Path) -> None:
    read_path = _read_file(tmp_path)
    request = _request(tmp_path, read_path=read_path)
    execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_successful_fake_oatk_runner(tmp_path),
    )
    manifest = _load_run_record(_expected_destination(request))
    stage_names = [stage.stage for stage in manifest.stages]
    assert stage_names == [
        "validate_input",
        "prepare_routing_context",
        "route_backend",
        "resolve_parameters",
        "resolve_environment",
        "resolve_resources",
        "prepare_workspace",
        "execute_backend",
        "collect_outputs",
        "normalize_outputs",
        "validate_outputs",
        "write_run_manifest",
        "write_primary_genome",
        "write_result",
        "publish_output",
    ]


def test_route_backend_runs_after_routing_context_and_records_evidence(
    tmp_path: Path,
) -> None:
    read_path = _read_file(tmp_path)
    request = _request(tmp_path, read_path=read_path)
    execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_successful_fake_oatk_runner(tmp_path),
    )
    manifest = _load_run_record(_expected_destination(request))
    stage_names = [stage.stage for stage in manifest.stages]
    # Context preparation precedes routing, which precedes parameter resolution.
    assert stage_names.index("prepare_routing_context") < stage_names.index("route_backend")
    assert stage_names.index("route_backend") < stage_names.index("resolve_parameters")
    assert manifest.routing_evidence is not None
    assert manifest.routing_evidence.rule_id == "explicit.backend"
    assert manifest.routing_evidence.taxon_group == "plant"
    assert manifest.routing_evidence.total_bases == 8  # "@r1\nACGTACGT\n+\nIIIIIIII\n"
    assert manifest.routing_evidence.genome_size_bp is None


def test_routing_context_uses_the_released_runtime_registry(tmp_path: Path) -> None:
    """The service must pass the live released runtime registry to routing, so a
    swapped registry (e.g. a private synthetic backend) gates auto routing."""
    read_path = _read_file(tmp_path)
    request = _request(tmp_path, read_path=read_path)
    captured: dict[str, object] = {}

    real_resolve = assembly_service.resolve_backend

    def spy_resolve(
        data: OrganelleData,
        *,
        organelle: Literal["mitochondrion", "plastid"],
        method: str = "auto",
        context: object | None = None,
        released_backend_ids: object | None = None,
    ) -> object:
        captured["released_backend_ids"] = released_backend_ids
        captured["context"] = context
        return real_resolve(
            data,
            organelle=organelle,
            method=method,
            context=context,  # type: ignore[arg-type]
            released_backend_ids=released_backend_ids,  # type: ignore[arg-type]
        )

    original_resolve = assembly_service.resolve_backend
    assembly_service.resolve_backend = spy_resolve
    try:
        execute_assembly(
            request,
            environment_manager=_fake_environment_manager(tmp_path / "cache"),
            runner=_successful_fake_oatk_runner(tmp_path),
        )
    finally:
        assembly_service.resolve_backend = original_resolve
    assert captured["released_backend_ids"] == tuple(assembly_service.RUNTIMES)
    assert captured["context"] is not None


def test_explicit_method_records_requested_backend(tmp_path: Path) -> None:
    read_path = _read_file(tmp_path)
    request = _request(tmp_path, read_path=read_path, method="oatk")
    result = execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_successful_fake_oatk_runner(tmp_path),
    )
    assert result.provenance is not None
    assert result.provenance.requested_backend == "oatk"


# ---------------------------------------------------------------------------
# Pre-start exceptions (typed, no destination)
# ---------------------------------------------------------------------------


def test_unsupported_oatk_auxiliary_fails_before_artifact_or_environment_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _hifi_data(_read_file(tmp_path), _profile_file(tmp_path))
    reference_path = tmp_path / "inputs" / "reference.fasta"
    reference_path.write_text(">reference\nACGT\n")
    reference = ArtifactRef.from_path(
        reference_path,
        kind="reference",
        format="fasta",
        media_type="text/x-fasta",
    )
    artifacts = dict(data.artifacts.items())
    artifacts["reference_fasta"] = reference
    payload = cast(dict[str, object], data.model_dump(mode="json")["payload"])
    payload["auxiliary"] = {
        "hmm_profiles_artifact": "hmm_profiles",
        "reference_fasta_artifact": "reference_fasta",
    }
    unsupported = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": artifacts,
            "payload": payload,
        }
    )
    request = AssemblyRequest(
        data=unsupported,
        organelle="mitochondrion",
        method="oatk",
    )

    def forbid_artifact_read(data: OrganelleData) -> None:
        raise AssertionError("unsupported input reached artifact I/O")

    monkeypatch.setattr(assembly_service, "_reverify_all_input_artifacts", forbid_artifact_read)

    with pytest.raises(OrganelleInputError) as raised:
        execute_assembly(
            request,
            environment_manager=object(),  # type: ignore[arg-type]
        )

    assert raised.value.code == "assembly.unsupported_data_profile"
    assert not _expected_destination(request).exists()


def test_changed_input_fails_reuse_before_execution(tmp_path: Path) -> None:
    read_path = _read_file(tmp_path)
    request = _request(tmp_path, read_path=read_path)
    execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_successful_fake_oatk_runner(tmp_path),
    )
    # mutate the read at the same path
    read_path.write_bytes(b"@r1\nTTTTGGGG\n+\nIIIIIIII\n")
    with pytest.raises((OrganelleExecutionError, OrganelleDependencyError)):
        execute_assembly(
            request,
            environment_manager=_fake_environment_manager(tmp_path / "cache"),
            runner=ForbiddenRunner(),
        )


def test_require_cache_unavailable_raises_before_destination(tmp_path: Path) -> None:
    read_path = _read_file(tmp_path)
    request = _request(
        tmp_path,
        read_path=read_path,
        environment_source="managed",
    )
    with pytest.raises((OrganelleExecutionError, OrganelleDependencyError)):
        execute_assembly(
            request,
            environment_manager=_fake_environment_manager(tmp_path / "cache"),
            runner=ForbiddenRunner(),
        )
    assert not (_expected_destination(request)).exists()


def test_launch_error_raises_before_destination(tmp_path: Path) -> None:
    read_path = _read_file(tmp_path)
    request = _request(tmp_path, read_path=read_path)

    class _LaunchErrorRunner(CommandRunner):
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
            return CommandOutcome(
                stage=stage,
                argv=argv,
                cwd=str(cwd),
                started=False,
                started_at=now,
                finished_at=now,
                duration_seconds=0.0,
                returncode=None,
                termination="launch_error",
                signal=None,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                launch_error="simulated launch failure",
            )

    with pytest.raises((OrganelleExecutionError, OrganelleDependencyError)):
        execute_assembly(
            request,
            environment_manager=_fake_environment_manager(tmp_path / "cache"),
            runner=_LaunchErrorRunner(),
        )
    assert not (_expected_destination(request)).exists()


# ---------------------------------------------------------------------------
# Post-start failures (status="failed", no primary genome)
# ---------------------------------------------------------------------------


def test_nonzero_exit_produces_failed_result(tmp_path: Path) -> None:
    read_path = _read_file(tmp_path)
    request = _request(tmp_path, read_path=read_path)
    result = execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_nonzero_fake_oatk_runner(tmp_path),
    )
    assert result.status == "failed"
    assert result.errors


def test_failed_result_publishes_machine_readable_evidence(tmp_path: Path) -> None:
    request = _request(tmp_path)
    manager = _fake_environment_manager(tmp_path / "cache")
    expected_profile = manager.expected_profile_identity(
        _SYNTHETIC_SPEC,
        organelle="mitochondrion",
        override=request.data.artifacts["hmm_profiles"],
    )

    result = execute_assembly(
        request,
        environment_manager=manager,
        runner=_nonzero_fake_oatk_runner(tmp_path),
    )

    assert {artifact.kind for artifact in result.artifacts} == {
        "assembly_run_manifest",
        "assembly_run_record",
        "assembly_run_observations",
        "assembly_stdout_log",
        "assembly_stderr_log",
    }
    assert all(artifact.resolve().is_file() for artifact in result.artifacts)
    restored = load_result(_expected_destination(request) / "result.json")
    assert restored == result
    manifest = _load_run_record(_expected_destination(request))
    # Failed manifests also record the validated input payload for QC.
    assert manifest.input_payload is not None
    assert manifest.input_payload.long_libraries[0].reads_artifact == "long_0_reads"
    profile_components = tuple(
        component for component in manifest.components if component.category == "profile"
    )
    assert profile_components == (expected_profile,)
    assert manifest.route_reason_code == "explicit.backend"
    assert result.provenance is not None
    assert result.provenance.model_hashes["hmm_profile"] == profile_components[0].sha256
    assert not result.provenance.database_hashes
    assert assembly_run_id_from_artifact(result.artifacts[0]) == manifest.run_manifest_id
    assert not (_expected_destination(request) / "primary_genome.json").exists()
    # the failed result is still published
    assert (_expected_destination(request) / "result.json").exists()


def test_missing_output_produces_failed_result(tmp_path: Path) -> None:
    read_path = _read_file(tmp_path)
    request = _request(tmp_path, read_path=read_path)

    class _NoOutputRunner(_FakeOatkRunner):
        def __init__(self, workspace_root: Path) -> None:
            super().__init__(workspace_root, mode="partial")

    result = execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_NoOutputRunner(tmp_path),
    )
    assert result.status == "failed"


# ---------------------------------------------------------------------------
# Exact reuse
# ---------------------------------------------------------------------------


def test_exact_completed_destination_is_reused_without_starting_oatk(tmp_path: Path) -> None:
    read_path = _read_file(tmp_path)
    request = _request(tmp_path, read_path=read_path)
    first = execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_successful_fake_oatk_runner(tmp_path),
    )
    forbidden = ForbiddenRunner()
    second = execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=forbidden,
    )
    assert second == first
    assert forbidden.calls == []


def test_exact_promoted_destination_is_reused_through_managed_link(tmp_path: Path) -> None:
    read_path = _read_file(tmp_path)
    request = _request(tmp_path, read_path=read_path)
    first = execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_successful_fake_oatk_runner(tmp_path),
    )
    published = write_assembly(first, output=tmp_path / "published")
    forbidden = ForbiddenRunner()

    reused = execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=forbidden,
    )

    assert reused == published
    assert forbidden.calls == []


def test_changed_expected_profile_identity_blocks_reuse(tmp_path: Path) -> None:
    request = _request(tmp_path)
    manager = _fake_environment_manager(tmp_path / "cache")
    execute_assembly(
        request,
        environment_manager=manager,
        runner=_successful_fake_oatk_runner(tmp_path),
    )

    class _ChangedProfileManager(_SyntheticEnvManager):
        def expected_profile_identity(
            self,
            spec: AssemblyEnvironmentSpec,
            *,
            organelle: Literal["mitochondrion", "plastid"],
            override: ArtifactRef | None = None,
        ) -> AssemblyComponentIdentity:
            current = super().expected_profile_identity(
                spec, organelle=organelle, override=override
            )
            return current.model_copy(update={"sha256": "0" * 64})

    with pytest.raises(OrganelleExecutionError) as raised:
        execute_assembly(
            request,
            environment_manager=_ChangedProfileManager(tmp_path / "cache"),
            runner=ForbiddenRunner(),
        )

    assert raised.value.code == "assembly.destination_conflict"


def test_tampered_published_output_blocks_reuse(tmp_path: Path) -> None:
    request = _request(tmp_path)
    execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_successful_fake_oatk_runner(tmp_path),
    )
    (_expected_destination(request) / "normalized" / "assembly.fasta").write_text(
        ">tampered\nAAAA\n"
    )

    with pytest.raises(OrganelleExecutionError) as raised:
        execute_assembly(
            request,
            environment_manager=_fake_environment_manager(tmp_path / "cache"),
            runner=ForbiddenRunner(),
        )

    assert raised.value.code == "assembly.destination_conflict"


def test_tampered_observations_binding_blocks_reuse(tmp_path: Path) -> None:
    request = _request(tmp_path)
    execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_successful_fake_oatk_runner(tmp_path),
    )
    destination = _expected_destination(request)
    observations_path = destination / "observations.json"
    observations = AssemblyRunObservations.model_validate_json(observations_path.read_bytes())
    observations = observations.model_copy(
        update={"run_manifest_id": "assembly-run:sha256:" + "0" * 64}
    )
    observations_path.write_text(
        json.dumps(observations.model_dump(mode="json"), sort_keys=True) + "\n"
    )
    result = load_result(destination / "result.json")
    replacement = ArtifactRef.from_path(
        observations_path,
        kind="assembly_run_observations",
        format="json",
        media_type="application/json",
    )
    artifacts = tuple(
        replacement if artifact.kind == "assembly_run_observations" else artifact
        for artifact in result.artifacts
    )
    save_contract(result.model_copy(update={"artifacts": artifacts}), destination / "result.json")

    with pytest.raises(OrganelleExecutionError) as raised:
        execute_assembly(
            request,
            environment_manager=_fake_environment_manager(tmp_path / "cache"),
            runner=ForbiddenRunner(),
        )

    assert raised.value.code == "assembly.destination_conflict"


def test_manifest_parameters_must_match_the_current_request(tmp_path: Path) -> None:
    request = _request(tmp_path)
    execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_successful_fake_oatk_runner(tmp_path),
    )
    result_path = _expected_destination(request) / "result.json"
    result = load_result(result_path)
    assert result.provenance is not None
    # Tamper the stored parameters hash so it no longer matches the request identity.
    tampered_provenance = result.provenance.model_copy(update={"parameters_hash": "0" * 64})
    save_contract(result.model_copy(update={"provenance": tampered_provenance}), result_path)

    with pytest.raises(OrganelleExecutionError) as raised:
        execute_assembly(
            request,
            environment_manager=_fake_environment_manager(tmp_path / "cache"),
            runner=ForbiddenRunner(),
        )

    assert raised.value.code == "assembly.destination_conflict"


def test_result_outputs_must_exactly_match_manifest_outputs(tmp_path: Path) -> None:
    request = _request(tmp_path)
    execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_successful_fake_oatk_runner(tmp_path),
    )
    destination = _expected_destination(request)
    extra_path = destination / "normalized" / "unmanifested.txt"
    extra_path.write_text("not declared by the run manifest\n")
    extra = ArtifactRef.from_path(
        extra_path,
        kind="assembly_output",
        format="txt",
        media_type="application/octet-stream",
    )
    result_path = destination / "result.json"
    result = load_result(result_path)
    save_contract(result.model_copy(update={"artifacts": (*result.artifacts, extra)}), result_path)

    with pytest.raises(OrganelleExecutionError) as raised:
        execute_assembly(
            request,
            environment_manager=_fake_environment_manager(tmp_path / "cache"),
            runner=ForbiddenRunner(),
        )

    assert raised.value.code == "assembly.destination_conflict"


def test_result_outputs_cannot_be_reused_by_two_manifest_roles(tmp_path: Path) -> None:
    request = _request(tmp_path)
    execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_successful_fake_oatk_runner(tmp_path),
    )
    destination = _expected_destination(request)
    manifest_path = destination / "assembly_run_manifest.json"
    record_path = destination / "assembly_run_record.json"
    observations_path = destination / "observations.json"
    genome_path = destination / "primary_genome.json"
    result_path = destination / "result.json"

    manifest = _load_run_record(destination)
    duplicated_outputs = list(manifest.outputs)
    duplicated_outputs[1] = ManifestArtifact(
        role=duplicated_outputs[1].role,
        artifact=duplicated_outputs[0].artifact,
    )
    manifest = manifest.model_copy(update={"outputs": tuple(duplicated_outputs)})
    manifest_path.write_bytes(manifest.canonical_bytes())
    record_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    canonical_artifact = manifest.as_artifact(str(manifest_path))

    observations = AssemblyRunObservations.model_validate_json(observations_path.read_bytes())
    observations_path.write_text(
        json.dumps(
            observations.model_copy(
                update={"run_manifest_id": manifest.run_manifest_id}
            ).model_dump(mode="json"),
            sort_keys=True,
        )
        + "\n"
    )
    genome = load_genome(genome_path)
    save_contract(
        genome.model_copy(update={"source_manifests": (canonical_artifact,)}), genome_path
    )

    replacements = {
        "assembly_run_manifest": canonical_artifact,
        "assembly_run_record": ArtifactRef.from_path(
            record_path,
            kind="assembly_run_record",
            format="json",
            media_type="application/json",
        ),
        "primary_genome_manifest": ArtifactRef.from_path(
            genome_path,
            kind="primary_genome_manifest",
            format="json",
            media_type="application/json",
        ),
        "assembly_run_observations": ArtifactRef.from_path(
            observations_path,
            kind="assembly_run_observations",
            format="json",
            media_type="application/json",
        ),
    }
    result = load_result(result_path)
    assert result.provenance is not None
    provenance = result.provenance.model_copy(update={"run_manifest_id": manifest.run_manifest_id})
    artifacts = tuple(replacements.get(artifact.kind, artifact) for artifact in result.artifacts)
    save_contract(
        result.model_copy(update={"provenance": provenance, "artifacts": artifacts}), result_path
    )

    with pytest.raises(OrganelleExecutionError) as raised:
        execute_assembly(
            request,
            environment_manager=_fake_environment_manager(tmp_path / "cache"),
            runner=ForbiddenRunner(),
        )

    assert raised.value.code == "assembly.destination_conflict"


def test_atomic_publish_never_replaces_a_racing_empty_destination(tmp_path: Path) -> None:
    request = _request(tmp_path)
    destination = _expected_destination(request)

    class _RacingRunner(_FakeOatkRunner):
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
            outcome = super().run(
                argv,
                stage=stage,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                env=env,
            )
            destination.mkdir()
            return outcome

    with pytest.raises(OrganelleExecutionError) as raised:
        execute_assembly(
            request,
            environment_manager=_fake_environment_manager(tmp_path / "cache"),
            runner=_RacingRunner(tmp_path / "scripts"),
        )

    assert raised.value.code == "assembly.destination_conflict"
    assert destination.is_dir()
    assert not any(destination.iterdir())


def test_failed_destination_is_archived_and_retry_proceeds(tmp_path: Path) -> None:
    """A failed run no longer wedges its destination forever (2026-08-15 fix).

    The original test pinned "failed destination is never overwritten" —
    which turned one bad read set into a permanent retry wedge. The fix
    preserves the evidence (archived aside with a timestamp) while letting
    the retry proceed to a fresh workspace.
    """
    read_path = _read_file(tmp_path)
    request = _request(tmp_path, read_path=read_path)
    first = execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_nonzero_fake_oatk_runner(tmp_path),
    )
    assert first.status == "failed"
    second = execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_successful_fake_oatk_runner(tmp_path),
    )
    assert second.status == "ok"


# ---------------------------------------------------------------------------
# KeyboardInterrupt cleanup
# ---------------------------------------------------------------------------


def test_keyboard_interrupt_leaves_no_destination(tmp_path: Path) -> None:
    read_path = _read_file(tmp_path)
    request = _request(tmp_path, read_path=read_path)

    class _InterruptRunner(CommandRunner):
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
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_assembly(
            request,
            environment_manager=_fake_environment_manager(tmp_path / "cache"),
            runner=_InterruptRunner(),
        )
    assert not (_expected_destination(request)).exists()
    # no temp siblings left in the managed operation directory
    op_root = _expected_destination(request).parent
    assert not op_root.exists() or not any(p.name.startswith(".") for p in op_root.iterdir())


# ---------------------------------------------------------------------------
# Fake-executable integration test (real subprocess, real filesystem)
# ---------------------------------------------------------------------------


def _write_fake_oatk(workspace_root: Path, mode: str = "success") -> Path:
    """Write a real executable Python script that emulates oatk's output contract."""
    workspace_root.mkdir(parents=True, exist_ok=True)
    script = workspace_root / "fake_oatk.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys, re\n"
        "mode = os.environ.get('OATK_FAKE_MODE', 'success')\n"
        "out_prefix = None\n"
        "target = 'mito'\n"
        "args = sys.argv[1:]\n"
        "for i, a in enumerate(args):\n"
        "    if a == '-o' and i + 1 < len(args):\n"
        "        out_prefix = args[i + 1]\n"
        "    if a == '-p':\n"
        "        target = 'pltd'\n"
        "if out_prefix is not None:\n"
        "    out_dir = os.path.dirname(out_prefix)\n"
        "    if out_dir:\n"
        "        os.makedirs(out_dir, exist_ok=True)\n"
        "if mode == 'nonzero':\n"
        "    sys.stderr.write('simulated failure\\n')\n"
        "    sys.exit(1)\n"
        "if mode == 'timeout':\n"
        "    import time; time.sleep(30)\n"
        "if out_prefix is None:\n"
        "    sys.exit(2)\n"
        "if mode == 'malformed':\n"
        "    with open(out_prefix + '.' + target + '.ctg.fasta', 'w') as f:\n"
        "        f.write('>ctg1\\nACGTXYZ\\n')\n"
        "    with open(out_prefix + '.' + target + '.gfa', 'w') as f:\n"
        "        f.write('H\\tVN:Z:1.0\\nS\\ts1\\tACGT\\n')\n"
        "    with open(out_prefix + '.' + target + '.ctg.bed', 'w') as f:\n"
        "        f.write('ctg1\\t0\\t3\\tgeneA\\t.\\t+\\n')\n"
        "    with open(out_prefix + '.annot_' + target + '.txt', 'w') as f:\n"
        "        f.write('annotation\\n')\n"
        "    with open(out_prefix + '.utg.final.gfa', 'w') as f:\n"
        "        f.write('H\\tVN:Z:1.0\\nS\\ts1\\tACGT\\n')\n"
        "    sys.exit(0)\n"
        "if mode == 'partial':\n"
        "    # only write the fasta, leave the rest missing\n"
        "    with open(out_prefix + '.' + target + '.ctg.fasta', 'w') as f:\n"
        "        f.write('>ctg1\\nACGT\\n')\n"
        "    sys.exit(0)\n"
        "# success\n"
        "with open(out_prefix + '.' + target + '.ctg.fasta', 'w') as f:\n"
        "    f.write('>ctg1 description\\nACGT\\n')\n"
        "with open(out_prefix + '.' + target + '.gfa', 'w') as f:\n"
        "    f.write('H\\tVN:Z:1.0\\nS\\ts1\\tACGT\\n')\n"
        "with open(out_prefix + '.' + target + '.ctg.bed', 'w') as f:\n"
        "    f.write('ctg1\\t0\\t3\\tgeneA\\t.\\t+\\n')\n"
        "with open(out_prefix + '.annot_' + target + '.txt', 'w') as f:\n"
        "    f.write('annotation\\n')\n"
        "with open(out_prefix + '.utg.final.gfa', 'w') as f:\n"
        "    f.write('H\\tVN:Z:1.0\\nS\\ts1\\tACGT\\n')\n"
        "sys.exit(0)\n"
    )
    script.chmod(0o755)
    return script


class _FakeOatkRunner(CommandRunner):
    """Runs a real fake-oatk subprocess, producing real output files.

    The fake oatk executable is written to workspace_root and the resolved argv
    points at the prepared environment's executable path. We override run() to
    invoke our fake script instead while keeping the real subprocess + filesystem
    pipeline.
    """

    def __init__(self, workspace_root: Path, mode: str = "success") -> None:
        super().__init__()
        self._workspace_root = workspace_root
        self._mode = mode
        self._script = _write_fake_oatk(workspace_root, mode)

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
        # Replace the oatk executable in argv with our fake script and run it
        # through the real CommandRunner so process/filesystem/observation are real.
        fake_argv = (str(self._script), *argv[1:])
        run_env = dict(env or os.environ)
        run_env["OATK_FAKE_MODE"] = self._mode
        return super().run(
            fake_argv,
            stage=stage,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            env=run_env,
        )


def _successful_fake_oatk_runner(tmp_path: Path) -> _FakeOatkRunner:
    return _FakeOatkRunner(tmp_path / "scripts", mode="success")


def _nonzero_fake_oatk_runner(tmp_path: Path) -> _FakeOatkRunner:
    return _FakeOatkRunner(tmp_path / "scripts", mode="nonzero")


def test_fake_executable_success_through_real_subprocess(tmp_path: Path) -> None:
    # read path with a space and $ to prove shell safety
    read_path = tmp_path / "inputs" / "sample $1 reads.fastq"
    read_path.parent.mkdir(parents=True, exist_ok=True)
    read_path.write_bytes(b"@r1\nACGT\n+\nIIII\n")
    request = _request(tmp_path, read_path=read_path)
    result = execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_successful_fake_oatk_runner(tmp_path),
    )
    assert result.status == "ok"
    genome = load_genome(_expected_destination(request) / "primary_genome.json")
    assert genome.sequence is not None


def test_fake_executable_nonzero_through_real_subprocess(tmp_path: Path) -> None:
    read_path = _read_file(tmp_path)
    request = _request(tmp_path, read_path=read_path)
    result = execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=_nonzero_fake_oatk_runner(tmp_path),
    )
    assert result.status == "failed"
    assert not (_expected_destination(request) / "primary_genome.json").exists()


def test_fake_executable_partial_output_through_real_subprocess(tmp_path: Path) -> None:
    read_path = _read_file(tmp_path)
    request = _request(tmp_path, read_path=read_path)
    runner = _FakeOatkRunner(tmp_path / "scripts", mode="partial")
    result = execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=runner,
    )
    assert result.status == "failed"
    manifest = _load_run_record(_expected_destination(request))
    failed = tuple(stage.stage for stage in manifest.stages if stage.status == "failed")
    assert failed == ("collect_outputs",)


def test_fake_executable_malformed_output_through_real_subprocess(tmp_path: Path) -> None:
    read_path = _read_file(tmp_path)
    request = _request(tmp_path, read_path=read_path)
    runner = _FakeOatkRunner(tmp_path / "scripts", mode="malformed")
    result = execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=runner,
    )
    assert result.status == "failed"
    assert not (_expected_destination(request) / "primary_genome.json").exists()
    assert (_expected_destination(request) / "result.json").exists()
    manifest = _load_run_record(_expected_destination(request))
    failed = tuple(stage.stage for stage in manifest.stages if stage.status == "failed")
    assert failed == ("normalize_outputs",)
    assert result.errors[0].suggested_action["failed_stage"] == "normalize_outputs"


def test_fake_executable_timeout_through_real_subprocess(tmp_path: Path) -> None:
    read_path = _read_file(tmp_path)
    request = _request(tmp_path, read_path=read_path)
    request = request.model_copy(update={"timeout_seconds": 1})
    runner = _FakeOatkRunner(tmp_path / "scripts", mode="timeout")
    result = execute_assembly(
        request,
        environment_manager=_fake_environment_manager(tmp_path / "cache"),
        runner=runner,
    )
    assert result.status == "failed"
    assert not (_expected_destination(request) / "primary_genome.json").exists()
    assert result.provenance is not None


def test_service_has_no_backend_name_conditionals() -> None:
    source = inspect.getsource(assembly_service.execute_assembly)
    assert '== "oatk"' not in source
    assert '== "himt"' not in source
    assert "OatkAdapter" not in source
    assert "OATK_ENVIRONMENT" not in source


class _ProbeResourceProvider(AssemblyResourceProvider):
    def expected(
        self,
        manager: EnvironmentManager,
        environment_spec: AssemblyEnvironmentSpec,
        request: AssemblyRequest,
        payload: AssemblyInputPayload,
    ) -> ExpectedBackendResources:
        return ExpectedBackendResources()

    def prepare(
        self,
        manager: EnvironmentManager,
        environment_spec: AssemblyEnvironmentSpec,
        request: AssemblyRequest,
        payload: AssemblyInputPayload,
        environment: PreparedEnvironment,
    ) -> PreparedBackendResources:
        return PreparedBackendResources()


class _ProbeAdapter:
    backend_id = "oatk"

    def preflight(self, context: AdapterContext) -> None:
        return None

    def build_command(self, context: AdapterContext) -> AssemblyCommand:
        return AssemblyCommand(stable_argv=("probe",), resolved_argv=("true",))

    def collect_outputs(self, context: AdapterContext) -> RawAssemblyOutputs:
        prefix = context.workspace / "backend" / context.environment.backend_id
        prefix.mkdir(parents=True, exist_ok=True)
        fasta = prefix / "assembly.fasta"
        fasta.write_text(">ctg1\nACGT\n")
        return RawAssemblyOutputs(
            outputs=(BackendOutput(role="assembly_fasta", path=fasta, format="fasta"),),
            primary_sequence_role="assembly_fasta",
        )

    def normalize(
        self,
        context: AdapterContext,
        raw: RawAssemblyOutputs,
        output_dir: Path,
    ) -> NormalizedAssemblyOutputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        source = raw.require("assembly_fasta").path
        destination = output_dir / "assembly.fasta"
        destination.write_bytes(source.read_bytes())
        return NormalizedAssemblyOutputs(
            outputs=(BackendOutput(role="assembly_fasta", path=destination, format="fasta"),),
            primary_sequence_role="assembly_fasta",
            record_count=1,
            total_bases=4,
        )


def _probe_environment_spec() -> AssemblyEnvironmentSpec:
    return AssemblyEnvironmentSpec(
        backend_id="oatk",
        contract_version="probe.environment.v1",
        platforms=(
            CondaPlatformSpec(
                platform="linux-64",
                package="probe=1.0=0",
                lock_resource=(
                    "organelleverse.assembly.resources.environments.oatk/linux-64.explicit.txt"
                ),
                executable_names=("true",),
                version_argv=("--version",),
            ),
        ),
    )


def _probe_runtime() -> BackendRuntime:
    return BackendRuntime(
        backend_id="oatk",
        environment_spec=_probe_environment_spec(),
        data_validator=validate_oatk_assembly_data,
        resource_provider=_ProbeResourceProvider(),
        adapter_factory=_ProbeAdapter,
    )


class _ProbeEnvManager(EnvironmentManager):
    def __init__(self, cache_root: Path) -> None:
        super().__init__(cache_root=cache_root)
        self._digest = "sha256:" + "a" * 64

    def prepare(
        self,
        spec: AssemblyEnvironmentSpec,
        *,
        policy: Literal["ensure", "require"],
        platform: str | None = None,
    ) -> PreparedEnvironment:
        return PreparedEnvironment(
            backend_id="oatk",
            carrier="conda",
            platform=platform or "linux-64",
            digest=self._digest,
            prefix=self.cache_root,
            executables=(PreparedExecutable(name="true", path=Path("/usr/bin/true")),),
            version="probe 1.0",
        )

    def expected_environment_digest(
        self,
        spec: AssemblyEnvironmentSpec,
        *,
        platform: str | None = None,
    ) -> str:
        return self._digest


def test_private_runtime_registry_executes_synthetic_backend(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    private = AssemblyRuntimeRegistry((_probe_runtime(),))
    original_runtimes = assembly_service.RUNTIMES
    assembly_service.RUNTIMES = private
    try:
        result = execute_assembly(
            request,
            environment_manager=_ProbeEnvManager(tmp_path / "cache"),
        )
        assert result.status == "ok"
        manifest = _load_run_record(_expected_destination(request))
        assert manifest.selected_backend == "oatk"
        assert manifest.primary_sequence_role == "assembly_fasta"
        assert len(manifest.outputs) == 1
        assert manifest.outputs[0].role == "assembly_fasta"
    finally:
        assembly_service.RUNTIMES = original_runtimes


# ---------------------------------------------------------------------------
# HiMT end-to-end execution through a fake executable
# ---------------------------------------------------------------------------

_FAKE_HIMT_SCRIPT = r"""
import argparse
import os
import sys
from pathlib import Path


def write_fasta(path: Path, seq: str, name: str = "seg1") -> None:
    path.write_text(f">{name}\n{seq}\n", encoding="utf-8")


def write_gfa(path: Path, seq: str, seg: str = "seg1") -> None:
    path.write_text(f"H\tVN:Z:1.0\nS\t{seg}\t{seq}\n", encoding="utf-8")


def main() -> None:
    if "--version" in sys.argv[1:]:
        print("HiMT 1.1.3")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="assemble")
    parser.add_argument("-i", dest="input")
    parser.add_argument("-o", dest="output", required=True)
    parser.add_argument("-s", dest="species")
    parser.add_argument("-d", dest="data_type")
    parser.add_argument("-k", type=int)
    parser.add_argument("-n", type=int)
    parser.add_argument("-t", type=int)
    parser.add_argument("-e", type=int)
    parser.add_argument("-b", type=int)
    parser.add_argument("-fd", type=int)
    parser.add_argument("-fp", type=float)
    parser.add_argument("-p", type=float)
    parser.add_argument("-c", type=float)
    parser.add_argument("-x", type=int)
    parser.add_argument("--no_flye_meta", action="store_true")
    args = parser.parse_args()

    mode = os.environ.get("ORGANELLEVERSE_FAKE_HIMT_MODE", "mito_only")
    requested = os.environ.get("ORGANELLEVERSE_FAKE_HIMT_REQUESTED", "mitochondrion")
    calls_path = os.environ.get("ORGANELLEVERSE_FAKE_HIMT_CALLS")

    if calls_path:
        Path(calls_path).parent.mkdir(parents=True, exist_ok=True)
        with open(calls_path, "a", encoding="utf-8") as handle:
            handle.write(f"{mode}\n")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "log.txt").write_text(f"mode={mode} requested={requested}\n", encoding="utf-8")

    if mode == "nonzero":
        sys.exit(1)

    seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGT"

    if mode == "malformed":
        (out / "himt_mitochondrial_raw.fa").write_text("not valid fasta", encoding="utf-8")
        (out / "himt_mitochondrial.gfa").write_text("bad gfa", encoding="utf-8")
        return

    if mode != "missing_requested":
        write_fasta(out / "himt_mitochondrial_raw.fa", seq)
        write_gfa(out / "himt_mitochondrial.gfa", seq)

    if mode in ("both_one_path", "both_two_paths", "missing_requested"):
        if requested == "plastid" and mode == "missing_requested":
            pass
        else:
            write_fasta(out / "chloroplast_path1.fa", seq)
            write_gfa(out / "himt_chloroplast.gfa", seq)
            if mode == "both_two_paths":
                write_fasta(out / "chloroplast_path2.fa", seq)

    if mode == "missing_requested":
        if requested == "mitochondrion":
            (out / "himt_mitochondrial_raw.fa").unlink(missing_ok=True)
            (out / "himt_mitochondrial.gfa").unlink(missing_ok=True)
        elif requested == "plastid":
            (out / "chloroplast_path1.fa").unlink(missing_ok=True)
            (out / "himt_chloroplast.gfa").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
"""


def _himt_environment_manager(tmp_path: Path) -> EnvironmentManager:
    """Return an offline environment manager backed by a synthetic HiMT prefix."""
    prefix = tmp_path / "himt_prefix"
    prefix.mkdir(parents=True, exist_ok=True)
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_path / "fake_himt.py"
    script_path.write_text(_FAKE_HIMT_SCRIPT, encoding="utf-8")

    platform_spec = HIMT_ENVIRONMENT.require_platform("linux-64")
    for name in platform_spec.executable_names:
        exe = bin_dir / name
        if name == "himt":
            wrapper = (
                "#!/usr/bin/env python3\n"
                "import subprocess, sys\n"
                f"sys.exit(subprocess.call([sys.executable, {str(script_path)!r}, *sys.argv[1:]]))\n"
            )
            exe.write_text(wrapper, encoding="utf-8")
        else:
            exe.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
        exe.chmod(0o755)

    meta = prefix / "conda-meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "himt-1.1.3-0.json").write_text(
        json.dumps({"name": "himt", "version": "1.1.3", "build": "0"})
    )
    (prefix / "version.txt").write_text("HiMT 1.1.3", encoding="utf-8")

    class _HimtEnvManager(EnvironmentManager):
        def __init__(self) -> None:
            super().__init__(cache_root=tmp_path / "cache")
            self._prefix = prefix
            self._digest = super().expected_environment_digest(
                HIMT_ENVIRONMENT, platform="linux-64"
            )

        def prepare(
            self,
            spec: AssemblyEnvironmentSpec,
            *,
            policy: Literal["ensure", "require"],
            platform: str | None = None,
        ) -> PreparedEnvironment:
            return PreparedEnvironment(
                backend_id="himt",
                carrier="conda",
                platform=platform or "linux-64",
                digest=self._digest,
                prefix=self._prefix,
                executables=tuple(
                    PreparedExecutable(name=name, path=bin_dir / name)
                    for name in platform_spec.executable_names
                ),
                version="HiMT 1.1.3",
            )

        def expected_environment_digest(
            self,
            spec: AssemblyEnvironmentSpec,
            *,
            platform: str | None = None,
        ) -> str:
            return self._digest

    return _HimtEnvManager()


def _himt_request(
    tmp_path: Path,
    *,
    organelle: str = "mitochondrion",
    backend_parameters: HimtParameters | None = None,
    technology: str = "pacbio_hifi",
    quality_state: str = "ccs",
) -> AssemblyRequest:
    read_path = _read_file(tmp_path)
    read_artifact = ArtifactRef.from_path(
        read_path, kind="long_read", format="fastq", media_type="application/x-fastq"
    )
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {"long_0_reads": read_artifact},
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "long_libraries": [
                    {
                        "technology": technology,
                        "quality_state": quality_state,
                        "reads_artifact": "long_0_reads",
                    }
                ],
            },
        }
    )
    return AssemblyRequest(
        data=data,
        organelle=organelle,  # type: ignore[arg-type]
        method="himt",  # type: ignore[arg-type]
        threads=2,
        backend_parameters=backend_parameters,
    )


def _set_himt_mode(mode: str, *, requested: str = "mitochondrion") -> None:
    os.environ["ORGANELLEVERSE_FAKE_HIMT_MODE"] = mode
    os.environ["ORGANELLEVERSE_FAKE_HIMT_REQUESTED"] = requested


def _clear_himt_mode() -> None:
    os.environ.pop("ORGANELLEVERSE_FAKE_HIMT_MODE", None)
    os.environ.pop("ORGANELLEVERSE_FAKE_HIMT_REQUESTED", None)


@pytest.mark.parametrize(
    ("technology", "quality_state", "expected_accuracy"),
    [
        ("pacbio_hifi", "ccs", 0.8),
        ("pacbio_clr", "raw", 0.3),
        ("pacbio_clr", "corrected", 0.8),
        ("ont", "raw", 0.3),
        ("ont", "corrected", 0.8),
        ("ont", "hq", 0.8),
        ("ont", "duplex", 0.8),
    ],
)
def test_himt_success_records_quality_resolved_accuracy(
    tmp_path: Path,
    technology: str,
    quality_state: str,
    expected_accuracy: float,
) -> None:
    _set_himt_mode("mito_only")
    try:
        request = _himt_request(
            tmp_path,
            technology=technology,
            quality_state=quality_state,
        )
        result = execute_assembly(request, environment_manager=_himt_environment_manager(tmp_path))
        assert result.status == "ok"
        assert result.provenance is not None
        assert result.provenance.actual_backend == "himt"
        assert result.provenance.attempted_backends == ("himt",)
        assert result.provenance.requested_backend == "himt"
        assert result.provenance.software_versions["himt"] == "HiMT 1.1.3"
        genome = load_genome(_expected_destination(request) / "primary_genome.json")
        assert genome.organelle == "mitochondrion"
        manifest = _load_run_record(_expected_destination(request))
        assert manifest.selected_backend == "himt"
        assert manifest.parameters.backend_parameters["accuracy"] == expected_accuracy
    finally:
        _clear_himt_mode()


def test_himt_success_plastid_binds_path1_as_primary(tmp_path: Path) -> None:
    _set_himt_mode("both_two_paths", requested="plastid")
    try:
        request = _himt_request(
            tmp_path,
            organelle="plastid",
            technology="ont",
            quality_state="raw",
        )
        result = execute_assembly(request, environment_manager=_himt_environment_manager(tmp_path))
        assert result.status == "ok"
        genome = load_genome(_expected_destination(request) / "primary_genome.json")
        assert genome.organelle == "plastid"
        manifest = _load_run_record(_expected_destination(request))
        roles = {item.role for item in manifest.outputs}
        assert "assembly_fasta" in roles
        assert "assembly_graph" in roles
        assert "alternate_assembly_fasta" in roles
        assert "detected_mitochondrion_fasta" in roles
        assert "detected_mitochondrion_graph" in roles
        assert manifest.primary_sequence_role == "assembly_fasta"
    finally:
        _clear_himt_mode()


def test_himt_success_mito_only_detects_plastid(tmp_path: Path) -> None:
    _set_himt_mode("both_one_path", requested="mitochondrion")
    try:
        request = _himt_request(tmp_path)
        result = execute_assembly(request, environment_manager=_himt_environment_manager(tmp_path))
        assert result.status == "ok"
        manifest = _load_run_record(_expected_destination(request))
        roles = {item.role for item in manifest.outputs}
        assert "assembly_fasta" in roles
        assert "assembly_graph" in roles
        assert "detected_plastid_path1" in roles
        assert "detected_plastid_graph" in roles
        assert "detected_plastid_path2" not in roles
    finally:
        _clear_himt_mode()


def test_himt_reuse_avoids_second_process_and_parameter_change_prevents_reuse(
    tmp_path: Path,
) -> None:
    calls = tmp_path / "calls.txt"
    os.environ["ORGANELLEVERSE_FAKE_HIMT_CALLS"] = str(calls)
    _set_himt_mode("mito_only")
    request = _himt_request(tmp_path)
    try:
        first = execute_assembly(
            request,
            environment_manager=_himt_environment_manager(tmp_path),
        )
        assert first.status == "ok"
        assert calls.read_text(encoding="utf-8").count("\n") == 1

        reused = execute_assembly(
            request,
            environment_manager=_himt_environment_manager(tmp_path),
        )
        assert reused == first
        assert calls.read_text(encoding="utf-8").count("\n") == 1

        changed = execute_assembly(
            _himt_request(
                tmp_path,
                backend_parameters=HimtParameters(backend="himt", kmer_length=31),
            ),
            environment_manager=_himt_environment_manager(tmp_path),
        )
        assert changed.status == "ok"
        assert calls.read_text(encoding="utf-8").count("\n") == 2
    finally:
        _clear_himt_mode()
        os.environ.pop("ORGANELLEVERSE_FAKE_HIMT_CALLS", None)


def test_himt_tampering_with_primary_output_blocks_reuse(tmp_path: Path) -> None:
    _set_himt_mode("mito_only")
    request = _himt_request(tmp_path)
    output_dir = _expected_destination(request)
    try:
        first = execute_assembly(
            request,
            environment_manager=_himt_environment_manager(tmp_path),
        )
        assert first.status == "ok"

        assembly_fasta = output_dir / "normalized" / "assembly.fasta"
        assembly_fasta.write_text(
            assembly_fasta.read_text(encoding="utf-8").rstrip() + "N\n",
            encoding="utf-8",
        )

        with pytest.raises(OrganelleExecutionError):
            execute_assembly(
                request,
                environment_manager=_himt_environment_manager(tmp_path),
            )
    finally:
        _clear_himt_mode()


def test_himt_failure_nonzero_publishes_failed_result(tmp_path: Path) -> None:
    _set_himt_mode("nonzero")
    request = _himt_request(tmp_path)
    try:
        result = execute_assembly(
            request,
            environment_manager=_himt_environment_manager(tmp_path),
        )
        assert result.status == "failed"
        assert not (_expected_destination(request) / "primary_genome.json").exists()
        assert all(artifact.kind != "primary_genome_manifest" for artifact in result.artifacts)
        manifest = _load_run_record(_expected_destination(request))
        assert manifest.selected_backend == "himt"
        assert any(
            stage.stage == "execute_backend" and stage.status == "failed"
            for stage in manifest.stages
        )
    finally:
        _clear_himt_mode()


def test_himt_failure_missing_requested_publishes_failed_result(tmp_path: Path) -> None:
    _set_himt_mode("missing_requested")
    request = _himt_request(tmp_path)
    try:
        result = execute_assembly(
            request,
            environment_manager=_himt_environment_manager(tmp_path),
        )
        assert result.status == "failed"
        assert not (_expected_destination(request) / "primary_genome.json").exists()
        manifest = _load_run_record(_expected_destination(request))
        assert any(
            stage.stage == "collect_outputs" and stage.status == "failed"
            for stage in manifest.stages
        )
    finally:
        _clear_himt_mode()


def test_himt_failure_malformed_output_publishes_failed_result(tmp_path: Path) -> None:
    _set_himt_mode("malformed")
    request = _himt_request(tmp_path)
    try:
        result = execute_assembly(
            request,
            environment_manager=_himt_environment_manager(tmp_path),
        )
        assert result.status == "failed"
        assert not (_expected_destination(request) / "primary_genome.json").exists()
        manifest = _load_run_record(_expected_destination(request))
        assert any(
            stage.stage == "normalize_outputs" and stage.status == "failed"
            for stage in manifest.stages
        )
    finally:
        _clear_himt_mode()


def test_himt_preflight_rejects_animal_plastid_without_creating_destination(
    tmp_path: Path,
) -> None:
    request = _himt_request(
        tmp_path,
        organelle="plastid",
        backend_parameters=HimtParameters(backend="himt", species="animal"),
    )
    with pytest.raises((OrganelleExecutionError, OrganelleInputError)):
        execute_assembly(request, environment_manager=_himt_environment_manager(tmp_path))
    assert not (_expected_destination(request)).exists()


def test_himt_preflight_rejects_two_long_libraries_without_creating_destination(
    tmp_path: Path,
) -> None:
    read_a = _read_file(tmp_path, name="a.fastq")
    read_b = _read_file(tmp_path, name="b.fastq")
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {
                "long_a": ArtifactRef.from_path(
                    read_a,
                    kind="long_read",
                    format="fastq",
                    media_type="application/x-fastq",
                ),
                "long_b": ArtifactRef.from_path(
                    read_b,
                    kind="long_read",
                    format="fastq",
                    media_type="application/x-fastq",
                ),
            },
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "long_libraries": [
                    {
                        "technology": "pacbio_hifi",
                        "quality_state": "ccs",
                        "reads_artifact": "long_a",
                    },
                    {
                        "technology": "ont",
                        "quality_state": "raw",
                        "reads_artifact": "long_b",
                    },
                ],
            },
        }
    )
    request = AssemblyRequest(
        data=data,
        organelle="mitochondrion",  # type: ignore[arg-type]
        method="himt",  # type: ignore[arg-type]
        threads=2,
    )
    with pytest.raises((OrganelleExecutionError, OrganelleInputError)):
        execute_assembly(request, environment_manager=_himt_environment_manager(tmp_path))
    assert _no_assembly_run_committed()
