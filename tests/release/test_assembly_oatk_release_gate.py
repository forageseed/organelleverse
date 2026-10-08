"""Real Oatk assembly release gate.

Marked ``release_assembly_oatk`` and excluded from the default test run. When
selected, these tests fail rather than skip if prerequisites are absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import pytest

from organelleverse.assembly.backends.base import (
    AdapterContext,
    AssemblyCommand,
    BackendOutput,
    NormalizedAssemblyOutputs,
    RawAssemblyOutputs,
)
from organelleverse.assembly.backends.oatk import OatkAdapter
from organelleverse.assembly.execution import CommandOutcome, CommandRunner

pytestmark = pytest.mark.release_assembly_oatk

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "assembly" / "oatk" / "ddAraThal4.json"


def _load_fixture_manifest() -> dict[str, object]:
    return json.loads(_FIXTURE_PATH.read_text())


def _cache_root() -> Path:
    env = os.environ.get("ORGANELLEVERSE_CACHE_ROOT")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "organelleverse"


def _expected_platform() -> str:
    env = os.environ.get("ORGANELLEVERSE_EXPECTED_CONDA_PLATFORM")
    if env:
        return env
    from organelleverse.assembly.environment_specs import normalize_platform

    return normalize_platform()


def test_managed_oatk_environment_install_and_probe() -> None:
    """Materialize the exact lock, prepare both managed profiles, re-run all probes."""
    from organelleverse.assembly.environment_specs import OATK_ENVIRONMENT, normalize_platform
    from organelleverse.assembly.environments import EnvironmentManager

    detected = normalize_platform()
    expected = _expected_platform()
    assert detected == expected, f"expected conda platform {expected!r} but host is {detected!r}"

    OATK_ENVIRONMENT.require_platform(detected)  # raises if unsupported
    manager = EnvironmentManager(cache_root=_cache_root())

    # Materialize the environment
    environment = manager.prepare(OATK_ENVIRONMENT, policy="ensure", platform=detected)
    assert environment.digest == manager.expected_environment_digest(
        OATK_ENVIRONMENT, platform=detected
    )
    assert environment.require_executable("oatk").is_file()
    assert environment.require_executable("nhmmscan").is_file()

    # Prepare both managed profiles
    for organelle in ("mitochondrion", "plastid"):
        profile = manager.prepare_profile(OATK_ENVIRONMENT, organelle=organelle, policy="ensure")
        assert profile.source == "managed"
        assert profile.path.is_file()


def test_mitochondrial_assembly_with_real_oatk() -> None:
    """Run real Oatk on ddAraThal4 HiFi data for the mitochondrion."""
    _run_real_assembly("mitochondrion")


def test_plastid_assembly_with_real_oatk() -> None:
    """Run real Oatk on ddAraThal4 HiFi data for the plastid."""
    _run_real_assembly("plastid")


def test_exact_reuse_avoids_second_oatk_run() -> None:
    """A second call with the same request reuses the published result."""
    from organelleverse.assembly.service import execute_assembly

    fixture = _load_fixture_manifest()
    cache = _cache_root()
    download = _download_fixture(fixture, cache)
    fasta = _decompress_gz(download, cache)

    from organelleverse.assembly.contracts import AssemblyRequest
    from organelleverse.core.artifacts import ArtifactRef
    from organelleverse.core.data import OrganelleData

    artifact = ArtifactRef.from_path(
        fasta, kind="long_read", format="fasta", media_type="text/x-fasta"
    )
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {"long_0_reads": artifact},
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "long_libraries": [
                    {
                        "technology": "pacbio_hifi",
                        "quality_state": "ccs",
                        "reads_artifact": "long_0_reads",
                    }
                ],
            },
        }
    )
    request = AssemblyRequest(
        data=data,
        organelle="mitochondrion",
        method="oatk",
        environment_source="auto",
    )
    first = execute_assembly(request, environment_manager=None)
    assert first.status == "ok"

    class ForbiddenRunner(CommandRunner):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[object] = []

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
            raise AssertionError("reuse should have prevented Oatk execution")

    second = execute_assembly(request, runner=ForbiddenRunner())
    assert second == first


def test_require_policy_fails_without_cache() -> None:
    """require policy must fail when the profile cache is absent."""
    from organelleverse.assembly.environment_specs import OATK_ENVIRONMENT
    from organelleverse.assembly.environments import EnvironmentManager
    from organelleverse.core.errors import OrganelleDependencyError

    cache = _cache_root() / "require_absent_test"
    manager = EnvironmentManager(cache_root=cache)
    with pytest.raises(OrganelleDependencyError) as raised:
        manager.prepare_profile(OATK_ENVIRONMENT, organelle="mitochondrion", policy="require")
    assert raised.value.code == "assembly.environment_unavailable"


class CorruptingOatkAdapter:
    """Delegates real preflight/build_command/collect_outputs to OatkAdapter,
    then corrupts a copy of the target FASTA before normalize() to prove real
    Oatk output that fails post-start validation becomes a failed Result
    (never a pre-start exception, never a fabricated success)."""

    backend_id = "oatk"

    def __init__(self) -> None:
        self._delegate = OatkAdapter()

    def preflight(self, context: AdapterContext) -> None:
        self._delegate.preflight(context)

    def build_command(self, context: AdapterContext) -> AssemblyCommand:
        return self._delegate.build_command(context)

    def collect_outputs(self, context: AdapterContext) -> RawAssemblyOutputs:
        return self._delegate.collect_outputs(context)

    def normalize(
        self,
        context: AdapterContext,
        raw: RawAssemblyOutputs,
        output_dir: Path,
    ) -> NormalizedAssemblyOutputs:
        corrupted = context.workspace / "corrupted_target.fasta"
        lines = raw.require("assembly_fasta").path.read_text().splitlines(keepends=True)
        corrupted_lines: list[str] = []
        done = False
        for line in lines:
            if not done and line.strip() and not line.startswith(">"):
                corrupted_lines.append("1" + line[1:])
                done = True
            else:
                corrupted_lines.append(line)
        corrupted.write_text("".join(corrupted_lines))
        corrupted_outputs = tuple(
            BackendOutput(
                role=item.role,
                path=corrupted if item.role == "assembly_fasta" else item.path,
                format=item.format,
            )
            if item.role == "assembly_fasta"
            else item
            for item in raw.outputs
        )
        corrupted_raw = raw.model_copy(update={"outputs": corrupted_outputs})
        return self._delegate.normalize(context, corrupted_raw, output_dir)


def test_corrupted_target_fasta_produces_post_start_failed_result() -> None:
    """Real Oatk execution whose output is corrupted before normalize() must
    fail closed as a status="failed" Result, never a crash and never ok=true
    with a usable primary genome."""
    from organelleverse.assembly.contracts import AssemblyRequest
    from organelleverse.assembly.service import execute_assembly
    from organelleverse.core.artifacts import ArtifactRef
    from organelleverse.core.data import OrganelleData

    fixture = _load_fixture_manifest()
    cache = _cache_root()
    download = _download_fixture(fixture, cache)
    fasta = _decompress_gz(download, cache)

    artifact = ArtifactRef.from_path(
        fasta, kind="long_read", format="fasta", media_type="text/x-fasta"
    )
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {"long_0_reads": artifact},
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "long_libraries": [
                    {
                        "technology": "pacbio_hifi",
                        "quality_state": "ccs",
                        "reads_artifact": "long_0_reads",
                    }
                ],
            },
        }
    )
    request = AssemblyRequest(
        data=data,
        organelle="mitochondrion",
        method="oatk",
        environment_source="auto",
    )
    result = execute_assembly(request, adapter=CorruptingOatkAdapter())

    assert result.status == "failed"
    assert result.errors
    run_dir = result.artifacts[0].resolve().parent
    assert not (run_dir / "primary_genome.json").exists()
    assert (run_dir / "result.json").exists()


def test_user_supplied_raw_hmm_profile_with_real_oatk() -> None:
    """A copied raw .fam works without managed sidecars and is recorded as user input."""
    _run_real_assembly("mitochondrion", use_custom_profile=True)


def test_agent_json_runs_real_oatk_end_to_end() -> None:
    """Explicit Oatk Agent transport reaches Oatk and preserves the evidence chain."""
    from organelleverse.assembly.manifests import (
        AssemblyRunManifest,
        assembly_run_id_from_artifact,
    )
    from organelleverse.assembly.operations import ASSEMBLE_SPEC
    from organelleverse.core.artifacts import ArtifactRef
    from organelleverse.core.data import OrganelleData
    from organelleverse.core.result import OrganelleResult
    from organelleverse.core.serialization import load_genome, load_result
    from organelleverse.operations import registry
    from organelleverse.operations.adapters import invoke_json

    fixture = _load_fixture_manifest()
    cache = _cache_root()
    download = _download_fixture(fixture, cache)
    fasta = _decompress_gz(download, cache)
    artifact = ArtifactRef.from_path(
        fasta, kind="long_read", format="fasta", media_type="text/x-fasta"
    )
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {"long_0_reads": artifact},
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "long_libraries": [
                    {
                        "technology": "pacbio_hifi",
                        "quality_state": "ccs",
                        "reads_artifact": "long_0_reads",
                    }
                ],
            },
        }
    )
    response = invoke_json(
        {
            "operation_id": "assembly.assemble",
            "input": data.model_dump(mode="json"),
            "parameters": {
                "organelle": "mitochondrion",
                "method": "oatk",
                "environment_source": "auto",
            },
        },
        registry=registry,
        granted_side_effects=set(ASSEMBLE_SPEC.side_effects),
    )

    assert response["ok"] is True, response.get("error")
    transported = OrganelleResult.model_validate(response["result"])
    record_path = next(
        artifact.resolve()
        for artifact in transported.artifacts
        if artifact.kind == "assembly_run_record"
    )
    run_dir = record_path.parent
    published = load_result(run_dir / "result.json")
    genome = load_genome(run_dir / "primary_genome.json")
    record = AssemblyRunManifest.model_validate_json(record_path.read_bytes())
    assert transported == published
    assert transported.status == "ok"
    assert transported.provenance is not None
    run_manifest = next(
        artifact for artifact in transported.artifacts if artifact.kind == "assembly_run_manifest"
    )
    run_record = next(
        artifact for artifact in transported.artifacts if artifact.kind == "assembly_run_record"
    )
    assert run_record.resolve() == record_path.resolve()
    assert run_record.sha256 == hashlib.sha256(record_path.read_bytes()).hexdigest()
    assert (run_dir / "assembly_run_manifest.json").read_bytes() == record.canonical_bytes()
    assert genome.source_manifests == (run_manifest,)
    assert (
        record.run_manifest_id
        == assembly_run_id_from_artifact(run_manifest)
        == transported.provenance.run_manifest_id
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_real_assembly(
    organelle: Literal["mitochondrion", "plastid"],
    *,
    use_custom_profile: bool = False,
) -> None:
    from organelleverse.assembly.contracts import AssemblyRequest
    from organelleverse.assembly.environment_specs import OATK_ENVIRONMENT
    from organelleverse.assembly.environments import EnvironmentManager
    from organelleverse.assembly.service import execute_assembly
    from organelleverse.core.artifacts import ArtifactRef
    from organelleverse.core.data import OrganelleData
    from organelleverse.core.serialization import load_genome

    fixture = _load_fixture_manifest()
    cache = _cache_root()
    download = _download_fixture(fixture, cache)
    fasta = _decompress_gz(download, cache)

    artifact = ArtifactRef.from_path(
        fasta, kind="long_read", format="fasta", media_type="text/x-fasta"
    )
    artifacts = {"long_0_reads": artifact}
    auxiliary: dict[str, str] = {}
    manager: EnvironmentManager | None = None
    if use_custom_profile:
        manager = EnvironmentManager(cache_root=cache)
        managed = manager.prepare_profile(
            OATK_ENVIRONMENT,
            organelle=organelle,
            policy="ensure",
        )
        custom_path = cache / "custom_profiles" / managed.path.name
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(managed.path, custom_path)
        custom_profile = ArtifactRef.from_path(
            custom_path,
            kind="hmm_profile",
            format="fam",
            media_type="text/plain",
        )
        artifacts["hmm_profiles"] = custom_profile
        auxiliary["hmm_profiles_artifact"] = "hmm_profiles"
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": artifacts,
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "long_libraries": [
                    {
                        "technology": "pacbio_hifi",
                        "quality_state": "ccs",
                        "reads_artifact": "long_0_reads",
                    }
                ],
                "auxiliary": auxiliary,
            },
        }
    )
    request = AssemblyRequest(
        data=data,
        organelle=organelle,
        method="oatk",
        environment_source="auto",
    )
    result = execute_assembly(request, environment_manager=manager)
    assert result.status == "ok", f"assembly failed: {result.errors}"
    genome_path = next(
        artifact.resolve()
        for artifact in result.artifacts
        if artifact.kind == "primary_genome_manifest"
    )
    run_dir = genome_path.parent
    genome = load_genome(genome_path)
    assert genome.sequence is not None
    if use_custom_profile:
        assert result.provenance is not None
        assert manager is not None
        expected_profile = manager.expected_profile_identity(
            OATK_ENVIRONMENT,
            organelle=organelle,
            override=artifacts["hmm_profiles"],
        )
        assert result.provenance.model_hashes["hmm_profile"] == expected_profile.sha256
        assert not result.provenance.database_hashes

    # Verify required markers are present in the gene annotations
    markers_key = "mitochondrial_markers" if organelle == "mitochondrion" else "plastid_markers"
    expected_markers = set(cast_markers(fixture[markers_key]))
    bed_path = run_dir / "gene_annotations.bed"
    if bed_path.is_file():
        from organelleverse.assembly.normalization import parse_gene_markers

        found = set(parse_gene_markers(bed_path))
        # Not all markers may be annotated by Oatk; require at least one match.
        assert found & expected_markers, (
            f"none of the expected {markers_key} found in annotations: "
            f"expected={expected_markers}, found={found}"
        )


def cast_markers(raw: object) -> list[str]:
    assert isinstance(raw, list)
    return [str(item) for item in cast("list[object]", raw)]


def _download_fixture(fixture: dict[str, object], cache: Path) -> Path:
    """Download the fixture into the content-addressed cache if absent; verify."""
    url = str(fixture["url"])
    expected_sha256 = str(fixture["sha256"])
    raw_size = fixture["size_bytes"]
    if not isinstance(raw_size, int):
        raise TypeError("fixture size_bytes must be an integer")
    expected_size = raw_size
    expected_md5 = str(fixture["md5"])

    downloads = cache / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    destination = downloads / f"{expected_sha256}.gz"

    if destination.is_file() and destination.stat().st_size == expected_size:
        return destination

    import urllib.request

    urllib.request.urlretrieve(url, destination)
    actual_size = destination.stat().st_size
    assert actual_size == expected_size, (
        f"downloaded size {actual_size} != expected {expected_size}"
    )
    actual_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
    assert actual_sha256 == expected_sha256, "downloaded SHA256 does not match fixture manifest"
    import hashlib as _hl

    actual_md5 = _hl.md5(destination.read_bytes()).hexdigest()
    assert actual_md5 == expected_md5, "downloaded MD5 does not match fixture manifest"
    return destination


def _decompress_gz(gz_path: Path, cache: Path) -> Path:
    """Decompress a .gz file into the cache; reuse if already decompressed."""
    out = gz_path.with_suffix("")
    if out.is_file():
        return out
    import gzip

    with gzip.open(gz_path, "rb") as src, out.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return out
