"""Non-skippable real-data release gate for managed HiMT assembly."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, cast

import pytest

import organelleverse as ov
from organelleverse.assembly.contracts import AssemblyRequest, LongReadLibrary
from organelleverse.assembly.environment_registry import InstallationRegistry
from organelleverse.assembly.environment_specs import HIMT_ENVIRONMENT, normalize_platform
from organelleverse.assembly.environments import EnvironmentManager
from organelleverse.assembly.execution import CommandOutcome, CommandRunner
from organelleverse.assembly.manifests import (
    AssemblyRunManifest,
    assembly_run_id_from_artifact,
)
from organelleverse.assembly.normalization import (
    normalize_fasta,
    normalize_gfa,
    validate_fasta_against_gfa,
)
from organelleverse.assembly.operations import ASSEMBLE_SPEC, ASSEMBLY_WRITE_SPEC
from organelleverse.assembly.service import execute_assembly
from organelleverse.core.data import OrganelleData
from organelleverse.core.result import OrganelleResult
from organelleverse.core.serialization import load_genome, load_result
from organelleverse.operations import registry
from organelleverse.operations.adapters import invoke_json
from tests.release.himt_fixture import PreparedFixture, prepare_himt_fixture

pytestmark = pytest.mark.release_assembly_himt

FixtureProfile = Literal["hifi", "clr", "ont"]


def _cache_root() -> Path:
    configured = os.environ.get("ORGANELLEVERSE_CACHE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "organelleverse").resolve()


def _release_output(cache_root: Path, name: str) -> Path:
    root = (cache_root / "test_output" / "himt-release").resolve(strict=False)
    output = (root / name).resolve(strict=False)
    if output.parent != root:
        raise ValueError("HiMT release output must be a direct child of its gate-owned root")
    return output


def _data_for_fixture(fixture: PreparedFixture) -> OrganelleData:
    return ov.io.read_reads(
        long_libraries=(
            LongReadLibrary(
                technology=fixture.technology,
                quality_state=fixture.quality_state,
                reads=fixture.path,
            ),
        )
    )


def _expected_fixture(profile: FixtureProfile) -> dict[str, object]:
    path = Path(__file__).parent / "fixtures" / "assembly" / "himt" / f"{profile}.expected.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected fixture manifest must be an object: {path}")
    return cast("dict[str, object]", payload)


def _assert_expected_fixture(fixture: PreparedFixture) -> None:
    expected = _expected_fixture(fixture.profile)
    assert expected == {
        "profile": fixture.profile,
        "sha256": fixture.sha256,
        "size_bytes": fixture.size_bytes,
        "records": fixture.records,
        "recipe_identity": fixture.recipe_identity,
    }


def _run_real(profile: FixtureProfile) -> tuple[Path, OrganelleResult]:
    fixture = prepare_himt_fixture(profile, _cache_root())
    _assert_expected_fixture(fixture)
    data = _data_for_fixture(fixture)
    output = _release_output(_cache_root(), f"real-{profile}")
    assembly_facade = cast(Any, ov.assembly)
    public_assemble = cast("Callable[..., OrganelleResult]", assembly_facade.assemble)
    result = public_assemble(
        data,
        organelle="mitochondrion",
        method="himt",
        environment_source="auto",
    )
    result = ov.assembly.write(result, output=output)
    assert result.status == "ok", result.errors
    assert result.provenance is not None
    assert result.provenance.actual_backend == "himt"
    assert result.provenance.attempted_backends == ("himt",)
    return output, result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_artifact_matches_disk(path: Path, sha256: str, size_bytes: int) -> None:
    assert path.is_file(), path
    assert path.stat().st_size == size_bytes
    assert _sha256(path) == sha256


def _assert_fasta_matches_gfa(fasta: Path, graph: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="himt-release-validate-") as temporary:
        root = Path(temporary)
        fasta_stats = normalize_fasta(fasta, root / "sequence.fasta")
        graph_stats = normalize_gfa(graph, root / "graph.gfa")
        validate_fasta_against_gfa(fasta_stats, graph_stats)


def _assert_evidence(
    output: Path,
    result: OrganelleResult,
    *,
    fixture: PreparedFixture,
    expected_accuracy: float,
    require_plastid: bool,
) -> None:
    published = load_result(output / "result.json")
    genome = load_genome(output / "primary_genome.json")
    record_path = output / "assembly_run_record.json"
    canonical_path = output / "assembly_run_manifest.json"
    manifest = AssemblyRunManifest.model_validate_json(record_path.read_bytes())

    assert result == published
    assert result.status == "ok"
    assert result.errors == ()
    assert result.provenance is not None
    assert result.provenance.actual_backend == "himt"
    assert result.provenance.requested_backend == "himt"
    assert result.provenance.attempted_backends == ("himt",)
    assert result.provenance.argv == manifest.stable_argv
    assert result.provenance.software_versions["himt"] == "1.1.3"

    assert manifest.selected_backend == "himt"
    assert manifest.requested_method == "himt"
    assert manifest.organelle == "mitochondrion"
    assert manifest.process_exit_code == 0
    assert manifest.parameters.backend_parameters["accuracy"] == expected_accuracy
    assert manifest.environment.platform == normalize_platform()
    assert manifest.stable_argv[0:2] == ("himt", "assemble")
    assert "role://artifact/long_0_reads" in manifest.stable_argv
    assert str(fixture.path) not in manifest.stable_argv
    assert tuple(stage.stage for stage in manifest.stages)[-4:] == (
        "write_run_manifest",
        "write_primary_genome",
        "write_result",
        "publish_output",
    )
    execute_stage = next(stage for stage in manifest.stages if stage.stage == "execute_backend")
    assert execute_stage.status == "ok"
    assert execute_stage.process_started is True
    assert execute_stage.exit_code == 0

    inputs = {item.role: item.artifact for item in manifest.input_artifacts}
    assert set(inputs) == {"long_0_reads"}
    assert inputs["long_0_reads"].sha256 == fixture.sha256
    assert inputs["long_0_reads"].size_bytes == fixture.size_bytes
    assert result.provenance.input_artifact_hashes == (fixture.sha256,)

    software = {(item.category, item.name): item for item in manifest.components}
    assert software[("software", "himt")].version == "1.1.3"
    manager = EnvironmentManager(cache_root=_cache_root())
    provider = next(
        item
        for item in InstallationRegistry(tool_root=manager.tool_root).locate("himt")
        if item.provider_digest == manifest.environment.digest
    )
    assert provider.version == "1.1.3"
    assert provider.platform == normalize_platform()

    run_manifest_artifact = next(
        artifact for artifact in result.artifacts if artifact.kind == "assembly_run_manifest"
    )
    run_record_artifact = next(
        artifact for artifact in result.artifacts if artifact.kind == "assembly_run_record"
    )
    assert canonical_path.read_bytes() == manifest.canonical_bytes()
    assert assembly_run_id_from_artifact(run_manifest_artifact) == manifest.run_manifest_id
    assert result.provenance.run_manifest_id == manifest.run_manifest_id
    assert run_manifest_artifact.resolve() == canonical_path.resolve()
    assert run_record_artifact.resolve() == record_path.resolve()
    assert len(genome.source_manifests) == 1
    assert (
        Path(genome.source_manifests[0].uri).resolve() == Path(run_manifest_artifact.uri).resolve()
    )
    assert genome.source_manifests[0].sha256 == run_manifest_artifact.sha256
    assert genome.sequence is not None
    assert genome.sequence.resolve().is_file()
    assert genome.sequence.size_bytes > 0
    _assert_artifact_matches_disk(
        genome.sequence.resolve(),
        genome.sequence.sha256,
        genome.sequence.size_bytes,
    )

    outputs = {item.role: item.artifact for item in manifest.outputs}
    assert {"assembly_fasta", "assembly_graph"} <= set(outputs)
    for artifact in outputs.values():
        _assert_artifact_matches_disk(artifact.resolve(), artifact.sha256, artifact.size_bytes)
    _assert_fasta_matches_gfa(
        outputs["assembly_fasta"].resolve(),
        outputs["assembly_graph"].resolve(),
    )

    plastid_roles = {role for role in outputs if role.startswith("detected_plastid_")}
    if require_plastid:
        assert {"detected_plastid_graph", "detected_plastid_path1"} <= plastid_roles
    if "detected_plastid_path1" in outputs:
        assert "detected_plastid_graph" in outputs
        _assert_fasta_matches_gfa(
            outputs["detected_plastid_path1"].resolve(),
            outputs["detected_plastid_graph"].resolve(),
        )
    if "detected_plastid_path2" in outputs:
        assert "detected_plastid_graph" in outputs
        _assert_fasta_matches_gfa(
            outputs["detected_plastid_path2"].resolve(),
            outputs["detected_plastid_graph"].resolve(),
        )

    raw_workspace = output / "backend" / "himt"
    assert raw_workspace.is_dir()
    himt_log = raw_workspace / "HiMT.log"
    flye_log = raw_workspace / "flye_output" / "flye.log"
    assert himt_log.is_file()
    assert himt_log.stat().st_size > 0
    assert flye_log.is_file()
    assert "Starting Flye 2.9.6" in flye_log.read_text(encoding="utf-8")

    for artifact in result.artifacts:
        _assert_artifact_matches_disk(artifact.resolve(), artifact.sha256, artifact.size_bytes)


def test_managed_himt_environment_install_and_probe() -> None:
    platform = normalize_platform()
    expected = os.environ["ORGANELLEVERSE_EXPECTED_CONDA_PLATFORM"]
    assert platform == expected
    manager = EnvironmentManager(cache_root=_cache_root())
    environment = manager.prepare(
        HIMT_ENVIRONMENT,
        policy="ensure",
        platform=platform,
    )
    assert environment.digest == manager.expected_environment_digest(
        HIMT_ENVIRONMENT,
        platform=platform,
    )
    assert environment.version == "HiMT 1.1.3"
    for name in ("himt", "flye", "makeblastdb", "tblastn", "minimap2", "miniprot"):
        assert environment.require_executable(name).is_file()


def test_real_hifi_himt_assembles_valid_mitochondrion_and_detects_plastid() -> None:
    fixture = prepare_himt_fixture("hifi", _cache_root())
    output, result = _run_real("hifi")
    _assert_evidence(
        output,
        result,
        fixture=fixture,
        expected_accuracy=0.8,
        require_plastid=True,
    )


def test_real_clr_himt_assembles_valid_mitochondrion() -> None:
    fixture = prepare_himt_fixture("clr", _cache_root())
    output, result = _run_real("clr")
    _assert_evidence(
        output,
        result,
        fixture=fixture,
        expected_accuracy=0.3,
        require_plastid=False,
    )


def test_real_ont_himt_assembles_valid_mitochondrion() -> None:
    fixture = prepare_himt_fixture("ont", _cache_root())
    output, result = _run_real("ont")
    _assert_evidence(
        output,
        result,
        fixture=fixture,
        expected_accuracy=0.3,
        require_plastid=False,
    )


def test_real_hifi_agent_json_preserves_evidence_chain() -> None:
    fixture = prepare_himt_fixture("hifi", _cache_root())
    _assert_expected_fixture(fixture)
    data = _data_for_fixture(fixture)
    output = _release_output(_cache_root(), "agent-hifi")
    response = invoke_json(
        {
            "operation_id": "assembly.assemble",
            "input": data.model_dump(mode="json"),
            "parameters": {
                "organelle": "mitochondrion",
                "method": "himt",
                "environment_source": "auto",
            },
        },
        registry=registry,
        granted_side_effects=set(ASSEMBLE_SPEC.side_effects),
    )
    assert response["ok"] is True, response.get("error")
    write_response = invoke_json(
        {
            "operation_id": "assembly.write",
            "input": response["result"],
            "parameters": {"output": str(output)},
        },
        registry=registry,
        granted_side_effects=set(ASSEMBLY_WRITE_SPEC.side_effects),
    )
    assert write_response["ok"] is True, write_response.get("error")
    transported = OrganelleResult.model_validate(write_response["result"])
    assert transported == load_result(output / "result.json")
    _assert_evidence(
        output,
        transported,
        fixture=fixture,
        expected_accuracy=0.8,
        require_plastid=True,
    )


class ForbiddenRunner(CommandRunner):
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
        del argv, stage, cwd, timeout_seconds, stdout_path, stderr_path, env
        raise AssertionError("exact result reuse must not launch a second HiMT process")


def _request_for_existing_hifi_run() -> AssemblyRequest:
    fixture = prepare_himt_fixture("hifi", _cache_root())
    return AssemblyRequest(
        data=_data_for_fixture(fixture),
        organelle="mitochondrion",
        method="himt",
        environment_source="auto",
    )


def test_real_himt_exact_reuse_avoids_second_process() -> None:
    _output, first = _run_real("hifi")
    request = _request_for_existing_hifi_run()
    second = execute_assembly(request, runner=ForbiddenRunner())
    assert second == first
