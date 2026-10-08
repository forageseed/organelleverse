from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import pytest
from pydantic import ValidationError

from organelleverse.assembly.backends.pmat_graph import PmatGraphAdapter, PmatGraphContext
from organelleverse.assembly.continuation import DirectoryManifest, DirectoryManifestFile
from organelleverse.assembly.data_contract import (
    pmat_graph_input_data_contract,
    validate_pmat_graph_build_data,
)
from organelleverse.assembly.environment_contracts import EnvironmentHint
from organelleverse.assembly.environment_specs import AssemblyEnvironmentSpec
from organelleverse.assembly.environments import (
    EnvironmentManager,
    PreparedEnvironment,
    PreparedExecutable,
)
from organelleverse.assembly.manifests import AssemblyRunManifest
from organelleverse.assembly.operations import PMAT_GRAPH_BUILD_SPEC
from organelleverse.assembly.pmat_graph import (
    PmatGraphBuildParameters,
    PmatGraphEnvironmentHint,
    execute_pmat_graph_build,
    pmat_graph_build,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleExecutionError, OrganelleInputError
from organelleverse.core.result import OrganelleResult
from organelleverse.operations import OperationRegistry
from organelleverse.runtime import managed_runs_root

_FAKE_PMAT = r"""#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("command")
p.add_argument("-i"); p.add_argument("-a"); p.add_argument("-o")
p.add_argument("-G"); p.add_argument("-x"); p.add_argument("-d")
p.add_argument("-s", nargs="*"); p.add_argument("-T")
a = p.parse_args()
if os.environ.get("PMAT_GRAPH_FAIL") == "1":
    sys.stderr.write("failed\n"); sys.exit(2)
root = Path(a.o); (root / "gfa_result").mkdir(parents=True)
target = a.G
fasta = f"PMAT_{target}.fa"
(root / "gfa_result" / fasta).write_text(">ctg\nACGTACGT\n")
(root / "gfa_result" / f"PMAT_{target}_main.gfa").write_text("H\tVN:Z:1.0\nS\tctg\tACGTACGT\n")
(root / "gfa_result" / f"PMAT_{target}_raw.gfa").write_text("H\tVN:Z:1.0\nS\tctg\tACGTACGT\n")
if target == "mt":
    (root / "PMAT_orgAss.txt").write_text("Mitochondrial assembly\n")
"""


def _artifact(path: Path, content: bytes, *, fmt: str) -> ArtifactRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return ArtifactRef.from_path(
        path,
        kind="pmat_continuation",
        format=fmt,
        media_type="application/octet-stream",
    )


def _data(tmp_path: Path, *, mutate: bool = False) -> OrganelleData:
    source = tmp_path / "source"
    subsample = _artifact(source / "subsample.fa", b">read\nACGT\n", fmt="fasta")
    contigs = _artifact(source / "all.fna", b">ctg\nACGTACGT\n", fmt="fasta")
    graph = _artifact(source / "graph.txt", b"1\t2\n", fmt="txt")
    subsample_manifest = DirectoryManifest(
        role="pmat_subsample",
        files=(
            DirectoryManifestFile(
                relative_path="PMAT_cut_seq.fa",
                artifact_role="pmat_subsample",
                sha256=subsample.sha256,
            ),
        ),
    )
    assembly_manifest = DirectoryManifest(
        role="pmat_assembly_result",
        files=(
            DirectoryManifestFile(
                relative_path="PMATAllContigs.fna",
                artifact_role="pmat_all_contigs",
                sha256=contigs.sha256,
            ),
            DirectoryManifestFile(
                relative_path="PMATContigGraph.txt",
                artifact_role="pmat_contig_graph",
                sha256=graph.sha256,
            ),
        ),
    )
    manifests = tmp_path / "manifests"
    sub_manifest = _artifact(
        manifests / "subsample.json", subsample_manifest.canonical_bytes(), fmt="json"
    )
    asm_manifest = _artifact(
        manifests / "assembly.json", assembly_manifest.canonical_bytes(), fmt="json"
    )
    if mutate:
        Path(subsample.uri).write_bytes(b">read\nTTTT\n")
    return OrganelleData.model_validate(
        {
            "modality": "pmat_graph_input",
            "artifacts": {
                "pmat_subsample": subsample,
                "pmat_all_contigs": contigs,
                "pmat_contig_graph": graph,
                "subsample_manifest": sub_manifest,
                "assembly_result_manifest": asm_manifest,
            },
            "payload": {
                "contract_version": "organelleverse.pmat-graph-input.v1",
                "subsample_manifest_artifact": "subsample_manifest",
                "assembly_result_manifest_artifact": "assembly_result_manifest",
                "file_artifact_roles": [
                    "pmat_subsample",
                    "pmat_all_contigs",
                    "pmat_contig_graph",
                ],
            },
        }
    )


class _Manager(EnvironmentManager):
    def __init__(self, root: Path) -> None:
        super().__init__(cache_root=root)
        executable = root / "bin" / "PMAT"
        executable.parent.mkdir(parents=True)
        executable.write_text(_FAKE_PMAT)
        executable.chmod(0o755)
        self.environment = PreparedEnvironment(
            backend_id="pmat",
            carrier="conda",
            platform="linux-64",
            digest="sha256:" + "a" * 64,
            prefix=root,
            executables=(PreparedExecutable(name="pmat", path=executable),),
            version="2.1.5",
        )

    def prepare(
        self,
        spec: AssemblyEnvironmentSpec,
        *,
        policy: Literal["ensure", "require"],
        platform: str | None = None,
    ) -> PreparedEnvironment:
        return self.environment


def _context(tmp_path: Path) -> PmatGraphContext:
    manager = _Manager(tmp_path / "env")
    workspace = tmp_path / "workspace"
    (workspace / "subsample").mkdir(parents=True)
    (workspace / "assembly_result").mkdir()
    return PmatGraphContext(
        workspace=workspace,
        environment=manager.environment,
        subsample_dir=workspace / "subsample",
        assembly_result_dir=workspace / "assembly_result",
        organelle="mitochondrion",
        taxon_group="plant",
        depth=12.5,
        seeds=(3, 9, 14),
        threads=6,
    )


def test_graph_build_operation_has_closed_agent_schema_and_is_released() -> None:
    registry = OperationRegistry(data_contracts=(pmat_graph_input_data_contract(),))
    registry.register(PMAT_GRAPH_BUILD_SPEC, pmat_graph_build)
    schema = registry.parameter_schema("assembly.pmat_graph_build")
    assert schema["additionalProperties"] is False
    properties = cast(dict[str, object], schema["properties"])
    assert set(properties) == {
        "organelle",
        "taxon_group",
        "depth",
        "seeds",
        "threads",
        "timeout_seconds",
        "environment_source",
        "backend_version",
        "environment_hint",
    }
    assert '"additionalProperties": false' in json.dumps(schema)
    from organelleverse.operations.registry import registry as released

    assert PMAT_GRAPH_BUILD_SPEC.operation_id in {spec.operation_id for spec in released.list()}


def test_contract_and_parameters_fail_closed(tmp_path: Path) -> None:
    assert validate_pmat_graph_build_data(_data(tmp_path)).file_artifact_roles
    with pytest.raises(ValidationError):
        PmatGraphBuildParameters(organelle="mitochondrion", seeds=(1, 1))
    with pytest.raises(ValidationError):
        PmatGraphBuildParameters(organelle="mitochondrion", depth=-1)
    with pytest.raises(ValidationError):
        PmatGraphEnvironmentHint(prefix=Path("relative"), unknown="value")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        EnvironmentHint.model_validate(
            PmatGraphEnvironmentHint(prefix=Path("relative")).model_dump(mode="python")
        )
    data = _data(tmp_path / "bad")
    raw = dict(data.payload)
    raw["file_artifact_roles"] = ("pmat_subsample",)
    invalid = OrganelleData.model_validate(
        {"modality": data.modality, "artifacts": dict(data.artifacts), "payload": raw}
    )
    with pytest.raises(OrganelleInputError):
        validate_pmat_graph_build_data(invalid)


def test_registry_forwards_exact_version_and_closed_environment_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_execute(data: OrganelleData, **kwargs: object) -> OrganelleResult:
        captured.update(kwargs)
        return OrganelleResult(
            operation_id="assembly.pmat_graph_build",
            scope="mitochondrion",
            status="ok",
        )

    monkeypatch.setattr(
        "organelleverse.assembly.pmat_graph.execute_pmat_graph_build",
        fake_execute,
    )
    registry = OperationRegistry(data_contracts=(pmat_graph_input_data_contract(),))
    registry.register(PMAT_GRAPH_BUILD_SPEC, pmat_graph_build)
    result = cast(
        OrganelleResult,
        registry.invoke(
            "assembly.pmat_graph_build",
            input=_data(tmp_path),
            parameters={
                "organelle": "mitochondrion",
                "backend_version": "2.8.3",
                "environment_hint": {"prefix": tmp_path.resolve()},
            },
        ),
    )
    assert result.status == "ok"
    assert captured["backend_version"] == "2.8.3"
    assert captured["environment_hint"] == EnvironmentHint(prefix=tmp_path.resolve())


def test_adapter_emits_every_real_graphbuild_flag(tmp_path: Path) -> None:
    command = PmatGraphAdapter().build_command(_context(tmp_path))
    assert command.stable_argv == (
        "PMAT",
        "graphBuild",
        "-i",
        "role://workspace/subsample",
        "-a",
        "role://workspace/assembly_result",
        "-o",
        "role://workspace/output",
        "-G",
        "mt",
        "-x",
        "0",
        "-d",
        "12.5",
        "-s",
        "3",
        "9",
        "14",
        "-T",
        "6",
    )
    assert command.resolved_argv[0].endswith("/bin/PMAT")
    assert not any("role://" in token for token in command.resolved_argv)


def test_execute_success_reuse_and_tamper_detection(tmp_path: Path) -> None:
    from organelleverse.assembly.environment_specs import PMAT_ORIENTATION_VERSION

    data = _data(tmp_path)
    manager = _Manager(tmp_path / "env")
    manager.environment = manager.environment.model_copy(
        update={"software_version": PMAT_ORIENTATION_VERSION}
    )
    first = execute_pmat_graph_build(
        data,
        organelle="mitochondrion",
        depth=2.5,
        seeds=(1, 2),
        environment_manager=manager,
    )
    output = Path(first.artifacts[0].uri).parent
    assert first.status == "ok"
    assert first.provenance.software_versions["pmat"] == PMAT_ORIENTATION_VERSION
    manifest = AssemblyRunManifest.model_validate_json(
        (output / "assembly_run_record.json").read_bytes()
    )
    assert manifest.operation_id == "assembly.pmat_graph_build"
    assert manifest.selected_backend == "pmat"
    assert "role://workspace/subsample" in manifest.stable_argv
    second = execute_pmat_graph_build(
        data,
        organelle="mitochondrion",
        depth=2.5,
        seeds=(1, 2),
        environment_manager=manager,
    )
    assert second.provenance == first.provenance
    (output / "normalized" / "assembly.fasta").write_text(">ctg\nTTTT\n")
    with pytest.raises(OrganelleExecutionError, match="not reusable"):
        execute_pmat_graph_build(
            data,
            organelle="mitochondrion",
            depth=2.5,
            seeds=(1, 2),
            environment_manager=manager,
        )


def test_execute_plastid_does_not_require_mitochondrial_assessment(tmp_path: Path) -> None:
    result = execute_pmat_graph_build(
        _data(tmp_path / "plastid"),
        organelle="plastid",
        environment_manager=_Manager(tmp_path / "plastid-env"),
    )
    output = Path(result.artifacts[0].uri).parent
    assert result.status == "ok"
    assert not (output / "normalized" / "assessment.txt").exists()


def test_started_failure_is_a_failed_result_and_input_tamper_is_prestart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PMAT_GRAPH_FAIL", "1")
    result = execute_pmat_graph_build(
        _data(tmp_path / "failed"),
        organelle="mitochondrion",
        environment_manager=_Manager(tmp_path / "failed-env"),
    )
    assert result.status == "failed"
    assert result.errors[0].suggested_action["failed_stage"] == "execute_backend"
    with pytest.raises(OrganelleExecutionError):
        execute_pmat_graph_build(
            _data(tmp_path / "tampered", mutate=True),
            organelle="mitochondrion",
            environment_manager=_Manager(tmp_path / "tampered-env"),
        )
    # The tampered input raised before any managed run was committed for it.
    assert len(list((managed_runs_root() / "assembly.pmat_graph_build").glob("sha256-*"))) == 1
