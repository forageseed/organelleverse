from __future__ import annotations

import importlib
from pathlib import Path

import organelleverse as ov
from organelleverse.operations import OperationRegistry, SideEffect
from organelleverse.operations.adapters import invoke_json
from organelleverse.quality_control.operations import ASSEMBLY_QC_SPEC, QC_WRITE_SPEC
from organelleverse.runtime import managed_runs_root

from .test_service import _install_deterministic_collectors
from .test_static import _publish_assembly_evidence


def test_qc_facade_exposes_only_canonical_actions() -> None:
    api = importlib.import_module("organelleverse.quality_control.api")

    assert ov.qc.annotation is api.annotation
    assert ov.qc.assembly is api.assembly
    assert ov.qc.write is api.write
    assert ov.qc.__all__ == [
        "annotation",
        "assembly",
        "compare_assembly_to_reference",
        "filter_long_reads",
        "filter_short_reads",
        "read_statistics",
        "write",
    ]


def test_direct_registry_and_agent_assembly_calls_agree(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)
    _install_deterministic_collectors(monkeypatch, tmp_path)
    registry = OperationRegistry()
    registry.register(ASSEMBLY_QC_SPEC, ov.qc.assembly)

    direct = ov.qc.assembly(published.result)
    assert direct.operation_version == ASSEMBLY_QC_SPEC.contract_version
    registered = registry.invoke(
        "qc.assembly",
        input=published.result,
        parameters={},
    )
    response = invoke_json(
        {
            "operation_id": "qc.assembly",
            "input": published.result.model_dump(mode="json"),
            "parameters": {},
        },
        registry=registry,
        granted_side_effects=set(ASSEMBLY_QC_SPEC.side_effects),
    )

    assert response["ok"] is True
    assert registered == direct
    assert response["result"] == direct.model_dump(mode="json")


def test_qc_assembly_publishes_into_the_managed_run_store(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)
    _install_deterministic_collectors(monkeypatch, tmp_path)
    isolated_cwd = tmp_path / "cwd"
    isolated_cwd.mkdir()
    monkeypatch.chdir(isolated_cwd)

    result = ov.qc.assembly(published.result)

    runs_root = managed_runs_root()
    assert all(Path(artifact.uri).is_relative_to(runs_root) for artifact in result.artifacts)
    assert all(artifact.resolve().is_file() for artifact in result.artifacts)
    # No user-selected destination is accepted, and no working-directory tree appears.
    assert not (isolated_cwd / ".organelleverse").exists()


def test_independent_qc_calls_reuse_one_managed_run(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)
    _install_deterministic_collectors(monkeypatch, tmp_path)
    import organelleverse.quality_control.service as service

    calls = 0
    original = service.collect_mapping_evidence

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "collect_mapping_evidence", counted)

    first = ov.qc.assembly(published.result)
    second = ov.qc.assembly(published.result)

    assert calls == 1
    assert second == first
    # The managed store holds exactly one published run for this operation.
    assert len(list((managed_runs_root() / "qc.assembly").iterdir())) == 1


def test_direct_registry_agent_and_top_level_write_use_qc_writer(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)
    _install_deterministic_collectors(monkeypatch, tmp_path)
    qc_result = ov.qc.assembly(published.result)
    registry = OperationRegistry()
    registry.register(QC_WRITE_SPEC, ov.qc.write)

    direct = ov.qc.write(qc_result, output=tmp_path / "direct")
    assert direct.operation_version == QC_WRITE_SPEC.contract_version
    registered = registry.invoke(
        "qc.write",
        input=qc_result,
        parameters={"output": tmp_path / "registry"},
    )
    response = invoke_json(
        {
            "operation_id": "qc.write",
            "input": qc_result.model_dump(mode="json"),
            "parameters": {"output": str(tmp_path / "agent")},
        },
        registry=registry,
        granted_side_effects={SideEffect.READ_FILES, SideEffect.WRITE_FILES},
    )
    top_level = ov.write(qc_result, tmp_path / "top-level")

    assert response["ok"] is True
    assert direct.operation_id == registered.operation_id == top_level.operation_id == "qc.write"
    assert response["result"]["operation_id"] == "qc.write"
    for candidate in ("registry", "agent", "top-level"):
        assert (tmp_path / candidate / "checks.csv").read_bytes() == (
            tmp_path / "direct" / "checks.csv"
        ).read_bytes()
