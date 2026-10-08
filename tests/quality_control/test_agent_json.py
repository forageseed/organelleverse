"""Agent JSON transport equivalence for the released QC operations.

The happy-path Result identity across direct Python, OperationRegistry, and
``invoke_json`` is locked in ``test_agent_contract.py``. These tests cover the
JSON-transport guarantees that are not implied by Result equality alone:

* independent invocation paths agree on every scientific field;
* the stable typed error envelope (``qc.*`` codes) is identical across all
  three paths, serialized unchanged by the Agent JSON adapter;
* side-effect grants are enforced only by the permission-aware JSON transport;
* the write contract error round-trips through Agent JSON as well.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.result import OrganelleResult
from organelleverse.operations import OperationRegistry
from organelleverse.operations.adapters import invoke_json
from organelleverse.quality_control.operations import ASSEMBLY_QC_SPEC, QC_WRITE_SPEC
from organelleverse.runtime import managed_runs_root

from .test_service import _install_deterministic_collectors
from .test_static import _publish_assembly_evidence

_GRANTS = set(ASSEMBLY_QC_SPEC.side_effects)


def _registry() -> OperationRegistry:
    registry = OperationRegistry()
    registry.register(ASSEMBLY_QC_SPEC, ov.qc.assembly)
    registry.register(QC_WRITE_SPEC, ov.qc.write)
    return registry


def _report_payload(result: OrganelleResult) -> dict[str, object]:
    artifact = next(item for item in result.artifacts if item.kind == "assembly_qc_report")
    return json.loads(Path(artifact.uri).read_bytes())


def _agent_assembly(registry, result: OrganelleResult, *, granted) -> dict[str, object]:
    return invoke_json(
        {
            "operation_id": "qc.assembly",
            "input": result.model_dump(mode="json"),
            "parameters": {},
        },
        registry=registry,
        granted_side_effects=granted,
    )


def test_independent_runs_agree_on_every_scientific_field(monkeypatch, tmp_path: Path) -> None:
    """Direct, Registry, and Agent JSON invocations agree on every field."""
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)
    _install_deterministic_collectors(monkeypatch, tmp_path)
    registry = _registry()

    direct = ov.qc.assembly(published.result)
    registered = registry.invoke(
        "qc.assembly",
        input=published.result,
        parameters={},
    )
    response = _agent_assembly(registry, published.result, granted=_GRANTS)
    assert response["ok"] is True
    agent = OrganelleResult.model_validate(response["result"])

    # The three invocation paths resolve to the same managed run, so the
    # scientific fields must agree across direct Python, Registry, and Agent.
    assert agent.metrics == direct.metrics == registered.metrics
    assert (
        [finding.code for finding in agent.findings]
        == [finding.code for finding in direct.findings]
        == [finding.code for finding in registered.findings]
    )

    direct_report = _report_payload(direct)
    agent_report = _report_payload(agent)
    registered_report = _report_payload(registered)
    for field in (
        "decision",
        "summary",
        "checks",
        "sequence_summaries",
        "junction_support",
        "repeat_support",
        "marker_hits",
        "error_candidates",
        "coverage_windows",
    ):
        assert direct_report[field] == agent_report[field] == registered_report[field]


def test_input_contract_error_envelope_is_identical_across_paths(
    monkeypatch, tmp_path: Path
) -> None:
    """A typed ``qc.input_contract_violation`` is raised unchanged by direct and
    Registry invocation, and serialized with the same code by Agent JSON."""
    _install_deterministic_collectors(monkeypatch, tmp_path)
    registry = _registry()
    bad_input = OrganelleResult(
        operation_id="not.assembly",
        scope="mitochondrion",
        status="ok",
    )

    with pytest.raises(OrganelleInputError) as direct_exc:
        ov.qc.assembly(bad_input)
    with pytest.raises(OrganelleInputError) as registry_exc:
        registry.invoke(
            "qc.assembly",
            input=bad_input,
            parameters={},
        )
    response = _agent_assembly(registry, bad_input, granted=_GRANTS)

    code = "qc.input_contract_violation"
    assert direct_exc.value.code == code
    assert registry_exc.value.code == code
    assert response["ok"] is False
    assert response["error"]["error_code"] == code


def test_side_effect_grant_denial_returns_permission_error(monkeypatch, tmp_path: Path) -> None:
    """Only the permission-aware JSON transport enforces side-effect grants;
    denying subprocess produces ``permission.denied`` without invoking science."""
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)
    _install_deterministic_collectors(monkeypatch, tmp_path)
    registry = _registry()

    response = _agent_assembly(
        registry,
        published.result,
        granted={side for side in ASSEMBLY_QC_SPEC.side_effects if side != "subprocess"},
    )

    assert response["ok"] is False
    assert response["error"]["error_code"] == "permission.denied"
    assert "subprocess" in response["error"]["details"]["missing_side_effects"]
    # Nothing was written: no managed run is committed when science never runs.
    assert not (managed_runs_root() / "qc.assembly").exists()


def test_write_contract_error_envelope_round_trips_via_agent(monkeypatch, tmp_path: Path) -> None:
    """A non-QC Result fed to ``qc.write`` surfaces ``qc.write_input_contract``
    with the same code through direct and Agent JSON paths."""
    source = tmp_path / "source"
    source.mkdir()
    _publish_assembly_evidence(source)
    registry = _registry()
    non_qc = OrganelleResult(
        operation_id="assembly.assemble",
        scope="mitochondrion",
        status="ok",
    )

    with pytest.raises(OrganelleInputError) as direct_exc:
        ov.qc.write(non_qc, output=tmp_path / "direct")
    response = invoke_json(
        {
            "operation_id": "qc.write",
            "input": non_qc.model_dump(mode="json"),
            "parameters": {"output": str(tmp_path / "agent")},
        },
        registry=registry,
        granted_side_effects=set(QC_WRITE_SPEC.side_effects),
    )

    assert direct_exc.value.code == "qc.write_input_contract"
    assert response["ok"] is False
    assert response["error"]["error_code"] == "qc.write_input_contract"
