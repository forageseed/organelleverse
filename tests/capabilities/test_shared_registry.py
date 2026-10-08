"""The public Python builder exposes the same checked registry used by Agents."""

from organelleverse.capabilities import CapabilityIndex, build_registry
from organelleverse.operations.adapters.json import invoke_json


def test_public_registry_keeps_release_contract_and_structured_errors() -> None:
    registry = build_registry(CapabilityIndex())
    assert {spec.operation_id for spec in registry.list()} == {
        "annotation.annotate",
        "assembly.assemble",
        "assembly.pmat_graph_build",
        "io.read_long_reads",
    }
    schema = registry.parameter_schema("io.read_long_reads")
    assert set(schema["required"]) == {"reads", "technology", "quality_state"}

    response = invoke_json(
        {"operation_id": "unknown.operation", "input": None, "parameters": {}},
        registry=registry,
        granted_side_effects=(),
    )
    assert response["ok"] is False
    assert response["error"]["error_code"] == "input.unknown_operation"
