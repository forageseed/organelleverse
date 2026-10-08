import pytest

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.registry import OperationRegistry
from organelleverse.optimization.profiles_v3 import identity_from_capability_entry


@pytest.fixture
def native_index(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    index = discover_capabilities()
    store = VerificationStore(home / "verifications")
    for kind in ("grid", "random", "space_filling", "successive_halving"):
        verify_capability(
            "optimization." + kind, store=store, environment=LocalVerificationEnvironment(index)
        )
    return discover_capabilities()


@pytest.mark.parametrize("kind", ["grid", "random", "space_filling"])
def test_admitted_strategy_bounds_and_resumes_without_scores(native_index, kind):
    entry = native_index.describe("optimization." + kind)
    binding = {
        "kind": kind,
        "identity": identity_from_capability_entry(entry).model_dump(mode="json"),
    }
    payload = {
        "binding": binding,
        "spec": {
            "parameters": [
                {
                    "name": "segment_length",
                    "kind": "integer",
                    "minimum": 100,
                    "maximum": 400,
                    "step": 100,
                }
            ],
            "seed": 932,
            "total_design_size": 4,
        },
        "remaining_budget": 2,
    }
    source = OrganelleResult(
        operation_id="optimization.prepare",
        status="ok",
        scope="mitochondrion",
        metrics={"optimization_strategy": payload},
    )
    op = OperationRegistry(capability_source=native_index.binding_source()).require(
        "optimization." + kind
    )
    first = op.invoke(source, {}).model_dump(mode="json")["metrics"]["optimization_proposals"]
    assert len(first["proposals"]) == 2
    assert all(
        set(row) <= {"parameters", "candidate_digest", "base_candidate_digest", "resource_rung"}
        for row in first["proposals"]
    )
    again = op.invoke(source, {}).model_dump(mode="json")["metrics"]["optimization_proposals"]
    assert first == again
    payload["state"] = first["state"]
    resumed = op.invoke(
        source.model_copy(update={"metrics": {"optimization_strategy": payload}}), {}
    )
    second = resumed.model_dump(mode="json")["metrics"]["optimization_proposals"]
    assert len(second["proposals"]) == 2 and second["exhausted"]
    assert {p["candidate_digest"] for p in first["proposals"]}.isdisjoint(
        p["candidate_digest"] for p in second["proposals"]
    )


def test_admitted_halving_waits_for_measured_observations(native_index):
    kinds = ("successive_halving", "grid")
    bindings = {
        kind: {
            "kind": kind,
            "identity": identity_from_capability_entry(
                native_index.describe("optimization." + kind)
            ).model_dump(mode="json"),
        }
        for kind in kinds
    }
    config = {
        "resource_parameter": "iterations",
        "rungs": [1, 2],
        "eta": 2,
        "base_strategy": bindings["grid"],
    }
    payload = {
        "binding": bindings["successive_halving"],
        "spec": {
            "parameters": [
                {"name": "width", "kind": "integer", "minimum": 1, "maximum": 2, "step": 1},
                {"name": "iterations", "kind": "integer", "minimum": 1, "maximum": 2, "step": 1},
            ],
            "seed": 1,
            "total_design_size": 2,
            "successive_halving": config,
            "objective_order": [["accuracy", "maximize"]],
        },
        "remaining_budget": 4,
    }
    source = OrganelleResult(
        operation_id="optimization.prepare",
        status="ok",
        scope="none",
        metrics={"optimization_strategy": payload},
    )
    op = OperationRegistry(capability_source=native_index.binding_source()).require(
        "optimization.successive_halving"
    )
    first = op.invoke(source, {}).model_dump(mode="json")["metrics"]["optimization_proposals"]
    assert all(row["resource_rung"] == 1 for row in first["proposals"])
    payload["state"] = first["state"]
    with pytest.raises(Exception, match="observe the current complete rung"):
        op.invoke(source.model_copy(update={"metrics": {"optimization_strategy": payload}}), {})
    payload["observations"] = [
        {
            "candidate_digest": row["candidate_digest"],
            "base_candidate_digest": row["base_candidate_digest"],
            "resource_rung": 1,
            "metrics": {"accuracy": float(row["parameters"]["width"])},
            "feasible": True,
        }
        for row in first["proposals"]
    ]
    promoted = op.invoke(
        source.model_copy(update={"metrics": {"optimization_strategy": payload}}), {}
    ).model_dump(mode="json")["metrics"]["optimization_proposals"]
    assert len(promoted["proposals"]) == 1
    assert promoted["proposals"][0]["parameters"] == {"width": 2, "iterations": 2}
