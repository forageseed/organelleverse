"""Independent benchmark, evaluator, and adoption gates for optimization v3."""

# pyright: reportArgumentType=false

from __future__ import annotations

import pytest
from pydantic import ValidationError

from organelleverse.optimization.contracts_v3 import (
    MetricConstraintV3,
    ObjectiveV3,
    OptimizationDeclarationV3,
)
from organelleverse.optimization.evaluation import (
    AdoptionPolicyV3,
    BenchmarkContractV3,
    BenchmarkSplitV3,
    EvaluatorContractV3,
)
from tests.optimization.test_contracts_v3 import (
    benchmark,
    declaration_payload,
    identity,
)

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64


@pytest.mark.parametrize("shared_field", ["split_id", "content_hash"])
def test_search_and_validation_must_be_independent(shared_field: str) -> None:
    search = BenchmarkSplitV3(
        split_id="search", content_hash=_HASH_A, case_count=4, artifact_ref="artifact:search"
    )
    validation_payload = {
        "split_id": "validation",
        "content_hash": _HASH_B,
        "case_count": 3,
        "artifact_ref": "artifact:validation",
    }
    validation_payload[shared_field] = getattr(search, shared_field)

    with pytest.raises(ValidationError):
        BenchmarkContractV3(
            benchmark_id="demo.benchmark",
            benchmark_version="1.0.0",
            search=search,
            validation=BenchmarkSplitV3(**validation_payload),
        )


def test_evaluator_and_adoption_policy_round_trip_without_scalar_score() -> None:
    evaluator = EvaluatorContractV3(identity=identity("demo.evaluator"))
    adoption = AdoptionPolicyV3(
        decision_rule="pareto_non_dominated",
        minimum_success_rate=0.8,
        maximum_failure_rate=0.2,
        maximum_wall_time_ratio=1.5,
        maximum_peak_memory_ratio=1.2,
        approval="human_required",
    )

    assert EvaluatorContractV3.model_validate(evaluator.model_dump()) == evaluator
    assert AdoptionPolicyV3.model_validate(adoption.model_dump()) == adoption
    assert evaluator.metrics_pointer == "/metrics/optimization_evaluation"
    assert not hasattr(adoption, "score")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_success_rate", 0),
        ("maximum_failure_rate", 1),
        ("maximum_wall_time_ratio", float("nan")),
        ("maximum_peak_memory_ratio", 0),
    ],
)
def test_adoption_policy_rejects_invalid_hard_gate_numbers(field: str, value: object) -> None:
    payload = {
        "decision_rule": "all_objectives",
        "minimum_success_rate": 0.8,
        "maximum_failure_rate": 0.2,
        "maximum_wall_time_ratio": 1.5,
        "maximum_peak_memory_ratio": 1.2,
        "approval": "automatic",
    }
    payload[field] = value
    with pytest.raises(ValidationError):
        AdoptionPolicyV3(**payload)


def test_objectives_and_constraints_cannot_alias_one_metric_pointer() -> None:
    payload = declaration_payload()
    pointer = "/metrics/optimization_evaluation/accuracy"
    payload["objectives"] = (
        ObjectiveV3(
            name="accuracy",
            metric_pointer=pointer,
            direction="maximize",
            minimum_improvement=0.01,
            improvement_mode="absolute",
            aggregation="mean",
        ),
    )
    payload["constraints"] = (
        MetricConstraintV3(
            name="accuracy_floor", metric_pointer=pointer, operator="ge", threshold=0.8
        ),
    )
    payload["benchmark"] = benchmark()

    with pytest.raises(ValidationError):
        OptimizationDeclarationV3(**payload)
