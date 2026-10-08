"""Strict and canonical v3 optimization control-plane contracts."""

# pyright: reportArgumentType=false, reportCallIssue=false, reportUnknownArgumentType=false, reportOperatorIssue=false

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from organelleverse.optimization.contracts_v3 import (
    BaselineCandidateV3,
    MetricConstraintV3,
    ObjectiveV3,
    OptimizationBudgetV3,
    OptimizationContractV3,
    OptimizationDeclarationV3,
    OptimizationProfileV3,
    PriorCandidateV3,
    RepeatPolicyV3,
    StrategyReferenceV3,
)
from organelleverse.optimization.evaluation import (
    AdoptionPolicyV3,
    BenchmarkContractV3,
    BenchmarkSplitV3,
    CapabilityIdentityV3,
    EvaluatorContractV3,
)
from organelleverse.optimization.models import ParameterDomain

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64
_HASH_C = "sha256:" + "c" * 64


def identity(capability_id: str = "demo.target") -> CapabilityIdentityV3:
    return CapabilityIdentityV3(
        capability_id=capability_id,
        bundle_version="1.2.3",
        contract_version="1.0",
        bundle_content_hash=_HASH_A if capability_id == "demo.target" else _HASH_B,
        execution_identity=None,
        implementation="native",
        surface="native",
    )


def benchmark() -> BenchmarkContractV3:
    return BenchmarkContractV3(
        benchmark_id="demo.benchmark",
        benchmark_version="1.0.0",
        search=BenchmarkSplitV3(
            split_id="search-v1", content_hash=_HASH_A, case_count=8, artifact_ref="artifact:a"
        ),
        validation=BenchmarkSplitV3(
            split_id="validation-v1",
            content_hash=_HASH_B,
            case_count=4,
            artifact_ref="artifact:b",
        ),
    )


def declaration_payload() -> dict[str, object]:
    return {
        "parameters": (
            ParameterDomain(name="iterations", kind="integer", minimum=2, maximum=8, step=2),
        ),
        "baseline": BaselineCandidateV3(parameters={"iterations": 2}),
        "priors": (),
        "objectives": (
            ObjectiveV3(
                name="accuracy",
                metric_pointer="/metrics/optimization_evaluation/accuracy",
                direction="maximize",
                minimum_improvement=0.01,
                improvement_mode="absolute",
                aggregation="mean",
            ),
        ),
        "constraints": (
            MetricConstraintV3(
                name="memory",
                metric_pointer="/metrics/optimization_evaluation/peak_memory",
                operator="le",
                threshold=2.0,
            ),
        ),
        "strategies": (StrategyReferenceV3(kind="grid", identity=identity("strategy.grid")),),
        "budget": OptimizationBudgetV3(
            max_trials=8,
            parallelism=2,
            max_wall_time_seconds=60,
            max_cpu_time_seconds=120,
            max_peak_memory_bytes=1024,
        ),
        "repeats": RepeatPolicyV3(search_repeats=1, validation_repeats=2, seed=7),
        "adoption": AdoptionPolicyV3(
            decision_rule="all_objectives",
            minimum_success_rate=0.8,
            maximum_failure_rate=0.2,
            maximum_wall_time_ratio=1.5,
            maximum_peak_memory_ratio=1.2,
            approval="human_required",
        ),
        "benchmark": benchmark(),
    }


def contract(**updates: object) -> OptimizationContractV3:
    payload = declaration_payload()
    payload.update(updates)
    return OptimizationContractV3(
        target=identity(),
        evaluator=EvaluatorContractV3(identity=identity("demo.evaluator")),
        **payload,
    )


def test_v3_contract_round_trips_strictly_with_stable_digest() -> None:
    first = contract()
    second = OptimizationContractV3.model_validate(first.model_dump(mode="json"))

    assert first.schema_version == "organelleverse.optimization.contract.v3"
    assert first == second
    assert first.digest == second.digest
    assert first.digest.startswith("sha256:")
    assert not hasattr(first, "score")


@pytest.mark.parametrize(
    ("model", "field", "bad"),
    [
        (OptimizationBudgetV3, "max_trials", "8"),
        (OptimizationBudgetV3, "parallelism", True),
        (RepeatPolicyV3, "seed", 1.0),
        (ObjectiveV3, "minimum_improvement", "0.1"),
        (MetricConstraintV3, "threshold", float("inf")),
    ],
)
def test_v3_control_fields_reject_coercion_and_non_finite_numbers(
    model: type[object], field: str, bad: object
) -> None:
    examples: dict[type[object], dict[str, object]] = {
        OptimizationBudgetV3: {
            "max_trials": 8,
            "parallelism": 2,
            "max_wall_time_seconds": 60,
            "max_cpu_time_seconds": 120,
            "max_peak_memory_bytes": 1024,
        },
        RepeatPolicyV3: {"search_repeats": 1, "validation_repeats": 2, "seed": 7},
        ObjectiveV3: {
            "name": "accuracy",
            "metric_pointer": "/metrics/optimization_evaluation/accuracy",
            "direction": "maximize",
            "minimum_improvement": 0.1,
            "improvement_mode": "absolute",
            "aggregation": "mean",
        },
        MetricConstraintV3: {
            "name": "memory",
            "metric_pointer": "/metrics/optimization_evaluation/memory",
            "operator": "le",
            "threshold": 2.0,
        },
    }
    payload = deepcopy(examples[model])
    payload[field] = bad

    with pytest.raises(ValidationError):
        model(**payload)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "field,value",
    [
        ("parameters", (ParameterDomain(name="x", kind="boolean"),) * 2),
        ("objectives", declaration_payload()["objectives"] * 2),
        ("constraints", declaration_payload()["constraints"] * 2),
        (
            "strategies",
            (
                StrategyReferenceV3(kind="grid", identity=identity("strategy.grid")),
                StrategyReferenceV3(kind="grid", identity=identity("strategy.random")),
            ),
        ),
    ],
)
def test_v3_declaration_rejects_duplicate_definitions(field: str, value: object) -> None:
    payload = declaration_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        OptimizationDeclarationV3(**payload)


def test_strategy_identity_must_also_be_unique() -> None:
    payload = declaration_payload()
    shared = identity("strategy.search")
    payload["strategies"] = (
        StrategyReferenceV3(kind="grid", identity=shared),
        StrategyReferenceV3(kind="random", identity=shared),
    )

    with pytest.raises(ValidationError):
        OptimizationDeclarationV3(**payload)


def test_v3_digest_normalizes_semantically_equal_numbers() -> None:
    integer = contract(
        constraints=(
            MetricConstraintV3(
                name="memory",
                metric_pointer="/metrics/optimization_evaluation/peak_memory",
                operator="le",
                threshold=2,
            ),
        )
    )
    floating = contract()

    assert integer.digest == floating.digest


def test_enabled_profile_cannot_exist_without_a_complete_contract() -> None:
    with pytest.raises(ValidationError):
        OptimizationProfileV3(status="enabled", target=identity(), contract=None, reason_code=None)


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"iterations": 2, "unknown": 1},
        {"iterations": 3},
        {"iterations": True},
    ],
)
def test_baseline_must_exactly_cover_closed_domains(parameters: dict[str, object]) -> None:
    payload = declaration_payload()
    payload["baseline"] = {"parameters": parameters}

    with pytest.raises(ValidationError):
        OptimizationDeclarationV3(**payload)


def test_every_prior_is_validated_and_minegraph_remains_an_initial_candidate() -> None:
    payload = declaration_payload()
    payload["priors"] = (
        PriorCandidateV3(
            prior_id="minegraph-compatible.v1",
            source="minegraph",
            parameters={"iterations": 4},
            evidence_refs=("artifact:recommendation",),
        ),
    )
    declaration = OptimizationDeclarationV3(**payload)

    assert declaration.priors[0].role == "initial_candidate"
    assert not hasattr(declaration.priors[0], "score")
    assert not hasattr(declaration.priors[0], "adopted")
    with pytest.raises(ValidationError):
        PriorCandidateV3(
            prior_id="minegraph-compatible.v1",
            source="minegraph",
            parameters={"iterations": 4},
            evidence_refs=("artifact:recommendation",),
            score=1.0,
        )


@pytest.mark.parametrize("prior_id", ["", "   "])
def test_prior_id_rejects_blank_values(prior_id: str) -> None:
    with pytest.raises(ValidationError):
        PriorCandidateV3(
            prior_id=prior_id,
            source="historical",
            parameters={"iterations": 4},
            evidence_refs=("artifact:history",),
        )


def test_duplicate_priors_are_rejected() -> None:
    payload = declaration_payload()
    payload["priors"] = (
        PriorCandidateV3(
            prior_id="prior-v1",
            source="historical",
            parameters={"iterations": 4},
            evidence_refs=("artifact:one",),
        ),
        PriorCandidateV3(
            prior_id="prior-v1",
            source="declared",
            parameters={"iterations": 10},
            evidence_refs=("artifact:two",),
        ),
    )

    with pytest.raises(ValidationError):
        OptimizationDeclarationV3(**payload)


def test_out_of_domain_prior_is_rejected() -> None:
    payload = declaration_payload()
    payload["priors"] = (
        PriorCandidateV3(
            prior_id="outside-v1",
            source="historical",
            parameters={"iterations": 10},
            evidence_refs=("artifact:outside",),
        ),
    )

    with pytest.raises(ValidationError):
        OptimizationDeclarationV3(**payload)


def test_categorical_candidate_membership_uses_canonical_numeric_semantics() -> None:
    payload = declaration_payload()
    payload["parameters"] = (ParameterDomain(name="mode", kind="categorical", values=(1, 2)),)
    payload["baseline"] = BaselineCandidateV3(parameters={"mode": 1.0})

    declaration = OptimizationDeclarationV3(**payload)

    assert declaration.baseline.parameters == {"mode": 1.0}


def test_v3_contract_surface_is_exported_from_optimization_package() -> None:
    import organelleverse.optimization as optimization

    expected = {
        "AdoptionPolicyV3",
        "BaselineCandidateV3",
        "BenchmarkContractV3",
        "BenchmarkSplitV3",
        "CapabilityIdentityV3",
        "EvaluatorContractV3",
        "MetricConstraintV3",
        "ObjectiveV3",
        "OptimizationBudgetV3",
        "OptimizationContractV3",
        "OptimizationDeclarationV3",
        "OptimizationProfileV3",
        "PriorCandidateV3",
        "RepeatPolicyV3",
        "StrategyReferenceV3",
        "classify_optimization_profile",
        "contract_v3_from_v2",
        "identity_from_capability_entry",
    }

    assert expected <= set(optimization.__all__)
    assert all(hasattr(optimization, name) for name in expected)
