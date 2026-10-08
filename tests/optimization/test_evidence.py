from __future__ import annotations

import math
from typing import Literal

import pytest
from pydantic import ValidationError

from organelleverse.core.result import ErrorDetail
from organelleverse.optimization.contracts_v3 import MetricConstraintV3, ObjectiveV3
from organelleverse.optimization.evaluation import (
    AdoptionPolicyV3,
    BenchmarkSplitV3,
    CapabilityIdentityV3,
)
from organelleverse.optimization.evidence import (
    CandidateAggregate,
    MetricAggregate,
    aggregate_repeats,
    decide_adoption,
    minimum_improvements_met,
    pareto_front,
)
from organelleverse.optimization.strategy_models import EvaluatorBinding
from organelleverse.plugin_experiments.adaptive_models import AttemptIdentity, RepeatEvidence


def objective(name: str, direction: Literal["maximize", "minimize"]) -> ObjectiveV3:
    return ObjectiveV3(
        name=name,
        metric_pointer=f"/metrics/optimization_evaluation/{name}",
        direction=direction,
        minimum_improvement=0.0,
        improvement_mode="absolute",
        aggregation="mean",
    )


def pointer(name: str) -> str:
    return f"/metrics/optimization_evaluation/{name}"


def validation_split() -> BenchmarkSplitV3:
    return BenchmarkSplitV3(
        split_id="validation-v1",
        content_hash="sha256:" + "f" * 64,
        case_count=4,
        artifact_ref="artifact:validation",
    )


def evaluator_binding() -> EvaluatorBinding:
    return EvaluatorBinding(
        identity=CapabilityIdentityV3(
            capability_id="demo.evaluator",
            bundle_version="1.0.0",
            contract_version="1.0",
            bundle_content_hash="sha256:" + "d" * 64,
            execution_identity=None,
            implementation="native",
            surface="native",
        ),
        evaluator_digest="sha256:" + "e" * 64,
    )


def aggregate(digest_digit: str, accuracy: float, runtime: float) -> CandidateAggregate:
    metrics = {
        (name if name.endswith(("_seconds", "_bytes")) else pointer(name)): MetricAggregate(
            value=value,
            variance=0.0,
            successful_runs=2,
            attempted_runs=2,
            failure_rate=0.0,
        )
        for name, value in {
            "accuracy": accuracy,
            "runtime": runtime,
            "memory": 10.0,
            "wall_time_seconds": 1.0,
            "cpu_time_seconds": 0.5,
            "peak_memory_bytes": 10.0,
        }.items()
    }
    return CandidateAggregate(
        split="validation",
        split_id="validation-v1",
        split_content_hash="sha256:" + "f" * 64,
        candidate_digest="sha256:" + digest_digit * 64,
        parameter_digest="sha256:" + digest_digit * 64,
        evaluator_digest="sha256:" + "e" * 64,
        evaluator_identity=evaluator_binding().identity,
        run_ids=(f"validation-{digest_digit}-0", f"validation-{digest_digit}-1"),
        metrics=metrics,
        feasible=True,
    )


def test_mixed_direction_pareto_excludes_dominated_and_constraint_failures() -> None:
    candidates = (
        aggregate("a", 0.9, 3.0),
        aggregate("b", 0.8, 2.0),
        aggregate("c", 0.7, 4.0),
        aggregate("d", 1.0, 1.0).model_copy(
            update={
                "metrics": {
                    **aggregate("d", 1.0, 1.0).metrics,
                    pointer("memory"): MetricAggregate(
                        value=30.0,
                        variance=0.0,
                        successful_runs=2,
                        attempted_runs=2,
                        failure_rate=0.0,
                    ),
                }
            }
        ),
    )
    memory = MetricConstraintV3(
        name="memory_limit",
        metric_pointer="/metrics/optimization_evaluation/memory",
        operator="le",
        threshold=20.0,
    )

    front = pareto_front(
        candidates,
        objectives=(objective("accuracy", "maximize"), objective("runtime", "minimize")),
        constraints=(memory,),
    )

    assert tuple(item.candidate_digest for item in front) == (
        "sha256:" + "a" * 64,
        "sha256:" + "b" * 64,
    )


def test_minimum_improvement_is_directional_and_inclusive() -> None:
    baseline = aggregate("a", 0.8, 4.0)
    boundary = aggregate("b", 0.9, 3.0)
    objectives = (
        objective("accuracy", "maximize").model_copy(update={"minimum_improvement": 0.1}),
        objective("runtime", "minimize").model_copy(update={"minimum_improvement": 1.0}),
    )

    assert minimum_improvements_met(boundary, baseline, objectives)
    assert not minimum_improvements_met(aggregate("c", 0.89, 3.0), baseline, objectives)


def test_relative_improvement_handles_zero_baseline_without_invented_score() -> None:
    baseline = aggregate("a", 0.0, 4.0)
    positive = aggregate("b", 0.1, 4.0)
    unchanged = aggregate("c", 0.0, 4.0)
    relative = (
        objective("accuracy", "maximize").model_copy(
            update={"minimum_improvement": 0.5, "improvement_mode": "relative"}
        ),
    )

    assert minimum_improvements_met(positive, baseline, relative)
    assert not minimum_improvements_met(unchanged, baseline, relative)


def test_raw_repeats_aggregate_mean_sample_variance_and_all_attempt_failures() -> None:
    attempts = tuple(
        RepeatEvidence(
            identity=AttemptIdentity(
                split="validation",
                split_id="validation-v1",
                split_content_hash="sha256:" + "f" * 64,
                candidate_digest="sha256:" + "a" * 64,
                parameter_digest="sha256:" + "a" * 64,
                logical_repeat_index=index,
                attempt_no=0,
                evaluator_digest="sha256:" + "e" * 64,
                evaluator_identity=evaluator_binding().identity,
            ),
            status="succeeded" if score is not None else "failed",
            run_id=f"run-{index}",
            seed=index,
            dispatch_digest="sha256:" + "c" * 64,
            granted_wall_time_seconds=10.0,
            granted_cpu_time_seconds=10.0,
            granted_peak_memory_bytes=100,
            metrics={} if score is None else {pointer("accuracy"): score},
            wall_time_seconds=1.0,
            cpu_time_seconds=0.5,
            peak_memory_bytes=10,
            error=(
                None
                if score is not None
                else ErrorDetail(code="evaluator.failed", message="synthetic")
            ),
        )
        for index, score in enumerate((0.8, 1.0, None))
    )
    policy = AdoptionPolicyV3(
        decision_rule="all_objectives",
        minimum_success_rate=0.6,
        maximum_failure_rate=0.4,
        maximum_wall_time_ratio=2.0,
        maximum_peak_memory_ratio=2.0,
        approval="automatic",
    )

    result = aggregate_repeats(
        attempts,
        objectives=(objective("accuracy", "maximize"),),
        constraints=(),
        adoption=policy,
        expected_repeats=3,
        max_wall_time_seconds=10,
        max_cpu_time_seconds=10,
        max_peak_memory_bytes=100,
    )

    assert result.metrics[pointer("accuracy")].value == 0.9
    assert math.isclose(result.metrics[pointer("accuracy")].variance, 0.02)
    assert math.isclose(result.metrics[pointer("accuracy")].failure_rate, 1 / 3)
    assert result.metrics["wall_time_seconds"].value == 1.0
    assert result.metrics["cpu_time_seconds"].value == 0.5
    assert result.metrics["peak_memory_bytes"].value == 10.0
    assert result.feasible


def test_all_failed_terminal_repeats_form_infeasible_auditable_aggregate() -> None:
    failed = RepeatEvidence(
        identity=AttemptIdentity(
            split="search",
            split_id="search-v1",
            split_content_hash="sha256:" + "d" * 64,
            candidate_digest="sha256:" + "a" * 64,
            parameter_digest="sha256:" + "a" * 64,
            logical_repeat_index=0,
            attempt_no=0,
            evaluator_digest=evaluator_binding().evaluator_digest,
            evaluator_identity=evaluator_binding().identity,
        ),
        status="failed",
        run_id="failed-run-0",
        seed=0,
        dispatch_digest="sha256:" + "c" * 64,
        granted_wall_time_seconds=1.0,
        granted_cpu_time_seconds=1.0,
        granted_peak_memory_bytes=1,
        error=ErrorDetail(code="evaluator.failed", message="synthetic"),
    )
    policy = AdoptionPolicyV3(
        decision_rule="all_objectives",
        minimum_success_rate=1.0,
        maximum_failure_rate=0.0,
        maximum_wall_time_ratio=1.0,
        maximum_peak_memory_ratio=1.0,
        approval="automatic",
    )
    result = aggregate_repeats(
        (failed,),
        objectives=(objective("accuracy", "maximize"),),
        constraints=(),
        adoption=policy,
        expected_repeats=1,
        max_wall_time_seconds=1,
        max_cpu_time_seconds=1,
        max_peak_memory_bytes=1,
    )
    assert not result.feasible
    assert result.run_ids == ("failed-run-0",)
    assert result.metrics["wall_time_seconds"].failure_rate == 1.0


def test_adoption_rejects_search_evidence_and_uses_only_fresh_validation() -> None:
    baseline = aggregate("a", 0.8, 4.0)
    search_only = aggregate("b", 1.0, 2.0).model_copy(
        update={
            "split": "search",
            "split_id": "search-v1",
            "split_content_hash": "sha256:" + "e" * 64,
        }
    )
    policy = AdoptionPolicyV3(
        decision_rule="all_objectives",
        minimum_success_rate=1.0,
        maximum_failure_rate=0.0,
        maximum_wall_time_ratio=1.0,
        maximum_peak_memory_ratio=1.0,
        approval="automatic",
    )
    objectives = (
        objective("accuracy", "maximize").model_copy(update={"minimum_improvement": 0.1}),
    )

    rejected = decide_adoption(
        baseline,
        (search_only,),
        objectives=objectives,
        constraints=(),
        adoption=policy,
        objective_order=("accuracy",),
        validation_split=validation_split(),
        evaluator_binding=evaluator_binding(),
    )
    adopted = decide_adoption(
        baseline,
        (aggregate("b", 0.9, 2.0),),
        objectives=objectives,
        constraints=(),
        adoption=policy,
        objective_order=("accuracy",),
        validation_split=validation_split(),
        evaluator_binding=evaluator_binding(),
    )

    assert rejected.status == "baseline_retained"
    assert rejected.reason_code == "optimization.validation_incomplete"
    assert adopted.status == "adopted"
    reused = aggregate("b", 0.9, 2.0).model_copy(update={"run_ids": baseline.run_ids})
    assert (
        decide_adoption(
            baseline,
            (reused,),
            objectives=objectives,
            constraints=(),
            adoption=policy,
            objective_order=("accuracy",),
            validation_split=validation_split(),
            evaluator_binding=evaluator_binding(),
        ).reason_code
        == "optimization.validation_incomplete"
    )
    shared_finalist_runs = aggregate("c", 0.95, 2.0).model_copy(
        update={"run_ids": aggregate("b", 0.9, 2.0).run_ids}
    )
    assert (
        decide_adoption(
            baseline,
            (aggregate("b", 0.9, 2.0), shared_finalist_runs),
            objectives=objectives,
            constraints=(),
            adoption=policy,
            objective_order=("accuracy",),
            validation_split=validation_split(),
            evaluator_binding=evaluator_binding(),
        ).reason_code
        == "optimization.validation_incomplete"
    )
    forged_evaluator = aggregate("b", 0.9, 2.0).model_copy(
        update={
            "evaluator_identity": evaluator_binding().identity.model_copy(
                update={"bundle_content_hash": "sha256:" + "c" * 64}
            )
        }
    )
    assert (
        decide_adoption(
            baseline,
            (forged_evaluator,),
            objectives=objectives,
            constraints=(),
            adoption=policy,
            objective_order=("accuracy",),
            validation_split=validation_split(),
            evaluator_binding=evaluator_binding(),
        ).reason_code
        == "optimization.validation_incomplete"
    )


def test_candidate_aggregate_rejects_noncanonical_parameter_digest() -> None:
    with pytest.raises(ValidationError, match="parameter digest"):
        CandidateAggregate.model_validate(
            aggregate("a", 0.9, 2.0).model_dump(mode="json")
            | {"parameter_digest": "sha256:" + "b" * 64}
        )


def test_even_median_and_hard_resource_boundaries_are_mechanical() -> None:
    attempts = tuple(
        RepeatEvidence(
            identity=AttemptIdentity(
                split="search",
                split_id="search-v1",
                split_content_hash="sha256:" + "d" * 64,
                candidate_digest="sha256:" + "a" * 64,
                parameter_digest="sha256:" + "a" * 64,
                logical_repeat_index=index,
                attempt_no=0,
                evaluator_digest="sha256:" + "e" * 64,
                evaluator_identity=evaluator_binding().identity,
            ),
            status="succeeded",
            run_id=f"median-{index}",
            seed=index,
            dispatch_digest="sha256:" + "c" * 64,
            granted_wall_time_seconds=10.0,
            granted_cpu_time_seconds=10.0,
            granted_peak_memory_bytes=100,
            metrics={pointer("accuracy"): value},
            wall_time_seconds=1.0,
            cpu_time_seconds=0.5,
            peak_memory_bytes=10,
        )
        for index, value in enumerate((0.2, 0.8, 1.0, 1.4))
    )
    median_objective = objective("accuracy", "maximize").model_copy(
        update={"aggregation": "median"}
    )
    policy = AdoptionPolicyV3(
        decision_rule="all_objectives",
        minimum_success_rate=1.0,
        maximum_failure_rate=0.0,
        maximum_wall_time_ratio=1.0,
        maximum_peak_memory_ratio=1.0,
        approval="automatic",
    )
    exact = aggregate_repeats(
        attempts,
        objectives=(median_objective,),
        constraints=(),
        adoption=policy,
        expected_repeats=4,
        max_wall_time_seconds=4,
        max_cpu_time_seconds=2,
        max_peak_memory_bytes=10,
    )
    exceeded = aggregate_repeats(
        attempts,
        objectives=(median_objective,),
        constraints=(),
        adoption=policy,
        expected_repeats=4,
        max_wall_time_seconds=3,
        max_cpu_time_seconds=2,
        max_peak_memory_bytes=10,
    )

    assert exact.metrics[pointer("accuracy")].value == 0.9
    assert math.isclose(exact.metrics[pointer("accuracy")].variance, 0.25)
    assert exact.feasible
    assert not exceeded.feasible


@pytest.mark.parametrize("bad_metric", [True, float("nan"), float("inf")])
def test_repeat_metrics_reject_bool_and_non_finite_values(bad_metric: object) -> None:
    with pytest.raises(ValidationError):
        RepeatEvidence.model_validate(
            {
                "identity": AttemptIdentity(
                    split="search",
                    split_id="search-v1",
                    split_content_hash="sha256:" + "d" * 64,
                    candidate_digest="sha256:" + "a" * 64,
                    parameter_digest="sha256:" + "a" * 64,
                    logical_repeat_index=0,
                    attempt_no=0,
                    evaluator_digest="sha256:" + "e" * 64,
                    evaluator_identity=evaluator_binding().identity,
                ),
                "status": "succeeded",
                "run_id": "bad-metric",
                "seed": 0,
                "dispatch_digest": "sha256:" + "c" * 64,
                "granted_wall_time_seconds": 1.0,
                "granted_cpu_time_seconds": 1.0,
                "granted_peak_memory_bytes": 1,
                "metrics": {"accuracy": bad_metric},
            }
        )


def test_pareto_keeps_equal_points_but_excludes_missing_metrics() -> None:
    first = aggregate("a", 0.9, 2.0)
    equal = aggregate("b", 0.9, 2.0)
    missing = aggregate("c", 1.0, 1.0).model_copy(
        update={
            "metrics": {pointer("accuracy"): aggregate("c", 1.0, 1.0).metrics[pointer("accuracy")]}
        }
    )

    front = pareto_front(
        (first, equal, missing),
        objectives=(objective("accuracy", "maximize"), objective("runtime", "minimize")),
    )
    assert tuple(item.candidate_digest for item in front) == (
        first.candidate_digest,
        equal.candidate_digest,
    )


def test_relative_improvement_uses_absolute_negative_baseline_denominator() -> None:
    baseline = aggregate("a", -10.0, 1.0)
    candidate = aggregate("b", -5.0, 1.0)
    relative = (
        objective("accuracy", "maximize").model_copy(
            update={"minimum_improvement": 0.5, "improvement_mode": "relative"}
        ),
    )
    assert minimum_improvements_met(candidate, baseline, relative)


def test_aggregate_rejects_running_attempts() -> None:
    running = RepeatEvidence(
        identity=AttemptIdentity(
            split="search",
            split_id="search-v1",
            split_content_hash="sha256:" + "d" * 64,
            candidate_digest="sha256:" + "a" * 64,
            parameter_digest="sha256:" + "a" * 64,
            logical_repeat_index=0,
            attempt_no=0,
            evaluator_digest="sha256:" + "e" * 64,
            evaluator_identity=evaluator_binding().identity,
        ),
        status="running",
        run_id="running-1",
        seed=0,
        dispatch_digest="sha256:" + "c" * 64,
        granted_wall_time_seconds=1.0,
        granted_cpu_time_seconds=1.0,
        granted_peak_memory_bytes=1,
    )
    policy = AdoptionPolicyV3(
        decision_rule="all_objectives",
        minimum_success_rate=1.0,
        maximum_failure_rate=0.0,
        maximum_wall_time_ratio=1.0,
        maximum_peak_memory_ratio=1.0,
        approval="automatic",
    )
    with pytest.raises(ValueError, match="running"):
        aggregate_repeats(
            (running,),
            objectives=(objective("accuracy", "maximize"),),
            constraints=(),
            adoption=policy,
            expected_repeats=1,
            max_wall_time_seconds=1,
            max_cpu_time_seconds=1,
            max_peak_memory_bytes=1,
        )


def test_full_metric_pointers_do_not_collide_on_equal_tail_tokens() -> None:
    left = objective("left", "maximize").model_copy(
        update={"metric_pointer": "/metrics/optimization_evaluation/left/value"}
    )
    right = objective("right", "maximize").model_copy(
        update={"metric_pointer": "/metrics/optimization_evaluation/right/value"}
    )
    candidate = aggregate("a", 0.0, 0.0).model_copy(
        update={
            "metrics": {
                **aggregate("a", 0.0, 0.0).metrics,
                left.metric_pointer: MetricAggregate(
                    value=1.0,
                    variance=0.0,
                    successful_runs=2,
                    attempted_runs=2,
                    failure_rate=0.0,
                ),
                right.metric_pointer: MetricAggregate(
                    value=2.0,
                    variance=0.0,
                    successful_runs=2,
                    attempted_runs=2,
                    failure_rate=0.0,
                ),
            }
        }
    )
    assert pareto_front((candidate,), objectives=(left, right)) == (candidate,)
