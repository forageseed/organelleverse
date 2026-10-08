"""Pure measured-evidence aggregation, Pareto, and adoption decisions."""

from __future__ import annotations

from decimal import Decimal
from statistics import mean, median, variance

from organelleverse.plugin_experiments.adaptive_models import (
    AdoptionDecision,
    CandidateAggregate,
    MetricAggregate,
    RepeatEvidence,
)

from .contracts_v3 import MetricConstraintV3, ObjectiveV3
from .evaluation import AdoptionPolicyV3, BenchmarkSplitV3
from .strategy_models import EvaluatorBinding


def aggregate_repeats(
    attempts: tuple[RepeatEvidence, ...],
    *,
    objectives: tuple[ObjectiveV3, ...],
    constraints: tuple[MetricConstraintV3, ...],
    adoption: AdoptionPolicyV3,
    expected_repeats: int,
    max_wall_time_seconds: int,
    max_cpu_time_seconds: int,
    max_peak_memory_bytes: int,
) -> CandidateAggregate:
    if not attempts:
        raise ValueError("repeat aggregation requires attempted evidence")
    if any(item.status == "running" for item in attempts):
        raise ValueError("running attempts cannot be aggregated")
    first = attempts[0].identity
    if any(
        (
            item.identity.split,
            item.identity.split_id,
            item.identity.split_content_hash,
            item.identity.candidate_digest,
            item.identity.parameter_digest,
            item.identity.resource_rung,
            item.identity.evaluator_digest,
            item.identity.evaluator_identity,
        )
        != (
            first.split,
            first.split_id,
            first.split_content_hash,
            first.candidate_digest,
            first.parameter_digest,
            first.resource_rung,
            first.evaluator_digest,
            first.evaluator_identity,
        )
        for item in attempts
    ):
        raise ValueError("repeat aggregation cannot mix evidence identities")
    required = {
        *(item.metric_pointer for item in objectives),
        *(item.metric_pointer for item in constraints),
    }
    latest_by_repeat: dict[int, RepeatEvidence] = {}
    for item in attempts:
        current = latest_by_repeat.get(item.identity.logical_repeat_index)
        if current is None or item.identity.attempt_no > current.identity.attempt_no:
            latest_by_repeat[item.identity.logical_repeat_index] = item
    successful = tuple(
        item
        for item in latest_by_repeat.values()
        if item.status == "succeeded" and required <= item.metrics.keys()
    )
    failure_rate = (len(attempts) - len(successful)) / len(attempts)
    aggregates: dict[str, MetricAggregate] = {}
    aggregations = {
        **{item.metric_pointer: item.aggregation for item in objectives},
        **{item.metric_pointer: "mean" for item in constraints},
    }
    for name in sorted(required):
        values = [float(item.metrics[name]) for item in successful]
        if not values:
            continue
        center = median(values) if aggregations[name] == "median" else mean(values)
        aggregates[name] = MetricAggregate(
            value=float(center),
            variance=float(variance(values)) if len(values) >= 2 else 0.0,
            successful_runs=len(successful),
            attempted_runs=len(attempts),
            failure_rate=float(failure_rate),
        )
    resource_values = {
        "wall_time_seconds": [float(item.wall_time_seconds) for item in attempts],
        "cpu_time_seconds": [float(item.cpu_time_seconds) for item in attempts],
        "peak_memory_bytes": [float(item.peak_memory_bytes) for item in attempts],
    }
    for name, values in resource_values.items():
        aggregates[name] = MetricAggregate(
            value=max(values) if name == "peak_memory_bytes" else float(mean(values)),
            variance=float(variance(values)) if len(values) >= 2 else 0.0,
            successful_runs=len(successful),
            attempted_runs=len(attempts),
            failure_rate=float(failure_rate),
        )
    complete = (
        len(latest_by_repeat) == expected_repeats
        and all(item.status in {"succeeded", "failed"} for item in latest_by_repeat.values())
        and required <= aggregates.keys()
    )
    success_rate = len(successful) / len(attempts)
    resources_ok = (
        sum(float(item.wall_time_seconds) for item in attempts) <= max_wall_time_seconds
        and sum(float(item.cpu_time_seconds) for item in attempts) <= max_cpu_time_seconds
        and max(item.peak_memory_bytes for item in attempts) <= max_peak_memory_bytes
    )
    candidate = CandidateAggregate(
        split=first.split,
        split_id=first.split_id,
        split_content_hash=first.split_content_hash,
        candidate_digest=first.candidate_digest,
        parameter_digest=first.parameter_digest,
        resource_rung=first.resource_rung,
        evaluator_digest=first.evaluator_digest,
        evaluator_identity=first.evaluator_identity,
        run_ids=tuple(item.run_id for item in attempts),
        metrics=aggregates,
        feasible=complete
        and success_rate >= adoption.minimum_success_rate
        and failure_rate <= adoption.maximum_failure_rate
        and resources_ok,
        failure_codes=tuple(
            sorted({item.error.code for item in attempts if item.error is not None})
        ),
    )
    if candidate.feasible and not _satisfies_constraints(candidate, constraints):
        return candidate.model_copy(update={"feasible": False})
    return candidate


def pareto_front(
    candidates: tuple[CandidateAggregate, ...],
    *,
    objectives: tuple[ObjectiveV3, ...],
    constraints: tuple[MetricConstraintV3, ...] = (),
) -> tuple[CandidateAggregate, ...]:
    feasible = tuple(
        candidate
        for candidate in candidates
        if candidate.feasible
        and _has_metrics(candidate, objectives, constraints)
        and _satisfies_constraints(candidate, constraints)
    )
    front = [
        candidate
        for candidate in feasible
        if not any(
            other is not candidate and _dominates(other, candidate, objectives)
            for other in feasible
        )
    ]
    return tuple(sorted(front, key=lambda item: item.candidate_digest))


def minimum_improvements_met(
    candidate: CandidateAggregate,
    baseline: CandidateAggregate,
    objectives: tuple[ObjectiveV3, ...],
) -> bool:
    if not candidate.feasible or not baseline.feasible:
        return False
    for objective in objectives:
        name = objective.metric_pointer
        if name not in candidate.metrics or name not in baseline.metrics:
            return False
        candidate_value = Decimal(str(candidate.metrics[name].value))
        baseline_value = Decimal(str(baseline.metrics[name].value))
        delta = (
            candidate_value - baseline_value
            if objective.direction == "maximize"
            else baseline_value - candidate_value
        )
        threshold = Decimal(str(objective.minimum_improvement))
        if objective.improvement_mode == "relative":
            improvement = Decimal("Infinity") if baseline_value == 0 and delta > 0 else delta
            if baseline_value != 0:
                improvement = delta / abs(baseline_value)
        else:
            improvement = delta
        if improvement < threshold:
            return False
    return True


def decide_adoption(
    baseline: CandidateAggregate,
    finalists: tuple[CandidateAggregate, ...],
    *,
    objectives: tuple[ObjectiveV3, ...],
    constraints: tuple[MetricConstraintV3, ...],
    adoption: AdoptionPolicyV3,
    objective_order: tuple[str, ...],
    validation_split: BenchmarkSplitV3,
    evaluator_binding: EvaluatorBinding,
) -> AdoptionDecision:
    if set(objective_order) != {item.name for item in objectives} or len(objective_order) != len(
        objectives
    ):
        raise ValueError("objective order must cover every objective exactly once")
    if (
        baseline.split != "validation"
        or baseline.split_id != validation_split.split_id
        or baseline.split_content_hash != validation_split.content_hash
        or baseline.evaluator_digest != evaluator_binding.evaluator_digest
        or baseline.evaluator_identity != evaluator_binding.identity
        or not baseline.run_ids
        or any(
            candidate.split != "validation"
            or candidate.split_id != validation_split.split_id
            or candidate.split_content_hash != validation_split.content_hash
            or candidate.evaluator_digest != evaluator_binding.evaluator_digest
            or candidate.evaluator_identity != evaluator_binding.identity
            or not candidate.run_ids
            or not set(candidate.run_ids).isdisjoint(baseline.run_ids)
            for candidate in finalists
        )
    ):
        return _retained("optimization.validation_incomplete")
    validation_run_ids = [
        *baseline.run_ids,
        *(run_id for item in finalists for run_id in item.run_ids),
    ]
    if len(validation_run_ids) != len(set(validation_run_ids)):
        return _retained("optimization.validation_incomplete")
    front = pareto_front(finalists, objectives=objectives, constraints=constraints)
    if not front:
        return _retained("optimization.no_feasible_candidate")
    eligible = tuple(
        candidate
        for candidate in front
        if minimum_improvements_met(candidate, baseline, objectives)
        and _resource_ratios_met(candidate, baseline, adoption)
    )
    if not eligible:
        return _retained("optimization.insufficient_improvement")
    by_name = {item.name: item for item in objectives}

    def rank(candidate: CandidateAggregate) -> tuple[float | str, ...]:
        values: list[float | str] = []
        for name in objective_order:
            objective = by_name[name]
            value = candidate.metrics[objective.metric_pointer].value
            values.append(-value if objective.direction == "maximize" else value)
        values.append(candidate.candidate_digest)
        return tuple(values)

    winner = min(eligible, key=rank)
    return AdoptionDecision(
        status="adopted",
        reason_code="optimization.candidate_adopted",
        candidate_digest=winner.candidate_digest,
        parameter_digest=winner.parameter_digest,
    )


def _retained(reason: str) -> AdoptionDecision:
    return AdoptionDecision(status="baseline_retained", reason_code=reason)


def _resource_ratios_met(
    candidate: CandidateAggregate,
    baseline: CandidateAggregate,
    adoption: AdoptionPolicyV3,
) -> bool:
    for name, ceiling in (
        ("wall_time_seconds", adoption.maximum_wall_time_ratio),
        ("peak_memory_bytes", adoption.maximum_peak_memory_ratio),
    ):
        if name not in candidate.metrics or name not in baseline.metrics:
            return False
        candidate_value = candidate.metrics[name].value
        baseline_value = baseline.metrics[name].value
        if baseline_value == 0:
            if candidate_value > 0:
                return False
        elif candidate_value / baseline_value > ceiling:
            return False
    return True


def _has_metrics(
    candidate: CandidateAggregate,
    objectives: tuple[ObjectiveV3, ...],
    constraints: tuple[MetricConstraintV3, ...],
) -> bool:
    required = {
        *(item.metric_pointer for item in objectives),
        *(item.metric_pointer for item in constraints),
    }
    return required <= candidate.metrics.keys()


def _satisfies_constraints(
    candidate: CandidateAggregate, constraints: tuple[MetricConstraintV3, ...]
) -> bool:
    for constraint in constraints:
        value = candidate.metrics[constraint.metric_pointer].value
        if constraint.operator == "le" and value > constraint.threshold:
            return False
        if constraint.operator == "ge" and value < constraint.threshold:
            return False
    return True


def _dominates(
    left: CandidateAggregate,
    right: CandidateAggregate,
    objectives: tuple[ObjectiveV3, ...],
) -> bool:
    no_worse = True
    strictly_better = False
    for objective in objectives:
        name = objective.metric_pointer
        left_value = left.metrics[name].value
        right_value = right.metrics[name].value
        if objective.direction == "maximize":
            no_worse &= left_value >= right_value
            strictly_better |= left_value > right_value
        else:
            no_worse &= left_value <= right_value
            strictly_better |= left_value < right_value
    return no_worse and strictly_better


__all__ = [
    "AdoptionDecision",
    "CandidateAggregate",
    "MetricAggregate",
    "RepeatEvidence",
    "aggregate_repeats",
    "decide_adoption",
    "minimum_improvements_met",
    "pareto_front",
]
