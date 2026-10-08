"""Durable running-to-terminal evaluator dispatch and aggregation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from organelleverse.core.errors import OrganelleContractError
from organelleverse.core.result import ErrorDetail
from organelleverse.plugin_experiments.adaptive_models import (
    AdaptiveStudyRecord,
    AttemptIdentity,
    CandidateAggregate,
    RepeatEvidence,
)
from organelleverse.plugin_experiments.store import ExperimentStore

from .evaluation import BenchmarkSplitV3
from .evaluator_runtime import evaluate_under_grant
from .evidence import aggregate_repeats
from .strategy_models import AdaptiveStudyRequest, CandidateProposal, EvaluatorBinding
from .study_models import EvaluationRequest, EvaluatorRunner, canonical_digest


def evaluate_candidate(
    *,
    record: AdaptiveStudyRecord,
    request: AdaptiveStudyRequest,
    proposal: CandidateProposal,
    split: Literal["search", "validation"],
    repeat_count: int,
    runner: EvaluatorRunner,
    store: ExperimentStore,
) -> tuple[CandidateAggregate, AdaptiveStudyRecord]:
    contract = request.contract
    benchmark = contract.benchmark.search if split == "search" else contract.benchmark.validation
    required = tuple(
        sorted(
            {
                *(item.metric_pointer for item in contract.objectives),
                *(item.metric_pointer for item in contract.constraints),
            }
        )
    )
    for repeat_index in range(repeat_count):
        matching = _matching_attempts(
            record, proposal, split, benchmark, repeat_index, request.evaluator_binding
        )
        latest = max(matching, key=lambda item: item.identity.attempt_no, default=None)
        if latest is not None and latest.status in {"succeeded", "failed"}:
            continue
        if latest is not None and latest.status == "running":
            raise _error("optimization.attempt_conflict", "an attempt is already running")
        attempt_no = 0 if latest is None else latest.identity.attempt_no + 1
        record = _dispatch_attempt(
            record=record,
            request=request,
            proposal=proposal,
            split=split,
            benchmark=benchmark,
            repeat_index=repeat_index,
            attempt_no=attempt_no,
            required=required,
            runner=runner,
            store=store,
        )
    attempts = tuple(
        item
        for item in record.attempts
        if item.identity.split == split
        and item.identity.split_id == benchmark.split_id
        and item.identity.split_content_hash == benchmark.content_hash
        and item.identity.candidate_digest == proposal.candidate_digest
        and item.identity.resource_rung == proposal.resource_rung
        and item.identity.evaluator_digest == request.evaluator_binding.evaluator_digest
        and item.identity.evaluator_identity == request.evaluator_binding.identity
    )
    aggregate = aggregate_repeats(
        attempts,
        objectives=contract.objectives,
        constraints=contract.constraints,
        adoption=contract.adoption,
        expected_repeats=repeat_count,
        max_wall_time_seconds=contract.budget.max_wall_time_seconds,
        max_cpu_time_seconds=contract.budget.max_cpu_time_seconds,
        max_peak_memory_bytes=contract.budget.max_peak_memory_bytes,
    )
    retained = tuple(
        item
        for item in record.aggregates
        if (item.split, item.split_id, item.candidate_digest, item.resource_rung)
        != (split, benchmark.split_id, proposal.candidate_digest, proposal.resource_rung)
    )
    record = store.update_study(record.model_copy(update={"aggregates": (*retained, aggregate)}))
    return aggregate, record


def _matching_attempts(
    record: AdaptiveStudyRecord,
    proposal: CandidateProposal,
    split: Literal["search", "validation"],
    benchmark: BenchmarkSplitV3,
    repeat_index: int,
    evaluator: EvaluatorBinding,
) -> tuple[RepeatEvidence, ...]:
    return tuple(
        item
        for item in record.attempts
        if item.identity.split == split
        and item.identity.split_id == benchmark.split_id
        and item.identity.split_content_hash == benchmark.content_hash
        and item.identity.candidate_digest == proposal.candidate_digest
        and item.identity.resource_rung == proposal.resource_rung
        and item.identity.logical_repeat_index == repeat_index
        and item.identity.evaluator_digest == evaluator.evaluator_digest
        and item.identity.evaluator_identity == evaluator.identity
    )


def _dispatch_attempt(
    *,
    record: AdaptiveStudyRecord,
    request: AdaptiveStudyRequest,
    proposal: CandidateProposal,
    split: Literal["search", "validation"],
    benchmark: BenchmarkSplitV3,
    repeat_index: int,
    attempt_no: int,
    required: tuple[str, ...],
    runner: EvaluatorRunner,
    store: ExperimentStore,
) -> AdaptiveStudyRecord:
    budget = request.contract.budget
    elapsed = (datetime.now(UTC) - record.submitted_at).total_seconds()
    wall_grant = min(
        budget.max_wall_time_seconds - elapsed,
        budget.max_wall_time_seconds - record.budget.wall_time_seconds,
    )
    cpu_grant = budget.max_cpu_time_seconds - record.budget.cpu_time_seconds
    if record.budget.attempted_runs >= budget.max_trials or wall_grant <= 0 or cpu_grant <= 0:
        raise _error("optimization.budget_exhausted", "hard evaluator budget is exhausted")
    identity = AttemptIdentity(
        split=split,
        split_id=benchmark.split_id,
        split_content_hash=benchmark.content_hash,
        candidate_digest=proposal.candidate_digest,
        parameter_digest=proposal.candidate_digest,
        resource_rung=proposal.resource_rung,
        logical_repeat_index=repeat_index,
        attempt_no=attempt_no,
        evaluator_digest=request.evaluator_binding.evaluator_digest,
        evaluator_identity=request.evaluator_binding.identity,
    )
    run_id = _run_id(record.study_id, identity)
    seed = request.contract.repeats.seed + repeat_index
    dispatch = EvaluationRequest(
        study_id=record.study_id,
        target=request.contract.target,
        contract_digest=request.contract.digest,
        evaluator_binding=request.evaluator_binding,
        identity=identity,
        benchmark_split=benchmark,
        parameters=proposal.parameters,
        input_digest=request.input_digest,
        fixed_parameters_digest=request.fixed_parameters_digest,
        input_artifact=request.input_artifact,
        fixed_parameters_artifact=request.fixed_parameters_artifact,
        environment_digest=request.environment_digest,
        seed=seed,
        run_id=run_id,
        required_metric_pointers=required,
        remaining_wall_time_seconds=wall_grant,
        remaining_cpu_time_seconds=cpu_grant,
        max_peak_memory_bytes=budget.max_peak_memory_bytes,
    )
    running = RepeatEvidence(
        identity=identity,
        status="running",
        run_id=run_id,
        seed=seed,
        dispatch_digest=dispatch.digest,
        granted_wall_time_seconds=wall_grant,
        granted_cpu_time_seconds=cpu_grant,
        granted_peak_memory_bytes=budget.max_peak_memory_bytes,
    )
    _require_runner_binding(runner, request.evaluator_binding)
    record = store.append_repeat_evidence(record.study_id, running)
    try:
        _require_runner_binding(runner, request.evaluator_binding)
        completed = evaluate_under_grant(runner, dispatch, running)
        _validate_completion(completed, running, required)
        if (
            completed.wall_time_seconds > wall_grant
            or completed.cpu_time_seconds > cpu_grant
            or completed.peak_memory_bytes > budget.max_peak_memory_bytes
        ):
            completed = completed.model_copy(
                update={
                    "status": "failed",
                    "error": ErrorDetail(
                        code="optimization.resource_constraint_failed",
                        message="evaluator exceeded its hard resource grant",
                    ),
                }
            )
    except Exception as error:
        detail = (
            ErrorDetail(code=error.code, message=error.message)
            if isinstance(error, OrganelleContractError)
            else ErrorDetail(
                code="optimization.evaluator_failed",
                message="the evaluator failed without trusted public details",
            )
        )
        completed = running.model_copy(update={"status": "failed", "error": detail})
    return store.append_repeat_evidence(record.study_id, completed)


def _require_runner_binding(runner: EvaluatorRunner, expected: EvaluatorBinding) -> None:
    if runner.binding != expected:
        raise _error("optimization.evaluator_mismatch", "evaluator runner binding mismatch")


def _validate_completion(
    completed: RepeatEvidence, running: RepeatEvidence, required: tuple[str, ...]
) -> None:
    if (
        completed.identity != running.identity
        or completed.run_id != running.run_id
        or completed.seed != running.seed
        or completed.dispatch_digest != running.dispatch_digest
        or completed.granted_wall_time_seconds != running.granted_wall_time_seconds
        or completed.granted_cpu_time_seconds != running.granted_cpu_time_seconds
        or completed.granted_peak_memory_bytes != running.granted_peak_memory_bytes
        or completed.status not in {"succeeded", "failed"}
    ):
        raise _error("optimization.evaluator_mismatch", "evaluator completion mismatch")
    if completed.status == "succeeded" and not set(required) <= completed.metrics.keys():
        raise _error("optimization.metric_invalid", "evaluator omitted required metrics")


def _run_id(study_id: str, identity: AttemptIdentity) -> str:
    digest = canonical_digest({"study_id": study_id, "attempt": identity.model_dump(mode="json")})
    return f"run-{digest.removeprefix('sha256:')}"


def _error(code: str, message: str) -> OrganelleContractError:
    return OrganelleContractError(code=code, message=message)


__all__ = ["evaluate_candidate"]
