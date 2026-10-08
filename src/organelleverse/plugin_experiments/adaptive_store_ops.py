"""Pure durable transitions for adaptive study records."""

from __future__ import annotations

from organelleverse.core.errors import OrganelleContractError
from organelleverse.core.result import ErrorDetail

from .adaptive_models import AdaptiveStudyRecord, BudgetAccount, RepeatEvidence

_TERMINAL = frozenset({"adopted", "baseline_retained", "failed"})
_TRANSITIONS: dict[str, frozenset[str]] = {
    "created": frozenset({"created", "baseline_search", "interrupted", "failed"}),
    "baseline_search": frozenset(
        {"baseline_search", "searching", "interrupted", "baseline_retained", "failed"}
    ),
    "searching": frozenset(
        {"searching", "shortlist", "interrupted", "baseline_retained", "failed"}
    ),
    "shortlist": frozenset(
        {"shortlist", "baseline_validation", "interrupted", "baseline_retained", "failed"}
    ),
    "baseline_validation": frozenset(
        {"baseline_validation", "validating", "interrupted", "baseline_retained", "failed"}
    ),
    "validating": frozenset(
        {"validating", "interrupted", "adopted", "baseline_retained", "failed"}
    ),
    "interrupted": frozenset(
        {
            "interrupted",
            "baseline_search",
            "searching",
            "shortlist",
            "baseline_validation",
            "validating",
            "baseline_retained",
            "failed",
        }
    ),
    "adopted": frozenset({"adopted"}),
    "baseline_retained": frozenset({"baseline_retained"}),
    "failed": frozenset({"failed"}),
}


def append_repeat(record: AdaptiveStudyRecord, evidence: RepeatEvidence) -> AdaptiveStudyRecord:
    if record.status in _TERMINAL:
        raise OrganelleContractError(
            code="optimization.study_terminal",
            message="terminal adaptive studies reject new repeat evidence",
            details={"study_id": record.study_id},
        )
    if (
        evidence.identity.evaluator_digest != record.evaluator_digest
        or evidence.identity.evaluator_identity != record.evaluator_identity
    ):
        raise OrganelleContractError(
            code="optimization.evaluator_mismatch",
            message="repeat evaluator binding does not match the adaptive study",
            details={"study_id": record.study_id},
        )
    logical = tuple(
        item for item in record.attempts if _logical_key(item) == _logical_key(evidence)
    )
    for existing in logical:
        if existing.identity == evidence.identity:
            if existing == evidence:
                return record
            if existing.status == "running" and evidence.status in {"succeeded", "failed"}:
                return _complete_running(record, existing, evidence)
            raise OrganelleContractError(
                code="optimization.attempt_conflict",
                message="an attempt identity already has different durable evidence",
                details={"study_id": record.study_id},
            )
    _validate_next_attempt(record, logical, evidence)
    if any(item.run_id == evidence.run_id for item in record.attempts):
        raise OrganelleContractError(
            code="optimization.attempt_conflict",
            message="a run ID cannot provide evidence for multiple attempts",
            details={"study_id": record.study_id},
        )
    budget = record.budget
    accounted = BudgetAccount(
        attempted_runs=budget.attempted_runs + 1,
        wall_time_seconds=float(budget.wall_time_seconds + evidence.wall_time_seconds),
        cpu_time_seconds=float(budget.cpu_time_seconds + evidence.cpu_time_seconds),
        peak_memory_bytes=max(budget.peak_memory_bytes, evidence.peak_memory_bytes),
    )
    return record.model_copy(update={"attempts": (*record.attempts, evidence), "budget": accounted})


def _logical_key(evidence: RepeatEvidence) -> tuple[object, ...]:
    identity = evidence.identity
    return (
        identity.split,
        identity.split_id,
        identity.split_content_hash,
        identity.candidate_digest,
        identity.resource_rung,
        identity.logical_repeat_index,
        identity.evaluator_digest,
    )


def _validate_next_attempt(
    record: AdaptiveStudyRecord,
    logical: tuple[RepeatEvidence, ...],
    evidence: RepeatEvidence,
) -> None:
    if not logical:
        valid = evidence.identity.attempt_no == 0
    else:
        latest = max(logical, key=lambda item: item.identity.attempt_no)
        valid = (
            latest.status == "interrupted"
            and evidence.identity.attempt_no == latest.identity.attempt_no + 1
        )
    if not valid:
        raise OrganelleContractError(
            code="optimization.attempt_sequence_invalid",
            message="attempts start at zero and advance only after interruption",
            details={"study_id": record.study_id},
        )


def _complete_running(
    record: AdaptiveStudyRecord,
    running: RepeatEvidence,
    completed: RepeatEvidence,
) -> AdaptiveStudyRecord:
    if (
        running.run_id != completed.run_id
        or running.seed != completed.seed
        or running.dispatch_digest != completed.dispatch_digest
        or running.granted_wall_time_seconds != completed.granted_wall_time_seconds
        or running.granted_cpu_time_seconds != completed.granted_cpu_time_seconds
        or running.granted_peak_memory_bytes != completed.granted_peak_memory_bytes
    ):
        raise OrganelleContractError(
            code="optimization.attempt_conflict",
            message="attempt completion changed its run identity",
            details={"study_id": record.study_id},
        )
    if (
        completed.wall_time_seconds < running.wall_time_seconds
        or completed.cpu_time_seconds < running.cpu_time_seconds
        or completed.peak_memory_bytes < running.peak_memory_bytes
    ):
        raise OrganelleContractError(
            code="optimization.attempt_conflict",
            message="attempt completion decreased recorded resource use",
            details={"study_id": record.study_id},
        )
    attempts = tuple(
        completed if item.identity == completed.identity else item for item in record.attempts
    )
    budget = record.budget
    accounted = BudgetAccount(
        attempted_runs=budget.attempted_runs,
        wall_time_seconds=float(
            budget.wall_time_seconds + completed.wall_time_seconds - running.wall_time_seconds
        ),
        cpu_time_seconds=float(
            budget.cpu_time_seconds + completed.cpu_time_seconds - running.cpu_time_seconds
        ),
        peak_memory_bytes=max(budget.peak_memory_bytes, completed.peak_memory_bytes),
    )
    return record.model_copy(update={"attempts": attempts, "budget": accounted})


def recover_interrupted(record: AdaptiveStudyRecord) -> AdaptiveStudyRecord:
    if record.status in _TERMINAL:
        return record
    changed = False
    attempts: list[RepeatEvidence] = []
    for attempt in record.attempts:
        if attempt.status == "running":
            attempt = attempt.model_copy(
                update={
                    "status": "interrupted",
                    "error": ErrorDetail(
                        code="experiment.run_interrupted",
                        message="the adaptive evaluator stopped before this attempt became terminal",
                    ),
                }
            )
            changed = True
        attempts.append(attempt)
    if not changed:
        return record
    consumed = BudgetAccount(
        attempted_runs=max(record.budget.attempted_runs, len(attempts)),
        wall_time_seconds=max(
            record.budget.wall_time_seconds,
            float(sum(item.wall_time_seconds for item in attempts)),
        ),
        cpu_time_seconds=max(
            record.budget.cpu_time_seconds,
            float(sum(item.cpu_time_seconds for item in attempts)),
        ),
        peak_memory_bytes=max(
            (record.budget.peak_memory_bytes, *(item.peak_memory_bytes for item in attempts))
        ),
    )
    return record.model_copy(
        update={"status": "interrupted", "attempts": tuple(attempts), "budget": consumed}
    )


def validate_replacement(existing: AdaptiveStudyRecord, replacement: AdaptiveStudyRecord) -> None:
    if existing.study_id != replacement.study_id:
        raise OrganelleContractError(
            code="optimization.study_identity_mismatch",
            message="adaptive study replacement changed its identity",
        )
    frozen = (
        "submitted_at",
        "contract_digest",
        "evaluator_digest",
        "evaluator_identity",
        "request",
    )
    if any(getattr(existing, name) != getattr(replacement, name) for name in frozen):
        raise OrganelleContractError(
            code="optimization.study_identity_mismatch",
            message="adaptive study immutable identity fields changed",
            details={"study_id": existing.study_id},
        )
    if existing.status in _TERMINAL and existing != replacement:
        raise OrganelleContractError(
            code="optimization.study_terminal",
            message="terminal adaptive studies are immutable",
            details={"study_id": existing.study_id},
        )
    if replacement.status not in _TRANSITIONS[existing.status]:
        raise OrganelleContractError(
            code="optimization.study_transition_invalid",
            message="adaptive study status transition is not allowed",
            details={
                "study_id": existing.study_id,
                "from": existing.status,
                "to": replacement.status,
            },
        )
    if replacement.attempts != existing.attempts or replacement.budget != existing.budget:
        raise OrganelleContractError(
            code="optimization.attempt_history_write_forbidden",
            message="repeat evidence and budget may change only through atomic append",
            details={"study_id": existing.study_id},
        )


__all__ = ["append_repeat", "recover_interrupted", "validate_replacement"]
