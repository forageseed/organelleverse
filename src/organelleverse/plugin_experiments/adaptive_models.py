"""Closed durable models for adaptive optimization studies."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import (
    Field,
    JsonValue,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_serializer,
    field_validator,
    model_validator,
)

from organelleverse.core.result import ErrorDetail
from organelleverse.operations.spec import StrictSpecModel
from organelleverse.optimization.evaluation import CapabilityIdentityV3
from organelleverse.optimization.strategies import candidate_digest

StrictNumber: TypeAlias = StrictInt | StrictFloat
Digest = Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
SafeText = Annotated[StrictStr, Field(min_length=1, pattern=r"^\S(?:.*\S)?$")]
OpaqueId = Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")]


class AttemptIdentity(StrictSpecModel):
    """Content-addressed identity of one evaluator attempt."""

    split: Literal["search", "validation"]
    split_id: SafeText
    split_content_hash: Digest
    candidate_digest: Digest
    parameter_digest: Digest
    resource_rung: StrictNumber | None = None
    logical_repeat_index: Annotated[StrictInt, Field(ge=0)]
    attempt_no: Annotated[StrictInt, Field(ge=0)]
    evaluator_digest: Digest
    evaluator_identity: CapabilityIdentityV3

    @model_validator(mode="after")
    def _candidate_is_canonical_parameter_identity(self) -> Self:
        if self.candidate_digest != self.parameter_digest:
            raise ValueError("candidate digest must equal the exact canonical parameter digest")
        return self

    @field_validator("resource_rung")
    @classmethod
    def _finite_rung(cls, value: StrictNumber | None) -> StrictNumber | None:
        if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
            raise ValueError("resource rung must be finite")
        return value


class RepeatEvidence(StrictSpecModel):
    """Raw durable result of one evaluator attempt."""

    identity: AttemptIdentity
    status: Literal["running", "succeeded", "failed", "interrupted"]
    run_id: SafeText
    seed: StrictInt
    dispatch_digest: Digest
    granted_wall_time_seconds: StrictNumber
    granted_cpu_time_seconds: StrictNumber
    granted_peak_memory_bytes: Annotated[StrictInt, Field(ge=1)]
    metrics: dict[StrictStr, StrictNumber] = Field(default_factory=dict)
    wall_time_seconds: StrictNumber = 0
    cpu_time_seconds: StrictNumber = 0
    peak_memory_bytes: Annotated[StrictInt, Field(ge=0)] = 0
    gpu_memory_bytes: Annotated[StrictInt, Field(ge=0)] | None = None
    error: ErrorDetail | None = None

    @model_validator(mode="after")
    def _validate_evidence(self) -> Self:
        numeric = (
            *self.metrics.values(),
            self.granted_wall_time_seconds,
            self.granted_cpu_time_seconds,
            self.wall_time_seconds,
            self.cpu_time_seconds,
        )
        if any(isinstance(value, bool) or not math.isfinite(value) for value in numeric):
            raise ValueError("repeat evidence numbers must be finite")
        if self.wall_time_seconds < 0 or self.cpu_time_seconds < 0:
            raise ValueError("repeat resource use must be non-negative")
        if self.granted_wall_time_seconds <= 0 or self.granted_cpu_time_seconds <= 0:
            raise ValueError("repeat resource grants must be positive")
        if self.status == "succeeded" and self.error is not None:
            raise ValueError("successful repeat evidence cannot carry an error")
        if self.status in {"failed", "interrupted"} and self.error is None:
            raise ValueError("failed repeat evidence requires an error")
        if self.status == "running" and self.error is not None:
            raise ValueError("running repeat evidence cannot carry an error")
        return self

    @field_serializer("error")
    def _serialize_error(self, error: ErrorDetail | None) -> object:
        return None if error is None else error.model_dump(mode="json", exclude={"object_id"})


class BudgetAccount(StrictSpecModel):
    """Resources durably consumed by recorded attempts."""

    attempted_runs: Annotated[StrictInt, Field(ge=0)] = 0
    wall_time_seconds: Annotated[StrictFloat, Field(ge=0)] = 0.0
    cpu_time_seconds: Annotated[StrictFloat, Field(ge=0)] = 0.0
    peak_memory_bytes: Annotated[StrictInt, Field(ge=0)] = 0


class MetricAggregate(StrictSpecModel):
    value: StrictFloat
    variance: Annotated[StrictFloat, Field(ge=0)]
    successful_runs: Annotated[StrictInt, Field(ge=0)]
    attempted_runs: Annotated[StrictInt, Field(ge=1)]
    failure_rate: Annotated[StrictFloat, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def _finite_values(self) -> Self:
        if not all(
            math.isfinite(value) for value in (self.value, self.variance, self.failure_rate)
        ):
            raise ValueError("aggregate numbers must be finite")
        if self.successful_runs > self.attempted_runs:
            raise ValueError("successful runs cannot exceed attempted runs")
        return self


class CandidateAggregate(StrictSpecModel):
    split: Literal["search", "validation"]
    split_id: SafeText
    split_content_hash: Digest
    candidate_digest: Digest
    parameter_digest: Digest
    resource_rung: StrictNumber | None = None
    evaluator_digest: Digest
    evaluator_identity: CapabilityIdentityV3
    run_ids: tuple[SafeText, ...] = Field(min_length=1)
    metrics: dict[StrictStr, MetricAggregate]
    feasible: bool
    failure_codes: tuple[SafeText, ...] = ()

    @model_validator(mode="after")
    def _closed_provenance(self) -> Self:
        if self.candidate_digest != self.parameter_digest:
            raise ValueError("aggregate candidate must retain its exact parameter digest")
        if len(self.run_ids) != len(set(self.run_ids)):
            raise ValueError("aggregate run IDs must be unique")
        counts = {
            (
                metric.successful_runs,
                metric.attempted_runs,
                metric.failure_rate,
            )
            for metric in self.metrics.values()
        }
        if len(counts) > 1:
            raise ValueError("aggregate metrics must share one repeat evidence population")
        if counts and next(iter(counts))[1] != len(self.run_ids):
            raise ValueError("aggregate attempted count must equal its durable run IDs")
        return self


class AdoptionDecision(StrictSpecModel):
    status: Literal["adopted", "baseline_retained", "failed"]
    reason_code: SafeText
    candidate_digest: Digest | None = None
    parameter_digest: Digest | None = None

    @model_validator(mode="after")
    def _adoption_has_identity(self) -> Self:
        if self.status == "adopted" and (
            self.candidate_digest is None or self.parameter_digest is None
        ):
            raise ValueError("adopted decision requires candidate and parameter digests")
        return self


class FinalExecutionLink(StrictSpecModel):
    # Historical ledgers accepted provider-qualified receipt references. Keep
    # this durable field backward compatible; new execute-best receipts are
    # constrained at their API boundary before a link can be created.
    run_receipt_id: SafeText
    executed_parameter_digest: Digest
    execution_status: Literal["succeeded", "failed"] | None = None
    linked_at: datetime


class FinalExecutionClaim(StrictSpecModel):
    """Durable ownership of the one allowed final execution side effect."""

    claim_id: OpaqueId
    executed_parameter_digest: Digest
    claimed_at: datetime


class ExecutionSelection(StrictSpecModel):
    source: Literal["adopted", "verified_baseline"]
    parameters: dict[StrictStr, JsonValue]
    parameter_digest: Digest

    @model_validator(mode="after")
    def _parameters_match_digest(self) -> Self:
        if self.parameter_digest != candidate_digest(self.parameters):
            raise ValueError("execution selection parameters must match their canonical digest")
        return self


class AdaptiveStudyRecord(StrictSpecModel):
    """Independent adaptive record stored beside, never inside, legacy records."""

    record_kind: Literal["adaptive_study.v1"] = "adaptive_study.v1"
    study_id: Annotated[StrictStr, Field(pattern=r"^study-[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
    status: Literal[
        "created",
        "baseline_search",
        "searching",
        "shortlist",
        "baseline_validation",
        "validating",
        "interrupted",
        "adopted",
        "baseline_retained",
        "failed",
    ]
    submitted_at: datetime
    completed_at: datetime | None = None
    contract_digest: Digest
    evaluator_digest: Digest
    evaluator_identity: CapabilityIdentityV3
    request: dict[StrictStr, JsonValue] = Field(default_factory=dict)
    strategy_state: dict[StrictStr, JsonValue] = Field(default_factory=dict)
    attempts: tuple[RepeatEvidence, ...] = ()
    aggregates: tuple[CandidateAggregate, ...] = ()
    pareto_candidate_digests: tuple[Digest, ...] = ()
    budget: BudgetAccount = Field(default_factory=BudgetAccount)
    decision: AdoptionDecision | None = None
    adopted_parameter_digest: Digest | None = None
    execution_selection: ExecutionSelection | None = None
    final_execution_claim: FinalExecutionClaim | None = None
    final_execution_link: FinalExecutionLink | None = None

    @model_validator(mode="after")
    def _validate_terminal_state(self) -> Self:
        terminal = {"adopted", "baseline_retained", "failed"}
        if self.status not in terminal:
            if any(
                value is not None
                for value in (
                    self.completed_at,
                    self.decision,
                    self.adopted_parameter_digest,
                    self.execution_selection,
                    self.final_execution_claim,
                    self.final_execution_link,
                )
            ):
                raise ValueError("nonterminal studies cannot carry terminal payloads")
        else:
            if self.completed_at is None or self.decision is None:
                raise ValueError("terminal studies require completion and decision")
            if self.decision.status != self.status:
                raise ValueError("study status and decision status must match")
            if self.status == "adopted":
                if self.adopted_parameter_digest != self.decision.parameter_digest:
                    raise ValueError("adopted digest must match the decision")
                if self.execution_selection is not None and (
                    self.execution_selection.source != "adopted"
                    or self.execution_selection.parameter_digest != self.adopted_parameter_digest
                ):
                    raise ValueError("adopted studies require their exact execution selection")
            elif self.adopted_parameter_digest is not None:
                raise ValueError("non-adopted studies cannot carry an adopted digest")
            if self.status == "baseline_retained" and self.execution_selection is not None:
                if self.execution_selection.source != "verified_baseline":
                    raise ValueError("baseline fallback requires a verified baseline selection")
            elif self.status == "failed" and self.execution_selection is not None:
                raise ValueError("failed studies cannot carry an execution selection")
            if self.final_execution_claim is not None:
                if self.execution_selection is None:
                    raise ValueError("final execution claims require an executable selection")
                if (
                    self.final_execution_claim.executed_parameter_digest
                    != self.execution_selection.parameter_digest
                ):
                    raise ValueError("final execution claim must match the executable selection")
                if self.final_execution_link is not None:
                    raise ValueError("final execution claim and link are mutually exclusive")
            if (
                self.final_execution_link is not None
                and self.execution_selection is not None
                and self.final_execution_link.executed_parameter_digest
                != self.execution_selection.parameter_digest
            ):
                raise ValueError("final execution must match the executable selection")
        identities = tuple(item.identity for item in self.attempts)
        if len(identities) != len(set(identities)):
            raise ValueError("attempt identities must be unique")
        run_ids = tuple(item.run_id for item in self.attempts)
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("attempt run IDs must be globally unique within a study")
        if any(
            item.identity.evaluator_digest != self.evaluator_digest
            or item.identity.evaluator_identity != self.evaluator_identity
            for item in self.attempts
        ):
            raise ValueError("attempt evaluator binding must match the study")
        return self


__all__ = [
    "AdaptiveStudyRecord",
    "AdoptionDecision",
    "AttemptIdentity",
    "BudgetAccount",
    "CandidateAggregate",
    "ExecutionSelection",
    "FinalExecutionClaim",
    "FinalExecutionLink",
    "MetricAggregate",
    "RepeatEvidence",
]
