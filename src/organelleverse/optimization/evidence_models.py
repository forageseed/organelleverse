"""Scientific evaluator evidence and parameter-selection contracts."""

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


class ExecutionSelection(StrictSpecModel):
    source: Literal["adopted", "verified_baseline"]
    parameters: dict[StrictStr, JsonValue]
    parameter_digest: Digest

    @model_validator(mode="after")
    def _parameters_match_digest(self) -> Self:
        if self.parameter_digest != candidate_digest(self.parameters):
            raise ValueError("execution selection parameters must match their canonical digest")
        return self

__all__ = ['AttemptIdentity', 'RepeatEvidence', 'BudgetAccount', 'MetricAggregate', 'CandidateAggregate', 'AdoptionDecision', 'ExecutionSelection']
