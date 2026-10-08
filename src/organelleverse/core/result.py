"""Immutable, human-readable and Agent-readable operation results."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, cast

from pydantic import Field, field_serializer, field_validator, model_validator

from .artifacts import ArtifactRef
from .base import StrictFrozenModel
from .errors import OrganelleExecutionError, OrganelleInternalError
from .frozen import FrozenJson, FrozenMap, freeze_json, thaw_json
from .provenance import ResultProvenance

ResultStatus = Literal["ok", "warning", "failed"]
ResultScope = Literal["mitochondrion", "plastid", "cytonuclear", "mixed", "none"]


class Finding(StrictFrozenModel[Literal["finding"]]):
    """One machine-readable observation associated with a result."""

    kind: Literal["finding"] = "finding"
    code: str
    metric: str = ""
    value: int | float | str | bool | None = None
    unit: str = ""
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence_artifact_ids: tuple[str, ...] = ()

    @field_validator("value")
    @classmethod
    def validate_finite_value(
        cls, value: int | float | str | bool | None
    ) -> int | float | str | bool | None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("finding value must be finite")
        return value


class ErrorDetail(StrictFrozenModel[Literal["error"]]):
    """Structured failure detail that can be raised or handled by an Agent."""

    kind: Literal["error"] = "error"
    code: str
    message: str
    details: FrozenMap[FrozenJson] = Field(default_factory=FrozenMap)
    retryable: bool = False
    suggested_action: FrozenMap[FrozenJson] = Field(default_factory=FrozenMap)

    @field_validator("details", "suggested_action", mode="before")
    @classmethod
    def freeze_objects(cls, value: object) -> FrozenMap[FrozenJson]:
        try:
            frozen = freeze_json({} if value is None else value)
        except OrganelleInternalError as error:
            raise ValueError(str(error)) from error
        if not isinstance(frozen, FrozenMap):
            raise ValueError("error mapping field must be a JSON object")
        return frozen

    @field_serializer("details", "suggested_action")
    def serialize_objects(self, value: FrozenMap[FrozenJson]) -> object:
        return thaw_json(value)


class OperationSuggestion(StrictFrozenModel[Literal["operation_suggestion"]]):
    """One deterministic next operation suggested by a result."""

    kind: Literal["operation_suggestion"] = "operation_suggestion"
    operation_id: str
    reason_code: str
    parameter_changes: FrozenMap[FrozenJson] = Field(default_factory=FrozenMap)

    @field_validator("parameter_changes", mode="before")
    @classmethod
    def freeze_parameters(cls, value: object) -> FrozenMap[FrozenJson]:
        try:
            frozen = freeze_json({} if value is None else value)
        except OrganelleInternalError as error:
            raise ValueError(str(error)) from error
        if not isinstance(frozen, FrozenMap):
            raise ValueError("parameter_changes must be a JSON object")
        return frozen

    @field_serializer("parameter_changes")
    def serialize_parameters(self, value: FrozenMap[FrozenJson]) -> object:
        return thaw_json(value)


class OrganelleResult(StrictFrozenModel[Literal["result"]]):
    """Immutable outcome of one named organelle analysis operation."""

    schema_version: Literal["organelleverse.result.v1"] = "organelleverse.result.v1"
    kind: Literal["result"] = "result"
    operation_id: str
    operation_version: str = "1.0"
    scope: ResultScope
    status: ResultStatus
    summary_text: str = ""
    metrics: FrozenMap[FrozenJson] = Field(default_factory=FrozenMap)
    findings: tuple[Finding, ...] = ()
    flags: tuple[str, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    provenance: ResultProvenance | None = None
    errors: tuple[ErrorDetail, ...] = ()
    suggested_operations: tuple[OperationSuggestion, ...] = ()

    @field_validator("metrics", mode="before")
    @classmethod
    def freeze_metrics(cls, value: object) -> FrozenMap[FrozenJson]:
        try:
            frozen = freeze_json({} if value is None else value)
        except OrganelleInternalError as error:
            raise ValueError(str(error)) from error
        if not isinstance(frozen, FrozenMap):
            raise ValueError("metrics must be a JSON object")
        return frozen

    @field_serializer("metrics")
    def serialize_metrics(self, value: FrozenMap[FrozenJson]) -> object:
        return thaw_json(value)

    @field_serializer("findings", "errors", "suggested_operations")
    def serialize_contract_items(
        self,
        value: tuple[Finding, ...] | tuple[ErrorDetail, ...] | tuple[OperationSuggestion, ...],
    ) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json", exclude={"object_id"}) for item in value]

    @field_serializer("artifacts")
    def serialize_artifacts(self, value: tuple[ArtifactRef, ...]) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json", exclude={"object_id"}) for item in value]

    @field_serializer("provenance")
    def serialize_provenance(self, value: ResultProvenance | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return value.model_dump(mode="json", exclude={"object_id"})

    @model_validator(mode="after")
    def validate_status(self) -> OrganelleResult:
        if self.status == "failed" and not self.errors:
            raise ValueError("failed result requires at least one error")
        if self.status == "ok" and self.errors:
            raise ValueError("ok result cannot contain errors")
        if (
            self.status == "ok"
            and self.provenance is not None
            and self.provenance.requested_backend
            and self.provenance.actual_backend
            and self.provenance.requested_backend != self.provenance.actual_backend
        ):
            raise ValueError("a requested-backend fallback requires warning status")
        return self

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "operation_id": self.operation_id,
            "operation_version": self.operation_version,
            "scope": self.scope,
            "status": self.status,
            "summary_text": self.summary_text,
            "metrics": thaw_json(self.metrics),
            "findings": [
                item.model_dump(mode="json", exclude={"object_id"}) for item in self.findings
            ],
            "flags": self.flags,
            "artifacts": [item.object_id for item in self.artifacts],
            "provenance": self.provenance.object_id if self.provenance is not None else None,
            "errors": [item.model_dump(mode="json", exclude={"object_id"}) for item in self.errors],
            "suggested_operations": [
                item.model_dump(mode="json", exclude={"object_id"})
                for item in self.suggested_operations
            ],
        }

    def summary(self) -> str:
        if self.summary_text:
            return self.summary_text
        return f"{self.operation_id}: {self.status}"

    def raise_for_failure(self) -> None:
        if self.status != "failed":
            return
        first = self.errors[0]
        details = cast(Mapping[str, Any], thaw_json(first.details))
        suggested_action = cast(Mapping[str, Any], thaw_json(first.suggested_action))
        raise OrganelleExecutionError(
            code=first.code,
            message=first.message,
            details=details,
            retryable=first.retryable,
            suggested_action=suggested_action,
        )
