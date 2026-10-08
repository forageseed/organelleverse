"""Closed parent-side contracts for the stdlib-only bundle worker protocol."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from organelleverse.operations.spec import StrictSpecModel

from .code_identity import ExecutionIdentity

WORKER_PROTOCOL = "organelleverse.bundle-worker.v1"

WorkerMode = Literal["inspect", "invoke"]
WorkerParameterKind = Literal[
    "positional_only",
    "positional_or_keyword",
    "keyword_only",
]
WorkerAnnotation = Literal[
    "str",
    "str_or_none",
    "path",
    "path_or_none",
    "int",
    "float",
    "bool",
    "list",
    "dict",
    "json",
    "plugin_context",
    "unsupported",
]


class WorkerParameter(StrictSpecModel):
    """Deterministic, deliberately small description of one Python parameter."""

    name: str = Field(min_length=1)
    kind: WorkerParameterKind
    required: bool
    default: JsonValue | None = None
    annotation: WorkerAnnotation


class WorkerInspection(StrictSpecModel):
    request_id: str = Field(min_length=1)
    execution_identity: ExecutionIdentity
    parameters: tuple[WorkerParameter, ...]


class WorkerResult(StrictSpecModel):
    request_id: str = Field(min_length=1)
    execution_identity: ExecutionIdentity
    value: JsonValue | None = None
    artifact_paths: tuple[str, ...] = ()


class WorkerError(StrictSpecModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    details: dict[str, JsonValue] = Field(default_factory=dict)


class WorkerRequest(StrictSpecModel):
    protocol: Literal["organelleverse.bundle-worker.v1"] = WORKER_PROTOCOL
    request_id: str = Field(min_length=1)
    operation_id: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    mode: WorkerMode
    execution_identity: ExecutionIdentity
    bundle_root: str = Field(min_length=1)
    contract: dict[str, JsonValue]
    worker_parameters: tuple[WorkerParameter, ...] = ()
    input: JsonValue | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    staging_root: str = Field(min_length=1)


class WorkerResponse(StrictSpecModel):
    protocol: Literal["organelleverse.bundle-worker.v1"] = WORKER_PROTOCOL
    request_id: str = Field(min_length=1)
    operation_id: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    mode: WorkerMode
    execution_identity: ExecutionIdentity
    status: Literal["ok", "error"]
    parameters: tuple[WorkerParameter, ...] = ()
    value: JsonValue | None = None
    artifact_paths: tuple[str, ...] = ()
    error: WorkerError | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> Self:
        if self.status == "ok" and self.error is not None:
            raise ValueError("successful worker response must not carry an error")
        if self.status == "error" and self.error is None:
            raise ValueError("failed worker response requires an error")
        if (
            self.mode == "inspect"
            and self.status == "ok"
            and (self.value is not None or self.artifact_paths)
        ):
            raise ValueError("inspection response must not carry an invocation result")
        if self.mode == "invoke" and self.status == "ok" and self.parameters:
            raise ValueError("invocation response must not carry signature parameters")
        return self


__all__ = [
    "WORKER_PROTOCOL",
    "WorkerAnnotation",
    "WorkerError",
    "WorkerInspection",
    "WorkerMode",
    "WorkerParameter",
    "WorkerParameterKind",
    "WorkerRequest",
    "WorkerResponse",
    "WorkerResult",
]
