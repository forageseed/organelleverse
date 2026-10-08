"""Closed durable and browser-safe models for normal plugin runs."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, cast

from pydantic import Field, JsonValue, field_serializer, field_validator, model_validator

from organelleverse.capabilities.plugin_descriptor import PluginDescriptor
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.spec import (
    InlineParameterValue,
    ParameterValue,
    PathParameterValue,
    StrictSpecModel,
)

__all__ = [
    "PluginRunArtifact",
    "PluginRunEvent",
    "PluginRunRecord",
    "PluginRunRequest",
    "RunContext",
    "RunContextArtifact",
    "RunContextDiagnostic",
    "RunError",
]

RunStatus = Literal["queued", "running", "succeeded", "failed"]
TerminalRunStatus = Literal["succeeded", "failed"]
ArtifactRole = Literal["output", "log", "evidence"]

_PATH_ENVELOPE_KEYS = frozenset({"kind", "value", "context"})
_INLINE_ENVELOPE_KEYS = frozenset({"kind", "value", "sha256", "format", "encoding"})


def _parse_transport_envelope(value: object) -> object:
    if not isinstance(value, dict):
        return value
    mapping = cast(dict[object, object], value)
    keys = set(mapping)
    kind = mapping.get("kind")
    if (
        kind == "path"
        and {"kind", "value"}.issubset(keys)
        and keys.issubset(_PATH_ENVELOPE_KEYS)
    ):
        return PathParameterValue.model_validate(mapping)
    if (
        kind == "inline"
        and {"kind", "value", "sha256", "format"}.issubset(keys)
        and keys.issubset(_INLINE_ENVELOPE_KEYS)
    ):
        return InlineParameterValue.model_validate(mapping)
    return mapping


class PluginRunRequest(StrictSpecModel):
    capability_id: str
    inputs: dict[str, str | ParameterValue]
    parameters: dict[str, ParameterValue | JsonValue] = Field(default_factory=dict)

    @field_validator("inputs", "parameters", mode="before")
    @classmethod
    def parse_tagged_parameter_values(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        parsed = dict(cast(dict[object, object], value))
        for name, item in parsed.items():
            parsed[name] = _parse_transport_envelope(item)
        return parsed

    @model_validator(mode="after")
    def reject_ambiguous_fields(self) -> PluginRunRequest:
        overlap = sorted(set(self.inputs) & set(self.parameters))
        if overlap:
            raise ValueError(f"plugin inputs and parameters overlap: {overlap}")
        return self


class RunError(StrictSpecModel):
    error_code: str
    message: str
    details: dict[str, JsonValue]
    retryable: bool
    suggested_action: dict[str, JsonValue]


class PluginRunEvent(StrictSpecModel):
    """One progress, stage, or log event emitted during a plugin run."""

    event_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    sequence: int = Field(ge=0)
    kind: Literal["stage", "tool", "log"]
    stage: str | None = None
    percent: int | None = Field(default=None, ge=0, le=100)
    message: str | None = None
    created_at: datetime



class PluginRunArtifact(StrictSpecModel):
    artifact_id: str
    output_name: str | None
    role: ArtifactRole
    format: str
    media_type: str
    size_bytes: int


class PluginRunRecord(StrictSpecModel):
    """One durable application receipt for a normal plugin invocation.

    The two optional identities permit lossless loading of pre-Task-1 desktop
    history only. Every newly submitted record supplies ``contract_identity``;
    an L6 result supplies ``l6_run_id`` from its provenance.
    """

    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    l6_run_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    capability_id: str
    contract_identity: str | None = None
    status: RunStatus
    submitted_at: datetime
    completed_at: datetime | None = None
    request: PluginRunRequest
    descriptor: PluginDescriptor
    result: OrganelleResult | None = None
    summary: dict[str, JsonValue] = Field(default_factory=dict)
    artifacts: tuple[PluginRunArtifact, ...] = ()
    error: RunError | None = None

    @field_serializer("result")
    def _serialize_result(self, result: OrganelleResult | None) -> object:
        if result is None:
            return None
        return result.model_dump(mode="json", exclude={"object_id"})


class RunContextArtifact(StrictSpecModel):
    object_id: str
    artifact_id: str
    output_name: str | None
    role: ArtifactRole
    format: str
    media_type: str
    size_bytes: int


class RunContextDiagnostic(StrictSpecModel):
    error_code: str
    message: str
    details: dict[str, JsonValue]
    retryable: bool
    suggested_action: dict[str, JsonValue]


class RunContext(StrictSpecModel):
    run_id: str
    l6_run_id: str | None
    capability_id: str
    contract_identity: str
    status: TerminalRunStatus
    request: PluginRunRequest
    result_id: str | None
    summary_text: str
    metrics: dict[str, JsonValue]
    findings: tuple[dict[str, JsonValue], ...]
    flags: tuple[str, ...]
    artifacts: tuple[RunContextArtifact, ...]
    diagnostics: tuple[RunContextDiagnostic, ...]
    provenance_links: tuple[str, ...]
