"""Generic, traceable missing-input requests.

A :class:`NeedsInputRequest` is a closed, deterministic envelope that an
operation raises (via :class:`~organelleverse.core.errors.OrganelleNeedsInput`)
when a person or Agent must supply information that cannot be inferred
scientifically. The model is deliberately generic: it records the missing
``field`` and the available ``choices`` but knows nothing about assembly,
genome size, or any answer payload a caller may later return.

The ``request_id`` is a content address over the operation's semantic
invocation payload (destination-free), the missing field, the envelope schema
version, and the ordered choices. Two requests that differ only in their
destination therefore share an identity, so an Agent can match a resumed answer
to the original invocation without re-running side-effecting work.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NEEDS_INPUT_SCHEMA_VERSION: Literal["organelleverse.needs-input.v1"] = (
    "organelleverse.needs-input.v1"
)


def _canonical_json_bytes(value: object) -> bytes:
    """Canonical UTF-8 JSON used for deterministic request identity."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class NeedsInputRequest(BaseModel):
    """Closed envelope describing one piece of missing invocation input."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["organelleverse.needs-input.v1"] = NEEDS_INPUT_SCHEMA_VERSION
    request_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    field: str = Field(min_length=1)
    choices: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_choices(self) -> NeedsInputRequest:
        if not self.choices:
            raise ValueError("choices must be non-empty")
        if any(not choice for choice in self.choices):
            raise ValueError("choices must be non-empty strings")
        if len(set(self.choices)) != len(self.choices):
            raise ValueError("choices must be unique")
        return self


def make_needs_input_request(
    *,
    semantic_payload: Mapping[str, object],
    field: str,
    choices: tuple[str, ...],
) -> NeedsInputRequest:
    """Build a :class:`NeedsInputRequest` with a deterministic ``request_id``.

    ``request_id`` is ``sha256:`` followed by the SHA256 of canonical JSON
    containing ``schema_version``, ``semantic_payload``, ``field``, and the
    ordered ``choices``. Destination paths are excluded on purpose: callers pass
    the existing semantic invocation payload, so two invocations that differ
    only in their output location share the same request identity.
    """
    ordered_choices = tuple(choices)
    canonical = _canonical_json_bytes(
        {
            "schema_version": NEEDS_INPUT_SCHEMA_VERSION,
            "semantic_payload": dict(semantic_payload),
            "field": field,
            "choices": ordered_choices,
        }
    )
    digest = hashlib.sha256(canonical).hexdigest()
    return NeedsInputRequest(
        request_id=f"sha256:{digest}",
        field=field,
        choices=ordered_choices,
    )
