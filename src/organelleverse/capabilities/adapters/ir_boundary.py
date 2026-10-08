"""Adapter decisions for the ``ir_boundary`` domain's restored capabilities.

**Zero-out ruling (owner mandate 2026-08-14, final):** ``compute_ir_boundary`` is INTERNAL - the plain-dict variant of the published ``ir_boundary.ir_boundary`` sequence operation (core-sequence input, non-core dict return); binding it would need a sequence-with-json-metric extension for no new scientific surface.
"""

from __future__ import annotations

from organelleverse.operations.spec import ParameterCodec, ResultCodec, SideEffect


@__import__("dataclasses").dataclass(frozen=True)
class ParameterOverride:
    name: str
    codec: ParameterCodec
    path_role: str | None = None


@__import__("dataclasses").dataclass(frozen=True)
class NamedParameterOverride:
    parameters: tuple[ParameterOverride, ...]
    result_codec: ResultCodec
    result_key: str | None
    side_effects: tuple[SideEffect, ...]

#: Capabilities this domain explicitly refuses to generate; see the module
#: docstring for the precise reason.
EXCLUDE = frozenset({"ir_boundary.compute_ir_boundary"})

OVERRIDES: dict[str, NamedParameterOverride] = {}

__all__ = ["EXCLUDE", "OVERRIDES"]
