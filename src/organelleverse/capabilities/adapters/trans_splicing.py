"""Adapter decisions for the ``trans_splicing`` domain's restored capabilities.

**Zero-out ruling (owner mandate 2026-08-14, final):** ``compute_trans_splicing`` is INTERNAL - unannotated ``genomes`` parameter in the sequence-non-core-return class (the internals of the published trans-splicing surface).
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
EXCLUDE = frozenset({"trans_splicing.compute_trans_splicing"})

OVERRIDES: dict[str, NamedParameterOverride] = {}

__all__ = ["EXCLUDE", "OVERRIDES"]
