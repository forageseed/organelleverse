"""Adapter decisions for the ``composition`` domain's restored capabilities.

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``composition.gc_content`` - ``canonical`` result_shape, but its first
  parameter is ``genome_fasta: str | Path`` (not a bare ``OrganelleGenome``),
  so it does not match the generator's generic ``canonical_core`` shape.
  Resolved below: ``genome_fasta`` gets the ``path`` codec (a real file the
  implementation reads via ``read_fasta``); the implementation already
  returns a real ``OrganelleResult`` it built itself, so
  ``result_codec = "canonical"`` revalidates it unchanged. ``window_size``
  keeps its own default (not exposed).
* ``composition.compute_gc_content`` - the ``json`` twin of ``gc_content``
  (``composition/gc_core.py``, no ``organelleverse.core`` types at all):
  same ``fasta_path: str | Path`` first parameter gets ``path``; its
  ``dict`` return is recursively finite JSON at every real call, so
  ``json_metric`` under ``"gc_metrics"``.
* ``composition.write_content`` - takes
  ``result: OrganelleResult | Mapping[str, Any]`` as its required first
  parameter. No implemented ``ParameterCodec`` can decode an
  ``OrganelleResult`` as an Agent-facing JSON argument
  (``ParameterCodec.LEGACY_RESULT`` exists in the schema but is explicitly
  unimplemented - see ``python_binding._UNIMPLEMENTED_PARAMETER_CODECS``),
  so no entry here - the generator fails closed for it with
  ``capability.no_adapter_override``.

Capability Plan 04 Task 1 adds ``FIXTURES`` for both bindable capabilities:
a 20 bp FASTA (6 G + 6 C + 8 A -> ``gc_content == 0.6``, one window since
20 bp < the default ``window_size`` of 500) is written to
``fixtures/basic/input/genome.fasta`` and the generator's own
``_capture_fixtures`` actually runs ``gc_content``/``compute_gc_content``
against it once, freezing whatever they *really* return - never a value
reasoned about from ``gc.py``'s source. Both implementations are pure
Python (no timestamps, no randomness, no external tool) - see
``composition/gc.py``'s ``_provenance``, which never sets
``started_at``/``finished_at`` - so an ``exact`` fixture never flakes here.
"""

from __future__ import annotations

from dataclasses import dataclass

from organelleverse.capabilities.adapters import FixtureCase, FixtureFile
from organelleverse.operations.spec import ParameterCodec, ResultCodec, SideEffect

_GC_FASTA = FixtureFile(relative_path="genome.fasta", content=">demo\nGGGGGGCCCCCCAAAAAAAA\n")


@dataclass(frozen=True)
class ParameterOverride:
    """One named, Agent-facing parameter the generator could not derive alone."""

    name: str
    codec: ParameterCodec
    path_role: str | None = None


@dataclass(frozen=True)
class NamedParameterOverride:
    """The full ``named_parameters`` binding plan for one restored capability."""

    parameters: tuple[ParameterOverride, ...]
    result_codec: ResultCodec
    result_key: str | None
    side_effects: tuple[SideEffect, ...]


OVERRIDES: dict[str, NamedParameterOverride] = {
    "composition.gc_content": NamedParameterOverride(
        parameters=(ParameterOverride(name="genome_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "composition.compute_gc_content": NamedParameterOverride(
        parameters=(ParameterOverride(name="fasta_path", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="gc_metrics",
        side_effects=(SideEffect.READ_FILES,),
    ),
}
"""``composition.write_content`` is deliberately absent - see the module
docstring for why no codec can decode its ``OrganelleResult | Mapping``
parameter honestly.
"""

FIXTURES: dict[str, tuple[FixtureCase, ...]] = {
    "composition.gc_content": (
        FixtureCase(
            case="basic",
            files=(_GC_FASTA,),
            parameters={"genome_fasta": _GC_FASTA.relative_path},
        ),
    ),
    "composition.compute_gc_content": (
        FixtureCase(
            case="basic",
            files=(_GC_FASTA,),
            parameters={"fasta_path": _GC_FASTA.relative_path},
        ),
    ),
}

__all__ = ["FIXTURES", "OVERRIDES", "NamedParameterOverride", "ParameterOverride"]
