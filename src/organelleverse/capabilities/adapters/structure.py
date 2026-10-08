"""Adapter decisions for the ``structure`` domain's restored capabilities.

``structure.introns`` matches the generator's generic ``canonical_core``
shape on its own (``genome: OrganelleGenome`` first parameter, ``->
OrganelleResult``) - no entry needed here for it.

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``structure.multiconf`` / ``.repeats`` / ``.resolve_configs`` -
  ``canonical`` result_shape, first parameter ``genome_fasta: str | Path``
  (not a bare ``OrganelleGenome``), so none matches ``canonical_core``.
  Resolved below: ``genome_fasta`` gets ``path`` (each really opens it via
  ``read_fasta``); all three already return a real ``OrganelleResult`` they
  built themselves, so ``result_codec = "canonical"``. Every other
  parameter (including ``resolve_configs``'s optional ``gfa_path``) keeps
  its own default. ``side_effects = ["read_files"]`` for all three.
* ``structure.compute_multiconf`` / ``.compute_repeats`` - the ``json``
  twins (``structure/structure_core.py``, no ``organelleverse.core``
  types): same ``fasta_path`` gets ``path``; both ``dict`` returns are
  recursively finite JSON at every real call, so ``json_metric`` under
  ``"multiconf"`` / ``"repeats"`` respectively. ``side_effects =
  ["read_files"]``.
* ``structure.write_introns`` / ``.write_multiconf`` / ``.write_repeats`` /
  ``.write_resolve_configs`` - **excluded**: all four take
  ``OrganelleResult | Mapping[str, Any]`` as a required parameter - the
  charter's ``LEGACY_RESULT`` gap verbatim. No entry here for any of the
  four.

Capability Plan 04 Task 1 adds ``FIXTURES`` for the two ``json_metric``
capabilities only. ``multiconf``/``.repeats``/``.resolve_configs`` are
deliberately absent from ``FIXTURES``: all three call ``datetime.now(UTC)``
directly and embed the real wall-clock result in their
``OrganelleResult.provenance`` (``structure.py:117,166,173,187,221,227,
243,257,266,434,471,481``) - every real run produces byte-different output,
so no ``exact`` fixture can ever pass twice, and a timestamp string is not
a numeric leaf ``numeric_tolerance`` could absorb either.
``compute_multiconf``/``compute_repeats`` (``structure_core.py``) return a
  plain ``dict`` with no provenance at all, so neither carries that
  non-determinism. ``compute_repeats`` scans exact tandem motifs using MISA's
  per-unit copy thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

from organelleverse.capabilities.adapters import FixtureCase, FixtureFile
from organelleverse.operations.spec import ParameterCodec, ResultCodec, SideEffect

# A 50 bp block (structure_core.compute_multiconf's default min_repeat_len)
# repeated twice, separated by a 10 bp spacer -> exactly one real direct
# repeat pair, confirmed interactively.
_REPEAT_BLOCK = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC"
assert len(_REPEAT_BLOCK) == 50
_MULTICONF_FASTA = FixtureFile(
    relative_path="multiconf.fasta",
    content=f">demo\n{_REPEAT_BLOCK}TTTTTTTTTT{_REPEAT_BLOCK}\n",
)

# A dinucleotide "AT" run of 10 copies, flanked by context below the
# mononucleotide threshold. MISA thresholds admit units AT, ATAT, and ATATAT.
_REPEATS_FASTA = FixtureFile(relative_path="repeats.fasta", content=f">demo\nGGGG{'AT' * 10}CCCC\n")


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
    "structure.multiconf": NamedParameterOverride(
        parameters=(ParameterOverride(name="genome_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "structure.repeats": NamedParameterOverride(
        parameters=(ParameterOverride(name="genome_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "structure.resolve_configs": NamedParameterOverride(
        parameters=(ParameterOverride(name="genome_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "structure.compute_multiconf": NamedParameterOverride(
        parameters=(ParameterOverride(name="fasta_path", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="multiconf",
        side_effects=(SideEffect.READ_FILES,),
    ),
    "structure.compute_repeats": NamedParameterOverride(
        parameters=(ParameterOverride(name="fasta_path", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="repeats",
        side_effects=(SideEffect.READ_FILES,),
    ),
}
"""``structure.write_introns``, ``.write_multiconf``, ``.write_repeats``,
and ``.write_resolve_configs`` are deliberately absent - see the module
docstring for why each's ``OrganelleResult | Mapping`` parameter has no
honest codec.
"""

FIXTURES: dict[str, tuple[FixtureCase, ...]] = {
    "structure.compute_multiconf": (
        FixtureCase(
            case="basic",
            files=(_MULTICONF_FASTA,),
            parameters={"fasta_path": _MULTICONF_FASTA.relative_path},
        ),
    ),
    "structure.compute_repeats": (
        FixtureCase(
            case="basic",
            files=(_REPEATS_FASTA,),
            parameters={"fasta_path": _REPEATS_FASTA.relative_path},
        ),
    ),
}

__all__ = ["FIXTURES", "OVERRIDES", "NamedParameterOverride", "ParameterOverride"]
