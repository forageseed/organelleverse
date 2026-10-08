"""Adapter decisions for the ``barcode`` domain's restored capabilities.

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``barcode.design_barcode`` - ``canonical`` result_shape, first parameter
  ``alignment_fasta: str | Path`` (not a bare ``OrganelleGenome``), so it
  does not match ``canonical_core``. Resolved below: ``alignment_fasta``
  gets the ``path`` codec (a real file ``read_fasta`` opens); the
  implementation already returns a real ``OrganelleResult`` it built
  itself, so ``result_codec = "canonical"``. ``min_window`` /
  ``min_p_variability`` keep their own defaults.
* ``barcode.identify`` - ``canonical``, two required file parameters
  (``query_fasta``, ``reference_fasta: str | Path``), both resolved to
  ``path``; same ``canonical`` result reasoning as ``design_barcode``.
* ``barcode.compute_barcode_candidates`` - the ``json`` twin of
  ``design_barcode`` (``barcode/barcode_core.py``): ``alignment_fasta``
  gets ``path``; its ``dict`` return is recursively finite JSON at every
  real call, so ``json_metric`` under ``"barcode_candidates"``.
* ``barcode.compute_jaccard_identity`` - the ``json`` twin of
  ``identify``: ``query_fasta`` and ``reference_fasta`` both get ``path``;
  ``json_metric`` under ``"jaccard_identity"``.

Capability Plan 04 Task 1 adds ``FIXTURES`` for the two ``json_metric``
capabilities only. ``design_barcode``/``identify`` are deliberately absent
from ``FIXTURES``: both call ``barcode._contract.utc_now()`` and embed real
wall-clock ``started_at``/``finished_at`` timestamps in their
``OrganelleResult.provenance`` (see ``barcode.py:38,61-62,107,112``) - every
real run produces byte-different output, so no ``exact`` fixture (and no
``numeric_tolerance`` one either - a timestamp string is not a numeric leaf)
can ever pass twice. ``compute_barcode_candidates``/``compute_jaccard_identity``
return a plain ``dict`` with no provenance at all, so they carry no such
non-determinism.
"""

from __future__ import annotations

from dataclasses import dataclass

from organelleverse.capabilities.adapters import FixtureCase, FixtureFile
from organelleverse.operations.spec import ParameterCodec, ResultCodec, SideEffect

# 120 bp, 2 sequences, 10 differing positions inside the first 100 bp ->
# design_barcode/compute_barcode_candidates' one 100 bp window (min_window's
# default) has p_variable == 0.1, above the default 0.05 threshold.
_ALIGNMENT_SEQ_A = "A" * 120
_ALIGNMENT_SEQ_B = "".join(
    "G" if i in range(0, 100, 10) else base for i, base in enumerate(_ALIGNMENT_SEQ_A)
)
_ALIGNMENT_FASTA = FixtureFile(
    relative_path="alignment.fasta",
    content=f">seqA\n{_ALIGNMENT_SEQ_A}\n>seqB\n{_ALIGNMENT_SEQ_B}\n",
)

# A 40 bp query that exactly matches one of two 40 bp references (k=31
# default -> both have real 31-mers) and is unrelated to the other, so
# compute_jaccard_identity's best match is unambiguous.
_QUERY_SEQ = "ACGT" * 10
_QUERY_FASTA = FixtureFile(relative_path="query.fasta", content=f">query\n{_QUERY_SEQ}\n")
_REFERENCE_FASTA = FixtureFile(
    relative_path="reference.fasta",
    content=f">match\n{_QUERY_SEQ}\n>distant\n{'T' * 40}\n",
)


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
    "barcode.design_barcode": NamedParameterOverride(
        parameters=(ParameterOverride(name="alignment_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "barcode.identify": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="query_fasta", codec=ParameterCodec.PATH),
            ParameterOverride(name="reference_fasta", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "barcode.compute_barcode_candidates": NamedParameterOverride(
        parameters=(ParameterOverride(name="alignment_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="barcode_candidates",
        side_effects=(SideEffect.READ_FILES,),
    ),
    "barcode.compute_jaccard_identity": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="query_fasta", codec=ParameterCodec.PATH),
            ParameterOverride(name="reference_fasta", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="jaccard_identity",
        side_effects=(SideEffect.READ_FILES,),
    ),
}
"""All 4 ledger records for this domain resolve - none is absent."""

FIXTURES: dict[str, tuple[FixtureCase, ...]] = {
    "barcode.compute_barcode_candidates": (
        FixtureCase(
            case="basic",
            files=(_ALIGNMENT_FASTA,),
            parameters={"alignment_fasta": _ALIGNMENT_FASTA.relative_path},
        ),
    ),
    "barcode.compute_jaccard_identity": (
        FixtureCase(
            case="basic",
            files=(_QUERY_FASTA, _REFERENCE_FASTA),
            parameters={
                "query_fasta": _QUERY_FASTA.relative_path,
                "reference_fasta": _REFERENCE_FASTA.relative_path,
            },
        ),
    ),
}

__all__ = ["FIXTURES", "OVERRIDES", "NamedParameterOverride", "ParameterOverride"]
