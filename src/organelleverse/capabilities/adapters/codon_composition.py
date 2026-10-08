"""Adapter decisions for the ``codon_composition`` domain's restored capabilities.

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``codon_composition.amino_acid`` / ``.codon_usage`` - ``canonical``
  result_shape, first parameter ``cds_fasta: str | Path`` (not a bare
  ``OrganelleGenome``), so neither matches ``canonical_core``. Resolved
  below: ``cds_fasta`` gets ``path`` (really opened via ``read_fasta`` in
  ``codon_core._count_codons``, which both call); both already return a
  real ``OrganelleResult`` they built themselves, so ``result_codec =
  "canonical"``. ``organelle``/``genetic_code``/``include_stop`` keep their
  own defaults. ``side_effects = ["read_files"]``.
* ``codon_composition.compute_amino_acid_composition`` /
  ``.compute_codon_usage`` - the ``json`` twins (``codon_core.py``, no
  ``organelleverse.core`` types): same ``cds_fasta`` gets ``path``; both
  ``dict`` returns are recursively finite JSON at every real call, so
  ``json_metric`` under ``"amino_acid_composition"`` /
  ``"codon_usage"`` respectively. ``side_effects = ["read_files"]``.
* ``codon_composition.compute_enc`` - resolved below: ``counts: dict[str,
  int]`` and ``codon_to_aa: dict[str, str]`` are both fully parameterized
  mappings (unlike the ``list[dict]`` gap documented in the ``phylogeny``
  adapter, both key and value types here are concrete JSON primitives), so
  both get ``json``. The bare ``float`` return is finite JSON, so
  ``json_metric`` under ``"enc"``.
* ``codon_composition.resolve_genetic_code`` - resolved below: both
  parameters (``organelle``, ``genetic_code``) have defaults, so zero
  parameters are declared (mirrors
  ``morphology.check_all_backends``/``phylogeny``'s
  zero-required-parameter capabilities). The bare ``int`` return is finite
  JSON, so ``json_metric`` under ``"genetic_code_id"``.
* ``codon_composition.write_amino_acid`` / ``.write_usage`` - **excluded**:
  both take ``OrganelleResult | Mapping[str, Any]`` as their required first
  parameter - the charter's ``LEGACY_RESULT`` gap verbatim. No entry here
  for either.

Capability Plan 04 Task 1 adds ``FIXTURES`` for all six bindable
capabilities in this domain - none uses a timestamp, a random seed, or an
external tool (``codon.py``/``codon_core.py`` only ever import Biopython's
static, version-pinned NCBI genetic-code tables), so every one is safe for
an ``exact`` fixture. ``_CDS_FASTA`` is a 9 bp CDS (``ATG GCT TAA`` = Met,
Ala, Stop) shared by the four ``cds_fasta``-parameterized capabilities.
``compute_enc`` needs no input file at all: ``_STANDARD_CODON_TABLE`` is
NCBI transl_table 1, written out by hand (matches
``organelleverse._bio.get_codon_table(1)`` - confirmed interactively, not
re-imported here so this adapter stays free of a Biopython import merely to
be loaded) and ``_ENC_COUNTS`` is a deterministic, non-zero count for every
codon so every synonymous family actually contributes.
"""

from __future__ import annotations

from dataclasses import dataclass

from organelleverse.capabilities.adapters import FixtureCase, FixtureFile
from organelleverse.operations.spec import ParameterCodec, ResultCodec, SideEffect

_CDS_FASTA = FixtureFile(relative_path="cds.fasta", content=">gene1\nATGGCTTAA\n")

# NCBI transl_table 1 (the standard genetic code), written out by hand -
# matches organelleverse._bio.get_codon_table(1).
_STANDARD_CODON_TABLE: dict[str, str] = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}
_ENC_COUNTS: dict[str, int] = {
    codon: (index % 5) + 1 for index, codon in enumerate(_STANDARD_CODON_TABLE)
}


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
    "codon_composition.amino_acid": NamedParameterOverride(
        parameters=(ParameterOverride(name="cds_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "codon_composition.codon_usage": NamedParameterOverride(
        parameters=(ParameterOverride(name="cds_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "codon_composition.compute_amino_acid_composition": NamedParameterOverride(
        parameters=(ParameterOverride(name="cds_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="amino_acid_composition",
        side_effects=(SideEffect.READ_FILES,),
    ),
    "codon_composition.compute_codon_usage": NamedParameterOverride(
        parameters=(ParameterOverride(name="cds_fasta", codec=ParameterCodec.PATH),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="codon_usage",
        side_effects=(SideEffect.READ_FILES,),
    ),
    "codon_composition.compute_enc": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="counts", codec=ParameterCodec.JSON),
            ParameterOverride(name="codon_to_aa", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="enc",
        side_effects=(),
    ),
    "codon_composition.resolve_genetic_code": NamedParameterOverride(
        parameters=(),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="genetic_code_id",
        side_effects=(),
    ),
}
"""``codon_composition.write_amino_acid`` and ``.write_usage`` are
deliberately absent - see the module docstring for why neither's
``OrganelleResult | Mapping`` parameter has an honest codec.
"""

FIXTURES: dict[str, tuple[FixtureCase, ...]] = {
    "codon_composition.amino_acid": (
        FixtureCase(
            case="basic",
            files=(_CDS_FASTA,),
            parameters={"cds_fasta": _CDS_FASTA.relative_path},
        ),
    ),
    "codon_composition.codon_usage": (
        FixtureCase(
            case="basic",
            files=(_CDS_FASTA,),
            parameters={"cds_fasta": _CDS_FASTA.relative_path},
        ),
    ),
    "codon_composition.compute_amino_acid_composition": (
        FixtureCase(
            case="basic",
            files=(_CDS_FASTA,),
            parameters={"cds_fasta": _CDS_FASTA.relative_path},
        ),
    ),
    "codon_composition.compute_codon_usage": (
        FixtureCase(
            case="basic",
            files=(_CDS_FASTA,),
            parameters={"cds_fasta": _CDS_FASTA.relative_path},
        ),
    ),
    "codon_composition.compute_enc": (
        FixtureCase(
            case="basic",
            parameters={"counts": _ENC_COUNTS, "codon_to_aa": _STANDARD_CODON_TABLE},
        ),
    ),
    "codon_composition.resolve_genetic_code": (FixtureCase(case="basic", parameters={}),),
}

__all__ = ["FIXTURES", "OVERRIDES", "NamedParameterOverride", "ParameterOverride"]
