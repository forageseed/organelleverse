"""Adapter decisions for the ``comparative`` domain's restored capabilities.

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``comparative.normalize_gene_names`` / ``.normalize_genes`` - resolved
  below: both take a required, fully parameterized ``list[str]`` (``names``
  / ``gene_names``) as their sole parameter, so it gets ``json``; both
  ``dict[str, str]`` returns are recursively finite JSON at every real
  call, so ``json_metric`` under ``"gene_name_map"`` for both. No side
  effects (pure string normalization, no I/O).
* ``comparative.compare_genes`` / ``.compare_genomes`` / ``.gene_table`` /
  ``.synteny`` - **excluded**: each takes a required
  ``genomes: list[OrganelleGenome]`` as its first parameter - the
  ``list[OrganelleGenome]`` blocker class (first identified in the
  ``ir_boundary`` adapter, confirmed by a real bind attempt below).
  ``canonical_core`` only recognizes a *bare* core-typed first parameter,
  never a list of one, and ``OrganelleGenome`` (a ``StrictFrozenModel``) is
  not a registered ``OperationParameterModel``, so ``list[OrganelleGenome]``
  is JSON-unsafe at the real runtime binder too - verified: binding
  ``comparative.compare_genes`` at ``verify_capability`` time raises
  ``OrganelleContractError: parameter 'genomes' has no JSON-safe annotation
  for codec 'json'``, the same error observed for ``phylogeny.build_mjn``'s
  ``list[dict]``. No entry here for any of the four.
* ``comparative.compute_gene_intersection`` / ``.compute_gene_sets`` /
  ``.compute_synteny_score`` - **excluded**: all three take a sole
  parameter, ``genomes``, with *no annotation at all*
  (``comparative/compare_core.py:6,11,32``: ``def compute_gene_sets(genomes)
  -> list[set[str]]``, etc.) - the same shape already documented for
  ``pangenome.compute_gene_pav``/``trans_splicing.compute_trans_splicing``.
  Even had the parameter been annotated, two of the three return a bare
  Python ``set`` (``set[str]``, or a ``list`` containing one) - not one of
  the finite-JSON types ``codecs._is_finite_json`` recognizes
  (``None``/``bool``/``str``/``int``/``float``/``list``/``tuple``/``dict``;
  a raw ``set`` is never accepted) - a second, independent reason neither
  would resolve even with an annotation. No entry here for any of the
  three.
* ``comparative.write_gene_comparison`` / ``.write_gene_table`` /
  ``.write_genome_comparison`` / ``.write_synteny`` - **excluded**: all
  four take ``OrganelleResult | Mapping[str, Any]`` as a required
  parameter - the charter's ``LEGACY_RESULT`` gap verbatim. No entry here
  for any of the four.
* ``comparative.compute_genome_identity`` - resolved below: first
  parameter ``reference_fasta: str | Path`` gets ``path`` (a real FASTA
  ``read_fasta`` opens), ``query_fastas: Sequence[str | Path]`` gets
  ``path`` with the list-of-paths plan (every element existence-checked
  and hashed), and the optional ``reference_genbank: str | Path | None``
  gets ``path`` with a nullable Agent field. The raw dict return is
  recursively finite JSON at every real call, so ``json_metric`` under
  ``"genome_identity"``. Side effects: reads the named files; the optional
  TSV table output is a non-Agent parameter left at its default. The
  implementation anchors with ``mappy`` (minimap2) and aligns each block
  with ``edlib``; both are declared dependencies, not external programs.

Capability Plan 04 Task 1 adds ``FIXTURES`` for both bindable capabilities.
Both are pure string normalization (no I/O, no provenance at all - see the
module docstring's own note that both declare ``side_effects=()``), so
``exact`` never flakes. The input list mixes a recognized alias
(``"nu1"`` -> ``nad1``), a case-varying alias (``"coxi"`` -> ``cox1``), a
second alias family (``"cytb"`` -> ``cob``), and an unrecognized name
(``"unknown_gene"``, which ``normalize_gene_names`` leaves as its own
lower-cased self) - confirmed interactively against ``_GENE_ALIASES``.
 **Zero-out rulings (owner mandate 2026-08-14, final):** ``compute_gene_intersection``/``compute_gene_sets``/
 ``compute_syntency_score``are INTERNAL - unannotated genomes plus bare-set returns (not finite JSON); internals of the published compare_genes surface.

"""

from __future__ import annotations

import random
from dataclasses import dataclass

from organelleverse.capabilities.adapters import FixtureCase, FixtureFile
from organelleverse.operations.spec import ParameterCodec, ResultCodec, SideEffect

_GENE_NAME_INPUT = ["nu1", "coxi", "cytb", "unknown_gene"]

# --- compute_genome_identity fixtures ---------------------------------------
# A 2400 bp reference built from a fixed 60-mer with deterministic
# substitutions every 29 and 47 bp (both periods coprime with 60, so the
# composed sequence is effectively aperiodic and anchors unambiguously).
# Query A carries ~1% substitutions, a 17 bp deletion and a 10 bp insertion;
# query B is the reverse complement with sparser substitutions, exercising
# the strand -1 anchoring path. Both anchor as a single minimap2 block.
_IDENTITY_WORD = ("ATGGCGTACGTTAGCCGATCGTACGATCGGATCCGATGCTAGCTAGCATCGATCGTACGTAGCTAGCAT")[:60]
_NEXT_BASE = {"A": "C", "C": "G", "G": "T", "T": "A"}


def _identity_base() -> str:
    chars = list(_IDENTITY_WORD * 40)
    for position in range(29, len(chars), 29):
        chars[position] = "ACGT"[(position // 29) % 4]
    for position in range(47, len(chars), 47):
        chars[position] = "TGCA"[(position // 47) % 4]
    return "".join(chars)


def _identity_query_a(base: str) -> str:
    chars = list(base)
    for position in range(37, len(chars), 101):
        chars[position] = _NEXT_BASE[chars[position]]
    del chars[700:717]
    chars[1500:1500] = list("GGGTTCCCAA")
    return "".join(chars)


def _identity_query_b(base: str) -> str:
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    chars = [complement[base_char] for base_char in reversed(base)]
    for position in range(53, len(chars), 137):
        chars[position] = _NEXT_BASE[chars[position]]
    return "".join(chars)


_IDENTITY_BASE = _identity_base()
_IDENTITY_REFERENCE_FASTA = FixtureFile(
    relative_path="reference.fasta",
    content=f">ref\n{_IDENTITY_BASE}\n",
)
_IDENTITY_QUERY_A_FASTA = FixtureFile(
    relative_path="query_a.fasta",
    content=f">query_a\n{_identity_query_a(_IDENTITY_BASE)}\n",
)
_IDENTITY_QUERY_B_FASTA = FixtureFile(
    relative_path="query_b.fasta",
    content=f">query_b\n{_identity_query_b(_IDENTITY_BASE)}\n",
)


# Fixed-seed circular plastomes, including one SSC isomer.
def _orientation_fixture():
    from Bio.Seq import reverse_complement

    rng = random.Random(113)
    lsc, ir, ssc = ["".join(rng.choices("ACGT", k=n)) for n in (4000, 800, 1600)]
    lsc, ssc = "AAAAA" + lsc[5:-5] + "AAAAA", "AAAAA" + ssc[5:-5] + "AAAAA"
    base = lsc + ir + ssc + reverse_complement(ir)
    flipped = lsc + ir + reverse_complement(ssc) + reverse_complement(ir)
    return FixtureFile(
        relative_path="plastomes.fasta",
        content=f">reference\n{base}\n>same\n{base}\n>ssc_isomer\n{flipped}\n",
    )


_ORIENTATION_FASTA = _orientation_fixture()


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
    "comparative.normalize_plastome_orientation": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="input_fasta", codec=ParameterCodec.PATH),
            ParameterOverride(name="reference_fasta", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.READ_FILES,),
    ),
    "comparative.compute_genome_identity": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="reference_fasta", codec=ParameterCodec.PATH),
            ParameterOverride(name="query_fastas", codec=ParameterCodec.PATH),
            ParameterOverride(name="reference_genbank", codec=ParameterCodec.PATH),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="genome_identity",
        side_effects=(SideEffect.READ_FILES,),
    ),
    "comparative.normalize_gene_names": NamedParameterOverride(
        parameters=(ParameterOverride(name="names", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="gene_name_map",
        side_effects=(),
    ),
    "comparative.normalize_genes": NamedParameterOverride(
        parameters=(ParameterOverride(name="gene_names", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="gene_name_map",
        side_effects=(),
    ),
}
"""``comparative.compare_genes``, ``.compare_genomes``,
``.compute_gene_intersection``, ``.compute_gene_sets``,
``.compute_synteny_score``, ``.gene_table``, ``.synteny``,
``.write_gene_comparison``, ``.write_gene_table``,
``.write_genome_comparison``, and ``.write_synteny`` are deliberately
absent - see the module docstring for why each fails closed honestly.
"""

FIXTURES: dict[str, tuple[FixtureCase, ...]] = {
    "comparative.normalize_plastome_orientation": (
        FixtureCase(
            case="basic",
            files=(_ORIENTATION_FASTA,),
            parameters={"input_fasta": _ORIENTATION_FASTA.relative_path},
        ),
    ),
    "comparative.normalize_gene_names": (
        FixtureCase(case="basic", parameters={"names": list(_GENE_NAME_INPUT)}),
    ),
    "comparative.normalize_genes": (
        FixtureCase(case="basic", parameters={"gene_names": list(_GENE_NAME_INPUT)}),
    ),
    "comparative.compute_genome_identity": (
        FixtureCase(
            case="single_query",
            files=(_IDENTITY_REFERENCE_FASTA, _IDENTITY_QUERY_A_FASTA),
            parameters={
                "reference_fasta": _IDENTITY_REFERENCE_FASTA.relative_path,
                "query_fastas": [_IDENTITY_QUERY_A_FASTA.relative_path],
            },
        ),
        FixtureCase(
            case="basic",
            files=(
                _IDENTITY_REFERENCE_FASTA,
                _IDENTITY_QUERY_A_FASTA,
                _IDENTITY_QUERY_B_FASTA,
            ),
            parameters={
                "reference_fasta": _IDENTITY_REFERENCE_FASTA.relative_path,
                "query_fastas": [
                    _IDENTITY_QUERY_A_FASTA.relative_path,
                    _IDENTITY_QUERY_B_FASTA.relative_path,
                ],
            },
        ),
    ),
}

__all__ = ["FIXTURES", "OVERRIDES", "NamedParameterOverride", "ParameterOverride"]

# Structural calls retain both coordinate axes and expose the same mappy extra.
OVERRIDES["comparative.detect_structural_variants"] = NamedParameterOverride(
    parameters=(
        ParameterOverride(name="reference_fasta", codec=ParameterCodec.PATH),
        ParameterOverride(name="query_fasta", codec=ParameterCodec.PATH),
        ParameterOverride(name="preset", codec=ParameterCodec.JSON),
        ParameterOverride(name="min_size", codec=ParameterCodec.JSON),
        ParameterOverride(name="topology", codec=ParameterCodec.JSON),
        ParameterOverride(name="plastome", codec=ParameterCodec.JSON),
    ),
    result_codec=ResultCodec.CANONICAL,
    result_key=None,
    side_effects=(SideEffect.READ_FILES,),
)


def _structural_fixture():
    from Bio.Seq import reverse_complement
    rng = random.Random(177)
    seq = "".join(rng.choices("ACGT", k=8000))
    query = seq[:2500] + reverse_complement(seq[2500:4000]) + seq[4000:]
    return FixtureCase(
        case="inversion",
        files=(FixtureFile("reference.fa", f">ref\n{seq}\n"),
               FixtureFile("query.fa", f">query\n{query}\n")),
        parameters={"reference_fasta": "reference.fa", "query_fasta": "query.fa", "topology": "linear"},
    )


FIXTURES["comparative.detect_structural_variants"] = (_structural_fixture(),)
