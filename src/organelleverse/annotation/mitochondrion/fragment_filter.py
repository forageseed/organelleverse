"""Drop short CDS that are redundant copies or remnants rather than genes.

Plant mitochondrial genomes carry many short copies of genes: pieces moved by
recombination across repeats, promiscuous plastid-derived stretches and
pseudogene remnants. Many keep an open reading frame, so they pass the CDS
checks and are published as extra CDS. On the RNA-truth cohort (38 genomes) a
fifth of OV's CDS that matched no truth gene were shorter than 60% of the
gene's reference protein.

A CDS shorter than ``SHORT_FRACTION`` of the *short end* (10th percentile) of the
packaged reference proteins of its gene is dropped when either holds. The short end,
not the median, because the references lean on angiosperms: a lineage whose gene is
genuinely shorter (rps2, rps4 in bryophytes and gymnosperms) still has references of
that length and must not be judged against the angiosperm median:

* the gene, or another member of its split-gene family (``ccmFC`` with
  ``ccmFC1``/``ccmFC2``), also has a copy of at least ``FULL_FRACTION`` of
  that length that shares no exon base with it: the short CDS is a redundant
  piece of a gene that is annotated in full; or
* the gene has no exon reference (it is a single-exon gene) and the CDS is one
  exon: a single-exon protein under 60% of its length is a pseudogene remnant.

What stays: pieces of genes that are spliced from several exons (the assembly
may not have joined a trans-spliced segment), a short CDS that is the only
copy of a split-gene family member's gene, and any CDS cut by a sequence end.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from .db import DBManager
from .models.gene import GeneAnnotation

#: A CDS below this fraction of the reference protein length counts as a fragment.
SHORT_FRACTION = 0.6
#: A copy at or above this fraction counts as the full-length gene.
FULL_FRACTION = 0.8

#: Genes the references store as one full gene plus separate fragment entries; a full
#: copy of any member makes a short copy of another member redundant. These are the
#: groups ``pcg`` already treats as related.
SPLIT_GENE_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("ccmfc", "ccmfc1", "ccmfc2"),
    ("ccmfn", "ccmfn1", "ccmfn2"),
)
_FAMILY = {member: group[0] for group in SPLIT_GENE_FAMILIES for member in group}


def family_of(gene_name: str) -> str:
    """Lower-cased family key of a gene: its split-gene family, else the gene itself."""

    name = gene_name.lower()
    return _FAMILY.get(name, name)


def _fasta_lengths(path: Path) -> list[int]:
    lengths, current = [], 0
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if current:
                    lengths.append(current)
                current = 0
            else:
                current += len(line.strip().rstrip("*"))
    if current:
        lengths.append(current)
    return lengths


#: Quantile of the reference protein lengths that marks the short end of a gene.
FLOOR_QUANTILE = 0.10


@lru_cache(maxsize=8)
def _reference_lengths(ref_dir: str) -> dict[str, tuple[float, float]]:
    """``gene -> (median, short end)`` of the reference protein lengths."""

    out: dict[str, tuple[float, float]] = {}
    for path in sorted(Path(ref_dir).glob("*.Protein.fasta")):
        lengths = sorted(_fasta_lengths(path))
        if lengths:
            floor = lengths[round(FLOOR_QUANTILE * (len(lengths) - 1))]
            out[path.name.split(".")[0].lower()] = (float(statistics.median(lengths)), float(floor))
    return out


def reference_protein_lengths(db_manager: DBManager) -> dict[str, float]:
    """Median reference protein length (amino acids) per lower-cased gene name."""

    return {
        gene: median
        for gene, (median, _) in _reference_lengths(str(db_manager.blast_ref_dir)).items()
    }


def reference_length_floors(db_manager: DBManager) -> dict[str, float]:
    """Short-end (10th percentile) reference protein length per lower-cased gene name."""

    return {
        gene: floor
        for gene, (_, floor) in _reference_lengths(str(db_manager.blast_ref_dir)).items()
    }


def multi_exon_genes(db_manager: DBManager) -> frozenset[str]:
    """Lower-cased genes that have an exon reference: the spliced genes."""

    return frozenset(
        path.name.split(".")[0].lower() for path in Path(db_manager.exon_ref_dir).glob("*.fasta")
    )


def protein_length(annotation: GeneAnnotation) -> float:
    """Amino acids encoded by the CDS, terminal stop excluded."""

    return annotation.total_exon_length / 3 - 1


def _share_bases(a: GeneAnnotation, b: GeneAnnotation) -> bool:
    return any(min(x.end, y.end) >= max(x.start, y.start) for x in a.exons for y in b.exons)


def drop_duplicate_fragments(
    annotations: Iterable[GeneAnnotation],
    ref_lengths: dict[str, float],
    spliced_genes: frozenset[str] | None = None,
    floors: dict[str, float] | None = None,
) -> tuple[list[GeneAnnotation], list[GeneAnnotation]]:
    """Return (kept, dropped): short CDS that are redundant pieces or single-exon remnants.

    ``spliced_genes`` names the genes that have an exon reference; without it only the
    redundant-piece rule applies. ``floors`` gives the short end of each gene's reference
    lengths, the yardstick for "short"; without it the median is used.
    """

    items = list(annotations)
    full: dict[str, list[GeneAnnotation]] = {}
    for a in items:
        ref = ref_lengths.get(a.gene_name.lower())
        if a.gene_type == "CDS" and ref and protein_length(a) >= FULL_FRACTION * ref:
            full.setdefault(family_of(a.gene_name), []).append(a)
    kept, dropped = [], []
    for a in items:
        ref = ref_lengths.get(a.gene_name.lower())
        is_fragment = (
            a.gene_type == "CDS"
            and ref is not None
            and not (a.is_partial_5prime or a.is_partial_3prime)
            and protein_length(a) < SHORT_FRACTION * (floors or {}).get(a.gene_name.lower(), ref)
            and (
                any(not _share_bases(a, f) for f in full.get(family_of(a.gene_name), []))
                or (
                    spliced_genes is not None
                    and len(a.exons) == 1
                    and a.gene_name.lower() not in spliced_genes
                    and family_of(a.gene_name) not in spliced_genes
                )
            )
        )
        (dropped if is_fragment else kept).append(a)
    return kept, dropped
