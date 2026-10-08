"""Regression tests for merging exon hits separated by sub-intron gaps.

cox1 is configured as a two-exon gene because many angiosperms carry a group I
intron in it. In genomes without the intron the two reference-exon hits land
next to each other; they must become one exon on either strand instead of a
fabricated 0-12 bp "intron" (seen in Arabidopsis NC_037304 and Medicago
NC_029641 before the fix).
"""

from __future__ import annotations

from organelleverse.annotation.mitochondrion.boundary import (
    SHORT_INTRON_THRESHOLD,
    _remove_short_introns,
)
from organelleverse.annotation.mitochondrion.models.gene import (
    ExonRecord,
    GeneAnnotation,
    Strand,
)
from organelleverse.annotation.mitochondrion.models.genome import GenomeSequence

GENOME = GenomeSequence(seqid="g", sequence="A" * 400_000)


def _gene(name: str, strand: Strand, exons: list[tuple[int, int]]) -> GeneAnnotation:
    return GeneAnnotation(
        gene_name=name,
        gene_type="CDS",
        product="",
        strand=strand,
        exons=[
            ExonRecord(start=start, end=end, strand=strand, number=i)
            for i, (start, end) in enumerate(exons, 1)
        ],
    )


def _coords(ann: GeneAnnotation) -> list[tuple[int, int]]:
    return [(exon.start, exon.end) for exon in ann.exons]


def test_minus_strand_short_gap_is_merged() -> None:
    # Minus-strand exons are stored in transcription order (high -> low).
    ann = _gene("cox1", Strand.MINUS, [(115632, 116345), (114762, 115619)])

    merged = _remove_short_introns(ann, GENOME)

    assert _coords(merged) == [(114762, 116345)]
    assert merged.exons[0].number == 1
    assert merged.exons[0].phase == 0


def test_abutting_exons_are_merged() -> None:
    ann = _gene("cox1", Strand.PLUS, [(179198, 179923), (179924, 180781)])

    merged = _remove_short_introns(ann, GENOME)

    assert _coords(merged) == [(179198, 180781)]


def test_overlapping_exon_hits_are_merged_to_their_union() -> None:
    ann = _gene("cox1", Strand.PLUS, [(1000, 1800), (1790, 2600)])

    merged = _remove_short_introns(ann, GENOME)

    assert _coords(merged) == [(1000, 2600)]


def test_real_intron_is_kept_and_order_preserved() -> None:
    gap = SHORT_INTRON_THRESHOLD + 800
    exons = [(10_000 + gap + 700, 10_000 + gap + 1400), (10_000, 10_700)]
    ann = _gene("cox2", Strand.MINUS, exons)

    merged = _remove_short_introns(ann, GENOME)

    assert merged is ann


def test_three_exons_merge_only_the_short_gap_and_recompute_phase() -> None:
    ann = _gene("nad4", Strand.PLUS, [(100, 199), (205, 400), (2000, 2300)])

    merged = _remove_short_introns(ann, GENOME)

    assert _coords(merged) == [(100, 400), (2000, 2300)]
    assert [exon.number for exon in merged.exons] == [1, 2]
    assert [exon.phase for exon in merged.exons] == [0, 301 % 3]


def test_scattered_trans_spliced_gene_is_left_alone() -> None:
    ann = _gene("nad1", Strand.PLUS, [(300_000, 300_100), (5_000, 5_200), (5_210, 5_400)])

    assert _remove_short_introns(ann, GENOME) is ann
