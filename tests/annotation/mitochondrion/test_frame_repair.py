"""Tests for PMGA-faithful reading-frame repair (pmga_frame_repair).

The function translates PMGA 03.CDSCheck.py: internal_edit (±1 base shifts at
exon 5' ends), check_reading2 (side trims), start_edit (3-step start slide)
and stop_edit (frame-aligning 3-step stop slide). Every test also pins the
convergent property: a gene the repair cannot fix is returned untouched, and
a gene that is already valid is never modified.
"""

from __future__ import annotations

import pytest

from organelleverse.annotation.mitochondrion.boundary import pmga_frame_repair
from organelleverse.annotation.mitochondrion.cds import _extract_cds_sequence
from organelleverse.annotation.mitochondrion.db import DBManager
from organelleverse.annotation.mitochondrion.models.gene import (
    ExonRecord,
    GeneAnnotation,
    Strand,
)
from organelleverse.annotation.mitochondrion.models.genome import GenomeSequence


def _rc(seq: str) -> str:
    return seq.translate(str.maketrans("ATGC", "TACG"))[::-1]


@pytest.fixture()
def db() -> DBManager:
    return DBManager()


def _gene(name: str, exons: list[tuple[int, int, Strand]]) -> GeneAnnotation:
    return GeneAnnotation(
        gene_name=name,
        gene_type="CDS",
        product="",
        strand=exons[0][2],
        exons=[
            ExonRecord(start=start, end=end, strand=strand)
            for start, end, strand in exons
        ],
    )


def test_valid_gene_passes_through_untouched(db: DBManager) -> None:
    genome = GenomeSequence(seqid="g", sequence="A" * 40 + "ATG" + "GGG" * 8 + "TAA")
    ann = _gene("atp9", [(41, 73, Strand.PLUS)])
    assert _extract_cds_sequence(ann, genome) == "ATG" + "GGG" * 8 + "TAA"

    repaired = pmga_frame_repair(ann, genome, db)

    assert repaired is ann


def test_single_exon_start_slide_accepts_atg(db: DBManager) -> None:
    # 5' boundary off by one codon: the first upstream in-frame ATG is 3 bp away.
    genome = GenomeSequence(
        seqid="g", sequence="A" * 20 + "ATG" + "GGG" * 6 + "TAA" + "C" * 20
    )
    ann = _gene("atp9", [(24, 44, Strand.PLUS)])  # starts at GGG
    assert _extract_cds_sequence(ann, genome)[:3] == "GGG"

    repaired = pmga_frame_repair(ann, genome, db)

    assert repaired is not ann
    final = _extract_cds_sequence(repaired, genome)
    assert final.startswith("ATG")
    assert final.endswith("TAA")


def test_minus_strand_start_slide_moves_high_coordinate(db: DBManager) -> None:
    # For a minus-strand gene the transcript 5' end is the high coordinate.
    transcript = "ATG" + "GGG" * 4 + "TAA"
    genome = GenomeSequence(seqid="g", sequence="A" * 10 + _rc(transcript) + "C" * 10)
    ann = _gene("atp9", [(11, 25, Strand.MINUS)])  # 5' end one codon short
    assert _extract_cds_sequence(ann, genome)[:3] != "ATG"

    repaired = pmga_frame_repair(ann, genome, db)

    assert repaired is not ann
    final = _extract_cds_sequence(repaired, genome)
    assert final == transcript


def test_stop_slide_aligns_frame_and_finds_stop(db: DBManager) -> None:
    # CDS ends one base short of a frame-valid stop: len%3==2, so the slide's
    # first step is 1 and lands exactly on the TAA.
    genome = GenomeSequence(
        seqid="g", sequence="A" * 20 + "ATG" + "GGG" * 6 + "TAA" + "C" * 20
    )
    ann = _gene("atp9", [(21, 43, Strand.PLUS)])  # 23 bp, ends at TA
    assert len(_extract_cds_sequence(ann, genome)) % 3 == 2

    repaired = pmga_frame_repair(ann, genome, db)

    assert repaired is not ann
    final = _extract_cds_sequence(repaired, genome)
    assert len(final) % 3 == 0
    assert final.endswith("TAA")


def test_stop_gain_accepted_for_listed_gene_with_exception(db: DBManager) -> None:
    # ccmFC is a packaged stop-gain gene: CAA at the 3' end is accepted and
    # marked with an RNA editing exception.
    assert db.is_stop_gain_gene("ccmFC")
    genome = GenomeSequence(
        seqid="g", sequence="A" * 20 + "ATG" + "GGG" * 6 + "CAA"
    )
    ann = _gene("ccmFC", [(21, 44, Strand.PLUS)])

    repaired = pmga_frame_repair(ann, genome, db)

    assert repaired is not ann
    assert _extract_cds_sequence(repaired, genome).endswith("CAA")
    assert "RNA editing" in repaired.exceptions


def test_stop_gain_not_accepted_for_unlisted_gene(db: DBManager) -> None:
    # cox1 is not a stop-gain gene: the same CAA terminal is left alone and
    # no slide finds a real stop, so the annotation is returned untouched.
    assert not db.is_stop_gain_gene("cox1")
    genome = GenomeSequence(
        seqid="g", sequence="A" * 20 + "ATG" + "GGG" * 6 + "CAA"
    )
    ann = _gene("cox1", [(21, 44, Strand.PLUS)])

    repaired = pmga_frame_repair(ann, genome, db)

    assert repaired is ann


def test_start_gain_accepts_acg_with_editing_exception(db: DBManager) -> None:
    # atp1 is a packaged start-gain gene (Ginkgo reference documents ACG).
    assert db.is_start_gain_gene("atp1")
    genome = GenomeSequence(
        seqid="g", sequence="A" * 20 + "ACG" + "GGG" * 6 + "TAA"
    )
    ann = _gene("atp1", [(24, 44, Strand.PLUS)])  # starts at GGG

    repaired = pmga_frame_repair(ann, genome, db)

    assert repaired is not ann
    final = _extract_cds_sequence(repaired, genome)
    assert final.startswith("ACG")
    assert "RNA editing" in repaired.exceptions


def test_multiexon_internal_edit_shifts_exon_boundary(db: DBManager) -> None:
    # nad4-like case: a 2-exon gene whose concatenation is frame-shifted and
    # carries an aligned internal stop. internal_edit tries the 5' extension
    # first; this construction makes the extension introduce a NEW stop (so it
    # is rejected) and the 1-base trim realigns the frame onto a clean CDS
    # ending in TAA.
    exon1 = "ATG" + "GGG" * 4                          # 15 bp, frame 0
    exon2 = "TCCTAGACTAAATTAA"                         # 16 bp
    genome = GenomeSequence(seqid="g", sequence="A" * 10 + exon1 + exon2 + "C" * 10)
    ann = _gene("nad1", [(11, 25, Strand.PLUS), (26, 41, Strand.PLUS)])
    seq = _extract_cds_sequence(ann, genome)
    assert len(seq) % 3 == 1
    from organelleverse.annotation.mitochondrion.pcg import translate_sequence

    assert "*" in translate_sequence(seq)[:-1]

    repaired = pmga_frame_repair(ann, genome, db)

    final = _extract_cds_sequence(repaired, genome)
    assert len(final) % 3 == 0
    assert final.startswith("ATG")
    assert final.endswith("TAA")
    assert "*" not in translate_sequence(final)[:-1]


def test_unfixable_gene_returned_untouched(db: DBManager) -> None:
    # No ATG/ACG/GTG upstream, no stop downstream: every pass fails and the
    # original annotation is returned unchanged.
    genome = GenomeSequence(seqid="g", sequence="TTT" * 100)
    ann = _gene("atp9", [(40, 60, Strand.PLUS)])

    repaired = pmga_frame_repair(ann, genome, db)

    assert repaired is ann


def test_pseudo_gene_skipped(db: DBManager) -> None:
    genome = GenomeSequence(seqid="g", sequence="A" * 20 + "ATG" + "GGG" * 6 + "TAA")
    ann = _gene("atp9", [(24, 44, Strand.PLUS)]).model_copy(update={"is_pseudo": True})

    repaired = pmga_frame_repair(ann, genome, db)

    assert repaired is ann
