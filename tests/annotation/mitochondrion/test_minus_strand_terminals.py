"""Start/stop corrections on minus-strand genes stored in transcription order.

The minus-strand branches take ``exons[0]`` as the 3' exon. Genes built by
BLAST list minus-strand exons high to low, so the stop search ran from exon 1's
donor and moved that splice site (rpl2 exon 1 lost 21 nt in Arabidopsis
NC_037304 and in 12 of 13 PMGA Table S2 species).
"""

from __future__ import annotations

from organelleverse.annotation.mitochondrion import boundary
from organelleverse.annotation.mitochondrion.boundary import (
    _correct_stop_codon_conservative,
    _in_genomic_order,
)
from organelleverse.annotation.mitochondrion.models.gene import ExonRecord, GeneAnnotation, Strand
from organelleverse.annotation.mitochondrion.models.genome import GenomeSequence


class _DB:
    @staticmethod
    def is_stop_gain_gene(name: str) -> bool:
        return False


def _gene(exons: list[tuple[int, int]]) -> GeneAnnotation:
    return GeneAnnotation(
        gene_name="rpl2",
        strand=Strand.MINUS,
        source_method="BLAST",
        exons=[ExonRecord(start=s, end=e, strand=Strand.MINUS, number=i) for i, (s, e) in enumerate(exons, 1)],
    )


def _genome() -> GenomeSequence:
    seq = ["C"] * 2000
    seq[100:103] = "TTA"  # 3' exon 101..400 ends in a stop (TAA on the minus strand)
    seq[991:994] = "TTA"  # in-frame stop 9 nt inside the intron next to exon 1's donor
    return GenomeSequence(seqid="g", sequence="".join(seq))


def test_stop_search_uses_the_three_prime_exon():
    ann = _gene([(1001, 1300), (101, 400)])  # transcription order
    out = _in_genomic_order(_correct_stop_codon_conservative, ann, _genome(), _DB(), 30)
    assert [(e.start, e.end) for e in out.exons] == [(1001, 1300), (101, 400)]


def test_minus_strand_stop_search_counts_codons_from_the_start():
    # 299 + 300 nt: the 3' end is 1 nt into a codon; the in-frame stop (TAA) sits 4 nt further on
    seq = ["C"] * 2000
    seq[97:100] = "TTA"
    ann = _gene([(1001, 1300), (102, 400)])
    out = _in_genomic_order(_correct_stop_codon_conservative, ann, GenomeSequence(seqid="g", sequence="".join(seq)), _DB(), 30)
    assert [(e.start, e.end) for e in out.exons] == [(1001, 1300), (98, 400)]


def test_phase_step_leaves_an_open_reading_frame_alone():
    ann = GeneAnnotation(
        gene_name="rps3",
        strand=Strand.MINUS,
        exons=[
            ExonRecord(start=1001, end=1074, strand=Strand.MINUS, number=1, phase=0),
            ExonRecord(start=101, end=400, strand=Strand.MINUS, number=2, phase=0),  # stale: 74 % 3 = 2
        ],
    )
    out = boundary._restore_phase_continuity(ann, GenomeSequence(seqid="g", sequence="C" * 2000))
    assert [(e.start, e.end) for e in out.exons] == [(1001, 1074), (101, 400)]


def test_order_restored_and_exons_renumbered():
    ann = _gene([(1001, 1300), (101, 400)])
    seen = []

    def step(a, *args):
        seen.append([(e.start, e.end) for e in a.exons])
        lo, hi = a.exons
        return a.model_copy(update={"exons": [lo.model_copy(update={"start": 95}), hi]})

    out = _in_genomic_order(step, ann)
    assert seen == [[(101, 400), (1001, 1300)]]
    assert [(e.start, e.end, e.number) for e in out.exons] == [(1001, 1300, 1), (95, 400, 2)]


def test_plus_strand_and_unchanged_genes_pass_through():
    ann = _gene([(1001, 1300), (101, 400)])
    assert _in_genomic_order(lambda a, *args: a, ann) is ann
    plus = ann.model_copy(update={"strand": Strand.PLUS})
    assert _in_genomic_order(lambda a, *args: "called", plus) == "called"


def test_db_refined_exons_carry_phases_of_their_new_lengths(monkeypatch):
    ann = GeneAnnotation(
        gene_name="ccmFC",
        strand=Strand.PLUS,
        exons=[
            ExonRecord(start=189483, end=190267, strand=Strand.PLUS, number=1, phase=0),
            ExonRecord(start=191201, end=191767, strand=Strand.PLUS, number=2, phase=0),
        ],
    )
    # the DB pins the Arabidopsis junction: exon 1 779 nt, so exon 2 starts in phase 779 % 3 = 2
    monkeypatch.setattr(boundary, "refine_exon_boundaries", lambda *a, **k: [(189483, 190261), (191221, 191767)])
    out = boundary._refine_exon_intron_boundaries(ann, GenomeSequence(seqid="g", sequence="A" * 200_000))
    assert [(e.start, e.end, e.phase) for e in out.exons] == [(189483, 190261, 0), (191221, 191767, 2)]
    # and the phase step no longer moves the pinned donor
    assert boundary._restore_phase_continuity(out, GenomeSequence(seqid="g", sequence="A" * 200_000)).exons[0].end == 190261


def test_boundary_db_margin_covers_a_23_nt_acceptor_shift():
    assert boundary.refine_exon_boundaries.__module__.endswith("boundary_db")
    from organelleverse.annotation.mitochondrion import boundary_db

    assert boundary_db._DEFAULT_MARGIN >= 23
