"""A CDS cut by the end of a molecule is kept as a partial gene."""

from __future__ import annotations

import pytest
from Bio.Seq import Seq

from organelleverse.annotation.mitochondrion.cds import partition_valid_cds
from organelleverse.annotation.mitochondrion.gff import _annotation_to_seq_features
from organelleverse.annotation.mitochondrion.models.gene import (
    ExonRecord,
    GeneAnnotation,
    Strand,
)
from organelleverse.annotation.mitochondrion.models.genome import ContigInfo, GenomeSequence

_FLANK = "ACGT" * 25  # 100 bases without an ORF to speak of
_BODY = "ATG" + "GCTAAA" * 30  # start codon + 60 codons, no stop


def _genome(sequence: str, contigs=None) -> GenomeSequence:
    return GenomeSequence(seqid="g", sequence=sequence, contig_map=contigs)


def _gene(start: int, end: int, strand: Strand, name: str = "atp1") -> GeneAnnotation:
    return GeneAnnotation(
        gene_name=name,
        exons=[ExonRecord(start=start, end=end, strand=strand)],
        strand=strand,
    )


def _revcomp(text: str) -> str:
    return str(Seq(text).reverse_complement())


def test_a_plus_gene_cut_at_the_3_end_becomes_a_partial_cds():
    cds = _BODY + "GC"  # runs off the end inside a codon
    genome = _genome(_FLANK + cds)
    gene = _gene(len(_FLANK) + 1, len(genome.sequence), Strand.PLUS)
    accepted, rejected = partition_valid_cds([gene], genome)
    assert not rejected
    (partial,) = accepted
    assert partial.is_partial_3prime and not partial.is_partial_5prime
    assert partial.total_exon_length % 3 == 0
    assert partial.exons[0].end == len(genome.sequence) - 2  # incomplete codon trimmed


def test_a_minus_gene_cut_at_the_molecule_start_becomes_a_partial_cds():
    # rps7-like: the gene lies on the minus strand and its stop codon is past position 1
    genome = _genome(_revcomp(_BODY) + _FLANK)
    gene = _gene(1, len(_BODY), Strand.MINUS)
    accepted, rejected = partition_valid_cds([gene], genome)
    assert not rejected
    (partial,) = accepted
    assert partial.is_partial_3prime and not partial.is_partial_5prime


def test_a_plus_gene_cut_at_the_5_end_becomes_a_partial_cds():
    cds = "GCTAAA" * 30 + "TAA"  # no start codon: the gene begins before the molecule
    genome = _genome(cds + _FLANK)
    gene = _gene(1, len(cds), Strand.PLUS)
    accepted, rejected = partition_valid_cds([gene], genome)
    assert not rejected
    (partial,) = accepted
    assert partial.is_partial_5prime and not partial.is_partial_3prime


def test_a_gene_short_of_its_stop_in_the_middle_of_the_sequence_stays_rejected():
    genome = _genome(_FLANK + _BODY + _FLANK)
    gene = _gene(len(_FLANK) + 1, len(_FLANK) + len(_BODY), Strand.PLUS)
    accepted, rejected = partition_valid_cds([gene], genome)
    assert not accepted and len(rejected) == 1


def test_an_internal_stop_is_not_excused_by_the_sequence_end():
    cds = "ATG" + "GCT" * 10 + "TAA" + "GCT" * 20  # stop codon inside, then the end
    genome = _genome(_FLANK + cds)
    gene = _gene(len(_FLANK) + 1, len(genome.sequence), Strand.PLUS)
    accepted, rejected = partition_valid_cds([gene], genome)
    assert not accepted and len(rejected) == 1


def test_a_fragment_too_short_to_be_a_gene_stays_rejected():
    cds = "ATG" + "GCT" * 8  # 27 bases, cut by the end
    genome = _genome(_FLANK + cds)
    gene = _gene(len(_FLANK) + 1, len(genome.sequence), Strand.PLUS)
    accepted, rejected = partition_valid_cds([gene], genome)
    assert not accepted and len(rejected) == 1


def test_a_complete_copy_of_the_gene_wins_over_the_cut_one():
    complete = "ATG" + "GCTAAA" * 30 + "TAA"
    genome = _genome(complete + _FLANK + _BODY)
    whole = _gene(1, len(complete), Strand.PLUS)
    cut = _gene(len(complete) + len(_FLANK) + 1, len(genome.sequence), Strand.PLUS)
    # the cut copy overlaps nothing, so both are kept: copies are real
    accepted, _ = partition_valid_cds([whole, cut], genome)
    assert len(accepted) == 2
    # a cut candidate sitting on the complete one is dropped
    twin = _gene(1, len(complete) - 3, Strand.PLUS)
    accepted, _ = partition_valid_cds([whole, twin], genome)
    assert accepted == [whole]


def test_the_end_of_an_inner_contig_of_a_merged_sequence_counts():
    first, second = _FLANK, _BODY
    merged = first + "N" * 200 + second
    contigs = [
        ContigInfo(original_id="a", start=1, end=len(first), length=len(first)),
        ContigInfo(
            original_id="b",
            start=len(first) + 201,
            end=len(merged),
            length=len(second),
        ),
    ]
    genome = _genome(merged, contigs)
    gene = _gene(len(first) + 201, len(merged), Strand.PLUS)
    accepted, rejected = partition_valid_cds([gene], genome)
    assert len(accepted) == 1 and not rejected


@pytest.mark.parametrize(
    ("strand", "text"),
    [(Strand.PLUS, "<"), (Strand.MINUS, ">")],
)
def test_a_5_prime_partial_is_written_with_an_open_end(strand, text):
    gene = _gene(1, 300, strand)
    gene = gene.model_copy(update={"is_partial_5prime": True})
    cds = next(f for f in _annotation_to_seq_features(gene) if f.type == "CDS")
    assert text in str(cds.location)
    assert "partial" in cds.qualifiers


@pytest.mark.parametrize(
    ("strand", "text"),
    [(Strand.PLUS, ">"), (Strand.MINUS, "<")],
)
def test_a_3_prime_partial_is_written_with_an_open_end(strand, text):
    gene = _gene(1, 300, strand)
    gene = gene.model_copy(update={"is_partial_3prime": True})
    cds = next(f for f in _annotation_to_seq_features(gene) if f.type == "CDS")
    assert text in str(cds.location)


def test_a_complete_gene_has_no_open_end():
    cds = next(f for f in _annotation_to_seq_features(_gene(1, 300, Strand.PLUS)) if f.type == "CDS")
    assert "<" not in str(cds.location) and ">" not in str(cds.location)
