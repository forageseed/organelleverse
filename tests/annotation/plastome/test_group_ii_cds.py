"""Two-exon group II introns must not be forced to nuclear AG acceptors."""

import pytest
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from organelleverse.annotation.plastome.cds_refine import refine_cds_features


@pytest.mark.parametrize("strand", [1, -1])
@pytest.mark.parametrize("acceptor", ["AC", "AT", "AG"])
def test_two_exon_refinement_recovers_protein_supported_join(strand, acceptor):
    e1 = "ATGGCTTGTGATGAATTTGGTCATATTAAACTGATGAATCCTCAACGTTCTACTGTTTGGATG"
    e2 = "GATGAAGCTTGTGGTCATATTAATCCTCAACGTTCTACTGTTTGGTATTTTAAACTGATGTAA"
    intron = "GTGCGT" + "TTATTCCATATCCCTTTATTCCTATCTTTA" + acceptor
    sequence = e1 + intron + e2
    exact = [(1, len(e1)), (len(e1) + len(intron) + 1, len(sequence))]
    approximate = [(1, len(e1) + 2), (exact[1][0] - 1, len(sequence))]
    if strand == -1:
        sequence = str(Seq(sequence).reverse_complement())
        exact = [(len(sequence) - b + 1, len(sequence) - a + 1) for a, b in exact]
        approximate = [(len(sequence) - b + 1, len(sequence) - a + 1) for a, b in approximate]
    feature = SeqFeature(
        CompoundLocation([FeatureLocation(a - 1, b, strand=strand) for a, b in approximate]),
        type="CDS",
        qualifiers={"gene": ["test"]},
    )
    protein = str(Seq(e1 + e2).translate(table=11)).rstrip("*")
    refine_cds_features(sequence, [feature], window=3, ref_proteins={"test": [protein]})
    actual = [(int(p.start) + 1, int(p.end)) for p in feature.location.parts]
    # AG has a one-base slide producing exactly the same spliced sequence;
    # neither protein nor RNA sequence can distinguish those coordinates.
    if acceptor in ("AC", "AT"):
        assert actual == exact
    assert str(feature.extract(Seq(sequence))) == e1 + e2
    assert str(feature.extract(Seq(sequence)).translate(table=11)).rstrip("*") == protein


@pytest.mark.parametrize("strand", [1, -1])
def test_motif_search_keeps_better_original_protein_alignment(strand):
    from organelleverse.annotation.plastome.splice_refine import refine_two_exon_cds

    e1 = "ATG" + "GCT" * 19
    e2 = "GAT" * 19 + "TAA"
    # The transferred splice starts with AA; a nearby GT donor adds one
    # lysine codon to the otherwise exact reference protein.
    intron = "AAAGT" + "C" * 21 + "AC"
    sequence = e1 + intron + e2
    original = [(1, len(e1)), (len(e1) + len(intron) + 1, len(sequence))]
    if strand == -1:
        sequence = str(Seq(sequence).reverse_complement())
        original = [(len(sequence) - b + 1, len(sequence) - a + 1) for a, b in original]
    protein = str(Seq(e1 + e2).translate(table=11)).rstrip("*")
    assert refine_two_exon_cds(sequence, original, protein, strand, window=4) == original


@pytest.mark.parametrize("strand", [1, -1])
@pytest.mark.parametrize("reverse_references", [False, True])
def test_multi_exon_refinement_uses_matching_reference_not_file_order(strand, reverse_references):
    e1 = "ATG" + "GCT" * 19
    e2 = "GAT" * 19 + "TAA"
    intron = "AAAGT" + "C" * 21 + "AC"
    sequence = e1 + intron + e2
    original = [(1, len(e1)), (len(e1) + len(intron) + 1, len(sequence))]
    if strand == -1:
        sequence = str(Seq(sequence).reverse_complement())
        original = [(len(sequence) - b + 1, len(sequence) - a + 1) for a, b in original]
    feature = SeqFeature(
        CompoundLocation([FeatureLocation(a - 1, b, strand=strand) for a, b in original]),
        type="CDS", qualifiers={"gene": ["test"]},
    )
    matching = str(Seq(e1 + e2).translate(table=11)).rstrip("*")
    # This reference contains an inserted lysine and favors the nearby GT
    # donor, despite matching the transferred gene less well overall.
    distant = str(Seq(e1 + "AAA" + e2).translate(table=11)).rstrip("*")
    references = [distant, matching]
    if reverse_references:
        references.reverse()
    refine_cds_features(sequence, [feature], window=4, ref_proteins={"test": references})
    assert [(int(p.start) + 1, int(p.end)) for p in feature.location.parts] == original


@pytest.mark.parametrize("strand", [1, -1])
@pytest.mark.parametrize("terminal_offset", [-2, -1, 1, 2])
def test_splice_and_terminal_frame_are_refined_jointly(strand, terminal_offset):
    e1 = "ATGGCTTGTGATGAATTTGGTCATATTAAACTGATGAATCCTCAACGTTCTACTGTTTGGATG"
    e2 = "GATGAAGCTTGTGGTCATATTAATCCTCAACGTTCTACTGTTTGGTATTTTAAACTGATGTAA"
    intron = "GTGCGT" + "TTATTCCATATCCCTTTATTCCTATCTTTA" + "AC"
    sequence = e1 + intron + e2 + "CCC"
    exact = [(1, len(e1)), (len(e1) + len(intron) + 1, len(sequence) - 3)]
    approximate = [(1, len(e1) + 2), (exact[1][0] - 1, exact[1][1] + terminal_offset)]
    if strand == -1:
        sequence = str(Seq(sequence).reverse_complement())
        exact = [(len(sequence) - b + 1, len(sequence) - a + 1) for a, b in exact]
        approximate = [(len(sequence) - b + 1, len(sequence) - a + 1) for a, b in approximate]
    feature = SeqFeature(
        CompoundLocation([FeatureLocation(a - 1, b, strand=strand) for a, b in approximate]),
        type="CDS", qualifiers={"gene": ["test"]},
    )
    protein = str(Seq(e1 + e2).translate(table=11)).rstrip("*")
    refine_cds_features(sequence, [feature], window=3, ref_proteins={"test": [protein]})
    assert [(int(p.start) + 1, int(p.end)) for p in feature.location.parts] == exact
    assert str(feature.extract(Seq(sequence))) == e1 + e2


@pytest.mark.parametrize("strand", [1, -1])
@pytest.mark.parametrize("terminal_codons", [90, 150])
def test_joint_refinement_extends_partial_last_exon_to_reference_extent(strand, terminal_codons):
    e1 = "ATG" + "GCT" * 19
    e2 = "GAT" * terminal_codons + "TAA"
    intron = "GT" + "C" * 28 + "AC"
    sequence = e1 + intron + e2
    exact = [(1, len(e1)), (len(e1) + len(intron) + 1, len(sequence))]
    # Only the first 15 codons of exon2 were transferred. The remaining
    # coding span exceeds the default 150nt terminal search window.
    transferred = [exact[0], (exact[1][0], exact[1][0] + 44)]
    if strand == -1:
        sequence = str(Seq(sequence).reverse_complement())
        exact = [(len(sequence) - b + 1, len(sequence) - a + 1) for a, b in exact]
        transferred = [(len(sequence) - b + 1, len(sequence) - a + 1) for a, b in transferred]
    feature = SeqFeature(
        CompoundLocation([FeatureLocation(a - 1, b, strand=strand) for a, b in transferred]),
        type="CDS", qualifiers={"gene": ["test"]},
    )
    protein = str(Seq(e1 + e2).translate(table=11)).rstrip("*")
    refine_cds_features(sequence, [feature], ref_proteins={"test": [protein]})
    assert [(int(p.start) + 1, int(p.end)) for p in feature.location.parts] == exact
    assert str(feature.extract(Seq(sequence))) == e1 + e2
