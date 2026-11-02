import pytest
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from organelleverse.annotation.plastome.exon_sequence_evidence import (
    coding_exon_domains,
    compare_exon_sequences,
)


def feature(lengths, strand=1, codon_start=1):
    parts = [FeatureLocation(100 * i, 100 * i + n, strand=strand) for i, n in enumerate(lengths)]
    if strand == -1:
        parts.reverse()
    location = CompoundLocation(parts) if len(parts) > 1 else parts[0]
    return SeqFeature(location, type="CDS", qualifiers={"codon_start": [str(codon_start)]})


def test_large_local_match_does_not_certify_first_exon():
    tail = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQV"
    source, target = "WWWWWWWWWW" + tail, "PPPPPPPPPP" + tail
    f = feature([30, 3 * len(tail) + 3])
    e = compare_exon_sequences("reference:1", f, source, "candidate:1", f, target)
    assert e.status == "unique_alignment"
    assert len(e.residue_pairs) >= len(tail)
    assert e.uncovered_source_residues[0] == tuple(range(10))
    assert e.uncovered_target_residues[0] == tuple(range(10))
    assert e.exon_pair_residue_counts[0] == (0, 0)
    assert e.source_domains[1].coding_bases_outside_translation == 3


@pytest.mark.parametrize("strand", [1, -1])
def test_biological_order_split_codon_and_codon_start(strand):
    f = feature([5, 7] if strand == 1 else [7, 5], strand, 2)
    domains = coding_exon_domains(f, "ACD")
    assert [d.coding_interval for d in domains] == [(-1, 4), (4, 11)]
    assert [d.protein_residues for d in domains] == [(0, 1), (1, 2)]
    assert [d.coding_bases_outside_translation for d in domains] == [0, 2]


def test_split_codon_corresponds_to_both_exons():
    f = feature([4, 8])
    e = compare_exon_sequences("source", f, "ACD", "IR:a", f, "ACD")
    assert e.uncovered_source_residues == ((), ())
    assert e.exon_pair_residue_counts == ((2, 1), (1, 2))
    other = compare_exon_sequences("source", f, "ACD", "IR:b", f, "ACD")
    assert e.target_id != other.target_id


def test_ambiguous_alignment_does_not_report_zero_loss():
    e = compare_exon_sequences("source", feature([9]), "AAA", "target", feature([12]), "AAAA")
    assert e.status == "ambiguous_alignment"
    assert e.source_domains and e.target_domains
    assert e.uncovered_source_residues == () and e.exon_pair_residue_counts == ()


def test_no_alignment_preserves_domains():
    e = compare_exon_sequences("source", feature([9]), "WWW", "target", feature([9]), "PPP")
    assert e.status == "no_alignment" and e.source_domains and e.target_domains


def test_invalid_phase_and_missing_translation_are_outside_domain():
    with pytest.raises(ValueError):
        coding_exon_domains(feature([9], codon_start=4), "ACD")
    with pytest.raises(ValueError):
        compare_exon_sequences("s", feature([9]), "", "t", feature([9]), "ACD")


def test_supplied_translation_outside_coding_dna_is_not_silent():
    f = feature([3])
    e = compare_exon_sequences("source", f, "ACD", "target", feature([9]), "ACD")
    assert e.source_residues_outside_coding_dna == (1, 2)
    assert e.target_residues_outside_coding_dna == ()
