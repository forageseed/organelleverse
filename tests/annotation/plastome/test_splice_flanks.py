"""Splice placements that differ only at an RNA editing site are resolved by reference join flanks."""

from __future__ import annotations

from types import SimpleNamespace

from Bio.Seq import Seq

from organelleverse.annotation.plastome.splice_flanks import (
    choose_candidate,
    collapse_edits,
    flank_distance,
    reference_junction_flanks,
)
from organelleverse.annotation.plastome.splice_refine import (
    refine_internal_boundaries,
    refine_two_exon_cds,
)

BODY1 = "GCTAAAGAAGGTCTGGCAAAAGTTGAAGGCGAAATCGCA"  # 13 codons, no stop
BODY2 = "GGTAAAGCTGAAGGCGTTCTGCGTAAAGAAGCAGGTTTCGAATAA"
LEFT = "TTTTTTTTTT"
INTRON = "GTGCGATTGAAATTCAATCCAAAAAAAGGGGTTTCTAT"  # group II-like: GTGCG ... AT


def _two_exon_genome():
    """Exon 1 ends ...TC, exon 2 starts A...: the junction codon is TC|A (Ser).

    Placing the intron one base upstream reads the intron's last T instead of
    exon 1's last C: codon TTA (Leu), which is what C-to-U editing makes of TCA.
    """
    exon1 = "ATG" + BODY1 + "TC"
    exon2 = "A" + BODY2
    genome = LEFT + exon1 + INTRON + exon2 + LEFT
    s1 = len(LEFT) + 1
    e1 = s1 + len(exon1) - 1
    s2 = e1 + len(INTRON) + 1
    e2 = s2 + len(exon2) - 1
    right = [(s1, e1), (s2, e2)]
    wrong = [(s1, e1 - 1), (s2 - 1, e2)]
    edited = str(Seq(exon1[:-1] + "T" + exon2).translate(table=11)).rstrip("*")
    return genome, right, wrong, edited, exon1, exon2


def _reference(exons: list[str], gene: str = "ndhA"):
    feats = [SimpleNamespace(feature_type="CDS_exon", gene=gene, feature_id=f"ref:7:exon{i}", reference_name="ref",
                             sequence=seq, exon_index=i, exon_count=len(exons))
             for i, seq in enumerate(exons, start=1)]
    feats.append(SimpleNamespace(feature_type="CDS", gene=gene, feature_id="ref:7", reference_name="ref",
                                 sequence="".join(exons), exon_index=1, exon_count=len(exons)))
    return SimpleNamespace(features=tuple(feats))


def test_reference_flanks_from_exon_features():
    flanks = reference_junction_flanks([_reference(["AAAAACCCCCGGGGG", "TTTTTAAAAACCCCC"])])
    assert flanks == {"ndhA": [[("CCCCCGGGGG", "TTTTTAAAAA")]]}


def test_edit_equivalence_ignores_c_t_only():
    assert collapse_edits("ACGTT") == collapse_edits("ACGCT")
    assert collapse_edits("ACGTT") != collapse_edits("ACGAT")


def test_flank_distance_uses_same_join_count_only():
    refs = [[("AAAC", "TTTT")], [("AAAC", "TTTT"), ("GG", "CC")]]
    assert flank_distance("AAACTTTT", [4], refs, flank=4) == 0
    assert flank_distance("AAATTTTT", [3], refs, flank=4) > 0
    assert flank_distance("AAACTTTT", [2, 4], [refs[0]], flank=4) is None


def test_two_exon_edit_site_at_join_follows_reference():
    genome, right, wrong, edited, exon1, exon2 = _two_exon_genome()
    # Protein alone keeps the transferred one-base-off model: the edited
    # reference protein matches its genomic T at least as well.
    assert refine_two_exon_cds(genome, wrong, edited, 1, window=3) == wrong
    flanks = reference_junction_flanks([_reference([exon1, exon2])])["ndhA"]
    assert refine_two_exon_cds(genome, wrong, edited, 1, window=3, junction_flanks=flanks) == right
    # Already right: stays right.
    assert refine_two_exon_cds(genome, right, edited, 1, window=3, junction_flanks=flanks) == right


def test_two_exon_minus_strand():
    genome, right, wrong, edited, exon1, exon2 = _two_exon_genome()
    L = len(genome)
    rc = str(Seq(genome).reverse_complement())
    flip = [(L - e + 1, L - s + 1) for s, e in wrong]
    want = [(L - e + 1, L - s + 1) for s, e in right]
    flanks = reference_junction_flanks([_reference([exon1, exon2])])["ndhA"]
    assert refine_two_exon_cds(rc, flip, edited, -1, window=3, junction_flanks=flanks) == want


def test_protein_still_decides_non_equivalent_candidates():
    # A reference whose flanks disagree cannot override a clearly better protein.
    cands = [(50.0, [(1, 9), (20, 29)], "ATGAAACCC" + "GGGTTTTAA", [9]),
             (10.0, [(1, 8), (19, 29)], "ATGAAACCA" + "GGGGTTTTAA", [8])]
    refs = [[("ATGAAACC", "AGGGGTTTTA")]]
    assert choose_candidate(cands, refs) == [(1, 9), (20, 29)]


def test_multi_exon_internal_search_follows_reference():
    genome, right, wrong, edited, exon1, exon2 = _two_exon_genome()
    # Append a third exon after a second intron; its join is unambiguous.
    exon3 = "GCTGAAGGTAAAGCAGGTCTGTAA"
    exon2_core = exon2[:-3]  # drop the stop: exon 3 carries it
    genome = LEFT + exon1 + INTRON + exon2_core + INTRON + exon3 + LEFT
    s1 = len(LEFT) + 1
    e1 = s1 + len(exon1) - 1
    s2 = e1 + len(INTRON) + 1
    e2 = s2 + len(exon2_core) - 1
    s3 = e2 + len(INTRON) + 1
    e3 = s3 + len(exon3) - 1
    right = [(s1, e1), (s2, e2), (s3, e3)]
    wrong = [(s1, e1 - 1), (s2 - 1, e2), (s3, e3)]
    edited = str(Seq(exon1[:-1] + "T" + exon2_core + exon3).translate(table=11)).rstrip("*")
    flanks = reference_junction_flanks([_reference([exon1, exon2_core, exon3], gene="clpP")])["clpP"]
    got = refine_internal_boundaries(genome, wrong, edited, 1, window=2, junction_flanks=flanks)
    assert got == right


def test_close_protein_scores_follow_reference_votes_only_with_tolerance():
    # Candidate 1 scores 2% below the protein-best but matches the reference join exactly.
    cands = [(100.0, [(1, 9), (20, 29)], "ATGAAACCC" + "GGGTTTTAA", [9]),
             (98.0, [(1, 12), (23, 29)], "ATGAAACCCAAA" + "TTTTAA", [12])]
    refs = [[("TGAAACCCAAA"[-10:], "TTTTAA")]] * 3  # three references agree
    assert choose_candidate(cands, refs, tolerance=0.0) == [(1, 9), (20, 29)]
    assert choose_candidate(cands, refs) == [(1, 12), (23, 29)]  # default PROTEIN_TOLERANCE = 0.06
    assert choose_candidate(cands, refs, tolerance=0.05) == [(1, 12), (23, 29)]
    assert choose_candidate(cands, refs, tolerance=0.01) == [(1, 9), (20, 29)]


def test_conflicting_reference_conventions_fall_back_to_group_ii_boundary():
    # One reference annotates the join like the intron's GTGCG..AT boundary, one a base upstream:
    # both flank sets match a candidate exactly, so the intron's own group II motif decides.
    genome, right, wrong, edited, exon1, exon2 = _two_exon_genome()
    refs = [_reference([exon1, exon2]), _reference([exon1[:-1], "T" + exon2])]
    for i, ref in enumerate(refs):  # distinct feature ids per reference
        for f in ref.features:
            f.reference_name = f"ref{i}"
    flanks = reference_junction_flanks(refs)["ndhA"]
    assert len(flanks) == 2
    assert refine_two_exon_cds(genome, wrong, edited, 1, window=3, junction_flanks=flanks) == right


def test_trans_spliced_cis_block_follows_reference():
    # rps12-like: exon 1 far away on the other strand; exons 2-3 cis-spliced with the edit site at their join.
    from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

    from organelleverse.annotation.plastome.cds_refine import refine_cds_features

    exon1 = "ATGCCAACTATTAAACAACTTATTAGAAAT"  # 30 nt, no stop
    exon2 = "GCAAAAGAAGGTCTGGCAAAAGTTGAAGGCGAAATCGCAGCTAAAGAAGGTCTGGCAAAAGTTGAAGGCGAAATCGCATC"
    exon3 = "A" + BODY2
    far = "C" * 21000
    genome = LEFT + str(Seq(exon1).reverse_complement()) + far + exon2 + INTRON + exon3 + LEFT
    e1 = (len(LEFT) + 1, len(LEFT) + len(exon1))
    s2 = e1[1] + len(far) + 1
    e2 = (s2, s2 + len(exon2) - 1)
    s3 = e2[1] + len(INTRON) + 1
    e3 = (s3, s3 + len(exon3) - 1)
    edited = str(Seq(exon1 + exon2[:-1] + "T" + exon3).translate(table=11)).rstrip("*")

    def feature(e2_end, e3_start):
        parts = [FeatureLocation(e1[0] - 1, e1[1], strand=-1), FeatureLocation(e2[0] - 1, e2_end, strand=1),
                 FeatureLocation(e3_start - 1, e3[1], strand=1)]
        return SeqFeature(CompoundLocation(parts), type="CDS", qualifiers={"gene": ["rps12"]})

    flanks = reference_junction_flanks([_reference([exon1, exon2, exon3], gene="rps12")])
    feat = feature(e2[1] - 1, e3[0] - 1)  # the join placed one base upstream
    refine_cds_features(genome, [feat], ref_proteins={"rps12": [edited]}, junction_flanks=flanks)
    got = [(int(p.start) + 1, int(p.end), p.strand) for p in feat.location.parts]
    assert got == [(e1[0], e1[1], -1), (e2[0], e2[1], 1), (e3[0], e3[1], 1)]


def test_a_single_dissenting_reference_cannot_overrule_protein():
    cands = [(100.0, [(1, 9), (20, 29)], "ATGAAACCC" + "GGGTTTTAA", [9]),
             (98.0, [(1, 12), (23, 29)], "ATGAAACCCAAA" + "TTTTAA", [12])]
    refs = [[("TGAAACCCAAA"[-10:], "TTTTAA")]]
    assert choose_candidate(cands, refs, tolerance=0.05) == [(1, 9), (20, 29)]


def test_group_ii_score_takes_the_best_equivalent_slide():
    from organelleverse.annotation.plastome.splice_flanks import group_ii_joins

    # exon ...G | GTGCG....AC | exon; sliding is impossible here, score 2.
    seq = "AAAG" + "GTGCGTTTTTTTAC" + "CCCC"
    assert group_ii_joins(seq, [(1, 4), (19, 22)]) == 2
    assert group_ii_joins(seq, [(1, 5), (20, 22)]) < 2
