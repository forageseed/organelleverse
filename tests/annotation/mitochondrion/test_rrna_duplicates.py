"""One rRNA locus must give one rRNA call, however many references hit it."""

from __future__ import annotations

from organelleverse.annotation.mitochondrion.rrna import RawRRNA, _merge_rrna_fragments


def _span(hits):
    return sorted((h.gene_name, h.start, h.end, h.strand) for h in hits)


def test_identical_and_shifted_hits_at_one_locus_collapse_to_the_longest():
    # Arabidopsis, leave-one-species-out: five references hit the same 5S and 18S loci
    hits = [
        RawRRNA("rrn5", 126001, 126118, -1, 70.0, "5S"),
        RawRRNA("rrn5", 126004, 126115, -1, 71.0, "5S"),
        RawRRNA("rrn5", 126004, 126116, -1, 69.0, "5S"),
        RawRRNA("rrn18", 126289, 128223, -1, 900.0, "18S"),
        RawRRNA("rrn18", 126290, 128223, -1, 910.0, "18S"),
        RawRRNA("rrn18", 126962, 128223, -1, 700.0, "18S"),
        RawRRNA("rrn18", 127087, 128222, -1, 600.0, "18S"),
    ]

    assert _span(_merge_rrna_fragments(hits)) == [
        ("rrn18", 126289, 128223, -1),
        ("rrn5", 126001, 126118, -1),
    ]


def test_a_longer_lower_scoring_hit_still_replaces_the_fragment_it_contains():
    hits = [
        RawRRNA("rrn18", 2000, 3000, 1, 950.0, "18S"),
        RawRRNA("rrn18", 1000, 3000, 1, 400.0, "18S"),
    ]

    assert _span(_merge_rrna_fragments(hits)) == [("rrn18", 1000, 3000, 1)]


def test_separate_copies_of_the_same_gene_are_kept():
    hits = [
        RawRRNA("rrn26", 5000, 8200, 1, 800.0, "26S"),
        RawRRNA("rrn26", 90000, 93200, -1, 790.0, "26S"),
        RawRRNA("rrn5", 100, 218, 1, 80.0, "5S"),
        RawRRNA("rrn5", 400000, 400118, 1, 79.0, "5S"),
    ]

    assert len(_merge_rrna_fragments(hits)) == 4


def test_fragments_overlapping_only_at_their_ends_are_still_merged():
    hits = [
        RawRRNA("rrn26", 5000, 6400, -1, 82.0, "26S"),
        RawRRNA("rrn26", 6350, 7600, -1, 83.0, "26S"),
    ]

    assert _span(_merge_rrna_fragments(hits)) == [("rrn26", 5000, 7600, -1)]


def test_hits_on_the_opposite_strand_are_not_treated_as_duplicates():
    hits = [
        RawRRNA("rrn18", 1000, 2900, 1, 900.0, "18S"),
        RawRRNA("rrn18", 1000, 2900, -1, 880.0, "18S"),
    ]

    assert len(_merge_rrna_fragments(hits)) == 2
