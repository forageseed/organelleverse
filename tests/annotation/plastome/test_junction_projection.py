from types import SimpleNamespace

import pytest
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from organelleverse.annotation.plastome import junction_projection as jp


def feature(parts, codon_start=1):
    return SeqFeature(
        CompoundLocation(parts), type="CDS", qualifiers={"codon_start": [str(codon_start)]}
    )


@pytest.mark.parametrize("strand", [1, -1])
@pytest.mark.parametrize("codon_start,expected", [(1, (4, 11)), (2, (3, 10)), (3, (2, 9))])
def test_junctions_follow_biological_order_and_codon_start(strand, codon_start, expected):
    parts = [
        FeatureLocation(20, 24, strand=strand),
        FeatureLocation(3, 10, strand=strand),
        FeatureLocation(40, 44, strand=strand),
    ]
    assert jp.coding_junctions(feature(parts, codon_start)) == expected


def test_shared_first_exon_and_mirrored_ir_keep_same_coding_architecture():
    shared = FeatureLocation(30, 34, strand=1)
    left = feature([shared, FeatureLocation(50, 57, strand=1), FeatureLocation(70, 74, strand=1)])
    right = feature([shared, FeatureLocation(13, 20, strand=-1), FeatureLocation(3, 7, strand=-1)])
    a = jp.junction_product("ref:CDS:left", left, "MKWFR*")
    b = jp.junction_product("ref:CDS:right", right, "MKWFR*")
    assert a.junctions == b.junctions == (4, 11)
    rows = jp.compare_junction_evidence(a, b, (a, b))
    assert [r.reference_id for r in rows] == [a.product_id, b.product_id]
    assert all(
        r.comparable and not r.gained_source_junctions and not r.lost_source_junctions for r in rows
    )


def test_equal_protein_does_not_erase_split_codon_boundary_difference():
    source = jp.JunctionProduct("source", "MKWFR", (3, 7))
    old = jp.JunctionProduct("current", "MKWFR", (3, 7))
    new = jp.JunctionProduct("candidate", "MKWFR", (3, 6))
    (evidence,) = jp.compare_junction_evidence(old, new, (source,))
    assert evidence.comparable
    assert evidence.current.matched_pairs == ((3, 3), (7, 7))
    assert evidence.candidate.matched_pairs == ((3, 3),)
    assert evidence.lost_source_junctions == (7,)


def test_terminal_extension_is_not_mistaken_for_a_different_homologous_junction():
    source = jp.JunctionProduct("source", "MKWFR", (3, 7))
    target = jp.JunctionProduct("extended", "QQMKWFR", (9, 13))
    assert jp.project_junctions(source, target).matched_pairs == ((3, 9), (7, 13))


def test_intervening_target_insertion_does_not_prove_between_codon_cut():
    indices = [[0, -1, 1, 2], [0, 1, 2, 3]]
    assert jp._matched_junctions((3,), (6,), indices) == ()


def test_unaligned_flanks_do_not_prove_a_cut():
    assert jp._matched_junctions((3,), (3,), [[0, 1], [-1, 1]]) == ()


def test_repeated_optimal_alignment_remains_ambiguous():
    source = jp.JunctionProduct("source", "MKWFR", (3,))
    target = jp.JunctionProduct("repeat", "MKWFRMKWFR", (3, 18))
    result = jp.project_junctions(source, target)
    assert result.status == "ambiguous_alignment" and result.alignment_count > 1
    assert result.matched_pairs == ()
    (row,) = jp.compare_junction_evidence(source, target, (source,))
    assert not row.comparable and row.gained_source_junctions == row.lost_source_junctions == ()


def test_alignment_count_overflow_is_explicitly_ambiguous(monkeypatch):
    class OverflowAlignments:
        def __len__(self):
            raise OverflowError("Too many optimal alignments")

    monkeypatch.setattr(jp, "_AL", SimpleNamespace(align=lambda a, b: OverflowAlignments()))
    result = jp.project_junctions(
        jp.JunctionProduct("s", "M", ()), jp.JunctionProduct("t", "M", ())
    )
    assert result.status == "ambiguous_alignment" and result.alignment_count is None


def test_no_alignment_is_not_junction_support(monkeypatch):
    monkeypatch.setattr(jp, "_AL", SimpleNamespace(align=lambda a, b: []))
    result = jp.project_junctions(
        jp.JunctionProduct("s", "M", ()), jp.JunctionProduct("t", "M", ())
    )
    assert result.status == "no_alignment" and result.matched_pairs == ()


def test_duplicate_source_identity_fails_instead_of_collapsing_reference_copies():
    p = jp.JunctionProduct("source", "MKWFR", (3,))
    with pytest.raises(ValueError, match="identities"):
        jp.compare_junction_evidence(p, p, (p, p))


def test_unscored_product_and_invalid_codon_start_fail_at_boundary():
    f = feature([FeatureLocation(0, 3), FeatureLocation(9, 12)])
    with pytest.raises(ValueError, match="translation"):
        jp.junction_product("id", f, "")
    f.qualifiers["codon_start"] = ["4"]
    with pytest.raises(ValueError, match="codon_start"):
        jp.coding_junctions(f)


def test_n_terminal_extension_loses_source_start_but_retains_junction_and_end():
    source = jp.JunctionProduct("short", "MKWFR", (3, 7))
    candidate = jp.JunctionProduct("long", "QQMKWFR", (9, 13))
    (row,) = jp.compare_junction_evidence(source, candidate, (source,))
    assert row.current.matched_translation_ends == ("N", "C")
    assert row.candidate.matched_translation_ends == ("C",)
    assert row.lost_source_junctions == ()
    assert row.lost_translation_ends == ("N",)


def test_c_terminal_extension_is_separate_from_internal_junction_support():
    source = jp.JunctionProduct("short", "MKWFR", (3,))
    candidate = jp.JunctionProduct("long", "MKWFRQQ", (3,))
    (row,) = jp.compare_junction_evidence(source, candidate, (source,))
    assert row.candidate.matched_pairs == ((3, 3),)
    assert row.candidate.matched_translation_ends == ("N",)
    assert row.lost_translation_ends == ("C",)


def test_long_reference_records_opposite_start_evidence_without_voting():
    current = jp.JunctionProduct("short", "MKWFR", (3,))
    candidate = jp.JunctionProduct("long", "QQMKWFR", (9,))
    short_row, long_row = jp.compare_junction_evidence(current, candidate, (current, candidate))
    assert short_row.lost_translation_ends == ("N",)
    assert long_row.gained_translation_ends == ("N",)
    assert short_row.comparable and long_row.comparable


def test_ambiguous_alignment_has_no_known_terminal_gain_or_loss():
    p = jp.JunctionProduct("source", "MKWFR", (3,))
    repeated = jp.JunctionProduct("repeat", "MKWFRMKWFR", (3, 18))
    (row,) = jp.compare_junction_evidence(p, repeated, (p,))
    assert row.candidate.matched_translation_ends == ()
    assert not row.comparable
    assert row.gained_translation_ends == row.lost_translation_ends == ()
