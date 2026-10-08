"""Structural trans-splicing detection and its GenBank / structure-map consequences."""

from __future__ import annotations

from organelleverse.annotation.models import (
    AnnotationFeature,
    FeatureQualifier,
    LocationPart,
)
from organelleverse.annotation.splice_layout import MAX_CIS_INTRON, parts_are_trans_spliced
from organelleverse.annotation.writer import _seq_feature

_L = 100_000


def test_a_short_forward_intron_is_cis():
    assert not parts_are_trans_spliced([(1000, 1200, 1), (2500, 2700, 1)], _L)


def test_a_minus_strand_cis_join_is_cis():
    assert not parts_are_trans_spliced([(2500, 2700, -1), (1000, 1200, -1)], _L)


def test_mixed_strands_are_trans():
    assert parts_are_trans_spliced([(1000, 1200, -1), (5000, 5200, 1)], _L)


def test_a_backward_jump_on_one_strand_is_trans():
    # nad5-like: exon 3 sits downstream of exons 4-5 along the circle
    parts = [(9000, 9200, 1), (9900, 10100, 1), (90000, 90100, 1), (2000, 2100, 1)]
    assert parts_are_trans_spliced(parts, 400_000)


def test_a_forward_gap_beyond_any_cis_intron_is_trans():
    # nad5-like: same strand, exon 2 -> 3 skips 123 kb
    parts = [(1000, 1200, 1), (125_000, 125_200, 1)]
    assert parts_are_trans_spliced(parts, 400_000, MAX_CIS_INTRON)
    # drawing code never reclassifies by distance
    assert not parts_are_trans_spliced(parts, 400_000)


def test_a_long_forward_intron_in_a_small_genome_is_not_a_backward_jump():
    assert not parts_are_trans_spliced([(1, 10, 1), (35_000, 35_100, 1)], 40_000)


def test_a_join_across_the_origin_is_contiguous_not_trans():
    assert not parts_are_trans_spliced([(_L - 300, _L, 1), (0, 400, 1)], _L)


def test_a_single_part_is_never_trans():
    assert not parts_are_trans_spliced([(10, 500, 1)], _L)


def _feature(parts, **qualifiers):
    return AnnotationFeature(
        feature_id="s:0:CDS",
        seqid="s",
        type="CDS",
        operator="join" if len(parts) > 1 else "single",
        parts=tuple(LocationPart(start=a, end=b, strand=c) for a, b, c in parts),
        qualifiers=tuple(FeatureQualifier(name=k, values=tuple(v)) for k, v in qualifiers.items()),
        parents=(),
    )


def test_the_writer_marks_a_long_one_strand_gap_as_trans_spliced():
    feature = _feature([(1000, 1200, 1), (125_000, 125_200, 1)], gene=["nad5"])
    assert "trans_splicing" in _seq_feature(feature, 400_000).qualifiers


def test_the_writer_marks_a_structurally_trans_spliced_cds():
    feature = _feature([(1000, 1200, -1), (5000, 5200, 1)], gene=["nad1"])
    assert "trans_splicing" in _seq_feature(feature, _L).qualifiers


def test_the_writer_leaves_a_cis_cds_alone():
    feature = _feature([(1000, 1200, 1), (2500, 2700, 1)], gene=["cox2"])
    assert "trans_splicing" not in _seq_feature(feature, _L).qualifiers


def test_an_existing_trans_splicing_qualifier_is_kept_once():
    feature = _feature([(1000, 1200, -1), (5000, 5200, 1)], trans_splicing=[])
    assert list(_seq_feature(feature, _L).qualifiers).count("trans_splicing") == 1
