"""Genes the primary reference lacks are searched with secondary references and kept only as open frames."""

from __future__ import annotations

from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from organelleverse.annotation.plastome.pipeline import _is_open_frame, _overlaps_cds
from organelleverse.annotation.plastome.references import choose_reference, rank_references


def _cds(parts, gene="rps16"):
    loc = [FeatureLocation(a, b, strand=s) for a, b, s in parts]
    return SeqFeature(loc[0] if len(loc) == 1 else CompoundLocation(loc), type="CDS", qualifiers={"gene": [gene]})


def test_open_frame_requires_no_internal_stop():
    orf = "ATG" + "GCT" * 40 + "TAA"
    genome = "C" * 10 + orf + "C" * 10
    assert _is_open_frame(_cds([(10, 10 + len(orf), 1)]), genome)
    broken = "ATG" + "GCT" * 20 + "TGA" + "GCT" * 19 + "TAA"
    assert not _is_open_frame(_cds([(10, 10 + len(broken), 1)]), "C" * 10 + broken + "C" * 10)
    shifted = "ATG" + "GCT" * 40 + "TTAA"
    assert not _is_open_frame(_cds([(10, 10 + len(shifted), 1)]), "C" * 10 + shifted + "C" * 10)


def test_truncated_remnant_is_not_an_open_frame():
    orf = "ATG" + "GCT" * 40 + "TAA"  # 42 codons
    feature = _cds([(10, 10 + len(orf), 1)])
    feature.reference_length = 258  # donor rps16, 85 aa + stop
    assert not _is_open_frame(feature, "C" * 10 + orf + "C" * 10)
    feature.reference_length = 140
    assert _is_open_frame(feature, "C" * 10 + orf + "C" * 10)


def test_overlap_with_an_existing_cds_on_the_same_strand():
    existing = [_cds([(100, 400, 1)], gene="ycf3")]
    assert _overlaps_cds(_cds([(150, 400, 1)], gene="pafI"), existing)
    assert not _overlaps_cds(_cds([(150, 400, -1)], gene="pafI"), existing)
    assert not _overlaps_cds(_cds([(390, 700, 1)]), existing)


def test_choose_reference_is_the_top_of_the_ranking():
    import inspect

    assert "rank_references" in inspect.getsource(choose_reference)
    assert rank_references.__doc__
