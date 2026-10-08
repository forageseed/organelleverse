"""CDS blocks with no bases between them are one exon (an intron lost in the target)."""

from __future__ import annotations

from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from organelleverse.annotation.plastome.pipeline import _merge_abutting_parts


def _cds(*parts):
    return SeqFeature(CompoundLocation([FeatureLocation(a, b, strand=s) for a, b, s in parts]), type="CDS")


def test_abutting_blocks_merge_on_both_strands():
    plus = _cds((54342, 54413, 1), (54413, 54705, 1), (55327, 55552, 1))  # Medicago clpP
    minus = _cds((900, 1000, -1), (800, 900, -1), (100, 300, -1))
    assert _merge_abutting_parts([plus, minus]) == 2
    assert [(int(p.start), int(p.end)) for p in plus.location.parts] == [(54342, 54705), (55327, 55552)]
    assert [(int(p.start), int(p.end)) for p in minus.location.parts] == [(800, 1000), (100, 300)]


def test_real_introns_and_origin_split_stay():
    intron = _cds((100, 200, 1), (900, 1000, 1))
    origin = _cds((149900, 150000, 1), (0, 300, 1))
    trans = _cds((100, 200, 1), (200, 260, -1))
    assert _merge_abutting_parts([intron, origin, trans]) == 0
    assert len(intron.location.parts) == len(origin.location.parts) == len(trans.location.parts) == 2
