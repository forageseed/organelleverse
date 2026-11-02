"""Geometry helpers of the standalone SAM 3 inference script (no torch, no sam3, no model is run)."""

from __future__ import annotations

import numpy as np
import pytest

from organelleverse.morphology import sam3_infer as S


def _disk_mask(shape, cy, cx, r):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def test_tile_grid_covers_image_and_is_flush_with_edges():
    tiles = S.tile_grid(2500, 1100, 1008, 256)
    covered = np.zeros((2500, 1100), bool)
    for y0, x0, y1, x1 in tiles:
        assert y1 - y0 <= 1008 and x1 - x0 <= 1008
        covered[y0:y1, x0:x1] = True
    assert covered.all()
    assert max(t[2] for t in tiles) == 2500 and max(t[3] for t in tiles) == 1100
    assert S.tile_grid(300, 400, 1008, 256) == [(0, 0, 300, 400)]
    with pytest.raises(ValueError):
        S.tile_grid(10, 10, 100, 100)


def test_pack_roundtrip_keeps_boxes_masks_scores():
    shape = (120, 160)
    props = [S.to_proposal(_disk_mask(shape, 30, 40, 12), (5, 7), 0.9, "text", 3),
             S.to_proposal(_disk_mask(shape, 80, 120, 20), (0, 0), 0.4, "exemplar", 1)]
    back = S.unpack(S.pack(props))
    assert len(back) == 2
    for a, b in zip(props, back, strict=True):
        assert a.box == b.box and a.ref == b.ref and a.source == b.source
        assert np.array_equal(a.mask, b.mask)
        assert b.score == pytest.approx(a.score)
    assert S.unpack(S.pack([])) == []


def test_suppress_drops_duplicates_and_tile_fragments_but_keeps_neighbours():
    shape = (200, 200)
    whole = S.to_proposal(_disk_mask(shape, 100, 100, 30), (0, 0), 0.8, "t", 0)
    same = S.to_proposal(_disk_mask(shape, 101, 100, 30), (0, 0), 0.7, "t", 1)
    half = _disk_mask(shape, 100, 100, 30)
    half[:, 100:] = False
    fragment = S.to_proposal(half, (0, 0), 0.95, "t", 2)  # higher score, but only part of the object
    other = S.to_proposal(_disk_mask(shape, 30, 30, 15), (0, 0), 0.5, "t", 3)
    kept = S.suppress([whole, same, fragment, other])
    refs = sorted(p.ref for p in kept)
    # the fragment is kept first (highest score); the whole object contains it and is <= 2.2x its
    # size, so it is the same object seen twice; the shifted copy goes by IoU; the neighbour stays
    assert 3 in refs and 1 not in refs
    assert len(refs) == 2


def test_touches_inner_edge_ignores_image_border():
    shape = (500, 500)
    m = np.zeros((200, 200), bool)
    m[0, 50] = True
    assert not S.touches_inner_edge(m, (0, 0, 200, 200), shape)   # top of the image
    assert S.touches_inner_edge(m, (100, 0, 300, 200), shape)     # top of an inner tile
    m2 = np.zeros((200, 200), bool)
    m2[100, 100] = True
    assert not S.touches_inner_edge(m2, (100, 100, 300, 300), shape)


def test_paint_never_overwrites_higher_priority_pixels():
    shape = (60, 60)
    a = S.to_proposal(_disk_mask(shape, 30, 25, 10), (0, 0), 0.9, "a", 1)
    b = S.to_proposal(_disk_mask(shape, 30, 35, 10), (0, 0), 0.5, "b", 2)
    labels = S.paint([a, b], shape)
    assert set(np.unique(labels)) == {0, 1, 2}
    assert (labels[_disk_mask(shape, 30, 25, 10)] == 1).all()


def test_stretch_and_interior_point():
    raw = np.linspace(0, 4095, 100 * 80, dtype=np.float32).reshape(100, 80)
    out = S.stretch_to_uint8(raw)
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255
    mask = np.zeros((40, 40), bool)
    mask[10:30, 5:15] = True
    y, x = S.interior_point(mask)
    # the distance ridge of a 20 x 10 bar is flat along y; x must sit on the centre line
    assert mask[y, x] and 14 <= y <= 25 and 9 <= x <= 10


def test_concept_phrase_translates_the_users_organelle_name():
    assert S.concept_phrase("线粒体") == ("membrane-bound organelle", True)
    assert S.concept_phrase(" Mitochondria ", "EM") == ("membrane-bound organelle", True)
    # not measured: the English noun, flagged unverified
    assert S.concept_phrase("叶绿体") == ("chloroplast", False)
    assert S.concept_phrase("线粒体", "fluorescence") == ("mitochondrion", False)
    assert S.concept_phrase("nucleolus") == ("nucleolus", False)
