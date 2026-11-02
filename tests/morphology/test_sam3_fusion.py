"""SAM 3 + model-probability fusion rules (pure numpy)."""

from __future__ import annotations

import numpy as np

from organelleverse.morphology import sam3_fusion as F
from organelleverse.morphology import sam3_infer as S


def _disk(shape, cy, cx, r):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


SHAPE = (120, 120)


def test_label_proposals_and_mean_probability():
    labels = np.zeros(SHAPE, np.int32)
    labels[_disk(SHAPE, 30, 30, 10)] = 1
    labels[_disk(SHAPE, 80, 80, 12)] = 3  # gaps in the label numbering are skipped
    props = F.label_proposals(labels)
    assert [p.ref for p in props] == [1, 3]
    prob = np.zeros(SHAPE, np.uint8)
    prob[labels == 1] = 204
    assert F.mean_probability(props[0], prob) == 0.8
    assert F.mean_probability(props[1], prob.astype(np.float32) / 255) == 0.0


def test_refine_takes_best_agreeing_hypothesis_and_gates_on_probability():
    cand_mask = _disk(SHAPE, 60, 60, 15)
    labels = cand_mask.astype(np.int32)
    cand = F.label_proposals(labels)
    tight = S.to_proposal(_disk(SHAPE, 60, 60, 13), (0, 0), 0.9, "refine0", 1)     # agrees, best score
    weaker = S.to_proposal(_disk(SHAPE, 60, 61, 15), (0, 0), 0.7, "refine1", 1)    # agrees, lower score
    cell = S.to_proposal(np.ones(SHAPE, bool), (0, 0), 0.99, "refine2", 1)        # latched onto everything
    prob = np.where(_disk(SHAPE, 60, 60, 14), 230, 40).astype(np.uint8)
    out = F.refine_objects(cand, [tight, weaker, cell], prob, min_iou=0.5, min_confidence=0.6, min_area=10)
    assert len(out) == 1 and out[0].source == "sam3_refined"
    assert np.array_equal(out[0].mask, tight.mask)
    # same candidate without an agreeing hypothesis is kept as drawn by the model
    out = F.refine_objects(cand, [cell], prob, min_iou=0.5, min_confidence=0.6, min_area=10)
    assert out[0].source == "model" and out[0].area == int(cand_mask.sum())
    # a stricter probability floor drops it
    assert F.refine_objects(cand, [tight], prob, min_iou=0.5, min_confidence=0.95, min_area=10) == []


def test_recover_adds_only_new_probable_objects():
    kept = [S.to_proposal(_disk(SHAPE, 30, 30, 10), (0, 0), 0.9, "model", 1)]
    prob = np.zeros(SHAPE, np.uint8)
    prob[_disk(SHAPE, 90, 90, 12)] = 128  # a faint object the model threshold missed
    repeat = S.to_proposal(_disk(SHAPE, 30, 31, 10), (0, 0), 0.8, "exemplar", 0)
    faint = S.to_proposal(_disk(SHAPE, 90, 90, 11), (0, 0), 0.6, "exemplar", 0)
    background = S.to_proposal(_disk(SHAPE, 90, 30, 11), (0, 0), 0.9, "exemplar", 0)
    out = F.add_recovered(kept, [repeat, faint, background], prob, SHAPE, min_score=0.5, min_confidence=0.35,
                          min_area=10)
    assert [p.source for p in out] == ["model", "sam3_recovered"]
    assert out[1].score == F.mean_probability(faint, prob)
    assert F.add_recovered(kept, [faint], prob, SHAPE, min_score=0.7, min_confidence=0.35, min_area=10) == kept
