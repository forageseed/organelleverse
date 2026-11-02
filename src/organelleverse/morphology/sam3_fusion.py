"""Fuse SAM 3 proposals with a calibrated model's probability map (pure numpy, no torch).

The lab mitochondria U-Net knows *what* a mitochondrion is in the user's EM; SAM 3 does not (zero-shot
text prompts find nothing on these images) but draws sharp boundaries once it is told *where* an
object is. The fusion therefore keeps the U-Net as the judge and SAM 3 as the draughtsman:

``refine``
    Every candidate object of the U-Net is replaced by the SAM 3 hypothesis with the highest
    predicted IoU among those that still overlap the candidate by IoU >= ``min_iou`` (SAM may
    otherwise latch onto the surrounding cell); a candidate without such a hypothesis is kept as
    it was. The object survives when its mean U-Net probability inside the final outline is at
    least ``min_confidence``.
``recover``
    SAM 3 exemplar proposals (prompted with the confident objects of the same image) that lie at
    most ``max_inside`` inside already kept objects are added when SAM scores them >= ``min_score``
    and the U-Net gives them a mean probability >= ``min_confidence`` -- a second, lower bar that
    lets SAM bring back faint or elongated mitochondria the U-Net threshold cut off.

All objects are :class:`~.sam3_infer.Proposal` (box + crop mask on the full image grid).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .sam3_infer import Proposal, pair_iou, paint

__all__ = ["add_recovered", "label_proposals", "mean_probability", "refine_objects"]


def label_proposals(labels: np.ndarray, source: str = "candidate") -> list[Proposal]:
    """One proposal per label of an instance map; ``ref`` is the label."""
    from scipy import ndimage as ndi

    out = []
    for k, sl in enumerate(ndi.find_objects(labels), 1):
        if sl is None:
            continue
        out.append(Proposal((sl[0].start, sl[1].start, sl[0].stop, sl[1].stop), labels[sl] == k, 1.0, source, k))
    return out


def mean_probability(p: Proposal, prob: np.ndarray) -> float:
    """Mean of ``prob`` (uint8 0..255 or float 0..1) inside the proposal."""
    y0, x0, y1, x1 = p.box
    values = prob[y0:y1, x0:x1][p.mask]
    if values.size == 0:
        return 0.0
    scale = 255.0 if prob.dtype == np.uint8 else 1.0
    return float(values.mean()) / scale


def refine_objects(candidates: Iterable[Proposal], hypotheses: Iterable[Proposal], prob: np.ndarray, *,
                   min_iou: float, min_confidence: float, min_area: int) -> list[Proposal]:
    """Candidates with their best agreeing SAM outline; score = mean probability; source refined/kept."""
    by_ref: dict[int, list[Proposal]] = {}
    for h in hypotheses:
        by_ref.setdefault(h.ref, []).append(h)
    out = []
    for cand in candidates:
        best = None
        for h in by_ref.get(cand.ref, []):
            if pair_iou(h, cand)[0] >= min_iou and (best is None or h.score > best.score):
                best = h
        obj = best if best is not None else cand
        if obj.area < min_area:
            continue
        conf = mean_probability(obj, prob)
        if conf >= min_confidence:
            out.append(Proposal(obj.box, obj.mask, conf, "sam3_refined" if best is not None else "model", cand.ref))
    return out


def add_recovered(kept: list[Proposal], pool: Iterable[Proposal], prob: np.ndarray, shape: tuple[int, int], *,
                  min_score: float, min_confidence: float, min_area: int,
                  max_inside: float = 0.2) -> list[Proposal]:
    """``kept`` plus pool proposals that pass both bars and do not repeat a kept object."""
    occupied = paint(kept, shape) > 0
    added = []
    for p in sorted(pool, key=lambda q: -q.score):
        if p.score < min_score or p.area < min_area:
            continue
        y0, x0, y1, x1 = p.box
        inside = (occupied[y0:y1, x0:x1] & p.mask).sum() / max(1, p.area)
        if inside > max_inside:
            continue
        conf = mean_probability(p, prob)
        if conf < min_confidence:
            continue
        added.append(Proposal(p.box, p.mask, conf, "sam3_recovered", p.ref))
        occupied[y0:y1, x0:x1] |= p.mask
    return kept + added
