"""Chloroplast bodies and stromule candidates for 2-D instance masks.

Definitions (each recorded in the result so tables stay interpretable):

* Body — every object is opened with a disk of ``body_radius_px`` and only its
  largest connected piece is kept, which removes thin protrusions (stromules,
  1–3 px bumps) from the chloroplast. An object that the opening would erase
  keeps its original pixels.
* Candidates — thin skeleton segments outside the bodies. Candidate pixels are
  the part of the segmentation removed by the opening plus ridge pixels of the
  intensity image, excluding the bodies dilated by 1 px. Ridge = Sato
  bright-ridge filter (``sigmas``) of the image rescaled so the background
  median is 0 and the median body intensity is 1, with the bodies and a 2 px
  halo set to background so body edges give no response. A pixel is a ridge
  pixel when its response passes a hysteresis threshold (high = background
  median + 8 robust σ, low = half of high; σ = 1.4826·MAD of the response
  over background far from any object) AND its smoothed intensity is at least
  ``max(0.05, 3 σ_noise)`` above the background, so dim stromules that fall
  below the segmentation threshold are found while noise is not. The mask is
  thinned to pixels at least half as bright as the local ridge peak.
* Candidates are skeletonised (8-connected). Length sums 8-neighbour steps as
  in :mod:`mito_metrics` (1 or √2, redundant diagonals dropped) plus the gap
  between the attaching skeleton pixel and the body surface; mean width is
  the candidate's pixel count per skeleton pixel.
* An object the opening erases completely is a small chloroplast and keeps its
  pixels, unless it is a thin filament (skeleton length >= ``min_length_px``
  and mean width <= ``max_width_px``): then it is no chloroplast, its pixels
  join the candidate pool, and it is counted in ``summary.free_filaments``
  when no stromule is attached to it.
* A candidate is a stromule when a skeleton pixel lies within ``attach_px`` of
  the 2 px halo zone around a body (it is assigned to the nearest body), its
  length is at least ``min_length_px``, its mean width is at most
  ``max_width_px`` and below half of that body's minor axis (rejects lobes).

These describe image morphology only; they do not establish that a
protrusion is a true stromule rather than e.g. an adjacent thylakoid tail or
an unrelated filament.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .mito_metrics import _edges

__all__ = ["annotate_stromule_rows", "chloroplast_bodies", "detect_stromules", "finalize_stromules",
           "stromule_summary"]

_EIGHT = np.ones((3, 3), dtype=int)
_ROOT_GAP_PX = 2
_DEFINITION = (
    "Body = per-object opening (largest piece kept); candidates = thin skeleton segments outside bodies "
    "(Sato ridge + thin residue) attached to a body; ridge = Sato response above a hysteresis threshold "
    "(high = background median + 8 robust sigma of the response, low = high/2) and smoothed intensity "
    "above background by max(0.05, 3 sigma) of the body-normalised scale; length = skeleton path + gap "
    "to the body surface; kept when length >= min_length_px, mean width (pixels per skeleton pixel) <= max_width_px "
    "and < 0.5 x the body minor axis"
)


def _disk(radius: int) -> np.ndarray:
    span = np.arange(-radius, radius + 1)
    return (span[:, None] ** 2 + span[None, :] ** 2) <= radius * radius


def chloroplast_bodies(labels: np.ndarray, *, radius_px: int) -> np.ndarray:
    """Instance map with thin protrusions removed from every object (see module docstring)."""
    return _open_bodies(labels, radius_px)[0]


def _open_bodies(labels: np.ndarray, radius_px: int) -> tuple[np.ndarray, list[int]]:
    """Bodies plus the labels the opening erased completely (those keep their pixels in the map)."""
    bodies = np.zeros(labels.shape, dtype=labels.dtype)
    erased: list[int] = []
    if radius_px < 1:
        return labels.copy(), erased
    structure = _disk(radius_px)
    margin = radius_px + 1
    height, width = labels.shape
    for label, crop in enumerate(ndi.find_objects(labels), start=1):
        if crop is None:
            continue
        y0, y1 = max(crop[0].start - margin, 0), min(crop[0].stop + margin, height)
        x0, x1 = max(crop[1].start - margin, 0), min(crop[1].stop + margin, width)
        mask = labels[y0:y1, x0:x1] == label
        # Image borders continue the object instead of eroding it.
        top = max(margin - (crop[0].start - y0), 0) if y0 == 0 else 0
        bottom = max(margin - (y1 - crop[0].stop), 0) if y1 == height else 0
        left = max(margin - (crop[1].start - x0), 0) if x0 == 0 else 0
        right = max(margin - (x1 - crop[1].stop), 0) if x1 == width else 0
        pad = ((top, bottom), (left, right))
        padded = np.pad(mask, pad, mode="edge")
        opened = ndi.binary_opening(padded, structure=structure)
        if not opened.any():
            kept = mask
            erased.append(label)
        else:
            pieces, count = ndi.label(opened, structure=_EIGHT)
            if count > 1:
                sizes = np.bincount(pieces.ravel())
                sizes[0] = 0
                opened = pieces == int(np.argmax(sizes))
            kept = opened[top:padded.shape[0] - bottom, left:padded.shape[1] - right]
        region = bodies[y0:y1, x0:x1]
        region[kept] = label
    return bodies, erased


def _is_filament(mask: np.ndarray, *, min_length: float, max_width: float) -> bool:
    """Thin and long: mean width (pixels per skeleton pixel) <= max_width and skeleton length >= min_length."""
    from skimage.morphology import skeletonize

    skeleton = skeletonize(np.pad(mask, 1))
    pixels = int(skeleton.sum())
    if pixels == 0:
        return False
    length = sum(step for _, _, step in _edges(skeleton))
    return length >= min_length and int(mask.sum()) / pixels <= max_width


def _defaults(pixel_size_um: float | None, body_radius_px: int | None, min_length_px: float | None,
              max_width_px: float | None) -> tuple[int, float, float]:
    if pixel_size_um:
        radius = body_radius_px if body_radius_px is not None else int(min(max(round(0.5 / pixel_size_um), 1), 6))
        minimum = min_length_px if min_length_px is not None else 1.0 / pixel_size_um
        maximum = max_width_px if max_width_px is not None else 1.0 / pixel_size_um
    else:
        radius = body_radius_px if body_radius_px is not None else 2
        minimum = min_length_px if min_length_px is not None else 5.0
        maximum = max_width_px if max_width_px is not None else 4.0
    return int(radius), float(minimum), float(maximum)


def _ridge_pixels(image: np.ndarray, foreground: np.ndarray, bodies: np.ndarray, *, sigmas: tuple[float, ...],
                  ) -> tuple[np.ndarray, np.ndarray | None, float, float]:
    """Bright-ridge mask, smoothed body-normalised image, the high threshold used and the intensity margin."""
    from skimage.filters import apply_hysteresis_threshold, sato

    values = image.astype(np.float64, copy=False)
    near = ndi.distance_transform_edt(~foreground)
    far = near > 3.0 * max(sigmas)
    if int(far.sum()) < 100:
        far = near > 1.0
    if not far.any():
        return np.zeros(image.shape, dtype=bool), None, 0.0, 0.0
    background = float(np.median(values[far]))
    contrast = float(np.median(values[bodies > 0])) - background
    if not contrast > 0:
        return np.zeros(image.shape, dtype=bool), None, 0.0, 0.0
    scaled = (values - background) / contrast
    # Bodies (and their blur halo, 2 px) are set to background so their edges cannot
    # produce ridge responses; stromule roots inside that zone are bridged by the gap term.
    zone = ndi.binary_dilation(bodies > 0, structure=_disk(_ROOT_GAP_PX))
    scaled_full = scaled
    scaled = np.where(zone, 0.0, scaled)
    response = sato(scaled, sigmas=sigmas, black_ridges=False)
    sample = response[far]  # far background: no body halo, no zeroed zone
    centre = float(np.median(sample))
    spread = 1.4826 * float(np.median(np.abs(sample - centre)))
    high = centre + 8.0 * spread
    smooth = np.where(zone, 0.0, ndi.gaussian_filter(scaled_full, min(sigmas)))
    noise = 1.4826 * float(np.median(np.abs(smooth[far] - np.median(smooth[far]))))
    margin = max(0.05, 3.0 * noise)
    ridge = apply_hysteresis_threshold(response, 0.5 * high, high) & (smooth > margin) & ~zone
    return ridge, ndi.gaussian_filter(scaled_full, 0.7), high, margin


def _empty_result(parameters: dict[str, Any], chloroplasts: int) -> dict[str, Any]:
    return {"definition": _DEFINITION, "parameters": parameters, "candidates": [],
            "summary": stromule_summary([], chloroplasts=chloroplasts, pixel_size_um=None)}


def detect_stromules(image: np.ndarray, labels: np.ndarray, *, pixel_size_um: float | None,
                     body_radius_px: int | None = None, min_length_px: float | None = None,
                     max_width_px: float | None = None, attach_px: int = 3,
                     sigmas: tuple[float, ...] = (1.0, 2.0)) -> tuple[np.ndarray, dict[str, Any]]:
    """Split ``labels`` into chloroplast bodies and find stromule candidates attached to them.

    Returns the body instance map (same label ids) and the ``stromules`` result dict.
    """
    from skimage.measure import regionprops
    from skimage.morphology import skeletonize

    radius, min_length, max_width = _defaults(pixel_size_um, body_radius_px, min_length_px, max_width_px)
    bodies, erased = _open_bodies(labels, radius)
    # An object the opening erases is either a small chloroplast (kept as it is) or a thin, long
    # filament: a stromule that segmentation split from its body, or a free filament. Filaments are
    # no chloroplasts; their pixels go to the candidate pool.
    boxes = ndi.find_objects(labels)
    filaments: list[int] = []
    for label in erased:
        box = boxes[label - 1]
        if _is_filament(labels[box] == label, min_length=min_length, max_width=max_width):
            filaments.append(label)
            bodies[labels == label] = 0
    parameters: dict[str, Any] = {
        "body_radius_px": radius, "min_length_px": round(min_length, 3), "max_width_px": round(max_width, 3),
        "attach_px": int(attach_px), "sigmas": [float(s) for s in sigmas], "ridge_threshold": 0.0,
    }
    chloroplasts = int(len(np.unique(bodies[bodies > 0])))
    result = _empty_result(parameters, chloroplasts)
    result["summary"]["free_filaments"] = len(filaments)
    if chloroplasts == 0:
        _apply_calibration(result, pixel_size_um)
        return bodies, result

    foreground = labels > 0
    ridge, smooth, high, margin = _ridge_pixels(image, foreground, bodies, sigmas=tuple(sigmas))
    parameters["ridge_threshold"] = round(high, 6)
    parameters["intensity_margin"] = round(margin, 6)
    body_mask = bodies > 0
    guard = ndi.binary_dilation(body_mask, structure=ndi.generate_binary_structure(2, 2))
    candidate_mask = ((foreground & ~body_mask) | ridge) & ~guard
    if smooth is not None:
        # Half-maximum thinning: keep candidate pixels at least half as bright as the local ridge peak,
        # so the blur support of the ridge filter does not inflate the width.
        peak = ndi.maximum_filter(np.where(candidate_mask, smooth, 0.0), footprint=_disk(3))
        candidate_mask = candidate_mask & (smooth >= 0.5 * peak)
    skeleton = skeletonize(candidate_mask)
    components, count = ndi.label(skeleton, structure=_EIGHT)
    if count == 0:
        _apply_calibration(result, pixel_size_um)
        return bodies, result

    pieces, _ = ndi.label(candidate_mask, structure=_EIGHT)
    piece_area = np.bincount(pieces.ravel())
    minor = {int(p.label): float(p.axis_minor_length if hasattr(p, "axis_minor_length") else p.minor_axis_length)
             for p in regionprops(bodies)}
    height, width = labels.shape
    found: list[dict[str, Any]] = []
    accepted_pieces: set[int] = set()
    for index, crop in enumerate(ndi.find_objects(components), start=1):
        if crop is None:
            continue
        mask = components[crop] == index
        ys, xs = np.nonzero(mask)
        ys, xs = ys + crop[0].start, xs + crop[1].start
        reach = int(math.ceil(attach_px)) + _ROOT_GAP_PX + 1
        y0, y1 = max(int(ys.min()) - reach, 0), min(int(ys.max()) + reach + 1, height)
        x0, x1 = max(int(xs.min()) - reach, 0), min(int(xs.max()) + reach + 1, width)
        window = bodies[y0:y1, x0:x1]
        if not window.any():
            continue
        distance, nearest = ndi.distance_transform_edt(window == 0, return_indices=True)
        d = distance[ys - y0, xs - x0]
        best = int(np.argmin(d))
        if d[best] > attach_px + _ROOT_GAP_PX:
            continue
        label = int(window[nearest[0][ys[best] - y0, xs[best] - x0], nearest[1][ys[best] - y0, xs[best] - x0]])
        piece = np.zeros((int(ys.max() - ys.min()) + 1, int(xs.max() - xs.min()) + 1), dtype=bool)
        piece[ys - ys.min(), xs - xs.min()] = True
        steps = sum(step for _, _, step in _edges(piece))
        length = steps + max(float(d[best]) - 1.0, 0.0)
        # Band width = pixels of the candidate piece per skeleton pixel (unbiased for 2- and 3-row bands).
        mean_width = float(piece_area[pieces[ys[0], xs[0]]]) / len(ys)
        if length < min_length or mean_width > max_width:
            continue
        if label not in minor or not mean_width < 0.5 * minor[label]:
            continue
        order = np.lexsort((xs, ys))
        accepted_pieces.add(int(pieces[ys[0], xs[0]]))
        found.append({
            "label": label, "length": round(length, 2), "mean_width": round(mean_width, 2),
            "skeleton": [[int(ys[i]), int(xs[i])] for i in order],
            "bbox": [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())],
        })
    # Filaments that ended up in no stromule are free filaments: not a chloroplast, not a stromule.
    free = sum(1 for label in filaments if not accepted_pieces & set(np.unique(pieces[labels == label]).tolist()))
    result["summary"]["free_filaments"] = free
    found.sort(key=lambda c: (c["label"], c["bbox"][0], c["bbox"][1], c["bbox"][2], c["bbox"][3]))
    result["candidates"] = [{"id": i, **c} for i, c in enumerate(found, start=1)]
    _apply_calibration(result, pixel_size_um)
    return bodies, result


def _apply_calibration(result: dict[str, Any], pixel_size_um: float | None) -> None:
    """Add the µm fields (calibrated images only) and the summary."""
    candidates = result["candidates"]
    if pixel_size_um:
        for c in candidates:
            c["length_um"] = round(c["length"] * pixel_size_um, 4)
            c["mean_width_um"] = round(c["mean_width"] * pixel_size_um, 4)
    result["summary"] = stromule_summary(candidates, chloroplasts=int(result["summary"]["chloroplasts"]),
                                         pixel_size_um=pixel_size_um,
                                         free_filaments=result["summary"].get("free_filaments"))


def stromule_summary(candidates: list[dict[str, Any]], *, chloroplasts: int, pixel_size_um: float | None,
                     free_filaments: int | None = None) -> dict[str, Any]:
    """Counts, frequency and total length of the given (accepted) candidates.

    ``free_filaments`` = thin objects that are neither chloroplasts nor attached stromules.
    """
    with_stromules = len({int(c["label"]) for c in candidates})
    total = float(sum(float(c["length"]) for c in candidates))
    summary: dict[str, Any] = {
        "chloroplasts": int(chloroplasts), "with_stromules": with_stromules,
        "frequency": round(with_stromules / chloroplasts, 4) if chloroplasts else None,
        "stromules": len(candidates), "total_length": round(total, 2),
    }
    if pixel_size_um:
        summary["total_length_um"] = round(total * pixel_size_um, 4)
    if free_filaments is not None:
        summary["free_filaments"] = int(free_filaments)
    return summary


def finalize_stromules(result: dict[str, Any], measured_labels: set[int], *,
                       pixel_size_um: float | None) -> dict[str, Any]:
    """Keep candidates of measured bodies only, renumber ids 1..n and recompute the summary."""
    kept = [c for c in result["candidates"] if int(c["label"]) in measured_labels]
    candidates = [{**c, "id": i} for i, c in enumerate(kept, start=1)]
    return {**result, "candidates": candidates,
            "summary": stromule_summary(candidates, chloroplasts=len(measured_labels), pixel_size_um=pixel_size_um,
                                        free_filaments=result["summary"].get("free_filaments"))}


def annotate_stromule_rows(rows: list[dict[str, Any]], candidates: list[dict[str, Any]], *,
                           pixel_size_um: float | None) -> None:
    """Add ``stromules`` / ``stromule_length`` (/ ``_um``) to measured rows in place."""
    count: dict[int, int] = {}
    length: dict[int, float] = {}
    for c in candidates:
        label = int(c["label"])
        count[label] = count.get(label, 0) + 1
        length[label] = length.get(label, 0.0) + float(c["length"])
    for row in rows:
        label = int(row.get("label", 0))
        row["stromules"] = count.get(label, 0)
        row["stromule_length"] = round(length.get(label, 0.0), 2)
        if pixel_size_um:
            row["stromule_length_um"] = round(length.get(label, 0.0) * pixel_size_um, 4)
