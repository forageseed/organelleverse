"""Mitochondrial shape and network metrics for 2-D instance masks.

Definitions (each recorded in the result so tables stay interpretable):

* ``length_rect`` / ``width_rect`` — long and short side of the minimum-area
  bounding rectangle of the object (rotating calipers over the convex hull of
  pixel corners), the length/width definition of Wang et al. 2025, Table 3.
  ``rect_angle_deg`` is the long side's angle to the image x axis.
* ``form_factor`` — P² / (4πA), the reciprocal of circularity (1 for a disc).
* Skeleton (``skimage.morphology.skeletonize``, 8-connected). Structural
  elements follow Ouellet et al. 2017 (PLoS Comput Biol 13:e1005612):
  ``ends`` are skeleton pixels with one neighbour; ``junctions`` are 8-connected
  clusters of pixels with three or more neighbours; ``tubules`` are the
  skeleton pieces left after removing junction pixels (a closed loop without
  junctions is one tubule). An object whose skeleton is a single pixel is a
  punctate mitochondrion counted as one tubule with two ends, so an isolated
  mitochondrion always has 3 elements as in Ouellet's EC definition.
* Skeleton length sums 8-neighbour steps (1 or √2) after dropping diagonal
  steps that duplicate an orthogonal path, so a straight run of n pixels has
  length n − 1.
* Image level: J/E = junctions/ends, T/E = tubules/ends, EC = (ends +
  tubules + junctions) / clusters, where each object is one cluster.

These describe image morphology only; they are not a fusion/fission readout
(that needs time-lapse tracking).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi

__all__ = ["annotate_objects", "network_summary", "object_network", "min_area_rect", "skeleton_raster"]

_EIGHT = np.ones((3, 3), dtype=int)


def min_area_rect(mask: np.ndarray) -> tuple[float, float, float]:
    """(length, width, angle_deg) of the minimum-area rectangle around ``mask``."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0, 0.0, 0.0
    corners = np.concatenate([
        np.stack([xs, ys], axis=1), np.stack([xs + 1, ys], axis=1),
        np.stack([xs, ys + 1], axis=1), np.stack([xs + 1, ys + 1], axis=1),
    ]).astype(np.float64)
    corners = np.unique(corners, axis=0)
    hull = _convex_hull(corners)
    best: tuple[float, float, float, float] | None = None  # area, long, short, angle
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        norm = math.hypot(edge[0], edge[1])
        if norm == 0:
            continue
        u = edge / norm
        v = np.array([-u[1], u[0]])
        pu, pv = hull @ u, hull @ v
        a, b = float(pu.max() - pu.min()), float(pv.max() - pv.min())
        area = a * b
        if best is None or area < best[0] - 1e-9:
            angle = math.degrees(math.atan2(u[1], u[0])) if a >= b else math.degrees(math.atan2(v[1], v[0]))
            best = (area, max(a, b), min(a, b), angle)
    if best is None:
        return 1.0, 1.0, 0.0
    angle = ((best[3] + 90.0) % 180.0) - 90.0
    return best[1], best[2], angle


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain; points are unique (x, y) rows."""
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]
    if len(pts) <= 2:
        return pts

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _skeleton(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize

    return skeletonize(mask.astype(bool))


def _neighbours(skeleton: np.ndarray) -> np.ndarray:
    count = ndi.convolve(skeleton.astype(np.uint8), _EIGHT.astype(np.uint8), mode="constant") - skeleton
    return np.where(skeleton, count, 0)


def _edges(skeleton: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    """8-neighbour steps between skeleton pixels, without redundant diagonals."""
    ys, xs = np.nonzero(skeleton)
    on = {(int(y), int(x)) for y, x in zip(ys, xs)}
    edges = []
    for y, x in on:
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            q = (y + dy, x + dx)
            if q not in on:
                continue
            if dy and dx and ((y, x + dx) in on or (y + dy, x) in on):
                continue  # an orthogonal two-step path already connects them
            edges.append(((y, x), q, math.sqrt(2.0) if dy and dx else 1.0))
    return edges


def object_network(mask: np.ndarray) -> dict[str, Any]:
    """Skeleton elements and lengths of one object (pixel units)."""
    skeleton = _skeleton(mask)
    pixels = int(skeleton.sum())
    if pixels == 0:
        return {"skeleton_length": 0.0, "ends": 0, "junctions": 0, "tubules": 0,
                "branch_lengths": [], "morphology_class": "empty"}
    if pixels == 1:
        return {"skeleton_length": 0.0, "ends": 2, "junctions": 0, "tubules": 1,
                "branch_lengths": [0.0], "morphology_class": "punctate"}
    degree = _neighbours(skeleton)
    ends = int((degree == 1).sum())
    junction_pixels = skeleton & (degree >= 3)
    junction_labels, junctions = ndi.label(junction_pixels, structure=_EIGHT)
    tubule_pixels = skeleton & ~junction_pixels
    tubule_labels, tubules = ndi.label(tubule_pixels, structure=_EIGHT)
    branch = np.zeros(tubules + 1)
    total = 0.0
    for a, b, step in _edges(skeleton):
        total += step
        la, lb = int(tubule_labels[a]), int(tubule_labels[b])
        # A step belongs to the tubule it touches; junction-to-junction steps to none.
        owner = la or lb
        if owner:
            branch[owner] += step
    return {
        "skeleton_length": total, "ends": ends, "junctions": int(junctions), "tubules": int(tubules),
        "branch_lengths": [float(v) for v in branch[1:]],
        "morphology_class": "network" if junctions else "rod",
    }


def annotate_objects(rows: list[dict[str, Any]], instance_labels: np.ndarray, *, pixel_size_um: float | None,
                     network: bool = True) -> None:
    """Add rectangle, form-factor and (optionally) skeleton metrics to measured rows in place."""
    objects = ndi.find_objects(instance_labels)
    for row in rows:
        label = int(row.get("label", 0))
        if label <= 0 or label > len(objects) or objects[label - 1] is None:
            continue
        crop = objects[label - 1]
        mask = instance_labels[crop] == label
        length, width, angle = min_area_rect(mask)
        row["length_rect"] = round(length, 2)
        row["width_rect"] = round(width, 2)
        row["rect_angle_deg"] = round(angle, 1)
        if pixel_size_um:
            row["length_rect_um"] = round(length * pixel_size_um, 4)
            row["width_rect_um"] = round(width * pixel_size_um, 4)
        perimeter, area = row.get("perimeter"), row.get("area_px")
        if isinstance(perimeter, (int, float)) and isinstance(area, (int, float)) and area > 0:
            row["form_factor"] = round(float(perimeter) ** 2 / (4 * math.pi * float(area)), 4)
        if not network:
            continue
        net = object_network(np.pad(mask, 1))
        branches = net.pop("branch_lengths")
        row.update({
            "skeleton_length": round(net["skeleton_length"], 2), "ends": net["ends"], "junctions": net["junctions"],
            "tubules": net["tubules"], "morphology_class": net["morphology_class"],
            "mean_branch_length": round(float(np.mean(branches)), 2) if branches else None,
            "max_branch_length": round(float(np.max(branches)), 2) if branches else None,
        })
        if pixel_size_um:
            row["skeleton_length_um"] = round(net["skeleton_length"] * pixel_size_um, 4)
            row["mean_branch_length_um"] = (round(float(np.mean(branches)) * pixel_size_um, 4) if branches else None)


def network_summary(rows: list[dict[str, Any]], *, pixel_size_um: float | None) -> dict[str, Any] | None:
    """Image-level structural elements and connectivity (Ouellet et al. 2017)."""
    measured = [row for row in rows if "skeleton_length" in row]
    if not measured:
        return None
    ends = sum(int(row["ends"]) for row in measured)
    junctions = sum(int(row["junctions"]) for row in measured)
    tubules = sum(int(row["tubules"]) for row in measured)
    lengths = [float(row["skeleton_length"]) for row in measured]
    total = float(sum(lengths))
    classes = {name: sum(1 for row in measured if row["morphology_class"] == name) for name in ("punctate", "rod", "network")}
    summary: dict[str, Any] = {
        "definition": "Ouellet et al. 2017 structural elements; each object is one cluster",
        "clusters": len(measured), "ends": ends, "tubules": tubules, "junctions": junctions,
        "junctions_per_end": round(junctions / ends, 4) if ends else None,
        "tubules_per_end": round(tubules / ends, 4) if ends else None,
        "elements_per_cluster": round((ends + tubules + junctions) / len(measured), 4),
        "total_skeleton_length": round(total, 2),
        "mean_branch_length": round(total / tubules, 2) if tubules else None,
        "largest_cluster_length_fraction": round(max(lengths) / total, 4) if total else None,
        "morphology_classes": classes,
    }
    if pixel_size_um:
        summary["total_skeleton_length_um"] = round(total * pixel_size_um, 4)
        summary["mean_branch_length_um"] = round(total / tubules * pixel_size_um, 4) if tubules else None
    return summary


def skeleton_raster(instance_labels: np.ndarray) -> np.ndarray:
    """uint8 overlay: 1 skeleton, 2 end, 3 junction (per object, objects never merged)."""
    out = np.zeros(instance_labels.shape, dtype=np.uint8)
    for label, crop in enumerate(ndi.find_objects(instance_labels), start=1):
        if crop is None:
            continue
        mask = np.pad(instance_labels[crop] == label, 1)
        skeleton = _skeleton(mask)
        degree = _neighbours(skeleton)
        layer = np.where(skeleton, 1, 0).astype(np.uint8)
        layer[(degree == 1) & skeleton] = 2
        layer[(degree >= 3) & skeleton] = 3
        region = out[crop]
        inner = layer[1:-1, 1:-1]
        region[inner > 0] = inner[inner > 0]
    return out
