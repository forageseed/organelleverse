"""Morphometric quantification of organelle segmentation label maps.

Pure-Python (numpy / scipy / scikit-image) — runs offline, no deep-learning
stack required. Consumes an integer label map produced by OrgSegNet (or any
semantic segmentation tool) where each pixel value is a class index, and
returns per-object morphometrics + per-class counts.

OrgSegNet (Plantorganelle Hunter, Feng et al. *Nat. Plants* 2023) class
indices::

    0 = background
    1 = Chloroplast
    2 = Mitochondria
    3 = Vacuole
    4 = Nucleus

Standard morphometrics follow scikit-image ``regionprops`` (area, perimeter,
equivalent_diameter, eccentricity). Circularity is the classical
``4*pi*area / perimeter**2`` (a perfect circle = 1). Mitochondria are small
and often touching; ``watershed_mito=True`` separates touching instances via
distance-transform watershed (scipy.ndimage) before measurement, matching the
standard instance-segmentation post-processing used in EM analysis.

References
----------
- Feng, X., Yu, Z., Fang, H. et al. (2023) *Plantorganelle Hunter: an
  open-source pipeline for plant organelle identification and segmentation.*
  Nature Plants 9:1705-1717. doi:10.1038/s41477-023-01527-5
- scikit-image: van der Walt, S. et al. (2014) *scikit-image: image processing
  in Python.* PeerJ 2:e453.
- Beucher, S. & Lantuéjoul, C. (1979) *Use of watersheds in contour
  detection.* Int. Workshop Image Processing, Rennes. (watershed)
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import measure as skmeasure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from ..core.errors import OrganelleInputError

__all__ = [
    "ORGSEG_CLASSES",
    "ORGSEG_PALETTE",
    "measure",
    "overlay",
    "summarize",
    "write_csv",
]

# OrgSegNet class index -> name (Plantorganelle Hunter, Nat. Plants 2023).
ORGSEG_CLASSES: tuple[str, ...] = (
    "background",
    "Chloroplast",
    "Mitochondria",
    "Vacuole",
    "Nucleus",
)
# Display palette (RGB), matching the OrgSegNet PlantCellDataset METAINFO.
ORGSEG_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),  # background - black
    1: (128, 0, 0),  # Chloroplast - maroon
    2: (0, 128, 0),  # Mitochondria - green
    3: (128, 128, 0),  # Vacuole - olive
    4: (0, 0, 128),  # Nucleus - navy
}


def _load_image(path: str | Path | np.ndarray) -> np.ndarray:
    """Load an image as a numpy array (supports TIF/PNG via PIL)."""
    if isinstance(path, np.ndarray):
        return path
    from PIL import Image

    img = Image.open(str(path))
    return np.array(img)


def _split_touching(mask: np.ndarray) -> np.ndarray:
    """Separate touching objects in a binary mask via distance-transform
    watershed (Beucher & Lantuéjoul 1979).

    Returns a labeled integer array (0 = background, 1..N = instances).
    Seed markers are the local maxima of the Euclidean distance transform,
    with ``min_distance`` set from the typical object radius so each blob
    yields one marker (avoids over-segmentation of wide objects).
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=np.int32)
    distance = ndi.distance_transform_edt(mask)
    # Estimate a per-image min_distance from the distance-transform peaks:
    # roughly the median object "radius" (peak distance value), clamped to a
    # sane range so tiny objects aren't over-split and large blobs get a
    # min_distance large enough to merge spurious double peaks.
    peak = float(distance.max())
    min_dist = int(max(3, min(round(peak * 0.8), 30)))
    coords = peak_local_max(distance, labels=mask, min_distance=min_dist)
    markers = np.zeros(distance.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i
    if markers.max() == 0:
        return skmeasure.label(mask).astype(np.int32)
    labels = watershed(-distance, markers, mask=mask)
    return labels


def _shape_complexity(
    instance_labels: np.ndarray, label: int, *, n_points: int = 24
) -> float | None:
    """OrgSegNet shape-complexity δ (Feng et al., Nat. Plants 2023, trait 1/3).

    Equally-spaced points on the object's longest contour form the nodes of a
    visibility graph; two nodes connect when the straight segment between them
    stays inside the object. δ = 2m / (n(n-1)): 1.0 for a convex (circle-like)
    profile, lower as outlines grow concave / complex. Scale-invariant.
    """
    contours = skmeasure.find_contours(instance_labels == label, 0.5)
    if not contours:
        return None
    contour = max(contours, key=len)
    if len(contour) < n_points:
        return None
    # Close the loop and resample at equal arc length.
    closed = np.vstack([contour, contour[:1]])
    seg = np.sqrt(((closed[1:] - closed[:-1]) ** 2).sum(axis=1))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total <= 0:
        return None
    targets = np.linspace(0.0, total, n_points, endpoint=False)
    pts = np.stack(
        [np.interp(targets, arc, closed[:, 0]), np.interp(targets, arc, closed[:, 1])],
        axis=1,
    )
    mask = instance_labels == label
    height, width = mask.shape
    visible = 0
    for a in range(n_points):
        for b in range(a + 1, n_points):
            p, q = pts[a], pts[b]
            chord = float(np.hypot(*(q - p)))
            steps = max(int(chord), 2)
            ts = np.linspace(0.0, 1.0, steps + 1)[1:-1]
            xs = np.rint(p[1] + ts * (q[1] - p[1])).astype(np.intp)
            ys = np.rint(p[0] + ts * (q[0] - p[0])).astype(np.intp)
            if not ((xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)).all():
                continue
            if mask[ys, xs].all():
                visible += 1
    return round(2.0 * visible / (n_points * (n_points - 1)), 4)


def _feret_diameter_min(
    instance_labels: np.ndarray, label: int, *, step_deg: float = 1.0
) -> float | None:
    """Minimum caliper (Feret) diameter by a 1° rotation scan over the
    object's pixels (ImageJ ``MinFeret``). Pixels are unit squares: the scan
    projects pixel centers and adds the unit-square projection (Minkowski
    correction), which matches ImageJ's polygon width on axis-aligned
    shapes instead of undershooting by one pixel."""
    coords = np.argwhere(instance_labels == label).astype(np.float64)
    if coords.shape[0] < 2:
        return None
    theta = np.deg2rad(np.arange(0.0, 180.0, step_deg))
    cos, sin = np.cos(theta), np.sin(theta)
    projections = coords @ np.stack([cos, sin])
    widths = projections.max(axis=0) - projections.min(axis=0) + np.abs(cos) + np.abs(sin)
    return float(widths.min())


def _gaussian_peak(values: np.ndarray) -> float | None:
    """Peak position of a least-squares Gaussian fit to the histogram —
    the OrgSegNet electron-density statistic (Feng et al. 2023, trait 2/3).
    Falls back to the sample mean when the fit does not converge (e.g. a
    uniform distribution has no curvature to fit)."""
    if values.size == 0:
        return None
    try:
        from scipy.optimize import curve_fit

        n_bins = int(np.clip(np.sqrt(values.size), 8, 64))
        hist, edges = np.histogram(values, bins=n_bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        mean0 = float(values.mean())
        std0 = max(float(values.std()), 1e-6)

        def _gauss(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
            return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        fitted, _ = curve_fit(
            _gauss, centers, hist.astype(np.float64), p0=[float(hist.max()), mean0, std0], maxfev=2000
        )
        peak = float(fitted[1])
        return peak if math.isfinite(peak) else None
    except (RuntimeError, ValueError):
        return None


def _collect_objects(
    instance_labels: np.ndarray,
    *,
    cls_idx: int,
    cls_name: str,
    img: np.ndarray | None,
    scale: float | None,
    pixel_size_um: float | None,
    min_area: int,
    label_source: str | None,
    background_mean: float | None,
    background_peak: float | None,
    counts: dict[str, int],
    per_object: list[dict[str, Any]],
) -> None:
    """Append regionprops rows for one instance-labeled class image.

    Metric sources (T-C1): every geometric quantity is a scikit-image
    ``regionprops`` property (never hand-written geometry); intensity
    distribution stats use numpy/scipy on the object's pixels. Definitions
    and ImageJ divergences are documented in
    ``docs/reports/morphology_metrics.md``.
    """
    base_props = (
        "label",
        "area",
        "perimeter",
        "equivalent_diameter",
        "eccentricity",
        "minor_axis_length",
        "major_axis_length",
        "bbox",
        # T-C1 additions (all scikit-image properties):
        "feret_diameter_max",  # maximum caliper diameter (pairwise boundary points)
        "solidity",  # area / convex_area
        "extent",  # area / bounding-box area
        "convex_area",  # area of the convex hull
        "orientation",  # radians, major axis vs image row axis
        "centroid",  # (row, col)
    )
    intensity_props = (
        "intensity_mean",
        "intensity_min",
        "intensity_max",
        "intensity_std",
        "centroid_weighted",
    )
    props = (
        skmeasure.regionprops_table(
            instance_labels,
            intensity_image=img,
            properties=(*base_props, *intensity_props),
        )
        if img is not None
        else skmeasure.regionprops_table(instance_labels, properties=base_props)
    )
    mi_col = "intensity_mean" if img is not None else None
    n = len(props["label"])
    for i in range(n):
        area_px = float(props["area"][i])
        if area_px < min_area:
            continue
        perim = float(props["perimeter"][i])
        ecc = float(props["eccentricity"][i])
        eq_d = float(props["equivalent_diameter"][i])
        minor = float(props["minor_axis_length"][i])
        major = float(props["major_axis_length"][i])
        feret = float(props["feret_diameter_max"][i])
        convex_area = float(props["convex_area"][i])
        bbox_height = float(props["bbox-2"][i] - props["bbox-0"][i])
        bbox_width = float(props["bbox-3"][i] - props["bbox-1"][i])
        circ = (4.0 * math.pi * area_px / (perim**2)) if perim > 0 else 0.0
        aspect = (minor / major) if major > 0 else 0.0
        # ImageJ's AR convention is major/minor (ours above is minor/major);
        # both are recorded so exports match whichever tool users compare to.
        aspect_imagej = (major / minor) if minor > 0 else None
        roundness = (4.0 * area_px / (math.pi * major**2)) if major > 0 else None
        if mi_col is not None:
            mi = props[mi_col][i]
            mi = float(mi) if mi == mi else None  # NaN check
        else:
            mi = None

        instance_label = int(props["label"][i])
        feret_min = _feret_diameter_min(instance_labels, instance_label)
        complexity = _shape_complexity(instance_labels, instance_label)

        # Intensity distribution stats: plain numpy/scipy statistics over the
        # object's pixels (not geometry). Median/skew/kurtosis are not
        # regionprops properties.
        intensity_median = intensity_skew = intensity_kurtosis = None
        raw_integrated_density = mode_intensity = None
        object_peak: float | None = None
        if img is not None:
            from scipy import stats as scipy_stats

            pixels = img[instance_labels == instance_label].astype(np.float64)
            if pixels.size:
                intensity_median = float(np.median(pixels))
                skew = float(scipy_stats.skew(pixels))
                kurt = float(scipy_stats.kurtosis(pixels))
                # undefined (0/0) for constant distributions -> None, never NaN
                # (the JSON result codec requires finite values)
                intensity_skew = skew if math.isfinite(skew) else None
                intensity_kurtosis = kurt if math.isfinite(kurt) else None
                # ImageJ RawIntDen: sum of the object's pixel values
                raw_integrated_density = round(float(pixels.sum()), 3)
                # ImageJ Mode: modal gray value (needs a discrete histogram)
                if img.dtype.kind == "f" or pixels.min() >= 0:
                    counts_hist = np.bincount(
                        np.rint(pixels - pixels.min()).astype(np.int64)
                    )
                    if counts_hist.size:
                        mode_intensity = round(
                            float(counts_hist.argmax() + pixels.min()), 3
                        )
                object_peak = _gaussian_peak(pixels)

        row: dict[str, Any] = {
            "label": int(props["label"][i]),
            "class": cls_idx,
            "class_name": cls_name,
            "area_px": round(area_px, 1),
            # physical columns stay empty without calibration — never a fake 1.0
            "area_um2": round(area_px * scale, 3) if scale else None,
            "perimeter": round(perim, 2),
            "perimeter_um": round(perim * pixel_size_um, 3) if pixel_size_um else None,
            "equivalent_diameter": round(eq_d, 2),
            "equivalent_diameter_um": round(eq_d * pixel_size_um, 3) if pixel_size_um else None,
            "eccentricity": round(ecc, 3),
            "circularity": round(circ, 3),
            "aspect_ratio": round(aspect, 3),
            "aspect_ratio_imagej": round(aspect_imagej, 3) if aspect_imagej else None,
            "roundness": round(roundness, 4) if roundness is not None else None,
            "feret_diameter_max": round(feret, 2),
            "feret_max_um": round(feret * pixel_size_um, 3) if pixel_size_um else None,
            "feret_diameter_min": round(feret_min, 2) if feret_min is not None else None,
            "feret_min_um": round(feret_min * pixel_size_um, 3)
            if feret_min is not None and pixel_size_um
            else None,
            # OrgSegNet trait 1/3: visibility-graph shape complexity
            "shape_complexity": complexity,
            "major_axis_length": round(major, 2),
            "minor_axis_length": round(minor, 2),
            "solidity": round(float(props["solidity"][i]), 4),
            "extent": round(float(props["extent"][i]), 4),
            "convex_area": round(convex_area, 1),
            "orientation": round(float(props["orientation"][i]), 4),
            "bbox": tuple(int(props[f"bbox-{j}"][i]) for j in range(4)),
            "bbox_width": int(bbox_width),
            "bbox_height": int(bbox_height),
            "centroid_y": round(float(props["centroid-0"][i]), 2),
            "centroid_x": round(float(props["centroid-1"][i]), 2),
            "mean_intensity": round(mi, 3) if mi is not None else None,
            "label_source": label_source,
            "pixel_size_um": pixel_size_um,
        }
        if img is not None:
            row.update(
                {
                    "intensity_min": float(props["intensity_min"][i]),
                    "intensity_max": float(props["intensity_max"][i]),
                    "intensity_std": round(float(props["intensity_std"][i]), 4),
                    "intensity_median": intensity_median,
                    "intensity_skew": round(intensity_skew, 4)
                    if intensity_skew is not None
                    else None,
                    "intensity_kurtosis": round(intensity_kurtosis, 4)
                    if intensity_kurtosis is not None
                    else None,
                    # integrated density = area * mean intensity (ImageJ IntDen)
                    "integrated_density": round(area_px * mi, 3) if mi is not None else None,
                    "raw_integrated_density": raw_integrated_density,
                    "mode_intensity": mode_intensity,
                    "centroid_weighted_y": round(float(props["centroid_weighted-0"][i]), 2),
                    "centroid_weighted_x": round(float(props["centroid_weighted-1"][i]), 2),
                }
            )
            # OrgSegNet-style electron density (trait 2/3): TEM brightness has
            # no absolute calibration, so the organelle/background ratio is
            # the comparable quantity. Feng et al. 2023 fit Gaussians to both
            # histograms and take the peak ratio mu_o/mu_b; each side falls
            # back to its mean when the fit does not converge.
            if background_mean is not None and mi is not None and background_mean > 0:
                numerator = object_peak if object_peak is not None else mi
                denominator = (
                    background_peak if background_peak is not None else background_mean
                )
                if denominator > 0:
                    row["electron_density_ratio"] = round(numerator / denominator, 4)
                    row["electron_density_ratio_mean"] = round(mi / background_mean, 4)
        counts[cls_name] += 1
        per_object.append(row)


def _finalize_spatial(
    per_object: list[dict[str, Any]],
    *,
    counts: dict[str, int],
    image_area_px: float,
    pixel_size_um: float | None,
    background_mean: float | None,
) -> dict[str, Any]:
    """Post-measurement spatial/summary metrics (stereology + NND).

    - Nearest-neighbor distance per object (centroid-based, across ALL
      objects regardless of class — the standard organelle-distribution
      statistic in volume-EM spatial analysis).
    - Area fraction (Vv estimate by the Delesse principle: organelle area /
      reference area) per class.
    - Number density per class per mm² when a pixel calibration exists.

    Mutates per-object rows (adds nearest_neighbor_px/um) and returns the
    envelope-level summary fields.
    """
    for row in per_object:
        row["nearest_neighbor_px"] = None
        row["nearest_neighbor_um"] = None
    nnd_summary: dict[str, Any] = {}
    if len(per_object) >= 2:
        from scipy.spatial import cKDTree

        points = np.array(
            [[row["centroid_y"], row["centroid_x"]] for row in per_object], dtype=np.float64
        )
        distances, _ = cKDTree(points).query(points, k=2)
        nnd = distances[:, 1]
        for row, d in zip(per_object, nnd, strict=True):
            row["nearest_neighbor_px"] = round(float(d), 2)
            if pixel_size_um:
                row["nearest_neighbor_um"] = round(float(d) * pixel_size_um, 3)
        nnd_summary = {
            "mean_px": round(float(nnd.mean()), 2),
            "min_px": round(float(nnd.min()), 2),
            "max_px": round(float(nnd.max()), 2),
        }

    area_fraction_pct: dict[str, float] = {}
    number_density_per_mm2: dict[str, float | None] = {}
    image_area_um2 = image_area_px * pixel_size_um**2 if pixel_size_um else None
    for cls_name, _ in counts.items():
        cls_rows = [r for r in per_object if r["class_name"] == cls_name]
        cls_area = sum(r["area_px"] for r in cls_rows)
        area_fraction_pct[cls_name] = (
            round(cls_area / image_area_px * 100.0, 3) if image_area_px > 0 else 0.0
        )
        number_density_per_mm2[cls_name] = (
            round(len(cls_rows) / image_area_um2 * 1e6, 3)
            if image_area_um2 and image_area_um2 > 0
            else None
        )
    return {
        "image_area_px": image_area_px,
        "area_fraction_pct": area_fraction_pct,
        "number_density_per_mm2": number_density_per_mm2,
        "nearest_neighbor_px": nnd_summary,
        "background_mean_intensity": background_mean,
    }


def measure(
    label_map: str | Path | np.ndarray,
    *,
    intensity_image: str | Path | np.ndarray | None = None,
    pixel_size_um: float | None = None,
    min_area: int = 10,
    watershed_classes: tuple[int, ...] = (2,),
    class_names: tuple[str, ...] = ORGSEG_CLASSES,
    instance_map: bool = False,
    label_source: str | None = None,
    backend: str = "native",
) -> dict[str, Any]:
    """Measure per-object morphometrics from a semantic-segmentation label map.

    Parameters
    ----------
    label_map
        Integer label map (array or path to a label PNG/TIF). Pixel values are
        class indices (0=background, 1=Chloroplast, ... for OrgSegNet output),
        or instance ids when ``instance_map`` is True.
    intensity_image
        Optional original EM image (array or path) for mean-intensity /
        electron-density measurements per object.
    pixel_size_um
        Physical pixel size in µm/pixel. When given, area is also reported in
        µm² (``area_um2 = area_px * pixel_size_um**2``).
    min_area
        Minimum object area in pixels; smaller objects (noise) are dropped.
    watershed_classes
        Class indices for which touching objects are split via distance-
        transform watershed before measurement. Default: ``(2,)`` (Mitochondria
        only), since mitochondria are small and frequently touching; chloroplasts
        / nuclei / vacuoles are larger and rarely need splitting.
    class_names
        Class-index -> name mapping (default OrgSegNet 5-class). With
        ``instance_map=True`` only ``class_names[1]`` is used, as the generic
        object class name.
    instance_map
        When True, the label map carries per-instance ids (0 = background,
        1..N = objects), as produced by micro-SAM's automatic instance
        segmentation. Every id is already one separated object, so no
        connected-component or watershed split is applied, and every object
        is reported under the single generic class ``class_names[1]``
        (instance segmentation carries no semantic class).
    label_source
        Provenance of the label map being measured, recorded verbatim on
        every per-object row and on the envelope (e.g. ``"model"`` for a
        pure model output, ``"human_corrected"`` for a label map that went
        through :func:`.correction.correct`). ``None`` means unknown and is
        recorded as such — never guessed.

    Returns
    -------
    dict
        ``per_object`` (list of dicts), ``counts`` (per-class int), ``n_total``,
        ``pixel_size_um``. Each per-object dict has: label, class, class_name,
        area_px, area_um2 (or None), perimeter, equivalent_diameter, eccentricity,
        circularity, aspect_ratio, mean_intensity (or None), bbox,
        label_source.
    """
    if backend not in {"native", "auto", "qupath"}:
        raise OrganelleInputError(
            code="morphology.unknown_measure_backend",
            message=f"unknown measure backend: {backend!r}",
            details={"backend": backend, "registered": ["native", "qupath"]},
        )
    if backend == "qupath":
        # Optional bridge backend (T-C2). Deterministic rule: only an explicit
        # request reaches QuPath; bridge failures propagate as structured
        # errors, never a silent native fallback. The project directory
        # derives from the label map location (no caller-chosen output path).
        from .qupath_bridge import measure_via_qupath

        if isinstance(label_map, np.ndarray):
            raise OrganelleInputError(
                code="qupath.requires_label_map_path",
                message="the qupath backend needs the label map as a file path",
                details={"backend": "qupath"},
            )
        label_path = Path(label_map)
        return measure_via_qupath(
            label_path,
            project_dir=label_path.parent / f"{label_path.stem}_qupath_project",
            min_area=min_area,
        )
    # backend="auto": native is the default and only implicit choice — the
    # bridge is opt-in by explicit request, never by what happens to be
    # installed.

    lab = _load_image(label_map)
    if lab.ndim == 3:
        # A colorized label PNG: reduce to its class index by summing channels.
        # OrgSegNet palette is unique per class, but a raw integer label PNG
        # loaded by PIL may come back as RGB; take the first channel if it's a
        # single-value image, else assume class index = R + 256*G + 65536*B is
        # wrong -- fall back to the max across channels.
        lab = lab[..., 0] if lab.shape[-1] == 3 else lab
    lab = lab.astype(np.int32)

    img = None
    background_mean: float | None = None
    background_peak: float | None = None
    if intensity_image is not None:
        img = _load_image(intensity_image)
        if img.ndim == 3:
            img = img.mean(axis=-1)  # grayscale electron density
        img = img.astype(np.float64)
        bg = img[lab == 0]
        if bg.size:
            background_mean = float(bg.mean())
            background_peak = _gaussian_peak(bg)

    scale = float(pixel_size_um) ** 2 if pixel_size_um else None
    per_object: list[dict[str, Any]] = []
    counts: dict[str, int] = {name: 0 for name in class_names if name != "background"}

    if instance_map:
        # Pre-segmented instance ids: each nonzero id is one object of the
        # single generic class; never re-split, never invent semantics.
        generic_class = class_names[1] if len(class_names) > 1 else "object"
        counts = {generic_class: 0}
        instance_labels = lab
        if int(instance_labels.max()) == 0:
            return {
                "per_object": [],
                "counts": counts,
                "n_total": 0,
                "n_classes_present": 0,
                "pixel_size_um": pixel_size_um,
                "label_source": label_source,
                **_finalize_spatial(
                    [],
                    counts=counts,
                    image_area_px=float(lab.shape[0] * lab.shape[1]),
                    pixel_size_um=pixel_size_um,
                    background_mean=background_mean,
                ),
            }
        _collect_objects(
            instance_labels,
            cls_idx=1,
            cls_name=generic_class,
            img=img,
            scale=scale,
            pixel_size_um=pixel_size_um,
            min_area=min_area,
            label_source=label_source,
            background_mean=background_mean,
            background_peak=background_peak,
            counts=counts,
            per_object=per_object,
        )
        return {
            "per_object": per_object,
            "counts": counts,
            "n_total": len(per_object),
            "n_classes_present": sum(1 for v in counts.values() if v > 0),
            "pixel_size_um": pixel_size_um,
            "label_source": label_source,
            **_finalize_spatial(
                per_object,
                counts=counts,
                image_area_px=float(lab.shape[0] * lab.shape[1]),
                pixel_size_um=pixel_size_um,
                background_mean=background_mean,
            ),
        }

    for cls_idx in range(1, len(class_names)):
        cls_name = class_names[cls_idx]
        mask = lab == cls_idx
        if not mask.any():
            continue

        if cls_idx in watershed_classes:
            instance_labels = _split_touching(mask)
        else:
            instance_labels = skmeasure.label(mask).astype(np.int32)

        _collect_objects(
            instance_labels,
            cls_idx=cls_idx,
            cls_name=cls_name,
            img=img,
            scale=scale,
            pixel_size_um=pixel_size_um,
            min_area=min_area,
            label_source=label_source,
            background_mean=background_mean,
            background_peak=background_peak,
            counts=counts,
            per_object=per_object,
        )

    return {
        "per_object": per_object,
        "counts": counts,
        "n_total": len(per_object),
        "n_classes_present": sum(1 for v in counts.values() if v > 0),
        "pixel_size_um": pixel_size_um,
        "label_source": label_source,
        **_finalize_spatial(
            per_object,
            counts=counts,
            image_area_px=float(lab.shape[0] * lab.shape[1]),
            pixel_size_um=pixel_size_um,
            background_mean=background_mean,
        ),
    }


def summarize(
    per_object: list[Mapping[str, str | int | float | bool | None]]
    | Mapping[str, list[Mapping[str, str | int | float | bool | None]]],
    *,
    by_class: bool = True,
) -> dict[str, Any]:
    """Aggregate per-object measurements into per-class summary statistics.

    Parameters
    ----------
    per_object
        Either the list of per-object dicts (from :func:`measure`) or the full
        dict returned by :func:`measure` (its ``per_object`` key is used).
    by_class
        When True, stats are grouped by ``class_name``; when False, a single
        all-objects summary is returned.

    Returns
    -------
    dict
        ``per_class`` (or ``all``): {class_name: {count, area_mean,
        area_std, perimeter_mean, circularity_mean, ...}}.
    """
    if isinstance(per_object, dict):
        per_object = per_object.get("per_object", [])
    if not per_object:
        return {"per_class": {}, "all": {"count": 0}}

    def _stats(values: list[float]) -> dict[str, float]:
        arr = np.array(values, dtype=np.float64)
        return {
            "mean": round(float(arr.mean()), 3) if arr.size else 0.0,
            "std": round(float(arr.std(ddof=1)), 3) if arr.size > 1 else 0.0,
            "min": round(float(arr.min()), 3) if arr.size else 0.0,
            "max": round(float(arr.max()), 3) if arr.size else 0.0,
        }

    fields = (
        "area_px",
        "area_um2",
        "perimeter",
        "equivalent_diameter",
        "eccentricity",
        "circularity",
        "aspect_ratio",
        "mean_intensity",
        "electron_density_ratio",
        "feret_diameter_max",
        "solidity",
        "extent",
        "orientation",
        "integrated_density",
        "nearest_neighbor_px",
        "nearest_neighbor_um",
    )

    def _group(objs: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"count": len(objs)}
        for f in fields:
            vals = [o[f] for o in objs if o.get(f) is not None]
            if vals:
                out[f] = _stats(vals)
        return out

    result: dict[str, Any] = {}
    if by_class:
        classes: dict[str, list[dict[str, Any]]] = {}
        for o in per_object:
            classes.setdefault(o["class_name"], []).append(o)
        result["per_class"] = {c: _group(objs) for c, objs in sorted(classes.items())}
    result["all"] = _group(per_object)
    return result


def overlay(
    image: str | Path | np.ndarray,
    label_map: str | Path | np.ndarray,
    *,
    alpha: float = 0.5,
    palette: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Render a colorized segmentation overlay on the source image.

    Parameters
    ----------
    image
        Source EM image (array or path).
    label_map
        Integer label map (array or path).
    alpha
        Overlay opacity (0 = source only, 1 = mask only).
    palette
        Class-index -> RGB. Defaults to :data:`ORGSEG_PALETTE`.

    Returns
    -------
    np.ndarray
        The overlay as a uint8 RGB array (H, W, 3). Compute-only; persist it
        through an explicit writer (e.g. ``ov.write``) if a file is needed.
    """
    if palette is None:
        palette = ORGSEG_PALETTE
    img = _load_image(image).astype(np.float64)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    # Normalize source to 0-255 for display.
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / (hi - lo + 1e-9) * 255.0

    lab = _load_image(label_map)
    if lab.ndim == 3:
        lab = lab[..., 0]
    lab = lab.astype(np.int32)

    color = np.zeros((*lab.shape, 3), dtype=np.float64)
    for cls_idx, rgb in palette.items():
        color[lab == cls_idx] = rgb

    out = img * (1.0 - alpha) + color * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def write_csv(
    per_object: list[dict[str, Any]] | dict[str, Any],
    output: str | Path,
    *,
    fields: tuple[str, ...] = (
        "label",
        "class",
        "class_name",
        "area_px",
        "area_um2",
        "perimeter",
        "perimeter_um",
        "equivalent_diameter",
        "eccentricity",
        "circularity",
        "aspect_ratio",
        "feret_diameter_max",
        "feret_max_um",
        "major_axis_length",
        "minor_axis_length",
        "solidity",
        "extent",
        "convex_area",
        "orientation",
        "bbox_width",
        "bbox_height",
        "centroid_y",
        "centroid_x",
        "mean_intensity",
        "intensity_min",
        "intensity_max",
        "intensity_std",
        "intensity_median",
        "intensity_skew",
        "intensity_kurtosis",
        "integrated_density",
        "raw_integrated_density",
        "mode_intensity",
        "electron_density_ratio",
        "electron_density_ratio_mean",
        "nearest_neighbor_px",
        "nearest_neighbor_um",
        "shape_complexity",
        "feret_diameter_min",
        "feret_min_um",
        "aspect_ratio_imagej",
        "roundness",
        "centroid_weighted_y",
        "centroid_weighted_x",
        "pixel_size_um",
        "label_source",
    ),
) -> Path:
    """Write per-object measurements to a CSV file (one object per row)."""
    if isinstance(per_object, dict):
        per_object = per_object.get("per_object", [])
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        for o in per_object:
            row = {f: o.get(f, "") for f in fields}
            w.writerow(row)
    return out
