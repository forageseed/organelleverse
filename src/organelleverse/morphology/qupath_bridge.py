"""Optional QuPath measurement backend for the morphology suite (T-C2).

Bridge: **paquo** (JPype, in-process JVM) — the T-C0 feasibility choice
(`docs/reports/qupath_bridge_feasibility.md`). QuPath supplies the
measurement口径; this module never adjusts numbers to match.

Fail-closed rules (task brief, one per row):

| failure | behavior here |
|---|---|
| JVM / paquo missing | ``qupath.bridge_unavailable`` naming what is missing |
| QuPath version outside the declared range | ``qupath.version_unsupported`` — never "try anyway" |
| project file locked (paquo readonly/OS error) | ``qupath.project_locked`` — no waiting, no forcing |
| bridge dies mid-run | the exception propagates into a failed result; partial results are never reported as success |

Explicitly requesting ``backend="qupath"`` without the bridge fails;
nothing ever falls back to native silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.errors import OrganelleDependencyError

__all__ = [
    "SUPPORTED_QUPATH_RANGE",
    "QUPath_BRIDGE_INFO",
    "check_qupath_bridge",
    "measure_via_qupath",
]

#: Verified against QuPath 0.5.1 (T-C0). Anything outside fails closed.
SUPPORTED_QUPATH_RANGE = ">=0.5,<0.6"

QUPath_BRIDGE_INFO: dict[str, str] = {
    "bridge": "paquo (JPype, in-process JVM)",
    "qupath_versions": SUPPORTED_QUPATH_RANGE,
    "install": "pip install paquo && paquo get_qupath 0.5.1 --install-path <dir>",
    "note": (
        "Optional measurement backend: QuPath-calibrated metrics via paquo. "
        "JVM stays out of core dependencies; the bridge runs in the worker "
        "process so a JVM crash fails the run instead of the app."
    ),
}


def check_qupath_bridge() -> dict[str, Any]:
    """Probe the bridge: paquo importable, JVM starts, QuPath version in range.

    Returns {available, bridge, qupath_version, reason} — ``available`` is
    False with a machine-readable reason, never a guess.
    """
    try:
        import paquo  # noqa: F401
    except ImportError:
        return {
            "available": False,
            "bridge": "paquo",
            "qupath_version": None,
            "reason": "paquo not installed",
        }
    try:
        from paquo.java import qupath_version
    except Exception as error:  # JVM start failure surfaces here
        return {
            "available": False,
            "bridge": "paquo",
            "qupath_version": None,
            "reason": f"JVM/QuPath failed to start: {error}",
        }
    version = str(qupath_version) if qupath_version else None
    if version is None:
        return {
            "available": False,
            "bridge": "paquo",
            "qupath_version": None,
            "reason": "no QuPath installation found",
        }
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    if Version(version) not in SpecifierSet(SUPPORTED_QUPATH_RANGE):
        return {
            "available": False,
            "bridge": "paquo",
            "qupath_version": version,
            "reason": f"QuPath {version} outside the declared range {SUPPORTED_QUPATH_RANGE}",
        }
    return {
        "available": True,
        "bridge": "paquo",
        "qupath_version": version,
        "reason": "",
    }


def measure_via_qupath(
    label_map: str | Path,
    *,
    project_dir: str | Path,
    min_area: int = 10,
) -> dict[str, Any]:
    """Measure a label map through QuPath and return native-schema rows.

    The output uses the same row keys as :func:`.measure` (schema parity);
    metrics QuPath does not compute the same way are ``None`` and the
    numeric divergences are recorded in ``docs/reports/morphology_metrics.md``.
    Errors from the bridge (locked project, dead JVM) propagate — the caller
    marks the run failed; partial results are never reported as success.
    """
    status = check_qupath_bridge()
    if not status["available"]:
        raise OrganelleDependencyError(
            code="qupath.bridge_unavailable",
            message=f"QuPath bridge unavailable: {status['reason']}",
            details={
                "bridge": status["bridge"],
                "qupath_version": status["qupath_version"],
                "install": QUPath_BRIDGE_INFO["install"],
                "supported_versions": SUPPORTED_QUPATH_RANGE,
            },
        )

    import numpy as np
    from skimage import measure as skmeasure

    from .measure import _load_image

    lab = _load_image(label_map)
    if lab.ndim == 3:
        lab = lab[..., 0]
    lab = np.asarray(lab).astype(np.int64)

    from paquo.projects import QuPathProject
    from shapely.geometry import Polygon

    rows: list[dict[str, Any]] = []
    project = Path(project_dir)
    project.mkdir(parents=True, exist_ok=True)
    try:
        with QuPathProject(str(project), mode="w") as qp:
            entry = qp.add_image(str(Path(label_map).resolve()))
            for obj_id in sorted(int(v) for v in np.unique(lab) if v != 0):
                mask = lab == obj_id
                if int(mask.sum()) < min_area:
                    continue
                contours = skmeasure.find_contours(mask.astype(np.uint8), 0.5)
                if not contours:
                    continue
                rings = [Polygon([(c, r) for r, c in ct]) for ct in contours if len(ct) >= 4]
                if not rings:
                    continue
                outer = max(rings, key=lambda p: p.area)
                holes = [
                    p for p in rings if p is not outer and p.representative_point().within(outer)
                ]
                poly = Polygon(outer.exterior.coords, [h.exterior.coords for h in holes])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    continue
                if poly.geom_type != "Polygon":
                    poly = max(poly.geoms, key=lambda g: g.area)
                entry.hierarchy.add_detection(poly)

            server = entry.java_object.readImageData().getServer()
            cal = server.getPixelCalibration()
            import jpype  # paquo runs inside JPype already
            from paquo.java import JClass

            ArrayList = JClass("java.util.ArrayList")
            ObjectMeasurements = JClass("qupath.lib.analysis.features.ObjectMeasurements")
            ShapeFeatures = JClass("qupath.lib.analysis.features.ObjectMeasurements$ShapeFeatures")
            java_list = ArrayList()
            for det in entry.hierarchy.detections:
                java_list.add(det.java_object)
            feats = jpype.JArray(ShapeFeatures)(
                [
                    ShapeFeatures.AREA,
                    ShapeFeatures.CIRCULARITY,
                    ShapeFeatures.LENGTH,
                    ShapeFeatures.SOLIDITY,
                    ShapeFeatures.MAX_DIAMETER,
                    ShapeFeatures.MIN_DIAMETER,
                ]
            )
            ObjectMeasurements.addShapeMeasurements(java_list, cal, feats)
            entry.save()
            for det in entry.hierarchy.detections:
                ms = dict(det.measurements)
                rows.append(
                    {
                        "label": len(rows) + 1,
                        "class": 1,
                        "class_name": "object",
                        "area_px": ms.get("Area px^2"),
                        "area_um2": None,
                        # QuPath "Length" is polygon-edge length — a different
                        # definition from skimage's pixel-contour perimeter
                        # (recorded in morphology_metrics.md).
                        "perimeter": ms.get("Length px"),
                        "perimeter_um": None,
                        "equivalent_diameter": None,
                        "equivalent_diameter_um": None,
                        "eccentricity": None,
                        "circularity": ms.get("Circularity"),
                        "aspect_ratio": None,
                        "aspect_ratio_imagej": None,
                        "roundness": None,
                        "feret_diameter_max": ms.get("Max diameter px"),
                        "feret_max_um": None,
                        "feret_diameter_min": ms.get("Min diameter px"),
                        "feret_min_um": None,
                        # OrgSegNet visibility-graph trait is native-only
                        "shape_complexity": None,
                        "major_axis_length": None,
                        "minor_axis_length": ms.get("Min diameter px"),
                        "solidity": ms.get("Solidity"),
                        "extent": None,
                        "convex_area": None,
                        "orientation": None,
                        "bbox": None,
                        "bbox_width": None,
                        "bbox_height": None,
                        "centroid_y": None,
                        "centroid_x": None,
                        "mean_intensity": None,
                        "intensity_min": None,
                        "intensity_max": None,
                        "intensity_std": None,
                        "intensity_median": None,
                        "intensity_skew": None,
                        "intensity_kurtosis": None,
                        "integrated_density": None,
                        "raw_integrated_density": None,
                        "mode_intensity": None,
                        "electron_density_ratio": None,
                        "electron_density_ratio_mean": None,
                        # QuPath carries no cross-object spatial pass, so the
                        # nearest-neighbor columns stay empty here.
                        "nearest_neighbor_px": None,
                        "nearest_neighbor_um": None,
                        "centroid_weighted_y": None,
                        "centroid_weighted_x": None,
                        "label_source": None,
                        "pixel_size_um": None,
                    }
                )
    except OSError as error:
        # paquo surfaces a locked/unwritable project as OSError
        raise OrganelleDependencyError(
            code="qupath.project_locked",
            message=f"QuPath project is locked or unwritable: {project}",
            details={"project": str(project), "reason": str(error)},
        ) from error

    return {
        "per_object": rows,
        "counts": {"object": len(rows)},
        "n_total": len(rows),
        "n_classes_present": 1 if rows else 0,
        "pixel_size_um": None,
        "label_source": None,
        "backend": "qupath",
        "qupath_version": status["qupath_version"],
        "bridge": status["bridge"],
    }
