"""Group comparison of organelle morphometrics with the biological replication kept.

Objects are nested in images, images in samples (the biological unit: a plant, an independently
grown culture or an independently treated sample) and samples in groups (genotype, treatment).
Organelles and image fields are subsamples, never independent replicates, so groups are compared
on one value per sample:

* object metrics (size, shape, network): the median over the sample's objects pooled across its
  images (organelle sizes are right-skewed, the median is stable);
* image metrics (objects per image, objects per 100 µm², area fraction, share of network objects):
  the mean over the sample's images.

For each group against the reference group and each metric: the difference and ratio of the group
means of the sample values, Welch's t interval and p value, Hedges' g, and Holm's correction across
the metrics compared for that group. With fewer than two samples in either group a comparison is
descriptive only. Size and shape metrics skip objects that touch the image border when
``exclude_edge`` is set (their outline is cut); counts always include them.

Mean intensity and the electron-density ratio depend on staining, section thickness and
acquisition settings, so they are compared only when ``compare_intensity`` is set by someone who
knows the images were acquired alike.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..core.errors import OrganelleInputError

__all__ = ["OBJECT_METRICS", "IMAGE_METRICS", "compare_groups", "write_tables", "markdown_tables"]

#: key -> (label, unit, uncalibrated key, uncalibrated unit). Field names of measure() rows and
#: mito_metrics.annotate_objects.
OBJECT_METRICS: dict[str, tuple[str, str, str | None, str]] = {
    "area_um2": ("Area", "µm²", "area_px", "px²"),
    "equivalent_diameter_um": ("Equivalent diameter", "µm", "equivalent_diameter", "px"),
    "length_rect_um": ("Length (minimum rectangle)", "µm", "length_rect", "px"),
    "width_rect_um": ("Width (minimum rectangle)", "µm", "width_rect", "px"),
    "feret_max_um": ("Maximum Feret diameter", "µm", "feret_diameter_max", "px"),
    "aspect_ratio_imagej": ("Aspect ratio (major/minor axis)", "", "aspect_ratio_imagej", ""),
    "circularity": ("Circularity (4πA/P²)", "", "circularity", ""),
    "solidity": ("Solidity (area/convex area)", "", "solidity", ""),
    "skeleton_length_um": ("Skeleton length", "µm", "skeleton_length", "px"),
    "junctions": ("Skeleton junctions per object", "", "junctions", ""),
}
INTENSITY_METRICS: dict[str, tuple[str, str, str | None, str]] = {
    "mean_intensity": ("Mean intensity", "a.u.", "mean_intensity", "a.u."),
    "electron_density_ratio": ("Electron-density ratio (object/background)", "", "electron_density_ratio", ""),
}
IMAGE_METRICS: dict[str, tuple[str, str]] = {
    "objects_per_image": ("Objects per image", "count"),
    "objects_per_100um2": ("Objects per 100 µm² analysed", "per 100 µm²"),
    "area_fraction_pct": ("Area fraction", "%"),
    "network_fraction_pct": ("Objects with a branched skeleton", "%"),
}
_ALPHA = 0.05


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _touches_edge(row: Mapping[str, Any], shape: tuple[int, int] | None) -> bool:
    bbox = row.get("bbox")
    if shape is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    r0, c0, r1, c1 = (int(v) for v in bbox)
    return r0 <= 0 or c0 <= 0 or r1 >= shape[0] or c1 >= shape[1]


def _shape(image: Mapping[str, Any]) -> tuple[int, int] | None:
    shape = image.get("image_shape")
    if isinstance(shape, (list, tuple)) and len(shape) == 2 and all(_number(v) for v in shape):
        return int(shape[0]), int(shape[1])
    return None


def _describe(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = statistics.fmean(values) if n else None
    sd = statistics.stdev(values) if n > 1 else None
    half = None
    if n > 1:
        from scipy import stats
        half = float(stats.t.ppf(1 - _ALPHA / 2, n - 1)) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd,
            "ci95_low": None if half is None else mean - half, "ci95_high": None if half is None else mean + half,
            "min": min(values) if n else None, "max": max(values) if n else None}


def _welch(a: list[float], b: list[float]) -> dict[str, Any]:
    """Welch's t test of mean(a) - mean(b) with its 95 % interval and Hedges' g."""
    from scipy import stats

    na, nb = len(a), len(b)
    diff = statistics.fmean(a) - statistics.fmean(b)
    out: dict[str, Any] = {"difference": diff, "ci95_low": None, "ci95_high": None, "t": None, "df": None,
                           "p": None, "hedges_g": None}
    if na < 2 or nb < 2:
        return out
    va, vb = statistics.variance(a), statistics.variance(b)
    se2 = va / na + vb / nb
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled > 0:
        out["hedges_g"] = diff / math.sqrt(pooled) * (1 - 3 / (4 * (na + nb) - 9))
    if se2 <= 0:
        return out
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t = diff / math.sqrt(se2)
    half = float(stats.t.ppf(1 - _ALPHA / 2, df)) * math.sqrt(se2)
    out.update({"ci95_low": diff - half, "ci95_high": diff + half, "t": t, "df": df,
                "p": float(2 * stats.t.sf(abs(t), df))})
    return out


def _holm(rows: list[dict[str, Any]]) -> None:
    tested = sorted((row for row in rows if row["p"] is not None), key=lambda row: row["p"])
    m, running = len(tested), 0.0
    for rank, row in enumerate(tested):
        running = max(running, min(1.0, (m - rank) * row["p"]))
        row["p_holm"] = running


def compare_groups(images: Sequence[Mapping[str, Any]], *, reference_group: str | None = None,
                   exclude_edge: bool = False, compare_intensity: bool = False) -> dict[str, Any]:
    """Image, sample and group tables of organelle morphometrics, and each group against the reference.

    Each item of ``images`` holds ``image`` (an id), ``group``, ``objects`` (measure() rows already
    restricted to the objects to count, e.g. those a person accepted), and optionally ``sample`` (the
    biological unit; the image itself when missing, which is recorded as an assumption),
    ``pixel_size_um``, ``image_shape`` (height, width in pixels) and ``analyzed_area_um2`` (defaults to
    the whole image when shape and pixel size are known).
    """
    if not images:
        raise OrganelleInputError(code="morphology.compare_empty", message="no images to compare")
    warnings: list[str] = []
    sample_group: dict[str, str] = {}
    seen: set[str] = set()
    for item in images:
        image, group = str(item.get("image") or ""), str(item.get("group") or "")
        if not image or not group:
            raise OrganelleInputError(code="morphology.compare_unlabelled",
                                      message="every image needs an id and a group", details={"image": image})
        if image in seen:
            raise OrganelleInputError(code="morphology.compare_duplicate", message=f"image {image} is listed twice")
        seen.add(image)
        sample = str(item.get("sample") or image)
        if sample_group.setdefault(sample, group) != group:
            raise OrganelleInputError(code="morphology.compare_sample_in_two_groups",
                                      message=f"sample {sample} is in groups {sample_group[sample]} and {group}")
    if all(not item.get("sample") for item in images):
        warnings.append("No sample (plant) was given for any image, so each image is treated as an independent "
                        "biological replicate; images from the same plant would make n too large.")
    elif any(not item.get("sample") for item in images):
        warnings.append("Some images have no sample and are treated as their own sample.")

    groups = list(dict.fromkeys(str(item["group"]) for item in images))
    reference = reference_group if reference_group is not None else groups[0]
    if reference not in groups:
        raise OrganelleInputError(code="morphology.compare_reference_missing",
                                  message=f"reference group {reference} has no images", details={"groups": groups})

    calibrated = all(_number(item.get("pixel_size_um")) for item in images)
    if not calibrated:
        warnings.append("Not every image has a pixel size, so sizes are compared in pixels; pixel values are "
                        "comparable only between images taken at the same magnification.")
    sizes = {_number(item.get("pixel_size_um")) for item in images}
    if calibrated and len(sizes) > 1:
        warnings.append(f"Pixel sizes differ between images ({', '.join(f'{s:g}' for s in sorted(sizes))} µm); "
                        "sizes were converted per image before aggregation.")

    catalogue = dict(OBJECT_METRICS)
    if compare_intensity:
        catalogue.update(INTENSITY_METRICS)
        warnings.append("Intensity metrics are compared on the user's statement that staining and acquisition "
                        "were the same for all images.")
    keys = {name: (name if calibrated else spec[2], spec[0], spec[1] if calibrated else spec[3])
            for name, spec in catalogue.items() if (name if calibrated else spec[2])}

    image_rows: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, list[float]]] = {}
    edge_total = 0
    for item in images:
        image, sample = str(item["image"]), str(item.get("sample") or item["image"])
        objects = [row for row in item.get("objects") or [] if isinstance(row, Mapping)]
        shape, pixel = _shape(item), _number(item.get("pixel_size_um"))
        edge = [row for row in objects if _touches_edge(row, shape)]
        edge_total += len(edge)
        measured = [row for row in objects if not (exclude_edge and _touches_edge(row, shape))]
        area = _number(item.get("analyzed_area_um2"))
        if area is None and shape and pixel:
            area = shape[0] * shape[1] * pixel * pixel
        row: dict[str, Any] = {"image": image, "sample": sample, "group": str(item["group"]),
                               "objects": len(objects), "edge_objects": len(edge),
                               "analyzed_area_um2": None if area is None else round(area, 4),
                               "objects_per_image": len(objects)}
        if area:
            row["objects_per_100um2"] = 100.0 * len(objects) / area
            covered = [_number(o.get("area_um2")) for o in objects]
            if all(v is not None for v in covered):
                row["area_fraction_pct"] = 100.0 * sum(covered) / area
        classes = [o.get("morphology_class") for o in objects if o.get("morphology_class")]
        if classes:
            row["network_fraction_pct"] = 100.0 * sum(c == "network" for c in classes) / len(classes)
        for name, (key, _, _) in keys.items():
            values = [v for o in measured if (v := _number(o.get(key))) is not None]
            pooled.setdefault(sample, {}).setdefault(name, []).extend(values)
            row[name] = statistics.median(values) if values else None
        image_rows.append(row)
    if edge_total:
        warnings.append(f"{edge_total} object(s) touch the image border; " + (
            "they are counted but left out of size and shape metrics." if exclude_edge else
            "their size is underestimated (set exclude_edge to leave them out of size and shape)."))

    metric_names = [name for name in keys if any(row.get(name) is not None for row in image_rows)]
    image_metric_names = [name for name in IMAGE_METRICS if any(row.get(name) is not None for row in image_rows)]
    sample_rows: list[dict[str, Any]] = []
    for sample, group in sample_group.items():
        mine = [row for row in image_rows if row["sample"] == sample]
        row = {"sample": sample, "group": group, "images": len(mine), "objects": sum(r["objects"] for r in mine)}
        for name in image_metric_names:
            values = [r[name] for r in mine if r.get(name) is not None]
            row[name] = statistics.fmean(values) if values else None
        for name in metric_names:
            values = pooled.get(sample, {}).get(name, [])
            row[name] = statistics.median(values) if values else None
        sample_rows.append(row)

    all_metrics = image_metric_names + metric_names
    group_rows: list[dict[str, Any]] = []
    for group in groups:
        mine = [row for row in sample_rows if row["group"] == group]
        entry = {"group": group, "samples": len(mine), "images": sum(r["images"] for r in mine),
                 "objects": sum(r["objects"] for r in mine), "metrics": {}}
        for name in all_metrics:
            values = [r[name] for r in mine if r.get(name) is not None]
            entry["metrics"][name] = _describe(values)
        group_rows.append(entry)
        if len(mine) < 2:
            warnings.append(f"Group {group} has {len(mine)} sample(s): its values are descriptive and no test "
                            "involving it is possible.")

    comparisons: list[dict[str, Any]] = []
    for group in groups:
        if group == reference:
            continue
        family = []
        for name in all_metrics:
            a = [r[name] for r in sample_rows if r["group"] == group and r.get(name) is not None]
            b = [r[name] for r in sample_rows if r["group"] == reference and r.get(name) is not None]
            if not a or not b:
                continue
            test = _welch(a, b)
            mean_b = statistics.fmean(b)
            family.append({"metric": name, "group": group, "reference": reference, "n": len(a), "n_reference": len(b),
                           "mean": statistics.fmean(a), "mean_reference": mean_b,
                           "ratio": statistics.fmean(a) / mean_b if mean_b else None, **test, "p_holm": None,
                           "status": "tested" if test["p"] is not None else "descriptive_only"})
        _holm(family)
        comparisons.extend(family)

    definitions = {name: {"label": keys[name][1], "unit": keys[name][2], "field": keys[name][0], "level": "object",
                          "sample_value": "median of the sample's objects"} for name in metric_names}
    definitions.update({name: {"label": IMAGE_METRICS[name][0], "unit": IMAGE_METRICS[name][1], "field": name,
                               "level": "image", "sample_value": "mean of the sample's images"}
                        for name in image_metric_names})
    return {"unit": "sample", "reference_group": reference, "groups": groups, "calibrated": calibrated,
            "exclude_edge": exclude_edge, "compare_intensity": compare_intensity,
            "metrics": definitions, "image_table": image_rows, "sample_table": sample_rows,
            "group_table": group_rows, "comparisons": comparisons, "warnings": warnings,
            "method": ("Sample-level values (object metrics: median of pooled objects; image metrics: mean of "
                       "images); groups compared with Welch's t test on sample values, 95 % CI of the difference, "
                       "Hedges' g, Holm correction across metrics within each group comparison.")}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value and (abs(value) < 10 ** -digits or abs(value) >= 10 ** 6):
            return f"{value:.{digits}g}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def write_tables(result: Mapping[str, Any], out_dir: str | Path) -> dict[str, str]:
    """Write the image, sample, group and comparison tables of compare_groups() as CSV files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    group_rows = []
    for entry in result["group_table"]:
        for name, d in entry["metrics"].items():
            group_rows.append({"group": entry["group"], "samples": entry["samples"], "images": entry["images"],
                               "objects": entry["objects"], "metric": name, "unit": result["metrics"][name]["unit"],
                               **d})
    tables = {"images.csv": result["image_table"], "samples.csv": result["sample_table"],
              "groups.csv": group_rows, "comparisons.csv": result["comparisons"]}
    paths = {}
    for name, rows in tables.items():
        fields = list(dict.fromkeys(key for row in rows for key in row)) or ["empty"]
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        paths[name] = str(out / name)
    return paths


def markdown_tables(result: Mapping[str, Any]) -> str:
    """Group summary and comparison tables of compare_groups() as Markdown."""
    metrics = result["metrics"]
    groups = result["group_table"]
    lines = ["| Metric | Unit | " + " | ".join(f"{g['group']} (n={g['samples']})" for g in groups) + " |",
             "|---|---|" + "---|" * len(groups)]
    for name, spec in metrics.items():
        cells = []
        for g in groups:
            d = g["metrics"].get(name) or {}
            cells.append("" if d.get("mean") is None else
                         _fmt(d["mean"]) + ("" if d.get("sd") is None else f" ± {_fmt(d['sd'])}"))
        lines.append(f"| {spec['label']} | {spec['unit']} | " + " | ".join(cells) + " |")
    text = "Mean ± SD of sample values\n\n" + "\n".join(lines)
    if result["comparisons"]:
        rows = ["| Metric | Group vs reference | Difference (95 % CI) | Ratio | Hedges' g | p (Welch) | p (Holm) |",
                "|---|---|---|---|---|---|---|"]
        for c in result["comparisons"]:
            ci = "" if c["ci95_low"] is None else f" ({_fmt(c['ci95_low'])} to {_fmt(c['ci95_high'])})"
            rows.append(f"| {metrics[c['metric']]['label']} | {c['group']} vs {c['reference']} (n={c['n']} vs "
                        f"{c['n_reference']}) | {_fmt(c['difference'])}{ci} | {_fmt(c['ratio'])} | "
                        f"{_fmt(c['hedges_g'])} | {_fmt(c['p'], 4)} | {_fmt(c['p_holm'], 4)} |")
        text += "\n\n" + "\n".join(rows)
    return text
