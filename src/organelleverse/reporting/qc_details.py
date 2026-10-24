"""Report detail for QC results: per-check table, coverage, gene recovery, comparison.

A QC Result's metrics only carry headline numbers; the individual checks, the
per-window coverage and the gene-profile assessment live in its canonical report
artifact. These helpers read that artifact (plain JSON) and return self-contained
HTML fragments - inline SVG and CSS bars, no scripts - for the report sections.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from html import escape
from typing import Any

from ..core.frozen import thaw_json
from ..core.result import OrganelleResult

# Check status -> (sort rank, dot class); "not_assessed" keeps the neutral dot.
_CHECK_STATUS = {
    "fail": (0, "failed"),
    "warn": (1, "warning"),
    "not_assessed": (2, ""),
    "pass": (3, "ok"),
}
_DECISION_CLASS = {"ready": "ok", "needs_review": "warning", "not_ready": "failed"}
_REPORT_KIND = {"qc.assembly": "assembly_qc_report", "qc.annotation": "annotation_qc_report"}


def qc_decision(result: OrganelleResult) -> tuple[str, str]:
    """The QC decision and its status class, or ``("", "")`` for other results."""
    if result.operation_id not in _REPORT_KIND:
        return "", ""
    decision = str(result.metrics.get("qc_decision", ""))
    return decision, _DECISION_CLASS.get(decision, "")


def qc_details(result: OrganelleResult) -> tuple[str, ...]:
    """HTML fragments for a successful QC or reference-comparison result."""
    if result.status == "failed":
        return ()
    operation = result.operation_id
    if operation == "qc.assembly":
        report = _report(result)
        return (_checks(report["checks"]), _coverage(report["coverage_windows"]))
    if operation == "qc.annotation":
        report = _report(result)
        return (_recovery(report["gene_profile_assessment"]), _checks(report["checks"]))
    if operation == "qc.compare_assembly_to_reference":
        return (_comparisons(thaw_json(result.metrics)["comparisons"]),)
    return ()


def _report(result: OrganelleResult) -> dict[str, Any]:
    (artifact,) = (a for a in result.artifacts if a.kind == _REPORT_KIND[result.operation_id])
    return json.loads(artifact.resolve().read_text(encoding="utf-8"))


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def _meter(fraction: float | None) -> str:
    width = 0 if fraction is None else max(0.0, min(1.0, fraction)) * 100
    return f"<span class='ov-meter'><i style='width:{width:.1f}%'></i></span>"


def _checks(checks: Sequence[dict[str, Any]]) -> str:
    ordered = sorted(checks, key=lambda c: _CHECK_STATUS[c["status"]][0])
    attention = sum(1 for c in checks if c["status"] in {"fail", "warn"})
    rows = "".join(
        f"<tr><td><i class='dot {_CHECK_STATUS[c['status']][1]}' title='{escape(c['status'])}'></i></td>"
        f"<td class='id'>{escape(c['check_id'])}</td><td>{escape(c['message'])}</td></tr>"
        for c in ordered
    )
    return (
        f"<details class='ov-group'{' open' if attention else ''}><summary>checks"
        f"<small>{len(checks)} checks, {attention} need attention</small></summary>"
        f"<table class='ov-checks'>{rows}</table></details>"
    )


def _coverage(windows: Sequence[dict[str, Any]]) -> str:
    """Mean depth per window for both coverage tracks; windows without any alignment marked."""
    if not windows:
        return ""
    lengths: dict[str, int] = {}
    for w in windows:
        lengths[w["sequence_id"]] = max(lengths.get(w["sequence_id"], 0), w["end"])
    offsets: dict[str, int] = {}
    total = 0
    for name, length in lengths.items():
        offsets[name] = total
        total += length
    width, height = 1000, 120
    depths = [
        d
        for w in windows
        for d in (w["any_alignment_mean_depth"], w["confident_mean_depth"])
        if d is not None
    ]
    top = max(depths, default=0) or 1.0

    def step(w: dict[str, Any], depth: float) -> list[str]:
        """The window as a horizontal segment, so even a single window is drawn."""
        y = height - depth / top * height
        first = offsets[w["sequence_id"]] + w["start"]
        last = offsets[w["sequence_id"]] + w["end"]
        return [f"{first / total * width:.1f},{y:.1f}", f"{last / total * width:.1f},{y:.1f}"]

    def line(css: str, key: str) -> str:
        runs: list[list[str]] = [[]]
        for w in windows:
            if w[key] is None:
                runs.append([])
            else:
                runs[-1].extend(step(w, w[key]))
        return "".join(
            f"<polyline class='{css}' fill='none' points='{' '.join(run)}'/>" for run in runs if run
        )

    gaps = "".join(
        f"<rect class='cov-gap' x='{(offsets[w['sequence_id']] + w['start']) / total * width:.1f}' "
        f"y='0' width='{max((w['end'] - w['start']) / total * width, 1):.1f}' height='{height}'/>"
        for w in windows
        if w["any_alignment_mean_depth"] == 0
    )
    seams = "".join(
        f"<line class='cov-seam' x1='{o / total * width:.1f}' x2='{o / total * width:.1f}' "
        f"y1='0' y2='{height}'/>"
        for o in list(offsets.values())[1:]
    )
    return (
        "<div class='ov-cov'>"
        f"<svg viewBox='0 0 {width} {height}' preserveAspectRatio='none' role='img' "
        f"aria-label='Mean depth per window along {len(lengths)} sequence(s)'>"
        f"{gaps}{seams}{line('cov-any', 'any_alignment_mean_depth')}"
        f"{line('cov-conf', 'confident_mean_depth')}</svg>"
        f"<p><span class='cov-key any'></span>any alignment "
        f"<span class='cov-key conf'></span>confident primary "
        f"<span class='cov-key gap'></span>no alignment &middot; "
        f"mean depth per window, 0-{top:.3g}x, {len(lengths)} sequence(s), {total:,} bp</p></div>"
    )


def _recovery(assessment: dict[str, Any]) -> str:
    expected = assessment["expected_profile_gene_count"]
    recovered = assessment["recovered_profile_gene_count"]
    fraction = recovered / expected if expected else None
    chips = ""
    for label, names in (
        ("Missing core genes", assessment["missing_expected_names"]),
        ("Duplicated core genes", assessment["duplicated_expected_names"]),
    ):
        if names:
            spans = "".join(f"<span>{escape(name)}</span>" for name in names)
            chips += f"<p class='ov-label'>{label}</p><div class='flags miss'>{spans}</div>"
    return (
        "<div class='ov-recovery'>"
        f"<p class='ov-label'>Core protein-coding genes recovered "
        f"<b>{recovered}/{expected}</b> ({_percent(fraction)})</p>{_meter(fraction)}{chips}</div>"
    )


def _comparisons(comparisons: Sequence[dict[str, Any]]) -> str:
    def cells(comparison: dict[str, Any] | None) -> str:
        if comparison is None:
            return "<td>-</td><td>-</td><td>-</td>"
        coverage = comparison["reference_coverage"]
        return (
            f"<td>{_meter(coverage)} {_percent(coverage)}</td>"
            f"<td>{_percent(comparison['alignment_identity'])}</td>"
            f"<td>{len(comparison['inverted_blocks'])}</td>"
        )

    rows = "".join(
        f"<tr><td class='id'>{escape(str(c['sample']))}</td><td>{c['query_length']:,}</td>"
        f"{cells(c['raw'])}{cells(c['normalized'])}</tr>"
        for c in comparisons
    )
    return (
        "<table class='ov-checks ov-compare'><thead><tr><th rowspan='2'>sample</th>"
        "<th rowspan='2'>length</th><th colspan='3'>raw</th><th colspan='3'>normalized</th></tr>"
        "<tr><th>reference coverage</th><th>identity</th><th>inverted blocks</th>"
        "<th>reference coverage</th><th>identity</th><th>inverted blocks</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
