"""HTML report construction and materialization.

Construction (:func:`build`) is compute-only: it validates and normalizes the
genome, results, title, backend, and images into an :class:`OrganelleReport`
*without creating any file*. Materialization (:func:`write`) is the only step
that renders and publishes the report to a user-selected destination.

The report is one styled, self-contained HTML page: a header with the
OrganelleVerse logo, an overview (status tally and genome facts), and one
section per ``OrganelleResult`` with metric tiles, collapsible nested
metrics, errors, embedded (base64) figures and provenance. No external
JS/CSS - the file is fully portable.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from html import escape
from importlib.resources import files
from pathlib import Path
from typing import cast

from ..core.artifacts import ArtifactRef
from ..core.errors import OrganelleInputError
from ..core.frozen import FrozenMap
from ..core.genome import OrganelleGenome
from ..core.result import Finding, OrganelleResult
from .qc_details import qc_decision, qc_details

# Report stylesheet: colour tokens (light + dark), class-based components, and
# the page layout (hero, nav, cards). Self-contained: the generated HTML needs
# no external assets.
_TOKENS = """
  --bg: #f4f6f3; --surface: #ffffff; --surface-2: #f7f9f7; --border: #e1e6e0;
  --text: #17201b; --muted: #5d6b63; --faint: #8a968f;
  --accent: #0b6e4f; --accent-soft: #e3f1ea; --accent-ink: #085a40;
  --ok: #2b8a3e; --ok-soft: #e6f4ea; --warning: #b45309; --warning-soft: #fdf1df;
  --failed: #c2272d; --failed-soft: #fdecec;
  --hero-1: #0b3d2e; --hero-2: #0b6e4f; --hero-ink: #f1faf5;
  --shadow: 0 1px 2px rgba(16, 32, 24, .05), 0 4px 16px rgba(16, 32, 24, .05);
  --radius: 12px;
  --mono: ui-monospace, SFMono-Regular, "Cascadia Mono", Menlo, Consolas, monospace;
"""
_DARK_TOKENS = """
    --bg: #0f1412; --surface: #161d1a; --surface-2: #1b2420; --border: #2a3530;
    --text: #e4ebe7; --muted: #9aa8a0; --faint: #6f7d75;
    --accent: #4cc497; --accent-soft: #173229; --accent-ink: #7fd9b4;
    --ok: #57c26f; --ok-soft: #16301e; --warning: #f0a54a; --warning-soft: #33261a;
    --failed: #f06a6f; --failed-soft: #3a1c1e;
    --hero-1: #0a2a20; --hero-2: #0e4a37; --hero-ink: #eaf6f0;
    --shadow: 0 1px 2px rgba(0, 0, 0, .3);
"""
_COMPONENT_CSS = """
.dot { flex: none; width: 8px; height: 8px; border-radius: 50%; background: var(--faint); }
.dot.ok { background: var(--ok); } .dot.warning { background: var(--warning); }
.dot.failed { background: var(--failed); }
.ov-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; }
.ov-stat { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px;
             padding: 12px 14px; }
.ov-stat b { display: block; font-size: 1.7rem; line-height: 1.1; font-variant-numeric: tabular-nums; }
.ov-stat span { font-size: .8rem; color: var(--muted); }
.ov-stat.ok b { color: var(--ok); } .ov-stat.warning b { color: var(--warning); }
.ov-stat.failed b { color: var(--failed); }
.ov-bar { display: flex; height: 8px; border-radius: 999px; overflow: hidden; margin: 16px 0 4px;
            background: var(--border); }
.ov-bar i { display: block; } .ov-bar .ok { background: var(--ok); }
.ov-bar .warning { background: var(--warning); } .ov-bar .failed { background: var(--failed); }
.ov-facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
              gap: 12px 20px; margin: 18px 0 0; padding-top: 16px; border-top: 1px dashed var(--border); }
.ov-facts dt { font-size: .74rem; font-weight: 400; color: var(--muted); }
.ov-facts dd { margin: 1px 0 0; font-weight: 600; overflow-wrap: anywhere; }
.ov-empty { color: var(--muted); margin: 14px 0 0; font-size: .9rem; }
.ov-eyebrow { margin: 0; font-size: .72rem; font-weight: 700; letter-spacing: .08em;
                text-transform: uppercase; color: var(--accent); }
.status { flex: none; font-size: .7rem; font-weight: 700; padding: 3px 10px; border-radius: 999px;
            letter-spacing: .06em; text-transform: uppercase; background: var(--ok-soft); color: var(--ok); }
.status.warning { background: var(--warning-soft); color: var(--warning); }
.status.failed { background: var(--failed-soft); color: var(--failed); }
.ov-summary { margin: 10px 0 0; color: var(--text); }
.ov-errors { margin: 12px 0 0; padding: 10px 14px; border-radius: 8px; background: var(--failed-soft);
               border-left: 3px solid var(--failed); font-size: .88rem; }
.ov-errors div + div { margin-top: 4px; }
.ov-errors code { font-family: var(--mono); font-size: .8rem; color: var(--failed); margin-right: 6px;
                    background: none; padding: 0; }
.ov-next { margin: 10px 0 0; font-size: .85rem; color: var(--muted); }
.ov-next code { font-family: var(--mono); font-size: .8rem; color: var(--accent-ink);
                  background: var(--accent-soft); border-radius: 6px; padding: 1px 6px; }
.flags { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 0; }
.flags span { background: var(--accent-soft); color: var(--accent-ink); padding: 2px 10px;
                border-radius: 999px; font-size: .76rem; font-family: var(--mono); }
.metrics { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px;
             margin: 16px 0 0; }
.metrics div { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px;
                 padding: 10px 12px; min-width: 0; }
.metrics dt { font-size: .74rem; font-weight: 400; color: var(--muted); font-family: var(--mono);
                overflow-wrap: anywhere; }
.metrics dd { margin: 2px 0 0; font-size: 1.15rem; font-weight: 650;
                font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.metrics dd.s { font-size: .9rem; font-weight: 500; }
.ov-group { margin: 12px 0 0; border: 1px solid var(--border); border-radius: 10px;
              background: var(--surface-2); }
.ov-group summary { cursor: pointer; padding: 9px 12px; font-family: var(--mono); font-size: .82rem;
                      font-weight: 600; }
.ov-group summary small { font-family: inherit; font-weight: 400; color: var(--faint);
                            margin-left: 10px; }
.ov-group dl { display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr));
                 gap: 0 18px; margin: 0; padding: 4px 12px 10px; border-top: 1px solid var(--border); }
.ov-group dl div { display: flex; justify-content: space-between; gap: 8px; padding: 3px 0;
                     border-bottom: 1px dotted var(--border); font-size: .8rem; }
.ov-group dt { font-family: var(--mono); font-weight: 400; color: var(--muted); }
.ov-group dd { margin: 0; font-variant-numeric: tabular-nums; font-weight: 600; }
.figures { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px;
             margin: 16px 0 0; }
.figures figure { margin: 0; background: #fff; border: 1px solid var(--border); border-radius: 10px;
                    padding: 10px; }
.figures img, .figures svg { display: block; max-width: 100%; height: auto; margin: 0 auto; }
.figures figcaption { margin-top: 8px; font-size: .76rem; color: #5d6b63; font-family: var(--mono);
                        overflow-wrap: anywhere; }
.provenance { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 16px 0 0; padding-top: 12px;
                border-top: 1px dashed var(--border); font-size: .76rem; color: var(--muted);
                font-family: var(--mono); }
.provenance b { font-weight: 600; color: var(--faint); margin-right: 4px; }
.ov-checks { width: 100%; border-collapse: collapse; font-size: .84rem; }
.ov-checks td, .ov-checks th { padding: 5px 10px; border-top: 1px solid var(--border);
                               text-align: left; vertical-align: top; }
.ov-checks th { font-size: .74rem; font-weight: 600; color: var(--muted); }
.ov-checks td:first-child { width: 22px; }
.ov-checks .dot { display: inline-block; margin-top: 4px; }
.ov-compare thead th[colspan] { text-align: center; }
.ov-checks .id { font-family: var(--mono); font-size: .78rem; white-space: nowrap; }
.ov-compare td, .ov-compare th { text-align: right; }
.ov-compare td:first-child, .ov-compare th:first-child { text-align: left; }
.ov-meter, .ov-mini { display: inline-block; vertical-align: middle; height: 8px; width: 90px;
                      border-radius: 999px; background: var(--border); overflow: hidden; }
.ov-meter { display: block; width: auto; height: 10px; margin: 6px 0 2px; }
.ov-compare .ov-meter { display: inline-block; width: 70px; height: 8px; margin: 0 6px 0 0; }
.ov-meter i { display: block; height: 100%; background: var(--accent); }
.ov-label { margin: 14px 0 0; font-size: .86rem; color: var(--muted); }
.ov-label b { color: var(--text); margin-left: 6px; }
.flags.miss span { background: var(--failed-soft); color: var(--failed); }
.ov-cov { margin: 16px 0 0; }
.ov-cov svg { display: block; width: 100%; height: 120px; background: var(--surface-2);
              border: 1px solid var(--border); border-radius: 10px; }
.ov-cov polyline, .ov-cov line { vector-effect: non-scaling-stroke; }
.ov-cov .cov-any { stroke: var(--faint); stroke-width: 1.5; }
.ov-cov .cov-conf { stroke: var(--accent); stroke-width: 1.5; }
.ov-cov .cov-gap { fill: var(--failed); opacity: .35; }
.ov-cov .cov-seam { stroke: var(--border); stroke-dasharray: 3 3; }
.ov-cov p { margin: 6px 0 0; font-size: .76rem; color: var(--muted); }
.cov-key { display: inline-block; width: 14px; height: 3px; margin: 0 4px 2px 10px;
           vertical-align: middle; background: var(--faint); }
.cov-key.conf { background: var(--accent); }
.cov-key.gap { height: 8px; background: var(--failed); opacity: .35; }
"""
_PAGE_CSS = """
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; background: var(--bg); color: var(--text); line-height: 1.55;
       font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue",
             "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
       -webkit-font-smoothing: antialiased; }
a { color: inherit; }
.ov-hero { background: linear-gradient(135deg, var(--hero-1), var(--hero-2)); color: var(--hero-ink);
           padding: 28px 16px 30px; }
.ov-hero-in, .ov-layout, .ov-foot { max-width: 1240px; margin: 0 auto; }
.ov-brand { display: flex; align-items: center; gap: 12px; }
.ov-brand img { width: 40px; height: 40px; border-radius: 10px; flex: none;
                box-shadow: 0 1px 3px rgba(0, 0, 0, .25); }
.ov-brand span { display: flex; flex-direction: column; font-size: 1.02rem; font-weight: 700;
                 letter-spacing: .01em; line-height: 1.2; }
.ov-brand small { font-size: .7rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
                  opacity: .7; }
.ov-hero h1 { margin: 10px 0 14px; font-size: 1.9rem; line-height: 1.2; font-weight: 700;
              letter-spacing: -.01em; }
.ov-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.ov-chips span { background: rgba(255, 255, 255, .12); border: 1px solid rgba(255, 255, 255, .18);
                 border-radius: 999px; padding: 3px 12px; font-size: .84rem; }
.ov-chips b { font-weight: 500; opacity: .7; margin-right: 6px; }
.ov-chips .sp { font-style: italic; }
.ov-layout { display: grid; grid-template-columns: 220px minmax(0, 1fr); gap: 28px;
             padding: 24px 16px 8px; }
.ov-nav { position: sticky; top: 16px; align-self: start; max-height: calc(100vh - 32px);
          overflow: auto; font-size: .86rem; }
.ov-nav p { margin: 0 0 8px; font-size: .72rem; font-weight: 700; letter-spacing: .08em;
            text-transform: uppercase; color: var(--faint); }
.ov-nav a { display: flex; align-items: center; gap: 8px; padding: 5px 10px; border-radius: 8px;
            text-decoration: none; color: var(--muted); overflow-wrap: anywhere; }
.ov-nav a:hover { background: var(--surface); color: var(--text); }
.ov-nav a small { color: var(--faint); }
main { min-width: 0; }
.ov-panel, section { background: var(--surface); border: 1px solid var(--border);
                     border-radius: var(--radius); box-shadow: var(--shadow);
                     padding: 20px 22px; margin: 0 0 18px; scroll-margin-top: 16px; }
.ov-panel h2 { margin: 0 0 14px; font-size: 1.05rem; }
section { border-top: 4px solid var(--ok); }
section.warning { border-top-color: var(--warning); }
section.failed { border-top-color: var(--failed); }
.ov-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
.ov-pills { display: flex; flex: none; gap: 6px; }
section h2 { margin: 2px 0 0; font-size: 1.2rem; line-height: 1.3; overflow-wrap: anywhere; }
.ov-foot { padding: 8px 16px 32px; color: var(--faint); font-size: .8rem; }
@media (max-width: 860px) {
  .ov-layout { grid-template-columns: minmax(0, 1fr); gap: 0; padding-top: 16px; }
  .ov-nav { position: static; max-height: none; display: flex; flex-wrap: wrap; gap: 6px;
            margin-bottom: 16px; }
  .ov-nav p { width: 100%; margin: 0; }
  .ov-nav a { background: var(--surface); border: 1px solid var(--border); padding: 3px 10px; }
  .ov-hero h1 { font-size: 1.5rem; }
  .ov-panel, section { padding: 16px; }
}
@media print {
  body { background: #fff; } .ov-nav { display: none; }
  .ov-layout { grid-template-columns: minmax(0, 1fr); }
  .ov-hero { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
  .ov-panel, section { box-shadow: none; break-inside: avoid; }
}
"""
_REPORT_CSS = (
    f":root {{ color-scheme: light dark;{_TOKENS}}}\n"
    f"@media (prefers-color-scheme: dark) {{\n  :root {{{_DARK_TOKENS}  }}\n}}"
    + _COMPONENT_CSS
    + _PAGE_CSS
)

_BACKENDS = ("auto", "jinja")


@dataclass(frozen=True)
class OrganelleReport:
    """A constructed, compute-only HTML report awaiting a writer.

    :func:`build` normalizes the inputs and records the requested backend; it
    creates no file. :func:`write` renders this report to a destination. The
    fields are the immutable, normalized inputs that fully determine the
    rendered output.
    """

    genome: OrganelleGenome
    results: tuple[OrganelleResult, ...]
    title: str
    backend: str
    images: tuple[Path, ...]


def build(
    genome: OrganelleGenome,
    results: list[OrganelleResult] | None = None,
    *,
    title: str = "OrganelleVerse Report",
    backend: str = "auto",
    images: list[str | Path] | None = None,
) -> OrganelleReport:
    """Construct a compute-only HTML report value (no file is created).

    Parameters
    ----------
    genome : OrganelleGenome
        The genome the report is about.
    results : list[OrganelleResult]
        Analysis results to embed (one section each, with metrics table,
        provenance, flags, and any figures referenced in
        ``result.artifacts``).
    title : report title.
    backend : ``"auto"`` (default) or ``"jinja"``; both select the in-tree
        HTML report. The MultiQC backend was removed; ``"multiqc"`` is refused.
    images : optional extra figure paths to embed at the top of the report
        (in addition to figures discovered in the result ``artifacts``).
    """
    if backend not in _BACKENDS:
        raise OrganelleInputError(
            code="reporting.unknown_backend",
            message=(
                "backend must be 'auto' or 'jinja'; the MultiQC backend was removed"
                if backend == "multiqc"
                else "backend must be 'auto' or 'jinja'"
            ),
            details={"backend": backend},
        )
    return OrganelleReport(
        genome=genome,
        results=tuple(results or []),
        title=title,
        backend=backend,
        images=tuple(Path(image) for image in (images or [])),
    )


def write(report: OrganelleReport, output: str | Path) -> OrganelleResult:
    """Render ``report`` to ``output`` and return the materialized result.

    Parameters
    ----------
    report : OrganelleReport
        The value returned by :func:`build`.
    output : path
        Output ``.html`` file.
    """
    report = _require_report(report)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    return _publish_jinja(report, out)


def _require_report(value: object) -> OrganelleReport:
    if isinstance(value, OrganelleReport):
        return value
    raise TypeError(
        "ov.report.write expects an OrganelleReport from ov.report.build(); "
        f"got {type(value).__name__}"
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SectionView:
    """Backend-neutral content of one report section (one result)."""

    group: str
    title: str
    status: str
    summary: str = ""
    flags: tuple[str, ...] = ()
    metrics: tuple[tuple[str, object], ...] = ()
    metric_groups: tuple[tuple[str, tuple[tuple[str, object], ...]], ...] = ()
    errors: tuple[tuple[str, str], ...] = ()
    suggestions: tuple[str, ...] = ()
    figures: tuple[Path, ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()
    decision: tuple[str, str] = ("", "")  # QC decision and its status class
    details: tuple[str, ...] = ()  # HTML fragments shown below the metrics


_STATUS_ORDER = ("ok", "warning", "failed")


def _render_jinja(report: OrganelleReport) -> str:
    genome = report.genome
    meta = genome.metadata
    chips = [("Organelle", genome.organelle, "")]
    if meta.species:
        chips.append(("Species", meta.species, "sp"))
    if meta.accession:
        chips.append(("Accession", meta.accession, ""))
    return _render_page(
        report.title,
        [_section_from_result(r) for r in report.results],
        chips=chips,
        facts=_genome_facts(genome),
        images=report.images,
        footer=f"{genome.organelle} / {meta.species or 'N/A'}",
    )


def _genome_facts(genome: OrganelleGenome) -> list[tuple[str, str]]:
    meta = genome.metadata
    facts = [
        ("Organelle", genome.organelle),
        ("Species", meta.species or "N/A"),
        ("Accession", meta.accession or "N/A"),
        ("Genetic code", str(meta.genetic_code)),
    ]
    facts += [
        (label, value)
        for label, value in (
            ("Assembly type", meta.assembly_type),
            ("Plastid type", meta.plastid_type),
            ("Source", meta.source),
        )
        if value
    ]
    return facts


def _section_from_result(r: OrganelleResult) -> _SectionView:
    suite, op = _operation_parts(r.operation_id)
    values, groups = _split_metrics(r.metrics)
    provenance: list[tuple[str, str]] = []
    pv = r.provenance
    if pv is not None:
        method = pv.actual_backend or pv.requested_backend
        if method:
            provenance.append(("method", method))
        provenance.append(("operation", pv.operation_id))
        if pv.package_version:
            provenance.append(("version", pv.package_version))
        if pv.git_commit:
            provenance.append(("commit", pv.git_commit[:10]))
        if pv.duration_seconds is not None:
            provenance.append(("time", f"{pv.duration_seconds:.3g} s"))
        provenance += [(str(k), str(v)) for k, v in pv.software_versions.items()]
    return _SectionView(
        group=suite,
        title=op,
        status=r.status,
        summary=r.summary_text,
        flags=r.flags,
        metrics=values,
        metric_groups=groups,
        errors=tuple((e.code, e.message) for e in r.errors),
        suggestions=tuple(s.operation_id for s in r.suggested_operations),
        figures=tuple(p for p in _artifact_paths(r) if _is_image(p)),
        provenance=tuple(provenance),
        decision=qc_decision(r),
        details=qc_details(r),
    )


@cache
def _logo_data_uri() -> str:
    """The desktop application's logo, downscaled for report headers."""
    data = (files(__package__) / "assets" / "logo.png").read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _logo_img() -> str:
    return f"<img src='{_logo_data_uri()}' alt='OrganelleVerse' width='40' height='40'>"


def _render_page(
    title: str,
    sections: Sequence[_SectionView],
    *,
    chips: Sequence[tuple[str, str, str]] = (),
    facts: Sequence[tuple[str, str]] = (),
    images: Sequence[Path] = (),
    footer: str = "",
) -> str:
    """Render the full self-contained report document."""
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{escape(title)}</title>",
        f"<style>{_REPORT_CSS}</style>",
        "</head><body>",
        "<header class='ov-hero'><div class='ov-hero-in'>",
        f"<div class='ov-brand'>{_logo_img()}<span>OrganelleVerse<small>report</small></span></div>",
        f"<h1>{escape(title)}</h1>",
    ]
    if chips:
        parts.append(
            "<div class='ov-chips'>"
            + "".join(
                f"<span class='{escape(cls)}'><b>{escape(label)}</b>{escape(value)}</span>"
                for label, value, cls in chips
            )
            + "</div>"
        )
    parts.append("</div></header>")

    # Navigation: overview, figures, then one entry per section.
    parts.append("<div class='ov-layout'><nav class='ov-nav'><p>Contents</p>")
    parts.append("<a href='#ov-overview'>Overview</a>")
    if images:
        parts.append("<a href='#ov-figures'>Figures</a>")
    for i, view in enumerate(sections, 1):
        group = f"<small>{escape(view.group)}</small>" if view.group else ""
        parts.append(
            f"<a href='#r{i}'><i class='dot {escape(view.status)}'></i>"
            f"<span>{group} {escape(view.title)}</span></a>"
        )
    parts.append("</nav><main>")

    parts.append("<div class='ov-panel' id='ov-overview'><h2>Overview</h2>")
    parts.append(_render_overview(sections, facts))
    parts.append("</div>")

    # Top-level figures supplied by the caller.
    if images:
        parts.append("<div class='ov-panel' id='ov-figures'><h2>Figures</h2>")
        parts.append(_render_figures(images))
        parts.append("</div>")

    for i, view in enumerate(sections, 1):
        parts.append(_render_section(view, f"r{i}"))

    parts.append("</main></div>")
    parts.append(
        "<footer class='ov-foot'>Generated by OrganelleVerse · "
        f"{len(sections)} result section(s)"
        + (f" · {escape(footer)}" if footer else "")
        + "</footer>"
    )
    parts.append("</body></html>")
    return "\n".join(parts)


def _render_overview(sections: Sequence[_SectionView], facts: Sequence[tuple[str, str]]) -> str:
    """Status tally, proportion bar, and input facts."""
    counts = {s: sum(1 for v in sections if v.status == s) for s in _STATUS_ORDER}
    parts = ["<div class='ov-stats'>"]
    parts.append(f"<div class='ov-stat'><b>{len(sections)}</b><span>Result sections</span></div>")
    for status in _STATUS_ORDER:
        parts.append(
            f"<div class='ov-stat {status}'><b>{counts[status]}</b>"
            f"<span>{status.capitalize()}</span></div>"
        )
    parts.append("</div>")
    if sections:
        parts.append(
            "<div class='ov-bar'>"
            + "".join(
                f"<i class='{status}' style='flex:{counts[status]}'></i>"
                for status in _STATUS_ORDER
                if counts[status]
            )
            + "</div>"
        )
    else:
        parts.append("<p class='ov-empty'>No result sections in this report.</p>")
    if facts:
        parts.append(
            "<dl class='ov-facts'>"
            + "".join(
                f"<div><dt>{escape(label)}</dt><dd>{escape(value)}</dd></div>"
                for label, value in facts
            )
            + "</dl>"
        )
    return "\n".join(parts)


def _render_section(view: _SectionView, anchor: str) -> str:
    return "\n".join(
        [
            f"<section class='{escape(view.status)}' id='{anchor}'>",
            "<div class='ov-head'><div>",
            f"<p class='ov-eyebrow'>{escape(view.group)}</p>" if view.group else "",
            f"<h2>{escape(view.title)}</h2></div>",
            "<div class='ov-pills'>"
            + _status_pill(view.status)
            + _decision_pill(view.decision)
            + "</div></div>",
            _render_section_body(view),
            "</section>",
        ]
    )


def _decision_pill(decision: tuple[str, str]) -> str:
    name, css = decision
    if not name:
        return ""
    return f"<span class='status {escape(css)}'>{escape(name.replace('_', ' '))}</span>"


def _status_pill(status: str) -> str:
    return f"<span class='status {escape(status)}'>{escape(status)}</span>"


def _render_section_body(view: _SectionView) -> str:
    """Everything in a section below its title: summary through provenance."""
    parts: list[str] = []
    if view.summary:
        parts.append(f"<p class='ov-summary'>{escape(view.summary)}</p>")
    if view.errors:
        parts.append(
            "<div class='ov-errors'>"
            + "".join(
                f"<div><code>{escape(code)}</code>{escape(message)}</div>"
                for code, message in view.errors
            )
            + "</div>"
        )
    if view.suggestions:
        parts.append(
            "<p class='ov-next'>Suggested next: "
            + " ".join(f"<code>{escape(op)}</code>" for op in view.suggestions)
            + "</p>"
        )
    if view.flags:
        parts.append(
            "<div class='flags'>"
            + "".join(f"<span>{escape(f)}</span>" for f in view.flags)
            + "</div>"
        )
    if view.metrics:
        parts.append("<dl class='metrics'>")
        for key, value in view.metrics:
            kind = "s" if isinstance(value, str) else "n"
            parts.append(
                f"<div><dt>{escape(str(key))}</dt>"
                f"<dd class='{kind}'>{escape(_fmt_value(value))}</dd></div>"
            )
        parts.append("</dl>")
    for name, group in view.metric_groups:
        parts.append(
            f"<details class='ov-group'><summary>{escape(name)}"
            f"<small>{len(group)} values</small></summary><dl>"
            + "".join(
                f"<div><dt>{escape(k)}</dt><dd>{escape(_fmt_value(v))}</dd></div>" for k, v in group
            )
            + "</dl></details>"
        )
    parts.extend(fragment for fragment in view.details if fragment)
    if view.figures:
        parts.append(_render_figures(view.figures))
    if view.provenance:
        parts.append(
            "<div class='provenance'>"
            + "".join(f"<span><b>{escape(k)}</b>{escape(v)}</span>" for k, v in view.provenance)
            + "</div>"
        )
    return "\n".join(parts)


def _render_figures(paths: Sequence[Path]) -> str:
    return (
        "<div class='figures'>"
        + "".join(
            f"<figure>{_embed_image(Path(p))}<figcaption>{escape(Path(p).name)}</figcaption></figure>"
            for p in paths
        )
        + "</div>"
    )


def _publish_jinja(report: OrganelleReport, out: Path) -> OrganelleResult:
    """Render the Jinja HTML and publish it atomically to ``out``."""
    html = _render_jinja(report)
    _atomic_write_text(out, html)
    results = report.results
    extra_images = report.images

    artifact = ArtifactRef.from_path(
        out,
        kind="report",
        format="html",
        media_type="text/html",
    )
    return OrganelleResult(
        operation_id="reporting.report",
        scope=report.genome.organelle,
        status="ok",
        artifacts=(artifact,),
        metrics=FrozenMap.from_json(
            {
                "results_count": len(results),
                "backend": "jinja2",
                "figures_embedded": (
                    sum(1 for r in results for p in _artifact_paths(r) if _is_image(p))
                    + len(extra_images)
                ),
            }
        ),
        findings=(
            Finding(code="report.sections", metric="report_sections", value=len(results)),
            Finding(code="report.backend", metric="backend", value="jinja2"),
        ),
        flags=("html_written",),
        summary_text=f"HTML report ({len(results)} sections, Jinja2 backend) → {out}.",
    )


def _atomic_write_text(destination: Path, text: str) -> None:
    """Write ``text`` to ``destination`` via temp-file + ``os.replace``.

    A crash or error during the write never leaves a partial destination: the
    temporary file is removed on any failure and the final name appears only
    via the atomic rename.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
# Short scalar lists (e.g. per-genome IR lengths) are shown inline; longer ones as a count.
_INLINE_LIST_MAX = 8


def _operation_parts(operation_id: str) -> tuple[str, str]:
    suite, separator, op = operation_id.partition(".")
    return (suite, op) if separator else ("result", suite)


def _artifact_paths(result: OrganelleResult) -> tuple[Path, ...]:
    return tuple(artifact.resolve() for artifact in result.artifacts)


def _is_image(path: Path) -> bool:
    return Path(path).suffix.lower() in _IMG_EXTS and Path(path).exists()


def _embed_image(path: Path) -> str:
    """Return an <img> tag with the image embedded as base64 (self-contained)."""
    p = Path(path)
    if not p.exists():
        return f"<p><em>(missing figure: {escape(p.name)})</em></p>"
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    if mime == "image/svg+xml":
        # SVG embeds cleanly inline (XML); no base64 needed.
        return f"<div class='fig'>{p.read_text(encoding='utf-8')}</div>"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"<img src='data:{mime};base64,{data}' alt='{escape(p.name)}' title='{escape(p.name)}'/>"


_MetricItems = tuple[tuple[str, object], ...]


def _split_metrics(
    metrics: Mapping[str, object],
) -> tuple[_MetricItems, tuple[tuple[str, _MetricItems], ...]]:
    """Split metrics into top-level display values and one-level nested groups.

    A one-element scalar list is shown as its value, short scalar lists are
    joined, other lists become an item count; deeper nesting is not displayed.
    """
    values: list[tuple[str, object]] = []
    groups: list[tuple[str, _MetricItems]] = []
    for k, v in metrics.items():
        if isinstance(v, Mapping):
            items = cast(Mapping[object, object], v).items()
            group = tuple(
                (str(kk), dv) for kk, vv in items if (dv := _display_value(vv)) is not None
            )
            if group:
                groups.append((str(k), group))
        elif (dv := _display_value(v)) is not None:
            values.append((str(k), dv))
    return tuple(values), tuple(groups)


def _display_value(v: object) -> object | None:
    if isinstance(v, (int, float, str)):
        return v
    if isinstance(v, Sequence):
        items = cast(Sequence[object], v)
        scalars = all(isinstance(x, (int, float, str)) for x in items)
        if not items:
            return "none"
        if scalars and len(items) == 1:
            return items[0]
        if scalars and len(items) <= _INLINE_LIST_MAX:
            return ", ".join(_fmt_value(x) for x in items)
        return f"[{len(items)} items]"
    return None


def _fmt_value(v: object) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.1f}" if abs(v) >= 1e4 else f"{v:.4g}"
    return str(v)
