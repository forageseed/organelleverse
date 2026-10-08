"""HTML report construction and materialization.

Construction (:func:`build`) is compute-only: it validates and normalizes the
genome, results, title, backend, and images into an :class:`OrganelleReport`
*without creating any file*. Materialization (:func:`write`) is the only step
that renders and publishes the report to a user-selected destination.

Two backends are supported, both behind the same compute-then-write contract:

  - **MultiQC** (default; the bioinformatics community-standard report
    framework — used by nf-core, Galaxy, etc.): when MultiQC is installed,
    :func:`write` drives it as a Python library
    (https://docs.seqera.io/multiqc/usage/scripts) to emit a publication-
    quality interactive report with collapsible sections, navigation sidebar,
    and embedded figures.

  - **Jinja2 in-tree** (automatic fallback when MultiQC is not installed): a
    styled, self-contained HTML page with a result table, per-section metric
    cards, provenance, and embedded (base64) figures. No external JS/CSS —
    the file is fully portable.

Both accept the same ``OrganelleGenome`` + ``list[OrganelleResult]`` and embed
any figures referenced in ``result.artifacts``.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from html import escape
from importlib.resources import as_file, files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from ..core.artifacts import ArtifactRef
from ..core.frozen import FrozenMap
from ..core.genome import OrganelleGenome
from ..core.result import ErrorDetail, Finding, OrganelleResult

# Report stylesheet, in three parts so the same section markup renders both as
# the standalone Jinja page and inside a MultiQC report:
#   - colour tokens (light + dark),
#   - class-based components, written with an ``&`` scope marker,
#   - the standalone page layout (hero, nav, cards), used only by the Jinja page.
# Self-contained: the generated HTML needs no external assets.
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
& .dot { flex: none; width: 8px; height: 8px; border-radius: 50%; background: var(--faint); }
& .dot.ok { background: var(--ok); } & .dot.warning { background: var(--warning); }
& .dot.failed { background: var(--failed); }
& .ov-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; }
& .ov-stat { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px;
             padding: 12px 14px; }
& .ov-stat b { display: block; font-size: 1.7rem; line-height: 1.1; font-variant-numeric: tabular-nums; }
& .ov-stat span { font-size: .8rem; color: var(--muted); }
& .ov-stat.ok b { color: var(--ok); } & .ov-stat.warning b { color: var(--warning); }
& .ov-stat.failed b { color: var(--failed); }
& .ov-bar { display: flex; height: 8px; border-radius: 999px; overflow: hidden; margin: 16px 0 4px;
            background: var(--border); }
& .ov-bar i { display: block; } & .ov-bar .ok { background: var(--ok); }
& .ov-bar .warning { background: var(--warning); } & .ov-bar .failed { background: var(--failed); }
& .ov-facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
              gap: 12px 20px; margin: 18px 0 0; padding-top: 16px; border-top: 1px dashed var(--border); }
& .ov-facts dt { font-size: .74rem; font-weight: 400; color: var(--muted); }
& .ov-facts dd { margin: 1px 0 0; font-weight: 600; overflow-wrap: anywhere; }
& .ov-empty { color: var(--muted); margin: 14px 0 0; font-size: .9rem; }
& .ov-eyebrow { margin: 0; font-size: .72rem; font-weight: 700; letter-spacing: .08em;
                text-transform: uppercase; color: var(--accent); }
& .status { flex: none; font-size: .7rem; font-weight: 700; padding: 3px 10px; border-radius: 999px;
            letter-spacing: .06em; text-transform: uppercase; background: var(--ok-soft); color: var(--ok); }
& .status.warning { background: var(--warning-soft); color: var(--warning); }
& .status.failed { background: var(--failed-soft); color: var(--failed); }
& .ov-summary { margin: 10px 0 0; color: var(--text); }
& .ov-errors { margin: 12px 0 0; padding: 10px 14px; border-radius: 8px; background: var(--failed-soft);
               border-left: 3px solid var(--failed); font-size: .88rem; }
& .ov-errors div + div { margin-top: 4px; }
& .ov-errors code { font-family: var(--mono); font-size: .8rem; color: var(--failed); margin-right: 6px;
                    background: none; padding: 0; }
& .ov-next { margin: 10px 0 0; font-size: .85rem; color: var(--muted); }
& .ov-next code { font-family: var(--mono); font-size: .8rem; color: var(--accent-ink);
                  background: var(--accent-soft); border-radius: 6px; padding: 1px 6px; }
& .flags { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 0; }
& .flags span { background: var(--accent-soft); color: var(--accent-ink); padding: 2px 10px;
                border-radius: 999px; font-size: .76rem; font-family: var(--mono); }
& .metrics { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px;
             margin: 16px 0 0; }
& .metrics div { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px;
                 padding: 10px 12px; min-width: 0; }
& .metrics dt { font-size: .74rem; font-weight: 400; color: var(--muted); font-family: var(--mono);
                overflow-wrap: anywhere; }
& .metrics dd { margin: 2px 0 0; font-size: 1.15rem; font-weight: 650;
                font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
& .metrics dd.s { font-size: .9rem; font-weight: 500; }
& .ov-group { margin: 12px 0 0; border: 1px solid var(--border); border-radius: 10px;
              background: var(--surface-2); }
& .ov-group summary { cursor: pointer; padding: 9px 12px; font-family: var(--mono); font-size: .82rem;
                      font-weight: 600; }
& .ov-group summary small { font-family: inherit; font-weight: 400; color: var(--faint);
                            margin-left: 10px; }
& .ov-group dl { display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr));
                 gap: 0 18px; margin: 0; padding: 4px 12px 10px; border-top: 1px solid var(--border); }
& .ov-group dl div { display: flex; justify-content: space-between; gap: 8px; padding: 3px 0;
                     border-bottom: 1px dotted var(--border); font-size: .8rem; }
& .ov-group dt { font-family: var(--mono); font-weight: 400; color: var(--muted); }
& .ov-group dd { margin: 0; font-variant-numeric: tabular-nums; font-weight: 600; }
& .figures { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px;
             margin: 16px 0 0; }
& .figures figure { margin: 0; background: #fff; border: 1px solid var(--border); border-radius: 10px;
                    padding: 10px; }
& .figures img, & .figures svg { display: block; max-width: 100%; height: auto; margin: 0 auto; }
& .figures figcaption { margin-top: 8px; font-size: .76rem; color: #5d6b63; font-family: var(--mono);
                        overflow-wrap: anywhere; }
& .provenance { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 16px 0 0; padding-top: 12px;
                border-top: 1px dashed var(--border); font-size: .76rem; color: var(--muted);
                font-family: var(--mono); }
& .provenance b { font-weight: 600; color: var(--faint); margin-right: 4px; }
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
# Inside a MultiQC report: tokens follow MultiQC's own light/dark toggle, and
# each result's content opens with its suite label and status pill.
_MULTIQC_EXTRA_CSS = """
.ov-mqc { color: var(--text); margin: 0 0 8px; }
.ov-mqc .ov-mqc-head { display: flex; align-items: center; gap: 10px; }
"""


def _scoped_components(scope: str) -> str:
    """Component rules with the ``&`` marker replaced by ``scope`` (or removed)."""
    return _COMPONENT_CSS.replace("& ", f"{scope} " if scope else "")


_REPORT_CSS = (
    f":root {{ color-scheme: light dark;{_TOKENS}}}\n"
    f"@media (prefers-color-scheme: dark) {{\n  :root {{{_DARK_TOKENS}  }}\n}}"
    + _scoped_components("")
    + _PAGE_CSS
)
_MULTIQC_CSS = (
    f".ov-mqc {{{_TOKENS}}}\n"
    f"[data-bs-theme='dark'] .ov-mqc {{{_DARK_TOKENS}}}"
    + _scoped_components(".ov-mqc")
    + _MULTIQC_EXTRA_CSS
)


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
    backend : ``"auto"`` (default) | ``"multiqc"`` | ``"jinja"``. The requested
        backend is recorded verbatim; the effective backend is resolved at
        write time so construction stays free of side effects beyond input
        normalization.
    images : optional extra figure paths to embed at the top of the report
        (in addition to figures discovered in the result ``artifacts``).
    """
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
        Output ``.html`` file. MultiQC also writes a ``*_data/`` dir next to it
        when its data directory is enabled.
    """
    report = _require_report(report)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if report.backend in ("auto", "multiqc"):
        try:
            import multiqc  # type: ignore # noqa: F401
        except ImportError:
            if report.backend == "multiqc":
                return OrganelleResult(
                    operation_id="reporting.report",
                    scope=report.genome.organelle,
                    status="failed",
                    summary_text="backend='multiqc' requested but MultiQC is not installed.",
                    errors=(
                        ErrorDetail(
                            code="reporting.multiqc_missing",
                            message="backend='multiqc' requested but MultiQC is not installed.",
                        ),
                    ),
                )
        else:
            return _report_multiqc(report, out)
    return _publish_jinja(report, out)


def _require_report(value: object) -> OrganelleReport:
    if isinstance(value, OrganelleReport):
        return value
    raise TypeError(
        "ov.report.write expects an OrganelleReport from ov.report.build(); "
        f"got {type(value).__name__}"
    )


# ---------------------------------------------------------------------------
# Backend 1 — Jinja2 (default, always available)
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
            _status_pill(view.status) + "</div>",
            _render_section_body(view),
            "</section>",
        ]
    )


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
# Backend 2 — MultiQC (publication-quality interactive report)
# ---------------------------------------------------------------------------


def _report_multiqc(report: OrganelleReport, out: Path) -> OrganelleResult:
    """Drive MultiQC as a Python library to build an interactive report.

    An "Overview" module (status tally, genome facts, caller figures) is
    followed by one module per ``OrganelleResult``. Each module holds one
    untitled section, built with MultiQC's module API the same way
    ``multiqc.add_custom_content_section()`` does, minus the repeated section
    heading. The section bodies are the Jinja report's markup; the
    OrganelleVerse stylesheet and logo are passed to ``multiqc.write_report()``
    through its documented ``custom_css_files`` and a session config file
    (https://docs.seqera.io/multiqc/usage/scripts).
    """
    import multiqc  # type: ignore
    from multiqc import report as mqc_report  # type: ignore
    from multiqc.base_module import BaseMultiqcModule  # type: ignore

    genome = report.genome
    results = report.results
    extra_images = report.images
    views = [_section_from_result(r) for r in results]

    multiqc.reset()

    def add_module(name: str, anchor: str, content: str) -> None:
        module = BaseMultiqcModule(name=name, anchor=f"{anchor}-module")
        module.add_section(anchor=anchor, content=f"<div class='ov-mqc'>{content}</div>")
        mqc_report.modules.append(module)

    overview = _render_overview(views, _genome_facts(genome))
    if extra_images:
        overview += _render_figures(extra_images)
    add_module("Overview", "ov_overview", overview)
    for i, view in enumerate(views, 1):
        eyebrow = f"<p class='ov-eyebrow'>{escape(view.group)}</p>" if view.group else ""
        head = f"<div class='ov-mqc-head'>{eyebrow}{_status_pill(view.status)}</div>"
        add_module(f"{view.group}.{view.title}", f"ov_r{i}", head + _render_section_body(view))

    with (
        TemporaryDirectory() as scratch,
        as_file(files(__package__) / "assets" / "logo.png") as logo,
    ):
        css = Path(scratch) / "organelleverse.css"
        css.write_text(_MULTIQC_CSS, encoding="utf-8")
        config = Path(scratch) / "multiqc_config.yaml"
        # JSON is valid YAML, and quotes the logo path safely.
        config.write_text(
            json.dumps(
                {
                    "custom_logo": str(logo),
                    "custom_logo_title": "OrganelleVerse",
                    "custom_logo_width": 56,
                }
            ),
            encoding="utf-8",
        )
        # MultiQC writes <filename> inside output_dir.
        written = multiqc.write_report(  # pyright: ignore[reportUnknownMemberType]
            title=report.title,
            output_dir=str(out.parent),
            filename=out.name,
            force=True,
            quiet=True,
            make_data_dir=False,
            config_files=[str(config)],
            custom_css_files=[str(css)],
        )
    # MultiQC may return None on success but the file is written.
    written_path = Path(written) if written else out
    if not written_path.exists():
        written_path = out  # fall back to the requested path

    artifact = ArtifactRef.from_path(
        written_path,
        kind="report",
        format="html",
        media_type="text/html",
    )
    return OrganelleResult(
        operation_id="reporting.report",
        scope=genome.organelle,
        status="ok",
        artifacts=(artifact,),
        metrics=FrozenMap.from_json(
            {
                "results_count": len(results),
                "backend": "multiqc",
                "figures_embedded": (
                    sum(1 for r in results for p in _artifact_paths(r) if _is_image(p))
                    + len(extra_images)
                ),
            }
        ),
        findings=(
            Finding(code="report.sections", metric="report_sections", value=len(results)),
            Finding(code="report.backend", metric="backend", value="multiqc"),
        ),
        flags=("html_written", "multiqc"),
        summary_text=(f"MultiQC report ({len(results)} sections + overview) → {written_path}."),
    )


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
