"""Reusable suite-level visualization helpers for OrganelleVerse.

These functions intentionally accept plain Python data structures. Suite code
can pass its existing metrics/tables without building a new plotting object,
and demo scripts can generate figures without external bioinformatics tools.
"""

from __future__ import annotations

import csv
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping, Sequence
from html import escape
from math import cos, log10, pi, sin, sqrt
from pathlib import Path
from typing import Any, TypeAlias, TypeGuard, cast

import matplotlib

matplotlib.use("Agg", force=True)
import itertools

import matplotlib.pyplot as plt  # type: ignore
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm  # type: ignore
from matplotlib.lines import Line2D  # type: ignore
from matplotlib.patches import (  # type: ignore
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

from ..core.frozen import thaw_json
from ..core.result import OrganelleResult
from ..selection.kaks import _CODONS as _STANDARD_CODONS
from .panel_capture import finish, journal_figsize, new_axes, renderer_panel
from .plot_object import OrganellePlot, data_plot_result, plot_result
from .style import FONT_PT, publication_rc

_PALETTE = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#6B7280",
)

TransferResultInput: TypeAlias = OrganelleResult | Mapping[str, Any]
_ALARM = "#D55E00"
_NEUTRAL = "#52606D"
_LIGHT = "#E5E7EB"
_NUMT_FRAGMENT = "#2C7FB8"
_NUMT_NEW_GENE = "#6D5ACF"
_NUMT_UN_GENE = "#F97316"
_IDEOGRAM_LABEL_STYLES = (
    {"color": "#2CA02C", "marker": "s"},
    {"color": "#6D4AA2", "marker": "o"},
    {"color": "#FF7F0E", "marker": "^"},
    {"color": "#1F77B4", "marker": "D"},
    {"color": "#D62728", "marker": "v"},
    {"color": "#17BECF", "marker": "P"},
    {"color": "#8C564B", "marker": "X"},
    {"color": "#BCBD22", "marker": "<"},
    {"color": "#E377C2", "marker": ">"},
    {"color": "#7F7F7F", "marker": "*"},
)
_IDEOGRAM_MARKERS = {
    f"label{index}": style for index, style in enumerate(_IDEOGRAM_LABEL_STYLES, start=1)
}
_IDEOGRAM_MARKERS.update(
    {
        "rRNA": _IDEOGRAM_MARKERS["label1"],
        "tRNA": _IDEOGRAM_MARKERS["label2"],
        "miRNA": _IDEOGRAM_MARKERS["label3"],
    }
)
_IDEOGRAM_MARKER_TRACK_GAP = 0.12
_ERC_METHODS = ("pearson", "spearman")
_SYNTENY_GROUP_COLORS = {
    "Wine": "#9BD3E0",
    "Table": "#D9E76C",
    "Syl": "#F4B6CF",
    "Vwr": "#B39DDB",
}
_SYNTENY_DIRECT = "#EF4444"
_SYNTENY_INVERTED = "#1E88E5"
_AA_ORDER = (
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "*",
)
_AA_NAMES = {
    "A": "Ala",
    "R": "Arg",
    "N": "Asn",
    "D": "Asp",
    "C": "Cys",
    "Q": "Gln",
    "E": "Glu",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "L": "Leu",
    "K": "Lys",
    "M": "Met",
    "F": "Phe",
    "P": "Pro",
    "S": "Ser",
    "T": "Thr",
    "W": "Trp",
    "Y": "Tyr",
    "V": "Val",
    "*": "Stop",
}


def plot_qc_dashboard(
    metrics: Mapping[str, str | int | float | bool | None | list[str]],
    *,
    title: str | None = None,
) -> OrganellePlot:
    """Render an evidence-driven assembly-QC dashboard (compute-only).

    The dashboard reflects the immutable QC decision and its evidence: check
    counts, both coverage tracks (any alignment, confident primary) with their
    zero-coverage bases, unsupported/contradicted junction counts, and flags. It
    does not assign or display a 0-100 score, a letter grade, or a
    GC-contamination claim; no universal pass threshold is applied. Render with
    ``ov.write(plot, output)``.

    ``metrics`` is the ``metrics`` of an ``ov.qc.assembly`` result. Its flags live on
    the result itself, so add them as ``{**qc.metrics, "flags": list(qc.flags)}``;
    without a ``flags`` entry the panel says "not provided" instead of "none".
    """

    def _render(path: str | Path) -> Path:
        return _render_qc_dashboard(metrics, path, title=title)

    return plot_result(
        "plot_qc_dashboard",
        _render,
        metrics={"render_method": "qc_dashboard"},
        flags=("qc_dashboard_prepared",),
        summary="Assembly QC dashboard prepared.",
    )


_LGA_CHECK_COLORS = {
    "Pass": _PALETTE[2],
    "Warning": _PALETTE[1],
    "Fail": _ALARM,
    "Not assessed": "#BFC5CC",
}
_LGA_TRACK = "#EEF0F2"
_LGA_MUTED = "#6B7280"


def _render_qc_dashboard(
    metrics: Mapping[str, Any],
    output: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_qc_dashboard`."""
    import textwrap

    out = _prepare_output(output)
    _apply_figure_style()

    decision = str(metrics.get("qc_decision", "insufficient_evidence"))
    decision_color = {
        "ready": _PALETTE[2],
        "needs_review": _PALETTE[1],
        "not_ready": _ALARM,
    }.get(decision, _LGA_MUTED)
    pass_count = int(_float(metrics.get("qc_pass_count", 0)))
    warning_count = int(_float(metrics.get("qc_warning_count", 0)))
    failure_count = int(_float(metrics.get("qc_failure_count", 0)))
    not_assessed_count = int(_float(metrics.get("qc_not_assessed_count", 0)))
    # The two coverage tracks of the assembly-QC result, which are not interchangeable.
    coverage_tracks = (
        (
            "any alignment",
            metrics.get("any_alignment_coverage_breadth"),
            metrics.get("any_alignment_zero_coverage_bases"),
        ),
        (
            "confident primary",
            metrics.get("confident_coverage_breadth"),
            metrics.get("confident_zero_coverage_bases"),
        ),
    )
    unsupported_junction_count = metrics.get("unsupported_junction_count")
    contradicted_junction_count = metrics.get("contradicted_junction_count")
    # Flags live on the QC result, not in its metrics: pass them as metrics["flags"].
    flags = None if "flags" not in metrics else [str(value) for value in metrics["flags"]]

    # One strip at double-column width: decision | checks | coverage | junctions.
    fig, axes = plt.subplots(
        1,
        4,
        figsize=journal_figsize(7.2, 1.75),
        gridspec_kw={"width_ratios": [0.9, 1.25, 1.55, 1.35], "wspace": 0.55},
    )
    _figure_title(fig, title)

    ax = axes[0]
    ax.axis("off")
    ax.set_title("Decision", loc="left")
    ax.text(
        0.0,
        0.72,
        decision.replace("_", " ").capitalize(),
        fontsize=FONT_PT["title"],
        fontweight="bold",
        color=decision_color,
        transform=ax.transAxes,
        va="center",
    )
    assessed = pass_count + warning_count + failure_count
    total = assessed + not_assessed_count
    ax.text(
        0.0,
        0.46,
        f"{pass_count} of {total} checks passed" if total else "No checks reported",
        color=_LGA_MUTED,
        transform=ax.transAxes,
        va="center",
    )

    ax = axes[1]
    labels = list(_LGA_CHECK_COLORS)
    counts = [pass_count, warning_count, failure_count, not_assessed_count]
    ax.barh(
        range(len(labels)),
        counts,
        color=[_LGA_CHECK_COLORS[label] for label in labels],
        height=0.62,
    )
    peak = max(1, *counts)
    for row, count in enumerate(counts):
        ax.text(count + peak * 0.04, row, str(count), va="center", ha="left")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(0, peak * 1.25)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Checks", loc="left")
    _despine(ax)
    ax.tick_params(axis="y", length=0, pad=3)

    ax = axes[2]
    ax.set_title("Coverage breadth", loc="left")
    ax.barh(range(2), [1.0, 1.0], color=_LGA_TRACK, height=0.5)
    for row, (_, breadth, _) in enumerate(coverage_tracks):
        if breadth is None:
            ax.text(0.5, row, "n/a", ha="center", va="center", color=_LGA_MUTED)
            continue
        value = max(0.0, min(1.0, _float(breadth)))
        ax.barh([row], [value], color=_PALETTE[0], height=0.5)
        text = f"{100 * value:.1f}%"
        if value >= 0.3:
            ax.text(value - 0.02, row, text, ha="right", va="center", color="white")
        else:
            ax.text(value + 0.02, row, text, ha="left", va="center", color="#1F2933")
    ax.set_xlim(0, 1)
    ax.set_ylim(1.6, -0.6)
    ax.set_yticks(
        range(2),
        [
            f"{label}\nzero: {'n/a' if zero is None else _format_bp(int(_float(zero)))}"
            for label, _, zero in coverage_tracks
        ],
    )
    ax.set_xticks([0, 0.5, 1.0], ["0", "50", "100"])
    ax.set_xlabel("Reference covered (%)")
    _despine(ax)
    ax.tick_params(axis="y", length=0, pad=3)

    ax = axes[3]
    ax.axis("off")
    ax.set_title("Junctions & flags", loc="left")
    unsupported_text = (
        "n/a"
        if unsupported_junction_count is None
        else str(int(_float(unsupported_junction_count)))
    )
    contradicted_text = (
        "n/a"
        if contradicted_junction_count is None
        else str(int(_float(contradicted_junction_count)))
    )
    flag_text = (
        "not provided"
        if flags is None
        else textwrap.fill(", ".join(flags), 26, break_long_words=False) if flags else "none"
    )
    lines = [
        ("Unsupported junctions", unsupported_text),
        ("Contradicted junctions", contradicted_text),
        ("Flags", flag_text),
    ]
    # count first, then its name, offset by the widest count
    from .style import text_width_pt

    axes_width_pt = max(1.0, ax.get_position().width * fig.get_figwidth() * 72)
    value_width = max(text_width_pt(value, FONT_PT["body"], bold=True) for _, value in lines[:2])
    label_x = min(0.5, (value_width + 5.0) / axes_width_pt)
    y = 0.86
    for label, value in lines[:2]:
        ax.text(0.0, y, value, transform=ax.transAxes, va="top", fontweight="bold")
        ax.text(label_x, y, label, color=_LGA_MUTED, transform=ax.transAxes, va="top")
        y -= 0.22
    ax.text(0.0, y, lines[2][0], color=_LGA_MUTED, transform=ax.transAxes, va="top")
    ax.text(0.0, y - 0.15, lines[2][1], transform=ax.transAxes, va="top", color="#1F2933")

    return _save(fig, out)


def plot_track_density(
    tracks: Sequence[Mapping[str, str | int | float | bool | None]],
    *,
    title: str | None = None,
    regions: Sequence[Mapping[str, Any]] | None = None,
) -> OrganellePlot:
    """Render aligned genomic-position tracks (compute-only)."""
    if not tracks:
        raise ValueError("tracks must contain at least one track")

    def _render(path: str | Path) -> Path:
        return _render_track_density(tracks, path, title=title, regions=regions)

    return plot_result(
        "plot_track_density",
        _render,
        metrics={"tracks": len(tracks)},
        summary="Windowed track density prepared.",
    )


_LGA_REGION_SHADE = "#F1F3F5"


def _lga_genome_axis(ax: Any, max_position: float, label: str = "Position") -> None:
    """Label a base-pair x axis in bp, kb or Mb (thousands separators for bp)."""
    from matplotlib.ticker import FuncFormatter  # type: ignore

    if max_position >= 2_000_000:
        scale, unit = 1_000_000, "Mb"
    elif max_position >= 2_000:
        scale, unit = 1_000, "kb"
    else:
        scale, unit = 1, "bp"

    def _format(value: float, _pos: int) -> str:
        if scale == 1:
            return f"{value:,.0f}"
        return f"{value / scale:,.2f}".rstrip("0").rstrip(".")

    ax.xaxis.set_major_formatter(FuncFormatter(_format))
    ax.set_xlabel(f"{label} ({unit})")


def _render_track_density(
    tracks: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    title: str | None = None,
    regions: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_track_density`."""
    out = _prepare_output(output)
    _apply_figure_style()

    n = len(tracks)
    fig, axes = plt.subplots(
        n, 1, figsize=journal_figsize(7.2, max(1.5, 0.95 * n + 0.45)), sharex=True
    )
    if n == 1:
        axes = [axes]
    _figure_title(fig, title)

    max_x = 0.0
    for i, (ax, track) in enumerate(zip(axes, tracks, strict=False)):
        x = _float_list(track.get("x", []), "track x")
        y = _float_list(track.get("y", []), "track y")
        if len(x) != len(y):
            raise ValueError("track x and y must have the same length")
        max_x = max([max_x, *x])
        # one hue for every track unless the caller colors them
        color = str(track.get("color") or _PALETTE[0])
        label = str(track.get("label", f"track {i + 1}"))
        dense = len(x) > 40
        ax.plot(
            x,
            y,
            marker="" if dense else "o",
            ms=2.2,
            lw=0.8,
            color=color,
            label=label,
            solid_joinstyle="round",
        )
        previous_end: float | None = None
        previous_shaded = False
        for region in regions or ():
            start = _float(region.get("start", 0.0))
            end = _float(region.get("end", start))
            # Every region is shaded, except that a region abutting a shaded
            # one is left white so tiled regions (LSC/IR/SSC) alternate
            # instead of washing the whole axis grey.
            abutting = previous_end is not None and abs(start - previous_end) <= 1
            shaded = not (abutting and previous_shaded)
            if shaded:
                ax.axvspan(start, end, color=_LGA_REGION_SHADE, lw=0, zorder=0)
            previous_end, previous_shaded = end, shaded
            if i == 0 and region.get("label"):
                ax.text(
                    (start + end) / 2,
                    1.03,
                    str(region["label"]),
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=FONT_PT["label"],
                    color=_NEUTRAL,
                    clip_on=False,
                )
        ylabel = str(track.get("ylabel", label))
        ax.set_ylabel(ylabel)
        if label != ylabel:
            # name the track in place instead of a one-entry legend
            ax.text(
                0.005,
                0.97,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=FONT_PT["label"],
                color=_NEUTRAL,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.6},
                zorder=4,
            )
        ax.margins(x=0)
        ax.tick_params(axis="y", labelsize=FONT_PT["tick"])
        _despine(ax)
    if regions:
        max_x = max([max_x, *(_float(region.get("end", 0.0)) for region in regions)])
    fig.align_ylabels(axes)
    _lga_genome_axis(axes[-1], max_x)
    return _save(fig, out)


def plot_matrix_heatmap(
    matrix: Sequence[Sequence[int | float]],
    *,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    title: str | None = None,
    center: float | None = None,
    cmap: str | None = None,
    annotate: bool = True,
    colorbar_label: str | None = None,
) -> OrganellePlot:
    """Render a small matrix heatmap with optional cell values (compute-only).

    Cells are square; ``center`` gives a diverging scale symmetric about it.
    ``colorbar_label`` names the colour scale (e.g. ``"Identity (%)"``).
    """
    values = _matrix(matrix)
    if not values:
        raise ValueError("matrix must contain at least one row")
    n_rows = len(values)
    n_cols = len(values[0])
    rows = list(row_labels or [str(i + 1) for i in range(n_rows)])
    cols = list(col_labels or [str(i + 1) for i in range(n_cols)])
    if len(rows) != n_rows:
        raise ValueError("row_labels length must match matrix rows")
    if len(cols) != n_cols:
        raise ValueError("col_labels length must match matrix columns")

    def _render(path: str | Path) -> Path:
        return _render_matrix_heatmap(
            values,
            rows,
            cols,
            path,
            title=title,
            center=center,
            cmap=cmap,
            annotate=annotate,
            colorbar_label=colorbar_label,
        )

    plot = plot_result(
        "plot_matrix_heatmap",
        _render,
        metrics={"rows": n_rows, "cols": n_cols},
        summary="Matrix heatmap prepared.",
    )
    plot._panel = renderer_panel(_render)
    return plot


#: Sequential scale for plain matrices: white to the Okabe-Ito blue family.
_MTX_SEQUENTIAL = ("#F5F8FB", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B")
_MTX_MM = 25.4


def _mtx_decimals(flat: Sequence[float]) -> int:
    """One number format per matrix: integers stay integers, others share decimals."""
    if all(float(v).is_integer() for v in flat):
        return 0
    peak = max(abs(v) for v in flat)
    if peak < 10:
        return 2
    return 1 if peak < 1000 else 0


def _mtx_cell_text(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    if text.startswith("-"):
        text = "\u2212" + text[1:]  # typographic minus
        if not text.strip("\u22120."):
            text = text[1:]  # no "-0.00"
    return text


def _mtx_text_color(rgba: Sequence[float]) -> str:
    """Dark or white cell text from the luminance of the cell colour itself."""
    red, green, blue = rgba[:3]
    return "#111827" if 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.5 else "white"


def _mtx_layout(
    rows: Sequence[str], cols: Sequence[str], annotate: bool
) -> tuple[float, float, float, bool]:
    """Journal width, height (inches), cell size (pt) and column-label rotation."""
    from .style import text_width_pt

    tick = FONT_PT["tick"]
    row_mm = max(text_width_pt(str(label), tick) for label in rows) / 72 * _MTX_MM + 2.5
    col_pt = max(text_width_pt(str(label), tick) for label in cols)
    cbar_mm = 16.0
    target_mm = 7.0 if annotate else 4.5
    n_rows, n_cols = len(rows), len(cols)
    width_mm = 183.0
    for journal_mm in (89.0, 120.0, 183.0):
        if row_mm + n_cols * target_mm + cbar_mm <= journal_mm:
            width_mm = journal_mm
            break
    cell_mm = min(target_mm, max(1.5, (width_mm - row_mm - cbar_mm) / n_cols))
    cell_pt = cell_mm / _MTX_MM * 72
    rotate = col_pt > 0.9 * cell_pt
    col_mm = (col_pt * 0.72 + tick * 0.72 if rotate else tick * 1.2) / 72 * _MTX_MM + 2.5
    height_mm = n_rows * cell_mm + col_mm + 6.0
    return width_mm / _MTX_MM, height_mm / _MTX_MM, cell_pt, rotate


def _render_matrix_heatmap(
    values: list[list[float]],
    rows: list[str],
    cols: list[str],
    output: str | Path,
    *,
    title: str | None = None,
    center: float | None = None,
    cmap: str | None = None,
    annotate: bool = True,
    colorbar_label: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_matrix_heatmap`."""
    from matplotlib.colors import LinearSegmentedColormap  # type: ignore

    from .style import text_width_pt

    out = _prepare_output(output)
    _apply_figure_style()

    n_rows = len(values)
    n_cols = len(values[0])

    flat = [v for row in values for v in row]
    if center is None:
        vmin = min(flat)
        vmax = max(flat)
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        norm = Normalize(vmin=vmin, vmax=vmax)
        chosen_cmap: Any = cmap or LinearSegmentedColormap.from_list(
            "ov_matrix_blue", _MTX_SEQUENTIAL
        )
    else:
        # symmetric about the centre so equal distances read as equal colours
        span = max(abs(v - center) for v in flat) or 1.0
        norm = TwoSlopeNorm(vmin=center - span, vcenter=center, vmax=center + span)
        chosen_cmap = cmap or "RdBu_r"

    width_in, height_in, cell_pt, rotate = _mtx_layout(rows, cols, annotate)
    fig, ax = new_axes(journal_figsize(width_in, height_in))
    im = ax.imshow(values, cmap=chosen_cmap, norm=norm, aspect="equal", interpolation="nearest")
    if rotate:
        ax.set_xticks(range(n_cols), cols, rotation=45, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticks(range(n_cols), cols)
    ax.set_yticks(range(n_rows), rows)
    ax.set_xticks([i - 0.5 for i in range(1, n_cols)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(1, n_rows)], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _panel_title(ax, title)

    if annotate and n_rows * n_cols <= 200:
        decimals = _mtx_decimals(flat)
        texts = [[_mtx_cell_text(v, decimals) for v in row] for row in values]
        widest = max(len(text) for row in texts for text in row)
        sample = max((t for row in texts for t in row if len(t) == widest), key=len)
        size = next(
            (
                pt
                for pt in (FONT_PT["label"],)
                if text_width_pt(sample, pt) <= 0.86 * cell_pt
            ),
            None,
        )
        if size is not None:
            cmap_obj = im.get_cmap()
            for r, row in enumerate(values):
                for c, value in enumerate(row):
                    ax.text(
                        c,
                        r,
                        texts[r][c],
                        ha="center",
                        va="center",
                        fontsize=size,
                        color=_mtx_text_color(cmap_obj(norm(value))),
                    )

    # colour bar as an inset so it matches the matrix height and stays in a panel
    # 2.2 mm wide whatever the axes size (also inside a composed panel)
    box = ax.get_position()
    box_w_pt = box.width * fig.get_figwidth() * 72
    box_h_pt = box.height * fig.get_figheight() * 72
    drawn_w_pt = min(box_w_pt, box_h_pt * n_cols / n_rows)
    bar_fraction = (2.2 / _MTX_MM * 72) / max(1e-6, drawn_w_pt)
    cax = ax.inset_axes((1.0 + bar_fraction * 1.2, 0.0, bar_fraction, 1.0))
    bar = fig.colorbar(im, cax=cax)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=FONT_PT["tick"], length=2, width=0.6, pad=1.5)
    if center == 0:
        from matplotlib.ticker import MaxNLocator  # type: ignore

        bar.locator = MaxNLocator(nbins=4, symmetric=True)
        bar.update_ticks()
    if colorbar_label:
        bar.set_label(colorbar_label, fontsize=FONT_PT["body"], labelpad=3)
    return _save(fig, out)


# Okabe-Ito vermillion/blue for direct/inverted blocks (colour-blind safe), and
# group tags from the remaining hues so the two never share a colour.
_LGA_SYNTENY_DIRECT = _PALETTE[5]
_LGA_SYNTENY_INVERTED = _PALETTE[0]
_LGA_SYNTENY_GROUP_CYCLE = (_PALETTE[2], _PALETTE[3], _PALETTE[1], _PALETTE[4], _PALETTE[7])
_LGA_SYNTENY_FRAME = "#9AA3AD"


def plot_synteny_matrix(
    blocks: Sequence[Mapping[str, str | int | float | None]] | Mapping[str, Sequence[Mapping[str, str | int | float | None]] | Mapping[str, Sequence[Mapping[str, str | int | float | None]]]],
    *,
    labels: Sequence[str] | None = None,
    groups: Mapping[str, str] | None = None,
    genome_lengths: Mapping[str, int | float] | None = None,
    title: str | None = None,
    direct_color: str = _LGA_SYNTENY_DIRECT,
    inverted_color: str = _LGA_SYNTENY_INVERTED,
) -> OrganellePlot:
    """Prepare a synteny matrix without writing; render it with ``ov.write``."""
    data = _synteny_matrix_data(blocks, labels=labels, genome_lengths=genome_lengths)

    def _render(output: str | Path) -> Path:
        return _render_synteny_matrix(
            blocks,
            output,
            labels=labels,
            groups=groups,
            genome_lengths=genome_lengths,
            title=title,
            direct_color=direct_color,
            inverted_color=inverted_color,
        )

    plot = plot_result(
        "plot_synteny_matrix",
        _render,
        metrics={"genomes": len(data["labels"]), "blocks": len(data["rows"])},
        summary="Synteny matrix prepared.",
    )
    plot._panel = renderer_panel(_render)
    return plot


def _lga_synteny_group_colors(
    names: Sequence[str], groups: Mapping[str, str] | None
) -> dict[str, str]:
    """Group -> tag colour; the historic named groups keep their colours."""
    colors: dict[str, str] = {}
    cycle = iter(itertools.cycle(_LGA_SYNTENY_GROUP_CYCLE))
    for group in _ordered_groups(names, groups):
        colors[group] = _SYNTENY_GROUP_COLORS.get(group) or next(cycle)
    return colors


def _render_synteny_matrix(
    blocks: Sequence[Mapping[str, str | int | float | None]] | Mapping[str, Sequence[Mapping[str, str | int | float | None]] | Mapping[str, Sequence[Mapping[str, str | int | float | None]]]],
    output: str | Path,
    *,
    labels: Sequence[str] | None = None,
    groups: Mapping[str, str] | None = None,
    genome_lengths: Mapping[str, int | float] | None = None,
    title: str | None = None,
    direct_color: str = _LGA_SYNTENY_DIRECT,
    inverted_color: str = _LGA_SYNTENY_INVERTED,
) -> Path:
    """Render a lower-triangle whole-genome pairwise synteny dotplot matrix.

    Each block row should name two genomes (``genome_a``/``genome_b`` or
    ``query``/``target`` aliases), coordinates on each genome, and optionally
    ``orientation``/``strand`` as direct or inverted. Row genomes are named on
    the left and column genomes below, each with its length; a coloured tag
    marks the group of each genome when ``groups`` is given.
    """
    data = _synteny_matrix_data(
        blocks,
        labels=labels,
        genome_lengths=genome_lengths,
    )
    if len(data["labels"]) < 2:
        raise ValueError("synteny matrix needs at least two genomes")
    out = _prepare_output(output)
    _apply_figure_style()

    names = data["labels"]
    rows = data["rows"]
    lengths = data["lengths"]
    index = {name: i for i, name in enumerate(names)}
    n = len(names)
    fig_size = max(3.4, min(7.2, 0.62 * n + 1.4))
    fig, ax = new_axes(journal_figsize(fig_size, fig_size))

    for i in range(1, n):
        y = n - 1 - i
        for j in range(i):
            ax.add_patch(
                Rectangle((j, y), 1, 1, fill=False, edgecolor=_LGA_SYNTENY_FRAME, linewidth=0.5)
            )

    for row in rows:
        a = row["genome_a"]
        b = row["genome_b"]
        ia = index[a]
        ib = index[b]
        if ia == ib:
            continue
        row_name = a if ia > ib else b
        col_name = b if ia > ib else a
        row_index = index[row_name]
        col_index = index[col_name]
        x_start, x_end = row["coords"][col_name]
        y_start, y_end = row["coords"][row_name]
        x_len = max(1.0, float(lengths[col_name]))
        y_len = max(1.0, float(lengths[row_name]))
        x0 = col_index + _clamp01(x_start / x_len)
        x1 = col_index + _clamp01(x_end / x_len)
        y_base = n - 1 - row_index
        y0 = y_base + _clamp01(y_start / y_len)
        y1 = y_base + _clamp01(y_end / y_len)
        orientation = row["orientation"]
        slope_product = (x1 - x0) * (y1 - y0)
        if (orientation == "inverted" and slope_product > 0) or (
            orientation == "direct" and slope_product < 0
        ):
            y0, y1 = y1, y0
        color = inverted_color if orientation == "inverted" else direct_color
        ax.plot([x0, x1], [y0, y1], color=color, lw=0.9, solid_capstyle="round")

    group_map = {str(name): str((groups or {}).get(name, "")) for name in names}
    group_colors = _lga_synteny_group_colors(names, groups)
    tag = 0.045  # group tag thickness, in cells
    gap = 0.03
    long_names = max(len(name) for name in names) > 9
    for i, name in enumerate(names):
        length_text = _format_bp(int(round(lengths[name])))
        color = group_colors.get(group_map[name])
        if i >= 1:  # row label on the left of the first column
            y_mid = n - 1 - i + 0.5
            if color:
                ax.add_patch(
                    Rectangle((-gap - tag, n - 1 - i + 0.04), tag, 0.92, color=color, lw=0,
                              clip_on=False)
                )
            x_text = -gap - (tag + gap if color else 0.0)
            ax.text(x_text, y_mid + 0.03, name, ha="right", va="bottom",
                    fontsize=FONT_PT["body"], color="#1F2933")
            ax.text(x_text, y_mid - 0.03, length_text, ha="right", va="top",
                    fontsize=FONT_PT["label"], color=_NEUTRAL)
        if i <= n - 2:  # column label under the bottom row
            x_mid = i + 0.5
            if color:
                ax.add_patch(
                    Rectangle((i + 0.04, -gap - tag), 0.92, tag, color=color, lw=0, clip_on=False)
                )
            y_text = -gap - (tag + gap if color else 0.0)
            if long_names:
                ax.text(x_mid + 0.25, y_text, f"{name} ({length_text})", ha="right", va="top",
                        rotation=35, rotation_mode="anchor", fontsize=FONT_PT["body"],
                        color="#1F2933")
            else:
                ax.text(x_mid, y_text, name, ha="center", va="top",
                        fontsize=FONT_PT["body"], color="#1F2933")
                ax.text(x_mid, y_text - 0.13, length_text, ha="center", va="top",
                        fontsize=FONT_PT["label"], color=_NEUTRAL)

    span = n - 1
    # Keep the genome names inside the axes so a composed panel reserves room
    # for them: margins are the measured label sizes over the cell size.
    from .style import text_width_pt

    right_pad = 1.25 if n == 2 else 0.02
    axes_pt = min(ax.get_position().width * fig.get_figwidth(),
                  ax.get_position().height * fig.get_figheight()) * 72
    cell_pt = max(1.0, axes_pt / (span + right_pad + 0.6))
    widest = max(text_width_pt(name, FONT_PT["body"]) for name in names)
    left_units = (widest + 6.0) / cell_pt + (tag + 2 * gap if group_colors else gap)
    if long_names:
        bottom_pt = widest * 0.6 + 12.0
    else:
        bottom_pt = FONT_PT["body"] + FONT_PT["label"] + 8.0
    bottom_units = bottom_pt / cell_pt + (tag + 2 * gap if group_colors else gap)
    ax.set_xlim(-left_units, span + right_pad)
    ax.set_ylim(-bottom_units, span + 0.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    if title:
        _panel_title(ax, title)

    handles = [
        Rectangle((0, 0), 1, 1, color=color, lw=0, label=group)
        for group, color in group_colors.items()
    ]
    handles += [
        Line2D([0], [0], color=direct_color, lw=1.2, label="Direct"),
        Line2D([0], [0], color=inverted_color, lw=1.2, label="Inverted"),
    ]
    # The empty upper triangle holds the key (beside the single cell when n == 2).
    ax.legend(
        handles=handles,
        loc="upper right" if n > 2 else "upper left",
        bbox_to_anchor=(span, span) if n > 2 else (span + 0.08, span),
        bbox_transform=ax.transData,
        frameon=False,
        handlelength=1.4,
        handleheight=0.7,
        borderaxespad=0.0,
        labelspacing=0.35,
    )
    return _save(fig, out)


def summarize_synteny_matrix(
    blocks: Sequence[Mapping[str, str | int | float | None]] | Mapping[str, Sequence[Mapping[str, str | int | float | None]]],
    *,
    labels: Sequence[str] | None = None,
    genome_lengths: Mapping[str, int | float] | None = None,
) -> dict[str, int | list[str] | dict[str, int | float]]:
    """Return summary metrics for whole-genome synteny matrix inputs."""
    data = _synteny_matrix_data(blocks, labels=labels, genome_lengths=genome_lengths)
    direct = sum(1 for row in data["rows"] if row["orientation"] == "direct")
    inverted = sum(1 for row in data["rows"] if row["orientation"] == "inverted")
    pair_keys = {
        tuple(sorted((row["genome_a"], row["genome_b"])))
        for row in data["rows"]
        if row["genome_a"] != row["genome_b"]
    }
    return {
        "genomes": len(data["labels"]),
        "pairs": len(pair_keys),
        "blocks": len(data["rows"]),
        "direct_blocks": direct,
        "inverted_blocks": inverted,
        "labels": data["labels"],
        "genome_lengths": data["lengths"],
    }


def plot_network(
    nodes: Sequence[Mapping[str, str | int | float | bool | None]],
    edges: Sequence[Mapping[str, str | int | float | bool | None]],
    *,
    title: str | None = None,
    node_groups: Mapping[str, str] | None = None,
) -> OrganellePlot:
    """Render a weighted labelled network with deterministic circular layout (compute-only)."""
    if not nodes:
        raise ValueError("nodes must contain at least one node")
    node_ids = [str(node.get("id", "")) for node in nodes]
    if any(not node_id for node_id in node_ids):
        raise ValueError("each node must define a non-empty id")
    known = set(node_ids)
    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source not in known or target not in known:
            missing = target if source in known else source
            raise ValueError(f"edge references unknown node: {missing}")

    def _render(path: str | Path) -> Path:
        return _render_network(nodes, edges, node_ids, path, title=title, node_groups=node_groups)

    plot = plot_result(
        "plot_network",
        _render,
        metrics={"nodes": len(node_ids), "edges": len(edges)},
        summary="Network plot prepared.",
    )
    plot._panel = renderer_panel(_render)
    return plot


#: Node colours: Okabe-Ito without the yellow, which vanishes on white.
_NET_GROUP_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#6B7280")
_NET_EDGE = "#8B95A1"
_NET_UNGROUPED = "#52606D"


def _net_label_anchor(x: float, y: float) -> tuple[str, str]:
    """Horizontal/vertical alignment that puts a label radially outside its node."""
    ha = "left" if x > 0.3 else "right" if x < -0.3 else "center"
    va = "bottom" if y > 0.3 else "top" if y < -0.3 else "center"
    return ha, va


def _render_network(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    node_ids: list[str],
    output: str | Path,
    *,
    title: str | None = None,
    node_groups: Mapping[str, str] | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_network`."""
    from .style import text_width_pt

    out = _prepare_output(output)
    _apply_figure_style()

    groups = {
        node_id: str(node_groups.get(node_id, "")) if node_groups else str(node.get("group", ""))
        for node_id, node in zip(node_ids, nodes, strict=False)
    }
    positions = _network_positions(node_ids, groups)
    named_groups = sorted({group for group in groups.values() if group})
    group_palette = {
        group: _NET_GROUP_COLORS[i % len(_NET_GROUP_COLORS)] for i, group in enumerate(named_groups)
    }
    group_palette[""] = _NET_UNGROUPED

    # legend rows below the circle: one column per entry while the row fits 89 mm
    weights = [abs(_float(edge.get("weight", 1.0))) for edge in edges]
    w_max = max(weights, default=1.0) or 1.0
    w_min = min(weights, default=w_max)
    show_groups = len(named_groups) > 1
    show_weights = bool(edges) and w_max - w_min > 1e-9
    entries = (named_groups if show_groups else []) + (
        [f"|weight| {w_min:.2g}", f"|weight| {w_max:.2g}"] if show_weights else []
    )
    legend_pt = [text_width_pt(entry, FONT_PT["label"]) + 20 for entry in entries]
    ncol = max(1, len(entries))
    while ncol > 1 and sum(sorted(legend_pt, reverse=True)[:ncol]) > 89 / 25.4 * 72 * 0.95:
        ncol -= 1
    legend_rows = -(-len(entries) // ncol) if entries else 0

    n_nodes = len(node_ids)
    base_area = 42.0 if n_nodes <= 20 else 26.0 if n_nodes <= 40 else 14.0
    width_in = 89 / 25.4
    fig, ax = new_axes(journal_figsize(width_in, width_in * 0.86 + legend_rows * 0.13))
    _panel_title(ax, title)

    def _line_width(weight: float) -> float:
        return 0.35 + 1.65 * weight / w_max

    for edge, weight in sorted(zip(edges, weights, strict=False), key=lambda item: item[1]):
        x1, y1 = positions[str(edge["source"])]
        x2, y2 = positions[str(edge["target"])]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=_NET_EDGE,
            lw=_line_width(weight),
            alpha=0.55,
            solid_capstyle="round",
            zorder=1,
        )
    for node, node_id in zip(nodes, node_ids, strict=False):
        x, y = positions[node_id]
        area = base_area * max(0.7, _float(node.get("size", 1.0)))
        ax.scatter(
            [x],
            [y],
            s=area,
            color=group_palette[groups[node_id]],
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        radius_pt = sqrt(area) / 2 + 2.0
        ha, va = _net_label_anchor(x, y)
        ax.annotate(
            str(node.get("label", node_id)),
            (x, y),
            xytext=(radius_pt * x, radius_pt * y),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=FONT_PT["label"],
            zorder=3,
        )
    if entries:
        handles: list[Any] = []
        if show_groups:
            handles += [
                Line2D(
                    [], [], ls="", marker="o", markersize=4.5, color=group_palette[group],
                    markeredgecolor="white", markeredgewidth=0.4, label=group,
                )
                for group in named_groups
            ]
        if show_weights:
            handles += [
                Line2D([], [], color=_NET_EDGE, alpha=0.8, lw=_line_width(w), label=label)
                for w, label in zip((w_min, w_max), entries[-2:], strict=True)
            ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=ncol,
            fontsize=FONT_PT["label"],
            handlelength=1.4,
            handletextpad=0.4,
            columnspacing=1.0,
            borderaxespad=0.0,
            frameon=False,
        )
    ax.set_aspect("equal")
    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.22, 1.22)
    ax.set_axis_off()
    return _save(fig, out)


def plot_selection_summary(
    records: Sequence[Mapping[str, str | int | float | bool | None]],
    *,
    title: str | None = None,
) -> OrganellePlot:
    """Render a gene-level selection-pressure summary (compute-only)."""
    if not records:
        raise ValueError("records must contain at least one selection record")

    def _render(path: str | Path) -> Path:
        return _render_selection_summary(records, path, title=title)

    plot = plot_result(
        "plot_selection_summary",
        _render,
        metrics={"records": len(records)},
        summary="Selection-pressure summary prepared.",
    )
    plot._panel = renderer_panel(_render)
    return plot


_SEL_BASE = "#7BA7CC"  # light member of the Okabe-Ito blue family
_SEL_SIGNIFICANT = _ALARM


def _sel_tick_labels(ax: Any, labels: Sequence[str], axes_width_pt: float) -> None:
    """Italic gene names, upright while they fit their slot, else at 45 degrees."""
    from .style import text_width_pt

    size = FONT_PT["tick"]
    slot = axes_width_pt / max(1, len(labels))
    widest = max(text_width_pt(label, size, italic=True) for label in labels)
    if widest <= 0.92 * slot:
        ax.set_xticks(range(len(labels)), labels, fontstyle="italic")
    else:
        ax.set_xticks(
            range(len(labels)), labels, fontstyle="italic",
            rotation=45, ha="right", rotation_mode="anchor",
        )


def _render_selection_summary(
    records: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_selection_summary`."""
    from matplotlib.patches import Patch  # type: ignore
    from matplotlib.ticker import MaxNLocator  # type: ignore
    from matplotlib.transforms import blended_transform_factory  # type: ignore

    out = _prepare_output(output)
    _apply_figure_style()

    genes = [str(r.get("gene", f"gene {i + 1}")) for i, r in enumerate(records)]
    omega = [_float(r.get("omega", r.get("kaks", 0.0))) for r in records]
    pvals = [_float(r.get("p_value", r.get("p", 1.0))) for r in records]
    sites = [_float(r.get("selected_sites", 0.0)) for r in records]
    significant = [p < 0.05 for p in pvals]
    colors = [_SEL_SIGNIFICANT if flag else _SEL_BASE for flag in significant]

    n = len(records)
    width_mm = 89.0 if n <= 10 else 120.0 if n <= 22 else 183.0
    fig, ax = new_axes(journal_figsize(width_mm / 25.4, 56 / 25.4))
    x = list(range(n))
    ax.bar(x, omega, color=colors, width=0.68, linewidth=0, zorder=2)
    ax.axhline(1.0, color=_NEUTRAL, lw=0.6, ls=(0, (3, 2)), zorder=1)
    # reference label inside the axes above a short end bar (so it stays inside a
    # composed panel), otherwise just right of the axes
    if max(omega[-2:]) < 0.85:
        ref_x, ref_ha, ref_va, ref_text = 1.0, "right", "bottom", "\u03c9 = 1"
    elif max(omega[:2]) < 0.85:
        ref_x, ref_ha, ref_va, ref_text = 0.0, "left", "bottom", "\u03c9 = 1"
    else:
        ref_x, ref_ha, ref_va, ref_text = 1.0, "left", "center", " \u03c9 = 1"
    ax.annotate(
        ref_text,
        (ref_x, 1.0),
        xycoords=blended_transform_factory(ax.transAxes, ax.transData),
        xytext=(0, 1.0 if ref_va == "bottom" else 0),
        textcoords="offset points",
        ha=ref_ha,
        va=ref_va,
        fontsize=FONT_PT["tick"],
        color=_NEUTRAL,
        annotation_clip=False,
    )
    top = max(max(omega), 1.0)
    for xi, value, n_sites in zip(x, omega, sites, strict=False):
        if n_sites:
            count = int(n_sites)
            ax.annotate(
                f"{count}",
                (xi, value),
                xytext=(0, 1.5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=FONT_PT["label"],
                color="#1F2933",
                bbox={"boxstyle": "square,pad=0.08", "fc": "white", "ec": "none"},
                zorder=3,
            )
    _sel_tick_labels(ax, genes, width_mm / 25.4 * 72 * 0.82)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylabel("Ka/Ks (\u03c9)")
    ax.set_ylim(0, top * 1.16)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    _panel_title(ax, title)
    _despine(ax)
    handles = []
    if any(significant):
        handles.append(Patch(color=_SEL_SIGNIFICANT, label="P < 0.05"))
        if not all(significant):
            handles.append(Patch(color=_SEL_BASE, label="P \u2265 0.05"))
    if any(sites):
        handles.append(
            Line2D([], [], ls="", marker="$n$", color="#1F2933", markersize=4.5,
                   label="Sites under selection")
        )
    if handles:
        ax.legend(
            handles=handles,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=len(handles),
            fontsize=FONT_PT["label"],
            handlelength=1.0,
            handleheight=0.8,
            handletextpad=0.4,
            columnspacing=1.0,
            borderaxespad=0.2,
            frameon=False,
        )
    return _save(fig, out)


def plot_transfer_schematic(
    events: Sequence[Mapping[str, str | int | float | bool | None]],
    *,
    title: str | None = None,
) -> OrganellePlot:
    """Render donor-recipient transfer events as linked coordinate bars (compute-only)."""
    if not events:
        raise ValueError("events must contain at least one transfer event")

    def _render(path: str | Path) -> Path:
        return _render_transfer_schematic(events, path, title=title)

    plot = plot_result(
        "plot_transfer_schematic",
        _render,
        metrics={"events": len(events)},
        summary="Transfer schematic prepared.",
    )
    plot._panel = renderer_panel(_render)
    return plot


_LGA_TRANSFER_BAR = "#DDE2E7"
_LGA_TRANSFER_DONOR_COLORS = (_PALETTE[0], _PALETTE[2], _PALETTE[5], _PALETTE[3], _PALETTE[1])


def _lga_transfer_span(
    event: Mapping[str, Any], role: str, maximum: float
) -> tuple[float, float]:
    start = _float(event.get(f"{role}_start", 0.0))
    end = _float(event.get(f"{role}_end", start))
    low, high = sorted((start, end))
    scale = max(1.0, maximum)
    return max(0.0, min(1.0, low / scale)), max(0.0, min(1.0, high / scale))


def _render_transfer_schematic(
    events: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_transfer_schematic`.

    Each genome is a bar scaled to its own length: ``donor_length`` /
    ``recipient_length`` in an event when given (the length is then printed at
    the bar end), otherwise the largest coordinate seen. Transferred segments
    are highlighted on both bars, arrows are coloured by donor genome and each
    event is labeled next to its recipient segment.
    """
    from matplotlib.path import Path as MplPath  # type: ignore

    from .style import text_width_pt

    out = _prepare_output(output)
    _apply_figure_style()

    entities = _ordered_entities(events)
    maxima = _entity_maxima(events, entities)
    declared: dict[str, float] = {}
    for event in events:
        for role in ("donor", "recipient"):
            value = event.get(f"{role}_length")
            if value not in (None, ""):
                name = str(event.get(role, role))
                declared[name] = max(declared.get(name, 0.0), _float(value))
    for name, value in declared.items():
        maxima[name] = max(maxima[name], value)
    y_pos = {entity: float(i) for i, entity in enumerate(reversed(entities))}
    donors = list(dict.fromkeys(str(event.get("donor", "donor")) for event in events))
    donor_color = {
        name: _LGA_TRANSFER_DONOR_COLORS[i % len(_LGA_TRANSFER_DONOR_COLORS)]
        for i, name in enumerate(donors)
    }
    fig, ax = new_axes(journal_figsize(7.2, 0.62 * len(entities) + 0.55))
    _panel_title(ax, title)
    axes_width_pt = max(1.0, 0.8 * ax.get_position().width * fig.get_figwidth() * 72)
    unit_per_pt = 1.25 / axes_width_pt  # x spans -0.2..1.05

    for entity in entities:
        y = y_pos[entity]
        ax.plot([0, 1], [y, y], color=_LGA_TRANSFER_BAR, lw=3.2, solid_capstyle="round",
                zorder=1)
        ax.text(-0.03, y, entity, ha="right", va="center", fontsize=FONT_PT["body"])
        if entity in declared:
            ax.text(1.02, y, _format_bp(int(declared[entity])), ha="left", va="center",
                    fontsize=FONT_PT["label"], color=_NEUTRAL)

    placed: dict[tuple[float, int, int], list[tuple[float, float]]] = {}
    for i, event in enumerate(events):
        donor = str(event.get("donor", "donor"))
        recipient = str(event.get("recipient", "recipient"))
        color = donor_color[donor]
        dy = y_pos[donor]
        ry = y_pos[recipient]
        d0, d1 = _lga_transfer_span(event, "donor", maxima[donor])
        r0, r1 = _lga_transfer_span(event, "recipient", maxima[recipient])
        for (s0, s1), y in (((d0, d1), dy), ((r0, r1), ry)):
            if s1 - s0 < 0.006:
                mid = (s0 + s1) / 2
                s0, s1 = mid - 0.003, mid + 0.003
            ax.plot([s0, s1], [y, y], color=color, lw=3.2, solid_capstyle="butt", zorder=2)
        dx = (d0 + d1) / 2
        rx = (r0 + r1) / 2
        side = 1 if dy > ry else -1  # the recipient side facing the donor
        # an S-curve leaving and entering the bars vertically never overshoots them
        y_start, y_end = dy - side * 0.08, ry + side * 0.08
        y_mid = (y_start + y_end) / 2
        ax.add_patch(
            FancyArrowPatch(
                path=MplPath(
                    [(dx, y_start), (dx, y_mid), (rx, y_mid), (rx, y_end)],
                    [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
                ),
                arrowstyle="-|>",
                mutation_scale=6,
                lw=0.8,
                color=color,
                shrinkA=0,
                shrinkB=0,
                zorder=3,
            )
        )
        label = str(event.get("label", f"event {i + 1}"))
        width = text_width_pt(label, FONT_PT["label"]) * unit_per_pt
        # Beside the arrow head, on the side the curve does not come from; a
        # label that would touch one already placed moves to a second row.
        from_left = dx <= rx
        left = rx + 0.012 if from_left else rx - 0.012 - width
        left = min(max(left, -0.19), 1.04 - width)
        tier = 0
        for tier in range(3):
            band = placed.setdefault((ry, side, tier), [])
            if all(left + width + 0.01 <= a or left - 0.01 >= b for a, b in band):
                break
        band.append((left, left + width))
        ax.text(
            left,
            ry + side * (0.2 + 0.17 * tier),
            label,
            ha="left",
            va="center",
            fontsize=FONT_PT["label"],
            color="#1F2933",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4},
            zorder=4,
        )
    ax.set_xlim(-0.2, 1.05 + (0.08 if declared else 0.0))
    ax.set_ylim(-0.55, len(entities) - 0.45)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax.spines["bottom"].set_bounds(0, 1)
    ax.set_xlabel("Relative position in genome (%)")
    ax.set_yticks([])
    _despine(ax)
    ax.spines["left"].set_visible(False)
    return _save(fig, out)


def ideogram(
    karyotype: Sequence[Mapping[str, str | int | float | None]] | Mapping[str, int | float],
    *,
    overlaid: Sequence[Mapping[str, Any]] | None = None,
    label: Sequence[Mapping[str, Any]] | None = None,
    label_names: Mapping[str, str] | None = None,
    profile: Sequence[Mapping[str, Any]] | None = None,
    title: str | None = None,
    max_mb: float | None = None,
    show_lengths: bool = False,
    chromosome_width: float | None = None,
    density_label: str | None = None,
    profile_labels: Sequence[str] | None = None,
) -> OrganelleResult:
    """Prepare a RIdeogram-style chromosome plot object without writing files.

    ``density_label`` names the overlaid heatmap in its colour key (default
    "Density") and ``profile_labels`` names the profile's filled area and
    line, in that order, in the key.
    """
    chrom_rows = _numt_chromosome_rows(karyotype)
    if not chrom_rows:
        raise ValueError("karyotype must contain at least one chromosome")
    density_rows = [_ideogram_density_row(row) for row in (overlaid or ())]
    label_rows = [_ideogram_label_row(row) for row in (label or ())]
    profile_rows = [_ideogram_profile_row(row) for row in (profile or ())]
    label_types = _ideogram_marker_types(label_rows)
    if len(label_types) > 10:
        raise ValueError("ideogram supports at most 10 label classes")
    resolved_label_names = _ideogram_label_names(label_rows, label_names)
    chrom_width = _ideogram_chromosome_width(chromosome_width)
    metrics = {
        "plot_kind": "ideogram",
        "chromosomes": chrom_rows,
        "density": density_rows,
        "label": label_rows,
        "label_names": resolved_label_names,
        "profile": profile_rows,
        "chromosome_count": len(chrom_rows),
        "density_count": len(density_rows),
        "label_count": len(label_rows),
        "label_types": label_types,
        "profile_count": len(profile_rows),
        "max_mb": max_mb,
        "title": title,
        "show_lengths": bool(show_lengths),
        "length_axis": "left" if show_lengths else "hidden",
        "chromosome_width": chrom_width,
        "density_label": density_label,
        "profile_labels": [str(value) for value in profile_labels] if profile_labels else None,
    }
    return data_plot_result(
        "ideogram",
        organelle="nuclear",
        metrics=metrics,
        key_findings=(
            {"metric": "chromosomes", "value": len(chrom_rows)},
            {"metric": "density_bins", "value": len(density_rows)},
            {"metric": "labels", "value": len(label_rows)},
            {"metric": "profile_points", "value": len(profile_rows)},
        ),
        flags=("ideogram_prepared",),
        summary=(
            f"Ideogram prepared for {len(chrom_rows)} chromosome(s), "
            f"{len(density_rows)} density bin(s)."
        ),
        method="matplotlib_ideogram_data",
    )


def write_ideogram(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
    *,
    width: float | None = None,
    height: float | None = None,
) -> Path:
    """Render a prepared RIdeogram-style chromosome plot object."""
    metrics = _result_metrics(result)
    return _render_ideogram(
        metrics.get("chromosomes", ()),
        metrics.get("density", ()),
        metrics.get("label", ()),
        metrics.get("profile", ()),
        output,
        title=metrics.get("title"),
        max_mb=metrics.get("max_mb"),
        show_lengths=bool(metrics.get("show_lengths", False)),
        label_names=metrics.get("label_names", {}),
        chromosome_width=metrics.get("chromosome_width"),
        figure_width=width,
        figure_height=height,
        density_label=metrics.get("density_label"),
        profile_labels=metrics.get("profile_labels"),
    )


def nuclear_transfer_ideogram(
    chromosomes: Sequence[Mapping[str, str | int | float | None]] | Mapping[str, int | float],
    fragments: (
        Sequence[Mapping[str, str | int | float | None]]
        | Mapping[
            str,
            Sequence[Mapping[str, str | int | float | None]]
            | Mapping[str, Sequence[Mapping[str, str | int | float | None]]],
        ]
    ),
    gene_hits: Sequence[Mapping[str, Any]] | None = None,
    *,
    title: str | None = None,
    panel_label: str | None = None,
    max_mb: float | None = None,
    transfer_label: str | None = None,
    plastid_fragments: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    color_by: str = "organelle",
    show_density: bool = True,
    density_window: int = 500_000,
    label_top: int = 0,
    centromeres: Mapping[str, Sequence[float]] | None = None,
    chromosome_order: Sequence[str] | None = None,
    nuclear_annotated: bool | None = None,
) -> OrganelleResult:
    """Prepare a NUMT/NUPT chromosome ideogram without writing files.

    Chromosomes are drawn vertically in natural order (or ``chromosome_order``,
    which also selects them), position 0 at the top, in Mb. Each insertion is a
    thin tick outside the chromosome: NUMT on the left, NUPT on the right, tick
    length/width by fragment length; the chromosome body is filled with the
    insertion density per ``density_window`` bp. Use
    :func:`write_nuclear_transfer_ideogram` to render the prepared result.

    Parameters
    ----------
    chromosomes : {name: length} or rows
        Chromosome lengths (bp or Mb); rows may carry ``centromere_start``/
        ``centromere_end``. Fragments on other sequences are counted as
        unplaced (``metrics["unplaced_candidates"]``) and not drawn.
    fragments : rows or transfer Result
        NUMT/NUPT candidates (``detect_transfers*`` Result or its candidate
        rows) or legacy fragment rows (``chromosome``, ``start``/``end``).
    gene_hits : rows, optional
        Legacy explicit gene markers (``class`` ``new_gene``/``un_gene`` are
        shown as within nuclear gene / intergenic).
    plastid_fragments : rows or transfer Result, optional
        NUPT candidates; ``fragments`` is then the NUMT set, drawn on the left.
    color_by : {"organelle", "evidence_level", "identity", "organelle_gene"}
        Tick colour. ``evidence_level`` names the validation layers;
        ``identity`` is continuous (recent insertions are near 100 %).
    label_top : int
        Annotate the N longest insertions with length and organelle gene.
    centromeres : {name: (start_bp, end_bp)}, optional
        Centromere spans drawn as a constriction.
    nuclear_annotated : bool, optional
        Whether a nuclear annotation was used. ``None`` infers it from any
        candidate naming a ``nuclear_gene``; without annotation no locus
        markers are drawn. With it, insertions are marked as within a nuclear
        gene or intergenic.
    """
    from . import transfer_plots as tp

    legacy_rows = _numt_chromosome_rows(chromosomes)
    if not legacy_rows:
        raise ValueError("chromosomes must contain at least one chromosome")
    chrom_rows = tp.normalize_chromosomes(
        (_transfer_ideogram_chromosome(row) for row in legacy_rows),
        order=chromosome_order,
        centromeres=centromeres,
    )
    sets = tp.build_transfer_sets(
        fragments, primary_label=transfer_label, plastid=plastid_fragments
    )
    hit_rows = [_numt_gene_hit_row(hit) for hit in (gene_hits or ())]
    annotated = bool(hit_rows) or tp.resolve_gene_annotation(sets, nuclear_annotated)
    placement = tp.placement_summary(sets, chrom_rows)
    resolved_transfer_label = (
        "NUMT + NUPT" if len(sets) > 1 else (transfer_label or sets[0]["transfer"])
    )
    fragment_rows = [_transfer_ideogram_fragment(row) for row in sets[0]["rows"]]
    plastid_rows = (
        [_transfer_ideogram_fragment(row) for row in sets[1]["rows"]] if len(sets) > 1 else []
    )
    metrics = {
        "plot_kind": "nuclear_transfer_ideogram",
        "chromosomes": [_transfer_ideogram_chromosome_metrics(row) for row in chrom_rows],
        "fragments": fragment_rows,
        "plastid_fragments": plastid_rows,
        "gene_hits": hit_rows,
        "chromosome_count": len(chrom_rows),
        "fragment_count": len(fragment_rows) + len(plastid_rows),
        "gene_hit_count": len(hit_rows),
        "transfer_label": resolved_transfer_label,
        "transfers": [item["transfer"] for item in sets],
        "panel_label": panel_label,
        "max_mb": max_mb,
        "title": title,
        "color_by": color_by,
        "show_density": bool(show_density),
        "density_window": int(density_window),
        "label_top": int(label_top),
        "nuclear_annotated": annotated,
        "drawn_fragment_count": placement["drawn_candidates"],
        "unplaced_candidates": placement["unplaced_candidates"],
        "unplaced_seqids": placement["unplaced_seqids"],
        "by_transfer": placement["by_transfer"],
        "note": placement["note"],
    }
    return data_plot_result(
        "nuclear_transfer_ideogram",
        organelle="nuclear",
        metrics=metrics,
        key_findings=(
            {"metric": "chromosomes", "value": len(chrom_rows)},
            {"metric": "fragments", "value": metrics["fragment_count"]},
            {"metric": "unplaced_fragments", "value": placement["unplaced_candidates"]},
            {"metric": "gene_hits", "value": len(hit_rows)},
        ),
        flags=("nuclear_transfer_ideogram_prepared",),
        summary=(
            f"{resolved_transfer_label} ideogram prepared for {len(chrom_rows)} "
            f"chromosome(s), {metrics['fragment_count']} fragment(s)."
        ),
        method="matplotlib_ideogram_data",
    )


def write_nuclear_transfer_ideogram(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
    *,
    width: float | None = None,
    height: float | None = None,
) -> Path:
    """Render a prepared nuclear-transfer ideogram to an image file."""
    metrics = _result_metrics(result)
    return _render_nuclear_transfer_ideogram(
        metrics.get("chromosomes", ()),
        metrics.get("fragments", ()),
        metrics.get("gene_hits", ()),
        output,
        **_ideogram_options(metrics),
        figure_width=width,
        figure_height=height,
    )


def plot_nuclear_transfer_ideogram(
    result: OrganelleResult | Mapping[str, Any],
    *,
    width: str | float | None = None,
    height: float | None = None,
) -> OrganellePlot:
    """Plot form of a prepared :func:`nuclear_transfer_ideogram` (compute-only).

    Renders with ``ov.write(plot, output)`` like the other plots and can be a
    panel of :func:`~organelleverse.visualization.compose_figure`.
    """
    from . import transfer_plots as tp

    metrics = _result_metrics(result)
    spec = _nuclear_transfer_ideogram_spec(
        metrics.get("chromosomes", ()),
        metrics.get("fragments", ()),
        metrics.get("gene_hits", ()),
        **_ideogram_options(metrics),
        figure_width=width,
        figure_height=height,
    )

    def _render(path: str | Path) -> Path:
        return tp.render_transfer_figure(spec, path)

    plot = plot_result(
        "plot_nuclear_transfer_ideogram",
        _render,
        organelle="nuclear",
        metrics={
            key: metrics.get(key)
            for key in (
                "transfer_label",
                "chromosome_count",
                "fragment_count",
                "drawn_fragment_count",
                "unplaced_candidates",
                "note",
            )
        }
        | {"caption": tp.transfer_caption(spec)},
        summary=str(metrics.get("note") or "Transfer ideogram prepared."),
    )
    plot._panel = tp.transfer_panel(spec)
    return plot


def _ideogram_options(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "title": metrics.get("title"),
        "panel_label": metrics.get("panel_label"),
        "max_mb": metrics.get("max_mb"),
        "transfer_label": str(metrics.get("transfer_label", "NUMT")),
        "plastid_fragments": metrics.get("plastid_fragments") or None,
        "color_by": str(metrics.get("color_by") or "organelle"),
        "show_density": bool(metrics.get("show_density", True)),
        "density_window": int(metrics.get("density_window") or 500_000),
        "label_top": int(metrics.get("label_top") or 0),
        "nuclear_annotated": metrics.get("nuclear_annotated"),
        "note": metrics.get("note"),
    }


def save_plot(
    plot: OrganelleResult | Mapping[str, Any],
    output: str | Path,
    *,
    width: float | None = None,
    height: float | None = None,
) -> Path:
    """Save a prepared OrganelleVerse plot object, similar to R ``ggsave``."""
    metrics = _result_metrics(plot)
    kind = str(metrics.get("plot_kind", ""))
    if kind == "ideogram":
        return write_ideogram(plot, output, width=width, height=height)
    if kind == "nuclear_transfer_ideogram":
        return write_nuclear_transfer_ideogram(plot, output, width=width, height=height)
    raise ValueError(f"unsupported plot kind for save_plot(): {kind or 'unknown'}")


def _result_metrics(
    value: OrganelleResult | Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return plot metrics as built-in containers the renderers can consume."""
    if isinstance(value, OrganelleResult):
        return cast(Mapping[str, Any], thaw_json(value.metrics))
    return value


_LGA_IDEOGRAM_OUTLINE = "#7A858C"
_LGA_IDEOGRAM_LEGEND_UNITS = 1.7  # width of the key column, in chromosome slots


def _render_ideogram(
    chromosomes: Sequence[Mapping[str, Any]] | Mapping[str, int | float],
    density: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    profile: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    title: str | None = None,
    max_mb: float | None = None,
    show_lengths: bool = False,
    label_names: Mapping[str, str] | None = None,
    chromosome_width: float | None = None,
    figure_width: float | None = None,
    figure_height: float | None = None,
    density_label: str | None = None,
    profile_labels: Sequence[str] | None = None,
) -> Path:
    chrom_rows = _numt_chromosome_rows(chromosomes)
    if not chrom_rows:
        raise ValueError("karyotype must contain at least one chromosome")
    out = _prepare_output(output)
    _apply_figure_style()

    chrom_names = [row["name"] for row in chrom_rows]
    x_by_chrom = {name: i + 1 for i, name in enumerate(chrom_names)}
    max_y = max_mb if max_mb is not None else max(row["length_mb"] for row in chrom_rows)
    max_y = max(1.0, float(max_y))
    density_by_chrom: dict[str, list[dict[str, Any]]] = {}
    for row in density:
        density_by_chrom.setdefault(_numt_chromosome_name(row), []).append(
            _ideogram_density_row(row)
        )
    labels_by_chrom: dict[str, list[dict[str, Any]]] = {}
    for row in labels:
        labels_by_chrom.setdefault(_numt_chromosome_name(row), []).append(_ideogram_label_row(row))
    profile_by_chrom: dict[str, list[dict[str, Any]]] = {}
    for row in profile:
        profile_by_chrom.setdefault(_numt_chromosome_name(row), []).append(
            _ideogram_profile_row(row)
        )

    chrom_width = _ideogram_chromosome_width(chromosome_width)
    inner_pad = chrom_width * 0.08
    profile_gap = 0.08
    profile_width = 0.28
    # horizontal extent of one chromosome slot to the right of its centre
    reach = chrom_width / 2
    if labels:
        reach = max(reach, chrom_width / 2 + _IDEOGRAM_MARKER_TRACK_GAP + 0.05)
    if profile:
        reach = (
            chrom_width / 2 + profile_gap
            + (_IDEOGRAM_MARKER_TRACK_GAP + 0.12 if labels else 0.0) + profile_width
        )
    has_key = bool(density) or bool(labels) or bool(profile and profile_labels)
    x_min = 1 - chrom_width / 2 - 0.3
    key_x = len(chrom_rows) + reach + 0.35
    x_max = key_x + (_LGA_IDEOGRAM_LEGEND_UNITS if has_key else -0.2)
    units = x_max - x_min
    fig_w = 0.45 * units + 0.5
    fig_h = fig_w * max(0.42, min(0.8, 3.2 / fig_w))
    fig, ax = new_axes(_figure_size(fig_w, fig_h, figure_width, figure_height))

    for row in chrom_rows:
        chrom = row["name"]
        x = x_by_chrom[chrom]
        length = min(row["length_mb"], max_y)
        has_density = bool(density_by_chrom.get(chrom))
        ax.add_patch(
            FancyBboxPatch(
                (x - chrom_width / 2, 0.0),
                chrom_width,
                length,
                boxstyle=f"round,pad=0,rounding_size={chrom_width / 2}",
                facecolor=_ideogram_density_color(0.0) if has_density else "white",
                edgecolor="none",
                zorder=1,
            )
        )
        for density_row in density_by_chrom.get(chrom, ()):
            start = max(0.0, min(length, _float(density_row.get("start_mb", 0.0))))
            end = max(0.0, min(length, _float(density_row.get("end_mb", start))))
            if end < start:
                start, end = end, start
            value = max(0.0, min(1.0, _float(density_row.get("value", 0.0))))
            if end > start:
                ax.add_patch(
                    Rectangle(
                        (x - chrom_width / 2 + inner_pad, start),
                        chrom_width - 2 * inner_pad,
                        end - start,
                        facecolor=_ideogram_density_color(value),
                        edgecolor="none",
                        zorder=2,
                    )
                )
        ax.add_patch(
            FancyBboxPatch(
                (x - chrom_width / 2, 0.0),
                chrom_width,
                length,
                boxstyle=f"round,pad=0,rounding_size={chrom_width / 2}",
                facecolor="none",
                edgecolor=_LGA_IDEOGRAM_OUTLINE,
                linewidth=0.6,
                zorder=4,
            )
        )
        _draw_ideogram_centromere(ax, x, chrom_width, row, max_y)
        _draw_ideogram_marker_labels(
            ax,
            _ideogram_marker_track_x(x, chrom_width),
            labels_by_chrom.get(chrom, ()),
            length,
        )
        _draw_ideogram_profile(
            ax,
            x
            + chrom_width / 2
            + profile_gap
            + (_IDEOGRAM_MARKER_TRACK_GAP + 0.12 if labels else 0.0),
            profile_width,
            profile_by_chrom.get(chrom, ()),
            length,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, max_y * 1.03)
    ax.set_xticks(list(x_by_chrom.values()), chrom_names)
    if show_lengths:
        ax.set_yticks(_ideogram_length_ticks(max_y))
        ax.set_ylabel("Position (Mb)")
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_bounds(0, max_y)
        ax.spines["left"].set_color(_NEUTRAL)
        ax.spines["left"].set_linewidth(0.6)
        ax.tick_params(
            axis="y", direction="out", length=2.5, pad=2, labelsize=FONT_PT["tick"],
            colors=_NEUTRAL,
        )
    else:
        ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=3, labelsize=FONT_PT["tick"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    if not show_lengths:
        ax.spines["left"].set_visible(False)
    if title:
        _panel_title(ax, title)
    if has_key:
        _lga_draw_ideogram_key(
            ax,
            key_x,
            max_y,
            density=bool(density),
            density_label=density_label or "Density",
            labels=labels,
            label_names=label_names,
            profile_labels=list(profile_labels or ()) if profile else [],
        )
    return _save(fig, out)


def _lga_draw_ideogram_key(
    ax: Any,
    x: float,
    max_y: float,
    *,
    density: bool,
    density_label: str,
    labels: Sequence[Mapping[str, Any]],
    label_names: Mapping[str, str] | None,
    profile_labels: Sequence[str],
) -> None:
    """Colour key and marker legend in the column right of the last chromosome."""
    top = max_y
    if density:
        bar_w, bar_h = 1.0, max_y * 0.03
        ax.text(x, top, density_label, ha="left", va="top", fontsize=FONT_PT["label"])
        bar_y = top - max_y * 0.075
        steps = 32
        for i in range(steps):
            ax.add_patch(
                Rectangle(
                    (x + bar_w * i / steps, bar_y),
                    bar_w / steps,
                    bar_h,
                    facecolor=_ideogram_density_color(i / (steps - 1)),
                    edgecolor="none",
                    clip_on=False,
                )
            )
        ax.add_patch(
            Rectangle((x, bar_y), bar_w, bar_h, facecolor="none",
                      edgecolor=_LGA_IDEOGRAM_OUTLINE, linewidth=0.4, clip_on=False)
        )
        for text, x_text, align in (("Low", x, "left"), ("High", x + bar_w, "right")):
            ax.text(x_text, bar_y - max_y * 0.01, text, ha=align, va="top",
                    fontsize=FONT_PT["label"], color=_NEUTRAL)
        top = bar_y - max_y * 0.09
    handles: list[Any] = []
    if labels:
        resolved = _ideogram_label_names(labels, label_names)
        for label_type in _ideogram_marker_types(labels):
            style = _ideogram_marker_style(label_type)
            handles.append(
                Line2D([0], [0], marker=style["marker"], color="none",
                       markerfacecolor=style["color"], markeredgewidth=0, markersize=4.0,
                       label=resolved.get(label_type, label_type))
            )
    if len(profile_labels) >= 1:
        handles.append(Rectangle((0, 0), 1, 1, color=_LGA_PROFILE_AREA, lw=0,
                                 label=profile_labels[0]))
    if len(profile_labels) >= 2:
        handles.append(Line2D([0], [0], color=_LGA_PROFILE_LINE, lw=0.8, label=profile_labels[1]))
    if handles:
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(x - 0.08, top),
            bbox_transform=ax.transData,
            frameon=False,
            handlelength=1.0,
            handleheight=0.8,
            handletextpad=0.4,
            labelspacing=0.3,
            borderaxespad=0.0,
            borderpad=0.0,
            fontsize=FONT_PT["label"],
        )


def _render_nuclear_transfer_ideogram(
    chromosomes: Sequence[Mapping[str, Any]] | Mapping[str, int | float],
    fragments: Sequence[Mapping[str, Any]],
    gene_hits: Sequence[Mapping[str, Any]],
    output: str | Path,
    **options: Any,
) -> Path:
    from . import transfer_plots as tp

    spec = _nuclear_transfer_ideogram_spec(chromosomes, fragments, gene_hits, **options)
    return tp.render_transfer_figure(spec, output)


def _nuclear_transfer_ideogram_spec(
    chromosomes: Sequence[Mapping[str, Any]] | Mapping[str, int | float],
    fragments: Sequence[Mapping[str, Any]],
    gene_hits: Sequence[Mapping[str, Any]],
    *,
    title: str | None = None,
    panel_label: str | None = None,
    max_mb: float | None = None,
    transfer_label: str = "NUMT",
    figure_width: float | None = None,
    figure_height: float | None = None,
    plastid_fragments: Sequence[Mapping[str, Any]] | None = None,
    color_by: str = "organelle",
    show_density: bool = True,
    density_window: int = 500_000,
    label_top: int = 0,
    nuclear_annotated: bool | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    from . import transfer_plots as tp

    legacy_rows = _numt_chromosome_rows(chromosomes)
    if not legacy_rows:
        raise ValueError("chromosomes must contain at least one chromosome")
    # prepared metrics already hold the drawing order; plain inputs are sorted
    prepared = not isinstance(chromosomes, Mapping) and all(
        isinstance(row, Mapping) and "length_bp" in row for row in chromosomes
    )
    chrom_rows = tp.normalize_chromosomes(
        (_transfer_ideogram_chromosome(row) for row in legacy_rows),
        order=[row["name"] for row in legacy_rows] if prepared else None,
    )
    label = None if plastid_fragments else transfer_label
    sets = tp.build_transfer_sets(fragments, primary_label=label, plastid=plastid_fragments)
    hits = []
    for hit in gene_hits or ():
        row = _numt_gene_hit_row(hit)
        hits.append(
            {
                "seqid": row["chromosome"],
                "position": row["position_mb"] * 1_000_000,
                "class": "within_gene" if row["class"] == "new_gene" else "intergenic",
            }
        )
    annotated = bool(hits) or tp.resolve_gene_annotation(sets, nuclear_annotated)
    spec = {
        "orientation": "vertical",
        "chromosomes": chrom_rows,
        "sets": sets,
        "gene_hits": hits,
        "gene_classes": annotated,
        "color_by": color_by,
        "show_density": show_density,
        "density_window": density_window,
        "label_top": label_top,
        "max_bp": float(max_mb) * 1_000_000 if max_mb else None,
        "title": title,
        "panel_label": panel_label,
        "note": note or tp.placement_summary(sets, chrom_rows)["note"],
        "figure_width": figure_width,
        "figure_height": figure_height,
    }
    return spec


def _transfer_ideogram_chromosome(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": row["name"],
        "length_bp": round(_float(row.get("length_mb")) * 1_000_000),
    }
    if row.get("centromere_start_mb") is not None and row.get("centromere_end_mb") is not None:
        out["centromere"] = [
            round(_float(row["centromere_start_mb"]) * 1_000_000),
            round(_float(row["centromere_end_mb"]) * 1_000_000),
        ]
    return out


def _transfer_ideogram_chromosome_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    centromere = row.get("centromere")
    return {
        "name": row["name"],
        "length_mb": row["length_bp"] / 1_000_000,
        "length_bp": row["length_bp"],
        "centromere_start_mb": centromere[0] / 1_000_000 if centromere else None,
        "centromere_end_mb": centromere[1] / 1_000_000 if centromere else None,
    }


def _transfer_ideogram_fragment(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "chromosome": row["seqid"],
        "start_mb": row["start"] / 1_000_000,
        "end_mb": row["end"] / 1_000_000,
        "length_mb": max(0.0, (row["end"] - row["start"]) / 1_000_000),
        **row,
    }


def plot_transfer_tracks(
    candidates: Sequence[Mapping[str, str | int | float | bool | None]],
    nuclear_fasta: str | Path | None = None,
    *,
    chromosome_lengths: Mapping[str, int] | None = None,
    color_by: str = "evidence_level",
    title: str | None = None,
    plastid_candidates: Sequence[Mapping[str, Any]] | None = None,
    transfer_label: str | None = None,
    centromeres: Mapping[str, Sequence[float]] | None = None,
    chromosome_order: Sequence[str] | None = None,
    show_density: bool = True,
    density_window: int = 500_000,
    label_top: int = 0,
    min_chromosome_fraction: float = 0.02,
    panel_label: str | None = None,
    width: float | None = None,
    height: float | None = None,
) -> OrganellePlot:
    """Render NUMT/NUPT insertions on linear chromosome tracks (compute-only).

    Each nuclear chromosome is a horizontal bar in natural order (1, 2, ...,
    10; or ``chromosome_order``, which also selects them) on a Mb axis. Every
    insertion is a thin semi-transparent tick: NUMT above the bar, NUPT below;
    tick length/width encode the fragment length. The bar is filled with the
    insertion density per ``density_window`` bp so dense regions stay
    readable. Render with ``ov.write(plot, output)``.

    Parameters
    ----------
    candidates : sequence of mappings
        ``detect_transfers*`` candidates (``nuclear_seqid``, ``nuclear_start``,
        ``nuclear_end``, ``length``, ``identity``, ``evidence_level``,
        ``depth_supported``, ``longread_linked``, ``organelle_gene``); a
        transfer Result is also accepted.
    nuclear_fasta : path, optional
        Read chromosome lengths (from ``.fai`` when present) when
        ``chromosome_lengths`` is not given; sequences shorter than
        ``min_chromosome_fraction`` of the longest count as unplaced.
    chromosome_lengths : {seqid: length}, optional
        Explicit chromosomes to draw; candidates elsewhere are counted in
        ``metrics["unplaced_candidates"]`` and not drawn.
    color_by : {"evidence_level", "organelle", "identity", "organelle_gene"}
        Tick colour. Evidence is named from the validation layers (alignment
        only, read depth, long-read linked, both); input with only the BLAST
        layer is shown as candidates (alignment only).
    plastid_candidates : sequence or transfer Result, optional
        NUPT candidates to draw below the bars; ``candidates`` is then NUMT.
    transfer_label : {"NUMT", "NUPT"}, optional
        Label of a single ``candidates`` set (guessed from the input otherwise).
    centromeres : {seqid: (start_bp, end_bp)}, optional
        Centromere spans drawn as a constriction.
    label_top : int
        Annotate the N longest insertions with length and organelle gene.
    """
    from . import transfer_plots as tp

    sets = tp.build_transfer_sets(
        candidates, primary_label=transfer_label, plastid=plastid_candidates
    )
    if not any(item["rows"] for item in sets):
        raise ValueError("candidates must contain at least one transfer candidate")
    chromosomes = tp.normalize_chromosomes(
        tp.resolve_track_chromosomes(
            sets,
            nuclear_fasta=nuclear_fasta,
            chromosome_lengths=chromosome_lengths,
            min_chromosome_fraction=min_chromosome_fraction,
        ),
        order=chromosome_order,
        centromeres=centromeres,
    )
    placement = tp.placement_summary(sets, chromosomes)
    spec = {
        "orientation": "horizontal",
        "chromosomes": chromosomes,
        "sets": sets,
        "gene_classes": False,
        "color_by": color_by,
        "show_density": show_density,
        "density_window": density_window,
        "label_top": label_top,
        "title": title,
        "panel_label": panel_label,
        "note": placement["note"],
        "figure_width": width,
        "figure_height": height,
    }

    def _render(path: str | Path) -> Path:
        return tp.render_transfer_figure(spec, path)

    plot = plot_result(
        "plot_transfer_tracks",
        _render,
        metrics={
            "candidates": sum(len(item["rows"]) for item in sets),
            "color_by": color_by,
            "transfers": [item["transfer"] for item in sets],
            "chromosomes": [row["name"] for row in chromosomes],
            "drawn_candidates": placement["drawn_candidates"],
            "unplaced_candidates": placement["unplaced_candidates"],
            "unplaced_seqids": placement["unplaced_seqids"],
            "by_transfer": placement["by_transfer"],
            "density_window": int(density_window),
            "caption": tp.transfer_caption(spec),
        },
        summary=f"Transfer track plot prepared ({placement['note']}).",
    )
    plot._panel = tp.transfer_panel(spec)
    return plot


def _render_transfer_tracks(
    candidates: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    nuclear_fasta: str | Path | None = None,
    chromosome_lengths: Mapping[str, int] | None = None,
    color_by: str = "evidence_level",
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_transfer_tracks`."""
    plot = plot_transfer_tracks(
        candidates,
        nuclear_fasta,
        chromosome_lengths=chromosome_lengths,
        color_by=color_by,
        title=title,
    )
    renderer = plot.renderer
    assert renderer is not None
    return renderer(output)


def _format_bp(length: int) -> str:
    """Human-readable bp length, e.g. 1500 -> '1.5 kb', 5_200_000 -> '5.2 Mb'."""
    if length >= 1_000_000:
        return f"{length / 1_000_000:.1f} Mb"
    if length >= 1_000:
        return f"{length / 1_000:.1f} kb"
    return f"{length} bp"


def plot_splicing_schematic(
    genes: Sequence[Mapping[str, str | list[Mapping[str, int | float]]]],
    *,
    title: str | None = None,
) -> OrganellePlot:
    """Render trans-spliced exon fragments and their gene-level connections (compute-only)."""
    if not genes:
        raise ValueError("genes must contain at least one gene")

    def _render(path: str | Path) -> Path:
        return _render_splicing_schematic(genes, path, title=title)

    plot = plot_result(
        "plot_splicing_schematic",
        _render,
        metrics={"genes": len(genes)},
        summary="Splicing schematic prepared.",
    )
    plot._panel = renderer_panel(_render)
    return plot


_LGA_SPLICE_EXON = _PALETTE[0]
_LGA_SPLICE_CIS = "#8B949E"
_LGA_SPLICE_TRANS = _PALETTE[5]


def _render_splicing_schematic(
    genes: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_splicing_schematic`.

    One lane per gene and contig, exons at their contig coordinates. Successive
    exons (input order) are joined by an arc within a lane or, when they lie on
    different contigs, by an accent-coloured connector between lanes.
    """
    from matplotlib.path import Path as MplPath  # type: ignore

    from .style import text_width_pt

    out = _prepare_output(output)
    _apply_figure_style()

    exons_by_gene: list[tuple[str, list[Mapping[str, Any]]]] = []
    for gene in genes:
        exons = list(gene.get("exons", []))
        if not exons:
            raise ValueError("each gene must contain at least one exon")
        exons_by_gene.append((str(gene.get("gene", "gene")), exons))
    contigs = {str(exon.get("contig", "contig")) for _, exons in exons_by_gene for exon in exons}
    single_contig = len(contigs) == 1
    lanes: list[tuple[str, str]] = []
    for gene_name, exons in exons_by_gene:
        for exon in exons:
            lane = (gene_name, str(exon.get("contig", "contig")))
            if lane not in lanes:
                lanes.append(lane)
    spacing = 1.3
    lane_y = {lane: (len(lanes) - 1 - i) * spacing for i, lane in enumerate(lanes)}
    max_end = max(
        1.0,
        max(max(_float(exon.get("start", 0.0)), _float(exon.get("end", 0.0)))
            for _, exons in exons_by_gene for exon in exons),
    )

    fig, ax = new_axes(journal_figsize(7.2, 0.4 * len(lanes) + 0.6))
    _panel_title(ax, title)
    axes_width_pt = max(1.0, 0.8 * ax.get_position().width * fig.get_figwidth() * 72)
    bp_per_pt = max_end / axes_width_pt
    min_width = max_end * 0.004
    label_rows: dict[tuple[str, str], list[list[tuple[float, float]]]] = {}
    has_trans = False

    for lane, y in lane_y.items():
        gene_name, contig = lane
        lane_exons = [exon for name, exons in exons_by_gene if name == gene_name
                      for exon in exons if str(exon.get("contig", "contig")) == contig]
        low = min(min(_float(e.get("start", 0.0)), _float(e.get("end", 0.0))) for e in lane_exons)
        high = max(max(_float(e.get("start", 0.0)), _float(e.get("end", 0.0))) for e in lane_exons)
        ax.plot([low, high], [y, y], color="#C9CED4", lw=0.6, zorder=1)
        ax.text(-0.015 * max_end, y + (0.0 if single_contig else 0.12), gene_name,
                ha="right", va="center" if single_contig else "bottom",
                fontstyle="italic", fontsize=FONT_PT["body"])
        if not single_contig:
            ax.text(-0.015 * max_end, y - 0.04, contig, ha="right", va="top",
                    fontsize=FONT_PT["label"], color=_NEUTRAL)

    for gene_name, exons in exons_by_gene:
        anchors: list[tuple[float, float, float, str]] = []  # (left, right, y, contig)
        for exon in exons:
            contig = str(exon.get("contig", "contig"))
            y = lane_y[(gene_name, contig)]
            start, end = sorted((_float(exon.get("start", 0.0)), _float(exon.get("end", 0.0))))
            if end - start < min_width:
                mid = (start + end) / 2
                start, end = mid - min_width / 2, mid + min_width / 2
            ax.add_patch(Rectangle((start, y - 0.17), end - start, 0.34,
                                   facecolor=_LGA_SPLICE_EXON, edgecolor="none", zorder=3))
            label = str(exon.get("exon", "exon"))
            if label.lower().startswith("exon "):
                label = label[5:]
            width = text_width_pt(label, FONT_PT["label"]) * bp_per_pt
            center = (start + end) / 2
            rows = label_rows.setdefault((gene_name, contig), [[], []])
            for tier, row in enumerate(rows):
                if all(center + width / 2 + 2 * bp_per_pt <= a
                       or center - width / 2 - 2 * bp_per_pt >= b for a, b in row):
                    row.append((center - width / 2, center + width / 2))
                    ax.text(center, y - 0.24 - 0.28 * tier, label, ha="center", va="top",
                            fontsize=FONT_PT["label"], color="#1F2933")
                    break
            anchors.append((start, end, y, contig))
        for (l1, r1, y1, c1), (l2, r2, y2, c2) in itertools.pairwise(anchors):
            x1, x2 = (l1 + r1) / 2, (l2 + r2) / 2
            if c1 == c2:
                # arc over the lane; taller for more distant exons
                h = 0.18 + 0.42 * min(1.0, abs(x2 - x1) / max_end) ** 0.5
                verts = [(x1, y1 + 0.17), (x1, y1 + 0.17 + h),
                         (x2, y2 + 0.17 + h), (x2, y2 + 0.17)]
                color, style = _LGA_SPLICE_CIS, "-"
            else:
                has_trans = True
                sign = 1 if y2 > y1 else -1
                ya, yb = y1 + sign * 0.17, y2 - sign * 0.17
                ym = (ya + yb) / 2
                verts = [(x1, ya), (x1, ym), (x2, ym), (x2, yb)]
                color, style = _LGA_SPLICE_TRANS, (0, (3, 1.5))
            ax.add_patch(
                FancyArrowPatch(
                    path=MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4,
                                         MplPath.CURVE4, MplPath.CURVE4]),
                    arrowstyle="-",
                    lw=0.7,
                    linestyle=style,
                    color=color,
                    zorder=2,
                )
            )

    # gene names sit inside the axes, so a composed panel keeps room for them
    widest = max(
        text_width_pt(gene_name, FONT_PT["body"], italic=True) for gene_name, _ in lanes
    )
    if not single_contig:
        widest = max(widest, *(text_width_pt(c, FONT_PT["label"]) for _, c in lanes))
    ax.set_xlim(-(widest + 8.0) * bp_per_pt * 0.8, max_end * 1.02)
    ax.spines["bottom"].set_bounds(0, max_end)
    ax.set_ylim(-0.95, (len(lanes) - 1) * spacing + 0.85)
    axis_label = f"Position on {next(iter(contigs))}" if single_contig else "Position on contig"
    _lga_genome_axis(ax, max_end, axis_label)
    ax.set_yticks([])
    _despine(ax)
    ax.spines["left"].set_visible(False)
    if has_trans:
        ax.legend(
            handles=[
                Line2D([0], [0], color=_LGA_SPLICE_CIS, lw=0.8, label="Same contig"),
                Line2D([0], [0], color=_LGA_SPLICE_TRANS, lw=0.8, ls=(0, (3, 1.5)),
                       label="Across contigs"),
            ],
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=2,
            frameon=False,
            handlelength=1.8,
            borderaxespad=0.2,
            fontsize=FONT_PT["label"],
        )
    return _save(fig, out)


def plot_localization_bars(
    predictions: Sequence[Mapping[str, str | Mapping[str, int | float]]],
    *,
    title: str | None = None,
) -> OrganellePlot:
    """Render per-protein subcellular-localization probabilities (compute-only)."""
    if not predictions:
        raise ValueError("predictions must contain at least one protein")

    def _render(path: str | Path) -> Path:
        return _render_localization_bars(predictions, path, title=title)

    plot = plot_result(
        "plot_localization_bars",
        _render,
        metrics={"proteins": len(predictions)},
        summary="Localization bar plot prepared.",
    )
    plot._panel = renderer_panel(_render)
    return plot


#: Fixed compartment colours (matched by name) so one compartment keeps its colour
#: across figures; organelles take the strong hues, the rest muted ones.
_LOC_COMPARTMENT_COLORS = (
    (("mito",), "#0072B2"),
    (("plastid", "chloro", "amyloplast"), "#009E73"),
    (("nucle",), "#E69F00"),
    (("cyto",), "#B5BEC8"),
    (("perox",), "#CC79A7"),
    (("secret", "extracell", "plasma", "membrane"), "#56B4E9"),
    (("vacuol",), "#8E6C8A"),
    (("reticulum", "er", "golgi"), "#D55E00"),
)
_LOC_FALLBACK = ("#6B7280", "#A3ACB6", "#374151", "#D1D5DB")


def _loc_compartment_order(totals: Mapping[str, float]) -> list[str]:
    """Organelle compartments first (mitochondrion, plastid), then by mean probability."""

    def rank(name: str) -> tuple[int, float, str]:
        lowered = name.lower()
        organelle = 0 if "mito" in lowered else 1 if ("plastid" in lowered or "chloro" in lowered) else 2
        return organelle, -totals[name], name

    return sorted(totals, key=rank)


def _loc_palette(compartments: Sequence[str]) -> dict[str, str]:
    palette: dict[str, str] = {}
    spare = iter(itertools.cycle(_LOC_FALLBACK))
    for name in compartments:
        lowered = name.lower()
        words = lowered.replace("-", " ").replace("_", " ").split()
        color = next(
            (
                hue
                for keys, hue in _LOC_COMPARTMENT_COLORS
                if any((key in words) if key == "er" else (key in lowered) for key in keys)
                and hue not in palette.values()
            ),
            None,
        )
        palette[name] = color or next(spare)
    return palette


def _render_localization_bars(
    predictions: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_localization_bars`."""
    from .style import text_width_pt

    out = _prepare_output(output)
    _apply_figure_style()

    probabilities = [
        pred.get("probabilities", {}) if isinstance(pred.get("probabilities", {}), Mapping) else {}
        for pred in predictions
    ]
    totals: dict[str, float] = {}
    for probs in probabilities:
        for comp, value in probs.items():
            totals[str(comp)] = totals.get(str(comp), 0.0) + _float(value)
    if not totals:
        raise ValueError("predictions must include probabilities")
    compartments = _loc_compartment_order(totals)
    proteins = [
        str(pred.get("protein", pred.get("id", f"protein {i + 1}")))
        for i, pred in enumerate(predictions)
    ]
    palette = _loc_palette(compartments)

    width_in = 120 / 25.4
    legend_entries = [comp for comp in compartments if totals[comp] > 0]
    legend_pt = [text_width_pt(comp, FONT_PT["label"]) + 16 for comp in legend_entries]
    ncol = max(1, len(legend_entries))
    while ncol > 1 and sum(sorted(legend_pt, reverse=True)[:ncol]) > width_in * 72 * 0.7:
        ncol -= 1
    legend_rows = -(-len(legend_entries) // ncol)
    height_in = 0.155 * len(proteins) + 0.42 + 0.12 * legend_rows
    fig, ax = new_axes(journal_figsize(width_in, height_in))

    y = list(range(len(predictions)))
    left = [0.0 for _ in predictions]
    segments: list[list[tuple[float, float, str]]] = [[] for _ in predictions]
    for comp in compartments:
        values = [_float(probs.get(comp, 0.0)) for probs in probabilities]
        if not any(values):
            continue
        ax.barh(
            y, values, left=left, label=comp, color=palette[comp], height=0.72,
            edgecolor="white", linewidth=0.4,
        )
        for i, value in enumerate(values):
            segments[i].append((left[i], value, palette[comp]))
        left = [a + b for a, b in zip(left, values, strict=False)]
    # the dominant probability, written into its own segment when it fits
    for yi, protein_segments in enumerate(segments):
        if not protein_segments:
            continue
        start, value, color = max(protein_segments, key=lambda seg: seg[1])
        if value >= 0.12:
            ax.text(
                start + value / 2,
                yi,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=FONT_PT["label"],
                color=_text_color_for_bar(color),
            )
    ax.set_yticks(y, proteins)
    ax.invert_yaxis()
    ax.set_ylim(len(proteins) - 0.5, -0.5)
    ax.set_xlim(0, max(1.0, max(left)))
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1"])
    ax.set_xlabel("Predicted probability")
    _panel_title(ax, title)
    _despine(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=ncol,
        fontsize=FONT_PT["label"],
        handlelength=1.0,
        handleheight=0.8,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0.2,
        frameon=False,
    )
    return _save(fig, out)


def plot_rscu_usage(
    rscu_rows: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    *,
    title: str | None = None,
    rna_labels: bool = True,
    include_stop: bool = False,
    layout: str = "stacked",
) -> OrganellePlot:
    """Render RSCU codon-usage bias as amino-acid grouped codon stacks (compute-only).

    ``layout="stacked"`` (default) stacks each amino acid's codons, preferred
    codons (RSCU > 1) at the bottom in the accent colour; ``layout="grouped"``
    draws one bar per codon, grouped by amino acid, against an RSCU = 1 line.
    """
    if layout not in {"stacked", "grouped"}:
        raise ValueError(f"unknown layout={layout!r}; expected 'stacked' | 'grouped'")
    rows = _rscu_rows(rscu_rows)
    if not include_stop:
        rows = [row for row in rows if row["aa"] != "*"]
    if not rows:
        raise ValueError("rscu_rows must contain at least one codon row")

    def _render(path: str | Path) -> Path:
        return _render_rscu_usage(rows, path, title=title, rna_labels=rna_labels, layout=layout)

    plot = plot_result(
        "plot_rscu_usage",
        _render,
        metrics={"codon_rows": len(rows)},
        summary="RSCU codon-usage plot prepared.",
    )
    plot._panel = renderer_panel(_render)
    return plot


_RSCU_PREFERRED = _PALETTE[0]  # RSCU > 1
_RSCU_OTHER = "#BCC8D4"  # RSCU <= 1, light member of the same blue-grey family
_RSCU_BAND = "#F2F4F7"  # alternate amino-acid background (grouped layout)
_RSCU_CODON_PT = FONT_PT["label"]


def _rscu_legend(ax: Any) -> None:
    from matplotlib.patches import Patch  # type: ignore

    ax.legend(
        handles=[
            Patch(color=_RSCU_PREFERRED, label="RSCU > 1 (preferred)"),
            Patch(color=_RSCU_OTHER, label="RSCU \u2264 1"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        fontsize=FONT_PT["label"],
        handlelength=1.0,
        handleheight=0.8,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.2,
        frameon=False,
    )


def _rscu_draw_grouped(
    ax: Any, fig: Any, by_aa: Mapping[str, list[dict[str, Any]]], ordered_aas: Sequence[str],
    rna_labels: bool,
) -> float:
    """One bar per codon, amino acids on alternating bands; returns the tallest bar."""
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory  # type: ignore

    from .style import text_width_pt

    gap = 0.8
    cursor = 0.0
    ticks: list[float] = []
    heights: list[float] = []
    tick_labels: list[str] = []
    centres: list[tuple[float, str]] = []
    tallest = 0.0
    for index, aa in enumerate(ordered_aas):
        family = sorted(by_aa[aa], key=lambda row: row["codon"])
        first = cursor
        for row in family:
            value = max(0.0, _float(row["rscu"]))
            ax.bar(
                cursor, value, width=0.78, linewidth=0, zorder=2,
                color=_RSCU_PREFERRED if value > 1.0 else _RSCU_OTHER,
            )
            ticks.append(cursor)
            heights.append(value)
            tick_labels.append(_display_codon(str(row["codon"]), rna_labels))
            tallest = max(tallest, value)
            cursor += 1.0
        last = cursor - 1.0
        if index % 2 == 1:
            ax.axvspan(first - 0.5, last + 0.5, color=_RSCU_BAND, lw=0, zorder=0)
        centres.append(((first + last) / 2, _AA_NAMES.get(aa, aa)))
        cursor += gap
    ax.set_xticks(ticks, tick_labels, rotation=90, fontsize=_RSCU_CODON_PT)
    ax.tick_params(axis="x", length=0, pad=1.5)
    codon_pt = max(text_width_pt(label, _RSCU_CODON_PT) for label in tick_labels)
    drop = ScaledTranslation(0, -(1.5 + codon_pt * 1.1 + 4.5) / 72, fig.dpi_scale_trans)
    transform = blended_transform_factory(ax.transData, ax.transAxes) + drop
    for centre, name in centres:
        ax.text(centre, 0, name, transform=transform, ha="center", va="top",
                fontsize=FONT_PT["tick"])
    ax.set_xlim(-0.9, cursor - gap - 0.1)
    ax.axhline(1.0, color=_NEUTRAL, lw=0.6, ls=(0, (3, 2)), zorder=3)
    # inside above the line when the last codon bar leaves room, else right of the axes
    inside = max(heights[-5:]) < 0.8
    ax.annotate(
        "RSCU = 1" if inside else " RSCU = 1",
        (1.0, 1.0),
        xycoords=blended_transform_factory(ax.transAxes, ax.transData),
        xytext=(0, 1.0 if inside else 0),
        textcoords="offset points",
        ha="right" if inside else "left",
        va="bottom" if inside else "center",
        fontsize=FONT_PT["tick"],
        color=_NEUTRAL,
        annotation_clip=False,
    )
    return tallest


def _render_rscu_usage(
    rows: list[dict[str, Any]],
    output: str | Path,
    *,
    title: str | None = None,
    rna_labels: bool = True,
    layout: str = "stacked",
) -> Path:
    """Private path-taking renderer backing :func:`plot_rscu_usage`."""
    from matplotlib.ticker import MaxNLocator  # type: ignore

    from .style import text_width_pt

    out = _prepare_output(output)
    _apply_figure_style()

    by_aa: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_aa.setdefault(str(row["aa"]), []).append(row)
    ordered_aas = [aa for aa in _AA_ORDER if aa in by_aa]
    ordered_aas.extend(sorted(aa for aa in by_aa if aa not in ordered_aas))
    if not ordered_aas:
        raise ValueError("rscu_rows must contain at least one codon row")

    if layout == "grouped":
        fig, ax = new_axes(journal_figsize(183 / 25.4, 64 / 25.4))
        top = _rscu_draw_grouped(ax, fig, by_aa, ordered_aas, rna_labels)
        ax.set_ylim(0, max(1.2, top * 1.08))
    else:
        width_mm = 183.0 if len(ordered_aas) > 10 else 120.0 if len(ordered_aas) > 5 else 89.0
        fig, ax = new_axes(journal_figsize(width_mm / 25.4, 62 / 25.4))
        stacks = {
            aa: sum(max(0.0, _float(row["rscu"])) for row in by_aa[aa]) for aa in ordered_aas
        }
        y_top = max(1.4, max(stacks.values()) * 1.04)
        # printed size of one RSCU unit and of one bar, to decide which codons fit
        box = ax.get_position()
        unit_pt = box.height * fig.get_figheight() * 72 * 0.9 / y_top
        bar_pt = box.width * fig.get_figwidth() * 72 / (len(ordered_aas) + 0.2) * 0.74 + 1.0
        for xi, aa in enumerate(ordered_aas):
            # preferred codons at the bottom, so the bias reads from the baseline up
            family = sorted(by_aa[aa], key=lambda row: (-_float(row["rscu"]), row["codon"]))
            bottom = 0.0
            for row in family:
                value = max(0.0, _float(row["rscu"]))
                color = _RSCU_PREFERRED if value > 1.0 else _RSCU_OTHER
                ax.bar(
                    xi, value, bottom=bottom, width=0.74, color=color,
                    edgecolor="white", linewidth=0.5, zorder=2,
                )
                label = _display_codon(str(row["codon"]), rna_labels)
                if (
                    value * unit_pt >= _RSCU_CODON_PT * 1.05
                    and text_width_pt(label, _RSCU_CODON_PT) <= bar_pt
                ):
                    ax.text(
                        xi, bottom + value / 2, label, ha="center", va="center",
                        fontsize=_RSCU_CODON_PT, color=_text_color_for_bar(color), zorder=3,
                    )
                bottom += value
        ax.set_xticks(range(len(ordered_aas)), [_AA_NAMES.get(aa, aa) for aa in ordered_aas])
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_xlim(-0.6, len(ordered_aas) - 0.4)
        ax.set_ylim(0, y_top)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_ylabel("RSCU")
    _panel_title(ax, title)
    _despine(ax)
    _rscu_legend(ax)
    return _save(fig, out)


def plot_rna_editing_summary(
    editing_sites: Sequence[Mapping[str, str | int | float | bool | None]],
    *,
    title: str | None = None,
    top_n: int = 25,
) -> OrganellePlot:
    """Render per-gene RNA-editing site counts, stacked by editing class (compute-only)."""
    if not editing_sites:
        raise ValueError("editing_sites must contain at least one site")

    def _render(path: str | Path) -> Path:
        return _render_rna_editing_summary(editing_sites, path, title=title, top_n=top_n)

    plot = plot_result(
        "plot_rna_editing_summary",
        _render,
        metrics={"editing_sites": len(editing_sites)},
        summary="RNA-editing summary prepared.",
    )
    plot._panel = renderer_panel(_render)
    return plot


_RNAED_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#6B7280")
#: Locus names that are not gene symbols and therefore stay upright.
_RNAED_UPRIGHT = frozenset({"intergenic", "unknown", "other"})


def _render_rna_editing_summary(
    editing_sites: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    title: str | None = None,
    top_n: int = 25,
) -> Path:
    """Private path-taking renderer backing :func:`plot_rna_editing_summary`."""
    out = _prepare_output(output)
    _apply_figure_style()

    from matplotlib.ticker import MaxNLocator  # type: ignore

    rows = [_editing_site_row(site, i) for i, site in enumerate(editing_sites)]
    gene_counts = _count_by(row["gene"] for row in rows)
    type_counts = _count_by(str(row["edit_type"]) for row in rows)
    # the most frequent class first, in the main colour; others in the next hues
    edit_types = sorted(type_counts, key=lambda key: (-type_counts[key], key))
    palette = {
        edit_type: _RNAED_COLORS[i % len(_RNAED_COLORS)] for i, edit_type in enumerate(edit_types)
    }
    top_genes = sorted(gene_counts, key=lambda key: (-gene_counts[key], key))[: max(1, top_n)]

    multi = len(edit_types) > 1
    height_in = 0.125 * len(top_genes) + 0.5 + (0.12 if multi else 0.0)
    fig, ax = new_axes(journal_figsize(89 / 25.4, height_in))
    y = list(range(len(top_genes)))
    left = [0 for _ in top_genes]
    for edit_type in edit_types:
        values = [
            sum(1 for row in rows if row["gene"] == gene and row["edit_type"] == edit_type)
            for gene in top_genes
        ]
        ax.barh(
            y, values, left=left, color=palette[edit_type], height=0.72, label=edit_type,
            edgecolor="white" if multi else "none", linewidth=0.4 if multi else 0,
        )
        left = [a + b for a, b in zip(left, values, strict=False)]
    for yi, total in zip(y, left, strict=False):
        ax.annotate(
            str(total), (total, yi), xytext=(2, 0), textcoords="offset points",
            va="center", ha="left", fontsize=FONT_PT["label"], color="#1F2933",
        )
    ax.set_yticks(y, top_genes)
    for tick in ax.get_yticklabels():
        if tick.get_text() not in _RNAED_UPRIGHT:
            tick.set_fontstyle("italic")
    ax.invert_yaxis()
    ax.set_ylim(len(top_genes) - 0.5, -0.6)
    max_total = max(left)
    x_max = max_total * 1.1 + 0.5
    ax.set_xlim(0, x_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    unit = edit_types[0] if not multi else ""
    total_sites = len(rows)
    note = f"{total_sites:,} sites in {len(gene_counts):,} genes"
    truncated = len(gene_counts) > len(top_genes)
    # the summary sits in the empty lower-right corner when the shortest bars leave it free
    corner_free = len(top_genes) >= 4 and max(left[-2:]) <= 0.45 * x_max
    if corner_free:
        ax.text(
            0.98, 0.02, note + (f"\ntop {len(top_genes)} shown" if truncated else ""),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FONT_PT["label"], color=_NEUTRAL, linespacing=1.2,
        )
        ax.set_xlabel(f"RNA editing sites ({unit})" if unit else "RNA editing sites")
    else:
        details = "; ".join(
            part for part in (unit, note, f"top {len(top_genes)} shown" if truncated else "") if part
        )
        ax.set_xlabel(f"RNA editing sites ({details})")
    _panel_title(ax, title)
    _despine(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    if multi:
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=len(edit_types),
            fontsize=FONT_PT["label"],
            handlelength=1.0,
            handleheight=0.8,
            handletextpad=0.4,
            columnspacing=1.0,
            borderaxespad=0.2,
            frameon=False,
        )
    return _save(fig, out)


_ERC_INK = "#1F2933"
_ERC_GREY_MID = "#8A94A3"
_ERC_GREY_BG = "#D5D9DF"
_ERC_HIT_LIGHT = "#F2B48C"
_ERC_DASH = (0, (3, 2))


def _erc_stat_symbol(method: str) -> str:
    return "\u03c1" if method == "spearman" else "r"


def _erc_strength_label(method: str) -> str:
    name = "Spearman" if method == "spearman" else "Pearson"
    return f"ERC strength, |{name} {_erc_stat_symbol(method)}|"


def _erc_axes_size_pt(ax: Any, *, bare: bool = False) -> tuple[float, float]:
    """Width and height of ``ax`` in points.

    A stand-alone figure still has matplotlib's default subplot box here, which
    ``tight_layout`` later enlarges to the canvas less its tick labels; the
    canvas is read instead (less a tick-label allowance unless ``bare``, axes
    drawn without ticks). Inside a composed panel the axes are already final.
    """
    fig_w, fig_h = ax.figure.get_size_inches()
    pos = ax.get_position()
    default_box = abs(pos.x0 - 0.125) < 1e-6 and abs(pos.width - 0.775) < 1e-6
    if default_box:
        if bare:
            return fig_w * 72.0 * 0.96, fig_h * 72.0 * 0.96
        return max(1.0, fig_w * 72.0 - 36.0), max(1.0, fig_h * 72.0 - 34.0)
    return pos.width * fig_w * 72.0, pos.height * fig_h * 72.0


def plot_erc_pair_scatter(
    points: Sequence[Mapping[str, str | int | float | bool | None]],
    *,
    method: str = "pearson",
    gene_a: str | None = None,
    gene_b: str | None = None,
    title: str | None = None,
    x_key: str = "rate_a",
    y_key: str = "rate_b",
) -> OrganellePlot:
    """Render branch-level ERC rates for one candidate gene pair (compute-only)."""
    method = _validate_erc_method(method)
    if len(points) < 2:
        raise ValueError("points must contain at least two branch observations")

    def _render(path: str | Path) -> Path:
        return _render_erc_pair_scatter(
            points,
            path,
            method=method,
            gene_a=gene_a,
            gene_b=gene_b,
            title=title,
            x_key=x_key,
            y_key=y_key,
        )

    plot = plot_result(
        "plot_erc_pair_scatter",
        _render,
        metrics={"method": method, "points": len(points)},
        summary="ERC pair scatter prepared.",
    )
    plot._panel = renderer_panel(_render)
    return plot


def _render_erc_pair_scatter(
    points: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    method: str,
    gene_a: str | None = None,
    gene_b: str | None = None,
    title: str | None = None,
    x_key: str = "rate_a",
    y_key: str = "rate_b",
) -> Path:
    """Private path-taking renderer backing :func:`plot_erc_pair_scatter`.

    One dot per branch, the least-squares line, and the correlation with its
    branch count written in the corner the trend leaves empty.
    """
    out = _prepare_output(output)
    _apply_figure_style()

    x = [_float(point.get(x_key, point.get("x", 0.0))) for point in points]
    y = [_float(point.get(y_key, point.get("y", 0.0))) for point in points]
    r_value = _correlation(x, y, method)
    label_a = gene_a or str(points[0].get("gene_a", "gene A"))
    label_b = gene_b or str(points[0].get("gene_b", "gene B"))

    fig, ax = new_axes(journal_figsize(3.5, 3.2))
    ax.scatter(x, y, s=14, color=_PALETTE[0], alpha=0.8, edgecolor="white", linewidth=0.3, zorder=3)
    fit = _linear_fit(x, y)
    if fit is not None:
        slope, intercept = fit
        x_lo, x_hi = min(x), max(x)
        ax.plot(
            [x_lo, x_hi],
            [slope * x_lo + intercept, slope * x_hi + intercept],
            color=_ERC_INK,
            lw=0.8,
            zorder=2,
        )
    # A positive trend leaves the upper-left corner empty, a negative one the upper-right.
    corner_x, corner_ha = (0.03, "left") if r_value >= 0 else (0.97, "right")
    ax.text(
        corner_x,
        0.97,
        f"{method.title()} {_erc_stat_symbol(method)} = {r_value:.2f}\nn = {len(x)} branches",
        transform=ax.transAxes,
        ha=corner_ha,
        va="top",
        fontsize=FONT_PT["label"],
        color=_ERC_INK,
        linespacing=1.3,
    )
    if min(x) >= 0 and min(y) >= 0:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    ax.set_xlabel(f"Relative branch length, {label_a}")
    ax.set_ylabel(f"Relative branch length, {label_b}")
    _panel_title(ax, title)
    _despine(ax)
    return _save(fig, out)


def plot_erc_distribution(
    pairs: Sequence[Mapping[str, str | int | float | bool | None]],
    *,
    method: str = "pearson",
    r_threshold: float = 0.15,
    title: str | None = None,
) -> OrganellePlot:
    """Render the global distribution of ERC strengths for one method (compute-only)."""
    rows = _erc_rows_for_method(pairs, method)

    def _render(path: str | Path) -> Path:
        return _render_erc_distribution(
            rows, path, method=method, r_threshold=r_threshold, title=title
        )

    plot = plot_result(
        "plot_erc_distribution",
        _render,
        metrics={"method": method, "pairs": len(rows)},
        summary="ERC distribution prepared.",
    )
    plot._panel = renderer_panel(_render)
    return plot


def _erc_pair_group_text(label: str) -> str:
    """``"a vs b"`` group labels as ``"a-b"``; a single group reads ``"within a"``."""
    if " vs " in label:
        return label.replace(" vs ", "\u2013")
    return f"within {label}"


def _render_erc_distribution(
    rows: list[dict[str, Any]],
    output: str | Path,
    *,
    method: str,
    r_threshold: float = 0.15,
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_erc_distribution`.

    Histogram density of |r| with the bins at or above the threshold in colour,
    the median and threshold marked, and an inset counting the pairs above
    the threshold by gene-group combination.
    """
    out = _prepare_output(output)
    _apply_figure_style()

    values = [row["abs_r"] for row in rows]
    bins = 40
    width = 1.0 / bins
    centers, density = _hist_density(values, bins=bins, low=0.0, high=1.0)
    threshold = max(0.0, min(1.0, _float(r_threshold)))
    symbol = _erc_stat_symbol(method)

    fig, ax = new_axes(journal_figsize(4.72, 3.0))
    colors = [
        _PALETTE[0] if center - width / 2 >= threshold - 1e-9 else _ERC_GREY_BG
        for center in centers
    ]
    ax.bar(centers, density, width=width, color=colors, edgecolor="white", linewidth=0.25)
    peak = max(density) if max(density) > 0 else 1.0
    ax.set_ylim(0, peak * 1.2)
    label_y = peak * 1.13

    median = _median(values)
    ax.axvline(threshold, color=_ERC_INK, lw=0.6, ls=_ERC_DASH, alpha=0.85)
    ax.axvline(median, color=_ERC_INK, lw=0.6, alpha=0.85)
    # Labels sit on the outer side of their lines: the smaller value's to the left
    # of its line, the larger value's to the right. A left label without room
    # before the y axis moves one row down: right of its own line when it fits
    # between the two lines, otherwise right of the larger value's line.
    from .style import text_width_pt

    width_pt, height_pt = _erc_axes_size_pt(ax)
    row_step = FONT_PT["label"] * 1.5 * (peak * 1.2) / max(1.0, height_pt)
    text_kw = {"va": "center", "fontsize": FONT_PT["label"], "color": _ERC_INK}
    (low_x, low_text), (high_x, high_text) = sorted(
        [(median, f"median {median:.2f}"), (threshold, f"|{symbol}| \u2265 {threshold:.2f}")],
        key=lambda item: item[0],
    )
    ax.text(high_x + 0.008, label_y, high_text, ha="left", **text_kw)
    low_width_pt = text_width_pt(low_text, FONT_PT["label"]) + 2
    if (low_x - 0.008) * width_pt >= low_width_pt:
        ax.text(low_x - 0.008, label_y, low_text, ha="right", **text_kw)
    elif (high_x - low_x - 0.016) * width_pt >= low_width_pt:
        ax.text(low_x + 0.008, label_y - row_step, low_text, ha="left", **text_kw)
    else:
        ax.text(high_x + 0.008, label_y - row_step, low_text, ha="left", **text_kw)
    ax.set_xlim(0, 1)
    ax.set_xlabel(_erc_strength_label(method))
    ax.set_ylabel("Density")
    _panel_title(ax, title)

    counts: dict[str, int] = {}
    for row in rows:
        if row["abs_r"] >= threshold:
            label = _erc_pair_group_label(row)
            counts[label] = counts.get(label, 0) + 1
    if counts:
        total = sum(counts.values())
        labels = sorted(counts, key=lambda label: (-counts[label], label))[:4]
        bars = [counts[label] for label in labels]
        inset = ax.inset_axes([0.72, 0.38, 0.25, 0.4])
        positions = list(range(len(labels)))
        inset.barh(positions, bars, color=_PALETTE[0], height=0.62)
        for pos, value in zip(positions, bars, strict=False):
            inset.text(
                value + max(bars) * 0.04,
                pos,
                f"{value:,}",
                ha="left",
                va="center",
                fontsize=FONT_PT["label"],
                color=_ERC_INK,
            )
        inset.set_yticks(
            positions, [_erc_pair_group_text(label) for label in labels], fontsize=FONT_PT["label"]
        )
        inset.invert_yaxis()
        inset.set_xlim(0, max(bars) * 1.3)
        inset.set_xticks([])
        inset.tick_params(axis="y", length=0, pad=2)
        for side in ("top", "right", "bottom"):
            inset.spines[side].set_visible(False)
        inset.set_title(
            f"Pairs with |{symbol}| \u2265 {threshold:.2f} (n = {total:,})",
            fontsize=FONT_PT["label"],
            loc="right",
            pad=3,
        )
    _despine(ax)
    return _save(fig, out)


def plot_erc_group_ridges(
    pairs: Sequence[Mapping[str, str | int | float | bool | None]],
    *,
    method: str = "pearson",
    group_order: Sequence[str] | None = None,
    r_threshold: float = 0.15,
    title: str | None = None,
) -> OrganellePlot:
    """Render group-wise ERC-strength distributions as stacked ridge bands (compute-only)."""
    rows = _erc_rows_for_method(pairs, method)

    def _render(path: str | Path) -> Path:
        return _render_erc_group_ridges(
            rows, path, method=method, group_order=group_order, r_threshold=r_threshold, title=title
        )

    plot = plot_result(
        "plot_erc_group_ridges",
        _render,
        metrics={"method": method, "pairs": len(rows)},
        summary="ERC group ridges prepared.",
    )
    plot._panel = renderer_panel(_render)
    return plot


def _erc_unit_density(values: Sequence[float]) -> tuple[list[float], list[float]]:
    """Gaussian kernel density of values in [0, 1], reflected at both bounds.

    Binned first (0.0025 wide) and smoothed by convolution, so the cost stays
    linear in the number of pairs of a genome-wide scan.
    """
    import numpy as np

    data = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    n = int(data.size)
    spread = float(data.std(ddof=1)) if n > 1 else 0.0
    q75, q25 = np.percentile(data, [75, 25])
    iqr = float(q75 - q25)
    sigma = min(spread, iqr / 1.349) if iqr > 0 else spread
    bandwidth = min(0.25, max(0.015, 0.9 * sigma * n ** (-0.2)))
    edges = np.linspace(-1.0, 2.0, 1201)
    counts, _ = np.histogram(np.concatenate([data, -data, 2.0 - data]), bins=edges)
    step = float(edges[1] - edges[0])
    half = int(np.ceil(4 * bandwidth / step))
    kernel = np.exp(-0.5 * ((np.arange(-half, half + 1) * step) / bandwidth) ** 2)
    smooth = np.convolve(counts, kernel, mode="same") / (n * bandwidth * sqrt(2 * pi))
    centers = (edges[:-1] + edges[1:]) / 2
    inside = (centers >= 0.0) & (centers <= 1.0)
    return centers[inside].tolist(), smooth[inside].tolist()


def _render_erc_group_ridges(
    rows: list[dict[str, Any]],
    output: str | Path,
    *,
    method: str,
    group_order: Sequence[str] | None = None,
    r_threshold: float = 0.15,
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_erc_group_ridges`.

    One smoothed |r| density per gene group (a pair counts towards both of its
    genes' groups), each scaled to the same peak height on a shared baseline
    grid, with the group median and the threshold marked. Group colours match
    :func:`plot_erc_network`.
    """
    out = _prepare_output(output)
    _apply_figure_style()

    group_values: dict[str, list[float]] = {}
    for row in rows:
        groups = {str(row.get("group_a", "unassigned")), str(row.get("group_b", "unassigned"))}
        for group in groups:
            group_values.setdefault(group, []).append(row["abs_r"])
    ordered = [group for group in (group_order or []) if group in group_values]
    ordered.extend(group for group in sorted(group_values) if group not in ordered)
    if not ordered:
        raise ValueError("pairs must include at least one group label")
    palette = _group_palette(group_values)
    ridge_height = 0.9

    fig, ax = new_axes(journal_figsize(4.72, max(2.0, 0.55 * len(ordered) + 0.9)))
    for i, group in enumerate(ordered):
        base = len(ordered) - i - 1
        values = group_values[group]
        xs, density = _erc_unit_density(values)
        peak = max(density) if density and max(density) > 0 else 1.0
        tops = [base + (value / peak) * ridge_height for value in density]
        color = palette[group]
        zorder = 2 + 2 * i  # lower ridges in front of the ones above
        ax.fill_between(xs, base, tops, color="white", lw=0, zorder=zorder)
        ax.fill_between(xs, base, tops, color=color, alpha=0.35, lw=0, zorder=zorder)
        ax.plot(xs, tops, color=color, lw=0.8, zorder=zorder + 0.5)
        ax.plot([0.0, 1.0], [base, base], color=_ERC_GREY_MID, lw=0.5, zorder=zorder + 0.5)
        median = _median(values)
        index = min(range(len(xs)), key=lambda k: abs(xs[k] - median))
        ax.plot([median, median], [base, tops[index]], color=_ERC_INK, lw=0.8, zorder=zorder + 0.6)
    threshold = max(0.0, min(1.0, _float(r_threshold)))
    ax.axvline(
        threshold, color=_ERC_INK, lw=0.6, ls=_ERC_DASH, alpha=0.8, zorder=2 * len(ordered) + 3
    )
    symbol = _erc_stat_symbol(method)
    ax.legend(
        handles=[
            Line2D([], [], color=_ERC_INK, lw=0.8, label="Median"),
            Line2D(
                [], [], color=_ERC_INK, lw=0.6, ls=_ERC_DASH, label=f"|{symbol}| = {threshold:.2f}"
            ),
        ],
        loc="upper right",
        fontsize=FONT_PT["label"],
        handlelength=1.6,
        borderaxespad=0.2,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.12, len(ordered) - 1 + ridge_height + 0.2)
    ax.set_yticks(
        [len(ordered) - i - 1 + 0.22 for i in range(len(ordered))],
        [f"{group}\nn = {len(group_values[group]):,}" for group in ordered],
    )
    ax.set_xlabel(_erc_strength_label(method))
    _panel_title(ax, title)
    _despine(ax)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    return _save(fig, out)


def plot_erc_network(
    pairs: Sequence[Mapping[str, str | int | float | bool | None]],
    *,
    method: str = "pearson",
    r_min: float = 0.15,
    q_max: float | None = None,
    top_n: int | None = 250,
    label_top_n: int = 28,
    title: str | None = None,
) -> OrganellePlot:
    """Render a filtered high-ERC network from a large pair table (compute-only)."""
    rows = _erc_rows_for_method(pairs, method)
    threshold = max(0.0, _float(r_min))
    filtered = [row for row in rows if row["abs_r"] >= threshold]
    if q_max is not None:
        q_limit = _float(q_max)
        filtered = [
            row for row in filtered if row.get("q_value") is None or row["q_value"] <= q_limit
        ]
    filtered.sort(key=lambda row: row["abs_r"], reverse=True)
    if top_n is not None:
        filtered = filtered[: max(1, int(top_n))]
    if not filtered:
        raise ValueError("no ERC pairs remain after filtering")

    def _render(path: str | Path) -> Path:
        return _render_erc_network(
            filtered, path, method=method, label_top_n=label_top_n, title=title
        )

    plot = plot_result(
        "plot_erc_network",
        _render,
        metrics={"method": method, "edges": len(filtered)},
        summary="ERC network prepared.",
    )
    plot._panel = renderer_panel(_render)
    return plot


def _erc_natural_key(label: str) -> tuple[Any, ...]:
    """Sort ``nad2`` before ``nad10``."""
    parts: list[Any] = []
    for is_digit, chunk in itertools.groupby(label, str.isdigit):
        text = "".join(chunk)
        parts.append((0, int(text), "") if is_digit else (1, 0, text.lower()))
    return tuple(parts)


def _erc_circle_layout(
    groups_in_order: Sequence[tuple[str, Sequence[str]]], gap_slots: float = 1.5
) -> tuple[dict[str, float], float]:
    """Angle of every node on one circle, clockwise from the top, a gap between groups."""
    gaps = gap_slots if len(groups_in_order) > 1 else 0.0
    total = sum(len(nodes) for _, nodes in groups_in_order) + gaps * len(groups_in_order)
    angles: dict[str, float] = {}
    slot = gaps / 2
    for _, nodes in groups_in_order:
        for node in nodes:
            angles[node] = pi / 2 - 2 * pi * (slot + 0.5) / total
            slot += 1
        slot += gaps
    return angles, total


def _render_erc_network(
    filtered: list[dict[str, Any]],
    output: str | Path,
    *,
    method: str,
    label_top_n: int = 28,
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_erc_network`.

    Genes sit on one circle in gene-group blocks (natural name order inside a
    block), sized by degree; edges bow towards the centre, wider and darker for
    stronger |r|, coloured by group when both genes share one and grey across
    groups, dashed for negative correlations. Labels run radially outside the
    circle so neighbours never overlap.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    from .style import text_width_pt

    out = _prepare_output(output)
    _apply_figure_style()

    degrees: dict[str, int] = {}
    groups: dict[str, str] = {}
    edges: list[dict[str, Any]] = []
    for row in filtered:
        gene_a = str(row["gene_a"])
        gene_b = str(row["gene_b"])
        degrees[gene_a] = degrees.get(gene_a, 0) + 1
        degrees[gene_b] = degrees.get(gene_b, 0) + 1
        groups.setdefault(gene_a, str(row.get("group_a", "unassigned")))
        groups.setdefault(gene_b, str(row.get("group_b", "unassigned")))
        edges.append(
            {
                "source": gene_a,
                "target": gene_b,
                "weight": row["abs_r"],
                "negative": _float(row.get("r", 0.0)) < 0,
            }
        )
    palette = _group_palette(groups.values())
    group_names = sorted(set(groups.values()))
    groups_in_order = [
        (
            group,
            sorted((gene for gene in degrees if groups[gene] == group), key=_erc_natural_key),
        )
        for group in group_names
    ]
    angles, total_slots = _erc_circle_layout(groups_in_order)
    labelled = set(sorted(degrees, key=lambda gene: (-degrees[gene], gene))[: max(0, label_top_n)])
    label_pt = FONT_PT["label"]
    widest = max(
        (text_width_pt(_compact_gene_label(gene), label_pt) for gene in labelled), default=0.0
    )
    max_degree = max(degrees.values())

    fig, ax = new_axes(journal_figsize(4.72, 5.0))
    width_pt, height_pt = _erc_axes_size_pt(ax, bare=True)
    legend_pt = 34.0  # two legend rows under the circle
    side_pt = max(60.0, min(width_pt, height_pt - legend_pt))
    # Circle radius 1 in data units; the limit leaves room for node and radial label.
    d_max = 10.0
    limit = 1.2
    for _ in range(3):
        room_pt = d_max / 2 + 2.0 + widest + 3.0
        limit = min(2.2, max(1.12, 1.0 / max(0.2, 1.0 - 2.0 * room_pt / side_pt)))
        radius_pt = side_pt / (2 * limit)
        spacing_pt = 2 * pi * radius_pt / max(1.0, total_slots)
        d_max = max(3.0, min(10.0, 0.85 * spacing_pt))
    d_min = max(2.5, 0.4 * d_max)
    data_per_pt = 2 * limit / side_pt

    def diameter(gene: str) -> float:
        return d_min + (d_max - d_min) * sqrt(degrees[gene] / max_degree)

    positions = {gene: (cos(theta), sin(theta)) for gene, theta in angles.items()}
    weights = [_float(edge["weight"]) for edge in edges]
    w_lo, w_hi = min(weights), max(weights)

    def edge_width(weight: float) -> float:
        share = (weight - w_lo) / (w_hi - w_lo) if w_hi > w_lo else 1.0
        return 0.35 + 1.25 * share

    for edge in sorted(edges, key=lambda item: _float(item["weight"])):
        source = str(edge["source"])
        target = str(edge["target"])
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        # Bow towards the centre, more for distant genes, so neighbours stay short arcs.
        pull = max(0.0, 1.0 - 0.5 * sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        control = ((x1 + x2) / 2 * pull, (y1 + y2) / 2 * pull)
        weight = _float(edge["weight"])
        share = (weight - w_lo) / (w_hi - w_lo) if w_hi > w_lo else 1.0
        same_group = groups[source] == groups[target]
        path = MplPath(
            [(x1, y1), control, (x2, y2)], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
        )
        ax.add_patch(
            PathPatch(
                path,
                facecolor="none",
                edgecolor=palette[groups[source]] if same_group else _ERC_GREY_MID,
                lw=edge_width(weight),
                alpha=0.25 + 0.5 * share,
                ls=_ERC_DASH if edge["negative"] else "-",
                capstyle="round",
                zorder=1,
            )
        )
    for group, genes in groups_in_order:
        ax.scatter(
            [positions[gene][0] for gene in genes],
            [positions[gene][1] for gene in genes],
            s=[diameter(gene) ** 2 for gene in genes],
            color=palette[group],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    for gene in labelled:
        theta = angles[gene]
        offset = 1.0 + (diameter(gene) / 2 + 2.0) * data_per_pt
        degrees_angle = theta * 180 / pi
        right_side = cos(theta) >= -1e-9
        ax.text(
            offset * cos(theta),
            offset * sin(theta),
            _compact_gene_label(gene),
            rotation=degrees_angle if right_side else degrees_angle + 180,
            rotation_mode="anchor",
            ha="left" if right_side else "right",
            va="center",
            fontsize=label_pt,
            color=_ERC_INK,
            zorder=4,
        )

    symbol = _erc_stat_symbol(method)
    group_handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            markersize=5,
            markerfacecolor=palette[group],
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=f"{group} ({len(genes)})",
        )
        for group, genes in groups_in_order
    ]
    encoding_handles = [
        Line2D([], [], color=_ERC_GREY_MID, lw=edge_width(w_lo), label=f"|{symbol}| = {w_lo:.2f}")
    ]
    if w_hi > w_lo:
        encoding_handles.append(
            Line2D(
                [], [], color=_ERC_GREY_MID, lw=edge_width(w_hi), label=f"|{symbol}| = {w_hi:.2f}"
            )
        )
    if any(edge["negative"] for edge in edges):
        encoding_handles.append(
            Line2D([], [], color=_ERC_GREY_MID, lw=0.8, ls=_ERC_DASH, label=f"{symbol} < 0")
        )
    min_degree = min(degrees.values())
    for degree in sorted({min_degree, max_degree}):
        encoding_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                markersize=d_min + (d_max - d_min) * sqrt(degree / max_degree),
                markerfacecolor=_ERC_GREY_MID,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=f"degree {degree}",
            )
        )
    legend_kw = {
        "loc": "lower center",
        "fontsize": FONT_PT["label"],
        "handlelength": 1.6,
        "columnspacing": 1.2,
        "handletextpad": 0.4,
        "borderaxespad": 0.0,
    }
    group_legend = ax.legend(
        handles=group_handles,
        ncol=len(group_handles),
        bbox_to_anchor=(0.5, 0.06),
        **legend_kw,
    )
    ax.add_artist(group_legend)
    ax.legend(
        handles=encoding_handles,
        ncol=len(encoding_handles),
        bbox_to_anchor=(0.5, 0.0),
        **legend_kw,
    )
    _panel_title(ax, title)
    ax.set_aspect("equal")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit - legend_pt * data_per_pt, limit)
    ax.set_axis_off()
    return _save(fig, out)


def plot_erc_significance(
    pairs: Sequence[Mapping[str, str | int | float | bool | None]],
    *,
    method: str = "pearson",
    highlight_genes: Sequence[str] | None = None,
    highlight_groups: Sequence[str] | None = None,
    q_threshold: float = 0.05,
    title: str | None = None,
    r_threshold: float = 0.15,
    strong_r: float | None = 0.5,
    label_top_n: int = 4,
) -> OrganellePlot:
    """Render ERC strength against significance, highlighting candidates (compute-only).

    Pairs pass when ``q <= q_threshold`` (``p`` when a row has no q value) and
    ``|r| >= r_threshold``; ``strong_r`` draws a lighter reference line and the
    ``label_top_n`` most significant highlighted pairs (then strongest) are named.
    """
    rows = _erc_rows_for_method(pairs, method)

    def _render(path: str | Path) -> Path:
        return _render_erc_significance(
            rows,
            path,
            method=method,
            highlight_genes=highlight_genes,
            highlight_groups=highlight_groups,
            q_threshold=q_threshold,
            title=title,
            r_threshold=r_threshold,
            strong_r=strong_r,
            label_top_n=label_top_n,
        )

    plot = plot_result(
        "plot_erc_significance",
        _render,
        metrics={"method": method, "pairs": len(rows)},
        summary="ERC significance plot prepared.",
    )
    plot._panel = renderer_panel(_render)
    return plot


def _erc_significance_score(row: Mapping[str, Any]) -> float:
    """-log10 of the q value, or of the p value when the row has no q value."""
    q_value = row.get("q_value")
    p_value = row.get("p_value")
    value = q_value if q_value is not None else p_value if p_value is not None else 1.0
    return _neg_log10(value)


def _erc_spread_positions(targets: Sequence[float], gap: float, low: float) -> list[float]:
    """Move 1-D label positions apart by at least ``gap`` and keep them above ``low``."""
    order = sorted(range(len(targets)), key=lambda i: -targets[i])
    placed = [0.0 for _ in targets]
    previous: float | None = None
    for i in order:
        value = targets[i] if previous is None else min(targets[i], previous - gap)
        placed[i] = value
        previous = value
    if order:
        shortfall = low - placed[order[-1]]
        if shortfall > 0:
            for i in order:
                placed[i] += shortfall
    return placed


def _erc_legend_spot(
    width: float,
    height: float,
    points_x: Sequence[float],
    points_y: Sequence[float],
    boxes: Sequence[tuple[float, float, float, float]],
    vlines: Sequence[float],
    hlines: Sequence[float],
    *,
    margin_x: float,
    margin_y: float,
) -> tuple[float, float] | None:
    """Upper-left corner (axes fractions) of the first empty legend rectangle.

    Scans from the top down and from the left across; a spot is empty when no
    point, box (``x0, y0, x1, y1``) or threshold line falls inside it with its
    margin. ``None`` when the axes have no such room.
    """
    import numpy as np

    px = np.asarray(points_x, dtype=float)
    py = np.asarray(points_y, dtype=float)
    if width + 2 * margin_x > 1.0 or height + 2 * margin_y > 1.0:
        return None
    for top in np.linspace(1.0 - margin_y, height + margin_y, 60):
        for left in np.linspace(margin_x, 1.0 - width - margin_x, 60):
            x0, x1 = left - margin_x, left + width + margin_x
            y0, y1 = top - height - margin_y, top + margin_y
            if any(x0 < v < x1 for v in vlines) or any(y0 < h < y1 for h in hlines):
                continue
            if any(bx0 < x1 and bx1 > x0 and by0 < y1 and by1 > y0 for bx0, by0, bx1, by1 in boxes):
                continue
            if np.any((px > x0) & (px < x1) & (py > y0) & (py < y1)):
                continue
            return float(left), float(top)
    return None


def _render_erc_significance(
    rows: list[dict[str, Any]],
    output: str | Path,
    *,
    method: str,
    highlight_genes: Sequence[str] | None = None,
    highlight_groups: Sequence[str] | None = None,
    q_threshold: float = 0.05,
    title: str | None = None,
    r_threshold: float = 0.15,
    strong_r: float | None = 0.5,
    label_top_n: int = 4,
) -> Path:
    """Private path-taking renderer backing :func:`plot_erc_significance`.

    Pairs outside the thresholds are pale grey, other passing pairs mid grey,
    highlighted pairs orange (pale when they fail). The ``label_top_n`` most
    significant highlighted pairs are labelled in stacks beside their points
    with leader lines; the legend takes the first empty spot of the axes.
    """
    out = _prepare_output(output)
    _apply_figure_style()

    highlight_gene_set = {str(gene) for gene in highlight_genes or ()}
    highlight_group_set = {str(group) for group in highlight_groups or ()}
    uses_q = any(row.get("q_value") is not None for row in rows)
    stat_name = "q" if uses_q else "P"
    symbol = _erc_stat_symbol(method)
    q_limit = _float(q_threshold)
    q_line = _neg_log10(q_limit)
    r_limit = max(0.0, _float(r_threshold))
    x = [row["abs_r"] for row in rows]
    y = [_erc_significance_score(row) for row in rows]
    highlighted = [
        (
            str(row["gene_a"]) in highlight_gene_set
            or str(row["gene_b"]) in highlight_gene_set
            or str(row.get("group_a", "")) in highlight_group_set
            or str(row.get("group_b", "")) in highlight_group_set
        )
        for row in rows
    ]
    passing = [xi >= r_limit and yi >= q_line for xi, yi in zip(x, y, strict=False)]

    fig, ax = new_axes(journal_figsize(4.72, 3.7))
    layers = (
        (False, False, _ERC_GREY_BG, 5, "Not passing"),
        (False, True, _ERC_GREY_MID, 8, "Passing"),
        (True, False, _ERC_HIT_LIGHT, 8, "Highlighted, not passing"),
        (True, True, _ALARM, 16, "Highlighted, passing"),
    )
    handles = []
    for zorder, (is_hit, is_pass, color, size, label) in enumerate(layers, start=1):
        idx = [i for i in range(len(rows)) if highlighted[i] == is_hit and passing[i] == is_pass]
        if not idx:
            continue
        ax.scatter(
            [x[i] for i in idx],
            [y[i] for i in idx],
            s=size,
            color=color,
            edgecolor="white" if is_hit and is_pass else "none",
            linewidth=0.3,
            zorder=zorder,
        )
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                markersize=sqrt(size),
                markerfacecolor=color,
                markeredgecolor="none",
                label=label,
            )
        )
    x_max = min(1.0, max(0.25, max(x) + 0.08))
    y_max = max(2.0, max(y) * 1.12)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    line_kw = {"color": _ERC_INK, "lw": 0.6, "ls": _ERC_DASH, "zorder": 0}
    ax.axhline(q_line, alpha=0.8, **line_kw)
    ax.annotate(
        f"{stat_name} = {q_limit:g}",
        xy=(1.0, q_line),
        xycoords=ax.get_yaxis_transform(),
        xytext=(0, 1.5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=FONT_PT["label"],
        color=_ERC_INK,
    )
    vertical = [(r_limit, 0.8)]
    if strong_r is not None and 0 < _float(strong_r) < x_max and _float(strong_r) != r_limit:
        vertical.append((_float(strong_r), 0.4))
    vertical_drawn: list[float] = []
    for value, alpha in vertical:
        if not 0 < value < x_max:
            continue
        vertical_drawn.append(value)
        ax.axvline(value, alpha=alpha, **line_kw)
        ax.annotate(
            f"|{symbol}| = {value:g}",
            xy=(value, 1.0),
            xycoords=ax.get_xaxis_transform(),
            xytext=(-1.5, -2),
            textcoords="offset points",
            rotation=90,
            ha="right",
            va="top",
            fontsize=FONT_PT["label"],
            color=_ERC_INK,
        )

    # Candidates: highlighted pairs that pass (all highlighted pairs when none
    # pass), most significant first, then strongest; every one of the first
    # label_top_n is labelled, label stacks get headroom instead of being dropped.
    from .style import text_width_pt

    candidates = [i for i in range(len(rows)) if highlighted[i] and passing[i]]
    if not candidates:
        candidates = [i for i in range(len(rows)) if highlighted[i]]
    chosen = sorted(candidates, key=lambda i: (-y[i], -x[i], i))[: max(0, label_top_n)]
    width_pt, height_pt = _erc_axes_size_pt(ax)
    x_per_pt = x_max / max(1.0, width_pt)
    label_pt = FONT_PT["label"]
    obstacles: list[tuple[float, float, float, float]] = []  # boxes in axes fractions
    sides = {
        left_side: [i for i in chosen if (x[i] >= x_max * 0.45) == left_side]
        for left_side in (True, False)
    }
    placements: dict[bool, list[float]] = {}
    gap = 0.0
    for _ in range(3):
        y_per_pt = y_max / max(1.0, height_pt)
        gap = label_pt * 1.45 * y_per_pt
        placements = {
            left_side: _erc_spread_positions([y[i] for i in side], gap, gap / 2)
            for left_side, side in sides.items()
            if side
        }
        top = max((max(placed) for placed in placements.values()), default=0.0)
        if top <= y_max - gap / 2:
            break
        y_max = top + gap
    ax.set_ylim(0, y_max)
    shift = 9.0 * x_per_pt
    for left_side, placed in placements.items():
        side = sides[left_side]
        column_x = min(x[i] for i in side) - shift if left_side else max(x[i] for i in side) + shift
        for i, label_y in zip(side, placed, strict=False):
            row = rows[i]
            text = (
                f"{_compact_gene_label(str(row['gene_a']))}\u2013"
                f"{_compact_gene_label(str(row['gene_b']))}"
            )
            ax.annotate(
                text,
                xy=(x[i], y[i]),
                xytext=(column_x, label_y),
                ha="right" if left_side else "left",
                va="center",
                fontsize=label_pt,
                color=_ERC_INK,
                zorder=6,
                bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.85},
                arrowprops={
                    "arrowstyle": "-",
                    "color": _ERC_GREY_MID,
                    "lw": 0.4,
                    "shrinkA": 1.0,
                    "shrinkB": 2.5,
                },
            )
            text_w = text_width_pt(text, label_pt) * x_per_pt
            x_lo = column_x - text_w if left_side else column_x
            x_hi = column_x if left_side else column_x + text_w
            obstacles.append(
                (
                    min(x_lo, x[i]) / x_max,
                    (min(label_y, y[i]) - gap / 2) / y_max,
                    max(x_hi, x[i]) / x_max,
                    (max(label_y, y[i]) + gap / 2) / y_max,
                )
            )

    # Threshold-line labels are obstacles for the legend as well.
    line_text_h = (label_pt * 1.2 + 2.0) / max(1.0, width_pt)
    for value in vertical_drawn:
        frac = value / x_max
        length = (text_width_pt(f"|{symbol}| = {value:g}", label_pt) + 3.0) / max(1.0, height_pt)
        obstacles.append((frac - line_text_h, 1.0 - length, frac, 1.0))
    q_frac = q_line / y_max
    q_text_w = (text_width_pt(f"{stat_name} = {q_limit:g}", label_pt) + 2.0) / max(1.0, width_pt)
    obstacles.append(
        (1.0 - q_text_w, q_frac, 1.0, q_frac + (label_pt * 1.2 + 2.0) / max(1.0, height_pt))
    )

    if handles:
        legend_kw = {
            "handles": handles,
            "loc": "upper left",
            "fontsize": label_pt,
            "handletextpad": 0.3,
            "handlelength": 1.2,
            "borderaxespad": 0.0,
            "frameon": False,
        }
        # Legend size from its text; then the first empty spot, top to bottom and
        # left to right, that crosses no point, line, label or line label.
        legend_w = (
            max(text_width_pt(str(handle.get_label()), label_pt) for handle in handles)
            + label_pt * (1.2 + 0.3 + 0.8)
        ) / max(1.0, width_pt)
        legend_h = (label_pt * (1.5 * len(handles) - 0.5 + 0.8)) / max(1.0, height_pt)
        spot = _erc_legend_spot(
            legend_w,
            legend_h,
            [xi / x_max for xi in x],
            [yi / y_max for yi in y],
            obstacles,
            [value / x_max for value in vertical_drawn],
            [q_frac],
            margin_x=3.0 / max(1.0, width_pt),
            margin_y=3.0 / max(1.0, height_pt),
        )
        if spot is None:  # no room inside: beside the plot
            ax.legend(bbox_to_anchor=(1.02, 1.0), **legend_kw)
        else:
            ax.legend(bbox_to_anchor=spot, **legend_kw)
    ax.set_xlabel(_erc_strength_label(method))
    ax.set_ylabel(f"\u2212log10 {stat_name}")
    _panel_title(ax, title)
    _despine(ax)
    return _save(fig, out)


def write_erc_visualization_report(
    figures: Sequence[Mapping[str, str]],
    output_html: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Write a static HTML gallery for ERC visualization previews."""
    out = _prepare_output(output_html)
    page_title = title or "ERC visualization preview"
    cards = []
    for figure in figures:
        path = Path(figure.get("path", ""))
        label = str(figure.get("title", path.stem or "Figure"))
        method = str(figure.get("method", ""))
        rel = path.name if path.parent == out.parent else str(path)
        method_html = f"<span>{escape(method)}</span>" if method else ""
        cards.append(
            "<section>"
            f'<div class="meta">{method_html}</div>'
            f"<h2>{escape(label)}</h2>"
            f'<img src="{escape(rel)}" alt="{escape(label)}" />'
            f"<p>{escape(path.name)}</p>"
            "</section>"
        )
    html = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8" />'
        f"<title>{escape(page_title)}</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:28px;color:#172033;background:#fff;}"
        "h1{font-size:24px;margin-bottom:6px;}h2{font-size:17px;margin:0 0 10px;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:24px;}"
        "section{border:1px solid #d9e2ec;border-radius:6px;padding:14px;background:#fff;}"
        "img{max-width:100%;height:auto;display:block;border:1px solid #eef2f7;}"
        "p{font-size:12px;color:#52606d;}.meta{height:20px;}"
        ".meta span{font-size:11px;text-transform:uppercase;letter-spacing:0;"
        "background:#edf2f7;border:1px solid #d9e2ec;border-radius:4px;padding:3px 6px;}"
        "</style></head><body>"
        f"<h1>{escape(page_title)}</h1>"
        "<p>Static ERC figures generated from large pair-table shaped data. "
        "Rows can be filtered by Pearson or Spearman correlation method.</p>"
        f'<div class="grid">{"".join(cards)}</div>'
        "</body></html>"
    )
    out.write_text(html)
    return out


def write_visualization_preview_report(
    figures: Sequence[Mapping[str, str]],
    output_html: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Write a static HTML gallery for generated visualization previews."""
    out = _prepare_output(output_html)
    page_title = title or "OrganelleVerse visualization preview"
    cards = []
    for figure in figures:
        path = Path(figure.get("path", ""))
        label = str(figure.get("title", path.stem or "Figure"))
        rel = path.name if path.parent == out.parent else str(path)
        cards.append(
            "<section>"
            f"<h2>{escape(label)}</h2>"
            f'<img src="{escape(rel)}" alt="{escape(label)}" />'
            f"<p>{escape(path.name)}</p>"
            "</section>"
        )
    html = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8" />'
        f"<title>{escape(page_title)}</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:28px;color:#172033;background:#fff;}"
        "h1{font-size:24px;margin-bottom:6px;}h2{font-size:17px;margin:0 0 10px;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:24px;}"
        "section{border:1px solid #d9e2ec;border-radius:6px;padding:14px;background:#fff;}"
        "img{max-width:100%;height:auto;display:block;border:1px solid #eef2f7;}"
        "p{font-size:12px;color:#52606d;}"
        "</style></head><body>"
        f"<h1>{escape(page_title)}</h1>"
        "<p>Static demo figures generated from synthetic OrganelleVerse-shaped data.</p>"
        f'<div class="grid">{"".join(cards)}</div>'
        "</body></html>"
    )
    out.write_text(html)
    return out


def _prepare_output(output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _apply_figure_style() -> None:
    """Journal typography (``style.publication_rc``) with open top/right spines."""
    plt.rcParams.update(
        {**publication_rc(), "figure.dpi": 120,
         "axes.spines.top": False, "axes.spines.right": False}
    )


def _save(fig: Any, output: Path) -> Path:
    return finish(fig, output)


def _panel_title(ax: Any, title: str | None) -> None:
    """A left-aligned panel title only when the caller gives one (no default titles)."""
    if title:
        ax.set_title(title, loc="left", fontsize=FONT_PT["title"], pad=3)


def _figure_title(fig: Any, title: str | None) -> None:
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=FONT_PT["title"])


def _figure_size(
    default_width: float,
    default_height: float,
    width: float | None = None,
    height: float | None = None,
) -> tuple[float, float]:
    # defaults are mapped to the journal width; explicit sizes are the caller's
    default_width, default_height = journal_figsize(default_width, default_height)
    resolved_width = default_width if width is None else float(width)
    resolved_height = default_height if height is None else float(height)
    if resolved_width <= 0 or resolved_height <= 0:
        raise ValueError("figure width and height must be positive")
    return resolved_width, resolved_height


def _despine(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_list(values: Iterable[Any], label: str) -> list[float]:
    try:
        return [float(v) for v in values]
    except TypeError as exc:
        raise ValueError(f"{label} must be an iterable of numbers") from exc


def _matrix(matrix: Sequence[Sequence[Any]]) -> list[list[float]]:
    rows = [[float(v) for v in row] for row in matrix]
    if not rows:
        return rows
    expected = len(rows[0])
    if expected == 0 or any(len(row) != expected for row in rows):
        raise ValueError("matrix rows must all have the same length")
    return rows


def _synteny_matrix_data(
    blocks: Sequence[Mapping[str, str | int | float | None]] | Mapping[str, Sequence[Mapping[str, str | int | float | None]] | Mapping[str, Sequence[Mapping[str, str | int | float | None]]]],
    *,
    labels: Sequence[str] | None,
    genome_lengths: Mapping[str, int | float] | None,
) -> dict[str, Any]:
    rows = [_synteny_row(row) for row in _synteny_raw_rows(blocks)]
    rows = [row for row in rows if row is not None]
    if not rows:
        raise ValueError("synteny blocks must contain at least one coordinate block")

    ordered_labels = list(labels or [])
    seen = set(ordered_labels)
    for row in rows:
        for name in (row["genome_a"], row["genome_b"]):
            if name not in seen:
                ordered_labels.append(name)
                seen.add(name)
    if len(ordered_labels) < 2:
        raise ValueError("synteny matrix needs at least two genomes")

    lengths = {str(key): max(1.0, float(value)) for key, value in (genome_lengths or {}).items()}
    for row in rows:
        for name, (start, end) in row["coords"].items():
            lengths[name] = max(lengths.get(name, 1.0), abs(start), abs(end), 1.0)
    for name in ordered_labels:
        lengths.setdefault(name, 1.0)

    known = set(ordered_labels)
    rows = [row for row in rows if row["genome_a"] in known and row["genome_b"] in known]
    return {"labels": ordered_labels, "rows": rows, "lengths": lengths}


def _synteny_raw_rows(
    blocks: Sequence[Mapping[str, str | int | float | None]] | Mapping[str, Sequence[Mapping[str, str | int | float | None]] | Mapping[str, Sequence[Mapping[str, str | int | float | None]]]],
) -> Iterable[Mapping[str, Any]]:
    if isinstance(blocks, OrganelleResult):
        return _synteny_raw_rows(thaw_json(blocks.metrics))
    if isinstance(blocks, Mapping):
        observed = blocks.get("observed_metrics")
        if isinstance(observed, Mapping):
            return _synteny_raw_rows(observed)
        for key in ("blocks", "synteny_blocks", "alignments", "matches", "rows"):
            value = blocks.get(key)
            if isinstance(value, (list, tuple)):
                return value
        return [blocks]
    return blocks


def _synteny_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    genome_a = _string_value(
        _case_get(
            row, "genome_a", "sample_a", "query", "query_id", "qseqid", "source", "genome1", "x_id"
        )
    )
    genome_b = _string_value(
        _case_get(
            row,
            "genome_b",
            "sample_b",
            "target",
            "target_id",
            "subject",
            "sseqid",
            "genome2",
            "y_id",
        )
    )
    if not genome_a or not genome_b:
        return None
    a_start = _coord_value(
        row, ("a_start", "start_a", "query_start", "q_start", "qstart", "start1", "x_start")
    )
    a_end = _coord_value(row, ("a_end", "end_a", "query_end", "q_end", "qend", "end1", "x_end"))
    b_start = _coord_value(
        row,
        (
            "b_start",
            "start_b",
            "target_start",
            "subject_start",
            "s_start",
            "sstart",
            "start2",
            "y_start",
        ),
    )
    b_end = _coord_value(
        row, ("b_end", "end_b", "target_end", "subject_end", "s_end", "send", "end2", "y_end")
    )
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return None
    orientation = _synteny_orientation(row, a_start, a_end, b_start, b_end)
    return {
        "genome_a": genome_a,
        "genome_b": genome_b,
        "coords": {
            genome_a: (a_start, a_end),
            genome_b: (b_start, b_end),
        },
        "orientation": orientation,
    }


def _coord_value(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    value = _case_get(row, *keys)
    if value is None or value == "":
        return None
    return _float(value)


def _synteny_orientation(
    row: Mapping[str, Any],
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> str:
    raw = str(_case_get(row, "orientation", "direction", "strand", "type") or "").strip().lower()
    if raw in {"-", "minus", "reverse", "inverted", "inversion", "inv"}:
        return "inverted"
    if raw in {"+", "plus", "forward", "direct", "same", "collinear"}:
        return "direct"
    return "direct" if (a_end - a_start) * (b_end - b_start) >= 0 else "inverted"


def _string_value(value: Any) -> str:
    return "" if value is None else str(value)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _synteny_group_palette(
    names: Sequence[str],
    groups: Mapping[str, str] | None,
) -> dict[str, str]:
    group_map = groups or {}
    fallback = _group_palette(group_map.get(name, name) for name in names)
    colors: dict[str, str] = {}
    for name in names:
        group = group_map.get(name, "")
        colors[name] = _SYNTENY_GROUP_COLORS.get(group, fallback.get(group or name, _NEUTRAL))
    return colors


def _ordered_groups(names: Sequence[str], groups: Mapping[str, str] | None) -> list[str]:
    if not groups:
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for name in names:
        group = str(groups.get(name, ""))
        if group and group not in seen:
            ordered.append(group)
            seen.add(group)
    return ordered


def _format_bp(value: int) -> str:
    if value >= 1_000_000:
        return _trim_float(value / 1_000_000, 1) + " Mb"
    if value >= 1_000:
        return _trim_float(value / 1_000, 0) + " kb"
    return f"{value} bp"


def _trim_float(value: float, digits: int) -> str:
    text = f"{value:.{digits}f}"
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".")


def _format_number(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.2g}"


def _text_color_for_value(value: float) -> str:
    return "white" if value > 0.58 or value < 0.18 else "#111827"


def _network_positions(
    node_ids: Sequence[str], groups: Mapping[str, str]
) -> dict[str, tuple[float, float]]:
    grouped: dict[str, list[str]] = {}
    for node_id in node_ids:
        grouped.setdefault(groups.get(node_id, ""), []).append(node_id)
    ordered = [node for group in sorted(grouped) for node in grouped[group]]
    positions = {}
    for i, node_id in enumerate(ordered):
        angle = 2 * pi * i / max(1, len(ordered)) + pi / 2
        positions[node_id] = (cos(angle), sin(angle))
    return positions


def _group_palette(groups: Iterable[str]) -> dict[str, str]:
    unique = sorted({str(group) for group in groups})
    return {group: _PALETTE[i % len(_PALETTE)] for i, group in enumerate(unique)}


def _rscu_rows(
    rscu_rows: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path | Any,
) -> list[dict[str, Any]]:
    if isinstance(rscu_rows, (str, Path)):
        raw_rows: Iterable[Mapping[str, Any]] = _read_rscu_table(Path(rscu_rows))
    elif isinstance(rscu_rows, OrganelleResult):
        return _rscu_rows(thaw_json(rscu_rows.metrics))
    elif isinstance(rscu_rows, Mapping):
        observed = rscu_rows.get("observed_metrics")
        if isinstance(observed, Mapping):
            return _rscu_rows(observed)
        codon_table = rscu_rows.get("codon_table")
        if isinstance(codon_table, (list, tuple)):
            raw_rows = codon_table
        else:
            rscu = rscu_rows.get("RSCU", rscu_rows.get("rscu"))
            if isinstance(rscu, Mapping):
                counts = rscu_rows.get("counts", rscu_rows.get("codon_counts", {}))
                raw_rows = [
                    {
                        "codon": codon,
                        "RSCU": value,
                        "count": counts.get(codon, 0) if isinstance(counts, Mapping) else 0,
                    }
                    for codon, value in rscu.items()
                ]
            else:
                raw_rows = [rscu_rows]
    else:
        raw_rows = rscu_rows
    rows = [_rscu_row(row) for row in raw_rows]
    return [row for row in rows if row is not None]


def _read_rscu_table(path: Path) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    if not lines:
        return []
    delimiter = "\t" if "\t" in lines[0] else ","
    return [dict(row) for row in csv.DictReader(lines, delimiter=delimiter)]


def _rscu_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    codon_raw = _case_get(row, "codon", "triplet")
    if codon_raw is None:
        return None
    codon = str(codon_raw).upper().replace("U", "T")
    if len(codon) != 3:
        return None
    aa_raw = _case_get(row, "AA", "aa", "amino_acid", "amino acid")
    aa = str(aa_raw) if aa_raw not in (None, "") else _STANDARD_CODONS.get(codon, "")
    if not aa:
        return None
    return {
        "codon": codon,
        "aa": aa,
        "count": _float(_case_get(row, "count", "counts") or 0),
        "rscu": _float(_case_get(row, "RSCU", "rscu") or 0),
    }


def _case_get(row: Mapping[str, Any], *keys: str) -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def _display_codon(codon: str, rna_labels: bool) -> str:
    text = codon.upper()
    return text.replace("T", "U") if rna_labels else text


def _numt_chromosome_rows(
    chromosomes: Sequence[Mapping[str, Any]] | Mapping[str, int | float],
) -> list[dict[str, Any]]:
    if isinstance(chromosomes, Mapping):
        raw_rows: Iterable[Any] = [
            {"chromosome": key, "length": value} for key, value in chromosomes.items()
        ]
    else:
        raw_rows = chromosomes
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            continue
        name = _numt_chromosome_name(row) or str(i + 1)
        length = _numt_length_mb(row)
        if length <= 0:
            continue
        ce_start, ce_end = _numt_centromere_mb(row)
        rows.append(
            {
                "name": name,
                "length_mb": length,
                "centromere_start_mb": ce_start,
                "centromere_end_mb": ce_end,
            }
        )
    return rows


def _draw_ideogram_centromere(
    ax: Any,
    x: float,
    width: float,
    row: Mapping[str, Any],
    max_y: float,
) -> None:
    ce_start = row.get("centromere_start_mb")
    ce_end = row.get("centromere_end_mb")
    if ce_start is None or ce_end is None:
        return
    low = max(0.0, min(max_y, _float(ce_start)))
    high = max(0.0, min(max_y, _float(ce_end)))
    if high < low:
        low, high = high, low
    if high <= low:
        high = min(max_y, low + max_y * 0.025)
        low = max(0.0, low - max_y * 0.025)
    mid = (low + high) / 2
    left = x - width / 2
    right = x + width / 2
    neck_half = width * 0.11
    edge_color = "#5B6770"
    for points in (
        [(left - 0.01, low), (x - neck_half, mid), (left - 0.01, high)],
        [(right + 0.01, low), (x + neck_half, mid), (right + 0.01, high)],
    ):
        ax.add_patch(Polygon(points, closed=True, facecolor="white", edgecolor="none", zorder=2.5))
        xs = [
            points[0][0] + (0.01 if points[0][0] < x else -0.01),
            points[1][0],
            points[2][0] + (0.01 if points[2][0] < x else -0.01),
        ]
        ys = [points[0][1], points[1][1], points[2][1]]
        ax.plot(xs, ys, color=edge_color, lw=0.62, zorder=2.8, solid_capstyle="round")


def _numt_chromosome_name(row: Mapping[str, Any]) -> str:
    value = _case_get(
        row, "chromosome", "chrom", "chr", "nuclear_seqid", "seqid", "sequence", "id", "name"
    )
    return "" if value in (None, "") else str(value)


def _numt_fragment_row(row: Mapping[str, Any]) -> dict[str, Any]:
    start, end = _numt_interval_mb(row)
    if end < start:
        start, end = end, start
    chrom = _numt_chromosome_name(row)
    return {
        "chromosome": chrom,
        "start_mb": start,
        "end_mb": end,
        "length_mb": max(0.0, end - start),
    }


def _numt_gene_hit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "chromosome": _numt_chromosome_name(row),
        "position_mb": _numt_position_mb(row),
        "class": _numt_gene_kind(row),
    }


def _is_transfer_result(value: object) -> TypeGuard[TransferResultInput]:
    if isinstance(value, OrganelleResult):
        return value.operation_id.startswith("transfer.")
    if not isinstance(value, Mapping):
        return False
    metrics = value.get("observed_metrics")
    if isinstance(metrics, Mapping) and "candidates" in metrics:
        return True
    if "candidates" in value:
        return True
    findings = value.get("key_findings")
    return isinstance(findings, Sequence) and any(
        isinstance(row, Mapping) and row.get("metric") in {"transfer", "candidate"}
        for row in findings
    )


def _transfer_candidate_rows(
    value: TransferResultInput,
) -> list[dict[str, Any]]:
    if isinstance(value, OrganelleResult):
        metrics = cast(Mapping[str, Any], thaw_json(value.metrics))
        findings: Sequence[Any] = value.findings
    else:
        raw_metrics = value.get("observed_metrics")
        metrics = raw_metrics if isinstance(raw_metrics, Mapping) else value
        findings = value.get("key_findings", ())
    candidates = metrics.get("candidates", ())
    if not isinstance(candidates, Sequence) or isinstance(candidates, str):
        candidates = ()
    rows = [dict(candidate) for candidate in candidates if isinstance(candidate, Mapping)]
    if rows:
        return rows
    return [
        dict(finding)
        for finding in findings
        if isinstance(finding, Mapping) and finding.get("metric") in {"transfer", "candidate"}
    ]


def _transfer_candidate_gene_hit(row: Mapping[str, Any]) -> dict[str, Any]:
    hit = dict(row)
    if _case_get(hit, "class", "category", "kind", "type", "gene_class") in (None, ""):
        hit["class"] = (
            "new_gene" if _case_get(hit, "nuclear_gene", "gene", "target_gene") else "un_gene"
        )
    return hit


def _transfer_result_label(
    value: TransferResultInput,
) -> str:
    organelle = ""
    if isinstance(value, OrganelleResult):
        organelle = value.scope
        if organelle in {"none", "mixed"}:
            organelle = str(value.metrics.get("organelle", ""))
    elif isinstance(value, Mapping):
        metrics = value.get("observed_metrics")
        if isinstance(metrics, Mapping):
            organelle = str(metrics.get("organelle", ""))
        if not organelle:
            organelle = str(value.get("organelle", ""))
    if not organelle:
        candidates = _transfer_candidate_rows(value)
        if candidates:
            organelle = str(candidates[0].get("organelle_seqid", ""))
    text = organelle.lower()
    if any(token in text for token in ("plastid", "chloroplast", "chloro", "cp")):
        return "NUPT"
    return "NUMT"


def _ideogram_density_row(row: Mapping[str, Any]) -> dict[str, Any]:
    start, end = _numt_interval_mb(row)
    if end < start:
        start, end = end, start
    value = _case_get(row, "value", "density", "score", "mean", "profile")
    return {
        "chromosome": _numt_chromosome_name(row),
        "start_mb": start,
        "end_mb": end,
        "value": _float(value),
    }


def _ideogram_label_row(row: Mapping[str, Any]) -> dict[str, Any]:
    label_type = str(_case_get(row, "type", "label", "class", "name", "category") or "marker")
    label_row = {
        "chromosome": _numt_chromosome_name(row),
        "position_mb": _numt_position_mb(row),
        "type": label_type,
    }
    display_name = _case_get(row, "label_name", "display_name", "legend_label")
    if display_name not in (None, ""):
        label_row["label_name"] = str(display_name)
    return label_row


def _ideogram_label_names(
    rows: Sequence[Mapping[str, Any]],
    label_names: Mapping[str, str] | None = None,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for row in rows:
        label_type = str(row.get("type", "marker"))
        display_name = row.get("label_name")
        if display_name not in (None, "") and label_type not in resolved:
            resolved[label_type] = str(display_name)
    for label_type, display_name in (label_names or {}).items():
        resolved[str(label_type)] = str(display_name)
    return resolved


def _ideogram_profile_row(row: Mapping[str, Any]) -> dict[str, Any]:
    area = _case_get(row, "area", "area_value", "orange", "polygon", "density")
    line = _case_get(row, "line", "line_value", "blue", "profile", "score")
    return {
        "chromosome": _numt_chromosome_name(row),
        "position_mb": _numt_position_mb(row),
        "area": _float(area),
        "line": _float(line),
    }


def _draw_ideogram_marker_labels(
    ax: Any,
    base_x: float,
    rows: Sequence[Mapping[str, Any]],
    length: float,
    *,
    avoid_ranges: Sequence[tuple[float, float]] = (),
) -> None:
    if not rows:
        return
    for layout in _layout_ideogram_marker_labels(rows, length, avoid_ranges):
        row = layout["row"]
        label_type = str(row.get("type", "marker"))
        style = _ideogram_marker_style(label_type)
        observed_y = _float(layout["observed_y"])
        display_y = _float(layout["display_y"])
        leader_start_x, leader_end_x = _ideogram_marker_leader_xs(base_x)
        ax.plot(
            [leader_start_x, leader_end_x],
            [observed_y, display_y],
            color=style["color"],
            lw=0.4,
            alpha=0.6,
            zorder=4.6,
            solid_capstyle="round",
        )
        ax.scatter(
            [base_x],
            [display_y],
            marker=style["marker"],
            s=9,
            color=style["color"],
            linewidths=0,
            alpha=0.95,
            zorder=5,
        )


def _ideogram_marker_track_x(chromosome_x: float, chromosome_width: float) -> float:
    return chromosome_x + chromosome_width / 2 + _IDEOGRAM_MARKER_TRACK_GAP


def _ideogram_chromosome_width(width: float | None = None) -> float:
    if width is None:
        return 0.25
    resolved = float(width)
    if resolved <= 0:
        raise ValueError("chromosome_width must be positive")
    return resolved


def _ideogram_marker_leader_xs(base_x: float) -> tuple[float, float]:
    return base_x - _IDEOGRAM_MARKER_TRACK_GAP + 0.012, base_x - 0.012


def _layout_ideogram_marker_labels(
    rows: Sequence[Mapping[str, Any]],
    length: float,
    avoid_ranges: Sequence[tuple[float, float]] = (),
) -> list[dict[str, Any]]:
    if not rows:
        return []
    length = max(0.0, float(length))
    gap = _ideogram_marker_vertical_gap(length)
    entries: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        observed_y = max(0.0, min(length, _float(row.get("position_mb", 0.0))))
        entries.append(
            {
                "index": index,
                "row": row,
                "observed_y": observed_y,
                "display_y": observed_y,
                "leader": False,
            }
        )

    for group in _ideogram_marker_layout_groups(entries, gap):
        observed_y = sum(_float(item["observed_y"]) for item in group) / len(group)
        if len(group) > 1:
            display_values = _ideogram_marker_stack_positions(
                observed_y,
                len(group),
                length,
                gap,
            )
            for entry, display_y in zip(group, display_values, strict=False):
                entry["display_y"] = display_y
                entry["leader"] = True
        else:
            display_y = observed_y
            group[0]["display_y"] = display_y
            group[0]["leader"] = abs(display_y - observed_y) > 1e-9
    return entries


def _ideogram_marker_layout_groups(
    entries: Sequence[dict[str, Any]],
    gap: float,
) -> list[list[dict[str, Any]]]:
    type_rank = {label: i for i, label in enumerate(("rRNA", "tRNA", "miRNA"))}
    ordered = sorted(
        entries,
        key=lambda item: (
            -_float(item["observed_y"]),
            type_rank.get(str(item["row"].get("type", "marker")), 99),
            int(item["index"]),
        ),
    )
    groups: list[list[dict[str, Any]]] = []
    for entry in ordered:
        if groups and abs(_float(groups[-1][-1]["observed_y"]) - _float(entry["observed_y"])) < gap:
            groups[-1].append(entry)
        else:
            groups.append([entry])
    for group in groups:
        group.sort(
            key=lambda item: (
                -_float(item["observed_y"]),
                type_rank.get(str(item["row"].get("type", "marker")), 99),
                int(item["index"]),
            )
        )
    return groups


def _ideogram_marker_stack_positions(
    observed_y: float,
    count: int,
    length: float,
    gap: float,
) -> list[float]:
    stack_height = max(0, count - 1) * gap
    center = observed_y
    top = center + stack_height / 2
    bottom = center - stack_height / 2
    if top > length:
        top = length
        bottom = max(0.0, top - stack_height)
    elif bottom < 0:
        bottom = 0.0
        top = min(length, bottom + stack_height)
    return [max(0.0, min(length, top - i * gap)) for i in range(count)]


def _ideogram_marker_vertical_gap(length: float) -> float:
    return max(0.18, min(0.42, max(1.0, length) * 0.01))


def _ideogram_marker_types(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    present = {str(row.get("type", "marker")) for row in rows}
    return sorted(present, key=_ideogram_marker_type_sort_key)


def _ideogram_marker_type_sort_key(label: str) -> tuple[int, int, str]:
    if label.startswith("label"):
        suffix = label[5:]
        if suffix.isdigit():
            return (0, int(suffix), label)
    legacy_order = {"rRNA": 1, "tRNA": 2, "miRNA": 3}
    if label in legacy_order:
        return (1, legacy_order[label], label)
    return (2, 0, label)


# Okabe-Ito hues for the up-to-ten marker classes (shapes stay as in
# _IDEOGRAM_LABEL_STYLES); green is left to the density heatmap.
_LGA_IDEOGRAM_MARKER_COLORS = (
    "#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9",
    "#000000", "#8C6D31", "#6B7280", "#B8860B", "#7B3294",
)
_LGA_PROFILE_AREA = "#F0B37E"
_LGA_PROFILE_LINE = _PALETTE[0]


def _ideogram_marker_style(label_type: str) -> dict[str, str]:
    style = _IDEOGRAM_MARKERS.get(label_type)
    if style is None:
        return {"color": _NEUTRAL, "marker": "o"}
    index = next(i for i, known in enumerate(_IDEOGRAM_LABEL_STYLES) if known is style)
    return {"color": _LGA_IDEOGRAM_MARKER_COLORS[index], "marker": style["marker"]}


def _draw_ideogram_profile(
    ax: Any,
    base_x: float,
    width: float,
    rows: Sequence[Mapping[str, Any]],
    length: float,
) -> None:
    if not rows:
        return
    prepared = sorted(
        (
            {
                "position_mb": max(0.0, min(length, _float(row.get("position_mb", 0.0)))),
                "area": max(0.0, _float(row.get("area", 0.0))),
                "line": max(0.0, _float(row.get("line", 0.0))),
            }
            for row in rows
        ),
        key=lambda row: row["position_mb"],
    )
    if len(prepared) < 2:
        return
    area_max = max(1.0, max(row["area"] for row in prepared))
    line_max = max(1.0, max(row["line"] for row in prepared))
    y = [row["position_mb"] for row in prepared]
    area_x = [base_x + width * 0.78 * (row["area"] / area_max) for row in prepared]
    line_x = [base_x + width * (0.28 + 0.72 * (row["line"] / line_max)) for row in prepared]
    ax.fill_betweenx(
        y, [base_x] * len(y), area_x, color=_LGA_PROFILE_AREA, linewidth=0, zorder=2.3
    )
    ax.plot(line_x, y, color=_LGA_PROFILE_LINE, lw=0.8, zorder=3.2, solid_capstyle="round")


def _ideogram_density_color(value: float) -> str:
    return _blend_hex("#EDF6F2", "#00785A", max(0.0, min(1.0, value)))


def _ideogram_length_ticks(max_y: float) -> list[int]:
    if max_y <= 20:
        step = 5
    elif max_y <= 60:
        step = 10
    else:
        step = 20
    ticks = list(range(0, int(max_y) + 1, step))
    if not ticks or ticks[-1] != int(max_y):
        ticks.append(int(max_y))
    return ticks


def _blend_hex(low: str, high: str, fraction: float) -> str:
    low = low.lstrip("#")
    high = high.lstrip("#")
    channels = []
    for i in range(0, 6, 2):
        a = int(low[i : i + 2], 16)
        b = int(high[i : i + 2], 16)
        channels.append(round(a + (b - a) * fraction))
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def _numt_length_mb(row: Mapping[str, Any]) -> float:
    value = _case_get(row, "length_mb", "length_mbp", "mb", "size_mb")
    if value not in (None, ""):
        return max(0.0, _float(value))
    value = _case_get(row, "length_bp", "size_bp", "bp")
    if value not in (None, ""):
        return max(0.0, _float(value) / 1_000_000)
    value = _case_get(row, "length", "size", "end")
    numeric = _float(value)
    if numeric > 1000:
        numeric /= 1_000_000
    return max(0.0, numeric)


def _numt_centromere_mb(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    start = _case_get(
        row,
        "centromere_start_mb",
        "centromere_start_mbp",
        "cen_start_mb",
        "ce_start_mb",
        "centromere_start",
        "cen_start",
        "ce_start",
    )
    end = _case_get(
        row,
        "centromere_end_mb",
        "centromere_end_mbp",
        "cen_end_mb",
        "ce_end_mb",
        "centromere_end",
        "cen_end",
        "ce_end",
    )
    if start in (None, "") or end in (None, ""):
        return None, None
    return _numt_coord_to_mb(start), _numt_coord_to_mb(end)


def _numt_interval_mb(row: Mapping[str, Any]) -> tuple[float, float]:
    start = _case_get(row, "start_mb", "nuclear_start_mb", "position_mb")
    end = _case_get(row, "end_mb", "nuclear_end_mb")
    if start not in (None, ""):
        s = _float(start)
        e = _float(end) if end not in (None, "") else s
        return s, e
    start = _case_get(row, "nuclear_start", "start", "sstart", "position", "pos")
    end = _case_get(row, "nuclear_end", "end", "send")
    s = _numt_coord_to_mb(start)
    e = _numt_coord_to_mb(end) if end not in (None, "") else s
    return s, e


def _numt_position_mb(row: Mapping[str, Any]) -> float:
    value = _case_get(row, "position_mb", "pos_mb")
    if value not in (None, ""):
        return _float(value)
    start, end = _numt_interval_mb(row)
    return (start + end) / 2


def _numt_coord_to_mb(value: Any) -> float:
    numeric = _float(value)
    if abs(numeric) > 1000:
        numeric /= 1_000_000
    return numeric


def _numt_gene_kind(row: Mapping[str, Any]) -> str:
    value = _case_get(row, "class", "category", "kind", "type", "gene_class")
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    text = text.replace(">", "_").replace("numt_", "")
    if "new" in text:
        return "new_gene"
    return "un_gene"


def _text_color_for_bar(color: str) -> str:
    color = color.lstrip("#")
    if len(color) != 6:
        return "#111827"
    red = int(color[0:2], 16) / 255
    green = int(color[2:4], 16) / 255
    blue = int(color[4:6], 16) / 255
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#111827" if luminance > 0.58 else "white"


def _ordered_entities(events: Sequence[Mapping[str, Any]]) -> list[str]:
    entities: list[str] = []
    for event in events:
        for key in ("donor", "recipient"):
            value = str(event.get(key, key))
            if value not in entities:
                entities.append(value)
    return entities


def _entity_maxima(
    events: Sequence[Mapping[str, Any]], entities: Sequence[str]
) -> dict[str, float]:
    maxima = {entity: 1.0 for entity in entities}
    for event in events:
        donor = str(event.get("donor", "donor"))
        recipient = str(event.get("recipient", "recipient"))
        maxima[donor] = max(maxima[donor], _float(event.get("donor_end", 1.0)))
        maxima[recipient] = max(maxima[recipient], _float(event.get("recipient_end", 1.0)))
    return maxima


def _editing_site_row(site: Mapping[str, Any], index: int) -> dict[str, Any]:
    position = _float(site.get("position", site.get("pos", site.get("site", index + 1))))
    ref = str(site.get("ref", site.get("reference", ""))).upper().replace("T", "U")
    alt = str(site.get("alt", site.get("edited", site.get("to", "")))).upper().replace("T", "U")
    edit_type = str(site.get("edit_type", site.get("type", "")))
    if not edit_type or edit_type in {"candidate", "site", "stop_gain"}:
        if ref and alt:
            edit_type = f"{ref}-to-{alt}"
        elif edit_type == "stop_gain":
            edit_type = "stop_gain"
        else:
            edit_type = "unknown"
    edit_type = edit_type.replace("->", "-to-").replace("2", "-to-")
    return {
        "gene": str(site.get("gene", site.get("locus", "intergenic"))),
        "position": max(0.0, position),
        "edit_type": edit_type,
        "confidence": max(0.0, min(1.0, _float(site.get("confidence", site.get("score", 0.0))))),
        "coverage": max(
            0.0, _float(site.get("coverage", site.get("depth", site.get("read_depth", 0.0))))
        ),
    }


def _count_by(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _validate_erc_method(method: str) -> str:
    normalized = str(method).lower()
    if normalized not in _ERC_METHODS:
        allowed = ", ".join(_ERC_METHODS)
        raise ValueError(f"method must be one of: {allowed}")
    return normalized


def _erc_rows_for_method(
    pairs: Sequence[Mapping[str, Any]],
    method: str,
) -> list[dict[str, Any]]:
    method = _validate_erc_method(method)
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        row_method = str(pair.get("method", method)).lower()
        if row_method != method:
            continue
        gene_a = str(pair.get("gene_a", pair.get("source", "")))
        gene_b = str(pair.get("gene_b", pair.get("target", "")))
        if not gene_a or not gene_b:
            continue
        r_value = _float(pair.get("r", pair.get("correlation", pair.get("value", 0.0))))
        p_raw = pair.get("p_value", pair.get("p", None))
        q_raw = pair.get("q_value", pair.get("q", pair.get("fdr", None)))
        rows.append(
            {
                "gene_a": gene_a,
                "gene_b": gene_b,
                "group_a": str(pair.get("group_a", pair.get("source_group", "unassigned"))),
                "group_b": str(pair.get("group_b", pair.get("target_group", "unassigned"))),
                "method": method,
                "r": r_value,
                "abs_r": abs(r_value),
                "p_value": _float(p_raw) if p_raw is not None else None,
                "q_value": _float(q_raw) if q_raw is not None else None,
            }
        )
    if not rows:
        raise ValueError(f"pairs must contain at least one {method} ERC row")
    return rows


def _erc_pair_group_label(row: Mapping[str, Any]) -> str:
    group_a = str(row.get("group_a", "unassigned"))
    group_b = str(row.get("group_b", "unassigned"))
    if group_a == group_b:
        return group_a
    return f"{group_a} vs {group_b}"


def _correlation(x: Sequence[float], y: Sequence[float], method: str) -> float:
    if len(x) != len(y):
        raise ValueError("x and y values must have the same length")
    if method == "spearman":
        return _pearson(_ranks(x), _ranks(y))
    return _pearson(x, y)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        raise ValueError("x and y values must have the same length")
    if len(x) < 2:
        return 0.0
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    dx = [value - x_mean for value in x]
    dy = [value - y_mean for value in y]
    denom = sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denom == 0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy, strict=False)) / denom


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0 for _ in values]
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank = (i + 1 + j) / 2
        for _, index in ordered[i:j]:
            ranks[index] = rank
        i = j
    return ranks


def _linear_fit(x: Sequence[float], y: Sequence[float]) -> tuple[float, float] | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denom = sum((value - x_mean) ** 2 for value in x)
    if denom == 0:
        return None
    slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y, strict=False)) / denom
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _hist_density(
    values: Sequence[float],
    *,
    bins: int,
    low: float,
    high: float,
) -> tuple[list[float], list[float]]:
    if not values:
        raise ValueError("values must contain at least one number")
    width = (high - low) / bins
    counts = [0 for _ in range(bins)]
    for raw in values:
        value = max(low, min(high, _float(raw)))
        index = bins - 1 if value == high else int((value - low) / width)
        counts[index] += 1
    centers = [low + (i + 0.5) * width for i in range(bins)]
    density = [count / (len(values) * width) for count in counts]
    return centers, density


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    midpoint = n // 2
    if n % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _neg_log10(value: Any) -> float:
    numeric = _float(value)
    if numeric <= 0:
        return 300.0
    return -log10(numeric)


def _short_label(label: str, max_len: int) -> str:
    if len(label) <= max_len:
        return label
    return label[: max(1, max_len - 1)] + "."


def _short_group_label(label: str) -> str:
    text = label.replace("non-mt-nuProtein", "non-mt")
    text = text.replace(" vs ", "\nvs ")
    return _short_label(text, 22)


def _compact_gene_label(label: str) -> str:
    for prefix, compact in (("mtOXPHOS_", "mt"), ("nuOXPHOS_", "nu"), ("gene_", "g")):
        if label.startswith(prefix):
            suffix = label.removeprefix(prefix).lstrip("0") or "0"
            return compact + suffix
    return _short_label(label, 13)


# mVISTA-style identity-plot palette: coding dark blue, UTR light blue,
# introns pale, non-coding genes deep pink, intergenic conserved pink.
_IDENTITY_CATEGORY_COLORS = {
    "cds": "#1D4E89",
    "utr": "#5BA4CF",
    "intron": "#C3D3E8",
    "nc_gene": "#D8678C",
    "noncoding": "#EE9FB8",
}
_IDENTITY_CURVE = "#333333"
_IDENTITY_SHADE_PRIORITY = ("cds", "utr", "intron", "nc_gene")


def _identity_row(row: Mapping[str, str | int | float | bool | None]) -> dict[str, Any]:
    try:
        sample = str(row["sample"])
        start = _float(row["start"])
        end = _float(row["end"])
        identity = _float(row["identity"])
    except KeyError as exc:
        raise ValueError(f"identity window row is missing {exc.args[0]!r}") from exc
    if end <= start:
        raise ValueError("identity window end must be greater than start")
    return {"sample": sample, "start": start, "end": end, "identity": identity}


def _identity_feature_row(
    row: Mapping[str, str | int | float | bool | None],
) -> dict[str, Any]:
    try:
        key = str(row["key"])
        name = str(row.get("name") or key)
        start = _float(row["start"])
        end = _float(row["end"])
    except KeyError as exc:
        raise ValueError(f"identity feature row is missing {exc.args[0]!r}") from exc
    strand = -1 if _float(row.get("strand", 1)) < 0 else 1
    category = str(row.get("category") or "noncoding")
    if category not in _IDENTITY_CATEGORY_COLORS:
        category = "noncoding"
    if end <= start:
        raise ValueError("identity feature end must be greater than start")
    return {
        "key": key,
        "name": name,
        "start": start,
        "end": end,
        "strand": strand,
        "category": category,
    }


def _window_category(
    start: float,
    end: float,
    intervals_by_category: Mapping[str, Sequence[tuple[float, float]]],
) -> str:
    """Return the highest-priority feature category overlapping a window."""
    for category in _IDENTITY_SHADE_PRIORITY:
        for part_start, part_end in intervals_by_category.get(category, ()):
            if part_start >= end:
                break
            if part_end > start:
                return category
    return "noncoding"


def plot_genome_identity(
    windows: Sequence[Mapping[str, str | int | float | bool | None]],
    features: Sequence[Mapping[str, str | int | float | bool | None]] = (),
    *,
    reference_length: int | float | None = None,
    region: Sequence[int | float] | None = None,
    min_identity: float = 50.0,
    conserved_threshold: float = 70.0,
    reference_name: str | None = None,
    title: str | None = None,
) -> OrganellePlot:
    """Prepare an mVISTA-style identity plot; render it with ``ov.write``.

    ``windows`` are the per-window rows from
    :func:`organelleverse.comparative.identity.compute_genome_identity`
    (``sample``/``start``/``end``/``identity`` at minimum). ``features`` are
    its flat ``features`` rows (one per location part); they draw the gene
    arrow track and color conserved windows by category (coding dark blue,
    UTR light blue, intron pale, non-coding gene deep pink, other conserved
    intergenic pink). One curve track is drawn per sample against reference
    coordinates, with the mVISTA defaults: y-axis floor ``min_identity`` =
    50%, conserved cutoff ``conserved_threshold`` = 70%. ``region``
    (``(start, end)``) zooms the x-axis to a reference interval.
    """
    rows = [_identity_row(row) for row in windows]
    if not rows:
        raise ValueError("plot_genome_identity requires at least one window row")
    if not 0.0 <= min_identity < 100.0:
        raise ValueError("min_identity must be in [0, 100)")
    if not min_identity < conserved_threshold <= 100.0:
        raise ValueError("conserved_threshold must be in (min_identity, 100]")
    feature_rows = [_identity_feature_row(row) for row in features]
    if reference_length is None:
        reference_length = max(
            [row["end"] for row in rows] + [row["end"] for row in feature_rows]
        )
    if region is not None:
        if len(region) != 2 or not 0 <= region[0] < region[1] <= reference_length:
            raise ValueError("region must satisfy 0 <= start < end <= reference_length")
        if not any(row["end"] > region[0] and row["start"] < region[1] for row in rows):
            raise ValueError("region must overlap at least one window")

    def _render(output: str | Path) -> Path:
        return _render_genome_identity(
            rows,
            feature_rows,
            output,
            reference_length=float(reference_length),
            region=region,
            min_identity=min_identity,
            conserved_threshold=conserved_threshold,
            reference_name=reference_name,
            title=title,
        )

    return plot_result(
        "plot_genome_identity",
        _render,
        metrics={
            "samples": len({row["sample"] for row in rows}),
            "windows": len(rows),
            "features": len(feature_rows),
            "reference_length": reference_length,
            "region": list(region) if region is not None else None,
        },
        summary="mVISTA-style genome identity plot prepared.",
    )


_LGA_IDENTITY_LABELS = {
    "cds": "Coding",
    "utr": "UTR",
    "intron": "Intron",
    "nc_gene": "Non-coding gene",
    "noncoding": "Conserved non-coding",
}
_LGA_GENE_LABEL_PT = FONT_PT["label"]


def _render_genome_identity(
    windows: list[dict[str, Any]],
    features: list[dict[str, Any]],
    output: str | Path,
    *,
    reference_length: float,
    region: Sequence[int | float] | None,
    min_identity: float,
    conserved_threshold: float,
    reference_name: str | None,
    title: str | None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_genome_identity`."""
    out = _prepare_output(output)
    _apply_figure_style()

    region_start, region_end = (0.0, float(reference_length))
    if region is not None:
        if len(region) != 2:
            raise ValueError("region must be a (start, end) pair")
        region_start, region_end = _float(region[0]), _float(region[1])
        if region_end <= region_start:
            raise ValueError("region end must be greater than region start")
    region_windows = [
        row for row in windows if row["end"] > region_start and row["start"] < region_end
    ]
    region_features = [
        row for row in features if row["end"] > region_start and row["start"] < region_end
    ]
    samples = list(dict.fromkeys(row["sample"] for row in region_windows))
    intervals_by_category: dict[str, list[tuple[float, float]]] = {}
    for row in region_features:
        intervals_by_category.setdefault(row["category"], []).append((row["start"], row["end"]))
    for intervals in intervals_by_category.values():
        intervals.sort()
    feature_edges = sorted({value for intervals in intervals_by_category.values()
                            for interval in intervals for value in interval})

    show_features = bool(region_features)
    panel_count = len(samples) + (1 if show_features else 0)
    gene_height = 0.85 if show_features else 0.0
    fig_height = min(6.6, 0.5 * len(samples) + gene_height + 0.55)
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=journal_figsize(7.2, fig_height),
        sharex=True,
        gridspec_kw={
            "height_ratios": ([2.1] if show_features else []) + [1.0] * len(samples)
        },
    )
    axes_list = list(axes) if isinstance(axes, Iterable) and not hasattr(axes, "plot") else [axes]
    track_axes = axes_list[1:] if show_features else axes_list
    present: set[str] = set()
    if show_features:
        present.update(
            _render_identity_gene_track(axes_list[0], region_features, region_start, region_end)
        )

    for ax, sample in zip(track_axes, samples, strict=True):
        sample_rows = sorted((row for row in region_windows if row["sample"] == sample),
                             key=lambda row: (row["start"] + row["end"]) / 2)
        centers = [(row["start"] + row["end"]) / 2 for row in sample_rows]
        # Each value is displayed in its center's cell. Overlapping sliding
        # windows must never make the curve double back along the x axis.
        edges = [sample_rows[0]["start"]]
        edges.extend((a + b) / 2 for a, b in itertools.pairwise(centers))
        edges.append(sample_rows[-1]["end"])
        x_steps: list[float] = []
        y_steps: list[float] = []
        polygons, colors = [], []
        for row, left, right in zip(sample_rows, edges[:-1], edges[1:], strict=True):
            identity = max(min_identity, min(100.0, row["identity"]))
            x_steps.extend((left, right))
            y_steps.extend((identity, identity))
            if row["identity"] < conserved_threshold:
                continue
            # Split at actual annotation boundaries; a single coding base
            # does not turn an entire intergenic window blue.
            cuts = [left, *feature_edges[bisect_right(feature_edges, left):
                                         bisect_left(feature_edges, right)], right]
            for start, end in itertools.pairwise(cuts):
                category = _window_category(start, end, intervals_by_category)
                polygons.append([(start, min_identity), (start, identity),
                                 (end, identity), (end, min_identity)])
                colors.append(_IDENTITY_CATEGORY_COLORS[category])
                present.add(category)
        ax.add_collection(PolyCollection(polygons, facecolors=colors, alpha=0.85, linewidths=0))
        ax.plot(x_steps, y_steps, color=_IDENTITY_CURVE, lw=0.45, solid_joinstyle="miter")
        ax.axhline(conserved_threshold, color="#8B949E", lw=0.5, ls=(0, (3, 2)), zorder=3)
        ax.set_ylim(min_identity, 100.0)
        ax.set_yticks([min_identity, conserved_threshold, 100.0])
        ax.set_yticklabels(
            [f"{min_identity:.0f}", f"{conserved_threshold:.0f}", "100"],
            fontsize=FONT_PT["tick"],
        )
        ax.set_xlim(region_start, region_end)
        # sample name as a small left-aligned header: no extra right margin
        ax.set_title(sample, loc="left", fontsize=FONT_PT["label"], pad=1.5, color="#1F2933")
        _despine(ax)
    track_axes[len(track_axes) // 2].set_ylabel("Identity (%)")
    axis_label = f"Position on {reference_name}" if reference_name else "Reference position"
    _lga_genome_axis(track_axes[-1], region_end, axis_label)
    track_axes[-1].set_xlim(region_start, region_end)
    fig.align_ylabels(axes_list)
    _figure_title(fig, title)
    order = [category for category in ("cds", "utr", "intron", "nc_gene", "noncoding")
             if category in present]
    if order:
        handles = [
            Rectangle((0, 0), 1, 1, color=_IDENTITY_CATEGORY_COLORS[category], lw=0, alpha=0.85)
            for category in order
        ]
        axes_list[0].legend(
            handles,
            [_LGA_IDENTITY_LABELS[category] for category in order],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02 if show_features else 1.12),
            ncol=len(order),
            frameon=False,
            handlelength=1.0,
            handleheight=0.8,
            handletextpad=0.4,
            columnspacing=1.0,
            borderaxespad=0.0,
            fontsize=FONT_PT["label"],
        )
    return _save(fig, out)


def _render_identity_gene_track(
    ax: Any,
    features: list[dict[str, Any]],
    region_start: float,
    region_end: float,
) -> set[str]:
    """Draw the mVISTA gene-arrow track: two lanes by strand, labeled arrows.

    A gene is labeled only where its name fits without touching a neighbour
    (two label rows per strand, longest genes first); returns the categories
    drawn so the legend lists only those.
    """
    from .style import text_width_pt

    groups: dict[str, dict[str, Any]] = {}
    for row in features:
        group = groups.setdefault(
            row["key"],
            {
                "name": row["name"],
                "strand": row["strand"],
                "category": row["category"],
                "parts": [],
            },
        )
        group["parts"].append((row["start"], row["end"]))
    span = max(1.0, region_end - region_start)
    fig = ax.figure
    axes_width_pt = max(1.0, 0.92 * ax.get_position().width * fig.get_figwidth() * 72)
    bp_per_pt = span / axes_width_pt
    head_cap = span * 0.006
    drawn: set[str] = set()
    for group in groups.values():
        parts = sorted(group["parts"])
        start = parts[0][0]
        end = parts[-1][1]
        strand = group["strand"]
        drawn.add(group["category"])
        color = _IDENTITY_CATEGORY_COLORS.get(group["category"], _IDENTITY_CATEGORY_COLORS["noncoding"])
        y = 1.0 if strand == 1 else 0.0
        # thin line across introns, thick boxes on the exons/parts
        ax.add_patch(Rectangle((start, y - 0.04), end - start, 0.08, color=color, lw=0))
        for part_start, part_end in parts:
            ax.add_patch(
                Rectangle((part_start, y - 0.16), part_end - part_start, 0.32, color=color, lw=0)
            )
        head = min(head_cap, (end - start) * 0.35)
        if head > 0:
            if strand == 1:
                tip, base = end, end - head
            else:
                tip, base = start, start + head
            ax.add_patch(
                Polygon(
                    [(tip, y), (base, y + 0.24), (base, y - 0.24)],
                    closed=True,
                    color=color,
                    lw=0,
                )
            )
        group["span"] = (start, end)
    # Label placement: longest genes first, two rows per strand, measured widths.
    label_rows = {1: ([], []), -1: ([], [])}
    label_y = {1: (1.58, 2.3), -1: (-0.58, -1.3)}
    gap = 2.0 * bp_per_pt
    for group in sorted(groups.values(), key=lambda g: g["span"][0] - g["span"][1]):
        start, end = group["span"]
        label = str(group["name"])
        width = text_width_pt(label, _LGA_GENE_LABEL_PT) * bp_per_pt
        center = (max(start, region_start) + min(end, region_end)) / 2
        center = min(max(center, region_start + width / 2), region_end - width / 2)
        left, right = center - width / 2 - gap, center + width / 2 + gap
        for row_index, placed in enumerate(label_rows[group["strand"]]):
            if all(right <= a or left >= b for a, b in placed):
                placed.append((left, right))
                ax.text(
                    center,
                    label_y[group["strand"]][row_index],
                    label,
                    ha="center",
                    va="center",
                    fontsize=_LGA_GENE_LABEL_PT,
                    fontstyle="italic",
                    color="#374151",
                )
                break
    ax.axhline(1.0, color="#D1D5DB", lw=0.5, zorder=0)
    ax.axhline(0.0, color="#D1D5DB", lw=0.5, zorder=0)
    ax.set_ylim(-1.65, 2.65)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["−", "+"], fontsize=FONT_PT["tick"])
    ax.set_ylabel("Genes")
    ax.set_xlim(region_start, region_end)
    _despine(ax)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    return drawn


__all__ = [
    "ideogram",
    "nuclear_transfer_ideogram",
    "plot_erc_distribution",
    "plot_erc_group_ridges",
    "plot_erc_network",
    "plot_erc_pair_scatter",
    "plot_erc_significance",
    "plot_genome_identity",
    "plot_localization_bars",
    "plot_matrix_heatmap",
    "plot_network",
    "plot_qc_dashboard",
    "plot_rna_editing_summary",
    "plot_rscu_usage",
    "plot_selection_summary",
    "plot_splicing_schematic",
    "plot_synteny_matrix",
    "plot_track_density",
    "plot_transfer_schematic",
    "save_plot",
    "summarize_synteny_matrix",
    "write_erc_visualization_report",
    "write_ideogram",
    "write_nuclear_transfer_ideogram",
    "write_visualization_preview_report",
]
