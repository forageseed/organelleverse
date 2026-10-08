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
    ConnectionPatch,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

from ..core.frozen import thaw_json
from ..core.result import OrganelleResult
from ..selection.kaks import _CODONS as _STANDARD_CODONS
from .plot_object import OrganellePlot, data_plot_result, plot_result

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
    counts, coverage breadth, zero-coverage bases, unsupported/contradicted
    junction counts, and flags. It does not assign or display a 0-100 score, a
    letter grade, or a GC-contamination claim; no universal pass threshold is
    applied. Render with ``ov.write(plot, output)``.
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


def _render_qc_dashboard(
    metrics: Mapping[str, Any],
    output: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_qc_dashboard`."""
    out = _prepare_output(output)
    _apply_figure_style()

    decision = str(metrics.get("qc_decision", "insufficient_evidence"))
    decision_color = {
        "ready": _PALETTE[2],
        "needs_review": _PALETTE[1],
        "not_ready": _ALARM,
    }.get(decision, _PALETTE[4])
    pass_count = int(_float(metrics.get("qc_pass_count", 0)))
    warning_count = int(_float(metrics.get("qc_warning_count", 0)))
    failure_count = int(_float(metrics.get("qc_failure_count", 0)))
    not_assessed_count = int(_float(metrics.get("qc_not_assessed_count", 0)))
    coverage_breadth = metrics.get("coverage_breadth")
    zero_coverage_bases = metrics.get("zero_coverage_bases")
    unsupported_junction_count = metrics.get("unsupported_junction_count")
    contradicted_junction_count = metrics.get("contradicted_junction_count")
    flags = [str(value) for value in metrics.get("flags", [])]

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.4))
    fig.suptitle(title or "Assembly QC evidence", x=0.02, ha="left", fontsize=10)

    ax = axes[0, 0]
    ax.axis("off")
    ax.set_title("Decision", loc="left")
    ax.text(
        0.0,
        0.5,
        decision,
        fontsize=15,
        fontweight="bold",
        color=decision_color,
        transform=ax.transAxes,
    )

    ax = axes[0, 1]
    ax.bar(
        ["pass", "warn", "fail", "n/a"],
        [pass_count, warning_count, failure_count, not_assessed_count],
        color=[_PALETTE[2], _PALETTE[1], _ALARM, _PALETTE[4]],
    )
    ax.set_title("Checks", loc="left")
    ax.set_ylabel("count")
    _despine(ax)

    ax = axes[1, 0]
    ax.set_title("Coverage breadth", loc="left")
    if coverage_breadth is None:
        ax.text(0.5, 0.5, "not assessed", ha="center", transform=ax.transAxes)
        ax.set_yticks([])
        ax.set_xticks([])
    else:
        breadth = max(0.0, min(1.0, _float(coverage_breadth)))
        ax.barh([0], [1.0], color="#F3F4F6", height=0.4)
        ax.barh([0], [breadth], color=_PALETTE[0], height=0.4)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1.0], ["0", "50%", "100%"])
        _despine(ax)
    zero_text = (
        "n/a" if zero_coverage_bases is None else _format_bp(int(_float(zero_coverage_bases)))
    )
    ax.set_xlabel(f"zero-coverage bases: {zero_text}")

    ax = axes[1, 1]
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
    lines = [
        ("Unsupported junctions", unsupported_text),
        ("Contradicted junctions", contradicted_text),
        ("Flags", ", ".join(flags) if flags else "none"),
    ]
    y = 0.78
    for label, value in lines:
        ax.text(0.0, y, label, fontweight="bold", transform=ax.transAxes)
        ax.text(0.46, y, value, transform=ax.transAxes)
        y -= 0.28

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
    fig, axes = plt.subplots(n, 1, figsize=(8.8, max(2.2, 1.65 * n)), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(title or "Windowed organelle tracks", x=0.02, ha="left", fontsize=10)

    for i, (ax, track) in enumerate(zip(axes, tracks, strict=False)):
        x = _float_list(track.get("x", []), "track x")
        y = _float_list(track.get("y", []), "track y")
        if len(x) != len(y):
            raise ValueError("track x and y must have the same length")
        color = str(track.get("color") or _PALETTE[i % len(_PALETTE)])
        label = str(track.get("label", f"track {i + 1}"))
        ax.plot(x, y, marker="o", ms=3, lw=1.5, color=color, label=label)
        for region in regions or ():
            start = _float(region.get("start", 0.0))
            end = _float(region.get("end", start))
            ax.axvspan(start, end, color="#D1D5DB", alpha=0.25, lw=0)
            if i == 0 and region.get("label"):
                ax.text(
                    (start + end) / 2,
                    0.98,
                    str(region["label"]),
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=7,
                    color=_NEUTRAL,
                )
        ax.set_ylabel(str(track.get("ylabel", label)))
        ax.legend(loc="upper right", frameon=False)
        _despine(ax)
    axes[-1].set_xlabel("Position (bp)")
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
) -> OrganellePlot:
    """Render a small matrix heatmap with optional cell values (compute-only)."""
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
        )

    return plot_result(
        "plot_matrix_heatmap",
        _render,
        metrics={"rows": n_rows, "cols": n_cols},
        summary="Matrix heatmap prepared.",
    )


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
) -> Path:
    """Private path-taking renderer backing :func:`plot_matrix_heatmap`."""
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
        chosen_cmap = cmap or "viridis"
    else:
        vmin = min(min(flat), center)
        vmax = max(max(flat), center)
        if vmin >= center:
            vmin = center - max(vmax - center, 1.0)
        if vmax <= center:
            vmax = center + max(center - vmin, 1.0)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        chosen_cmap = cmap or "coolwarm"

    fig_w = max(4.6, min(10.0, 0.52 * n_cols + 2.6))
    fig_h = max(3.8, min(9.0, 0.46 * n_rows + 2.2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, cmap=chosen_cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(n_cols), cols, rotation=45, ha="right")
    ax.set_yticks(range(n_rows), rows)
    ax.set_title(title or "Matrix heatmap", loc="left")
    if annotate and n_rows * n_cols <= 200:
        for r, row in enumerate(values):
            for c, value in enumerate(row):
                ax.text(
                    c,
                    r,
                    _format_number(value),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=_text_color_for_value(norm(value)),
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, out)


def plot_synteny_matrix(
    blocks: Sequence[Mapping[str, str | int | float | None]] | Mapping[str, Sequence[Mapping[str, str | int | float | None]] | Mapping[str, Sequence[Mapping[str, str | int | float | None]]]],
    *,
    labels: Sequence[str] | None = None,
    groups: Mapping[str, str] | None = None,
    genome_lengths: Mapping[str, int | float] | None = None,
    title: str | None = None,
    direct_color: str = _SYNTENY_DIRECT,
    inverted_color: str = _SYNTENY_INVERTED,
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

    return plot_result(
        "plot_synteny_matrix",
        _render,
        metrics={"genomes": len(data["labels"]), "blocks": len(data["rows"])},
        summary="Synteny matrix prepared.",
    )


def _render_synteny_matrix(
    blocks: Sequence[Mapping[str, str | int | float | None]] | Mapping[str, Sequence[Mapping[str, str | int | float | None]] | Mapping[str, Sequence[Mapping[str, str | int | float | None]]]],
    output: str | Path,
    *,
    labels: Sequence[str] | None = None,
    groups: Mapping[str, str] | None = None,
    genome_lengths: Mapping[str, int | float] | None = None,
    title: str | None = None,
    direct_color: str = _SYNTENY_DIRECT,
    inverted_color: str = _SYNTENY_INVERTED,
) -> Path:
    """Render a lower-triangle whole-genome pairwise synteny dotplot matrix.

    Each block row should name two genomes (``genome_a``/``genome_b`` or
    ``query``/``target`` aliases), coordinates on each genome, and optionally
    ``orientation``/``strand`` as direct or inverted.
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
    fig_size = max(4.6, min(13.5, 0.66 * n + 2.2))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    for i in range(1, n):
        y = n - 1 - i
        for j in range(i):
            ax.add_patch(Rectangle((j, y), 1, 1, fill=False, edgecolor="#737373", linewidth=0.6))

    direct_blocks = inverted_blocks = 0
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
        if orientation == "inverted":
            inverted_blocks += 1
            color = inverted_color
        else:
            direct_blocks += 1
            color = direct_color
        ax.plot([x0, x1], [y0, y1], color=color, lw=1.2, alpha=0.95, solid_capstyle="round")

    group_colors = _synteny_group_palette(names, groups)
    for i, name in enumerate(names):
        y = (n - 1) if i == 0 else (n - i)
        ax.text(
            i + 0.04,
            y + 0.03,
            name,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=group_colors[name],
        )

    ax.set_xlim(-0.02, n + 1.05)
    ax.set_ylim(-0.02, n + 0.28)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    if title:
        ax.set_title(title, loc="left", pad=10)

    orientation_handles = [
        Line2D([0], [0], color=direct_color, lw=1.6, label="Direct"),
        Line2D([0], [0], color=inverted_color, lw=1.6, label="Inverted"),
    ]
    group_labels = []
    for group in _ordered_groups(names, groups):
        color = _SYNTENY_GROUP_COLORS.get(
            group,
            group_colors[next(name for name in names if (groups or {}).get(name, "") == group)],
        )
        group_labels.append(Line2D([0], [0], color=color, lw=3.0, label=group))
    if group_labels:
        legend = ax.legend(
            handles=group_labels + orientation_handles,
            loc="center left",
            bbox_to_anchor=(0.76, 0.64),
            frameon=False,
        )
        ax.add_artist(legend)
    else:
        ax.legend(
            handles=orientation_handles,
            loc="center left",
            bbox_to_anchor=(0.76, 0.64),
            frameon=False,
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

    return plot_result(
        "plot_network",
        _render,
        metrics={"nodes": len(node_ids), "edges": len(edges)},
        summary="Network plot prepared.",
    )


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
    out = _prepare_output(output)
    _apply_figure_style()

    groups = {
        node_id: str(node_groups.get(node_id, "")) if node_groups else str(node.get("group", ""))
        for node_id, node in zip(node_ids, nodes, strict=False)
    }
    positions = _network_positions(node_ids, groups)
    group_palette = _group_palette(groups.values())

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.set_title(title or "Network", loc="left")
    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        weight = abs(_float(edge.get("weight", 1.0)))
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#9CA3AF",
            lw=max(0.6, min(4.0, 0.8 + 2.5 * weight)),
            alpha=0.55,
            zorder=1,
        )
    for node, node_id in zip(nodes, node_ids, strict=False):
        x, y = positions[node_id]
        group = groups[node_id]
        size = 330 * max(0.7, _float(node.get("size", 1.0)))
        ax.scatter(
            [x], [y], s=size, color=group_palette[group], edgecolor="white", linewidth=1.0, zorder=2
        )
        if y > 0.74:
            label_y, va = y + 0.11, "bottom"
        elif y < -0.74:
            label_y, va = y - 0.12, "top"
        else:
            label_y, va = y + 0.09, "bottom"
        ax.text(
            x, label_y, str(node.get("label", node_id)), ha="center", va=va, fontsize=8, zorder=3
        )
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.28, 1.45)
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

    return plot_result(
        "plot_selection_summary",
        _render,
        metrics={"records": len(records)},
        summary="Selection-pressure summary prepared.",
    )


def _render_selection_summary(
    records: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_selection_summary`."""
    out = _prepare_output(output)
    _apply_figure_style()

    genes = [str(r.get("gene", f"gene {i + 1}")) for i, r in enumerate(records)]
    omega = [_float(r.get("omega", r.get("kaks", 0.0))) for r in records]
    pvals = [_float(r.get("p_value", r.get("p", 1.0))) for r in records]
    sites = [_float(r.get("selected_sites", 0.0)) for r in records]
    colors = [_ALARM if p < 0.05 else _PALETTE[0] for p in pvals]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = list(range(len(records)))
    ax.bar(x, omega, color=colors, width=0.66)
    ax.axhline(1.0, color=_NEUTRAL, lw=1.0, ls="--")
    ax.text(
        len(records) - 0.35, 1.02, "omega = 1", ha="right", va="bottom", fontsize=7, color=_NEUTRAL
    )
    for xi, value, n_sites in zip(x, omega, sites, strict=False):
        if n_sites:
            ax.text(
                xi,
                value + max(omega) * 0.04,
                f"{int(n_sites)} sites",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x, genes, rotation=35, ha="right")
    ax.set_ylabel("omega / KaKs")
    ax.set_title(title or "Selection pressure summary", loc="left")
    ax.set_ylim(0, max(max(omega) * 1.25, 1.25))
    _despine(ax)
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

    return plot_result(
        "plot_transfer_schematic",
        _render,
        metrics={"events": len(events)},
        summary="Transfer schematic prepared.",
    )


def _render_transfer_schematic(
    events: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_transfer_schematic`."""
    out = _prepare_output(output)
    _apply_figure_style()

    entities = _ordered_entities(events)
    maxima = _entity_maxima(events, entities)
    y_pos = {entity: i for i, entity in enumerate(reversed(entities))}
    fig, ax = plt.subplots(figsize=(9.0, max(3.6, 0.8 * len(entities) + 2.2)))
    ax.set_title(title or "DNA transfer schematic", loc="left")

    for entity in entities:
        y = y_pos[entity]
        ax.plot([0, 1], [y, y], color="#CBD5E1", lw=8, solid_capstyle="round")
        ax.text(-0.03, y, entity, ha="right", va="center", fontweight="bold")
    for i, event in enumerate(events):
        donor = str(event.get("donor", "donor"))
        recipient = str(event.get("recipient", "recipient"))
        dy = y_pos[donor]
        ry = y_pos[recipient]
        dx = _scaled_mid(event, "donor", maxima[donor])
        rx = _scaled_mid(event, "recipient", maxima[recipient])
        color = _PALETTE[i % len(_PALETTE)]
        ax.scatter([dx, rx], [dy, ry], color=color, s=42, zorder=3)
        con = ConnectionPatch(
            (dx, dy),
            (rx, ry),
            "data",
            "data",
            arrowstyle="-|>",
            shrinkA=4,
            shrinkB=4,
            mutation_scale=12,
            lw=1.4,
            color=color,
            alpha=0.78,
            axesA=ax,
            axesB=ax,
            connectionstyle="arc3,rad=0.22",
        )
        ax.add_artist(con)
        label = str(event.get("label", f"event {i + 1}"))
        ax.text(
            (dx + rx) / 2,
            (dy + ry) / 2 + 0.16,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
        )
    ax.set_xlim(-0.22, 1.05)
    ax.set_ylim(-0.7, len(entities) - 0.25)
    ax.set_xlabel("Scaled coordinate within each genome")
    ax.set_yticks([])
    _despine(ax)
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
) -> OrganelleResult:
    """Prepare a RIdeogram-style chromosome plot object without writing files."""
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
    )


def nuclear_transfer_ideogram(
    chromosomes: Sequence[Mapping[str, str | int | float | None]] | Mapping[str, int | float],
    fragments: (Sequence[Mapping[str, str | int | float | None]] | Mapping[str, Sequence[Mapping[str, str | int | float | None]] | Mapping[str, Sequence[Mapping[str, str | int | float | None]]]]),
    gene_hits: Sequence[Mapping[str, Any]] | None = None,
    *,
    title: str | None = None,
    panel_label: str | None = "C",
    max_mb: float | None = None,
    transfer_label: str | None = None,
) -> OrganelleResult:
    """Prepare a panel-C style NUMT/NUPT chromosome ideogram without writing files.

    Use :func:`write_nuclear_transfer_ideogram` to render the prepared result.
    """
    chrom_rows = _numt_chromosome_rows(chromosomes)
    if not chrom_rows:
        raise ValueError("chromosomes must contain at least one chromosome")
    if _is_transfer_result(fragments) and gene_hits is None:
        candidate_rows = _transfer_candidate_rows(fragments)
        fragment_rows = [_numt_fragment_row(candidate) for candidate in candidate_rows]
        hit_rows = [
            _numt_gene_hit_row(_transfer_candidate_gene_hit(candidate))
            for candidate in candidate_rows
        ]
        resolved_transfer_label = transfer_label or _transfer_result_label(fragments)
    else:
        fragment_rows = [_numt_fragment_row(fragment) for fragment in fragments]  # type: ignore[union-attr]
        hit_rows = [_numt_gene_hit_row(hit) for hit in (gene_hits or ())]
        resolved_transfer_label = transfer_label or "NUMT"
    metrics = {
        "plot_kind": "nuclear_transfer_ideogram",
        "chromosomes": chrom_rows,
        "fragments": fragment_rows,
        "gene_hits": hit_rows,
        "chromosome_count": len(chrom_rows),
        "fragment_count": len(fragment_rows),
        "gene_hit_count": len(hit_rows),
        "transfer_label": resolved_transfer_label,
        "panel_label": panel_label,
        "max_mb": max_mb,
        "title": title,
    }
    return data_plot_result(
        "nuclear_transfer_ideogram",
        organelle="nuclear",
        metrics=metrics,
        key_findings=(
            {"metric": "chromosomes", "value": len(chrom_rows)},
            {"metric": "fragments", "value": len(fragment_rows)},
            {"metric": "gene_hits", "value": len(hit_rows)},
        ),
        flags=("nuclear_transfer_ideogram_prepared",),
        summary=(
            f"{resolved_transfer_label} ideogram prepared for {len(chrom_rows)} "
            f"chromosome(s), {len(fragment_rows)} fragment(s)."
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
    chromosomes = metrics.get("chromosomes", ())
    fragments = metrics.get("fragments", ())
    gene_hits = metrics.get("gene_hits", ())
    return _render_nuclear_transfer_ideogram(
        chromosomes,
        fragments,
        gene_hits,
        output,
        title=metrics.get("title"),
        panel_label=metrics.get("panel_label", "C"),
        max_mb=metrics.get("max_mb"),
        transfer_label=str(metrics.get("transfer_label", "NUMT")),
        figure_width=width,
        figure_height=height,
    )


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

    marker_only = bool(labels) and not density and not profile
    if marker_only:
        fig_w = max(5.4, min(7.0, 0.22 * len(chrom_rows) + 1.5))
        fig_h = 4.6
    else:
        fig_w = max(7.4, min(10.8, 0.36 * len(chrom_rows) + 2.2))
        fig_h = 5.2
    fig_w, fig_h = _figure_size(fig_w, fig_h, figure_width, figure_height)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    chrom_width = _ideogram_chromosome_width(chromosome_width)
    inner_pad = chrom_width * 0.08
    profile_gap = 0.08
    profile_width = 0.28

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
                facecolor="#D7F0EA" if has_density else "white",
                edgecolor="#7A858C",
                linewidth=0.75,
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
                        alpha=0.86,
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
                edgecolor="#7A858C",
                linewidth=0.75,
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

    ax.set_xlim(0.45, len(chrom_rows) + 1.05)
    ax.set_ylim(0, max_y * 1.03)
    ax.set_xticks(list(x_by_chrom.values()), chrom_names)
    if show_lengths:
        ax.set_yticks(_ideogram_length_ticks(max_y))
        ax.set_ylabel("Mb", fontsize=8, labelpad=10)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_bounds(0, max_y)
        ax.spines["left"].set_color("#52606D")
        ax.spines["left"].set_linewidth(0.8)
        ax.tick_params(axis="y", direction="out", length=3, pad=2, labelsize=6, colors="#52606D")
    else:
        ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=2, labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    if not show_lengths:
        ax.spines["left"].set_visible(False)
    if title:
        ax.set_title(title, loc="left", pad=6)
    if density:
        _draw_ideogram_density_legend(ax)
    _draw_ideogram_marker_legend(
        ax,
        labels,
        below_density=bool(density),
        label_names=label_names,
    )
    return _save(fig, out)


def _render_nuclear_transfer_ideogram(
    chromosomes: Sequence[Mapping[str, Any]] | Mapping[str, int | float],
    fragments: Sequence[Mapping[str, Any]],
    gene_hits: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    title: str | None = None,
    panel_label: str | None = "C",
    max_mb: float | None = None,
    transfer_label: str = "NUMT",
    figure_width: float | None = None,
    figure_height: float | None = None,
) -> Path:
    chrom_rows = _numt_chromosome_rows(chromosomes)
    if not chrom_rows:
        raise ValueError("chromosomes must contain at least one chromosome")
    out = _prepare_output(output)
    _apply_figure_style()

    chrom_names = [row["name"] for row in chrom_rows]
    x_by_chrom = {name: i + 1 for i, name in enumerate(chrom_names)}
    max_y = max_mb if max_mb is not None else max(row["length_mb"] for row in chrom_rows)
    max_y = max(1.0, float(max_y))

    fig_w = max(3.6, min(4.35, 0.12 * len(chrom_rows) + 1.7))
    fig_w, fig_h = _figure_size(fig_w, 3.05, figure_width, figure_height)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    chrom_width = 0.34

    for row in chrom_rows:
        x = x_by_chrom[row["name"]]
        length = min(row["length_mb"], max_y)
        ax.add_patch(
            FancyBboxPatch(
                (x - chrom_width / 2, 0.0),
                chrom_width,
                length,
                boxstyle=f"round,pad=0,rounding_size={chrom_width / 2}",
                facecolor="white",
                edgecolor="#5B6770",
                linewidth=0.65,
                zorder=1,
            )
        )
        _draw_ideogram_centromere(ax, x, chrom_width, row, max_y)

    for fragment in fragments:
        chrom = _numt_chromosome_name(fragment)
        if chrom not in x_by_chrom:
            continue
        start, end = _numt_interval_mb(fragment)
        if end < start:
            start, end = end, start
        y = max(0.0, min(max_y, (start + end) / 2))
        x = x_by_chrom[chrom]
        ax.plot(
            [x - chrom_width * 0.36, x + chrom_width * 0.36],
            [y, y],
            color=_NUMT_FRAGMENT,
            lw=0.9,
            solid_capstyle="butt",
            zorder=3,
        )

    for hit in gene_hits:
        chrom = _numt_chromosome_name(hit)
        if chrom not in x_by_chrom:
            continue
        y = max(0.0, min(max_y, _numt_position_mb(hit)))
        kind = _numt_gene_kind(hit)
        if kind == "new_gene":
            marker = "s"
            color = _NUMT_NEW_GENE
            size = 6
            x = x_by_chrom[chrom] - chrom_width * 0.12
        else:
            marker = "o"
            color = _NUMT_UN_GENE
            size = 7
            x = x_by_chrom[chrom] + chrom_width * 0.48
        ax.scatter([x], [y], marker=marker, s=size, color=color, linewidths=0, zorder=4)

    ax.set_xlim(0.45, len(chrom_rows) + 0.55)
    ax.set_ylim(0, max_y)
    ax.set_xticks(list(x_by_chrom.values()), chrom_names)
    ax.set_yticks([tick for tick in range(0, int(max_y) + 1, 5)])
    ax.set_ylabel("MB", fontsize=7, labelpad=4)
    if title:
        ax.set_title(title, loc="left", pad=6)
    ax.spines["left"].set_bounds(0, max_y)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=1, labelsize=6)
    ax.tick_params(axis="y", direction="out", length=3, pad=1, labelsize=6)
    ax.grid(False)

    handles = [
        Line2D([0], [0], color=_NUMT_FRAGMENT, lw=1.0, label=f"{transfer_label} fragments"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=_NUMT_NEW_GENE,
            markeredgewidth=0,
            markersize=3.2,
            label=f"{transfer_label} -> new-gene",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_NUMT_UN_GENE,
            markeredgewidth=0,
            markersize=3.2,
            label=f"{transfer_label} -> un-gene",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.55, 1.02),
        frameon=False,
        borderaxespad=0,
        handlelength=1.2,
        handletextpad=0.35,
        labelspacing=0.12,
        fontsize=5.5,
    )
    if panel_label:
        ax.text(
            -0.12,
            1.02,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    return _save(fig, out)


def plot_transfer_tracks(
    candidates: Sequence[Mapping[str, str | int | float | bool | None]],
    nuclear_fasta: str | Path | None = None,
    *,
    chromosome_lengths: Mapping[str, int] | None = None,
    color_by: str = "evidence_level",
    title: str | None = None,
) -> OrganellePlot:
    """Render NUMT/NUPT insertions as dots on linear chromosome tracks (compute-only).

    Each nuclear chromosome is drawn as a horizontal bar (length = chromosome
    length in bp, x = chromosome name). Insertion candidates are plotted as
    markers at their nuclear position; marker size encodes the fragment length
    and colour encodes either ``evidence_level`` or ``organelle_gene``. Render
    with ``ov.write(plot, output)``.

    Parameters
    ----------
    candidates : sequence of mappings
        ``detect_transfers_evidence`` key_findings (each carrying
        ``nuclear_seqid``, ``nuclear_start``, ``nuclear_end``, ``length``,
        ``evidence_level``, ``organelle_gene``).
    nuclear_fasta : path, optional
        Read chromosome lengths from this FASTA when ``chromosome_lengths`` is
        not given.
    chromosome_lengths : {seqid: length}, optional
        Explicit chromosome lengths; takes precedence over ``nuclear_fasta``.
    color_by : {"evidence_level", "organelle_gene"}
        How to colour the insertion markers.
    """
    if not candidates:
        raise ValueError("candidates must contain at least one transfer candidate")

    def _render(path: str | Path) -> Path:
        return _render_transfer_tracks(
            candidates,
            path,
            nuclear_fasta=nuclear_fasta,
            chromosome_lengths=chromosome_lengths,
            color_by=color_by,
            title=title,
        )

    return plot_result(
        "plot_transfer_tracks",
        _render,
        metrics={"candidates": len(candidates), "color_by": color_by},
        summary="Transfer track plot prepared.",
    )


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
    out = _prepare_output(output)
    _apply_figure_style()

    # resolve chromosome lengths
    lengths = dict(chromosome_lengths) if chromosome_lengths else {}
    if nuclear_fasta is not None and not lengths:
        from .._bio import read_fasta

        for name, seq in read_fasta(Path(nuclear_fasta)):
            lengths[name] = len(seq)
    # ensure every referenced chromosome has a length (fall back to max insert end)
    for c in candidates:
        sid = str(c.get("nuclear_seqid", ""))
        end = int(c.get("nuclear_end", 0))
        if sid and sid not in lengths:
            lengths[sid] = max(end, int(c.get("nuclear_start", 0)))
    if not lengths:
        raise ValueError(
            "no chromosome lengths available; pass nuclear_fasta or chromosome_lengths"
        )

    # order chromosomes by length (longest at top)
    chroms = sorted(lengths, key=lambda s: lengths[s], reverse=True)
    y_pos = {c: i for i, c in enumerate(chroms)}
    max_len = max(lengths.values()) or 1

    fig_h = max(3.0, 0.55 * len(chroms) + 2.0)
    fig, ax = plt.subplots(figsize=(10.0, fig_h))
    ax.set_title(title or "NUMT/NUPT insertions across nuclear chromosomes", loc="left")

    # draw chromosome bars
    for c in chroms:
        y = y_pos[c]
        L = lengths[c]
        ax.barh(y, L, height=0.55, color="#E5E7EB", edgecolor="#9CA3AF", zorder=1)
        # chromosome name + length label on the left
        ax.text(-max_len * 0.02, y, c, ha="right", va="center", fontweight="bold", fontsize=8)
        ax.text(
            L + max_len * 0.01,
            y,
            _format_bp(L),
            ha="left",
            va="center",
            fontsize=6,
            color="#6B7280",
        )

    # colour mapping
    if color_by == "organelle_gene":
        genes = sorted({str(c.get("organelle_gene", "") or "?") for c in candidates})
        gene_color = {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(genes)}
    else:  # evidence_level
        ev_levels = sorted({int(c.get("evidence_level", 0)) for c in candidates})
        ev_color = {lv: _PALETTE[lv % len(_PALETTE)] for lv in ev_levels}

    # plot insertions
    max_frag = max((int(c.get("length", 1)) for c in candidates), default=1) or 1
    for c in candidates:
        sid = str(c.get("nuclear_seqid", ""))
        if sid not in y_pos:
            continue
        y = y_pos[sid]
        start = int(c.get("nuclear_start", 0))
        end = int(c.get("nuclear_end", start))
        mid = (start + end) / 2
        frag = max(1, int(c.get("length", end - start + 1)))
        size = 20 + 80 * (frag / max_frag)  # scale marker by fragment length
        if color_by == "organelle_gene":
            color = gene_color.get(str(c.get("organelle_gene", "") or "?"), _NEUTRAL)
        else:
            color = ev_color.get(int(c.get("evidence_level", 0)), _NEUTRAL)
        ax.scatter(
            [mid],
            [y],
            color=color,
            s=size,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        # insertion span as a short horizontal tick
        ax.plot(
            [start, end], [y, y], color=color, lw=2.2, alpha=0.7, zorder=2, solid_capstyle="round"
        )

    ax.set_yticks([])
    ax.set_xlim(-max_len * 0.12, max_len * 1.08)
    ax.set_ylim(-0.7, len(chroms) - 0.3)
    ax.set_xlabel("Chromosome position (bp)")
    _despine(ax)

    # legend
    if color_by == "organelle_gene":
        from matplotlib.lines import Line2D as _L

        handles = [
            _L(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=col,
                markersize=7,
                label=g or "intergenic",
            )
            for g, col in gene_color.items()
        ]
        ax.legend(
            handles=handles, title="Organelle gene", loc="upper right", framealpha=0.9, fontsize=6
        )
    else:
        from matplotlib.lines import Line2D as _L

        handles = [
            _L(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=col,
                markersize=7,
                label=f"level {lv}",
            )
            for lv, col in ev_color.items()
        ]
        ax.legend(
            handles=handles, title="Evidence level", loc="upper right", framealpha=0.9, fontsize=6
        )
    return _save(fig, out)


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

    return plot_result(
        "plot_splicing_schematic",
        _render,
        metrics={"genes": len(genes)},
        summary="Splicing schematic prepared.",
    )


def _render_splicing_schematic(
    genes: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_splicing_schematic`."""
    out = _prepare_output(output)
    _apply_figure_style()

    rows: list[tuple[str, Mapping[str, Any]]] = []
    for gene in genes:
        exons = list(gene.get("exons", []))
        if not exons:
            raise ValueError("each gene must contain at least one exon")
        for exon in exons:
            rows.append((str(gene.get("gene", "gene")), exon))

    fig, ax = plt.subplots(figsize=(9.0, max(3.8, 0.62 * len(rows) + 1.8)))
    ax.set_title(title or "Trans-splicing schematic", loc="left")
    y = len(rows) - 1
    points: dict[str, list[tuple[float, float]]] = {}
    max_end = max(1.0, max(_float(exon.get("end", 1.0)) for _, exon in rows))
    gene_palette = _group_palette(gene_name for gene_name, _ in rows)
    for gene_name, exon in rows:
        start = _float(exon.get("start", 0.0)) / max_end
        end = _float(exon.get("end", start)) / max_end
        width = max(0.014, end - start)
        color = gene_palette[gene_name]
        ax.add_patch(
            Rectangle((start, y - 0.15), width, 0.3, facecolor=color, edgecolor="white", lw=0.8)
        )
        label = str(exon.get("exon", "exon"))
        contig = str(exon.get("contig", "contig"))
        ax.text(start, y + 0.2, label, fontsize=7, ha="left", va="bottom")
        ax.text(-0.02, y, f"{gene_name} / {contig}", ha="right", va="center")
        points.setdefault(gene_name, []).append((start + width / 2, y))
        y -= 1
    for _gene_name, coords in points.items():
        coords = sorted(coords, key=lambda xy: xy[1], reverse=True)
        for (x1, y1), (x2, y2) in itertools.pairwise(coords):
            arrow = FancyArrowPatch(
                (x1, y1 - 0.18),
                (x2, y2 + 0.18),
                connectionstyle="arc3,rad=-0.28",
                arrowstyle="-",
                lw=1.1,
                color="#6B7280",
                alpha=0.8,
            )
            ax.add_patch(arrow)
    ax.set_xlim(-0.24, 1.03)
    ax.set_ylim(-0.7, len(rows) - 0.25)
    ax.set_xlabel("Scaled genomic position")
    ax.set_yticks([])
    _despine(ax)
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

    return plot_result(
        "plot_localization_bars",
        _render,
        metrics={"proteins": len(predictions)},
        summary="Localization bar plot prepared.",
    )


def _render_localization_bars(
    predictions: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_localization_bars`."""
    out = _prepare_output(output)
    _apply_figure_style()

    compartments = sorted(
        {
            str(comp)
            for pred in predictions
            for comp in (
                pred.get("probabilities", {}).keys()
                if isinstance(pred.get("probabilities", {}), Mapping)
                else ()
            )
        }
    )
    if not compartments:
        raise ValueError("predictions must include probabilities")
    proteins = [
        str(pred.get("protein", pred.get("id", f"protein {i + 1}")))
        for i, pred in enumerate(predictions)
    ]
    palette = _group_palette(compartments)

    fig, ax = plt.subplots(figsize=(8.6, max(3.6, 0.42 * len(proteins) + 2.0)))
    left = [0.0 for _ in predictions]
    y = list(range(len(predictions)))
    for comp in compartments:
        values = []
        for pred in predictions:
            probs = pred.get("probabilities", {})
            values.append(_float(probs.get(comp, 0.0)) if isinstance(probs, Mapping) else 0.0)
        ax.barh(y, values, left=left, label=comp, color=palette[comp], height=0.66)
        left = [a + b for a, b in zip(left, values, strict=False)]
    ax.set_yticks(y, proteins)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.0, max(left)))
    ax.set_xlabel("Probability")
    ax.set_title(title or "Subcellular localization probabilities", loc="left")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
        ncol=min(4, len(compartments)),
    )
    _despine(ax)
    return _save(fig, out)


def plot_rscu_usage(
    rscu_rows: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    *,
    title: str | None = None,
    rna_labels: bool = True,
    include_stop: bool = False,
) -> OrganellePlot:
    """Render RSCU codon-usage bias as amino-acid grouped codon stacks (compute-only)."""
    rows = _rscu_rows(rscu_rows)
    if not include_stop:
        rows = [row for row in rows if row["aa"] != "*"]
    if not rows:
        raise ValueError("rscu_rows must contain at least one codon row")

    def _render(path: str | Path) -> Path:
        return _render_rscu_usage(rows, path, title=title, rna_labels=rna_labels)

    return plot_result(
        "plot_rscu_usage",
        _render,
        metrics={"codon_rows": len(rows)},
        summary="RSCU codon-usage plot prepared.",
    )


def _render_rscu_usage(
    rows: list[dict[str, Any]],
    output: str | Path,
    *,
    title: str | None = None,
    rna_labels: bool = True,
) -> Path:
    """Private path-taking renderer backing :func:`plot_rscu_usage`."""
    out = _prepare_output(output)
    _apply_figure_style()

    by_aa: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_aa.setdefault(str(row["aa"]), []).append(row)
    ordered_aas = [aa for aa in _AA_ORDER if aa in by_aa]
    ordered_aas.extend(sorted(aa for aa in by_aa if aa not in ordered_aas))
    if not ordered_aas:
        raise ValueError("rscu_rows must contain at least one codon row")

    fig, ax = plt.subplots(figsize=(max(6.8, 0.44 * len(ordered_aas) + 2.2), 4.8))
    max_stack = 0.0
    for xi, aa in enumerate(ordered_aas):
        family = sorted(by_aa[aa], key=lambda row: row["codon"])
        bottom = 0.0
        for j, row in enumerate(family):
            value = max(0.0, _float(row["rscu"]))
            color = _PALETTE[j % len(_PALETTE)]
            ax.bar(
                xi, value, bottom=bottom, width=0.72, color=color, edgecolor="white", linewidth=0.45
            )
            if value >= 0.18:
                label = _display_codon(str(row["codon"]), rna_labels)
                ax.text(
                    xi,
                    bottom + value / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=5.7,
                    color=_text_color_for_bar(color),
                )
            bottom += value
        max_stack = max(max_stack, bottom)

    ax.axhline(1.0, color=_NEUTRAL, lw=0.8, ls="--", alpha=0.72)
    ax.text(
        len(ordered_aas) - 0.55,
        1.04,
        "RSCU = 1",
        ha="right",
        va="bottom",
        fontsize=7,
        color=_NEUTRAL,
    )
    ax.set_xticks(
        range(len(ordered_aas)), [_AA_NAMES.get(aa, aa) for aa in ordered_aas], rotation=0
    )
    ax.set_xlim(-0.6, len(ordered_aas) - 0.4)
    ax.set_ylim(0, max(1.4, max_stack * 1.08))
    ax.set_ylabel("RSCU value")
    ax.set_title(title or "RSCU codon usage bias", loc="left")
    _despine(ax)
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

    return plot_result(
        "plot_rna_editing_summary",
        _render,
        metrics={"editing_sites": len(editing_sites)},
        summary="RNA-editing summary prepared.",
    )


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

    rows = [_editing_site_row(site, i) for i, site in enumerate(editing_sites)]
    gene_counts = _count_by(row["gene"] for row in rows)
    edit_types = sorted({str(row["edit_type"]) for row in rows})
    palette = _group_palette(edit_types)
    top_genes = sorted(gene_counts, key=lambda key: (-gene_counts[key], key))[: max(1, top_n)]

    fig, ax = plt.subplots(figsize=(7.0, max(3.8, 0.28 * len(top_genes) + 1.7)))
    y = list(range(len(top_genes)))
    left = [0 for _ in top_genes]
    for edit_type in edit_types:
        values = [
            sum(1 for row in rows if row["gene"] == gene and row["edit_type"] == edit_type)
            for gene in top_genes
        ]
        ax.barh(y, values, left=left, color=palette[edit_type], height=0.66, label=edit_type)
        left = [a + b for a, b in zip(left, values, strict=False)]
    for yi, total in zip(y, left, strict=False):
        ax.text(total + 0.2, yi, str(total), va="center", ha="left", fontsize=7)
    ax.set_yticks(y, top_genes)
    ax.invert_yaxis()
    max_total = max(left)
    ax.set_xlim(0, max_total + max(0.5, max_total * 0.08))
    ax.set_xlabel("Number of RNA editing sites")
    ax.set_title(title or "RNA editing sites by gene", loc="left")
    ax.legend(loc="lower right", frameon=False)
    _despine(ax)
    return _save(fig, out)


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

    return plot_result(
        "plot_erc_pair_scatter",
        _render,
        metrics={"method": method, "points": len(points)},
        summary="ERC pair scatter prepared.",
    )


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
    """Private path-taking renderer backing :func:`plot_erc_pair_scatter`."""
    out = _prepare_output(output)
    _apply_figure_style()

    x = [_float(point.get(x_key, point.get("x", 0.0))) for point in points]
    y = [_float(point.get(y_key, point.get("y", 0.0))) for point in points]
    r_value = _correlation(x, y, method)
    label_a = gene_a or str(points[0].get("gene_a", "gene A"))
    label_b = gene_b or str(points[0].get("gene_b", "gene B"))

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.scatter(x, y, s=26, color="#4B5563", alpha=0.86, edgecolor="white", linewidth=0.35)
    fit = _linear_fit(x, y)
    if fit is not None:
        slope, intercept = fit
        xs = [min(x), max(x)]
        ax.plot(xs, [slope * xi + intercept for xi in xs], color="#EF4444", lw=1.2)
    ax.text(
        0.04,
        0.92,
        f"{method.title()} ERC r = {r_value:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": _LIGHT},
    )
    ax.set_xlabel(f"Relative branch length: {label_a}")
    ax.set_ylabel(f"Relative branch length: {label_b}")
    ax.set_title(title or f"ERC pair scatter: {label_a} vs {label_b}", loc="left")
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

    return plot_result(
        "plot_erc_distribution",
        _render,
        metrics={"method": method, "pairs": len(rows)},
        summary="ERC distribution prepared.",
    )


def _render_erc_distribution(
    rows: list[dict[str, Any]],
    output: str | Path,
    *,
    method: str,
    r_threshold: float = 0.15,
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_erc_distribution`."""
    out = _prepare_output(output)
    _apply_figure_style()

    values = [row["abs_r"] for row in rows]
    centers, density = _hist_density(values, bins=44, low=0.0, high=1.0)
    threshold = max(0.0, min(1.0, _float(r_threshold)))

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.fill_between(centers, density, color="#9BD3E0", alpha=0.72)
    ax.plot(centers, density, color="#172033", lw=0.9)
    ax.axvline(threshold, color="white", lw=1.5, ls="--")
    ax.axvline(threshold, color=_NEUTRAL, lw=0.7, ls="--", alpha=0.85)
    ax.text(
        threshold + 0.01,
        max(density) * 0.83,
        f"r >= {threshold:.2f}",
        ha="left",
        va="center",
        fontsize=7,
        color=_NEUTRAL,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("ERC strength (|r|)")
    ax.set_ylabel("Density")
    ax.set_title(title or f"Distribution of ERC strengths ({method})", loc="left")

    counts: dict[str, int] = {}
    for row in rows:
        if row["abs_r"] >= threshold:
            label = _erc_pair_group_label(row)
            counts[label] = counts.get(label, 0) + 1
    if counts:
        inset = ax.inset_axes([0.54, 0.46, 0.4, 0.4])
        labels = sorted(counts, key=lambda label: counts[label], reverse=True)[:4]
        bars = [counts[label] for label in labels]
        inset.bar(
            range(len(labels)),
            bars,
            color=[_PALETTE[i % len(_PALETTE)] for i in range(len(labels))],
        )
        inset.set_title(f"{sum(counts.values())} pairs above threshold", fontsize=7)
        inset.set_xticks(
            range(len(labels)),
            [_short_group_label(label) for label in labels],
            rotation=25,
            ha="right",
            fontsize=6,
        )
        inset.set_ylabel("Pairs", fontsize=6)
        inset.tick_params(axis="y", labelsize=6)
        _despine(inset)
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

    return plot_result(
        "plot_erc_group_ridges",
        _render,
        metrics={"method": method, "pairs": len(rows)},
        summary="ERC group ridges prepared.",
    )


def _render_erc_group_ridges(
    rows: list[dict[str, Any]],
    output: str | Path,
    *,
    method: str,
    group_order: Sequence[str] | None = None,
    r_threshold: float = 0.15,
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_erc_group_ridges`."""
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

    fig, ax = plt.subplots(figsize=(6.8, max(3.8, 0.72 * len(ordered) + 1.4)))
    for i, group in enumerate(ordered):
        base = len(ordered) - i - 1
        centers, density = _hist_density(group_values[group], bins=34, low=0.0, high=1.0)
        max_density = max(density) if density else 1.0
        scaled = [(value / max_density) * 0.55 if max_density else 0.0 for value in density]
        color = _PALETTE[i % len(_PALETTE)]
        ax.fill_between(
            centers, [base] * len(centers), [base + v for v in scaled], color=color, alpha=0.64
        )
        ax.plot(centers, [base + v for v in scaled], color="#172033", lw=0.7)
        median = _median(group_values[group])
        ax.vlines(median, base, base + 0.48, color="#111827", lw=1.0)
        ax.text(
            1.02,
            base + 0.18,
            f"n={len(group_values[group])}",
            ha="left",
            va="center",
            fontsize=7,
            color=_NEUTRAL,
        )
    threshold = max(0.0, min(1.0, _float(r_threshold)))
    ax.axvline(threshold, color=_NEUTRAL, lw=0.8, ls="--", alpha=0.8)
    ax.set_xlim(0, 1.16)
    ax.set_ylim(-0.25, len(ordered) - 0.35)
    ax.set_yticks([len(ordered) - i - 1 + 0.2 for i in range(len(ordered))], ordered)
    ax.set_xlabel("ERC strength (|r|)")
    ax.set_title(title or f"Group-wise ERC distributions ({method})", loc="left")
    _despine(ax)
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

    return plot_result(
        "plot_erc_network",
        _render,
        metrics={"method": method, "edges": len(filtered)},
        summary="ERC network prepared.",
    )


def _render_erc_network(
    filtered: list[dict[str, Any]],
    output: str | Path,
    *,
    method: str,
    label_top_n: int = 28,
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_erc_network`."""
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
        edges.append({"source": gene_a, "target": gene_b, "weight": row["abs_r"]})
    node_ids = sorted(degrees, key=lambda gene: (groups.get(gene, ""), -degrees[gene], gene))
    positions = _network_positions(node_ids, groups)
    palette = _group_palette(groups.values())
    labelled = set(sorted(degrees, key=lambda gene: (-degrees[gene], gene))[: max(0, label_top_n)])

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        weight = _float(edge.get("weight", 0.0))
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#9CA3AF",
            lw=max(0.45, min(2.5, 0.4 + 2.0 * weight)),
            alpha=0.36,
            zorder=1,
        )
    for gene in node_ids:
        x, y = positions[gene]
        group = groups.get(gene, "unassigned")
        size = 90 + min(420, degrees.get(gene, 1) * 28)
        ax.scatter(
            [x], [y], s=size, color=palette[group], edgecolor="white", linewidth=0.9, zorder=2
        )
    for gene in node_ids:
        if gene not in labelled:
            continue
        x, y = positions[gene]
        label_y = y + (0.08 if y >= 0 else -0.08)
        va = "bottom" if y >= 0 else "top"
        ax.text(x, label_y, _compact_gene_label(gene), ha="center", va=va, fontsize=6.5, zorder=3)
    legend_handles = [
        ax.scatter(
            [], [], s=52, color=palette[group], edgecolor="white", linewidth=0.8, label=group
        )
        for group in sorted(set(groups.values()))
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            frameon=False,
            ncol=min(4, len(legend_handles)),
        )
    ax.set_title(title or f"High-ERC network ({method}, top {len(filtered)} edges)", loc="left")
    ax.set_aspect("equal")
    ax.set_xlim(-1.24, 1.24)
    ax.set_ylim(-1.28, 1.36)
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
) -> OrganellePlot:
    """Render ERC strength against significance, highlighting candidates (compute-only)."""
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
        )

    return plot_result(
        "plot_erc_significance",
        _render,
        metrics={"method": method, "pairs": len(rows)},
        summary="ERC significance plot prepared.",
    )


def _render_erc_significance(
    rows: list[dict[str, Any]],
    output: str | Path,
    *,
    method: str,
    highlight_genes: Sequence[str] | None = None,
    highlight_groups: Sequence[str] | None = None,
    q_threshold: float = 0.05,
    title: str | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_erc_significance`."""
    out = _prepare_output(output)
    _apply_figure_style()

    highlight_gene_set = {str(gene) for gene in highlight_genes or ()}
    highlight_group_set = {str(group) for group in highlight_groups or ()}
    x = [row["abs_r"] for row in rows]
    y = [_neg_log10(row.get("q_value", row.get("p_value", 1.0))) for row in rows]
    highlighted = [
        (
            str(row["gene_a"]) in highlight_gene_set
            or str(row["gene_b"]) in highlight_gene_set
            or str(row.get("group_a", "")) in highlight_group_set
            or str(row.get("group_b", "")) in highlight_group_set
        )
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    normal_x = [value for value, is_hit in zip(x, highlighted, strict=False) if not is_hit]
    normal_y = [value for value, is_hit in zip(y, highlighted, strict=False) if not is_hit]
    hit_x = [value for value, is_hit in zip(x, highlighted, strict=False) if is_hit]
    hit_y = [value for value, is_hit in zip(y, highlighted, strict=False) if is_hit]
    ax.scatter(normal_x, normal_y, s=14, color="#B8C0CC", alpha=0.72, edgecolor="none")
    ax.scatter(hit_x, hit_y, s=32, color=_ALARM, alpha=0.9, edgecolor="white", linewidth=0.45)
    ax.axhline(_neg_log10(q_threshold), color=_ALARM, lw=0.9, ls=(0, (5, 4)), alpha=0.85)
    ax.axvline(0.15, color=_NEUTRAL, lw=0.8, ls="--", alpha=0.68)
    ax.axvline(0.5, color=_NEUTRAL, lw=0.8, ls="--", alpha=0.42)

    strict_labelled = [
        row
        for row in rows
        if str(row["gene_a"]) in highlight_gene_set and str(row["gene_b"]) in highlight_gene_set
    ]
    label_candidates = strict_labelled or [
        row for row, is_hit in zip(rows, highlighted, strict=False) if is_hit
    ]
    labelled = sorted(
        label_candidates,
        key=lambda row: (row["abs_r"], _neg_log10(row.get("q_value", row.get("p_value", 1.0)))),
        reverse=True,
    )[:4]
    for i, row in enumerate(labelled):
        label = (
            f"{_compact_gene_label(str(row['gene_a']))}-{_compact_gene_label(str(row['gene_b']))}"
        )
        offset = [0.11, -0.11, 0.07, -0.07][i % 4]
        if row["abs_r"] >= 0.52:
            label_x = row["abs_r"] - 0.018
            ha = "right"
        else:
            label_x = row["abs_r"] + 0.014
            ha = "left"
        ax.text(
            label_x,
            _neg_log10(row.get("q_value", row.get("p_value", 1.0))) + offset,
            label,
            fontsize=6.2,
            ha=ha,
            va="center",
        )
    ax.set_xlim(0, min(1.0, max(0.25, max(x) + 0.08)))
    ax.set_ylim(0, max(2.0, max(y) * 1.15))
    ax.set_xlabel("ERC strength (|r|)")
    ax.set_ylabel("-log10(q-value)")
    ax.set_title(title or f"ERC significance screen ({method})", loc="left")
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
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: Any, output: Path) -> Path:
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def _figure_size(
    default_width: float,
    default_height: float,
    width: float | None = None,
    height: float | None = None,
) -> tuple[float, float]:
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
            lw=0.34,
            alpha=0.55,
            zorder=4.6,
            solid_capstyle="round",
        )
        ax.scatter(
            [base_x],
            [display_y],
            marker=style["marker"],
            s=5.8,
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


def _draw_ideogram_marker_legend(
    ax: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    below_density: bool = False,
    label_names: Mapping[str, str] | None = None,
) -> None:
    if not rows:
        return
    resolved_label_names = _ideogram_label_names(rows, label_names)
    handles = []
    for label_type in _ideogram_marker_types(rows):
        style = _ideogram_marker_style(label_type)
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="none",
                markerfacecolor=style["color"],
                markeredgewidth=0,
                markersize=3.6,
                label=resolved_label_names.get(label_type, label_type),
            )
        )
    anchor_y = 0.805 if below_density else 0.88
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.84, anchor_y),
        frameon=False,
        handlelength=0.65,
        handletextpad=0.22,
        labelspacing=0.06,
        fontsize=5.6,
    )


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


def _ideogram_marker_style(label_type: str) -> dict[str, str]:
    return _IDEOGRAM_MARKERS.get(label_type, {"color": _NEUTRAL, "marker": "o"})


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
        y, [base_x] * len(y), area_x, color="#FB8D62", alpha=0.94, linewidth=0, zorder=2.3
    )
    ax.plot(line_x, y, color="#7597C7", lw=1.6, zorder=3.2, solid_capstyle="round")


def _draw_ideogram_density_legend(ax: Any) -> None:
    x, y, width, height = _ideogram_density_legend_bounds()
    legend_ax = ax.inset_axes([x, y, width, height])
    steps = 28
    for i in range(steps):
        legend_ax.add_patch(
            Rectangle(
                (i / steps, 0),
                1 / steps,
                1,
                transform=legend_ax.transAxes,
                facecolor=_ideogram_density_color(i / max(1, steps - 1)),
                edgecolor="none",
            )
        )
    legend_ax.set_axis_off()
    label_y = y - 0.012
    ax.text(x, label_y, "Low", transform=ax.transAxes, ha="left", va="top", fontsize=5.8)
    ax.text(x + width, label_y, "High", transform=ax.transAxes, ha="right", va="top", fontsize=5.8)


def _ideogram_density_legend_bounds() -> tuple[float, float, float, float]:
    return (0.735, 0.885, 0.105, 0.030)


def _ideogram_density_color(value: float) -> str:
    return _blend_hex("#EAF7F4", "#2BA36A", max(0.0, min(1.0, value)))


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


def _scaled_mid(event: Mapping[str, Any], role: str, maximum: float) -> float:
    start = _float(event.get(f"{role}_start", 0.0))
    end = _float(event.get(f"{role}_end", start))
    return max(0.0, min(1.0, ((start + end) / 2) / max(1.0, maximum)))


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
    fig_height = max(3.0, 0.72 * len(samples) + (1.7 if show_features else 1.0))
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(11.5, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0] * panel_count},
    )
    axes_list = list(axes) if isinstance(axes, Iterable) and not hasattr(axes, "plot") else [axes]
    track_axes = axes_list[1:] if show_features else axes_list
    if show_features:
        _render_identity_gene_track(axes_list[0], region_features, region_start, region_end)

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
        ax.add_collection(PolyCollection(polygons, facecolors=colors, alpha=0.85, linewidths=0))
        ax.plot(x_steps, y_steps, color=_IDENTITY_CURVE, lw=0.7)
        ax.axhline(conserved_threshold, color="#9CA3AF", lw=0.6, ls="--")
        ax.set_ylim(min_identity, 100.0)
        ax.set_yticks([min_identity, conserved_threshold, 100.0])
        ax.set_yticklabels(
            [f"{min_identity:.0f}", f"{conserved_threshold:.0f}", "100"],
            fontsize=6,
        )
        ax.set_xlim(region_start, region_end)
        ax.text(
            1.005,
            0.5,
            sample,
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=7,
        )
        _despine(ax)
    track_axes[0].set_ylabel("Identity (%)")
    track_axes[-1].set_xlabel("Reference position (bp)")
    default_title = "Genome identity"
    if reference_name:
        default_title = f"Identity to {reference_name}"
    fig.suptitle(title or default_title, x=0.02, ha="left", fontsize=10)
    if show_features:
        handles = [
            Rectangle((0, 0), 1, 1, color=_IDENTITY_CATEGORY_COLORS[category], lw=0)
            for category in ("cds", "utr", "intron", "nc_gene", "noncoding")
        ]
        labels = ["Coding", "UTR", "Intron", "Non-coding gene", "Conserved non-coding"]
        axes_list[0].legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02),
            ncol=5,
            frameon=False,
            handlelength=1.2,
            columnspacing=0.9,
        )
    return _save(fig, out)


def _render_identity_gene_track(
    ax: Any,
    features: list[dict[str, Any]],
    region_start: float,
    region_end: float,
) -> None:
    """Draw the mVISTA gene-arrow track: two lanes by strand, labeled arrows."""
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
    forward_labels: list[list[float]] = [[], []]
    reverse_labels: list[list[float]] = [[], []]
    for group in groups.values():
        parts = sorted(group["parts"])
        start = parts[0][0]
        end = parts[-1][1]
        strand = group["strand"]
        color = _IDENTITY_CATEGORY_COLORS.get(group["category"], _IDENTITY_CATEGORY_COLORS["noncoding"])
        y = 1.0 if strand == 1 else 0.0
        ax.add_patch(Rectangle((start, y - 0.055), end - start, 0.11, color=color, lw=0))
        for part_start, part_end in parts:
            ax.add_patch(
                Rectangle((part_start, y - 0.17), part_end - part_start, 0.34, color=color, lw=0)
            )
        head = min(900.0, (end - start) * 0.15)
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
        label_rows = forward_labels if strand == 1 else reverse_labels
        label_y_options = (1.42, 1.78) if strand == 1 else (-0.78, -1.14)
        label = group["name"]
        label_width = max(0.01, len(label) * 0.0045) * (region_end - region_start)
        placed = False
        for row_index in (0, 1):
            row = label_rows[row_index]
            if not row or start > row[-1] + (region_end - region_start) * 0.002:
                row.extend((start, start + label_width))
                ax.text(
                    (start + end) / 2,
                    label_y_options[row_index],
                    label,
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color="#374151",
                )
                placed = True
                break
        if not placed:
            # Both label rows are busy here: draw the arrow without a label.
            label_rows[0].extend((start, start + label_width))
    ax.axhline(1.0, color="#D1D5DB", lw=0.8)
    ax.axhline(0.0, color="#D1D5DB", lw=0.8)
    ax.set_ylim(-1.6, 2.2)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["-", "+"], fontsize=7)
    ax.set_ylabel("Genes", fontsize=7)
    ax.set_xlim(region_start, region_end)
    _despine(ax)


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
