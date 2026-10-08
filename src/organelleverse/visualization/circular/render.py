"""Slot-parameterized ring drawing primitives (Cartesian arc renderers).

Ported from `pan_circular.py`'s `_draw_density_band` / `_draw_sv_arcs` /
`_draw_diversity_arcs` / `_draw_pan_segment_arcs` / `_draw_ring_band` /
`_arc_xy` / `_density_norm` / `_draw_heatmap_colorbar`, with the hard-coded
`_R_*` module-level radius constants replaced by a `Slot(r_in, r_out)`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..ogdraw import _bp_to_deg, _draw_arc_block
from .prepare import DensityData, DiversityData, SegmentData, SVData
from .tracks import Slot, allocate_rings


def density_norm(density: Sequence[float]) -> tuple[Any, Any]:
    import matplotlib
    from matplotlib.colors import Normalize

    arr = np.asarray(density, dtype=float)
    nz = arr[arr > 0]
    vmax = float(np.percentile(nz, 98)) if nz.size else 1.0
    vmax = max(1.0, vmax)
    cmap = matplotlib.colormaps.get_cmap("Blues").copy()
    return Normalize(vmin=0.0, vmax=vmax), cmap


def draw_density(ax: Any, data: DensityData, slot: Slot, genome: int) -> None:
    """Per-window variant-density band: full-height arcs graded light->dark blue."""
    if not any(data.density):
        return
    norm, cmap = density_norm(data.density)
    for start, end, value in zip(data.bin_starts, data.bin_ends, data.density, strict=False):
        _draw_arc_block(
            ax,
            float(start),
            float(end),
            slot.r_in,
            slot.r_out,
            genome,
            cmap(norm(float(value))),
            lw=0.0,
            edge="none",
        )


def draw_density_colorbar(fig: Any, data: DensityData) -> None:
    norm, cmap = density_norm(data.density)
    vmax = float(norm.vmax)
    cax = fig.add_axes([0.085, 0.10, 0.014, 0.14])
    steps = 128
    y_edges = np.linspace(0.0, vmax, steps + 1)
    grad = np.linspace(0.0, vmax, steps).reshape(-1, 1)
    cax.pcolormesh(
        np.array([0.0, 1.0]),
        y_edges,
        grad,
        cmap=cmap,
        norm=norm,
        shading="flat",
        antialiased=False,
    )
    cax.set_xticks([])
    cax.set_ylim(0.0, vmax)
    cax.yaxis.set_ticks_position("right")
    if vmax <= 6:
        cax.set_yticks(list(range(0, int(vmax) + 1)))
    cax.tick_params(labelsize=6, length=2, width=0.5, pad=1.5)
    for spine in cax.spines.values():
        spine.set_linewidth(0.4)
    cax.set_ylabel("variant density / window", fontsize=6.2, labelpad=3)
    cax.yaxis.set_label_position("left")


def draw_ring_band(ax: Any, r_in: float, r_out: float, color: str) -> None:
    from matplotlib.patches import Wedge

    ax.add_patch(
        Wedge(
            (0, 0), r_out, 0, 360, width=r_out - r_in, facecolor=color, edgecolor="none", zorder=0.5
        )
    )


def _arc_xy(bp_values: np.ndarray, radii: np.ndarray, genome: int) -> tuple[np.ndarray, np.ndarray]:
    ang = np.radians([_bp_to_deg(float(bp), genome) for bp in bp_values])
    return radii * np.cos(ang), radii * np.sin(ang)


def draw_sv(ax: Any, data: SVData, slot: Slot, genome: int) -> None:
    density = data.density
    if density is None or not np.any(density):
        return
    draw_ring_band(ax, slot.r_in, slot.r_out, "#FBF3E7")
    values = np.asarray(density, dtype=float)
    max_value = float(np.nanmax(values)) or 1.0
    for start, end, value in zip(data.bin_starts, data.bin_ends, values, strict=False):
        if value <= 0:
            continue
        r_out = slot.r_in + (value / max_value) * (slot.r_out - slot.r_in)
        _draw_arc_block(
            ax, float(start), float(end), slot.r_in, r_out, genome, "#A63603", lw=0.0, edge="none"
        )


def draw_diversity(ax: Any, data: DiversityData, slot: Slot, genome: int) -> None:
    if len(data.bin_starts) == 0:
        return
    draw_ring_band(ax, slot.r_in, slot.r_out, "#EFE7CC")
    mids = np.array([(a + b) / 2 for a, b in zip(data.bin_starts, data.bin_ends, strict=False)])
    max_value = max(
        float(np.nanmax(data.snp)),
        float(np.nanmax(data.indel)),
        1e-9,
    )
    span = slot.r_out - slot.r_in
    for values, color, lw, fill in (
        (data.snp, "#1B7837", 1.5, True),
        (data.indel, "#762A83", 1.3, False),
    ):
        if not np.any(values):
            continue
        radii = slot.r_in + (np.asarray(values, dtype=float) / max_value) * span
        bp_closed = np.r_[mids, mids[0]]
        r_closed = np.r_[radii, radii[0]]
        x, y = _arc_xy(bp_closed, r_closed, genome)
        if fill:
            xb, yb = _arc_xy(bp_closed, np.full_like(r_closed, slot.r_in), genome)
            ax.fill(
                np.r_[x, xb[::-1]],
                np.r_[y, yb[::-1]],
                color=color,
                alpha=0.12,
                edgecolor="none",
                zorder=1,
            )
        ax.plot(x, y, color=color, lw=lw, zorder=3, solid_capstyle="round")


def draw_segments(ax: Any, data: SegmentData, slot: Slot, genome: int) -> None:
    colors = {
        "core": "#55C667",
        "shell": "#35B8C8",
        "cloud": "#7E62D9",
        "private": "#D95F9F",
        "variable": "#E6AB02",
    }
    for i, row in enumerate(data.rows):
        color = colors.get(row.segment_class.lower(), colors["variable"])
        r_in = slot.r_in + (i % 3) * 0.016
        span = 0.05 if row.coverage is None else max(0.024, 0.052 * float(row.coverage))
        _draw_arc_block(
            ax,
            float(row.start),
            float(row.end),
            r_in,
            r_in + span,
            genome,
            color,
            lw=0.0,
            edge="white",
        )


# ── plot_circular orchestration ──────────────────────────────────────────────
# Generalizes pan_circular.py's hard-coded five/six-layer figure to N declared
# tracks: gene rings form the OGDraw frame, inner tracks share the radial band
# between the frame and the centre (allocated by `allocate_rings`), and an
# optional centre track (Bandage graph inset or schematic) fills the middle.
_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
_GENE_RING_RADII = (0.905, 0.838)
_R_CENTER_GRAPH = 0.44
_R_CENTER_LABEL = 0.44

_TRACK_DRAWERS = {
    "density": draw_density,
    "sv": draw_sv,
    "diversity": draw_diversity,
    "segments": draw_segments,
}


def _draw_bandage_inset(ax: Any, image_path: Any, radius: float = _R_CENTER_GRAPH) -> None:
    """Overlay a Bandage-rendered graph image at the map centre.

    Ported from `pan_circular.py`'s `_draw_bandage_inset`: the opaque white
    canvas Bandage exports on is masked to transparent so only the graph
    itself sits over the rings, then the image is placed via `inset_axes`
    with a data-space transform so it scales with the figure.
    """
    try:
        import matplotlib.image as mpimg

        image = mpimg.imread(str(image_path))
    except Exception:
        return
    image = np.asarray(image, dtype=float)
    if image.max() > 1.0:
        image = image / 255.0
    if image.ndim == 3 and image.shape[2] == 3:
        image = np.dstack([image, np.ones(image.shape[:2])])
    if image.ndim == 3 and image.shape[2] == 4:
        image = image.copy()
        white = np.all(image[..., :3] > 0.94, axis=-1)
        image[white, 3] = 0.0
    box = [-radius, -radius, 2 * radius, 2 * radius]
    inset = ax.inset_axes(box, transform=ax.transData)
    inset.imshow(image, interpolation="bilinear")
    inset.set_axis_off()
    inset.patch.set_alpha(0.0)


def _draw_ring_numbering(ax: Any, entries: Sequence[tuple[str | None, float]]) -> None:
    """Black bold roman numerals I..N near the top, outer->inner order.

    Each entry is ``(label_override, radius)``; a track's own `.label`
    overrides the positional roman numeral (numbering still advances).
    """
    ink = "#111111"
    ang = math.radians(88.0)
    for idx, (label, radius) in enumerate(entries):
        text = label if label else (_ROMAN[idx] if idx < len(_ROMAN) else str(idx + 1))
        ax.text(
            radius * math.cos(ang),
            radius * math.sin(ang),
            text,
            color=ink,
            fontsize=9.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=25,
        )


def _draw_variant_legend(ax: Any, present_kinds: set[str]) -> None:
    """Compact variation-track legend, generalizing pan_circular's fixed one.

    Only includes entries for track kinds actually present on this figure.
    """
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    handles: list[Any] = []
    if "sv" in present_kinds:
        handles.append(mpatches.Patch(color="#A63603", label="SV density"))
    if "density" in present_kinds:
        handles.append(mpatches.Patch(color="#9EC9E2", label="SNP/indel density"))
    if "diversity" in present_kinds:
        handles.append(mlines.Line2D([], [], color="#1B7837", lw=1.8, label="SNP diversity"))
        handles.append(mlines.Line2D([], [], color="#762A83", lw=1.6, label="Indel diversity"))
    if "segments" in present_kinds:
        handles.append(mpatches.Patch(color="#55C667", label="Core pan segment"))
        handles.append(mpatches.Patch(color="#35B8C8", label="Shell/variable segment"))
    if not handles:
        return
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        bbox_transform=ax.transData,
        frameon=False,
        fontsize=8,
        handlelength=1.0,
        labelspacing=0.3,
        borderaxespad=0,
        title="Variation tracks",
        title_fontsize=8.5,
    )
    leg._legend_box.align = "left"
    ax.add_artist(leg)


def _draw_info_box(ax: Any, info_lines: Sequence[str]) -> None:
    """Small white info card in the upper-left, mirroring pan_circular's summary box."""
    ax.text(
        -2.15,
        1.40,
        "\n".join(info_lines),
        transform=ax.transData,
        ha="left",
        va="top",
        fontsize=7.5,
        linespacing=1.3,
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#D0D0D0",
            "linewidth": 0.5,
        },
    )


def plot_circular(
    reference_gbk: Any,
    output: Any,
    *,
    tracks: Sequence[Any],
    title: str = "",
    figsize: tuple[float, float] = (13, 13),
    dpi: int = 300,
    work_dir: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    info_lines: list[str] | None = None,
) -> Path:
    """Stack declared tracks into an OGDraw-based circular figure.

    Generalizes `pan_circular.py`'s hard-coded five/six-layer figure to N
    declared `tracks`: `GeneRing`s form the outer frame (drawn by OGDraw
    itself), a single `is_center` track fills the middle, and every other
    track shares the inner radial band allocated by `allocate_rings`.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    from ..ogdraw import draw_mito_map, parse_genbank

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(work_dir) if work_dir is not None else out.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)

    parsed = parse_genbank(reference_gbk)

    gene_rings = [t for t in tracks if t.kind == "genes"]
    center = next((t for t in tracks if t.is_center), None)
    inner = [t for t in tracks if not t.is_center and t.kind != "genes"]
    slots = allocate_rings([t.weight for t in inner]) if inner else []

    density_for_colorbar: DensityData | None = None
    present_kinds: set[str] = set()

    def inner_tracks(ax: Any, genome_length: int) -> None:
        nonlocal density_for_colorbar
        for track, slot in zip(inner, slots, strict=False):
            data = track.resolve(genome_length, work)
            draw_fn = _TRACK_DRAWERS.get(track.kind)
            if draw_fn is None:
                continue
            draw_fn(ax, data, slot, genome_length)
            present_kinds.add(track.kind)
            if track.kind == "density":
                density_for_colorbar = data

        center_kind: str | None = None
        if center is not None:
            cdata = center.resolve(genome_length, work)
            if center.kind == "graph" and getattr(cdata, "path", None) is not None:
                _draw_bandage_inset(ax, cdata.path)
                center_kind = "graph"
            elif center.kind == "schematic":
                draw_segments(ax, cdata, Slot(0.0, _R_CENTER_GRAPH), genome_length)
                present_kinds.add("segments")
                center_kind = "schematic"

        entries: list[tuple[str | None, float]] = []
        for track, radius in zip(gene_rings, _GENE_RING_RADII, strict=False):
            entries.append((track.label, radius))
        for track, slot in zip(inner, slots, strict=False):
            entries.append((track.label, (slot.r_in + slot.r_out) / 2))
        if center is not None and center_kind is not None:
            entries.append((center.label, _R_CENTER_LABEL))
        _draw_ring_numbering(ax, entries)

        _draw_variant_legend(ax, present_kinds)
        if density_for_colorbar is not None:
            draw_density_colorbar(ax.figure, density_for_colorbar)
        if info_lines:
            _draw_info_box(ax, info_lines)

    reference_name = str((metadata or {}).get("reference") or Path(reference_gbk).stem)
    draw_mito_map(
        parsed,
        genome_name=reference_name,
        output_file=out,
        width=figsize[0],
        height=figsize[1],
        dpi=dpi,
        title=title,
        draw_gc=False,
        draw_center_text=False,
        draw_legend=True,
        inner_tracks=inner_tracks,
        fold_minus_labels_out=True,
        minus_label_mode=(gene_rings[0].minus_label if gene_rings else "on_ring"),
    )
    return out
