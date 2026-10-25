"""Drawing for the population / pangenome comparison figures, at printed size.

Every figure draws into one *host* axes (``draw(host, metrics, legend)``): the
host is switched off and the real axes are inset in millimetres, with tick
labels and axis titles kept inside the host box. The same code therefore
renders a stand-alone figure (host = whole page) and a panel of
:func:`~organelleverse.visualization.compose.compose_figure` (host = one grid
cell, letter and header added by the composer, legend handles returned for
the merged legend). Type follows :mod:`.style`: 7 pt text, 6 pt ticks and
notes, 6.5 pt gene / SV labels, nothing below 5 pt, no figure titles.
"""

from __future__ import annotations

import csv
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .style import (
    FONT_PT,
    MAX_HEIGHT_MM,
    add_panel_label,
    floor_font,
    publication_style,
    width_inches,
)

BODY = FONT_PT["body"]  # 7 pt: axis titles, legend, row labels
TICK = FONT_PT["tick"]  # 6 pt: tick labels, method notes
LABEL = 6.5  # gene / SV / peak labels
LABEL_MIN = FONT_PT["min"]  # 5 pt floor when rows are dense
SUBSCRIPTED = 7.2  # labels with mathtext subscripts (0.7 x) stay >= 5 pt
MM = 1 / 25.4

INK = "#222222"
MUTED = "#6b7280"
GRID = "#e5e7eb"
HOMOLOGOUS = "#3B6FB6"
SV_RED = "#D1495B"
PEAK_RED = "#C81D25"
ABSENT = "#E4E7EB"
FST_LINE = "#1F5F8B"
PI_LINE = "#3D405B"


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def figure_width_mm(width: str | float) -> float:
    return width_inches(width) * 25.4


def _host_mm(host) -> tuple[float, float]:
    fig = host.figure
    pos = host.get_position()
    w_in, h_in = fig.get_size_inches()
    return pos.width * w_in * 25.4, pos.height * h_in * 25.4


def _inset(host, left: float, bottom: float, width: float, height: float, **kw):
    """Axes at ``left, bottom`` (mm from the host's lower-left corner), ``width`` x ``height`` mm."""
    w, h = _host_mm(host)
    ax = host.inset_axes((left / w, bottom / h, width / w, height / h), **kw)
    return ax


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=TICK, length=2.5, width=0.6, pad=1.5)


def _kb_axis(ax, label: str) -> None:
    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
    ax.set_xlabel(label, fontsize=BODY, labelpad=2)


def _text_mm(text: str, size: float) -> float:
    """Rough rendered width of ``text`` in mm (sans serif ≈ 0.55 em per character)."""
    return len(text) * size * 0.55 / 72 * 25.4


def _contig_label(name: str | None) -> str:
    return str(name).split("#", 1)[0] if name else "reference"


def _free_level(placed, mid: float, half: float) -> int:
    level = 0
    while any(lv == level and abs(mid - x) < half + w for x, w, lv in placed):
        level += 1
    return level


NOTE_LINE_MM = 2.85
YLABEL_MM = 9.0  # y-axis titles sit this far left of the axes in every scan panel


def _wrap(text: str, width_mm: float) -> str:
    import textwrap

    chars = max(20, int(width_mm / (LABEL * 0.55 / 72 * 25.4)))
    return "\n".join(textwrap.wrap(text, chars, break_on_hyphens=False))


def _fst_note(m) -> str:
    a, b = m["groups"]
    sizes = m.get("group_sizes") or {}
    note = f"{a} vs {b}"
    if sizes:
        note += f" (n = {sizes.get(a, '?')} / {sizes.get(b, '?')})"
    if m.get("window_size"):
        note += f" · {m['window_size'] / 1000:g}-kb windows, {m['step'] / 1000:g}-kb step"
    if m.get("min_snps"):
        note += f" · gaps: < {m['min_snps']} SNPs"
    return note


def _pi_note(m) -> str:
    a, b = m["groups"]
    sizes = m.get("group_sizes") or {}
    return (
        f"mean π: {a} {m['mean_pi_a']:.2e} (n = {sizes.get(a)}), {b} {m['mean_pi_b']:.2e} (n = {sizes.get(b)})"
        f" · {m['window_size'] / 1000:g}-kb windows, {m['step'] / 1000:g}-kb step"
    )


def _ylabel(ax, text: str, size: float) -> None:
    """Y-axis title at a fixed distance (mm) left of the axes, so stacked panels align."""
    ax.set_ylabel(text, fontsize=size)
    width_mm, _ = _host_mm(ax)
    ax.yaxis.set_label_coords(-YLABEL_MM / width_mm, 0.5)


def _note(ax, text: str) -> None:
    """Method line(s) in the top margin, left-aligned with the plotting area."""
    ax.annotate(
        text,
        (0, 1),
        xycoords="axes fraction",
        xytext=(0, 3),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=LABEL,
        color=MUTED,
        linespacing=1.1,
    )


def write_tsv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str], suffix: str = ".tsv"
) -> None:
    """Data table next to a figure: ``<stem><suffix>``."""
    with path.with_name(path.stem + suffix).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(fields), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})


# ---------------------------------------------------------------------------
# A / B: scans along the genome
# ---------------------------------------------------------------------------

SCAN_MARGINS = {"left": 12.0, "bottom": 8.5, "right": 9.0, "top": 4.0}


def _draw_fst_axes(ax, m, *, xlabel: bool, note: str) -> None:
    import numpy as np

    rows = m["windows"]
    x = np.array([(w["start"] + w["end"]) / 2 for w in rows], dtype=float)
    y = np.array([np.nan if w["fst"] is None else w["fst"] for w in rows], dtype=float)
    ax.fill_between(x, 0, np.clip(y, 0, None), color=FST_LINE, alpha=0.18, lw=0)
    ax.plot(x, y, color=FST_LINE, lw=0.7, zorder=3)
    finite = y[~np.isnan(y)]
    lo = min(0.0, float(finite.min())) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 1.0
    pad = (hi - lo) * 0.14 or 0.05
    ax.set_ylim(lo - (pad * 0.3 if lo < 0 else 0), hi + pad * (2.0 if m["highlights"] else 1.0))
    ax.set_xlim(0, m["genome_length"])
    if m.get("top_cut") is not None:
        cut = m["top_cut"]
        ax.axhline(cut, color=MUTED, lw=0.6, ls=(0, (4, 3)), zorder=2)
        ax.annotate(
            f"top {m['top_fraction']:.0%}",
            (1, cut),
            xycoords=("axes fraction", "data"),
            xytext=(2, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=TICK,
            color=MUTED,
        )
        top = y >= cut
        ax.scatter(x[top], y[top], s=5, color=PEAK_RED, zorder=4, lw=0)
    from matplotlib.patches import Rectangle

    y0, y1 = ax.get_ylim()
    box_top = y0 + (y1 - y0) * 0.84
    for box in m["highlights"]:
        ax.add_patch(
            Rectangle(
                (box["start"], y0),
                box["end"] - box["start"] + 1,
                box_top - y0,
                fill=False,
                edgecolor=PEAK_RED,
                lw=0.7,
                ls=(0, (3, 2)),
                zorder=5,
            )
        )
        if box.get("label"):
            ax.annotate(
                box["label"],
                ((box["start"] + box["end"]) / 2, box_top),
                xytext=(0, 1.5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=LABEL,
                fontstyle="italic",
                color=PEAK_RED,
                zorder=6,
            )
    if lo < 0:
        ax.axhline(0, color=INK, lw=0.5)
    estimator = {"heterozygosity": "H", "weir_cockerham": "WC", "hudson": "Hudson"}.get(
        m["estimator"], m["estimator"]
    )
    # 7.2 pt keeps the mathtext subscript (0.7 x) above the 5 pt floor
    _ylabel(ax, f"$F_{{ST}}$ ({estimator})", SUBSCRIPTED)
    _style_axes(ax)
    if xlabel:
        _kb_axis(ax, f"Position on {_contig_label(m.get('contig'))} (kb)")
    else:
        ax.tick_params(labelbottom=False)
    _note(ax, note)


def _tex(label: str) -> str:
    return str(label).replace("\\", "").replace(" ", "\\ ").replace("_", "\\_").replace("#", "\\#")


def _draw_pi_axes(ax, m, *, xlabel: bool, note: str) -> None:
    import numpy as np

    rows = m["windows"]
    x = np.array([(w["start"] + w["end"]) / 2 for w in rows], dtype=float)
    y = np.array([np.nan if w["ratio"] is None else w["ratio"] for w in rows], dtype=float)
    log = bool(m.get("log_scale"))
    if log:
        from matplotlib.ticker import FuncFormatter, LogLocator

        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_locator(LogLocator(base=4))
        ax.yaxis.set_minor_locator(LogLocator(base=2))
        # plain numbers: 2^n tick labels would put a 4.2 pt superscript on the page
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v:g}" if v >= 1 else f"1/{1 / v:g}")
        )
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ok = ~np.isnan(y)
    ax.fill_between(x, 1.0, y, where=ok & (y >= 1), color="#81B29A", alpha=0.35, lw=0, interpolate=True)
    ax.fill_between(x, 1.0, y, where=ok & (y < 1), color="#E07A5F", alpha=0.25, lw=0, interpolate=True)
    ax.plot(x, y, color=PI_LINE, lw=0.7, zorder=3)
    ax.axhline(1.0, color=INK, lw=0.5, zorder=2)
    finite = y[ok]
    hi = float(finite.max()) if finite.size else 2.0
    lo = float(finite.min()) if finite.size else 0.5
    if log:
        bottom = min(lo, 1.0) / 1.3
        ax.set_ylim(bottom, 2 ** (math.log2(bottom) + (math.log2(hi) - math.log2(bottom)) / 0.78))
    else:
        ax.set_ylim(0, hi / 0.78 + 0.05)
    ax.set_xlim(0, m["genome_length"])
    if m.get("top_cut") is not None:
        ax.axhline(m["top_cut"], color=MUTED, lw=0.6, ls=(0, (4, 3)), zorder=2)
        ax.annotate(
            f"top {m['top_fraction']:.0%}",
            (1, m["top_cut"]),
            xycoords=("axes fraction", "data"),
            xytext=(2, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=TICK,
            color=MUTED,
        )
    for peak in m["peaks"]:
        px = (peak["peak_start"] + peak["peak_end"]) / 2
        ax.scatter([px], [peak["peak_value"]], marker="v", s=12, color=PEAK_RED, zorder=5, lw=0)
        ax.annotate(
            peak.get("label") or "",
            (px, 0.86),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=LABEL,
            fontstyle="italic",
            color=PEAK_RED,
            zorder=6,
        )
    a, b = m["groups"]
    _ylabel(ax, f"π$_{{\\mathrm{{{_tex(a)}}}}}$ / π$_{{\\mathrm{{{_tex(b)}}}}}$", SUBSCRIPTED)
    _style_axes(ax)
    if xlabel:
        _kb_axis(ax, f"Position on {_contig_label(m.get('contig'))} (kb)")
    else:
        ax.tick_params(labelbottom=False)
    _note(ax, note)


def _scan_axes(host, notes: Sequence[str]):
    """One scan axes per note, stacked; each top margin holds its wrapped note."""
    w, h = _host_mm(host)
    mg = SCAN_MARGINS
    inner_w = w - mg["left"] - mg["right"]
    wrapped = [_wrap(n, inner_w) for n in notes]
    tops = [1.8 + NOTE_LINE_MM * (t.count("\n") + 1) for t in wrapped]
    inner_h = (h - mg["bottom"] - sum(tops) - 1.5 * (len(notes) - 1)) / len(notes)
    axes, y = [], h
    for top in tops:
        y -= top + inner_h
        axes.append(_inset(host, mg["left"], y, inner_w, inner_h))
        y -= 1.5
    return axes, wrapped


def draw_fst(host, m, legend: bool) -> list:
    host.set_axis_off()
    (ax,), (note,) = _scan_axes(host, [_fst_note(m)])
    _draw_fst_axes(ax, m, xlabel=True, note=note)
    return []


def draw_pi(host, m, legend: bool) -> list:
    host.set_axis_off()
    (ax,), (note,) = _scan_axes(host, [_pi_note(m)])
    _draw_pi_axes(ax, m, xlabel=True, note=note)
    return []


def draw_stack(host, m, legend: bool) -> list:
    host.set_axis_off()
    (ax_a, ax_b), (note_a, note_b) = _scan_axes(host, [_fst_note(m["fst"]), _pi_note(m["pi_ratio"])])
    ax_b.sharex(ax_a)
    _draw_fst_axes(ax_a, m["fst"], xlabel=False, note=note_a)
    _draw_pi_axes(ax_b, m["pi_ratio"], xlabel=True, note=note_b)
    if legend:  # stand-alone: letter the two scans; in a composite the stack is one panel
        for ax, letter in zip((ax_a, ax_b), m.get("panel_labels") or ("a", "b"), strict=False):
            add_panel_label(ax, letter, x_pt=-(SCAN_MARGINS["left"] - 1.0) / 25.4 * 72, y_pt=3)
    return []


# ---------------------------------------------------------------------------
# C: SV schematic
# ---------------------------------------------------------------------------

ROW_MM = 7.0


def _row_map(row, cap: float):
    cuts = [(b["start"], b["end"]) for b in row["blocks"] if b["kind"] == "sv" and b["end"] - b["start"] > cap]

    def to_x(pos: float) -> float:
        shift = 0.0
        for a, b in cuts:
            if pos >= b:
                shift += (b - a) - cap
            elif pos > a:
                shift += (pos - a) - cap * (pos - a) / (b - a)
        return pos - shift

    return to_x


def schematic_label_mm(m) -> float:
    return min(70.0, max(14.0, max(_text_mm(r["label"], BODY) for r in m["rows"]) + 3.0))


def draw_schematic(host, m, legend: bool) -> list:
    from matplotlib.patches import Rectangle

    host.set_axis_off()
    rows = m["rows"]
    n = len(rows)
    w, h = _host_mm(host)
    label_mm = schematic_label_mm(m)
    right_mm = 16.0
    ref_mm = 13.0
    legend_mm = 6.0 if legend else 0.0
    ax = _inset(host, label_mm, ref_mm + legend_mm, w - label_mm - right_mm, h - ref_mm - legend_mm - 1.0)
    ax2 = _inset(host, label_mm, legend_mm, w - label_mm - right_mm, ref_mm, sharex=ax)
    ref_len = rows[0]["length"] or 1
    cap = 0.35 * ref_len
    maps = [_row_map(r, cap) for r in rows]
    shifts = []
    for row, to_x in zip(rows, maps, strict=False):
        row_x, ref_x = row.get("align", [0, 0])
        shifts.append(maps[0](ref_x) - to_x(row_x))

    def xs(i, pos):
        return shifts[i] + maps[i](pos)

    x0 = min(xs(i, 0) for i in range(n))
    x1 = max(xs(i, rows[i]["length"]) for i in range(n))
    span = (x1 - x0) or 1
    ax_w_mm = w - label_mm - right_mm
    bp_per_mm = span / ax_w_mm
    bar = 0.5
    bar_mm = bar * (h - ref_mm - legend_mm - 1.0) / (n + 0.3)
    ys = [n - 1 - i for i in range(n)]
    for i, row in enumerate(rows):
        y = ys[i]
        for block in row["blocks"]:  # blocks tile the whole row; no baseline under the labels
            a, b = xs(i, block["start"]), xs(i, block["end"])
            color = HOMOLOGOUS if block["kind"] == "homologous" else SV_RED
            ax.add_patch(Rectangle((a, y - bar / 2), b - a, bar, facecolor=color, edgecolor="white", lw=0.4, zorder=3))
            if block["kind"] != "sv":
                continue
            length = block["end"] - block["start"]
            broken = length > cap
            if broken:
                mid = (a + b) / 2
                for dx in (-0.9, 0.9):
                    xx = [mid + (dx - 0.6) * bp_per_mm, mid + (dx + 0.6) * bp_per_mm]
                    yy = [y - bar * 0.62, y + bar * 0.62]
                    ax.plot(xx, yy, color="white", lw=1.8, zorder=4, solid_capstyle="butt")
                    ax.plot(xx, yy, color=INK, lw=0.5, zorder=5)
            # label inside the block; with a break mark, in the part left of it
            right = (a + b) / 2 - 2.2 * bp_per_mm if broken else b
            room = (right - a) / bp_per_mm - 1.0
            text = f"{block.get('sv') or ''} {length / 1000:.1f} kb".strip()
            if _text_mm(text, LABEL) > room:
                text = block.get("sv") or ""
            if text and _text_mm(text, LABEL) <= room and bar_mm >= LABEL * 1.2 / 72 * 25.4:
                ax.text(
                    (a + right) / 2,
                    y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=LABEL,
                    color="white",
                    fontweight="bold",
                    zorder=6,
                )
        ax.annotate(
            row["label"],
            (0, y),
            xycoords=("axes fraction", "data"),
            xytext=(-3, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=BODY,
            fontstyle="italic",
        )
        note = f"{row['length'] / 1000:.1f} kb"
        if row.get("collinear_fraction", 1) < m.get("min_span", 0.8):
            note += f" ({row['collinear_fraction']:.0%} collinear)"
        ax.annotate(
            note,
            (xs(i, row["length"]), y),
            xytext=(2, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=LABEL,
            color=MUTED,
            linespacing=0.95,
        )
    for i in range(n - 1):
        upper, lower = rows[i], rows[i + 1]
        for key, x_up in upper["edge_positions"].items():
            x_low = lower["edge_positions"].get(key)
            if x_low is not None:
                ax.plot(
                    [xs(i, x_up), xs(i + 1, x_low)],
                    [ys[i] - bar / 2, ys[i + 1] + bar / 2],
                    color="#7a8699",
                    lw=0.45,
                    ls=(0, (2.5, 2)),
                    zorder=2,
                )
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.55, n - 0.25)
    ax.set_axis_off()
    # reference line with SV sites, labels stacked when they collide
    ref_offset = int(m.get("reference_anchor_start", m["locus"]["start"])) - 1

    def ref_x(coord):
        return xs(0, coord - ref_offset)

    ax2.plot([xs(0, 0), xs(0, ref_len)], [0, 0], color=INK, lw=0.7)
    for anchor, ha in ((0, "left"), (ref_len, "right")):
        ax2.annotate(
            f"{ref_offset + max(anchor, 1):,}",
            (xs(0, anchor), 0),
            xytext=(0, -2),
            textcoords="offset points",
            ha=ha,
            va="top",
            fontsize=LABEL,
            color=MUTED,
        )
    ax2.annotate(
        _contig_label(m["reference"]),
        (0, 0),
        xycoords=("axes fraction", "data"),
        xytext=(-3, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=BODY,
    )
    placed: list[tuple[float, float, int]] = []
    for sv in m["svs"]:
        a, b = ref_x(sv["ref_start"] - 1), ref_x(sv["ref_end"])
        mid = (a + b) / 2
        if sv["in_reference"] and sv["ref_end"] >= sv["ref_start"]:
            ax2.plot([a, a, b, b], [0.1, 0.28, 0.28, 0.1], color=SV_RED, lw=0.7)
            coords = f"{sv['ref_start']:,}\u2013{sv['ref_end']:,}"
        else:
            ax2.plot([mid], [0.18], marker="v", color=SV_RED, ms=3, lw=0)
            coords = f"ins. after {sv['ref_end']:,}"
        text = f"{sv['id']} {coords}"
        half = _text_mm(text, LABEL) / 2 * bp_per_mm
        level = _free_level(placed, mid, half)
        placed.append((mid, half, level))
        ax2.text(mid, 0.36 + 0.32 * level, text, ha="center", va="bottom", fontsize=LABEL, color=SV_RED)
    ax2.set_ylim(-0.7, 0.8 + 0.32 * max([lv for _, _, lv in placed], default=0))
    ax2.set_axis_off()
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=HOMOLOGOUS, label="homologous sequence"),
        Rectangle((0, 0), 1, 1, facecolor=SV_RED, label="SV / variable sequence"),
    ]
    if legend:
        host.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(label_mm / w, 0.0),
            ncol=2,
            frameon=False,
            fontsize=BODY,
            handlelength=1.0,
            handleheight=0.8,
            borderaxespad=0,
        )
        return []
    return handles


# ---------------------------------------------------------------------------
# D: SV frequency
# ---------------------------------------------------------------------------


def _stars_text(stars: str) -> tuple[str, float, str]:
    if "*" in stars:
        return stars, BODY, INK
    return stars, LABEL, MUTED


CAPTION_LINE_MM = 2.85


def frequency_caption_lines(m, width_mm: float) -> str:
    return _wrap(m.get("caption") or "", width_mm)


def draw_frequency(host, m, legend: bool) -> list:
    from matplotlib.patches import Patch

    host.set_axis_off()
    w, h = _host_mm(host)
    legend_mm = 40.0 if legend else 0.0
    left = 11.0
    plot_w = w - left - 1.0 - legend_mm
    if legend:
        caption = frequency_caption_lines(m, w - left - 1.0)
    else:  # a composite panel: one short line; the full method is metrics["caption"] for the legend
        caption = _wrap(m.get("caption_short") or "", w - left - 1.0)
    caption_mm = (caption.count("\n") + 1) * CAPTION_LINE_MM + 1.0 if caption else 0.0
    bottom = 8.0 + caption_mm
    ax = _inset(host, left, bottom, max(plot_w, 10.0), max(h - bottom - 6.0, 10.0))
    labels, colors, svs = m["groups"], m["colors"], m["svs"]
    k = len(labels)
    bar = 0.8 / k
    for i, sv in enumerate(svs):
        for j, group in enumerate(labels):
            c = sv["counts"][group]
            total = c["present"] + c["absent"]
            if not total:
                continue
            x = i - 0.4 + bar * (j + 0.5)
            present = 100 * c["present"] / total
            ax.bar(x, present, bar * 0.9, color=colors[group], edgecolor="white", lw=0.25, zorder=3)
            ax.bar(x, 100 - present, bar * 0.9, bottom=present, color=ABSENT, edgecolor="white", lw=0.25, zorder=3)
        text, size, color = _stars_text(sv["stars"])
        ax.plot([i - 0.4 + bar * 0.5, i + 0.4 - bar * 0.5], [103, 103], color=INK, lw=0.5, clip_on=False)
        ax.text(i, 103.6, text, ha="center", va="bottom", fontsize=size, color=color, clip_on=False)
    ax.set_xticks(range(len(svs)))
    ax.set_xticklabels(
        [f"{sv['id']}\n{sv['length'] / 1000:.1f} kb" if sv.get("length") else sv["id"] for sv in svs],
        fontsize=TICK,
    )
    ax.set_xlim(-0.6, len(svs) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Genomes with SV (%)", fontsize=BODY, labelpad=2)
    _style_axes(ax)
    ax.tick_params(axis="x", length=0)
    if caption:
        host.text(left / w, (caption_mm - 0.5) / h, caption, transform=host.transAxes, ha="left", va="top",
                  fontsize=LABEL, color=MUTED, linespacing=1.15)
    units = m.get("group_units") or m["group_sizes"]
    handles = [
        Patch(facecolor=colors[g], label=f"{g} present (n = {units[g]})" if legend else g) for g in labels
    ]
    handles.append(Patch(facecolor=ABSENT, label="absent"))
    if legend:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                  fontsize=BODY, handlelength=1.0, handleheight=0.8, borderaxespad=0)
        return []
    return handles


# ---------------------------------------------------------------------------
# E: presence tracks
# ---------------------------------------------------------------------------

#: Row pitch giving 6.5 pt sample names (0.78 of the pitch is text).
TRACK_PITCH_MM = 3.0
#: Smallest pitch a composite panel may squeeze rows to; rows then show indices.
TRACK_PITCH_MIN_MM = 1.6
GROUP_LABEL_MM = 17.0


def track_bands(m) -> dict[str, float]:
    levels = max(1, min(3, len(m["svs"])))
    return {
        "header": 8.0,
        "sv": (2.5 + 3.0 * levels) if m["svs"] else 0.0,
        "genes": 10.0 if m["genes"] else 0.0,
        "axis": 7.5,
    }


def track_layout(m) -> dict[str, float]:
    """Stand-alone pitch (6.5 pt names) and heights, capped at the page height."""
    n = len(m["rows"])
    bands = track_bands(m)
    fixed = sum(bands.values())
    pitch = float(m.get("row_height") or TRACK_PITCH_MM)
    if fixed + n * pitch > MAX_HEIGHT_MM:
        pitch = max(TRACK_PITCH_MIN_MM, (MAX_HEIGHT_MM - fixed) / max(n, 1))
    return {**bands, "pitch": pitch, "height": min(MAX_HEIGHT_MM, fixed + n * pitch)}


def _row_label_plan(rows, pitch_mm: float) -> tuple[list[tuple[int, str]], float, str]:
    """(row index, text) to print, font size, mode: names if they fit at >= 6.5 pt, else every k-th index."""
    size_fit = pitch_mm / 25.4 * 72 * 0.78
    if size_fit >= LABEL:
        return [(i, r["label"]) for i, r in enumerate(rows)], LABEL, "names"
    step = max(1, math.ceil(LABEL / 72 * 25.4 / 0.78 / pitch_mm))
    step = next(s for s in (1, 2, 5, 10, 20, 50) if s >= step)
    picks = [i for i in range(len(rows)) if (i + 1) % step == 0 or i == 0]
    return [(i, str(i + 1)) for i in picks], LABEL, "index"


def draw_tracks(host, m, legend: bool) -> list:
    from matplotlib.patches import Patch, Rectangle

    host.set_axis_off()
    rows = m["rows"]
    n = len(rows)
    start, end = m["region"]["start"], m["region"]["end"]
    span = end - start + 1
    bands = track_bands(m)
    w, h = _host_mm(host)
    cols = m["annotation_columns"][1:]  # the group is labelled directly on the right
    fixed = sum(bands.values())
    pitch = max(0.3, (h - fixed) / max(n, 1))
    labels, label_pt, mode = _row_label_plan(rows, pitch)
    label_mm = max(_text_mm(t, label_pt) for _, t in labels) + 2.0
    strip_mm = 2.8 * len(cols)
    legend_mm = 36.0 if legend else 0.0
    left = label_mm + strip_mm + 1.2
    plot_w = w - left - GROUP_LABEL_MM - legend_mm
    track_h = n * pitch
    bottom_track = h - bands["header"] - track_h
    ax = _inset(host, left, bottom_track, plot_w, track_h)
    ax_lab = _inset(host, label_mm + 0.2, bottom_track, strip_mm + 0.01, track_h)
    colors = m["palette"]["Group"]
    for i, row in enumerate(rows):
        y = n - 1 - i
        ax.add_patch(Rectangle((start, y - 0.06), span, 0.12, facecolor=ABSENT, lw=0, zorder=1))
        for a, b in row["blocks"]:
            ax.add_patch(Rectangle((a, y - 0.4), b - a + 1, 0.8, facecolor=colors[row["group"]], lw=0, zorder=2))
        for j, col in enumerate(cols):
            ax_lab.add_patch(
                Rectangle((j + 0.08, y - 0.42), 0.84, 0.84, facecolor=m["palette"][col][row["annotations"][col]], lw=0)
            )
    # group brackets with direct labels on the right
    bounds = [0] + [i for i in range(1, n) if rows[i]["group"] != rows[i - 1]["group"]] + [n]
    for a, b in itertools.pairwise(bounds):
        if a:
            ax.axhline(n - a - 0.5, color=MUTED, lw=0.4, ls=(0, (2, 2)), zorder=4)
        group = rows[a]["group"]
        top_y, bottom_y = n - a - 0.5 + 0.1, n - b + 0.5 - 0.1
        ax.plot([1.012, 1.02, 1.02, 1.012], [top_y, top_y, bottom_y, bottom_y], color=colors[group], lw=0.8,
                transform=ax.get_yaxis_transform(), clip_on=False)
        ax.annotate(f"{group}\n(n = {b - a})", (1.02, (top_y + bottom_y) / 2), xycoords=("axes fraction", "data"),
                    xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=LABEL,
                    color=colors[group], linespacing=1.05)
    for sv in m["svs"]:
        ax.axvspan(sv["start"], sv["end"], color=SV_RED, alpha=0.05, lw=0, zorder=0)
    ax.set_xlim(start, end)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_axis_off()
    ax_lab.set_xlim(0, max(1, len(cols)))
    ax_lab.set_ylim(-0.6, n - 0.4)
    ax_lab.set_xticks([j + 0.5 for j in range(len(cols))])
    ax_lab.set_xticklabels(cols, rotation=90, fontsize=LABEL)
    ax_lab.xaxis.tick_top()
    ax_lab.tick_params(axis="x", length=0, pad=1)
    ax_lab.set_yticks([n - 1 - i for i, _ in labels])
    ax_lab.set_yticklabels([t for _, t in labels], fontsize=floor_font(label_pt))
    ax_lab.tick_params(axis="y", length=1.2 if mode == "index" else 0, width=0.4, pad=1.2)
    for spine in ax_lab.spines.values():
        spine.set_visible(False)
    if mode == "index":
        ax_lab.annotate("row", (0, 1), xycoords="axes fraction", xytext=(-2, 1.5), textcoords="offset points",
                        ha="right", va="bottom", fontsize=LABEL, color=MUTED)
    # SV ids (coordinates are in the <stem>.svs.tsv table); stacked when they collide
    sv_bottom = bottom_track - bands["sv"]
    bp_per_mm = span / plot_w
    if bands["sv"]:
        ax_sv = _inset(host, left, sv_bottom, plot_w, bands["sv"], sharex=ax)
        placed: list[tuple[float, float, int]] = []
        for sv in m["svs"]:
            a, b = sv["start"], sv["end"]
            ax_sv.plot([a, a, b, b], [0.0, 0.55, 0.55, 0.0], color=SV_RED, lw=0.6)
            mid = (a + b) / 2
            half = (_text_mm(sv["id"], LABEL) / 2 + 0.6) * bp_per_mm
            level = _free_level(placed, mid, half)
            placed.append((mid, half, level))
            ax_sv.text(mid, 0.85 + 1.0 * level, sv["id"], ha="center", va="top", fontsize=LABEL, color=SV_RED)
        levels = max([lv for _, _, lv in placed], default=0)
        ax_sv.set_ylim(1.9 + levels, -0.1)
        ax_sv.set_axis_off()
    gene_bottom = sv_bottom - bands["genes"]
    last = None
    if bands["genes"]:
        ax_gene = _inset(host, left, gene_bottom, plot_w, bands["genes"], sharex=ax)
        draw_genes(ax_gene, m["genes"], start, end, bp_per_mm)
        last = ax_gene
    axis_ax = last or _inset(host, left, gene_bottom, plot_w, 0.01, sharex=ax)
    axis_ax.spines["bottom"].set_visible(True)
    axis_ax.xaxis.set_visible(True)
    axis_ax.tick_params(axis="x", labelbottom=True, labelsize=TICK, length=2.5, width=0.6, pad=1.5)
    _kb_axis(axis_ax, f"Position on {_contig_label(m.get('reference'))} (kb)")
    key = [
        Patch(facecolor=MUTED, label="present (colour = group)"),
        Patch(facecolor=ABSENT, label="absent"),
    ]
    if legend:
        legend_handles = list(key)
        for col in cols:
            legend_handles.append(Patch(facecolor="none", edgecolor="none", label=col))
            for cat, color in m["palette"][col].items():
                count = sum(1 for r in rows if r["annotations"][col] == cat)
                if count:
                    legend_handles.append(Patch(facecolor=color, label=f"{cat} ({count})"))
        lg = host.legend(handles=legend_handles, loc="upper left",
                         bbox_to_anchor=((w - legend_mm + 1.0) / w, (h - bands["header"]) / h),
                         frameon=False, fontsize=LABEL, handlelength=0.9, handleheight=0.8, borderaxespad=0)
        for text in lg.get_texts():
            if text.get_text() in cols:
                text.set_fontweight("bold")
        return []
    handles = [Patch(facecolor=colors[g], label=g) for g in m["groups"]]
    handles.append(Patch(facecolor=ABSENT, label="absent"))
    for col in cols:
        for cat, color in m["palette"][col].items():
            if any(r["annotations"][col] == cat for r in rows):
                handles.append(Patch(facecolor=color, label=f"{col}: {cat}"))
    return handles


def draw_genes(ax, genes, start: int, end: int, bp_per_mm: float) -> None:
    """Gene arrows on a [-1, 1] track; italic labels on stacked levels (no overlap)."""
    from matplotlib.patches import Polygon

    span = max(1, end - start + 1)
    head = 1.2 * bp_per_mm
    ax.axhline(0, color=MUTED, lw=0.5, zorder=1)
    placed: list[tuple[float, float, int]] = []
    for gene in genes:
        a, b = max(start, gene["start"]), min(end, gene["end"])
        tip = min(head, (b - a) * 0.6)
        y0, y1 = -0.3, 0.3
        if gene["strand"] >= 0:
            pts = [(a, y0), (b - tip, y0), (b, 0), (b - tip, y1), (a, y1)]
        else:
            pts = [(b, y0), (a + tip, y0), (a, 0), (a + tip, y1), (b, y1)]
        ax.add_patch(Polygon(pts, closed=True, facecolor=gene["color"], edgecolor=INK, lw=0.35, zorder=3))
        mid = (a + b) / 2
        half = _text_mm(gene["name"], LABEL) / 2 * bp_per_mm + 0.4 * bp_per_mm
        level = _free_level(placed, mid, half)
        placed.append((mid, half, level))
        ax.text(mid, 0.45 + 0.75 * level, gene["name"], ha="center", va="bottom", fontsize=LABEL,
                fontstyle="italic", color=INK, zorder=4)
    levels = max([lv for _, _, lv in placed], default=0)
    ax.set_ylim(-0.5, 1.4 + 0.75 * levels)
    ax.set_xlim(start, start + span - 1)
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelbottom=False)


# ---------------------------------------------------------------------------
# sizes, stand-alone rendering, panels
# ---------------------------------------------------------------------------


def height_mm(kind: str, m) -> float:
    width = figure_width_mm(m["width"])
    if m.get("height"):
        return min(MAX_HEIGHT_MM, float(m["height"]))
    if kind == "plot_fst_scan" or kind == "plot_pi_ratio":
        return 58.0
    if kind == "plot_fst_pi_stack":
        return 105.0
    if kind == "plot_sv_schematic":
        return min(MAX_HEIGHT_MM, len(m["rows"]) * float(m.get("row_height") or ROW_MM) + 20.0)
    if kind == "plot_sv_frequency":
        lines = frequency_caption_lines(m, width - 12.0).count("\n") + 1 if m.get("caption") else 0
        return 50.0 + lines * CAPTION_LINE_MM
    if kind == "plot_presence_tracks":
        return track_layout(m)["height"]
    return width * 0.4


DRAWERS: Mapping[str, Callable[[Any, Mapping[str, Any], bool], list]] = {
    "plot_fst_scan": draw_fst,
    "plot_pi_ratio": draw_pi,
    "plot_fst_pi_stack": draw_stack,
    "plot_sv_schematic": draw_schematic,
    "plot_sv_frequency": draw_frequency,
    "plot_presence_tracks": draw_tracks,
}


def aspect(kind: str, m) -> float:
    """Height / width for compose_figure; tracks keep >= 1.8 mm rows in a half-width cell."""
    width = figure_width_mm(m["width"])
    value = height_mm(kind, m) / width
    if kind == "plot_presence_tracks":
        compact = len(m["rows"]) * 1.8 + sum(track_bands(m).values())
        value = max(value, compact / (width / 2))
    return value


def render(kind: str, path: Path, m, sidecar: Callable[[Path], None]) -> Path:
    """Stand-alone figure at printed size (``width`` x the kind's height)."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    width = figure_width_mm(m["width"])
    height = height_mm(kind, m)
    with publication_style():
        fig = plt.figure(figsize=(width * MM, height * MM))
        try:
            host = fig.add_axes((0, 0, 1, 1))
            DRAWERS[kind](host, m, True)
            fig.savefig(path, dpi=m.get("dpi") or 600, bbox_inches="tight", pad_inches=0.03)
        finally:
            plt.close(fig)
    sidecar(path)
    return path


def panel_header(kind: str, m) -> str:
    return str(m.get("title") or "")


__all__ = ["DRAWERS", "aspect", "draw_genes", "height_mm", "panel_header", "render", "write_tsv"]
