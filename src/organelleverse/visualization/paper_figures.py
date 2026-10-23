"""Standard comparative figures of organelle-genome papers.

Four figures that plastome/mitogenome papers draw by hand today, in the OGDraw
map's style (same functional-class palette and LSC/IR/SSC colours, fonts that
scale with the figure size):

- :func:`plot_ir_boundaries` — IRscope-style LSC/IRb/SSC/IRa junction panel
- :func:`plot_nucleotide_diversity` — sliding-window π with region band and peak genes
- :func:`plot_repeat_summary` — SSR type/motif and dispersed-repeat length classes
- :func:`plot_gene_presence` — species x gene presence/pseudogene/absence matrix

Like every OrganelleVerse plot they are compute-only: the call reads the
inputs, computes everything it draws into ``metrics`` and returns an
:class:`OrganellePlot`; ``ov.write(plot, "fig.svg")`` renders PNG/SVG/PDF from
those metrics alone and writes the drawn numbers to a TSV beside the image.
A serialized result (capability call, JSON round trip) therefore renders
identically through :func:`restore_paper_figure`.

Figures are drawn at their printed size with :mod:`.style` (Nature-family
widths ``"single"``/``"onehalf"``/``"double"`` or millimetres, 7 pt body, 6 pt
ticks, 6-6.5 pt gene labels, 8 pt panel titles, nothing below 5 pt, no
in-figure title, at most 170 mm tall). Each plot also carries a
:class:`~.plot_object.PanelDrawer`, so :func:`~.compose.compose_figure` can set
it as one lettered panel of a composite.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from itertools import accumulate, pairwise
from pathlib import Path
from typing import Any

from ..core.errors import OrganelleInputError, OrganelleParameterError
from .features import annotation_rows, record_display_name, resolve_ir_regions
from .ogdraw import GENE_COLORS, REGION_COLORS, classify_gene, legend_rows, record_organelle
from .plot_object import OrganellePlot, PanelDrawer, plot_result
from .style import (
    FONT_FAMILY,
    FONT_PT,
    MAX_HEIGHT_MM,
    add_panel_label,
    publication_style,
    width_inches,
)

#: Okabe-Ito colours for samples (colour-blind safe), then a muted extension.
SAMPLE_COLORS = (
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#999933",
    "#882255", "#44AA99", "#332288", "#AA4499", "#117733",
)
_MOTIF_COLORS = (
    "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948", "#B07AA1",
    "#FF9DA7", "#9C755F", "#86BCB6", "#D4A6C8", "#A0CBE8",
)
_OTHER_COLOR = "#C8C8C8"
#: Ranges and spacers in labels ("30-39", "trnH-psbA") use the typographic en dash.
EN_DASH = chr(0x2013)

JUNCTIONS = (
    ("JLB", "LSC", "IRb"),
    ("JSB", "IRb", "SSC"),
    ("JSA", "SSC", "IRa"),
    ("JLA", "IRa", "LSC"),
)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _paths(value, label: str = "genbank_path") -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    paths = [Path(p) for p in value or ()]
    if not paths:
        raise OrganelleParameterError(
            code=f"visualization.paper.{label}", message=f"{label} needs at least one file."
        )
    for path in paths:
        if not path.is_file():
            raise OrganelleInputError(
                code="visualization.paper.missing_file", message=f"File not found: {path}"
            )
    return paths


def _records(path: Path) -> list[Any]:
    from Bio import SeqIO  # type: ignore

    try:
        records = list(SeqIO.parse(str(path), "genbank"))
    except (OSError, ValueError) as exc:
        raise OrganelleInputError(code="visualization.paper.genbank", message=f"{path}: {exc}") from exc
    if not records:
        raise OrganelleInputError(
            code="visualization.paper.genbank", message=f"No GenBank records in {path}."
        )
    return records


def _single_record(path: Path):
    records = _records(path)
    if len(records) != 1:
        raise OrganelleInputError(
            code="visualization.paper.single_record",
            message=f"{path} holds {len(records)} records; one circular genome per file is required.",
        )
    return records[0]


def _names(names, count: int, defaults: list[str]) -> list[str]:
    if names is None:
        return defaults
    names = [str(n) for n in names]
    if len(names) != count:
        raise OrganelleParameterError(
            code="visualization.paper.names",
            message=f"names has {len(names)} entries for {count} inputs.",
        )
    return names


def _loci(record, organelle: str) -> list[dict[str, Any]]:
    """Gene spans (0-based start, end exclusive) for flank lookups.

    Cis genes span their introns, as IRscope draws them; a trans-spliced gene
    (``/trans_splicing``, mixed strands, or a split ``gene`` join such as the
    ndhH fragment NCBI writes across a wheat IR/SSC junction) contributes each
    part on its own, and so does a join across the origin, so no locus
    pretends to cover the space between its parts. tRNAs/rRNAs without a
    ``/gene`` are named from ``/product``; other loci without one
    (locus-tag-only ORFs) are left out.
    """
    length = len(record)
    seen: set[tuple] = set()
    loci = []
    for row in annotation_rows(record, organelle, strict=False):
        feature = record.features[row["feature_index"]]
        name = feature_gene_name(feature)
        if name is None:
            continue
        if not feature.qualifiers.get("gene"):
            name = _gene_key(name, True)[1]
            row = {**row, "gene": name, "category": classify_gene(name, organelle)}
        exons = row["exons"]
        lo, hi = min(e["start"] for e in exons), max(e["end"] for e in exons)
        split = (
            row["splicing"] == "trans"
            or "trans_splicing" in feature.qualifiers
            or len({e["strand"] for e in exons}) > 1
            or hi - lo + 1 > length / 2
        )
        if split:
            parts = [(e["start"] - 1, e["end"], e["strand"]) for e in exons]
        else:
            parts = [(lo - 1, hi, exons[0]["strand"])]
        for start, end, strand in parts:
            key = (row["gene"], start, end, strand)
            if key in seen:
                continue
            seen.add(key)
            loci.append(
                {"gene": row["gene"], "category": row["category"], "start": start, "end": end,
                 "strand": -1 if strand == -1 else 1}
            )
    return loci


def _write_tsv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _natural_key(name: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


_PT_MM = 25.4 / 72
#: Gene names in labels: FONT_PT["label"] (6 pt) is the dense-label size; names
#: that have the room are set half a point larger, as on the OGDraw maps.
LABEL_PT = 6.5


def _text_mm(text: str, size_pt: float, italic: bool = False) -> float:
    """Rendered width of ``text`` in millimetres, measured with the figure font."""
    return _text_pt(text, italic) * size_pt / 10 * _PT_MM


@lru_cache(maxsize=4096)
def _text_pt(text: str, italic: bool) -> float:
    """Advance width of ``text`` at 10 pt in the publication font, in points."""
    from matplotlib.font_manager import FontProperties  # type: ignore
    from matplotlib.textpath import TextPath  # type: ignore

    if not text.strip():
        return 2.8 * len(text)
    prop = FontProperties(family=_figure_font(), style="italic" if italic else "normal")
    # Trailing and leading spaces have no ink; bracket them so they still count.
    return (TextPath((0, 0), f"|{text}|", size=10, prop=prop).get_extents().width
            - TextPath((0, 0), "||", size=10, prop=prop).get_extents().width)


@lru_cache(maxsize=1)
def _figure_font() -> str:
    """The first installed family of the publication font list (as matplotlib will pick)."""
    from matplotlib import font_manager  # type: ignore

    for family in FONT_FAMILY:
        try:
            font_manager.findfont(font_manager.FontProperties(family=family),
                                  fallback_to_default=False)
        except ValueError:
            continue
        return family
    return "DejaVu Sans"


def _wrap(text: str, size_pt: float, width_mm: float) -> list[str]:
    lines: list[str] = []
    for word in text.split():
        if lines and _text_mm(f"{lines[-1]} {word}", size_pt) <= width_mm:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    return lines


def _axes_mm(ax) -> tuple[float, float]:
    box = ax.get_position()
    fig = ax.figure
    return box.width * fig.get_figwidth() * 25.4, box.height * fig.get_figheight() * 25.4


def _mm_axes(ax, width_mm: float, height_mm: float) -> None:
    """Data units are millimetres; equal aspect keeps them so inside any panel box."""
    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, height_mm)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N")
    ax.axis("off")


def _tint(color: str, amount: float = 0.55) -> str:
    """``color`` mixed with white (hatch patterns are avoided: PDF audits misread them)."""
    from matplotlib.colors import to_hex, to_rgb  # type: ignore

    return to_hex(tuple(c + (1 - c) * amount for c in to_rgb(color)))


def _figure(width: float, height: float):
    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    return plt, plt.figure(figsize=(width, height))


def _save(fig, plt, path: Path, dpi: int) -> None:
    """Saved at exactly the drawn size: the printed width is the figure width."""
    try:
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)


def _render_mm(path: Path, m: Mapping[str, Any], width_mm: float, height_mm: float, draw) -> None:
    """One full-figure axes in millimetre units (schematic figures)."""
    with publication_style():
        plt, fig = _figure(width_mm / 25.4, height_mm / 25.4)
        draw(fig.add_axes((0, 0, 1, 1)))
        _save(fig, plt, path, int(m["dpi"]))


def _render_axes(path, m, width_mm, height_mm, draw) -> None:
    """A full-figure axes the drawer insets itself, exactly as it does inside a panel box."""
    with publication_style():
        plt, fig = _figure(width_mm / 25.4, height_mm / 25.4)
        draw(fig.add_axes((0, 0, 1, 1)))
        _save(fig, plt, path, int(m["dpi"]))


def _inset_mm(ax, left: float, right: float, bottom: float, top: float) -> None:
    """Shrink ``ax`` by millimetre margins so its ticks and labels fit its original box."""
    fig = ax.figure
    box = ax.get_position()
    fw, fh = fig.get_figwidth() * 25.4, fig.get_figheight() * 25.4
    ax.set_position((box.x0 + left / fw, box.y0 + bottom / fh,
                     box.width - (left + right) / fw, box.height - (bottom + top) / fh))


def _category_handles(categories, organelle: str) -> list[Any]:
    from matplotlib.patches import Patch  # type: ignore

    labels = dict(legend_rows(organelle))
    return [Patch(facecolor=GENE_COLORS.get(c, GENE_COLORS["other"]), edgecolor="black",
                  linewidth=0.4, label=labels.get(c, c)) for c in categories]


def _wrapped_spans(start: int, length: int, total: int) -> list[tuple[int, int]]:
    """A region (0-based start, length) as one or two in-genome spans."""
    end = start + length
    if end <= total:
        return [(start, end)]
    return [(start, total), (0, end - total)]


# ---------------------------------------------------------------------------
# 1. IR boundaries (IRscope style)
# ---------------------------------------------------------------------------


def plot_ir_boundaries(
    genbank_path: Sequence[str | Path],
    *,
    names: Sequence[str] | None = None,
    window: int = 1500,
    genes_per_side: int = 1,
    width: str | float = "double",
    dpi: int = 600,
) -> OrganellePlot:
    """IRscope-style junction panel: one row per plastome, LSC/IRb/SSC/IRa side by side.

    ``genbank_path`` lists one single-record plastid GenBank per genome (≥ 2
    for a comparison). The quadripartite spans come from the circular IR
    detector shared with the OGDraw/structure maps
    (:func:`~organelleverse.visualization.features.resolve_ir_regions`), so
    every map of the same file shows the same boundaries. A genome without a
    detectable IR (papilionoid legumes, many conifers) is drawn as one
    single-copy bar and flagged, never failed.

    At each junction (JLB = LSC/IRb, JSB = IRb/SSC, JSA = SSC/IRa,
    JLA = IRa/LSC) the genes crossing it, or else the ``genes_per_side``
    nearest named genes on each side within ``window`` bp (a crossing gene
    counts as the nearest on both sides), are taken from the annotation
    (no gene list is hard-coded: rps19, rpl2, rpl22, ycf1, ndhF, trnH, trnN …
    appear wherever the file puts them). A crossing gene reports the bp on
    each side; a flanking gene reports its gap to the junction. Bars are not
    to scale; genes are drawn to scale inside ``±window`` bp of each junction.

    ``metrics["genomes"]`` holds region lengths, junction coordinates (the
    1-based last base of the left region) and every drawn gene; ``ov.write``
    also writes ``<stem>.junctions.tsv``.
    """
    if window < 100:
        raise OrganelleParameterError(
            code="visualization.ir_boundaries.window", message="window must be at least 100 bp."
        )
    if genes_per_side < 0:
        raise OrganelleParameterError(
            code="visualization.ir_boundaries.genes_per_side",
            message="genes_per_side must be zero or positive.",
        )
    paths = _paths(genbank_path)
    records = [_single_record(path) for path in paths]
    labels = _names(
        names, len(records), [record_display_name(r, p.stem) for r, p in zip(records, paths, strict=True)]
    )
    genomes = []
    for path, record, label in zip(paths, records, labels, strict=True):
        length = len(record)
        regions = resolve_ir_regions(record, "plastid", None)
        loci = _loci(record, "plastid")
        junctions = []
        for jid, left, right in JUNCTIONS if regions else ():
            position = regions[right]["start"]
            junctions.append(
                {
                    "id": jid,
                    "left": left,
                    "right": right,
                    "position": position or length,
                    "genes": _junction_genes(loci, position, length, window, genes_per_side),
                }
            )
        genomes.append(
            {
                "name": label,
                "file": path.name,
                "accession": record.id,
                "length": length,
                "ir_detected": bool(regions),
                "regions": {k: regions[k] for k in ("LSC", "IRb", "SSC", "IRa")} if regions else {},
                "junctions": junctions,
            }
        )
    metrics = {
        "genomes": genomes,
        "window": window,
        "genes_per_side": genes_per_side,
        "width_mm": round(width_inches(width) * 25.4, 2),
        "dpi": dpi,
        "coordinate_system": "regions: 0-based start + length; junction position: 1-based last base "
        "of the left region; gene start/end: 1-based inclusive",
    }
    metrics["height_mm"] = round(_ir_layout(metrics, metrics["width_mm"])["height"], 2)
    flags = ("ir_not_detected",) if not all(g["ir_detected"] for g in genomes) else ()
    return _with_panel(plot_result(
        "plot_ir_boundaries",
        lambda path: _render_ir_boundaries(Path(path), metrics),
        organelle="plastid",
        method="irscope_style",
        metrics=metrics,
        flags=flags + _height_flag(metrics),
        summary=f"IR boundary panel for {len(genomes)} plastome(s) prepared.",
    ), metrics)


def _junction_genes(loci, position: int, length: int, window: int, per_side: int) -> list[dict]:
    crossing, left, right = [], [], []
    half = length // 2
    for locus in loci:
        d0 = (locus["start"] - position + half) % length - half
        d1 = d0 + (locus["end"] - locus["start"])
        row = {
            "gene": locus["gene"],
            "category": locus["category"],
            "strand": locus["strand"],
            "start": locus["start"] + 1,
            "end": locus["end"],
            "offset_start": d0,
            "offset_end": d1,
        }
        if d0 < 0 < d1:
            crossing.append({**row, "relation": "crosses", "left_bp": -d0, "right_bp": d1})
        elif d1 <= 0 and -d1 <= window:
            left.append({**row, "relation": "left", "distance": -d1})
        elif d0 >= 0 and d0 <= window:
            right.append({**row, "relation": "right", "distance": d0})
    left.sort(key=lambda r: (r["distance"], r["gene"]))
    right.sort(key=lambda r: (r["distance"], r["gene"]))
    crossing.sort(key=lambda r: r["offset_start"])
    # A gene across the junction is the nearest gene on both sides (IRscope's rule).
    flank = max(0, per_side - len(crossing))
    return crossing + left[:flank] + right[:flank]


def _ir_fonts() -> dict[str, float]:
    return {
        "name": FONT_PT["body"], "len": LABEL_PT, "gene": LABEL_PT, "num": LABEL_PT,
        "region": LABEL_PT, "head": FONT_PT["title"], "sub": LABEL_PT, "note": LABEL_PT,
    }


_IR_NOTE = (
    "Region bars are not to scale; genes are drawn to scale within ±{window:,} bp of each "
    "junction. Numbers beside a junction give bp from the junction (a crossing gene: bp on "
    "each side). Plus-strand genes above, minus-strand genes below each bar."
)


def _ir_layout(m: Mapping[str, Any], width_mm: float) -> dict[str, Any]:
    """Millimetre layout of the IR panel at ``width_mm`` with fixed label layers.

    Every row has the same height: from the bar outwards, a box layer, a number
    layer (bp sit immediately left and right of the junction line) and a name
    layer, above the bar for plus-strand genes and mirrored below for minus.
    """
    f = _ir_fonts()
    window = int(m["window"])
    name_w = max(
        max(_text_mm(g["name"], f["name"], True) for g in m["genomes"]),
        max(_text_mm(f"{g['length']:,} bp", f["len"]) for g in m["genomes"]),
    ) + 2.5
    x0, x1 = name_w, width_mm - 0.5
    unit = (x1 - x0) / 4  # half LSC, IRb, SSC, IRa, half LSC
    layer = max(f["num"], f["gene"]) * 1.35 * _PT_MM
    geometry = {"bar_h": 3.4, "gap": 0.3, "box_h": 1.9, "layer": layer}
    side = geometry["gap"] + geometry["box_h"] + 0.2 + 2 * layer
    row_h = geometry["bar_h"] + 2 * side + 2.4
    note = _wrap(_IR_NOTE.format(window=window), f["note"], width_mm - 1.0)
    head_h = 7.5
    note_h = len(note) * f["note"] * 1.35 * _PT_MM + 1.5
    return {
        "fonts": f, "x0": x0, "x1": x1, "unit": unit, "window": window,
        "xs": [x0 + unit * (0.5 + k) for k in range(4)], "bp_mm": 0.42 * unit / window,
        "row_h": row_h, "side": side, "note": note, "head_h": head_h, "note_h": note_h,
        "width": width_mm, "height": head_h + row_h * len(m["genomes"]) + note_h, **geometry,
    }


def _junction_sides(junction) -> dict[int, dict[str, Any]]:
    """Per strand: the crossing gene(s), else the nearest gene left and right, as drawn."""
    sides: dict[int, dict[str, Any]] = {}
    for gene in junction["genes"]:
        strand = 1 if gene["strand"] != -1 else -1
        side = sides.setdefault(strand, {"crosses": [], "left": None, "right": None})
        if gene["relation"] == "crosses":
            side["crosses"].append(gene)
        elif side[gene["relation"]] is None:
            side[gene["relation"]] = gene
    for side in sides.values():
        if side["crosses"]:
            side["left"] = side["right"] = None
    return sides


def _draw_ir_boundaries(ax, m: Mapping[str, Any], legend: bool) -> list[Any]:
    from matplotlib.patches import Rectangle  # type: ignore

    width_mm = _axes_mm(ax)[0]
    lay = _ir_layout(m, width_mm)
    f = lay["fonts"]
    height = lay["height"]
    _mm_axes(ax, width_mm, height)
    y_head = height - lay["head_h"] / 2
    for xj, (jid, left, right) in zip(lay["xs"], JUNCTIONS, strict=True):
        ax.text(xj, y_head + 1.3, jid, ha="center", va="center", fontsize=f["head"],
                fontweight="bold")
        ax.text(xj, y_head - 1.6, f"{left}/{right}", ha="center", va="center", fontsize=f["sub"],
                color="0.35")
    seg_x = [lay["x0"], *lay["xs"], lay["x1"]]
    categories: dict[str, None] = {}
    bar = {"lw": 0.4, "edgecolor": "0.35", "zorder": 1}
    for index, genome in enumerate(m["genomes"]):
        yc = height - lay["head_h"] - index * lay["row_h"] - 1.2 - lay["side"] - lay["bar_h"] / 2
        ax.text(0.3, yc + 1.3, genome["name"], fontsize=f["name"], fontstyle="italic",
                va="center", ha="left")
        ax.text(0.3, yc - 1.6, f"{genome['length']:,} bp", fontsize=f["len"], va="center",
                ha="left", color="0.3")
        if genome["ir_detected"]:
            for i, region in enumerate(("LSC", "IRb", "SSC", "IRa", "LSC")):
                a, b = seg_x[i], seg_x[i + 1]
                ax.add_patch(Rectangle((a, yc - lay["bar_h"] / 2), b - a, lay["bar_h"],
                                       facecolor=REGION_COLORS[region], **bar))
                length = genome["regions"][region]["length"]
                for label in ((region,) if i == 4 else
                              (f"{region} {length:,} bp", f"{length:,} bp", region)):
                    if _text_mm(label, f["region"]) <= b - a - 1.0:
                        ax.text((a + b) / 2, yc, label, ha="center", va="center",
                                fontsize=f["region"], zorder=2)
                        break
        else:
            ax.add_patch(Rectangle((lay["x0"], yc - lay["bar_h"] / 2), lay["x1"] - lay["x0"],
                                   lay["bar_h"], facecolor="#e6e6e6", **bar))
            ax.text((lay["x0"] + lay["x1"]) / 2, yc, "No inverted repeat detected",
                    ha="center", va="center", fontsize=f["region"], color="0.3", zorder=2)
        reach = lay["bar_h"] / 2 + lay["gap"] + lay["box_h"]
        for xj, junction in zip(lay["xs"], genome["junctions"], strict=False):
            # The junction line crosses the bar and the box layers, never a label layer.
            ax.plot([xj, xj], [yc - reach, yc + reach], color="0.35", lw=0.4, ls=(0, (2, 1.2)),
                    zorder=3.5)
            for strand, side in _junction_sides(junction).items():
                for gene in (*side["crosses"], side["left"], side["right"]):
                    if gene is not None:
                        categories[gene["category"]] = None
                        _draw_junction_box(ax, xj, gene, strand, yc, lay)
                _draw_junction_labels(ax, xj, side, strand, yc, lay)
    for i, line in enumerate(lay["note"]):
        ax.text(0.3, lay["note_h"] - 1.0 - i * f["note"] * 1.35 * _PT_MM, line,
                fontsize=f["note"], color="0.35", va="top", ha="left")
    return [] if legend else _category_handles(categories, "plastid")


def _draw_junction_box(ax, xj, gene, strand, yc, lay):
    from matplotlib.patches import Rectangle  # type: ignore

    window, bp_mm, box_h = lay["window"], lay["bp_mm"], lay["box_h"]
    a = xj + max(gene["offset_start"], -window) * bp_mm
    b = max(xj + min(gene["offset_end"], window) * bp_mm, a + 0.25)
    y0 = (yc + lay["bar_h"] / 2 + lay["gap"] if strand == 1
          else yc - lay["bar_h"] / 2 - lay["gap"] - box_h)
    color = GENE_COLORS.get(gene["category"], GENE_COLORS["other"])
    ax.add_patch(Rectangle((a, y0), b - a, box_h, facecolor=color, edgecolor="none", zorder=3))
    edge = {"color": "0.2", "lw": 0.4, "zorder": 4, "solid_capstyle": "butt"}
    ax.plot([a, b], [y0, y0], **edge)
    ax.plot([a, b], [y0 + box_h, y0 + box_h], **edge)
    # A clipped end (the gene continues beyond the drawn window) gets no closing edge.
    if gene["offset_start"] >= -window:
        ax.plot([a, a], [y0, y0 + box_h], **edge)
    if gene["offset_end"] <= window:
        ax.plot([b, b], [y0, y0 + box_h], **edge)
    if gene["relation"] != "crosses":
        near = b if gene["relation"] == "left" else a
        if abs(xj - near) > 0.3:
            mid = y0 + box_h / 2
            ax.annotate("", xy=(xj, mid), xytext=(near, mid),
                        arrowprops={"arrowstyle": "<->", "lw": 0.4, "color": "0.25",
                                    "shrinkA": 0, "shrinkB": 0, "mutation_scale": 4},
                        zorder=4)


def _draw_junction_labels(ax, xj, side, strand, yc, lay):
    """bp beside the junction line on the number layer, names on the name layer."""
    f = lay["fonts"]
    base = lay["bar_h"] / 2 + lay["gap"] + lay["box_h"] + 0.2
    y_num = yc + strand * (base + lay["layer"] / 2)
    y_name = yc + strand * (base + lay["layer"] * 1.5)
    gap = 0.6
    italic = {"fontsize": f["gene"], "fontstyle": "italic", "va": "center", "zorder": 5}
    number = {"fontsize": f["num"], "va": "center", "zorder": 5}
    if side["crosses"]:
        first = side["crosses"][0]
        ax.text(xj - gap, y_num, f"{first['left_bp']:,} bp", ha="right", **number)
        ax.text(xj + gap, y_num, f"{first['right_bp']:,} bp", ha="left", **number)
        ax.text(xj, y_name, "/".join(g["gene"] for g in side["crosses"]), ha="center", **italic)
        return
    if side["left"] is not None:
        ax.text(xj - gap, y_num, f"{side['left']['distance']:,} bp", ha="right", **number)
        ax.text(xj - gap, y_name, side["left"]["gene"], ha="right", **italic)
    if side["right"] is not None:
        ax.text(xj + gap, y_num, f"{side['right']['distance']:,} bp", ha="left", **number)
        ax.text(xj + gap, y_name, side["right"]["gene"], ha="left", **italic)


def _render_ir_boundaries(path: Path, m: Mapping[str, Any]) -> Path:
    width_mm = float(m["width_mm"])
    height_mm = _ir_layout(m, width_mm)["height"]
    _render_mm(path, m, width_mm, height_mm, lambda ax: _draw_ir_boundaries(ax, m, True))
    _write_tsv(
        path.with_suffix(".junctions.tsv"),
        ["genome", "accession", "junction", "position", "gene", "strand", "start", "end",
         "relation", "left_bp", "right_bp", "distance"],
        [
            {"genome": g["name"], "accession": g["accession"], "junction": j["id"],
             "position": j["position"], **gene}
            for g in m["genomes"]
            for j in g["junctions"]
            for gene in j["genes"]
        ],
    )
    return path


# ---------------------------------------------------------------------------
# 2. Nucleotide diversity (π) sliding window
# ---------------------------------------------------------------------------


def plot_nucleotide_diversity(
    alignment_fasta: str | Path | None = None,
    *,
    windows: Sequence[Mapping[str, int | float | str | bool | None]] | None = None,
    genbank_path: str | Path | None = None,
    reference_index: int = 0,
    window_size: int = 600,
    step: int = 200,
    top_n: int = 5,
    threshold: float | None = None,
    show_ir: bool | None = None,
    width: str | float = "double",
    height_mm: float = 60.0,
    dpi: int = 600,
) -> OrganellePlot:
    """Sliding-window π along the genome, with the LSC/IR/SSC band and peak genes.

    π comes from :func:`organelleverse.diversity.compute_nucleotide_diversity`,
    whose input is an existing **multiple sequence alignment** (equal-length
    FASTA; it does not align). Pass ``alignment_fasta`` to compute it here
    (DnaSP's 600/200 bp window/step by default), or pass ``windows`` already
    computed by ``compute_nucleotide_diversity(...)["windows"]`` or
    ``sliding_window_diversity(...).metrics["windows"]``.

    The x axis is the reference sequence (row ``reference_index`` of the
    alignment, gaps removed) whenever it can be: windows that carry
    ``ref_start``/``ref_end`` use them, otherwise alignment columns are mapped
    through the reference row of ``alignment_fasta``. ``genbank_path`` is that
    reference's annotation: it supplies the region band (same IR detector as
    the OGDraw map; ``show_ir`` as there) and the gene names of the peaks, and
    must have the reference's ungapped length.

    Peaks: with ``threshold`` every run of windows with π ≥ threshold is
    labelled (dashed line drawn); otherwise the ``top_n`` highest
    non-overlapping windows. A peak is named by the gene it overlaps most, or
    ``geneA-geneB`` when it lies mainly in a spacer. ``ov.write`` also writes
    ``<stem>.windows.tsv``.
    """
    if top_n < 0 or (threshold is not None and threshold < 0):
        raise OrganelleParameterError(
            code="visualization.diversity.peaks", message="top_n and threshold must be non-negative."
        )
    overall_pi = None
    n_sequences = None
    if windows is None:
        if alignment_fasta is None:
            raise OrganelleParameterError(
                code="visualization.diversity.input",
                message="Pass alignment_fasta (an existing alignment) or precomputed windows.",
            )
        from ..diversity.diversity_core import compute_nucleotide_diversity

        computed = compute_nucleotide_diversity(alignment_fasta, window_size=window_size, step=step)
        windows = computed["windows"]
        overall_pi, n_sequences = computed["pi"], computed["n_sequences"]
    rows = [dict(w) for w in windows]
    if not rows:
        raise OrganelleInputError(
            code="visualization.diversity.empty",
            message="No windows: the alignment is shorter than one window or has < 2 sequences.",
        )
    reference_name, reference_length = None, None
    if all(r.get("ref_start") is not None and r.get("ref_end") is not None for r in rows):
        coordinate = "reference"
        for r in rows:
            r["x_start"], r["x_end"] = int(r["ref_start"]), int(r["ref_end"])
    elif alignment_fasta is not None:
        coordinate = "reference"
        reference_name, reference_length, mapped = _map_to_reference(
            Path(alignment_fasta), reference_index, rows
        )
        rows = mapped
    else:
        coordinate = "alignment"
        for r in rows:
            r["x_start"], r["x_end"] = int(r["start"]), int(r["end"])
    regions: dict[str, Any] = {}
    loci: list[dict[str, Any]] = []
    organelle = "plastid"
    if genbank_path is not None:
        if coordinate != "reference":
            raise OrganelleParameterError(
                code="visualization.diversity.coordinates",
                message="Gene and region labels need reference coordinates: pass alignment_fasta "
                "or windows with ref_start/ref_end.",
            )
        record = _single_record(Path(genbank_path))
        if reference_length is not None and reference_length != len(record):
            raise OrganelleInputError(
                code="visualization.diversity.reference_length",
                message=f"Reference row has {reference_length:,} bases but the GenBank record has "
                f"{len(record):,}; annotate the aligned reference itself.",
            )
        organelle = record_organelle(record)
        regions = resolve_ir_regions(record, organelle, show_ir)
        loci = _loci(record, organelle)
        reference_name = record_display_name(record, record.id)
        reference_length = len(record)
    peaks = _diversity_peaks(rows, top_n, threshold, loci)
    pis = [float(r["pi"]) for r in rows]
    metrics = {
        "windows": [
            {k: r.get(k) for k in ("start", "end", "x_start", "x_end", "pi")} for r in rows
        ],
        "coordinate": coordinate,
        "reference_name": reference_name,
        "reference_length": reference_length,
        "organelle": organelle,
        "regions": regions,
        "peaks": peaks,
        "threshold": threshold,
        "top_n": top_n,
        "mean_pi": round(sum(pis) / len(pis), 6),
        "max_pi": max(pis),
        "overall_pi": overall_pi,
        "n_sequences": n_sequences,
        "window_size": window_size if alignment_fasta is not None else None,
        "step": step if alignment_fasta is not None else None,
        "width_mm": round(width_inches(width) * 25.4, 2),
        "height_mm": float(height_mm),
        "dpi": dpi,
    }
    return _with_panel(plot_result(
        "plot_nucleotide_diversity",
        lambda path: _render_nucleotide_diversity(Path(path), metrics),
        organelle=organelle,
        method="sliding_window_pi",
        metrics=metrics,
        flags=_height_flag(metrics),
        summary=f"π profile over {len(rows)} windows with {len(peaks)} labelled peak(s) prepared.",
    ), metrics)


def _map_to_reference(path: Path, index: int, rows: list[dict]) -> tuple[str, int, list[dict]]:
    from .._bio import read_fasta

    records = read_fasta(path)
    if not 0 <= index < len(records):
        raise OrganelleParameterError(
            code="visualization.diversity.reference_index",
            message=f"reference_index {index} is outside the {len(records)}-sequence alignment.",
        )
    name, row = records[index]
    called = list(accumulate(0 if c in "-." else 1 for c in row))
    mapped = []
    for r in rows:
        start, end = int(r["start"]), int(r["end"])
        before = called[start - 2] if start > 1 else 0
        last = called[min(end, len(called)) - 1]
        if last <= before:
            continue  # the window is all gap in the reference
        mapped.append({**r, "x_start": before + 1, "x_end": last})
    return name, called[-1] if called else 0, mapped


def _diversity_peaks(rows, top_n, threshold, loci) -> list[dict[str, Any]]:
    candidates: list[tuple[float, int, int]] = []  # (pi, x_start, x_end)
    if threshold is not None:
        run: list[dict] = []
        for r in [*sorted(rows, key=lambda r: r["x_start"]), None]:
            if r is not None and float(r["pi"]) >= threshold and (
                not run or r["x_start"] <= run[-1]["x_end"] + 1
            ):
                run.append(r)
                continue
            if run:
                best = max(run, key=lambda w: float(w["pi"]))
                candidates.append((float(best["pi"]), run[0]["x_start"], run[-1]["x_end"]))
            run = [r] if r is not None and float(r["pi"]) >= threshold else []
        candidates.sort(reverse=True)
        candidates = candidates[:15]
    else:
        for r in sorted(rows, key=lambda r: -float(r["pi"])):
            if len(candidates) >= top_n or float(r["pi"]) <= 0:
                break
            if any(r["x_start"] <= b and a <= r["x_end"] for _, a, b in candidates):
                continue
            candidates.append((float(r["pi"]), r["x_start"], r["x_end"]))
    peaks = []
    for pi, a, b in candidates:
        peak_rows = [r for r in rows if a <= r["x_start"] and r["x_end"] <= b]
        best = max(peak_rows, key=lambda w: float(w["pi"])) if peak_rows else None
        x = (best["x_start"] + best["x_end"]) / 2 if best else (a + b) / 2
        peaks.append({"start": a, "end": b, "x": x, "pi": pi, "label": _region_label(a, b, loci)})
    return sorted(peaks, key=lambda p: p["x"])


def _region_label(start: int, end: int, loci) -> str:
    """Gene a 1-based window overlaps most, else its flanking ``A-B`` spacer."""
    if not loci:
        return ""
    lo, hi = start - 1, end
    best, best_overlap = None, 0
    for locus in loci:
        overlap = min(hi, locus["end"]) - max(lo, locus["start"])
        if overlap > best_overlap:
            best, best_overlap = locus, overlap
    if best is not None and best_overlap >= 0.5 * (hi - lo):
        return best["gene"]
    left = max((loc for loc in loci if loc["end"] <= hi), key=lambda loc: loc["end"], default=None)
    right = min((loc for loc in loci if loc["start"] >= lo), key=lambda loc: loc["start"],
                default=None)
    if left is not None and right is not None and left["gene"] != right["gene"]:
        return f"{left['gene']}{EN_DASH}{right['gene']}"
    return best["gene"] if best is not None else ""


def _draw_nucleotide_diversity(ax, m: Mapping[str, Any], legend: bool) -> list[Any]:
    from matplotlib.patches import Rectangle  # type: ignore
    from matplotlib.ticker import FuncFormatter  # type: ignore
    from matplotlib.transforms import blended_transform_factory  # type: ignore

    _inset_mm(ax, 11.0, 2.5, 8.5, 1.5)  # ticks and axis titles stay inside the panel box
    w_mm, h_mm = _axes_mm(ax)
    rows = m["windows"]
    xs = [(r["x_start"] + r["x_end"]) / 2 for r in rows]
    ys = [float(r["pi"]) for r in rows]
    total = m["reference_length"] or max(r["x_end"] for r in rows)
    band = 3.2 / h_mm if m["regions"] else 0.0  # region band inside the top of the axes
    room = 1 - band - (0.02 if band else 0.0)
    strip = blended_transform_factory(ax.transData, ax.transAxes)
    for region, span in m["regions"].items():
        for a, b in _wrapped_spans(span["start"], span["length"], total):
            ax.add_patch(Rectangle((a, 1 - band), b - a, band, transform=strip,
                                   facecolor=REGION_COLORS[region], edgecolor="0.35", lw=0.3,
                                   clip_on=False, zorder=3))
            if _text_mm(region, LABEL_PT) < (b - a) / total * w_mm - 1:
                ax.text((a + b) / 2, 1 - band / 2, region, transform=strip, ha="center",
                        va="center", fontsize=LABEL_PT, zorder=4)
    line = "#1F4E79"
    ax.fill_between(xs, ys, color=line, alpha=0.22, lw=0, zorder=1)
    ax.plot(xs, ys, color=line, lw=0.6, zorder=2)
    ax.set_xlim(0, total)
    y_max = max(ys) or 1.0
    fs = LABEL_PT
    data_per_mm = total / w_mm
    line_frac = fs * 1.45 * _PT_MM / h_mm
    peaks = [p for p in sorted(m["peaks"], key=lambda p: -p["pi"]) if p["label"]]
    top = y_max / (room - 0.02)
    placed: list[dict[str, Any]] = []
    for _ in range(4):  # label height depends on the y range, which depends on the labels
        step = line_frac * top
        placed = []
        highest = y_max
        for peak in peaks:
            half = _text_mm(peak["label"], fs, True) * data_per_mm / 2 + 0.6 * data_per_mm
            spot = _place_label(peak, half, step, placed, xs, ys, total)
            if spot is not None:
                placed.append(spot)
                highest = max(highest, spot["y"] + step * 0.5)
        needed = highest / room
        if needed <= top * 1.001:
            break
        top = needed
    ax.set_ylim(0, top)
    for spot in placed:
        peak = spot["peak"]
        ax.plot([peak["x"], spot["x"]], [peak["pi"], spot["y"] - step * 0.45], color="0.35",
                lw=0.4, zorder=3)
        ax.text(spot["x"], spot["y"], peak["label"], ha="center", va="center", fontsize=fs,
                fontstyle="italic", zorder=4)
    if m.get("threshold") is not None:
        ax.axhline(m["threshold"], color="#B22222", lw=0.5, ls=(0, (4, 2)), zorder=1.5)
        ax.text(total, m["threshold"], f"π = {m['threshold']:g} ", color="#B22222",
                fontsize=LABEL_PT, va="bottom", ha="right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=FONT_PT["tick"])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}"))
    where = (f"Position in {m['reference_name']} (kb)" if m["coordinate"] == "reference"
             and m.get("reference_name") else
             "Reference position (kb)" if m["coordinate"] == "reference" else
             "Alignment position (kb)")
    ax.set_xlabel(where, fontsize=FONT_PT["body"], labelpad=2)
    ax.set_ylabel("Nucleotide diversity (π)", fontsize=FONT_PT["body"], labelpad=2)
    return []


def _place_label(peak, half, step, placed, xs, ys, total, tries: int = 40):
    """Lowest label spot above ``peak`` that touches no curve, label or leader line.

    Spots are tried level by level, first straight above the peak and then
    shifted left and right by one label width (the leader then slants), so a
    peak standing under a neighbour's label still gets its own label. Returns
    ``None`` only when nothing fits within ``tries`` levels.
    """
    import bisect

    shifts = (0.0, 1.1, -1.1, 2.2, -2.2)
    y = peak["pi"] + step * 0.6
    for _ in range(tries):
        for shift in shifts:
            lx = min(max(peak["x"] + shift * 2 * half, half), total - half)
            rect = (lx - half, y - step / 2, lx + half, y + step / 2)
            lo, hi = bisect.bisect_left(xs, rect[0]), bisect.bisect_right(xs, rect[2])
            curve = max(ys[max(lo - 1, 0):min(hi + 1, len(ys))], default=0.0)
            if curve + step * 0.15 > rect[1]:
                continue  # the label would sit on the curve
            leader = ((peak["x"], peak["pi"]), (lx, y - step * 0.45))
            if any(_rects_meet(rect, other["rect"]) or _segment_meets(leader, other["rect"])
                   or _segment_meets(other["leader"], rect) for other in placed):
                continue
            return {"x": lx, "y": y, "rect": rect, "leader": leader, "peak": peak}
        y += step
    return None


def _rects_meet(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _segment_meets(segment, rect) -> bool:
    """Whether a line segment touches an axis-aligned rectangle (scale-free test)."""
    (x0, y0), (x1, y1) = segment
    if max(x0, x1) < rect[0] or min(x0, x1) > rect[2] or max(y0, y1) < rect[1] or \
            min(y0, y1) > rect[3]:
        return False
    if rect[0] <= x0 <= rect[2] and rect[1] <= y0 <= rect[3]:
        return True
    corners = ((rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3]))
    sides = [(x1 - x0) * (cy - y0) - (y1 - y0) * (cx - x0) for cx, cy in corners]
    return min(sides) <= 0 <= max(sides)


def _render_nucleotide_diversity(path: Path, m: Mapping[str, Any]) -> Path:
    width_mm, height_mm = float(m["width_mm"]), float(m["height_mm"])
    _render_axes(path, m, width_mm, height_mm, lambda ax: _draw_nucleotide_diversity(ax, m, True))
    labels = {(p["start"], p["end"]): p["label"] for p in m["peaks"]}
    _write_tsv(
        path.with_suffix(".windows.tsv"),
        ["start", "end", "x_start", "x_end", "pi", "peak_label"],
        [
            {**r, "peak_label": next((label for (a, b), label in labels.items()
                                      if a <= r["x_start"] and r["x_end"] <= b), "")}
            for r in m["windows"]
        ],
    )
    return path


# ---------------------------------------------------------------------------
# 3. SSR and dispersed-repeat summary
# ---------------------------------------------------------------------------

SSR_TYPES = ("mono", "di", "tri", "tetra", "penta", "hexa")
_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def motif_class(unit: str) -> str:
    """MISA-style motif class: ``A/T``, ``AT/AT``, ``AG/CT``, ``AAT/ATT`` …

    A motif, its rotations, its reverse complement and their rotations are one
    class, written as the smallest rotation of the motif and of its complement.
    """
    unit = unit.upper()
    rc = unit.translate(_COMPLEMENT)[::-1]

    def smallest(m: str) -> str:
        return min(m[i:] + m[:i] for i in range(len(m)))

    first, second = sorted((smallest(unit), smallest(rc)))
    return f"{first}/{second}"


def _primitive(unit: str) -> bool:
    """False for a motif that is a shorter motif repeated (AA, ATAT, AATAAT)."""
    return len(unit) == 1 or unit not in (unit + unit)[1:-1]


def plot_repeat_summary(
    fasta_path: Sequence[str | Path],
    *,
    ssrs: Sequence[Sequence[Mapping[str, str | int]]] | None = None,
    repeats: Sequence[Sequence[Mapping[str, str | int | list[int]]]] | None = None,
    names: Sequence[str] | None = None,
    min_repeat_len: int = 30,
    top_motifs: int = 10,
    width: str | float = "double",
    height_mm: float = 75.0,
    dpi: int = 600,
) -> OrganellePlot:
    """SSR and dispersed-repeat statistics for one or more genomes.

    Inputs are the existing analyses, one entry per genome:
    ``compute_repeats(fa)["ssrs"]`` (exact SSRs, MISA 10/5/4/3/3/3 thresholds)
    and ``compute_multiconf(fa, min_repeat_len)["repeats"]`` (direct and
    inverted repeat pairs). ``fasta_path`` runs both here, one genome per
    FASTA file; pass ``[]`` with precomputed ``ssrs``/``repeats`` instead
    (the capability binding has no optional list of paths). Nothing is
    re-detected when results are passed.

    Panels: (a) SSR counts by unit length (mono…hexa); (b) SSR motif classes
    (``A/T``, ``AT/AT`` …, the ``top_motifs`` most frequent plus "other"),
    stacked per genome; (c) dispersed repeats by length class, forward
    (direct) solid and palindromic (inverted) in a light tint of the same
    colour. With several genomes
    every panel groups by genome in one colour per genome. The plastid IR
    itself is an inverted repeat and falls in the longest class. SSR rows whose
    unit is a shorter motif repeated (``AA``, ``ATAT``) are the same locus as
    the primitive row ``compute_repeats`` also reports, so only the primitive
    one is counted, as MISA does (``non_primitive_skipped`` per genome). ``ov.write``
    also writes ``<stem>.repeats.tsv``.
    """
    if isinstance(fasta_path, (str, Path)) or fasta_path:
        if ssrs is not None or repeats is not None:
            raise OrganelleParameterError(
                code="visualization.repeats.input",
                message="Pass fasta_path or precomputed ssrs/repeats, not both.",
            )
        from ..structure.structure_core import compute_multiconf, compute_repeats

        files = _paths(fasta_path, "fasta_path")
        ssrs = [compute_repeats(p)["ssrs"] for p in files]
        repeats = [compute_multiconf(p, min_repeat_len=min_repeat_len)["repeats"] for p in files]
        defaults = [p.stem for p in files]
    else:
        ssrs = _samples(ssrs, "ssrs", "unit")
        repeats = _samples(repeats, "repeats", "type")
        count = len(ssrs or repeats or [])
        defaults = [f"genome {i + 1}" for i in range(count)]
    if not ssrs and not repeats:
        raise OrganelleParameterError(
            code="visualization.repeats.input", message="No SSR or repeat results were given."
        )
    if ssrs and repeats and len(ssrs) != len(repeats):
        raise OrganelleParameterError(
            code="visualization.repeats.samples",
            message=f"{len(ssrs)} SSR results but {len(repeats)} repeat results.",
        )
    labels = _names(names, len(ssrs or repeats), defaults)
    bins = _length_bins(repeats, min_repeat_len) if repeats else []
    samples = []
    for i, label in enumerate(labels):
        sample: dict[str, Any] = {"name": label}
        if ssrs:
            # compute_repeats scans each unit length on its own, so an A12 run is also
            # reported as (AA)6 and (AAA)4. MISA counts primitive motifs only; every
            # non-primitive row is the same locus as its primitive one, which is kept.
            every = list(ssrs[i])
            rows = [r for r in every if _primitive(str(r["unit"]))]
            sample["non_primitive_skipped"] = len(every) - len(rows)
            types = Counter(SSR_TYPES[len(str(r["unit"])) - 1] for r in rows if 1 <= len(str(r["unit"])) <= 6)
            sample["ssr_total"] = len(rows)
            sample["ssr_types"] = {t: types.get(t, 0) for t in SSR_TYPES}
            sample["motifs"] = dict(Counter(motif_class(str(r["unit"])) for r in rows))
        if repeats:
            rows = list(repeats[i])
            sample["repeat_total"] = len(rows)
            sample["repeat_types"] = {
                kind: sum(r["type"] == kind for r in rows) for kind in ("direct", "inverted")
            }
            sample["repeat_bins"] = {
                b["label"]: {
                    kind: sum(r["type"] == kind and b["min"] <= int(r["length"]) <= b["max"]
                              for r in rows)
                    for kind in ("direct", "inverted")
                }
                for b in bins
            }
        samples.append(sample)
    motif_order: list[str] = []
    if ssrs:
        overall = Counter()
        for sample in samples:
            overall.update(sample["motifs"])
        ranked = sorted(overall, key=lambda c: (-overall[c], len(c), c))
        motif_order = ranked[:top_motifs]
        if len(ranked) > top_motifs:
            motif_order.append("other")
    metrics = {
        "samples": samples,
        "ssr_types": list(SSR_TYPES) if ssrs else [],
        "motif_order": motif_order,
        "length_bins": bins,
        "min_repeat_len": min_repeat_len if repeats else None,
        "width_mm": round(width_inches(width) * 25.4, 2),
        "height_mm": float(height_mm),
        "dpi": dpi,
    }
    return _with_panel(plot_result(
        "plot_repeat_summary",
        lambda path: _render_repeat_summary(Path(path), metrics),
        organelle="none",
        method="ssr_repeat_summary",
        metrics=metrics,
        flags=_height_flag(metrics),
        summary=f"Repeat summary for {len(samples)} genome(s) prepared.",
    ), metrics)


def _samples(value, label: str, row_key: str):
    """One list of rows per genome from a list of lists, one result dict, or one list."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return [list(value.get(label, []))]
    value = list(value)
    if value and isinstance(value[0], Mapping):
        if row_key in value[0]:
            return [value]
        return [list(v.get(label, [])) for v in value]
    return [list(v) for v in value]


def _length_bins(repeats, min_len: int) -> list[dict[str, Any]]:
    lengths = [int(r["length"]) for rows in repeats for r in rows]
    floor = (min(min_len, *lengths) if lengths else min_len) // 10 * 10
    bins = [{"label": f"{floor + 10 * i}{EN_DASH}{floor + 10 * i + 9}", "min": floor + 10 * i,
             "max": floor + 10 * i + 9} for i in range(5)]
    bins.append({"label": f"≥{floor + 50}", "min": floor + 50, "max": 10**12})
    return bins


def _draw_repeat_panels(fig, region_mm, m: Mapping[str, Any], *, letters: bool,
                        legend: bool) -> list[Any]:
    """Draw panels a-c inside ``region_mm`` = (left, bottom, width, height) of ``fig`` in mm."""
    import numpy as np  # type: ignore
    from matplotlib.patches import Patch  # type: ignore

    rx, ry, rw, rh = region_mm
    fig_w, fig_h = fig.get_figwidth() * 25.4, fig.get_figheight() * 25.4
    samples = m["samples"]
    n = len(samples)
    colors = [SAMPLE_COLORS[i % len(SAMPLE_COLORS)] for i in range(n)]
    names = [s["name"] for s in samples]
    panels = (["types", "motifs"] if m["ssr_types"] else []) + (
        ["lengths"] if m["length_bins"] else [])
    order = m["motif_order"]
    motif_cols = 3
    motif_rows = math.ceil(len(order) / motif_cols) if "motifs" in panels else 0
    line = FONT_PT["body"] * 1.45 * _PT_MM
    sample_cols = max(1, min(n, int(rw // (max(_text_mm(x, FONT_PT["body"], True)
                                                for x in names) + 6))))
    sample_rows = math.ceil(n / sample_cols) if legend and n > 1 else 0
    bottom = ry + 8.5 + motif_rows * line + (1.5 if motif_rows else 0) + sample_rows * line
    top = ry + rh - 4.5
    gap = 4.0
    lefts = {"types": 8.5, "motifs": max(_text_mm(x, FONT_PT["body"], True) for x in names) + 2.5,
             "lengths": 8.5}
    shares = {"types": 0.27, "motifs": 0.38, "lengths": 0.35}
    free = rw - sum(lefts[p] for p in panels) - gap * (len(panels) - 1) - 1.0
    total_share = sum(shares[p] for p in panels)
    x = rx
    axes = {}
    for panel in panels:
        x += lefts[panel]
        w = free * shares[panel] / total_share
        axes[panel] = fig.add_axes((x / fig_w, bottom / fig_h, w / fig_w, (top - bottom) / fig_h))
        x += w + gap
    titles = {"types": "SSR type", "motifs": "SSR motif class", "lengths": "Dispersed repeats"}
    group_w = 0.8
    for k, panel in enumerate(panels):
        ax = axes[panel]
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=FONT_PT["tick"])
        ax.set_title(titles[panel], loc="left", fontsize=FONT_PT["title"], pad=3)
        if letters:
            add_panel_label(ax, "abc"[k], x_pt=-lefts[panel] / _PT_MM + 6, y_pt=3)
        if panel == "types":
            xs = np.arange(len(m["ssr_types"]))
            bw = group_w / n
            for i, sample in enumerate(samples):
                values = [sample["ssr_types"][t] for t in m["ssr_types"]]
                bars = ax.bar(xs - group_w / 2 + bw * (i + 0.5), values, bw * 0.92,
                              color=colors[i], lw=0)
                if n <= 3:
                    for bar, v in zip(bars, values, strict=True):
                        if v:
                            ax.text(bar.get_x() + bar.get_width() / 2, v, str(v), ha="center",
                                    va="bottom", fontsize=LABEL_PT)
            ax.set_xticks(xs, m["ssr_types"])
            ax.set_ylabel("Number of SSRs", fontsize=FONT_PT["body"], labelpad=2)
        elif panel == "motifs":
            ys = np.arange(n)
            left = np.zeros(n)
            palette = {c: _MOTIF_COLORS[i % len(_MOTIF_COLORS)] for i, c in enumerate(order)}
            palette["other"] = _OTHER_COLOR
            for motif in order:
                values = np.array([
                    sum(v for c, v in s["motifs"].items() if c not in order[:-1])
                    if motif == "other" else s["motifs"].get(motif, 0)
                    for s in samples
                ])
                ax.barh(ys, values, 0.66, left=left, color=palette[motif], lw=0.25,
                        edgecolor="white", label=motif)
                left += values
            ax.set_yticks(ys, names, fontstyle="italic", fontsize=FONT_PT["body"])
            ax.invert_yaxis()
            ax.set_xlabel("Number of SSRs", fontsize=FONT_PT["body"], labelpad=2)
            height_mm = (top - bottom)
            ax.legend(loc="upper left", bbox_to_anchor=(-lefts[panel] / (free * shares[panel] / total_share),
                                                        -8.0 / height_mm),
                      ncol=motif_cols, fontsize=FONT_PT["body"], frameon=False, handlelength=1.0,
                      handleheight=0.8, columnspacing=1.0, borderaxespad=0, borderpad=0)
        else:
            labels = [b["label"] for b in m["length_bins"]]
            xs = np.arange(len(labels))
            bw = group_w / n
            for i, sample in enumerate(samples):
                direct = np.array([sample["repeat_bins"][b]["direct"] for b in labels])
                inverted = np.array([sample["repeat_bins"][b]["inverted"] for b in labels])
                pos = xs - group_w / 2 + bw * (i + 0.5)
                ax.bar(pos, direct, bw * 0.92, color=colors[i], lw=0)
                ax.bar(pos, inverted, bw * 0.92, bottom=direct, color=_tint(colors[i]), lw=0)
            ax.set_xticks(xs, labels)
            ax.set_xlabel("Repeat length (bp)", fontsize=FONT_PT["body"], labelpad=2)
            ax.set_ylabel("Number of repeat pairs", fontsize=FONT_PT["body"], labelpad=2)
            ax.legend(
                handles=[Patch(facecolor="0.35", lw=0, label="Forward (direct)"),
                         Patch(facecolor=_tint("#595959"), lw=0, label="Palindromic (inverted)")],
                fontsize=FONT_PT["body"], frameon=False, loc="upper right", handlelength=1.0,
                handleheight=0.8, borderaxespad=0.2,
            )
    handles = [Patch(facecolor=c, lw=0, label=name) for c, name in zip(colors, names, strict=True)]
    if legend and n > 1:
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(rx / fig_w, ry / fig_h),
                   ncol=sample_cols, frameon=False, handlelength=1.0, handleheight=0.8,
                   columnspacing=1.2, borderaxespad=0, borderpad=0,
                   prop={"size": FONT_PT["body"], "style": "italic"})
    return [] if legend or n < 2 else handles


def _draw_repeat_summary(ax, m: Mapping[str, Any], legend: bool) -> list[Any]:
    """Panel form: the three sub-panels share the composer's axes box."""
    fig = ax.figure
    box = ax.get_position()
    region = (box.x0 * fig.get_figwidth() * 25.4, box.y0 * fig.get_figheight() * 25.4,
              box.width * fig.get_figwidth() * 25.4, box.height * fig.get_figheight() * 25.4)
    ax.axis("off")
    return _draw_repeat_panels(fig, region, m, letters=False, legend=legend)


def _render_repeat_summary(path: Path, m: Mapping[str, Any]) -> Path:
    width_mm, height_mm = float(m["width_mm"]), float(m["height_mm"])
    with publication_style():
        plt, fig = _figure(width_mm / 25.4, height_mm / 25.4)
        _draw_repeat_panels(fig, (0.0, 0.5, width_mm - 0.5, height_mm - 0.5), m, letters=True,
                            legend=True)
        _save(fig, plt, path, int(m["dpi"]))
    rows = []
    for sample in m["samples"]:
        for t, v in sample.get("ssr_types", {}).items():
            rows.append({"genome": sample["name"], "table": "ssr_type", "class": t, "count": v})
        for c, v in sorted(sample.get("motifs", {}).items(), key=lambda kv: -kv[1]):
            rows.append({"genome": sample["name"], "table": "ssr_motif", "class": c, "count": v})
        for b, counts in sample.get("repeat_bins", {}).items():
            for kind, v in counts.items():
                rows.append({"genome": sample["name"], "table": f"repeat_{kind}", "class": b,
                             "count": v})
    _write_tsv(path.with_suffix(".repeats.tsv"), ["genome", "table", "class", "count"], rows)
    return path


# ---------------------------------------------------------------------------
# 4. Gene presence/absence matrix
# ---------------------------------------------------------------------------

#: Renamed plastid/mitochondrial genes: the older or alternative symbol → the one drawn.
GENE_SYNONYMS = {
    "pafi": "ycf3", "pafii": "ycf4", "lhba": "psbZ", "ycf9": "psbZ", "ycf10": "cemA",
    "ycf5": "ccsA", "ycf6": "petN", "pbf1": "psbN", "tatc": "mttB",
}
STATUS_ORDER = ("present", "pseudogene", "transferred", "absent")
STATUS_COLORS = {
    "present": "#2E5E7E", "pseudogene": "#A3BFD4", "transferred": "#D98B3A", "absent": "#F1F1F1",
}
_TRNA = re.compile(r"^[-_ ]?(fmet|fm|[a-z]{3}(?![a-z])|[a-z])(?:[-_ ]?\(?([acgtu]{3})\)?)?$")
_AMINO_ACIDS = {
    "ala": "A", "arg": "R", "asn": "N", "asp": "D", "cys": "C", "gln": "Q", "glu": "E",
    "gly": "G", "his": "H", "ile": "I", "leu": "L", "lys": "K", "met": "M", "phe": "F",
    "pro": "P", "ser": "S", "thr": "T", "trp": "W", "tyr": "Y", "val": "V", "fmet": "fM",
    "fm": "fM",
}
_RRNA = re.compile(r"^rrn(\d+(?:\.\d+)?)s?$", re.IGNORECASE)
_TRANSFER_NOTE = re.compile(r"transferred to (the )?nucle", re.IGNORECASE)


def _gene_key(raw: str, trna_anticodon: bool) -> tuple[str, str, bool]:
    """(match key, display name, has anticodon) for one annotated gene name."""
    name = raw.strip()
    # "trnA-UGC", "trnfM", "tRNA-Ala (UGC)", "tRNA-fMet", "tRNAI(CAT)" all name tRNAs;
    # "tRNA…" (capital RNA) is the long form, so "trnA-UGC" is not read as tRNA-UGC.
    # A suffix other than the anticodon ("trnD-ct", plastid-derived) stays its own gene.
    long_form = name[:4].upper() == "TRNA" and name[1:4] == "RNA"
    body = name[4:] if long_form else name[3:] if name[:3].lower() == "trn" else None
    trna = _TRNA.match(body.lower()) if body is not None else None
    if trna:
        code = trna.group(1)
        aa = _AMINO_ACIDS.get(code) if len(code) > 1 else code.upper()
        if aa:
            anticodon = (trna.group(2) or "").upper().replace("T", "U")
            display = f"trn{aa}-{anticodon}" if trna_anticodon and anticodon else f"trn{aa}"
            return display.lower(), display, bool(anticodon)
    rrna = _RRNA.match(name)
    if rrna:
        display = f"rrn{rrna.group(1)}"
        return display.lower(), display, False
    target = GENE_SYNONYMS.get(name.lower())
    if target:
        return target.lower(), target, False
    return name.lower(), name, False


_RRNA_PRODUCT = re.compile(r"^(\d+(?:\.\d+)?)S ribosomal RNA", re.IGNORECASE)
_ANTICODON_SEQ = re.compile(r"seq:([acgtu]{3})", re.IGNORECASE)
_ANTICODON_NOTE = re.compile(r"\(([ACGTU]{3})\)")


def feature_gene_name(feature) -> str | None:
    """``/gene``, or for a tRNA/rRNA without one the name its ``/product`` gives.

    RefSeq plastomes such as rice NC_001320 annotate most tRNAs and rRNAs with
    only ``/locus_tag`` and ``/product="tRNA-Gln"`` / ``"16S ribosomal RNA"``;
    the anticodon then comes from ``/anticodon`` or a note like ``tRNA-Gln(UUG)``.
    """
    q = feature.qualifiers
    gene = (q.get("gene") or [""])[0].strip()
    if gene:
        return gene
    product = (q.get("product") or [""])[0].strip()
    if feature.type == "tRNA" and product.lower().startswith("trna-"):
        found = _ANTICODON_SEQ.search(" ".join(q.get("anticodon", []))) or _ANTICODON_NOTE.search(
            " ".join(q.get("note", []))
        )
        return f"{product} ({found.group(1).upper()})" if found else product
    if feature.type == "rRNA":
        rrna = _RRNA_PRODUCT.match(product)
        if rrna:
            return f"rrn{rrna.group(1)}"
    return None


def _genome_genes(records) -> list[tuple[str, str, bool, bool]]:
    """(raw name, feature kind, pseudo, transfer note) for every named gene-level feature."""
    out = []
    for record in records:
        for feature in record.features:
            if feature.type not in ("gene", "CDS", "tRNA", "rRNA"):
                continue
            raw = feature_gene_name(feature)
            if not raw:
                continue
            q = feature.qualifiers
            pseudo = "pseudo" in q or "pseudogene" in q
            note = " ".join(q.get("note", []))
            out.append((raw, feature.type, pseudo, bool(_TRANSFER_NOTE.search(note))))
    return out


def plot_gene_presence(
    genbank_path: Sequence[str | Path],
    *,
    names: Sequence[str] | None = None,
    tree_path: str | Path | None = None,
    order: str = "auto",
    genes: Sequence[str] | None = None,
    transferred: Mapping[str, Sequence[str]] | None = None,
    variable_only: bool = False,
    include_trna: bool = True,
    include_rrna: bool = True,
    include_orfs: bool = False,
    trna_names: str = "auto",
    organelle: str | None = None,
    width: str | float = "double",
    cell_mm: float = 2.6,
    columns_per_block: int | None = None,
    dpi: int = 600,
) -> OrganellePlot:
    """Species x gene matrix: present, pseudogene, transferred to the nucleus, absent.

    ``genbank_path`` lists one GenBank file per genome (a file with several
    records — a multichromosomal mitogenome — is one genome). A gene is
    *present* when any copy is annotated without ``/pseudo``/``/pseudogene``,
    a *pseudogene* when every copy is, *transferred* when ``transferred``
    lists it for that genome (``{genome name: [genes]}``) or a feature note
    says "transferred to the nucleus" and no functional copy remains, and
    *absent* otherwise. Names are matched case-insensitively, renamed genes
    are unified (``pafI``→``ycf3`` …) and ``rrn16S``→``rrn16``. tRNAs keep the
    anticodon (``trnI-CAU``) when every genome names it (``trna_names="auto"``
    or ``"anticodon"``), else they are compared per amino acid (``trnI``).

    Columns group by OGDraw functional class (coloured header, same palette as
    the maps); ``genes`` fixes a subset and order instead; ``variable_only``
    keeps genes not present everywhere. Rows follow ``tree_path`` (Newick;
    tips matched to genome names, accessions or file stems, extra tips
    pruned) drawn on the left, or a hierarchical clustering of the status
    profiles (``order="cluster"``, the ``"auto"`` choice for ≥ 3 genomes
    without a tree), or the input order (``"input"``). Genes that do not fit
    the printed ``width`` at ``cell_mm`` per column wrap into stacked,
    balanced blocks breaking between functional classes where they fit
    (``columns_per_block`` fixes the block size instead; ``0`` keeps one
    row). ``ov.write`` also writes ``<stem>.matrix.tsv``.
    """
    if order not in ("auto", "tree", "cluster", "input"):
        raise OrganelleParameterError(
            code="visualization.gene_presence.order",
            message="order must be 'auto', 'tree', 'cluster' or 'input'.",
        )
    if trna_names not in ("auto", "anticodon", "amino_acid"):
        raise OrganelleParameterError(
            code="visualization.gene_presence.trna_names",
            message="trna_names must be 'auto', 'anticodon' or 'amino_acid'.",
        )
    paths = _paths(genbank_path)
    genomes = [_records(p) for p in paths]
    labels = _names(names, len(paths), [
        record_display_name(records[0], p.stem) for records, p in zip(genomes, paths, strict=True)
    ])
    organelle = organelle or Counter(record_organelle(r[0]) for r in genomes).most_common(1)[0][0]
    raw = [_genome_genes(records) for records in genomes]
    trnas = [(n, kind) for entries in raw for n, kind, _, _ in entries
             if kind == "tRNA" or n.lower().startswith("trn")]
    every_has_anticodon = all(_gene_key(n, True)[2] for n, _ in trnas) if trnas else False
    use_anticodon = trna_names == "anticodon" or (trna_names == "auto" and every_has_anticodon)

    spellings: dict[str, Counter] = {}
    status_sets: list[dict[str, str]] = []
    for entries in raw:
        found: dict[str, set[str]] = {}
        for name, _kind, pseudo, note in entries:
            key, display, _ = _gene_key(name, use_anticodon)
            is_trna, is_rrna = key.startswith("trn"), key.startswith("rrn")
            if (is_trna and not include_trna) or (is_rrna and not include_rrna):
                continue
            if not include_orfs and re.match(r"^orf", key):
                continue
            spellings.setdefault(key, Counter())[display] += 1
            found.setdefault(key, set()).add("transferred" if note else "pseudogene" if pseudo
                                             else "present")
        status_sets.append({
            key: next(s for s in ("present", "transferred", "pseudogene") if s in states)
            for key, states in found.items()
        })
    display_of = {key: counts.most_common(1)[0][0] for key, counts in spellings.items()}
    if genes is not None:
        keys = []
        for gene in genes:
            key = _gene_key(str(gene), use_anticodon)[0]
            display_of.setdefault(key, _gene_key(str(gene), use_anticodon)[1])
            keys.append(key)
    else:
        rank = {cat: i for i, (cat, _) in enumerate(legend_rows(organelle))}
        keys = sorted(display_of, key=lambda k: (
            rank.get(classify_gene(display_of[k], organelle), len(rank)), _natural_key(display_of[k])
        ))
    transfers = {str(k).lower(): {_gene_key(str(g), use_anticodon)[0] for g in v}
                 for k, v in (transferred or {}).items()}
    matrix = []
    for label, path, records, statuses in zip(labels, paths, genomes, status_sets, strict=True):
        aliases = {label.lower(), path.stem.lower(), records[0].id.lower(),
                   records[0].id.split(".")[0].lower()}
        listed = set().union(*(transfers.get(a, set()) for a in aliases))
        row = []
        for key in keys:
            state = statuses.get(key, "absent")
            if key in listed and state != "present":
                state = "transferred"
            row.append(state)
        matrix.append(row)
    if variable_only:
        keep = [i for i in range(len(keys)) if any(row[i] != "present" for row in matrix)]
        keys = [keys[i] for i in keep]
        matrix = [[row[i] for i in keep] for row in matrix]
    if not keys:
        raise OrganelleInputError(
            code="visualization.gene_presence.empty", message="No genes left to draw."
        )
    resolved = order
    if order == "auto":
        resolved = "tree" if tree_path else "cluster" if len(paths) >= 3 else "input"
    if resolved == "tree" and not tree_path:
        raise OrganelleParameterError(
            code="visualization.gene_presence.tree", message="order='tree' needs tree_path."
        )
    rows_order = list(range(len(paths)))
    segments: list[list[float]] = []
    if resolved == "tree":
        rows_order, segments = _tree_layout(Path(tree_path), labels, paths, genomes)
    elif resolved == "cluster" and len(paths) >= 3:
        rows_order, segments = _cluster_layout(matrix)
    genes_out = [{"gene": display_of[k], "category": classify_gene(display_of[k], organelle)}
                 for k in keys]
    species = [labels[i] for i in rows_order]
    ordered = [matrix[i] for i in rows_order]
    metrics = {
        "organelle": organelle,
        "species": species,
        "files": [paths[i].name for i in rows_order],
        "genes": genes_out,
        "matrix": ordered,
        "counts": [{status: row.count(status) for status in STATUS_ORDER} for row in ordered],
        "order": resolved,
        "tree_segments": segments,
        "trna_names": "anticodon" if use_anticodon else "amino_acid",
        "width_mm": round(width_inches(width) * 25.4, 2),
        "cell_mm": float(cell_mm),
        "columns_per_block": columns_per_block,
        "dpi": dpi,
    }
    layout = _presence_layout(metrics, metrics["width_mm"])
    metrics["height_mm"] = round(layout["height"], 2)
    metrics["blocks"] = [len(b["columns"]) for b in layout["blocks"]]
    return _with_panel(plot_result(
        "plot_gene_presence",
        lambda path: _render_gene_presence(Path(path), metrics),
        organelle=organelle,
        method="gene_presence_matrix",
        metrics=metrics,
        flags=_height_flag(metrics) + (("wider_than_page",) if layout["overflow"] else ()),
        summary=f"Gene presence matrix of {len(species)} genomes x {len(keys)} genes prepared.",
    ), metrics)


def _tree_layout(tree_path: Path, labels, paths, genomes) -> tuple[list[int], list[list[float]]]:
    from Bio import Phylo  # type: ignore

    try:
        tree = Phylo.read(str(tree_path), "newick")
    except Exception as exc:  # Bio.Phylo raises several parser error types
        raise OrganelleInputError(code="visualization.gene_presence.tree",
                                  message=f"Cannot read Newick tree {tree_path}: {exc}") from exc

    def norm(text: str) -> str:
        return re.sub(r"[\s_]+", " ", str(text)).strip().lower()

    lookup: dict[str, int] = {}
    for i, (label, path, records) in enumerate(zip(labels, paths, genomes, strict=True)):
        for alias in (label, path.stem, records[0].id, records[0].id.split(".")[0]):
            lookup.setdefault(norm(alias), i)
    tips = tree.get_terminals()
    matched = {tip: lookup.get(norm(tip.name or "")) for tip in tips}
    found = sorted(i for i in matched.values() if i is not None)
    if found != list(range(len(labels))):
        missing = [labels[i] for i in range(len(labels)) if i not in found]
        raise OrganelleInputError(
            code="visualization.gene_presence.tree_tips",
            message=f"Tree tips do not match every genome once; unmatched: {missing}",
        )
    for tip, index in matched.items():
        if index is None:
            tree.prune(tip)
    tips = tree.get_terminals()
    order = [lookup[norm(tip.name)] for tip in tips]
    depths = tree.depths()
    if not any(depths.values()):
        depths = tree.depths(unit_branch_lengths=True)
    max_depth = max(depths[t] for t in tips) or 1.0
    y_of = {tip: float(i) for i, tip in enumerate(tips)}
    segments: list[list[float]] = []

    # Children are placed before their parent's horizontal segments are drawn.
    def walk(clade):
        for child in clade.clades:
            walk(child)
        if not clade.is_terminal():
            ys = [y_of[c] for c in clade.clades]
            y_of[clade] = (min(ys) + max(ys)) / 2
            x = depths[clade] / max_depth
            segments.append([x, min(ys), x, max(ys), 0])
            for child in clade.clades:
                segments.append([x, y_of[child], depths[child] / max_depth, y_of[child], 0])

    walk(tree.root)
    for tip in tips:
        if depths[tip] / max_depth < 0.999:
            segments.append([depths[tip] / max_depth, y_of[tip], 1.0, y_of[tip], 1])
    return order, segments


def _cluster_layout(matrix) -> tuple[list[int], list[list[float]]]:
    from scipy.cluster.hierarchy import dendrogram, linkage  # type: ignore
    from scipy.spatial.distance import pdist  # type: ignore

    codes = [[STATUS_ORDER.index(state) for state in row] for row in matrix]
    tree = linkage(pdist(codes, metric="hamming"), method="average")
    dendro = dendrogram(tree, no_plot=True)
    leaves = list(dendro["leaves"])
    top = max(max(d) for d in dendro["dcoord"]) or 1.0
    segments = []
    for xs, ds in zip(dendro["icoord"], dendro["dcoord"], strict=True):
        points = [(1 - d / top, (i - 5.0) / 10.0) for i, d in zip(xs, ds, strict=True)]
        for (x0, y0), (x1, y1) in pairwise(points):
            segments.append([x0, y0, x1, y1, 0])
    return leaves, segments


def _blocks(genes, limit: int) -> list[list[int]]:
    """Split gene columns into the fewest stacked blocks of at most ``limit`` columns.

    Among those splits, the one that tears the fewest functional classes wins,
    then the one whose longest block is shortest (dynamic programming over cut
    positions), so blocks come out balanced and break between classes whenever
    a class fits in a block.
    """
    n = len(genes)
    if limit <= 0 or n <= limit:
        return [list(range(n))]
    count = math.ceil(n / limit)
    torn = [0] + [int(genes[i - 1]["category"] == genes[i]["category"]) for i in range(1, n)]
    torn.append(0)
    inf = (n + 1, n + 1)
    # best[k][i]: (classes torn, longest block) for the first i columns in k blocks.
    best = [[inf] * (n + 1) for _ in range(count + 1)]
    cut = [[0] * (n + 1) for _ in range(count + 1)]
    best[0][0] = (0, 0)
    for k in range(1, count + 1):
        for i in range(1, n + 1):
            for j in range(max(0, i - limit), i):
                prev = best[k - 1][j]
                if prev == inf:
                    continue
                score = (prev[0] + (torn[j] if j else 0), max(prev[1], i - j))
                if score < best[k][i]:
                    best[k][i], cut[k][i] = score, j
    blocks, i = [], n
    for k in range(count, 0, -1):
        j = cut[k][i]
        blocks.append(list(range(j, i)))
        i = j
    return blocks[::-1]


def _presence_layout(m: Mapping[str, Any], width_mm: float) -> dict[str, Any]:
    """Millimetre layout of the matrix at ``width_mm``: blocks that fit the width, legend rows."""
    cell = float(m["cell_mm"])
    species, genes = m["species"], m["genes"]
    f_sp, f_gene, f_leg = FONT_PT["body"], LABEL_PT, FONT_PT["body"]
    margin, group_gap, strip_h, block_gap = 0.5, 0.7, 1.1, 2.5
    tree_w = 11.0 if m["tree_segments"] else 0.0
    name_w = max(_text_mm(n, f_sp, True) for n in species) + 2.0
    avail = width_mm - 2 * margin - tree_w - name_w

    def place(columns):
        xs, x, previous = [], 0.0, None
        for i in columns:
            if previous is not None and genes[i]["category"] != previous:
                x += group_gap
            xs.append(x)
            x += cell
            previous = genes[i]["category"]
        return xs, x

    limit = m.get("columns_per_block")
    if limit is None:
        limit = max(1, int(avail // cell))
        while limit > 1 and any(place(b)[1] > avail for b in _blocks(genes, limit)):
            limit -= 1
    blocks = []
    for columns in _blocks(genes, int(limit)):
        xs, block_w = place(columns)
        gene_h = max(_text_mm(genes[i]["gene"], f_gene, True) for i in columns) + 0.8
        blocks.append({"columns": columns, "xs": xs, "width": block_w, "gene_h": gene_h,
                       "height": gene_h + strip_h + 0.6 + len(species) * cell})
    rank = dict(legend_rows(m["organelle"]))
    entries = [("status", st) for st in STATUS_ORDER if any(st in row for row in m["matrix"])]
    entries += [("category", c) for c in dict.fromkeys(g["category"] for g in genes)]
    legend_lines: list[list[tuple[str, str, float]]] = [[]]
    used = 0.0
    for kind, key in entries:
        text = _STATUS_LABEL.get(key, key) if kind == "status" else rank.get(key, key)
        w = 2.4 + _text_mm(text, f_leg) + 2.5
        if used + w > width_mm - 2 * margin and legend_lines[-1]:
            legend_lines.append([])
            used = 0.0
        legend_lines[-1].append((kind, key, w))
        used += w
    line = f_leg * 1.6 * _PT_MM
    legend_h = len(legend_lines) * line + 1.5
    height = (2 * margin + sum(b["height"] for b in blocks) + block_gap * (len(blocks) - 1)
              + legend_h)
    return {"blocks": blocks, "cell": cell, "margin": margin, "strip_h": strip_h,
            "block_gap": block_gap, "tree_w": tree_w, "name_w": name_w, "legend": legend_lines,
            "legend_line": line, "legend_h": legend_h, "rank": rank, "height": height,
            "columns_per_block": int(limit), "fonts": (f_sp, f_gene, f_leg),
            "overflow": any(b["width"] > avail + 1e-6 for b in blocks)}


_STATUS_LABEL = {"transferred": "transferred to nucleus"}


def _status_style(state: str) -> dict[str, Any]:
    if state == "pseudogene":
        return {"facecolor": STATUS_COLORS[state], "edgecolor": STATUS_COLORS["present"],
                "lw": 0.35}
    if state == "absent":
        return {"facecolor": STATUS_COLORS[state], "edgecolor": "#D4D4D4", "lw": 0.25}
    return {"facecolor": STATUS_COLORS[state], "edgecolor": "none"}


def _draw_gene_presence(ax, m: Mapping[str, Any], legend: bool) -> list[Any]:
    from matplotlib.patches import Patch, Rectangle  # type: ignore

    width_mm = _axes_mm(ax)[0]
    lay = _presence_layout(m, width_mm)
    species, genes, matrix = m["species"], m["genes"], m["matrix"]
    f_sp, f_gene, f_leg = lay["fonts"]
    cell, margin = lay["cell"], lay["margin"]
    height = lay["height"]
    _mm_axes(ax, width_mm, height)
    x0 = margin + lay["tree_w"] + lay["name_w"]
    inset = 0.1 * cell
    block_top = height - margin
    for block in lay["blocks"]:
        y_top = block_top - block["height"] + len(species) * cell  # top edge of row 0
        for i, gx in zip(block["columns"], block["xs"], strict=True):
            gene = genes[i]
            ax.add_patch(Rectangle((x0 + gx, y_top + 0.6), cell, lay["strip_h"],
                                   facecolor=GENE_COLORS.get(gene["category"],
                                                             GENE_COLORS["other"]),
                                   edgecolor="white", lw=0.2))
            ax.text(x0 + gx + cell / 2, y_top + 0.6 + lay["strip_h"] + 0.4, gene["gene"],
                    rotation=90, ha="center", va="bottom", fontsize=f_gene,
                    fontstyle="italic")
        for r, (name, row) in enumerate(zip(species, matrix, strict=True)):
            yc = y_top - (r + 0.5) * cell
            ax.text(margin + lay["tree_w"] + 0.8, yc, name, ha="left", va="center",
                    fontsize=f_sp, fontstyle="italic")
            for i, gx in zip(block["columns"], block["xs"], strict=True):
                side = cell - 2 * inset
                ax.add_patch(Rectangle((x0 + gx + inset, yc - side / 2), side, side,
                                       **_status_style(row[i])))
        if m["tree_segments"]:
            tx0, tx1 = margin, margin + lay["tree_w"] - 0.4
            for xa, ya, xb, yb, dotted in m["tree_segments"]:
                ax.plot([tx0 + xa * (tx1 - tx0), tx0 + xb * (tx1 - tx0)],
                        [y_top - (ya + 0.5) * cell, y_top - (yb + 0.5) * cell],
                        color="0.55" if dotted else "0.15", lw=0.35 if dotted else 0.5,
                        ls=(0, (1, 1.5)) if dotted else "-", solid_capstyle="round")
        block_top -= block["height"] + lay["block_gap"]
    handles = []
    for line in lay["legend"]:
        for kind, key, _w in line:
            if kind == "status":
                handles.append(Patch(label=_STATUS_LABEL.get(key, key), **_status_style(key)))
            else:
                handles.append(Patch(facecolor=GENE_COLORS.get(key, GENE_COLORS["other"]),
                                     edgecolor="none", label=lay["rank"].get(key, key)))
    if not legend:
        return handles
    y = margin + lay["legend_h"] - 0.8 - lay["legend_line"] / 2
    for line in lay["legend"]:
        lx = margin
        for kind, key, w in line:
            sq = 2.0
            style = (_status_style(key) if kind == "status" else
                     {"facecolor": GENE_COLORS.get(key, GENE_COLORS["other"]),
                      "edgecolor": "none"})
            label = (_STATUS_LABEL.get(key, key) if kind == "status"
                     else lay["rank"].get(key, key))
            ax.add_patch(Rectangle((lx, y - sq / 2), sq, sq, **style))
            ax.text(lx + sq + 0.6, y, label, va="center", fontsize=f_leg)
            lx += w
        y -= lay["legend_line"]
    return []


def _render_gene_presence(path: Path, m: Mapping[str, Any]) -> Path:
    width_mm = float(m["width_mm"])
    height_mm = _presence_layout(m, width_mm)["height"]
    _render_mm(path, m, width_mm, height_mm, lambda ax: _draw_gene_presence(ax, m, True))
    genes, species, matrix = m["genes"], m["species"], m["matrix"]
    _write_tsv(
        path.with_suffix(".matrix.tsv"),
        ["genome", *[g["gene"] for g in genes]],
        [{"genome": name, **{g["gene"]: state for g, state in zip(genes, row, strict=True)}}
         for name, row in zip(species, matrix, strict=True)],
    )
    return path


# ---------------------------------------------------------------------------
# serialized results → renderer
# ---------------------------------------------------------------------------

_RENDERERS = {
    "visualization.plot_ir_boundaries": _render_ir_boundaries,
    "visualization.plot_nucleotide_diversity": _render_nucleotide_diversity,
    "visualization.plot_repeat_summary": _render_repeat_summary,
    "visualization.plot_gene_presence": _render_gene_presence,
}
PAPER_FIGURE_OPERATIONS = frozenset(_RENDERERS)

_DRAWERS = {
    "visualization.plot_ir_boundaries": (_draw_ir_boundaries, "IR boundaries"),
    "visualization.plot_nucleotide_diversity": (_draw_nucleotide_diversity,
                                                "Nucleotide diversity"),
    "visualization.plot_repeat_summary": (_draw_repeat_summary, "Repeats"),
    "visualization.plot_gene_presence": (_draw_gene_presence, "Gene content"),
}


def _panel_aspect(op: str, m: Mapping[str, Any]) -> float:
    """Height / width of the panel at the figure's own printed width.

    The IR panel and the gene matrix are laid out for their width (full width by
    default), so compose them with ``columns=1``; the axes plots scale freely.
    """
    return float(m["height_mm"]) / float(m["width_mm"])


def _with_panel(plot: OrganellePlot, metrics: Mapping[str, Any]) -> OrganellePlot:
    draw, header = _DRAWERS[plot.operation_id]
    plot._panel = PanelDrawer(lambda ax, legend: draw(ax, metrics, legend),
                              _panel_aspect(plot.operation_id, metrics), header)
    return plot


def _height_flag(metrics: Mapping[str, Any]) -> tuple[str, ...]:
    return ("taller_than_page",) if float(metrics["height_mm"]) > MAX_HEIGHT_MM else ()


def restore_paper_figure(result) -> OrganellePlot:
    """Reattach rendering to a serialized paper-figure Result; nothing is recomputed."""
    from ..core.frozen import thaw_json

    plot = OrganellePlot.model_validate(result.model_dump(mode="python"))
    metrics = thaw_json(plot.metrics)
    render = _RENDERERS[plot.operation_id]
    plot._renderer = lambda path: render(Path(path), metrics)
    return _with_panel(plot, metrics)


__all__ = [
    "PAPER_FIGURE_OPERATIONS",
    "motif_class",
    "plot_gene_presence",
    "plot_ir_boundaries",
    "plot_nucleotide_diversity",
    "plot_repeat_summary",
    "restore_paper_figure",
]
