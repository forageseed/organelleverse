"""Multi-sample organelle genome stability figures.

Three publication figures for DNA-maintenance mutants (msh1, recA, whirly,
polIb ...), populations, regenerants or CMS lines aligned to one organelle
reference:

* :func:`plot_depth_profile` - samples x windows log2 relative depth
  (diverging, white = the sample's own median or a reference sample),
  mitochondrion and plastid side by side, a genotype strip, an OGDraw-coloured
  gene track and repeat-pair arcs; ``mode="ridgeline"`` draws one trace per
  sample, ``mode="tracks"`` zooms into one region with one area track per
  sample on its own depth axis.
* :func:`plot_variant_frequencies` - one column per sample, SNV/InDel and
  mitochondrion/plastid facets; fractions under ``threshold`` are summarized as
  a density band, only fractions at or above it are drawn as points, coloured
  and shaped by category.
* :func:`plot_variant_site_heatmap` - sites at or above ``threshold`` in at
  least one sample x samples, showing which variants recur across samples.

The inputs are the results of
:func:`organelleverse.heteroplasmy_panel.summarize_depth_windows` and
:func:`organelleverse.heteroplasmy_panel.summarize_variant_fractions` (or their
``metrics``). Every call is compute-only; ``ov.write(plot, "fig.pdf")`` renders
PNG/SVG/PDF by suffix and writes the drawn data next to it as TSV.

Figures follow :mod:`.style`: drawn at printed size (``width`` = ``"single"``,
``"onehalf"``, ``"double"`` or mm), fixed point sizes, no in-figure title. Each
plot also carries a :class:`~.plot_object.PanelDrawer`, so
``compose_figure([...])`` can letter and align them; the depth and frequency
figures share their left and right margins, so stacked panels line up.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from ..core.errors import OrganelleInputError, OrganelleParameterError
from ..core.frozen import thaw_json
from .features import annotation_rows
from .ogdraw import GENE_COLORS, _load_genbank_records, legend_rows, record_organelle
from .plot_object import OrganellePlot, PanelDrawer, plot_result
from .style import FONT_PT, MAX_HEIGHT_MM, publication_style, width_inches

#: Okabe-Ito order; the first group (usually the wild type) gets the neutral grey.
GROUP_COLORS = ("#7F7F7F", "#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#F0E442")
#: Category -> (colour, marker, label); colour-blind safe and redundant with shape.
#: Low-confidence points are drawn hollow, small and underneath the others.
CATEGORY_STYLE = {
    "other": ("#0072B2", "o", "other"),
    "repeat_related": ("#D55E00", "^", "repeat-related"),
    "low_confidence": ("#A6A6A6", "o", "low confidence"),
}
SCOPE_LABEL = {"mitochondrion": "Mitochondrion", "plastid": "Plastid", None: "Sequence"}
SCOPE_SHORT = {"mitochondrion": "mt", "plastid": "pt"}
_SCOPE_ORDER = {"mitochondrion": 0, "plastid": 1}

#: ``summarize_depth_windows(...).metrics`` at its real depth (the Result itself is accepted too).
DepthSummary = Mapping[
    str,
    int | float | str | None | Sequence[str]
    | Sequence[Mapping[str, str | int | float | None | Sequence[float]]],
]
#: ``summarize_variant_fractions(...).metrics`` at its real depth (the Result itself is accepted too).
VariantSummary = Mapping[
    str,
    int | float | str | None | Sequence[str]
    | Mapping[str, int | float | Sequence[str]]
    | Sequence[Mapping[str, str | int | float | bool | None]],
]

# Layout in millimetres at printed size.
_RIGHT_MM = 14.0  # colour bar column; also kept by the frequency figure so panels align
_LINE_MM = 3.6  # one legend row at the body size
#: Tick, sample and feature labels: never below 6.5 pt at printed size.
_TICK = max(FONT_PT["tick"], 6.5)
_LABEL = max(FONT_PT["label"], 6.5)
_PT_MM = 25.4 / 72


def _metrics(value: Any, operation: str) -> dict[str, Any]:
    metrics = getattr(value, "metrics", value)
    if getattr(value, "status", "ok") != "ok":
        raise OrganelleInputError(
            code="visualization.stability.failed_input",
            message=f"{operation} input result has status {value.status}: {value.summary_text}",
        )
    if not isinstance(metrics, Mapping):
        raise OrganelleInputError(
            code="visualization.stability.input",
            message="Pass the summarize_* result or its metrics mapping.",
        )
    return thaw_json(metrics) if not isinstance(metrics, dict) else metrics


def _samples(samples: Sequence[str], groups: Mapping[str, str] | None, order: Sequence[str] | None):
    """Ordered samples with their group and group colour; groups stay contiguous."""
    ordered = list(order) if order is not None else list(samples)
    missing = set(ordered) - set(samples)
    if missing or len(set(ordered)) != len(ordered):
        raise OrganelleParameterError(
            code="visualization.stability.sample_order",
            message=f"sample_order must list each sample once; unknown: {sorted(missing)}",
        )
    group_of = {s: (groups or {}).get(s, "") for s in ordered}
    names = list(dict.fromkeys(group_of[s] for s in ordered))
    ordered.sort(key=lambda s: names.index(group_of[s]))
    colors = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(names)}
    return [{"sample": s, "group": group_of[s], "color": colors[group_of[s]]} for s in ordered]


def _x_positions(layout, gap: float = 0.6):
    xs, x, previous = [], 0.0, None
    for item in layout:
        if previous is not None and item["group"] != previous:
            x += gap
        xs.append(x)
        x += 1.0
        previous = item["group"]
    return xs


def _group_spans(layout, xs):
    spans = []
    for item, x in zip(layout, xs, strict=True):
        if spans and spans[-1]["group"] == item["group"]:
            spans[-1]["end"] = x
        else:
            spans.append({"group": item["group"], "color": item["color"], "start": x, "end": x})
    return spans


def _check_width(width: str | float, dpi: int) -> None:
    try:
        width_inches(width)
    except ValueError as exc:
        raise OrganelleParameterError(code="visualization.stability.width", message=str(exc)) from exc
    if dpi < 72:
        raise OrganelleParameterError(code="visualization.stability.dpi", message="dpi must be >= 72.")


# ── layout helpers ──────────────────────────────────────────────────────────────


def _text_mm(text: str, size_pt: float) -> float:
    """Approximate printed width of ``text`` (Arial-like average glyph = 0.55 em)."""
    return len(text) * 0.55 * size_pt * _PT_MM


def _left_mm(layout) -> float:
    """Left margin for sample labels + genotype strip; shared so stacked figures align."""
    return max(22.0, max(_text_mm(i["sample"], _TICK) for i in layout) + 7.0)


class _Box:
    """Places sub-axes inside a host axes by millimetres measured from its top-left."""

    def __init__(self, host) -> None:
        width, height = host.figure.get_size_inches()
        pos = host.get_position()
        self.host = host
        self.w = pos.width * width * 25.4
        self.h = pos.height * height * 25.4
        host.axis("off")

    def axes(self, x: float, y: float, w: float, h: float):
        return self.host.inset_axes((x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h))

    def text(self, x: float, y: float, s: str, **kw):
        return self.host.text(x / self.w, 1 - y / self.h, s, transform=self.host.transAxes, **kw)

    def legend(self, handles, y: float, columns: int):
        return self.host.legend(
            handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1 - y / self.h), ncol=columns,
            fontsize=FONT_PT["body"], frameon=False, handlelength=1.0, handleheight=0.8,
            columnspacing=1.2, borderaxespad=0, borderpad=0,
        )


def _legend_columns(handles, width_mm: float) -> int:
    widest = max((_text_mm(h.get_label(), FONT_PT["body"]) for h in handles), default=10) + 6.0
    return max(1, min(len(handles), int(width_mm // widest)))


def _legend_height(handles, width_mm: float) -> float:
    if not handles:
        return 0.0
    return 2.0 + _LINE_MM * math.ceil(len(handles) / _legend_columns(handles, width_mm))


def _render(path: Path, spec: Mapping[str, Any], draw: Callable, height: Callable[[float], float]) -> Path:
    """Standalone render at printed width; ``height(width_mm)`` sizes the canvas."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width_mm = width_inches(spec["width"]) * 25.4
    with publication_style():
        fig = plt.figure(figsize=(width_mm / 25.4, min(height(width_mm), MAX_HEIGHT_MM) / 25.4))
        try:
            host = fig.add_axes((0, 0, 1, 1))
            draw(host, spec, True)
            fig.savefig(path, dpi=spec["dpi"], bbox_inches="tight", pad_inches=0.04)
        finally:
            plt.close(fig)
    return path


def _panel(spec, draw, height) -> PanelDrawer:
    width_mm = width_inches(spec["width"]) * 25.4
    return PanelDrawer(lambda ax, legend: draw(ax, spec, legend), aspect=height(width_mm) / width_mm)


def _style_axis(ax, *, left=True, bottom=True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(labelsize=_TICK, length=2.0, width=0.5)


# ── 1. depth profile ────────────────────────────────────────────────────────────


def _gene_blocks(genbank_paths, seqids: set[str]):
    blocks: dict[str, list[dict[str, Any]]] = {}
    if not genbank_paths:
        return blocks
    try:
        records = [record for _, record in _load_genbank_records(genbank_paths)]
    except (OSError, ValueError) as exc:
        raise OrganelleInputError(code="visualization.stability.genbank", message=str(exc)) from exc
    for record in records:
        names = {record.id, record.name, record.id.split(".")[0]}
        seqid = next((s for s in seqids if s in names or s.split(".")[0] in names), None)
        if seqid is None:
            continue
        organelle = record_organelle(record)
        for row in annotation_rows(record, organelle, strict=False):
            for exon in row["exons"]:
                blocks.setdefault(seqid, []).append(
                    {
                        "gene": row["gene"],
                        "category": row["category"],
                        "organelle": organelle,
                        "start": exon["start"],
                        "end": exon["end"],
                        "strand": exon["strand"],
                    }
                )
    return blocks


def plot_depth_profile(
    depth: DepthSummary,
    *,
    groups: Mapping[str, str] | None = None,
    sample_order: Sequence[str] | None = None,
    genbank_paths: str | Path | Sequence[str | Path] | None = None,
    repeats: Sequence[Mapping[str, str | int | Sequence[int]]] | None = None,
    min_arc_length: int = 500,
    mode: Literal["heatmap", "ridgeline", "tracks"] = "heatmap",
    region: Sequence[str | int] | None = None,
    limit: float = 2.0,
    reference_sample: str | None = None,
    width: str | float = "double",
    dpi: int = 600,
) -> OrganellePlot:
    """Prepare a multi-sample read-depth figure.

    ``depth`` is ``summarize_depth_windows(...)`` (or its metrics).

    ``mode="heatmap"`` (default): samples x windows, one panel per sequence
    (mitochondrion left, plastid right). Colour is the log2 ratio to the sample's
    own median window depth on that sequence, clipped to ``+-limit`` (white = 0,
    red = more reads, blue = fewer). ``reference_sample`` (e.g. the wild type)
    subtracts that sample's profile window by window, so biases every sample
    shares (GC, mappability) cancel. ``mode="ridgeline"`` draws the same values as
    one trace per sample.

    ``mode="tracks"``: zoom into ``region = [seqid, start, end]`` (1-based,
    inclusive) and draw each sample as an area track of mean window depth on its
    own y axis, so a region absent from a sample drops to 0 whatever that
    sample's overall coverage. Use small windows (100-200 bp) for zooms.

    ``groups`` maps sample -> genotype (left strip / track colour; samples of a
    group stay together, in ``sample_order`` or input order). ``genbank_paths``
    adds a gene track coloured by OGDraw functional class (records matched to
    sequences by accession; gene names are written in the tracks zoom).
    ``repeats`` (``compute_multiconf(...)["repeats"]``) adds arcs linking the two
    copies of each repeat pair of at least ``min_arc_length`` bp (blue direct,
    vermillion inverted, width ~ length). ``width`` is the printed width
    (``"single"`` 89 mm, ``"onehalf"`` 120 mm, ``"double"`` 183 mm or mm).
    Rendering also writes ``<stem>.depth.tsv``.
    """
    data = _metrics(depth, "plot_depth_profile")
    if mode not in ("heatmap", "ridgeline", "tracks") or limit <= 0 or min_arc_length < 1:
        raise OrganelleParameterError(
            code="visualization.stability.parameters",
            message="mode is heatmap, ridgeline or tracks; limit > 0; min_arc_length >= 1.",
        )
    _check_width(width, dpi)
    layout = _samples(data["samples"], groups, sample_order)
    contigs = sorted(data["contigs"], key=lambda c: _SCOPE_ORDER.get(c["organelle"], 2))
    zoom = None
    if mode == "tracks":
        lengths = {c["seqid"]: c["length"] for c in contigs}
        if region is None or len(region) != 3 or str(region[0]) not in lengths:
            raise OrganelleParameterError(
                code="visualization.stability.region",
                message="mode='tracks' needs region=[seqid, start, end] on a summarized sequence.",
            )
        seqid, start, end = str(region[0]), int(region[1]), int(region[2])
        if not 1 <= start < end <= lengths[seqid]:
            raise OrganelleParameterError(
                code="visualization.stability.region",
                message=f"region must satisfy 1 <= start < end <= {lengths[seqid]:,}.",
            )
        zoom = {"seqid": seqid, "start": start, "end": end}
        contigs = [c for c in contigs if c["seqid"] == seqid]
    seqids = {c["seqid"] for c in contigs}
    genes = _gene_blocks(genbank_paths, seqids)
    arcs = [
        {
            "seqid": str(r["sequence_id"]),
            "a": int(r["positions"][0]),
            "b": int(r["positions"][1]),
            "length": int(r["length"]),
            "type": str(r.get("type", "direct")),
        }
        for r in repeats or ()
        if str(r["sequence_id"]) in seqids and int(r["length"]) >= min_arc_length
    ]
    profiles = [p for p in data["profiles"] if p["seqid"] in seqids]
    if reference_sample is not None:
        if reference_sample not in data["samples"]:
            raise OrganelleParameterError(
                code="visualization.stability.reference_sample",
                message=f"reference_sample {reference_sample!r} is not one of the samples.",
            )
        base = {p["seqid"]: p["log2_ratio"] for p in profiles if p["sample"] == reference_sample}
        profiles = [
            {**p, "log2_ratio": [round(v - b, 4) for v, b in zip(p["log2_ratio"], base[p["seqid"]], strict=True)]}
            for p in profiles
        ]
    spec = {
        "layout": layout,
        "contigs": contigs,
        "window_size": data["window_size"],
        "profiles": profiles,
        "reference_sample": reference_sample,
        "region": zoom,
        "genes": genes,
        "arcs": arcs,
        "min_arc_length": min_arc_length,
        "mode": mode,
        "limit": limit,
        "width": width,
        "dpi": dpi,
    }
    spec["caption"] = _depth_caption(spec)
    plot = plot_result(
        "plot_depth_profile",
        lambda path: _render_depth_profile(Path(path), spec),
        organelle=contigs[0]["organelle"] if len(contigs) == 1 else "mixed",
        method="stability_depth",
        metrics=spec,
        key_findings=({"metric": "samples", "value": len(layout)},
                      {"metric": "repeat_arcs", "value": len(arcs)}),
        summary=f"Read depth of {len(layout)} samples over {len(contigs)} sequences prepared ({mode}).",
    )
    plot._panel = _panel(spec, _draw_depth, lambda w: _depth_height(spec, w, True))
    return plot


def _depth_legend(spec) -> list:
    from matplotlib.patches import Patch

    grouped = len({i["group"] for i in spec["layout"]}) > 1
    handles = []
    if grouped and spec["mode"] != "tracks":
        seen = dict.fromkeys((i["group"], i["color"]) for i in spec["layout"])
        handles += [Patch(color=c, label=g) for g, c in seen]
    present = {b["category"] for blocks in spec["genes"].values() for b in blocks
               if _visible(spec, b["start"], b["end"])}
    order = [key for organelle in ("mito", "plastid") for key, _ in legend_rows(organelle)]
    handles += [Patch(facecolor=GENE_COLORS.get(k, GENE_COLORS["other"]), edgecolor="#333333", lw=0.3,
                      label=_GENE_SHORT.get(k, k)) for k in dict.fromkeys(order) if k in present]
    types = {a["type"] for a in _arcs_in_view(spec)}
    handles += [Patch(color=color, label=f"{kind} repeat")
                for kind, color in (("direct", "#0072B2"), ("inverted", "#D55E00")) if kind in types]
    return handles


#: Short legend names for the OGDraw functional classes; the full names go to the caption.
_GENE_SHORT = {
    "complex_I": "complex I", "complex_II": "complex II", "complex_III": "complex III",
    "complex_IV": "complex IV", "atp_synthase": "ATP synthase", "cytochrome_c": "cyt c biogenesis",
    "rna_pol": "RNA polymerase", "ribo_SSU": "ribosomal SSU", "ribo_LSU": "ribosomal LSU",
    "maturase": "maturase", "other": "other genes", "orf": "ORF", "tRNA": "tRNA", "rRNA": "rRNA",
    "photosystem_I": "photosystem I", "photosystem_II": "photosystem II", "cytochrome_b6f": "cyt b6f",
    "rubisco": "RuBisCO", "clp_protease": "clpP", "ycf": "ycf",
}


def _depth_caption(spec) -> str:
    """Figure-legend text for the depth figure; the drawn legend keeps only keys."""
    window = spec["window_size"]
    if spec["mode"] == "tracks":
        zoom = spec["region"]
        text = (f"Mean read depth in {window} bp windows over {zoom['seqid']}:{zoom['start']:,}-"
                f"{zoom['end']:,}, one track per sample, each on its own y axis (0 = no reads). ")
    else:
        base = (f"the {spec['reference_sample']} profile (log2 ratio minus that of {spec['reference_sample']})"
                if spec["reference_sample"] else "the sample's own median window depth on that sequence")
        text = (f"Rows are samples grouped by genotype (left strip); columns are {window} bp windows. "
                f"Colour is log2 mean depth relative to {base}, clipped at ±{spec['limit']:g}; "
                "white = no change, red = more reads, blue = fewer. ")
    if spec["genes"]:
        text += "Gene track: OGDraw functional classes, + strand above the line, - strand below. "
    if spec["arcs"]:
        text += (f"Arcs join the two copies of repeat pairs of at least {spec['min_arc_length']:,} bp "
                 "(blue direct, vermillion inverted; width ~ repeat length). ")
    return text.rstrip()


def _visible(spec, start: int, end: int) -> bool:
    zoom = spec["region"]
    return zoom is None or (start <= zoom["end"] and end >= zoom["start"])


def _arcs_in_view(spec) -> list:
    return [a for a in spec["arcs"]
            if _visible(spec, a["a"], a["a"] + a["length"]) or _visible(spec, a["b"], a["b"] + a["length"])]


def _depth_parts(spec) -> dict[str, float]:
    """Fixed heights (mm) of the depth figure apart from the legend."""
    n = len(spec["layout"])
    tracks = spec["mode"] == "tracks"
    body = n * (9.0 + 1.6) if tracks else n * (3.2 if spec["mode"] == "heatmap" else 5.0)
    genes = any(_visible(spec, b["start"], b["end"]) for v in spec["genes"].values() for b in v)
    return {
        "title": 4.0,
        "body": body,
        "genes": (_GENE_NAMED_MM if tracks else _GENE_PLAIN_MM) if genes else 0.0,
        "arcs": 9.0 if _arcs_in_view(spec) else 0.0,
        "axis": 8.0,
    }


def _depth_height(spec, width_mm: float, legend: bool) -> float:
    parts = _depth_parts(spec)
    extra = 1.5 * (bool(parts["genes"]) + bool(parts["arcs"]))
    handles = _depth_legend(spec) if legend else []
    return sum(parts.values()) + extra + _legend_height(handles, width_mm)


def _draw_depth(host, spec, legend: bool):
    if spec["mode"] == "tracks":
        return _draw_depth_tracks(host, spec, legend)
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MplPath

    box = _Box(host)
    layout, contigs, limit = spec["layout"], spec["contigs"], spec["limit"]
    window = spec["window_size"]
    profile = {(p["sample"], p["seqid"]): p for p in spec["profiles"]}
    n = len(layout)
    heat = spec["mode"] == "heatmap"
    parts = _depth_parts(spec)
    handles = _depth_legend(spec)
    flexible = box.h - _depth_height(spec, box.w, legend) + parts["body"]
    body = max(flexible, 0.5 * parts["body"])
    left = _left_mm(layout)
    plot_w = box.w - left - _RIGHT_MM
    lengths = [c["length"] for c in contigs]
    shares = [max(length, 0.32 * max(lengths)) for length in lengths]
    gap = 4.0
    widths = [(plot_w - gap * (len(contigs) - 1)) * s / sum(shares) for s in shares]
    cmap = LinearSegmentedColormap.from_list(
        "ov_depth", ["#053061", "#2166AC", "#92C5DE", "#FFFFFF", "#F4A582", "#B2182B", "#67001F"]
    )
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)
    top = parts["title"]
    y_genes = top + body + 1.5
    y_arcs = y_genes + (parts["genes"] + 1.5 if parts["genes"] else 0.0)
    y_axis = y_arcs + (parts["arcs"] if parts["arcs"] else 0.0)
    image = None
    x = left
    longest = max((a["length"] for a in spec["arcs"]), default=1)
    for col, (contig, w) in enumerate(zip(contigs, widths, strict=True)):
        seqid, length = contig["seqid"], contig["length"]
        ax = box.axes(x, top, w, body)
        starts = np.arange(0, length, window)
        if heat:
            matrix = np.array([profile[(i["sample"], seqid)]["log2_ratio"] for i in layout], dtype=float)
            image = ax.imshow(np.clip(matrix, -limit, limit), aspect="auto", interpolation="nearest",
                              cmap=cmap, norm=norm, extent=(0, length, n, 0))
            for i in range(1, n):
                if layout[i]["group"] != layout[i - 1]["group"]:
                    ax.axhline(i, color="white", lw=1.0)
            ax.set_ylim(n, 0)
        else:
            mids = np.minimum(starts + window / 2, length)
            for i, item in enumerate(layout):
                values = np.clip(profile[(item["sample"], seqid)]["log2_ratio"], -limit, limit)
                base = n - 1 - i
                trace = base + 0.45 * values / limit
                ax.fill_between(mids, base, trace, where=values >= 0, color="#B2182B", lw=0,
                                alpha=0.75, interpolate=True)
                ax.fill_between(mids, base, trace, where=values < 0, color="#2166AC", lw=0,
                                alpha=0.75, interpolate=True)
                ax.plot(mids, trace, color=item["color"], lw=0.5)
                ax.axhline(base, color="#BBBBBB", lw=0.3)
            ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(0, length)
        ax.set_xticks([])
        ax.set_yticks([])
        _style_axis(ax, left=col == 0 and not heat, bottom=False)
        if heat:
            for side in ax.spines.values():
                side.set_visible(True)
                side.set_linewidth(0.5)
        label = SCOPE_LABEL.get(contig["organelle"], contig["organelle"] or "Sequence")
        ax.set_title(f"{label} · {seqid} · {length:,} bp", loc="left", fontsize=FONT_PT["body"], pad=3)
        last = ax
        if parts["genes"]:
            gax = box.axes(x, y_genes, w, parts["genes"])
            _draw_gene_track(gax, spec["genes"].get(seqid, ()), 0, length, width_mm=w, names=False)
            if col == 0:
                box.text(left - 1.0, y_genes + parts["genes"] / 2, "genes +/\u2212", ha="right",
                         va="center", fontsize=_TICK)
            last = gax
        if parts["arcs"]:
            aax = box.axes(x, y_arcs, w, parts["arcs"])
            for arc in spec["arcs"]:
                if arc["seqid"] != seqid:
                    continue
                a = arc["a"] + arc["length"] / 2
                b = arc["b"] + arc["length"] / 2
                color = "#0072B2" if arc["type"] == "direct" else "#D55E00"
                drop = -min(1.0, 0.15 + abs(b - a) / length * 1.6) * 1.33
                curve = MplPath([(a, 0), (a, drop), (b, drop), (b, 0)],
                                [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
                aax.add_patch(PathPatch(curve, facecolor="none", edgecolor=color, alpha=0.8,
                                        lw=0.4 + 1.2 * math.sqrt(arc["length"] / longest)))
                for start in (arc["a"], arc["b"]):
                    aax.add_patch(Rectangle((start, -0.04), arc["length"], 0.12, color=color, lw=0))
            aax.set_xlim(0, length)
            aax.set_ylim(-1.1, 0.12)
            aax.set_yticks([])
            _style_axis(aax, left=False)
            if col == 0:
                box.text(left - 1.0, y_arcs + parts["arcs"] / 2, f"repeats ≥{spec['min_arc_length']:,} bp",
                         ha="right", va="center", fontsize=_TICK)
            last = aax
        ticks = np.arange(0, length + 1, _tick_step(length))
        last.set_xticks(ticks)
        last.set_xticklabels([f"{t // 1000:.0f}" for t in ticks])
        last.set_xlabel("Position (kb)", fontsize=FONT_PT["body"], labelpad=2)
        last.spines["bottom"].set_visible(True)
        last.tick_params(axis="x", labelsize=_TICK, length=2.0, width=0.5)
        x += w + gap
    strip = box.axes(left - 3.0, top, 2.2, body)
    rows_y = list(range(n)) if heat else [n - 1 - i - 0.5 for i in range(n)]
    for item, y in zip(layout, rows_y, strict=True):
        strip.add_patch(Rectangle((0, y), 1, 1, color=item["color"], lw=0))
    strip.set_xlim(0, 1)
    strip.set_ylim((n, 0) if heat else (-0.6, n - 0.4))
    strip.set_xticks([])
    strip.set_yticks([y + 0.5 for y in rows_y])
    strip.set_yticklabels([i["sample"] for i in layout], fontsize=_TICK)
    strip.tick_params(axis="y", length=0, pad=1.5)
    for side in strip.spines.values():
        side.set_visible(False)
    if heat:
        cax = box.axes(box.w - _RIGHT_MM + 2.5, top, 2.0, min(body, 28.0))
        bar = host.figure.colorbar(image, cax=cax, extend="both")
        bar.set_label(f"log2 depth vs {spec['reference_sample']}" if spec["reference_sample"]
                      else "log2 depth / median", fontsize=_TICK, labelpad=2)
        bar.ax.tick_params(labelsize=_TICK, length=2, width=0.5)
        bar.outline.set_linewidth(0.5)
    if not legend:
        return handles
    if handles:
        box.legend(handles, y_axis + parts["axis"] + 1.0, _legend_columns(handles, box.w))
    return []


#: Gene track geometry in mm: blocks reach 2.6 mm each side of the axis; named
#: tracks add two label levels per strand, one 6 pt line (2.6 mm) apart.
_GENE_BLOCK_MM = 2.6
_GENE_LEVELS_MM = (3.0, 5.6)
_GENE_NAMED_MM = 16.0
_GENE_PLAIN_MM = 6.5


def _draw_gene_track(ax, blocks, lo: int, hi: int, *, width_mm: float, names: bool) -> int:
    """Gene blocks (+ strand above the line); with ``names``, returns how many names did not fit."""
    from matplotlib.patches import Rectangle

    ax.axhline(0, color="#555555", lw=0.4)
    shown = [b for b in blocks if b["start"] <= hi and b["end"] >= lo]
    bp_per_mm = (hi - lo) / width_mm
    for block in shown:
        y = 0.2 if block["strand"] == 1 else -_GENE_BLOCK_MM
        color = GENE_COLORS.get(block["category"], GENE_COLORS["other"])
        ax.add_patch(Rectangle((block["start"] - 1, y), max(block["end"] - block["start"] + 1, 0.3 * bp_per_mm),
                               _GENE_BLOCK_MM - 0.2, facecolor=color, edgecolor="#333333", lw=0.2))
    dropped = 0
    if names:
        # One name per gene; a name takes the first level where it clears its neighbour,
        # and is left out (counted) rather than shrunk when neither level has room.
        spans: dict[tuple[str, int], list[int]] = {}
        for b in shown:
            spans.setdefault((b["gene"], b["strand"]), []).extend((max(b["start"], lo), min(b["end"], hi)))
        level_end: dict[tuple[int, int], float] = {}
        for (gene, strand), coords in sorted(spans.items(), key=lambda kv: min(kv[1])):
            mid = (min(coords) + max(coords)) / 2
            half = (_text_mm(gene, _LABEL) / 2 + 0.6) * bp_per_mm
            for level, offset in enumerate(_GENE_LEVELS_MM):
                if level_end.get((strand, level), -math.inf) < mid - half:
                    level_end[(strand, level)] = mid + half
                    ax.text(mid, offset if strand == 1 else -offset, gene, ha="center",
                            va="bottom" if strand == 1 else "top", fontsize=_LABEL,
                            style="italic", clip_on=False)
                    break
            else:
                dropped += 1
    top = _GENE_NAMED_MM / 2 if names else _GENE_PLAIN_MM / 2
    ax.set_xlim(lo, hi)
    ax.set_ylim(-top, top)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    return dropped


def _draw_depth_tracks(host, spec, legend: bool):
    import numpy as np
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator

    box = _Box(host)
    layout, window = spec["layout"], spec["window_size"]
    zoom = spec["region"]
    lo, hi = zoom["start"] - 1, zoom["end"]
    profile = {p["sample"]: p for p in spec["profiles"]}
    parts = _depth_parts(spec)
    handles = _depth_legend(spec)
    n = len(layout)
    row_gap = 1.6
    row_h = max((box.h - _depth_height(spec, box.w, legend) + parts["body"]) / n - row_gap, 4.0)
    left = 16.0
    plot_w = box.w - left - 4.0
    first = lo // window
    last = math.ceil(hi / window)
    edges = np.arange(first, last + 1) * window
    y = parts["title"]
    ticks = _zoom_ticks(lo, hi)
    axes = []
    for i, item in enumerate(layout):
        values = np.asarray(profile[item["sample"]]["mean_depth"][first:last], dtype=float)
        ax = box.axes(left, y, plot_w, row_h)
        ax.fill_between(edges[:-1], values, step="post", color=item["color"], alpha=0.55, lw=0)
        ax.step(edges[:-1], values, where="post", color=item["color"], lw=0.6)
        ax.set_xlim(lo, hi)
        # Headroom keeps the sample name clear of the trace.
        ax.set_ylim(0, max(float(values.max()) * 1.5, 1.0))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
        ax.set_xticks(ticks)
        ax.grid(color="0.9", lw=0.4)
        ax.set_axisbelow(True)
        _style_axis(ax)
        ax.tick_params(axis="x", labelbottom=False)
        ax.text(0.995, 0.96, item["sample"], transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT_PT["body"], color="#222222")
        if i == 0:
            ax.set_title(f"{SCOPE_LABEL.get(spec['contigs'][0]['organelle'], 'Sequence')} · {zoom['seqid']} · "
                         f"{zoom['start']:,}\u2013{zoom['end']:,} bp · {window} bp windows",
                         loc="left", fontsize=FONT_PT["body"], pad=3)
        axes.append(ax)
        y += row_h + row_gap
    box.text(3.0, parts["title"] + (y - parts["title"]) / 2, "Read depth (\u00d7)", rotation=90,
             ha="center", va="center", fontsize=FONT_PT["body"])
    bottom = axes[-1]
    dropped = 0
    if parts["genes"]:
        gax = box.axes(left, y + 0.5, plot_w, parts["genes"])
        dropped = _draw_gene_track(gax, spec["genes"].get(zoom["seqid"], ()), lo, hi, width_mm=plot_w,
                                   names=True)
        box.text(left - 1.0, y + 0.5 + parts["genes"] / 2, "genes +/\u2212", ha="right", va="center",
                 fontsize=_TICK)
        y += parts["genes"] + 2.0
        bottom = gax
    arcs = _arcs_in_view(spec)
    if arcs:
        aax = box.axes(left, y, plot_w, parts["arcs"] * 0.6)
        for arc in arcs:
            color = "#0072B2" if arc["type"] == "direct" else "#D55E00"
            for start in (arc["a"], arc["b"]):
                if start <= hi and start + arc["length"] >= lo:
                    aax.add_patch(Rectangle((start, 0), arc["length"], 1, color=color, lw=0))
                    aax.text(start + arc["length"] / 2, 1.1, f"{arc['length']:,} bp", ha="center",
                             va="bottom", fontsize=_LABEL, clip_on=False)
        aax.set_xlim(lo, hi)
        aax.set_ylim(0, 2.2)
        aax.set_yticks([])
        aax.set_xticks([])
        for side in aax.spines.values():
            side.set_visible(False)
        box.text(left - 1.0, y + parts["arcs"] * 0.3, "repeat copies", ha="right", va="center",
                 fontsize=_TICK)
        y += parts["arcs"] * 0.6
        bottom = aax
    bottom.set_xticks(ticks)
    bottom.set_xticklabels([_kb(t) for t in ticks])
    bottom.tick_params(axis="x", labelbottom=True, labelsize=_TICK, length=2, width=0.5)
    bottom.spines["bottom"].set_visible(True)
    bottom.set_xlabel("Position (kb)", fontsize=FONT_PT["body"], labelpad=2)
    if parts["genes"] and dropped:
        box.text(left + plot_w, y + parts["axis"] - 1.0, f"{dropped} gene names not shown (space)",
                 ha="right", va="bottom", fontsize=_LABEL, color="#555555")
    if not legend:
        return handles
    if handles:
        box.legend(handles, y + parts["axis"], _legend_columns(handles, box.w))
    return []


def _zoom_ticks(lo: int, hi: int) -> list[int]:
    span = hi - lo
    for step in (500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000):
        if span / step <= 8:
            return list(range(math.ceil(lo / step) * step, hi + 1, step))
    return [lo, hi]


def _kb(position: int) -> str:
    return f"{position / 1000:g}"


def _render_depth_profile(path: Path, spec: Mapping[str, Any]) -> Path:
    _render(path, spec, _draw_depth, lambda w: _depth_height(spec, w, True))
    zoom = spec["region"]
    with path.with_suffix(".depth.tsv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        ratio = f"log2_vs_{spec['reference_sample']}" if spec["reference_sample"] else "log2_ratio"
        writer.writerow(["sample", "group", "seqid", "start", "end", "mean_depth", ratio])
        profile = {(p["sample"], p["seqid"]): p for p in spec["profiles"]}
        window = spec["window_size"]
        for item in spec["layout"]:
            for contig in spec["contigs"]:
                p = profile[(item["sample"], contig["seqid"])]
                for i, (mean, value) in enumerate(zip(p["mean_depth"], p["log2_ratio"], strict=True)):
                    start, end = i * window + 1, min((i + 1) * window, contig["length"])
                    if zoom and (end < zoom["start"] or start > zoom["end"]):
                        continue
                    writer.writerow([item["sample"], item["group"], contig["seqid"], start, end, mean, value])
    return path


def _tick_step(length: int) -> int:
    for step in (10_000, 20_000, 50_000, 100_000, 200_000, 500_000):
        if length / step <= 8:
            return step
    return 1_000_000


# ── 2. per-sample frequency panels ──────────────────────────────────────────────


def _variant_table(data):
    sites = {s["site_id"]: s for s in data["sites"]}
    return sites, data["fractions"]


def plot_variant_frequencies(
    variants: VariantSummary,
    *,
    groups: Mapping[str, str] | None = None,
    sample_order: Sequence[str] | None = None,
    threshold: float = 0.05,
    sqrt_scale: bool = True,
    width: str | float = "double",
    dpi: int = 600,
) -> OrganellePlot:
    """Prepare per-sample SNV/InDel fraction panels (rows: organelle, columns: kind).

    ``variants`` is ``summarize_variant_fractions(...)`` (or its metrics). In each
    sample column, fractions >= ``threshold`` are points coloured and shaped by
    category (other, repeat-related, low confidence); the detected fractions below
    it are summarized as a density band (a kernel density of sqrt(fraction)),
    since thousands of low-frequency calls would hide each other as points. The
    number of points per column is printed above it. Genotype groups get a tinted
    background and a gap. ``sqrt_scale`` stretches the low end of the y axis.
    ``width`` is the printed width. Rendering also writes ``<stem>.frequencies.tsv``.
    """
    data = _metrics(variants, "plot_variant_frequencies")
    if not 0 < threshold < 1:
        raise OrganelleParameterError(
            code="visualization.stability.threshold", message="threshold must be in (0, 1)."
        )
    _check_width(width, dpi)
    layout = _samples(data["samples"], groups, sample_order)
    sites, rows = _variant_table(data)
    scopes = sorted({s["scope"] for s in sites.values()} or set(data.get("references", {})),
                    key=lambda s: _SCOPE_ORDER.get(s, 2))
    kinds = [k for k in ("SNV", "InDel") if any(s["kind"] == k for s in sites.values())] or ["SNV"]
    points, bands = [], []
    for r in rows:
        f = r["alt_fraction"]
        if f is None:
            continue
        site = sites[r["site_id"]]
        key = {"sample": r["sample"], "scope": site["scope"], "kind": site["kind"]}
        if f >= threshold:
            points.append({**key, "site_id": r["site_id"], "fraction": f, "category": r["category"],
                           "alt_count": r["alt_count"], "depth": r["depth"]})
        elif r["detected"]:
            bands.append({**key, "fraction": f})
    spec = {
        "layout": layout,
        "scopes": scopes,
        "kinds": kinds,
        "points": points,
        "bands": bands,
        "threshold": threshold,
        "min_alt_fraction": data.get("thresholds", {}).get("min_alt_fraction"),
        "criteria": {**data.get("thresholds", {}), "control_samples": data.get("control_samples", [])},
        "sqrt_scale": sqrt_scale,
        "width": width,
        "dpi": dpi,
    }
    spec["caption"] = _frequency_caption(spec)
    plot = plot_result(
        "plot_variant_frequencies",
        lambda path: _render_variant_frequencies(Path(path), spec),
        organelle=scopes[0] if len(scopes) == 1 else "mixed",
        method="stability_frequencies",
        metrics=spec,
        key_findings=({"metric": "points_at_or_above_threshold", "value": len(points)},
                      {"metric": "detected_below_threshold", "value": len(bands)}),
        summary=f"{len(points)} sample-variant fractions >= {threshold:g} and {len(bands)} below it prepared.",
    )
    plot._panel = _panel(spec, _draw_frequencies, lambda w: _frequency_height(spec, w, True))
    return plot


def _kde_band(values, lo, hi, transform, inverse):
    import numpy as np

    t = transform(np.asarray(values, dtype=float))
    grid = np.linspace(transform(lo), transform(hi), 80)
    if len(t) < 2 or float(np.std(t)) == 0.0:
        density = np.exp(-0.5 * ((grid - t.mean()) / ((grid[-1] - grid[0]) / 25)) ** 2)
    else:
        bw = 1.06 * float(np.std(t)) * len(t) ** (-1 / 5)
        density = np.exp(-0.5 * ((grid[:, None] - t[None, :]) / bw) ** 2).sum(axis=1)
    return inverse(grid), density / density.max()


def _frequency_legend(spec) -> list:
    """Keys only for what is drawn; the band and threshold are explained in the caption."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    present = {p["category"] for p in spec["points"]}
    handles = []
    for c in ("other", "repeat_related", "low_confidence"):
        if c in present:
            color, marker, label = CATEGORY_STYLE[c]
            hollow = c == "low_confidence"
            handles.append(Line2D([], [], ls="", marker=marker, markersize=3.2 if hollow else 3.8,
                                  markerfacecolor="none" if hollow else color, markeredgecolor=color,
                                  markeredgewidth=0.6, label=label))
    if len({i["group"] for i in spec["layout"]}) > 1:
        seen = dict.fromkeys((i["group"], i["color"]) for i in spec["layout"])
        handles += [Patch(color=c, alpha=0.35, label=g) for g, c in seen]
    return handles


def _frequency_caption(spec) -> str:
    threshold, floor = spec["threshold"], spec["min_alt_fraction"] or 0.0
    t = spec.get("criteria") or {}
    rules = []
    if t.get("min_depth") is not None:
        rules.append(f"depth < {t['min_depth']}")
    if t.get("min_alt_reads") is not None:
        rules.append(f"< {t['min_alt_reads']} supporting reads")
    controls = ", ".join(t.get("control_samples") or ())
    low = "homology-masked (organelle-to-organelle transfer)"
    if controls:
        low += f", also present in the control ({controls}) at \u2265 {floor:g}"
    if rules:
        low += ", or " + " / ".join(rules)
    return (
        "Each column is one sample; genotype groups are shaded and separated. Points are sites whose "
        f"alternate-read fraction is \u2265 {threshold:g} in that sample (dashed line, labelled on the axis): "
        "filled blue circles, other; vermillion triangles, repeat-related (overlapping a repeat copy); "
        f"small hollow grey circles, low confidence ({low}). The number above each column counts the "
        "points that are not low confidence. Shapes below the line are kernel densities of the detected "
        f"fractions between {floor:g} and {threshold:g}" + (" (square-root axis)." if spec["sqrt_scale"] else ".")
        + " Fractions are read fractions, not molecule fractions."
    )


def _frequency_parts(spec) -> dict[str, float]:
    tallest = max(_text_mm(i["sample"], _TICK) for i in spec["layout"])
    return {"row": 30.0, "title": 8.0, "gap": 3.0, "labels": tallest * 0.72 + 3.0}


def _frequency_height(spec, width_mm: float, legend: bool) -> float:
    parts = _frequency_parts(spec)
    rows = len(spec["scopes"])
    handles = _frequency_legend(spec) if legend else []
    return (rows * (parts["row"] + parts["title"]) + (rows - 1) * parts["gap"] + parts["labels"]
            + 1.0 + _legend_height(handles, width_mm))


def _draw_frequencies(host, spec, legend: bool):
    import numpy as np
    from matplotlib.patches import Rectangle

    box = _Box(host)
    layout, scopes, kinds = spec["layout"], spec["scopes"], spec["kinds"]
    threshold = spec["threshold"]
    parts = _frequency_parts(spec)
    handles = _frequency_legend(spec)
    flexible = box.h - _frequency_height(spec, box.w, legend)
    row_h = max(parts["row"] + flexible / len(scopes), 12.0)
    xs = _x_positions(layout)
    x_of = {item["sample"]: x for item, x in zip(layout, xs, strict=True)}
    spans = _group_spans(layout, xs)
    grouped = len(spans) > 1
    forward, inverse = (np.sqrt, np.square) if spec["sqrt_scale"] else ((lambda v: v), (lambda v: v))
    band_floor = spec["min_alt_fraction"] or 0.0
    rng = np.random.default_rng(7)
    left = _left_mm(layout)
    col_gap = 8.0
    panel_w = (box.w - left - _RIGHT_MM - col_gap * (len(kinds) - 1)) / len(kinds)
    ticks = [0, 0.01, 0.1, 0.25, 0.5, 1.0] if spec["sqrt_scale"] else [0, 0.25, 0.5, 0.75, 1]
    ticks = sorted({*ticks, threshold} - {t for t in ticks if t and abs(t - threshold) / threshold < 0.5})
    rule = "#666666"
    y = 0.0
    for i, scope in enumerate(scopes):
        y += parts["title"]
        for j, kind in enumerate(kinds):
            ax = box.axes(left + j * (panel_w + col_gap), y, panel_w, row_h)
            if spec["sqrt_scale"]:
                ax.set_yscale("function", functions=(np.sqrt, np.square))
            for span in spans if grouped else ():
                ax.add_patch(Rectangle((span["start"] - 0.5, 0), span["end"] - span["start"] + 1, 1,
                                       facecolor=span["color"], alpha=0.10, lw=0, zorder=0))
            ax.axhline(threshold, color=rule, lw=0.5, ls=(0, (2.5, 2)), zorder=2.5)
            for item in layout:
                x = x_of[item["sample"]]
                low = [b["fraction"] for b in spec["bands"]
                       if b["sample"] == item["sample"] and b["scope"] == scope and b["kind"] == kind]
                if low:
                    # Density only spans [floor, threshold): it never reaches above the line.
                    ys, dens = _kde_band(low, band_floor, threshold, forward, inverse)
                    ax.fill_betweenx(ys, x - 0.3 * dens, x + 0.3 * dens, color=item["color"],
                                     alpha=0.4, lw=0, zorder=2)
                pts = [p for p in spec["points"]
                       if p["sample"] == item["sample"] and p["scope"] == scope and p["kind"] == kind]
                for category in ("low_confidence", "repeat_related", "other"):
                    sel = [p["fraction"] for p in pts if p["category"] == category]
                    if not sel:
                        continue
                    color, marker, _ = CATEGORY_STYLE[category]
                    jitter = rng.uniform(-0.28, 0.28, len(sel))
                    if category == "low_confidence":
                        ax.scatter(x + jitter, sel, s=3, marker=marker, facecolors="none", edgecolors=color,
                                   linewidths=0.4, zorder=1.5)
                    else:
                        ax.scatter(x + jitter, sel, s=7, marker=marker, facecolors=color, edgecolors="white",
                                   linewidths=0.25, zorder=3)
                kept = sum(p["category"] != "low_confidence" for p in pts)
                ax.text(x, 1.02, str(kept), ha="center", va="bottom", fontsize=_LABEL,
                        color="#333333", transform=ax.get_xaxis_transform(), clip_on=False)
            ax.set_xlim(-0.6, xs[-1] + 0.6)
            ax.set_ylim(0, 1)
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:g}" for t in ticks])
            for label, value in zip(ax.get_yticklabels(), ticks, strict=True):
                if value == threshold:
                    label.set_color(rule)
                    label.set_fontweight("bold")
            ax.set_xticks(xs)
            ax.grid(axis="y", color="0.92", lw=0.4)
            ax.set_axisbelow(True)
            _style_axis(ax)
            ax.tick_params(axis="x", length=0)
            if i == len(scopes) - 1:
                ax.set_xticklabels([item["sample"] for item in layout], rotation=45, ha="right",
                                   rotation_mode="anchor", fontsize=_TICK)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel("Alt read fraction", fontsize=FONT_PT["body"], labelpad=2)
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.set_title(f"{SCOPE_LABEL.get(scope, scope)} · {kind}", loc="left",
                         fontsize=FONT_PT["body"], pad=11)
        y += row_h + (parts["gap"] if i < len(scopes) - 1 else 0.0)
    if not legend:
        return handles
    box.legend(handles, y + parts["labels"] + 1.0, _legend_columns(handles, box.w))
    return []


def _render_variant_frequencies(path: Path, spec: Mapping[str, Any]) -> Path:
    _render(path, spec, _draw_frequencies, lambda w: _frequency_height(spec, w, True))
    with path.with_suffix(".frequencies.tsv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(["sample", "scope", "kind", "site_id", "alt_fraction", "drawn_as", "category"])
        for p in spec["points"]:
            writer.writerow([p["sample"], p["scope"], p["kind"], p["site_id"], p["fraction"], "point",
                             p["category"]])
        for b in spec["bands"]:
            writer.writerow([b["sample"], b["scope"], b["kind"], "", b["fraction"], "density", ""])
    return path


# ── 3. site x sample recurrence heatmap ─────────────────────────────────────────


def plot_variant_site_heatmap(
    variants: VariantSummary,
    *,
    groups: Mapping[str, str] | None = None,
    sample_order: Sequence[str] | None = None,
    threshold: float = 0.05,
    max_sites: int = 40,
    include_low_confidence: bool = True,
    width: str | float = "double",
    dpi: int = 600,
) -> OrganellePlot:
    """Prepare a sites x samples fraction heatmap of variants >= ``threshold`` somewhere.

    Rows are the sites whose fraction reaches ``threshold`` in at least one
    sample, ordered by organelle and position; columns are samples grouped by
    genotype. Cells show the alternate-read fraction on a square-root colour
    scale (grey = no fraction: no spanning reads). Side bars give organelle,
    SNV/InDel and site category (homology-masked or carried by a control = low
    confidence, inside a repeat copy = repeat-related), and the number of samples
    at or above ``threshold`` (recurrence). With more than ``max_sites`` rows the
    most recurrent (then highest) sites are kept and the result is flagged;
    ``include_low_confidence=False`` drops low-confidence sites before choosing.
    Row labels are printed when the rows fit at the label size.
    Rendering also writes ``<stem>.sites.tsv``.
    """
    data = _metrics(variants, "plot_variant_site_heatmap")
    if not 0 < threshold < 1 or max_sites < 1:
        raise OrganelleParameterError(
            code="visualization.stability.threshold",
            message="threshold must be in (0, 1) and max_sites >= 1.",
        )
    _check_width(width, dpi)
    layout = _samples(data["samples"], groups, sample_order)
    sites, rows = _variant_table(data)
    cell: dict[str, dict[str, float | None]] = {}
    for r in rows:
        cell.setdefault(r["site_id"], {})[r["sample"]] = r["alt_fraction"]

    def site_category(s) -> str:
        if s.get("masked") or s.get("in_control"):
            return "low_confidence"
        return "repeat_related" if s.get("in_repeat") else "other"

    chosen = []
    for site_id, values in cell.items():
        if not include_low_confidence and site_category(sites[site_id]) == "low_confidence":
            continue
        hits = sum(1 for v in values.values() if v is not None and v >= threshold)
        if hits:
            top = max(v for v in values.values() if v is not None)
            chosen.append((hits, top, site_id))
    chosen_all = list(chosen)
    truncated = len(chosen) > max_sites
    chosen = sorted(chosen, reverse=True)[:max_sites]
    keep = [sites[sid] for _, _, sid in chosen]
    keep.sort(key=lambda s: (_SCOPE_ORDER.get(s["scope"], 2), s["seqid"], s["position"], s["alt"]))
    table = []
    for s in keep:
        values = [cell[s["site_id"]].get(item["sample"]) for item in layout]
        table.append(
            {
                "site_id": s["site_id"],
                "label": f"{SCOPE_SHORT.get(s['scope'], s['seqid'])} {s['position']:,} "
                + (f"{s['ref']}>{s['alt']}" if s["kind"] == "SNV" else
                   (f"+{s['alt'][1:]}" if len(s["alt"]) > len(s["ref"]) else f"\u2212{s['ref'][1:]}")),
                "scope": s["scope"],
                "kind": s["kind"],
                "category": site_category(s),
                "values": values,
                "recurrence": sum(1 for v in values if v is not None and v >= threshold),
            }
        )
    spec = {"layout": layout, "rows": table, "threshold": threshold, "truncated": truncated,
            "width": width, "dpi": dpi}
    spec["caption"] = (
        f"Rows: variant sites whose alternate-read fraction is ≥ {threshold:g} in at least one sample"
        + (f" (the {len(table)} most recurrent of {len(chosen_all)})" if truncated else "")
        + ("" if include_low_confidence else ", low-confidence sites excluded")
        + "; columns: samples grouped by genotype. Colour: alternate-read fraction (square-root scale); "
        "grey: no reads span the site. Side bars: organelle, SNV/InDel, category (low confidence = "
        "homology-masked or present in a control sample; repeat-related = overlapping a repeat copy). "
        f"Right: number of samples ≥ {threshold:g}."
    )
    plot = plot_result(
        "plot_variant_site_heatmap",
        lambda path: _render_variant_site_heatmap(Path(path), spec),
        organelle="mixed",
        method="stability_site_heatmap",
        metrics=spec,
        key_findings=({"metric": "sites", "value": len(table)},
                      {"metric": "recurrent_sites", "value": sum(r["recurrence"] > 1 for r in table)}),
        flags=("sites_truncated",) if truncated else (),
        summary=f"{len(table)} sites with a fraction >= {threshold:g} in at least one sample prepared.",
    )
    plot._panel = _panel(spec, _draw_site_heatmap, lambda w: _site_height(spec, w, True))
    return plot


_SITE_ROW_MM = 2.8  # a labelled row: 6.5 pt text + leading
_SITE_BODY_MAX_MM = 120.0


def _site_parts(spec) -> dict[str, float]:
    n = max(1, len(spec["rows"]))
    labelled = n * _SITE_ROW_MM <= _SITE_BODY_MAX_MM
    tallest = max(_text_mm(i["sample"], _TICK) for i in spec["layout"])
    return {
        "header": 7.0,
        "body": max(18.0, n * _SITE_ROW_MM if labelled else _SITE_BODY_MAX_MM),
        "labels": max(tallest * 0.72, _text_mm("organelle", _TICK)) + 3.0,
        "labelled": float(labelled),
    }


def _site_legend(spec) -> list:
    from matplotlib.patches import Patch

    sides = (
        ({"mitochondrion": "#6A3D9A", "plastid": "#33A02C"}, {r["scope"] for r in spec["rows"]}),
        ({"SNV": "#444444", "InDel": "#BBBBBB"}, {r["kind"] for r in spec["rows"]}),
        ({k: v[0] for k, v in CATEGORY_STYLE.items()}, {r["category"] for r in spec["rows"]}),
    )
    handles = []
    for colors, present in sides:
        handles += [Patch(color=colors[k], label=CATEGORY_STYLE[k][2] if k in CATEGORY_STYLE else k)
                    for k in colors if k in present]
    handles.append(Patch(color="#E3E3E3", label="no spanning reads"))
    return handles


def _site_height(spec, width_mm: float, legend: bool) -> float:
    parts = _site_parts(spec)
    handles = _site_legend(spec) if legend else []
    return parts["header"] + parts["body"] + parts["labels"] + 1.0 + _legend_height(handles, width_mm)


def _draw_site_heatmap(host, spec, legend: bool):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap, PowerNorm
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator

    box = _Box(host)
    layout, table = spec["layout"], spec["rows"]
    n_rows, n_cols = max(1, len(table)), len(layout)
    parts = _site_parts(spec)
    handles = _site_legend(spec)
    body = max(parts["body"] + box.h - _site_height(spec, box.w, legend), 10.0)
    labelled = bool(parts["labelled"])
    xs = _x_positions(layout, gap=0.35)
    spans = _group_spans(layout, xs)
    label_w = (max(_text_mm(r["label"], _LABEL) for r in table) + 2.0) if labelled and table else 2.0
    bar_w, bar_gap = 2.4, 0.6
    sides_w = 3 * bar_w + 2 * bar_gap
    recurrence_w = 22.0
    heat_w = min(box.w - label_w - sides_w - 2.0 - recurrence_w - 3.0 - _RIGHT_MM, 9.0 * (xs[-1] + 1))
    top = parts["header"]
    x0 = label_w + sides_w + 2.0
    ax = box.axes(x0, top, heat_w, body)

    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#E3E3E3")
    matrix = np.array([[np.nan if v is None else v for v in r["values"]] for r in table] or [[np.nan] * n_cols])
    mesh = None
    for x, column in zip(xs, matrix.T, strict=True):
        mesh = ax.imshow(column[:, None], aspect="auto", cmap=cmap, norm=PowerNorm(0.5, 0, 1),
                         extent=(x - 0.5, x + 0.5, n_rows, 0), interpolation="nearest")
    ax.set_xlim(-0.5, xs[-1] + 0.5)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks(xs)
    ax.set_xticklabels([i["sample"] for i in layout], rotation=45, ha="right", rotation_mode="anchor",
                       fontsize=_TICK)
    ax.tick_params(left=False, labelleft=False, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    head = box.axes(x0, top - 2.6, heat_w, 2.0)
    for item, x in zip(layout, xs, strict=True):
        head.add_patch(Rectangle((x - 0.5, 0), 1, 1, color=item["color"], lw=0))
    head.set_xlim(-0.5, xs[-1] + 0.5)
    head.set_ylim(0, 1)
    head.axis("off")
    if len(spans) > 1:
        for span in spans:
            head.text((span["start"] + span["end"]) / 2, 1.35, span["group"], ha="center", va="bottom",
                      fontsize=_TICK)
    side_specs = (
        ("organelle", [r["scope"] for r in table], {"mitochondrion": "#6A3D9A", "plastid": "#33A02C"}),
        ("type", [r["kind"] for r in table], {"SNV": "#444444", "InDel": "#BBBBBB"}),
        ("category", [r["category"] for r in table], {k: v[0] for k, v in CATEGORY_STYLE.items()}),
    )
    for col, (name, values, colors) in enumerate(side_specs):
        sax = box.axes(label_w + col * (bar_w + bar_gap), top, bar_w, body)
        keys = list(colors)
        codes = np.array([[keys.index(v)] for v in values] or [[0]])
        sax.imshow(codes, aspect="auto", cmap=ListedColormap([colors[k] for k in keys]),
                   vmin=-0.5, vmax=len(keys) - 0.5, extent=(0, 1, n_rows, 0), interpolation="nearest")
        sax.set_xticks([0.5])
        sax.set_xticklabels([name], rotation=90, fontsize=_TICK)
        sax.tick_params(length=0)
        for side in sax.spines.values():
            side.set_visible(False)
        if col == 0 and labelled and table:
            sax.set_yticks(np.arange(n_rows) + 0.5)
            sax.set_yticklabels([r["label"] for r in table], fontsize=_LABEL)
        else:
            sax.set_yticks([])
        sax.set_ylim(n_rows, 0)
    rax = box.axes(x0 + heat_w + 3.0, top, recurrence_w, body)
    rax.barh(np.arange(n_rows) + 0.5, [r["recurrence"] for r in table], height=0.75, color="#555555")
    rax.set_xlim(0, n_cols)
    rax.set_ylim(n_rows, 0)
    rax.set_yticks([])
    rax.set_xlabel(f"Samples ≥ {spec['threshold']:g}", fontsize=FONT_PT["body"], labelpad=2)
    rax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    _style_axis(rax, left=False)
    cax = box.axes(x0 + heat_w + 3.0 + recurrence_w + 5.0, top, 2.0, min(body, 28.0))
    bar = host.figure.colorbar(mesh, cax=cax)
    bar.set_ticks([0, 0.05, 0.25, 0.5, 1])
    bar.set_label("Alt read fraction", fontsize=_TICK, labelpad=2)
    bar.ax.tick_params(labelsize=_TICK, length=2, width=0.5)
    bar.outline.set_linewidth(0.5)
    if spec["truncated"]:
        box.text(x0 + heat_w + 3.0, top - 1.0, f"Top {len(table)} sites by recurrence", ha="left",
                 va="bottom", fontsize=_TICK, color="#444444")
    if not legend:
        return handles
    box.legend(handles, top + body + parts["labels"] + 1.0, _legend_columns(handles, box.w))
    return []


def _render_variant_site_heatmap(path: Path, spec: Mapping[str, Any]) -> Path:
    _render(path, spec, _draw_site_heatmap, lambda w: _site_height(spec, w, True))
    with path.with_suffix(".sites.tsv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(["site_id", "scope", "kind", "category", "recurrence",
                         *[item["sample"] for item in spec["layout"]]])
        for r in spec["rows"]:
            writer.writerow([r["site_id"], r["scope"], r["kind"], r["category"], r["recurrence"],
                             *["" if v is None else v for v in r["values"]]])
    return path


_RENDERERS = {
    "visualization.plot_depth_profile": (_render_depth_profile, _draw_depth, _depth_height),
    "visualization.plot_variant_frequencies": (_render_variant_frequencies, _draw_frequencies,
                                               _frequency_height),
    "visualization.plot_variant_site_heatmap": (_render_variant_site_heatmap, _draw_site_heatmap,
                                                _site_height),
}


def _restore_stability_plot(result) -> OrganellePlot:
    """Reattach rendering to a serialized stability-plot Result (its metrics are the spec)."""
    plot = OrganellePlot.model_validate(result.model_dump(mode="python"))
    spec = thaw_json(plot.metrics)
    render, draw, height = _RENDERERS[result.operation_id]
    plot._renderer = lambda path: render(Path(path), spec)
    plot._panel = _panel(spec, draw, lambda w: height(spec, w, True))
    return plot


__all__ = [
    "CATEGORY_STYLE",
    "GROUP_COLORS",
    "plot_depth_profile",
    "plot_variant_frequencies",
    "plot_variant_site_heatmap",
]
