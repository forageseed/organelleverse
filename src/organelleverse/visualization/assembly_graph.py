"""OrganelleVerse-drawn assembly graph with repeat recombination conformations.

``plot_assembly_graph`` replaces the Bandage screenshot with a figure drawn by
OrganelleVerse itself:

* the network panel is the GFA graph; every segment is a fixed-width ribbon
  labelled with its name and depth, recombining repeats carry an ``R1``,
  ``R2``... badge in their own colour;
* with a plastid assembly, plastid-derived pieces (MTPTs, found with the
  transfer suite's LOSAT/blastn thresholds) are painted green on the
  mitochondrial ribbons and joined by translucent homology ribbons to their
  donor positions on a LSC/IRb/SSC/IRa plastid map; segments are coloured
  mitochondrial / shared, and reads that span an MTPT into mitochondrial
  flanks are written next to its ribbon;
* ``style="insets"`` adds a dashed group per repeat with the molecules each
  conformation spells; ``style="triptych"`` draws network -> single master
  circle (repeat copies as coloured blocks joined by dashed chords) and
  network -> sub-genomic circles produced by direct-repeat recombination;
* conformation frequencies come only from long reads spanning a repeat with
  unique anchors on both sides, either counted here (``reads``) or taken from
  an ``ovasm run`` ``evidence.json``; otherwise conformations are candidates.

``gfa`` may be a GFA file, ``OrganelleData``, or an ``ovasm run`` output
directory (one organelle, or the parent of ``mitochondrion/`` and
``plastid/``). The call is compute-only; ``ov.write(plot, "figure.svg" |
".png" | ".pdf")`` draws it at printed journal size as lettered panels
(``visualization.style``) and writes ``<figure>.conformations.tsv``,
``<figure>.caption.txt`` (title and method notes for the figure legend) and,
with a plastid, ``<figure>.mtpt.tsv`` beside it. The network can also be a
panel of ``compose_figure``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from organelleverse._public_validation import validated_public

from ..core.data import OrganelleData
from ..core.errors import OrganelleParameterError
from ..structure.graph_conformation import (
    PROPORTION_DEFINITIONS,
    analyze_conformations,
    parse_node_label,
)
from ..structure.graph_origin import load_ovasm_run
from .gfa_graph import _resolve_gfa_input
from .graph_layout import graph_layout
from .plot_object import OrganellePlot, plot_result
from .style import (
    FONT_PT,
    MAX_HEIGHT_MM,
    MM_PER_INCH,
    add_panel_label,
    publication_style,
    width_inches,
)

#: Unique-segment palette (``color_by="segment"``): muted and free of the
#: saturated hues used for repeat badges and of the plastid green.
PALETTE = (
    "#4C78A8",
    "#B279A2",
    "#72B7B2",
    "#9D755D",
    "#5D69B1",
    "#79706E",
    "#2E86AB",
    "#A0CBE8",
    "#D4A6C8",
    "#499894",
    "#86BCB6",
    "#BAB0AC",
    "#3F6E8C",
    "#8E6C8A",
    "#5F9EA0",
    "#B6992D",
)
#: One highlight colour for every recombining repeat; ``R1``, ``R2``... labels
#: tell them apart.
REPEAT_COLOR = "#B0177A"
ORIGIN_COLORS = {"mitochondrial": "#E39B4A", "shared": "#A3A8AE"}
MTPT_COLOR = "#3C9A4A"
PLASTID_REGION_COLORS = {"LSC": "#4FA35C", "IR": "#2B7A3D", "SSC": "#8CC97A", "plastome": "#4FA35C"}
LINK_COLOR = "#8A94A3"
UNSUPPORTED_COLOR = "#9AA3AF"
INK = "#1F2933"
MUTED = "#4B5563"
_READ_NAMES = {"hifi": "HiFi", "ont": "ONT", "clr": "PacBio CLR"}



@validated_public('visualization.plot_assembly_graph')
def plot_assembly_graph(
    gfa: str | Path | OrganelleData,
    *,
    reads: str | Path | Sequence[str | Path] | None = None,
    read_type: str = "hifi",
    plastid: str | Path | None = None,
    organelle: str = "mito",
    title: str = "",
    display_names: Mapping[str, str] | None = None,
    style: str = "insets",
    color_by: str = "auto",
    show_conformations: bool = True,
    max_insets: int = 6,
    ribbon_min_length: int = 0,
    flank: int = 1000,
    anchor: int = 500,
    min_identity: float | None = None,
    min_spanning_reads: int = 3,
    mtpt_min_length: int = 100,
    mtpt_min_identity: float = 80.0,
    threads: int = 4,
    minimap2: str | Path | None = None,
    layout_seed: int = 0,
    width: str | float = "double",
    dpi: int = 600,
) -> OrganellePlot:
    """Prepare an assembly-graph figure with repeat conformations (and MTPTs).

    Parameters
    ----------
    gfa:
        GFA1 assembly graph (ovasm, PMAT, Flye, ...), ``OrganelleData`` with a
        ``gfa``/``graph`` artifact, or an ``ovasm run`` output directory. A
        run directory supplies ``evidence.json`` repeat pairings and, for an
        ``organelle="both"`` run, the plastid assembly.
    reads:
        Long reads (FASTA/FASTQ, gzip allowed). When given, spanning reads are
        counted here for repeats and MTPTs and take precedence over
        ``evidence.json``.
    plastid:
        Plastid assembly (FASTA, first record used, or an ovasm plastid
        directory) for MTPT detection and the plastid map.
    style:
        ``insets`` (conformation insets beside the network) or ``triptych``
        (network -> single master circle / sub-genomic circles).
    color_by:
        ``segment`` (one colour per segment), ``origin`` (mitochondrial /
        shared, MTPT in green; needs a plastid) or ``auto`` (origin when a
        plastid is available).
    ribbon_min_length:
        MTPTs at least this long are summarised by the dashed connectors (one
        per mitochondrial segment and plastid region); every MTPT is painted on
        its segment and listed in the table.
    title:
        Goes into ``<figure>.caption.txt`` (journal figures carry no title).
    width:
        Printed width: ``single`` (89 mm), ``onehalf`` (120 mm), ``double``
        (183 mm) or millimetres; text is sized for that print size.
    """
    width_inches(width)
    if dpi <= 0 or max_insets < 0:
        raise OrganelleParameterError(
            code="visualization.assembly_graph.parameters",
            message="dpi must be positive and max_insets non-negative.",
            details={"dpi": dpi, "max_insets": max_insets},
        )
    if style not in {"insets", "triptych"} or color_by not in {"auto", "segment", "origin"}:
        raise OrganelleParameterError(
            code="visualization.assembly_graph.style",
            message="style must be insets|triptych and color_by auto|segment|origin.",
            details={"style": style, "color_by": color_by},
        )
    pairings = None
    decisive = None
    if not isinstance(gfa, OrganelleData) and Path(gfa).is_dir():
        run = load_ovasm_run(gfa)
        gfa_path, resolved_organelle = run["gfa"], organelle
        pairings, decisive = run["pairings"], run["decisive"]
        plastid = plastid if plastid is not None else run["plastid"]
    else:
        gfa_path, resolved_organelle = _resolve_gfa_input(gfa, organelle)
    if color_by == "origin" and plastid is None:
        raise OrganelleParameterError(
            code="visualization.assembly_graph.origin_needs_plastid",
            message="color_by='origin' needs a plastid assembly.",
        )
    read_paths = [reads] if isinstance(reads, (str, Path)) else list(reads or ())
    analysis = analyze_conformations(
        gfa_path,
        reads=read_paths,
        read_type=read_type,
        flank=flank,
        anchor=anchor,
        min_identity=min_identity,
        min_spanning_reads=min_spanning_reads,
        threads=threads,
        minimap2=minimap2,
        pairings=pairings,
        plastid=plastid,
        mtpt_min_length=mtpt_min_length,
        mtpt_min_identity=mtpt_min_identity,
    )
    mode = "origin" if color_by == "origin" or (color_by == "auto" and analysis["plastid"]) else "segment"
    repeats = analysis["repeats"]
    _number_repeats(repeats, analysis["master_molecules"])
    names = [row["name"] for row in analysis["segments"]]
    labels = _display_names(names, display_names or {})
    by_repeat = {row["name"]: row for row in repeats}
    palette = iter(PALETTE * (len(names) // len(PALETTE) + 1))
    for row in analysis["segments"]:
        row["label"] = labels[row["name"]]
        repeat = by_repeat.get(row["name"])
        row["kind"] = repeat["kind"] if repeat else "unique"
        row["repeat_id"] = repeat["id"] if repeat else None
        if repeat:
            row["color"] = repeat["color"]
        elif mode == "origin":
            row["color"] = ORIGIN_COLORS[row["origin"]]
        else:
            row["color"] = next(palette)
    plastid_graph = _plastid_graph(analysis["plastid"]) if analysis["plastid"] else None
    homology = (
        _homology(analysis["plastid"], plastid_graph, analysis["segments"], ribbon_min_length)
        if plastid_graph
        else []
    )
    layout = graph_layout(
        [*analysis["segments"], *(plastid_graph["segments"] if plastid_graph else [])],
        [*analysis["links"], *(plastid_graph["links"] if plastid_graph else [])],
        last=[row["name"] for row in plastid_graph["segments"]] if plastid_graph else (),
        seed=layout_seed,
    )
    insets = [
        row["name"] for row in sorted(repeats, key=_inset_rank) if row["configurations"]
    ][: max_insets if show_conformations and style == "insets" else 0]
    supported = [row for row in repeats if row["status"] == "supported"]
    metrics: dict[str, Any] = {
        "method": "organelleverse_graph",
        "style": style,
        "color_by": mode,
        "title": title,
        "width": width,
        "dpi": dpi,
        "nodes": analysis["graph"]["segments"],
        "edges": analysis["graph"]["links"],
        "paths": analysis["graph"]["paths"],
        "total_length": analysis["graph"]["total_length"],
        "single_copy_depth": analysis["graph"]["single_copy_depth"],
        "repeat_count": sum(row["kind"] == "repeat" for row in repeats),
        "branch_count": sum(row["kind"] == "branch" for row in repeats),
        "supported_repeats": len(supported),
        "linearization_decisive": decisive,
        "segments": analysis["segments"],
        "links": analysis["links"],
        "repeats": repeats,
        "master_molecules": analysis["master_molecules"],
        "read_evidence": analysis["read_evidence"],
        "plastid": analysis["plastid"],
        "plastid_graph": plastid_graph,
        "homology": homology,
        "ribbon_min_length": ribbon_min_length,
        "layout": layout,
        "insets": insets,
        "frequency_source": _frequency_source(analysis["read_evidence"]),
        "proportion_definition": (
            PROPORTION_DEFINITIONS[analysis["read_evidence"]["source"]] if analysis["read_evidence"] else None
        ),
    }
    if analysis["plastid"]:
        fragments = analysis["plastid"]["fragments"]
        metrics["mtpt_count"] = len(fragments)
        metrics["mtpt_bp"] = sum(f["length"] for f in fragments)
        metrics["mtpt_integrated"] = sum(f["integration"] == "supported" for f in fragments)
    flags = ["graph_rendered", "native_renderer"]
    flags.append("conformations_read_supported" if supported else "conformations_candidate_only")
    if analysis["plastid"]:
        flags.append("mtpt_annotated")
    findings = [
        {"metric": "nodes", "value": metrics["nodes"]},
        {"metric": "edges", "value": metrics["edges"]},
        {"metric": "repeats", "value": len(repeats)},
        {"metric": "supported_repeats", "value": len(supported)},
    ]
    if analysis["plastid"]:
        findings.append({"metric": "mtpt_bp", "value": metrics["mtpt_bp"], "unit": "bp"})
    plot = plot_result(
        "plot_assembly_graph",
        _renderer(metrics),
        organelle=resolved_organelle,
        method="organelleverse_graph",
        metrics=metrics,
        key_findings=tuple(findings),
        flags=tuple(flags),
        summary=_summary(metrics),
    )
    return _with_panel(plot, plot.metrics)


def _plastid_graph(plastid: Mapping[str, Any]) -> dict[str, Any]:
    """The plastid as assemblers draw it: LSC - IR - SSC with the IR collapsed."""
    reference = plastid["reference"]
    regions = reference["regions"]
    if not regions:
        row = {"name": "pt", "label": "plastid", "length": reference["length"], "region": "plastome"}
        return {"segments": [row], "links": [{"source": "pt+", "target": "pt+"}]}
    segments = [
        {"name": f"pt.{region}", "label": region, "length": regions[key]["length"], "region": region}
        for region, key in (("LSC", "LSC"), ("IR", "IRb"), ("SSC", "SSC"))
    ]
    links = [
        {"source": "pt.LSC+", "target": "pt.IR+"},
        {"source": "pt.IR+", "target": "pt.SSC+"},
        {"source": "pt.SSC+", "target": "pt.IR-"},
        {"source": "pt.IR-", "target": "pt.LSC+"},
    ]
    for row in segments:
        row.update(kind="plastid", color=PLASTID_REGION_COLORS[row["region"]])
    return {"segments": segments, "links": links}


def _plastid_position(position: float, plastid: Mapping[str, Any]) -> tuple[str, float]:
    """Plastid-graph segment and fraction of a 0-based donor position (IRa folds onto IR)."""
    reference = plastid["reference"]
    regions = reference["regions"]
    length = reference["length"]
    if not regions:
        return "pt", position / length
    for key, region in regions.items():
        offset = (position - region["start"]) % length
        if offset < region["length"]:
            fraction = offset / region["length"]
            if key == "IRa":
                return "pt.IR", 1 - fraction
            return f"pt.{'IR' if key == 'IRb' else key}", fraction
    raise AssertionError("quadripartite regions cover the plastome")


def _homology(plastid, plastid_graph, segments, min_length) -> list[dict[str, Any]]:
    """One connector per mitochondrial segment and plastid region they share MTPTs with.

    Every MTPT (>= ``min_length``) donor block is assigned to its plastid-graph
    segment; MTPTs between the same mitochondrial segment and plastid region
    are summed into one connector anchored at their length-weighted mean
    positions, so the figure shows where plastid sequence went, not each HSP.
    """
    lengths = {row["name"]: row["length"] for row in segments}
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for fragment in plastid["fragments"]:
        if fragment["length"] < min_length:
            continue
        mito_at = ((fragment["start"] - 1 + fragment["end"]) / 2) / lengths[fragment["segment"]]
        targets = {}
        for block in fragment["donor_blocks"]:
            target, at = _plastid_position((block["start"] - 1 + block["end"]) / 2, plastid)
            targets.setdefault(target, (at, block["identity"]))
        for target, (at, identity) in targets.items():
            group = groups.setdefault(
                (fragment["segment"], target),
                {"segment": fragment["segment"], "plastid_segment": target, "mtpts": [], "_w": 0.0,
                 "_mito": 0.0, "_pt": 0.0, "_ident": 0.0},
            )
            weight = fragment["length"]
            group["mtpts"].append(fragment["id"])
            group["_w"] += weight
            group["_mito"] += mito_at * weight
            group["_pt"] += at * weight
            group["_ident"] += identity * weight
    rows = []
    by_id = {f["id"]: f for f in plastid["fragments"]}
    for group in groups.values():
        weight = group.pop("_w")
        members = [by_id[i] for i in group["mtpts"]]
        assessed = [f for f in members if f["spanning_reads"] is not None]
        rows.append(
            {
                **group,
                "segment_at": round(group.pop("_mito") / weight, 4),
                "plastid_at": round(group.pop("_pt") / weight, 4),
                "identity": round(group.pop("_ident") / weight, 2),
                "total_length": int(weight),
                "mtpt_count": len(members),
                "assessed": len(assessed),
                "integrated": sum(f["integration"] == "supported" for f in assessed),
            }
        )
    return sorted(rows, key=lambda row: -row["total_length"])


def _frequency_source(evidence: Mapping[str, Any] | None) -> str:
    if evidence is None:
        return "none: no reads or evidence.json, conformations are candidates only"
    if evidence["source"] == "pairings":
        return "ovasm evidence.json: reads spanning each repeat between its entries and exits"
    return "long reads spanning each repeat with unique anchors on both sides"


def _number_repeats(repeats: list[dict[str, Any]], master: Sequence[Mapping[str, Any]] | None) -> None:
    """R1, R2, ... in order of first appearance along the master molecules."""
    order: dict[str, int] = {}
    for molecule in master or ():
        for label in molecule["steps"]:
            order.setdefault(parse_node_label(label)[0], len(order))
    repeats.sort(key=lambda row: (order.get(row["name"], len(order)), row["name"]))
    for index, row in enumerate(repeats):
        row["id"] = f"R{index + 1}"
        row["color"] = REPEAT_COLOR


def _summary(metrics: Mapping[str, Any]) -> str:
    text = (
        f"Assembly graph with {metrics['nodes']} segment(s), {metrics['edges']} link(s) and "
        f"{len(metrics['repeats'])} recombining repeat(s)"
    )
    parts = []
    for row in metrics["repeats"]:
        if row["status"] == "supported":
            freqs = " / ".join(_fraction(c) for c in row["configurations"])
            parts.append(f"{row['id']} {row['name']}: {freqs} junction-pairing reads")
    if parts:
        text += "; " + "; ".join(parts)
    if metrics.get("plastid"):
        text += (
            f"; {metrics['mtpt_count']} MTPT(s), {metrics['mtpt_bp']} bp, "
            f"{metrics['mtpt_integrated']} with integration-spanning reads"
        )
    return text + ". Render with ov.write(plot, output)."


def _inset_rank(row: Mapping[str, Any]) -> tuple:
    order = {"supported": 0, "insufficient_reads": 1, "candidate_only": 1, "not_spannable": 2}
    return (order.get(row["status"], 3), row["kind"] != "repeat", -row["length"])


def _display_names(names: Sequence[str], given: Mapping[str, str]) -> dict[str, str]:
    prefix = ""
    if len(names) > 1 and all("." in name for name in names):
        heads = {name.split(".", 1)[0] for name in names}
        if len(heads) == 1:
            prefix = next(iter(heads)) + "."
    return {name: str(given.get(name, name[len(prefix) :])) for name in names}



# ----------------------------------------------------------------------- render
#
# Figures are drawn at printed size (organelleverse.visualization.style): the
# point sizes below are what the reader sees. Panels sit on a regular grid with
# equal gutters, a bold lower-case letter and a left-aligned header on one
# baseline; the title and method notes go to <figure>.caption.txt.

_GUTTER_MM = 4.0
_LEFT_MM = 3.5  # room for the panel letters left of the first column
_HEADER_MM = 4.5
_LEGEND_ROW_MM = 3.8
_RIBBON = 3.4  # points, fixed for every segment
_RING = 2.4  # points, molecule rings in conformation panels
_LABEL_PT = max(6.5, FONT_PT["label"])  # segment, ring and note labels


def _with_panel(plot: OrganellePlot, metrics: Mapping[str, Any]) -> OrganellePlot:
    """Let ``compose_figure`` place the network as one panel of a larger figure."""
    from .plot_object import PanelDrawer

    plot._panel = PanelDrawer(
        lambda ax, legend: _draw_network_panel(ax, metrics, legend),
        aspect=_network_aspect(metrics),
        header=metrics.get("title") or "Assembly graph",
    )
    return plot


def _renderer(metrics: Mapping[str, Any]):
    def render(path: str | Path) -> Path:
        return _render_assembly_graph(Path(path), metrics)

    return render


def _restore_assembly_graph_plot(result) -> OrganellePlot:
    """Reattach rendering to a serialized, fully computed assembly-graph Result."""
    plot = OrganellePlot.model_validate(result.model_dump(mode="python"))
    plot._renderer = _renderer(plot.metrics)
    return _with_panel(plot, plot.metrics)


def _render_assembly_graph(path: Path, metrics: Mapping[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with publication_style():
        fig = _figure(metrics)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=int(metrics["dpi"]), facecolor="white")
        plt.close(fig)
    _write_tables(path, metrics)
    path.with_name(path.stem + ".caption.txt").write_text(_caption(metrics) + "\n", encoding="utf-8")
    return path


# -------------------------------------------------------------- figure layout


def _network_aspect(metrics) -> float:
    points = np.concatenate([np.asarray(v) for v in metrics["layout"].values()])
    span = np.ptp(points, axis=0) + 3.0
    return float(np.clip(span[1] / span[0], 0.3, 0.8))


def _figure(metrics):
    """Lettered grid: network on top, conformation panels below, legend last."""
    import matplotlib.pyplot as plt

    width_mm = width_inches(metrics.get("width", "double")) * MM_PER_INCH
    content_mm = width_mm - _LEFT_MM
    lower = _lower_panels(metrics)
    columns = len(lower) if metrics.get("style") == "triptych" else min(3, max(len(lower), 2))
    lower_rows = math.ceil(len(lower) / columns) if lower else 0
    panel_w = (content_mm - _GUTTER_MM * (columns - 1)) / columns
    lower_h = 62.0 if metrics.get("style") == "triptych" else panel_w * 0.68
    handles = _legend_handles(metrics)
    legend_cols = 3 if width_mm > 150 else 2
    legend_h = _LEGEND_ROW_MM * math.ceil(len(handles) / legend_cols) + 2.0
    fixed = (_HEADER_MM + _GUTTER_MM) * lower_rows + lower_h * lower_rows + _HEADER_MM + legend_h
    network_h = float(np.clip(content_mm * _network_aspect(metrics), 40.0, MAX_HEIGHT_MM - fixed))
    total_h = fixed + network_h
    fig = plt.figure(figsize=(width_mm / MM_PER_INCH, total_h / MM_PER_INCH))

    def axes(x_mm, top_mm, w_mm, h_mm):
        left = (_LEFT_MM + x_mm) / width_mm
        return fig.add_axes((left, 1 - (top_mm + h_mm) / total_h, w_mm / width_mm, h_mm / total_h))

    letters = iter("abcdefghijklmnopqrstuvwxyz")
    top = _HEADER_MM
    network = axes(0, top, content_mm, network_h)
    _draw_network_panel(network, metrics, False)
    _header(network, next(letters), "Assembly graph" + (" with plastid graph" if metrics.get("plastid_graph") else ""))
    top += network_h + _GUTTER_MM + _HEADER_MM
    for index, (title, draw) in enumerate(lower):
        row, column = divmod(index, columns)
        ax = axes(column * (panel_w + _GUTTER_MM), top + row * (lower_h + _GUTTER_MM + _HEADER_MM), panel_w, lower_h)
        draw(ax)
        _header(ax, next(letters), title)
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        ncol=legend_cols,
        fontsize=FONT_PT["body"],
        handlelength=1.2,
        handleheight=0.8,
        columnspacing=1.4,
        borderaxespad=0.2,
        frameon=False,
    )
    return fig


def _header(ax, letter: str, text: str) -> None:
    add_panel_label(ax, letter)
    ax.set_title(text, loc="left", fontsize=FONT_PT["body"], pad=3)


def _lower_panels(metrics) -> list[tuple[str, Any]]:
    if metrics.get("style") == "triptych":
        return [
            ("Representative circular path (assembled repeat pairings)", lambda ax: _draw_master(ax, metrics)),
            ("Candidate recombination paths (direct repeats)", lambda ax: _draw_subgenomic(ax, metrics)),
        ]
    rows = [row for row in metrics["repeats"] if row["name"] in metrics["insets"]]
    segments = {r["name"]: r for r in metrics["segments"]}
    return [
        (
            f"{row['id']} {segments[row['name']]['label']} · {_kb(row['length'])}, {row['depth']:.0f}\u00d7",
            (lambda r: lambda ax: _draw_repeat_panel(ax, r, metrics))(row),
        )
        for row in rows
    ]


# ---------------------------------------------------------------- network panel


def _midpoint(points) -> np.ndarray:
    chain = np.asarray(points)
    return chain[len(chain) // 2]


def _smooth(points: np.ndarray, samples: int = 12) -> np.ndarray:
    """Uniform Catmull-Rom curve through the chain points."""
    if len(points) < 3:
        return points
    padded = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])
    out = []
    t = np.linspace(0, 1, samples, endpoint=False)[:, None]
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        out.append(
            0.5
            * (
                2 * p1
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
            )
        )
    out.append(points[-1:])
    return np.vstack(out)


def _subcurve(curve: np.ndarray, start: float, end: float) -> np.ndarray:
    """Part of ``curve`` between fractions ``start`` and ``end`` of its arc length."""
    steps = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = cumulative[-1] or 1.0
    positions = np.linspace(start, end, max(3, int(40 * (end - start)) + 2)) * total
    xs = np.interp(positions, cumulative, curve[:, 0])
    ys = np.interp(positions, cumulative, curve[:, 1])
    return np.column_stack([xs, ys])


def _darker(color: str, factor: float = 0.66) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(color)
    return (r * factor, g * factor, b * factor)


def _end_geometry(points: np.ndarray, node: tuple[str, str], *, exit_end: bool):
    forward_end = (node[1] == "+") == exit_end
    tip, inner = (points[-1], points[-2]) if forward_end else (points[0], points[1])
    direction = tip - inner
    norm = np.linalg.norm(direction) or 1.0
    return tip, direction / norm


def _data_per_point(ax, lo, hi) -> float:
    """Layout units per printed point for an equal-aspect axes showing [lo, hi]."""
    fig_w, fig_h = ax.figure.get_size_inches()
    box = ax.get_position()
    width_pt, height_pt = box.width * fig_w * 72, box.height * fig_h * 72
    return max((hi[0] - lo[0]) / width_pt, (hi[1] - lo[1]) / height_pt)


def _draw_network_panel(ax, metrics, legend: bool) -> list:
    """Graph (with the plastid graph and MTPT connectors); legend handles when not drawn."""
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyArrowPatch, PathPatch
    from matplotlib.path import Path as MplPath

    ax.axis("off")
    plastid_graph = metrics.get("plastid_graph") or {"segments": [], "links": []}
    segments = {row["name"]: row for row in [*metrics["segments"], *plastid_graph["segments"]]}
    layout = {name: np.asarray(points) for name, points in metrics["layout"].items()}
    curves = {name: _smooth(points) for name, points in layout.items()}
    points = np.concatenate(list(curves.values()))
    lo, hi = points.min(axis=0), points.max(axis=0)
    pad = 0.06 * float(max(hi - lo))
    lo, hi = lo - pad, hi + pad
    scale = _data_per_point(ax, lo, hi)
    center = (lo + hi) / 2
    half = np.array([(ax.get_position().width * ax.figure.get_size_inches()[0] * 72),
                     (ax.get_position().height * ax.figure.get_size_inches()[1] * 72)]) * scale / 2
    ax.set_xlim(center[0] - half[0], center[0] + half[0])
    ax.set_ylim(center[1] - half[1], center[1] + half[1])
    ax.set_aspect("equal", adjustable="box", anchor="C")
    for link in [*metrics["links"], *plastid_graph["links"]]:
        source, target = parse_node_label(link["source"]), parse_node_label(link["target"])
        p0, u0 = _end_geometry(layout[source[0]], source, exit_end=True)
        p3, u3 = _end_geometry(layout[target[0]], target, exit_end=False)
        gap = float(np.linalg.norm(p3 - p0))
        if gap < 1e-6:
            normal = np.array([-u0[1], u0[0]])
            c1, c2 = p0 + 0.9 * (u0 + normal), p0 + 0.9 * (u0 - normal)
        else:
            reach = 0.45 * gap + 0.35
            c1, c2 = p0 + reach * u0, p3 + reach * u3
        path = MplPath([p0, c1, c2, p3], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        ax.add_patch(PathPatch(path, fill=False, lw=0.7, ec=LINK_COLOR, capstyle="round", zorder=1))
    for name, row in segments.items():
        curve = curves[name]
        outline = _darker(row["color"])
        ax.plot(curve[:, 0], curve[:, 1], color=outline, lw=_RIBBON + 0.8, solid_capstyle="butt", zorder=2)
        ax.plot(curve[:, 0], curve[:, 1], color=row["color"], lw=_RIBBON, solid_capstyle="butt", zorder=3)
        ax.add_patch(
            FancyArrowPatch(
                curve[max(0, len(curve) - 4)],
                curve[-1],
                arrowstyle="-|>,head_length=0.5,head_width=0.35",
                mutation_scale=6,
                color=outline,
                lw=0,
                shrinkA=0,
                shrinkB=0,
                zorder=5,
            )
        )
    plastid = metrics.get("plastid")
    if plastid:
        for fragment in plastid["fragments"]:
            length = segments[fragment["segment"]]["length"]
            piece = _subcurve(curves[fragment["segment"]], (fragment["start"] - 1) / length, fragment["end"] / length)
            ax.plot(piece[:, 0], piece[:, 1], color=MTPT_COLOR, lw=_RIBBON, solid_capstyle="butt", zorder=4)
        _draw_homology(ax, metrics, curves, scale)
    _place_labels(ax, segments, curves, scale, pe)
    return [] if legend else _legend_handles(metrics)


def _place_labels(ax, segments, curves, scale, pe) -> None:
    """Name + depth beside each ribbon, avoiding ribbons and earlier labels (in points)."""
    occupied = np.concatenate(list(curves.values()))
    taken: list[tuple[np.ndarray, float, float]] = []
    for name, row in sorted(segments.items(), key=lambda item: -item[1]["length"]):
        curve = curves[name]
        emphasis = row["kind"] != "unique"
        size = _LABEL_PT + (0.5 if emphasis else 0.0)
        if row["kind"] == "plastid":
            text = f"{row['label']} {_kb(row['length'])}"
        else:
            badge = f"{row['repeat_id']} " if row.get("repeat_id") else ""
            text = f"{badge}{row['label']} {row['depth']:.1f}\u00d7"
        half_w = 0.29 * size * len(text) * scale
        half_h = 0.65 * size * scale
        best = None
        for fraction in (0.5, 0.35, 0.65, 0.2, 0.8):
            i = min(len(curve) - 2, int(fraction * (len(curve) - 1)))
            tangent = curve[i + 1] - curve[i]
            tangent /= np.linalg.norm(tangent) or 1.0
            normal = np.array([-tangent[1], tangent[0]])
            for side in (1, -1):
                for distance_pt in (_RIBBON / 2 + 1.5 + size * 0.8, _RIBBON / 2 + 3 + size * 1.6):
                    spot = curve[i] + side * distance_pt * scale * normal
                    inside = (np.abs(occupied[:, 0] - spot[0]) < half_w) & (np.abs(occupied[:, 1] - spot[1]) < half_h)
                    clash = sum(
                        abs(spot[0] - other[0]) < half_w + other_w and abs(spot[1] - other[1]) < half_h + other_h
                        for other, other_w, other_h in taken
                    )
                    score = 6 * clash + int(inside.sum()) + 0.05 * distance_pt
                    if best is None or score < best[0]:
                        best = (score, spot)
        assert best is not None
        spot = best[1]
        taken.append((spot, half_w, half_h))
        ax.text(
            spot[0],
            spot[1],
            text,
            ha="center",
            va="center",
            fontsize=size,
            fontweight="bold" if emphasis else "normal",
            color=row["color"] if emphasis else INK,
            zorder=7,
            path_effects=[pe.withStroke(linewidth=1.6, foreground="white")],
        )


def _curve_point(curve: np.ndarray, fraction: float) -> np.ndarray:
    piece = _subcurve(curve, max(0.0, fraction - 0.02), min(1.0, fraction + 0.02))
    return piece[len(piece) // 2]


def _draw_homology(ax, metrics, curves, scale) -> None:
    """Dashed MTPT -> plastid connectors: sequence homology, not assembly links."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    # Integration read counts are in the caption and <figure>.mtpt.tsv: at print
    # size one label per connector cannot be placed without collisions.
    rows = metrics.get("homology") or []
    longest = max((row["total_length"] for row in rows), default=1)
    for row in rows:
        a = _curve_point(curves[row["segment"]], row["segment_at"])
        b = _curve_point(curves[row["plastid_segment"]], row["plastid_at"])
        direction = b - a
        control = (a + b) / 2 + np.array([-direction[1], direction[0]]) * 0.12
        path = MplPath([a, control, b], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        unsupported = row["assessed"] > 0 and row["integrated"] == 0
        ax.add_patch(
            PathPatch(
                path,
                fill=False,
                ec=UNSUPPORTED_COLOR if unsupported else MTPT_COLOR,
                lw=0.5 + 1.6 * math.sqrt(row["total_length"] / longest),
                ls=(0, (1, 1.5)) if unsupported else (0, (3, 2)),
                alpha=0.75,
                capstyle="round",
                zorder=1.2,
            )
        )


def _legend_handles(metrics) -> list:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    def patch(color, label):
        return Patch(fc=color, ec=_darker(color), lw=0.4, label=label)

    handles = []
    if metrics["color_by"] == "origin":
        handles.append(patch(ORIGIN_COLORS["mitochondrial"], "Mitochondrial"))
    if metrics.get("plastid"):
        handles += [
            patch(PLASTID_REGION_COLORS["LSC"], "Plastid (LSC / IR / SSC)"),
            patch(MTPT_COLOR, "Plastid-derived in mitochondrion (MTPT)"),
        ]
    if any(row.get("origin") == "shared" for row in metrics["segments"]):
        handles.append(patch(ORIGIN_COLORS["shared"], "Shared (≥90% plastid-identical)"))
    if metrics["repeats"]:
        handles.append(patch(REPEAT_COLOR, f"Repeat (R1\u2013R{len(metrics['repeats'])}, recombining)"))
    handles.append(Line2D([], [], color=LINK_COLOR, lw=0.9, label="Assembly-graph link"))
    if metrics.get("homology"):
        handles.append(
            Line2D([], [], color=MTPT_COLOR, lw=1.0, ls=(0, (3, 2)),
                   label="MTPT sequence homology, summed per region (not a graph link)")
        )
        if any(row["assessed"] and not row["integrated"] for row in metrics["homology"]):
            handles.append(
                Line2D([], [], color=UNSUPPORTED_COLOR, lw=1.0, ls=(0, (1, 1.5)), label="MTPT without spanning reads")
            )
    return handles


def _caption(metrics) -> str:
    parts = []
    if metrics.get("title"):
        parts.append(f"{metrics['title']}.")
    parts.append(
        f"Mitochondrial assembly graph ({metrics['nodes']} segments, {metrics['edges']} links); "
        f"labels give segment name and depth; single-copy depth {metrics['single_copy_depth']:.1f}\u00d7."
    )
    evidence = metrics.get("read_evidence")
    if evidence and evidence["source"] == "reads":
        parts.append(
            "Percentages are junction-pairing read support at each repeat, not abundances of whole "
            f"circular molecules: a/b gives a, the {_READ_NAMES[evidence['read_type']]} reads whose alignment "
            f"covers the repeat plus ≥{evidence['anchor']} bp of unique flank on both sides and whose best-scoring "
            "flank pair belongs to that conformation's two predecessor-repeat-successor pairings, and b, "
            f"all such reads of the repeat (minimap2 {evidence['preset']}; ties excluded)."
        )
    elif evidence:
        parts.append(
            "Percentages are junction-pairing read support at each repeat, not abundances of whole circular "
            "molecules: a/b gives the ovasm evidence.json pairing reads (entry-repeat-exit traversals) "
            "of that conformation over those of both conformations."
        )
    else:
        parts.append("No read evidence: conformations are candidates and frequencies were not estimated.")
    parts.append(
        "Circular paths are spelled from the graph with every other repeat in its assembled pairing; "
        "no read spans a whole path, so they are representative and candidate recombination paths, "
        "not observed molecules."
    )
    plastid = metrics.get("plastid")
    if plastid:
        parts.append(
            f"MTPTs: {plastid['backend']}, ≥{plastid['min_identity']:g}% identity, ≥{plastid['min_length']} bp; "
            "each dashed connector sums all MTPTs between one mitochondrial segment and one plastid region "
            "(sequence homology, not an assembly link; width scales with total MTPT length). The plastid is drawn as its "
            "quadripartite graph (LSC\u2013IR\u2013SSC, IR collapsed) from the representative molecule."
        )
        assessed = [f for f in plastid["fragments"] if f["spanning_reads"] is not None]
        if assessed:
            counts = [f["spanning_reads"] for f in assessed]
            supported = sum(f["integration"] == "supported" for f in assessed)
            parts.append(
                f"{supported} of {len(assessed)} assessable MTPTs are spanned into mitochondrial sequence on both "
                f"sides by ≥{evidence['min_spanning_reads']} reads ({min(counts)}\u2013{max(counts)} reads; "
                "per-MTPT counts in the .mtpt.tsv table); "
                f"{len(plastid['fragments']) - len(assessed)} lie within {evidence['anchor']} bp of a segment end and were not assessed."
            )
    if metrics.get("linearization_decisive") is False:
        parts.append("The ovasm linearization was not decisive: the assembled path is one of equally supported paths.")
    return " ".join(parts)


# ---------------------------------------------------------- conformation panels


def _fraction(conf) -> str:
    return f"{conf['supporting_reads']}/{conf['proportion_denominator']} ({100 * conf['proportion']:.1f}%)"


def _frequency_text(row, conf, *, unit: bool = True) -> str:
    """Junction-pairing read support as numerator/denominator (percentage)."""
    if conf.get("proportion") is not None:
        if not unit:
            return _fraction(conf)
        noun = "pairing reads" if row.get("support_source") == "pairings" else "spanning reads"
        return f"{conf['supporting_reads']}/{conf['proportion_denominator']} {noun} ({100 * conf['proportion']:.1f}%)"
    if row["status"] == "not_spannable" or conf.get("supporting_reads") is None:
        return "candidate (no spanning reads)" if row["status"] == "not_spannable" else "candidate"
    return f"{conf['supporting_reads']}/{conf['proportion_denominator']} reads (below minimum)"


def _status_text(row) -> str:
    status = row["status"]
    source = " (ovasm evidence)" if row.get("support_source") == "pairings" else ""
    if status == "supported":
        noun = "pairing reads" if row.get("support_source") == "pairings" else "spanning reads"
        return f"a/b of {row['spanning_reads']} {noun}{source}"
    if status == "not_spannable":
        return f"not spannable: longest read {_kb(row['longest_touching_read'])} < {_kb(row['required_read_length'])}"
    if status == "insufficient_reads":
        extra = f", {row['ambiguous_reads']} ambiguous" if row.get("ambiguous_reads") else ""
        return f"{row['spanning_reads']} spanning reads{extra}{source}"
    if status == "complex":
        return "not a 2\u00d72 repeat"
    return "no read evidence: candidate conformations"


def _kb(length: int) -> str:
    return f"{length / 1000:.1f} kb" if length >= 1000 else f"{length} bp"


def _draw_repeat_panel(ax, row, metrics) -> None:
    """Both conformations of one repeat side by side, junction-pairing support underneath."""
    ax.axis("off")
    segments = {r["name"]: r for r in metrics["segments"]}
    confs = row["configurations"]
    for index, conf in enumerate(confs):
        left = index / len(confs)
        ring_ax = ax.inset_axes((left + 0.02, 0.28, 1 / len(confs) - 0.04, 0.62))
        ring_ax.axis("off")
        molecules = conf.get("molecules")
        if molecules is None:
            _draw_local_pairs(ring_ax, conf, row, segments)
        else:
            _draw_molecule_row(ring_ax, molecules, segments, metrics)
        center = left + 0.5 / len(confs)
        ax.text(center, 0.2, f"Conf. {index + 1}" + (" (assembled)" if conf["assembled"] else ""),
                transform=ax.transAxes, ha="center", va="center", fontsize=_LABEL_PT, color=INK,
                fontweight="bold" if conf["assembled"] else "normal")
        ax.text(center, 0.09, _frequency_text(row, conf, unit=False), transform=ax.transAxes, ha="center", va="center",
                fontsize=FONT_PT["body"], fontweight="bold" if conf.get("proportion") is not None else "normal",
                color=INK if conf.get("proportion") is not None else MUTED)
    ax.text(0.0, 1.0, _status_text(row), transform=ax.transAxes, ha="left", va="top",
            fontsize=_LABEL_PT, color=MUTED)


def _draw_molecule_row(ax, molecules, segments, metrics, *, plus=False, ring=_RING) -> None:
    largest = max(m["length"] for m in molecules)
    radii = [0.45 + 0.55 * math.sqrt(m["length"] / largest) for m in molecules]
    x = 0.0
    centers = []
    for index, radius in enumerate(radii):
        if index:
            x += 0.75 if plus else 0.4
        x += radius
        centers.append(x)
        x += radius
    ax.set_xlim(-0.1, x + 0.1)
    ax.set_ylim(-1.35, 1.1)
    for index, (molecule, radius, cx) in enumerate(zip(molecules, radii, centers, strict=True)):
        if molecule["circular"]:
            _draw_ring(ax, molecule, (cx, 0.0), radius, segments, metrics, width=ring)
        else:
            _draw_linear(ax, molecule, (cx, 0.0), radius, segments)
        if plus and index:
            ax.text(cx - radius - 0.37, 0, "+", ha="center", va="center", fontsize=FONT_PT["title"], color=MUTED)
    ax.set_aspect("equal", adjustable="datalim", anchor="C")


def _arc_spans(molecule, segments) -> list[tuple[str, str, float, float]]:
    steps = [parse_node_label(label) for label in molecule["steps"]]
    lengths = np.array([segments[name]["length"] for name, _ in steps], dtype=float)
    share = np.maximum(lengths / lengths.sum(), 0.03)
    share /= share.sum()
    spans = []
    start = 0.0
    for (name, orient), part in zip(steps, share, strict=True):
        spans.append((name, orient, start, start + float(part)))
        start += float(part)
    return spans


def _ring_points(center, radius, start, end, samples=60):
    theta = np.linspace(math.pi / 2 - 2 * math.pi * start, math.pi / 2 - 2 * math.pi * end, samples)
    return center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)


def _draw_ring(ax, molecule, center, radius, segments, metrics, *, width=_RING, label=True) -> list:
    """Ring of coloured steps (MTPTs in green); returns step spans for annotation."""
    cx, cy = center
    spans = _arc_spans(molecule, segments)
    fragments = (metrics.get("plastid") or {}).get("fragments", [])
    for name, orient, start, end in spans:
        row = segments[name]
        lw = width + (0.6 if row["kind"] != "unique" else 0)
        xs, ys = _ring_points(center, radius, start, end)
        ax.plot(xs, ys, color=_darker(row["color"]), lw=lw + 0.6, solid_capstyle="butt")
        ax.plot(xs, ys, color=row["color"], lw=lw, solid_capstyle="butt")
        for fragment in fragments:
            if fragment["segment"] != name:
                continue
            lo, hi = (fragment["start"] - 1) / row["length"], fragment["end"] / row["length"]
            if orient == "-":
                lo, hi = 1 - hi, 1 - lo
            xs, ys = _ring_points(center, radius, start + lo * (end - start), start + hi * (end - start), 12)
            ax.plot(xs, ys, color=MTPT_COLOR, lw=lw, solid_capstyle="butt")
        x0, y0 = _ring_points(center, radius, start, start, 1)
        ax.plot([cx + (x0[0] - cx) * 0.93, cx + (x0[0] - cx) * 1.07],
                [cy + (y0[0] - cy) * 0.93, cy + (y0[0] - cy) * 1.07], color="white", lw=0.5)
    if label:
        text = _kb(molecule["length"])
        if 0.62 * _LABEL_PT * len(text) < 1.6 * radius * _points_per_unit(ax) - 2 * width:
            ax.text(cx, cy, text, ha="center", va="center", fontsize=_LABEL_PT, color=INK)
        else:
            ax.text(cx, cy - radius - 0.12, text, ha="center", va="top", fontsize=_LABEL_PT, color=INK)
    return spans


def _points_per_unit(ax) -> float:
    """Printed points per data unit of an equal-aspect axes (before autoscaling)."""
    fig_w, fig_h = ax.figure.get_size_inches()
    box = ax.get_position()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return min(box.width * fig_w * 72 / (x1 - x0), box.height * fig_h * 72 / (y1 - y0))


def _draw_linear(ax, molecule, center, radius, segments) -> None:
    cx, cy = center
    for name, _, start, end in _arc_spans(molecule, segments):
        xs = [cx - radius + start * 2 * radius, cx - radius + end * 2 * radius]
        color = segments[name]["color"]
        ax.plot(xs, [cy, cy], color=_darker(color), lw=_RING + 0.6, solid_capstyle="butt")
        ax.plot(xs, [cy, cy], color=color, lw=_RING, solid_capstyle="butt")
    ax.text(cx, cy - 0.3, _kb(molecule["length"]) + " (linear)", ha="center", va="center",
            fontsize=_LABEL_PT, color=INK)


def _draw_local_pairs(ax, conf, row, segments) -> None:
    """When the graph does not spell molecules: draw the p - R - s pairs."""
    for index, (left, right) in enumerate(conf["pairs"]):
        y = 0.45 - index * 0.9
        for x0, x1, label in ((-1.5, -0.55, left), (-0.5, 0.5, f"{row['name']}+"), (0.55, 1.5, right)):
            name = parse_node_label(label)[0]
            color = segments[name]["color"]
            ax.plot([x0, x1], [y, y], color=_darker(color), lw=_RING + 0.6, solid_capstyle="butt")
            ax.plot([x0, x1], [y, y], color=color, lw=_RING, solid_capstyle="butt")
            ax.text((x0 + x1) / 2, y + 0.25, segments[name]["label"], ha="center", fontsize=_LABEL_PT, color=INK)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal", adjustable="datalim", anchor="C")


def _draw_master(ax, metrics) -> None:
    ax.axis("off")
    segments = {row["name"]: row for row in metrics["segments"]}
    molecules = metrics.get("master_molecules")
    if not molecules:
        ax.text(0.5, 0.5, "The graph does not spell a unique circular path\n(unresolved branching).",
                transform=ax.transAxes, ha="center", va="center", fontsize=FONT_PT["body"], color=MUTED)
        return
    repeats = {row["name"]: row for row in metrics["repeats"]}
    main = max(molecules, key=lambda m: (sum(parse_node_label(s)[0] in repeats for s in m["steps"]), m["length"]))
    others = [m for m in molecules if m is not main]
    radii = [0.22 + 0.38 * math.sqrt(m["length"] / main["length"]) for m in others]
    ax.set_xlim(-1.4, max(1.4, 1.5 + sum(0.45 + 2 * r for r in radii)))
    ax.set_ylim(-1.55, 1.35)
    spans = _draw_ring(ax, main, (0.0, 0.0), 1.0, segments, metrics, width=3.2, label=False)
    copies: dict[str, list[float]] = {}
    for name, _, start, end in spans:
        if name in repeats:
            copies.setdefault(name, []).append((start + end) / 2)
    for name, positions in copies.items():
        row = repeats[name]
        for position in positions:
            angle = math.pi / 2 - 2 * math.pi * position
            ax.text(1.2 * math.cos(angle), 1.2 * math.sin(angle), row["id"], ha="center", va="center",
                    fontsize=_LABEL_PT + 0.5, fontweight="bold", color=row["color"])
        for a, b in itertools.pairwise(positions):
            pa = np.array([math.cos(math.pi / 2 - 2 * math.pi * a), math.sin(math.pi / 2 - 2 * math.pi * a)]) * 0.92
            pb = np.array([math.cos(math.pi / 2 - 2 * math.pi * b), math.sin(math.pi / 2 - 2 * math.pi * b)]) * 0.92
            _chord(ax, pa, pb, row["color"])
    ax.text(0, -1.3, f"representative path \u00b7 {_kb(main['length'])}", ha="center", va="top",
            fontsize=FONT_PT["body"], color=INK)
    x = 1.45
    for molecule, radius in zip(others, radii, strict=True):
        ax.text(x + 0.12, -0.75, "+", ha="center", va="center", fontsize=FONT_PT["title"], color=MUTED)
        _draw_ring(ax, molecule, (x + 0.35 + radius, -0.75), radius, segments, metrics)
        x += 0.45 + 2 * radius
    ax.set_aspect("equal", adjustable="datalim", anchor="C")


def _chord(ax, a, b, color) -> None:
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    control = (a + b) * 0.25
    path = MplPath([a, control, b], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    ax.add_patch(PathPatch(path, fill=False, ec=color, lw=0.7, ls=(0, (3, 2))))


def _draw_subgenomic(ax, metrics) -> None:
    ax.axis("off")
    segments = {row["name"]: row for row in metrics["segments"]}
    rows = []
    isomers = []
    for repeat in metrics["repeats"]:
        assembled = next((c for c in repeat["configurations"] if c["assembled"]), None)
        for conf in repeat["configurations"]:
            if conf is assembled or not conf.get("molecules"):
                continue
            base = len(assembled["molecules"]) if assembled and assembled.get("molecules") else 1
            if len(conf["molecules"]) > base:
                rows.append((repeat, conf))
            elif assembled and assembled.get("molecules"):
                isomers.append(repeat)
    if not rows:
        ax.text(0.0, 0.6, "No direct repeat splits the representative path.", transform=ax.transAxes,
                fontsize=FONT_PT["body"], color=MUTED, va="center")
    shown = rows[:4]
    height = min(0.88 / max(len(shown), 1), 0.3)
    for index, (repeat, conf) in enumerate(shown):
        top = 1.0 - index * height
        ax.text(0.0, top - height * 0.25,
                f"{repeat['id']} {segments[repeat['name']]['label']} \u00b7 {_kb(repeat['length'])} direct repeat",
                transform=ax.transAxes, fontsize=FONT_PT["body"], fontweight="bold", color=repeat["color"], va="center")
        ax.text(0.0, top - height * 0.5, "recombinant pairing support:", transform=ax.transAxes,
                fontsize=_LABEL_PT, color=MUTED, va="center")
        ax.text(0.0, top - height * 0.75, _frequency_text(repeat, conf), transform=ax.transAxes,
                fontsize=FONT_PT["body"], fontweight="bold" if conf.get("proportion") is not None else "normal",
                color=INK if conf.get("proportion") is not None else MUTED, va="center")
        ring_ax = ax.inset_axes((0.46, top - height, 0.54, height * 0.96))
        ring_ax.axis("off")
        _draw_molecule_row(ring_ax, conf["molecules"], segments, metrics, plus=True)
    notes = []
    if len(rows) > len(shown):
        notes.append(f"{len(rows) - len(shown)} more in the table")
    if isomers:
        parts = []
        for repeat in isomers:
            props = [c.get("proportion") for c in repeat["configurations"]]
            text = (
                " / ".join(_fraction(c) for c in repeat["configurations"])
                if all(p is not None for p in props)
                else "candidate"
            )
            parts.append(f"{repeat['id']} {text}")
        notes.append("Inverted repeats (isomeric paths, no split): " + "; ".join(parts))
    if notes:
        ax.text(0.0, 0.0, "   ".join(notes), transform=ax.transAxes, fontsize=_LABEL_PT, color=MUTED, va="bottom")


# ----------------------------------------------------------------------- tables


def _write_tables(path: Path, metrics) -> None:
    header = [
        "repeat_id",
        "repeat",
        "kind",
        "length",
        "depth",
        "copy_ratio",
        "status",
        "support_source",
        "configuration",
        "assembled",
        "pairs",
        "supporting_reads",
        "proportion",
        "molecules",
    ]
    lines = ["\t".join(header)]
    for row in metrics["repeats"]:
        for index, conf in enumerate(row["configurations"], start=1):
            molecules = conf.get("molecules")
            described = (
                "unresolved"
                if molecules is None
                else ";".join(f"{'circular' if m['circular'] else 'linear'}:{m['length']}" for m in molecules)
            )
            values = (
                row["id"],
                row["name"],
                row["kind"],
                row["length"],
                row["depth"],
                row["copy_ratio"],
                row["status"],
                row.get("support_source") or "",
                index,
                conf["assembled"],
                ",".join(f"{a}>{row['name']}>{b}" for a, b in conf["pairs"]),
                "" if conf["supporting_reads"] is None else conf["supporting_reads"],
                "" if conf["proportion"] is None else conf["proportion"],
                described,
            )
            lines.append("\t".join(str(v) for v in values))
    path.with_name(path.stem + ".conformations.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plastid = metrics.get("plastid")
    if plastid:
        rows = ["\t".join(["id", "segment", "start", "end", "length", "identity", "strand", "donor_blocks", "spanning_reads", "integration"])]
        for f in plastid["fragments"]:
            blocks = ";".join(f"{b['start']}-{b['end']}:{b['identity']}" for b in f["donor_blocks"])
            rows.append(
                "\t".join(
                    str(v)
                    for v in (
                        f["id"], f["segment"], f["start"], f["end"], f["length"], f["identity"], f["strand"],
                        blocks, "" if f["spanning_reads"] is None else f["spanning_reads"], f["integration"],
                    )
                )
            )
        path.with_name(path.stem + ".mtpt.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


__all__ = ["graph_layout", "plot_assembly_graph"]
