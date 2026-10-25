"""Readable skeleton of a pangenome GFA, drawn by OrganelleVerse.

A PGGB/minigraph graph of a dozen organelle genomes has thousands of nodes,
most of them the few bases either side of a SNP or small indel. Drawn as is
(Bandage) that is a hairball; the centre of a pangenome circle needs the
large-scale structure instead: which stretches every genome shares, which
only some carry, and how they are joined.

``pangenome_skeleton`` keeps nodes of at least ``min_block`` bp as anchors,
links anchors that follow each other in some genome's path (skipping the
short nodes between them, so small-variant bubbles collapse onto one link),
closes each path into a circle for circular organelle genomes, and merges
unbranched runs of anchors carried by the same genomes into one block. The
threshold grows until at most ``max_blocks`` blocks remain. Each block keeps
its length, how many genomes carry it (core = all, private = one, shell =
between) and its span on the reference path; each link keeps how many genomes
traverse it.

``draw_pangenome_graph`` lays the blocks out with the assembly-graph stress
layout (``assembly_graph.graph_layout``) and draws them as ribbons inside a
circle of a given radius on an existing axes.
"""

from __future__ import annotations

import gzip
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import OrganelleInputError

Oriented = tuple[str, str]

#: Presence classes, shared with the legend of the pan circle.
CLASS_COLORS = {"core": "#2A6F97", "shell": "#E9A23B", "private": "#C8553D"}
CLASS_LABELS = {
    "core": "core (all genomes)",
    "shell": "shell (some genomes)",
    "private": "private (one genome)",
}
LINK_COLOR = "#8A94A3"


def _flip(node: Oriented) -> Oriented:
    return (node[0], "-" if node[1] == "+" else "+")


def _canonical(u: Oriented, v: Oriented) -> tuple[Oriented, Oriented]:
    """One key for a bidirected edge and its reverse-complement reading (more '+' wins)."""
    reverse = (_flip(v), _flip(u))
    return min((u, v), reverse, key=lambda e: ((e[0][1] == "-") + (e[1][1] == "-"), e))


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def read_gfa_paths(path: str | Path) -> tuple[dict[str, int], dict[str, list[Oriented]]]:
    """Segment lengths and oriented paths (GFA1 ``P`` and GFA1.1 ``W`` lines)."""
    lengths: dict[str, int] = {}
    paths: dict[str, list[Oriented]] = {}
    with _open(Path(path)) as handle:
        for line in handle:
            kind = line[:2]
            if kind == "S\t":
                fields = line.rstrip("\n").split("\t")
                length = len(fields[2]) if fields[2] != "*" else 0
                for tag in fields[3:]:
                    if tag.startswith("LN:i:"):
                        length = int(tag[5:])
                lengths[fields[1]] = length
            elif kind == "P\t":
                fields = line.rstrip("\n").split("\t")
                paths[fields[1]] = [(step[:-1], step[-1]) for step in fields[2].split(",") if step]
            elif kind == "W\t":
                fields = line.rstrip("\n").split("\t")
                name = f"{fields[1]}#{fields[2]}#{fields[3]}"
                walk = fields[6]
                steps: list[Oriented] = []
                i = 0
                while i < len(walk):
                    orient = "+" if walk[i] == ">" else "-"
                    j = i + 1
                    while j < len(walk) and walk[j] not in "<>":
                        j += 1
                    steps.append((walk[i + 1 : j], orient))
                    i = j
                paths[name] = steps
    if not lengths:
        raise OrganelleInputError(
            code="visualization.pan_graph.empty",
            message="The pangenome GFA has no segments.",
            details={"path": str(path)},
        )
    if not paths:
        raise OrganelleInputError(
            code="visualization.pan_graph.no_paths",
            message="The pangenome GFA has no P/W paths, so genome membership is unknown.",
            details={"path": str(path)},
        )
    return lengths, paths


def path_sample(name: str) -> str:
    """PAN-SN sample of a path name (``sample#haplotype#contig``)."""
    return name.split("#", 1)[0]


def pangenome_skeleton(
    gfa: str | Path,
    *,
    max_blocks: int = 60,
    min_block: int | None = None,
    circular: bool = True,
    reference: str | None = None,
    growth: float = 1.5,
) -> dict[str, Any]:
    """Collapse a pangenome GFA into at most ``max_blocks`` blocks (see module doc).

    ``reference`` picks the path whose coordinates the blocks report (a path
    name, a PAN-SN sample, or a substring of one); the first path otherwise.
    """
    lengths, raw_paths = read_gfa_paths(gfa)
    samples = sorted({path_sample(name) for name in raw_paths})
    bit = {sample: 1 << i for i, sample in enumerate(samples)}
    ref_name = _pick_reference(raw_paths, reference)

    units: dict[str, _Unit] = {}
    for name, steps in raw_paths.items():
        b = bit[path_sample(name)]
        for node, _ in steps:
            unit = units.get(node)
            if unit is None:
                units[node] = _Unit(lengths.get(node, 0), b)
            else:
                unit.mask |= b
    offset = 0
    for node, _ in raw_paths[ref_name]:
        unit = units[node]
        if unit.ref_start is None:
            unit.ref_start, unit.ref_end = offset, offset + unit.length
        offset += unit.length
    paths = {name: list(steps) for name, steps in raw_paths.items()}

    total_bp = sum(lengths.values())
    # Round 0 only merges unbranched runs; later rounds drop ever longer short
    # blocks (bubbles of small variants first) and merge what they left behind.
    target = int(min_block) if min_block else None
    threshold = 0
    while True:
        units, paths = _coarsen(units, paths, threshold, circular)
        if target is not None:
            if threshold >= target:
                break
        elif len(units) <= max_blocks or threshold >= total_bp:
            break
        threshold = max(10, math.ceil(threshold * growth))
        if target is not None:
            threshold = min(threshold, target)
    blocks, links = _finish(units, paths, bit, ref_name, circular)
    return {
        "genomes": len(samples),
        "samples": samples,
        "reference_path": ref_name,
        "raw_nodes": len(lengths),
        "raw_bp": total_bp,
        "min_block": threshold,
        "circular": circular,
        "represented_bp": sum(b["length"] for b in blocks),
        "blocks": blocks,
        "links": links,
    }


class _Unit:
    __slots__ = ("length", "mask", "ref_end", "ref_start")

    def __init__(self, length: int, mask: int, ref_start=None, ref_end=None) -> None:
        self.length = length
        self.mask = mask
        self.ref_start = ref_start
        self.ref_end = ref_end


def _pick_reference(paths: Mapping[str, Any], reference: str | None) -> str:
    names = list(paths)
    if reference:
        for test in (
            lambda n: n == reference,
            lambda n: path_sample(n) == reference,
            lambda n: reference in n,
        ):
            hit = next((n for n in names if test(n)), None)
            if hit is not None:
                return hit
    return names[0]


def _path_pairs(steps: list[Oriented], circular: bool):
    pairs = list(pairwise(steps))
    if circular and steps:
        # a genome collapsed into one block still closes on itself
        pairs.append((steps[-1], steps[0]))
    return pairs


def _coarsen(units, paths, threshold, circular):
    """Drop units shorter than ``threshold``, then merge unbranched same-genome runs."""
    kept = {u for u, unit in units.items() if unit.length >= threshold}
    paths = {name: [s for s in steps if s[0] in kept] for name, steps in paths.items()}
    out: dict[Oriented, set[Oriented]] = {}
    into: dict[Oriented, set[Oriented]] = {}
    for steps in paths.values():
        for u, v in _path_pairs(steps, circular):
            for a, c in ((u, v), (_flip(v), _flip(u))):
                out.setdefault(a, set()).add(c)
                into.setdefault(c, set()).add(a)

    def extend(chain: list[Oriented], visited: set[str], mask: int) -> None:
        cur = chain[-1]
        while True:
            nexts = out.get(cur, ())
            if len(nexts) != 1:
                return
            nxt = next(iter(nexts))
            if nxt[0] in visited or into.get(nxt) != {cur} or units[nxt[0]].mask != mask:
                return
            chain.append(nxt)
            visited.add(nxt[0])
            cur = nxt

    ref_order = sorted(
        kept,
        key=lambda u: (units[u].ref_start if units[u].ref_start is not None else math.inf, u),
    )
    visited: set[str] = set()
    place: dict[str, tuple[str, str]] = {}  # unit -> (merged id, its orientation in the run)
    merged: dict[str, _Unit] = {}
    for unit_id in ref_order:
        if unit_id in visited:
            continue
        visited.add(unit_id)
        forward = [(unit_id, "+")]
        extend(forward, visited, units[unit_id].mask)
        backward = [(unit_id, "-")]
        extend(backward, visited, units[unit_id].mask)
        chain = [_flip(x) for x in reversed(backward[1:])] + forward
        new_id = f"u{len(merged)}"
        members = [units[x[0]] for x in chain]
        starts = [m.ref_start for m in members if m.ref_start is not None]
        ends = [m.ref_end for m in members if m.ref_end is not None]
        merged[new_id] = _Unit(
            sum(m.length for m in members),
            members[0].mask,
            min(starts) if starts else None,
            max(ends) if ends else None,
        )
        for member, orient in chain:
            place[member] = (new_id, orient)
    rewritten: dict[str, list[Oriented]] = {}
    for name, steps in paths.items():
        new_steps: list[Oriented] = []
        for unit_id, orient in steps:
            new_id, run_orient = place[unit_id]
            step = (new_id, "+" if orient == run_orient else "-")
            if not new_steps or new_steps[-1] != step:
                new_steps.append(step)
        if circular and len(new_steps) > 1 and new_steps[0] == new_steps[-1]:
            new_steps.pop()
        rewritten[name] = new_steps
    return merged, rewritten


def _finish(units, paths, bit, ref_name, circular):
    genome_count = len(bit)
    order = sorted(
        units,
        key=lambda u: (
            units[u].ref_start if units[u].ref_start is not None else math.inf,
            -units[u].length,
            u,
        ),
    )
    names = {unit_id: f"b{i}" for i, unit_id in enumerate(order, 1)}
    blocks = []
    for unit_id in order:
        unit = units[unit_id]
        genomes = bin(unit.mask).count("1")
        blocks.append(
            {
                "name": names[unit_id],
                "length": unit.length,
                "genomes": genomes,
                "class": _presence_class(genomes, genome_count),
                "ref_start": unit.ref_start,
                "ref_end": unit.ref_end,
            }
        )
    link_masks: dict[tuple[Oriented, Oriented], int] = {}
    for name, steps in paths.items():
        b = bit[path_sample(name)]
        for u, v in _path_pairs(steps, circular):
            key = _canonical((names[u[0]], u[1]), (names[v[0]], v[1]))
            link_masks[key] = link_masks.get(key, 0) | b
    links = [
        {"source": f"{s[0]}{s[1]}", "target": f"{t[0]}{t[1]}", "genomes": bin(mask).count("1")}
        for (s, t), mask in sorted(
            link_masks.items(), key=lambda kv: (_block_key(kv[0][0]), _block_key(kv[0][1]))
        )
    ]
    return blocks, links


def _block_key(node: Oriented) -> tuple[int, str]:
    return (int(node[0][1:]), node[1])


def _presence_class(genomes: int, total: int) -> str:
    if genomes >= total:
        return "core"
    if genomes <= 1:
        return "private"
    return "shell"


# ---------------------------------------------------------------------- drawing


def skeleton_layout(skeleton: Mapping[str, Any], *, seed: int = 0) -> dict[str, np.ndarray]:
    """Polyline per block from the assembly-graph stress layout."""
    from .graph_layout import graph_layout

    segments = [{"name": b["name"], "length": int(b["length"])} for b in skeleton["blocks"]]
    links = [{"source": e["source"], "target": e["target"]} for e in skeleton["links"]]
    raw = graph_layout(segments, links, seed=seed)
    return {name: np.asarray(points, dtype=float) for name, points in raw.items()}


def draw_pangenome_graph(
    ax: Any,
    skeleton: Mapping[str, Any],
    *,
    radius: float,
    unit_in: float,
    layout: Mapping[str, np.ndarray] | None = None,
    label_blocks: int = 6,
    font_pt: float = 6.5,
    named_labels: bool = False,
    bounds: tuple[float, float, float, float] | None = None,
) -> int:
    """Draw the skeleton centred at the origin of ``ax``, inside ``radius``.

    ``unit_in`` (inches per data unit) sizes ribbons and fonts so the drawing
    keeps its proportions at any figure size; ``layout`` is a precomputed
    :func:`skeleton_layout` (computed here when omitted). ``bounds``
    (x0, y0, x1, y1) fits the drawing to a box instead of the circle. Returns
    the number of blocks that got a label.
    """
    from matplotlib.patches import FancyArrowPatch, PathPatch
    from matplotlib.path import Path as MplPath

    from ..structure.graph_conformation import parse_node_label
    from .assembly_graph import _end_geometry, _smooth

    blocks = {b["name"]: b for b in skeleton["blocks"]}
    if not blocks:
        return 0
    layout = dict(layout) if layout is not None else skeleton_layout(skeleton)
    points = np.concatenate(list(layout.values()))
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    reach = float(np.linalg.norm(points - center, axis=1).max()) or 1.0
    scale = 0.9 * radius / reach
    if bounds is not None:
        span = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-9)
        scale = 0.92 * min((bounds[2] - bounds[0]) / span[0], (bounds[3] - bounds[1]) / span[1])
    layout = {name: (pts - center) * scale for name, pts in layout.items()}

    pt_per_unit = unit_in * 72
    ribbon = max(1.2, 0.0105 * pt_per_unit)
    total = max(1, int(skeleton.get("genomes", 1)))
    link_pts: list[np.ndarray] = []  # sampled link curves, kept clear of labels
    t = np.linspace(0.0, 1.0, 24)[:, None]
    for link in skeleton["links"]:
        source, target = parse_node_label(link["source"]), parse_node_label(link["target"])
        p0, u0 = _end_geometry(layout[source[0]], source, exit_end=True)
        p3, u3 = _end_geometry(layout[target[0]], target, exit_end=False)
        gap = float(np.linalg.norm(p3 - p0))
        bend = 0.04 * radius / 0.42
        if gap < 1e-6:
            normal = np.array([-u0[1], u0[0]])
            c1, c2 = p0 + 2.2 * bend * (u0 + normal), p0 + 2.2 * bend * (u0 - normal)
        else:
            far = 0.45 * gap + bend
            c1, c2 = p0 + far * u0, p3 + far * u3
        path = MplPath(
            [p0, c1, c2, p3], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        )
        link_pts.append(
            (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t**2 * c2 + t**3 * p3
        )
        width = ribbon * (0.2 + 0.5 * math.sqrt(link["genomes"] / total))
        ax.add_patch(
            PathPatch(
                path, fill=False, lw=width, ec=LINK_COLOR, capstyle="round", zorder=6, alpha=0.9
            )
        )
    # Private blocks first, core last, so the shared backbone stays on top.
    rank = {"private": 0, "shell": 1, "core": 2}
    for name, row in sorted(blocks.items(), key=lambda kv: rank[kv[1]["class"]]):
        curve = _smooth(layout[name])
        color = CLASS_COLORS[row["class"]]
        z = 7 + 0.1 * rank[row["class"]]
        ax.plot(
            curve[:, 0],
            curve[:, 1],
            color="white",
            lw=ribbon + 1.0,
            solid_capstyle="butt",
            zorder=z,
        )
        ax.plot(
            curve[:, 0], curve[:, 1], color=color, lw=ribbon, solid_capstyle="butt", zorder=z + 0.05
        )
        tail = curve[max(0, len(curve) - 3)]
        ax.add_patch(
            FancyArrowPatch(
                tail,
                curve[-1],
                arrowstyle="-|>,head_length=0.5,head_width=0.32",
                mutation_scale=ribbon * 1.6,
                color=color,
                lw=0,
                shrinkA=0,
                shrinkB=0,
                zorder=z + 0.05,
            )
        )
    largest = sorted(blocks.values(), key=lambda b: -b["length"])[:label_blocks]
    font = font_pt  # printed points
    # Labels go where their box clears every earlier label and every ribbon
    # point; a label that fits nowhere is left out (the block table lists all).
    char_w, line_h = 0.55 * font / pt_per_unit, 1.25 * font / pt_per_unit
    ribbon_pts = np.concatenate([_smooth(layout[name]) for name in blocks] + link_pts)
    ribbon_pad = 0.6 * ribbon / pt_per_unit
    placed: list[tuple[float, float, float, float]] = []
    labelled = 0
    for row in largest:
        text = f"{row['name']} {_kb(row['length'])}" if named_labels else _kb(row["length"])
        half_w, half_h = char_w * len(text) / 2, line_h / 2
        curve = _smooth(layout[row["name"]])
        spot = None
        for fraction in (0.5, 0.35, 0.65, 0.2, 0.8):
            i = min(len(curve) - 2, max(1, int(fraction * (len(curve) - 1))))
            tangent = curve[i + 1] - curve[i - 1]
            tangent = tangent / (np.linalg.norm(tangent) or 1.0)
            normal = np.array([-tangent[1], tangent[0]])
            for side in (1, -1):
                for gap in (1.0, 1.8):
                    c = curve[i] + side * normal * (half_h + ribbon_pad) * gap
                    box = (c[0] - half_w, c[1] - half_h, c[0] + half_w, c[1] + half_h)
                    if bounds is not None and not (
                        bounds[0] <= box[0]
                        and box[2] <= bounds[2]
                        and bounds[1] <= box[1]
                        and box[3] <= bounds[3]
                    ):
                        continue
                    if bounds is None and np.linalg.norm(c) + half_w > radius:
                        continue
                    if any(_boxes_overlap(box, other) for other in placed):
                        continue
                    inside = (
                        (ribbon_pts[:, 0] > box[0] - ribbon_pad)
                        & (ribbon_pts[:, 0] < box[2] + ribbon_pad)
                        & (ribbon_pts[:, 1] > box[1] - ribbon_pad)
                        & (ribbon_pts[:, 1] < box[3] + ribbon_pad)
                    )
                    if inside.any():
                        continue
                    spot, placed_box = c, box
                    break
                if spot is not None:
                    break
            if spot is not None:
                break
        if spot is None:
            continue
        placed.append(placed_box)
        labelled += 1
        ax.text(
            spot[0],
            spot[1],
            text,
            ha="center",
            va="center",
            fontsize=font,
            color="#1F2933",
            zorder=9,
        )
    return labelled


def _boxes_overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _kb(length: int) -> str:
    return f"{length / 1000:.1f} kb" if length < 10_000 else f"{length / 1000:.0f} kb"


def skeleton_rows(skeleton: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Block table for the sidecar TSV."""
    return [dict(row) for row in skeleton.get("blocks", ())]


def path_window_coverage(
    gfa: str | Path,
    reference: str,
    starts: Sequence[int],
    ends: Sequence[int],
) -> dict[str, list[float]]:
    """Share of each reference window's bp that each genome's path also traverses.

    The reference path is laid out in its own coordinates; a genome covers a
    stretch of it when its path passes through the same graph nodes. A window
    a genome does not reach at all was not aligned to it (absent or unaligned
    sequence), as opposed to one it reaches but where a variant site has no
    allele for it.
    """
    lengths, paths = read_gfa_paths(gfa)
    ref_name = _pick_reference(paths, reference)
    node_sets: dict[str, set[str]] = {}
    for name, steps in paths.items():
        if name == ref_name:
            continue
        node_sets.setdefault(path_sample(name), set()).update(node for node, _ in steps)
    edges = np.asarray([*starts, ends[-1]], dtype=float)
    width = np.maximum(1.0, np.asarray(ends, dtype=float) - np.asarray(starts, dtype=float))
    step_start, step_end, step_node = [], [], []
    offset = 0
    for node, _ in paths[ref_name]:
        step_start.append(offset)
        offset += lengths.get(node, 0)
        step_end.append(offset)
        step_node.append(node)
    a = np.asarray(step_start, dtype=float)
    b = np.asarray(step_end, dtype=float)
    coverage: dict[str, list[float]] = {}
    for sample, nodes in sorted(node_sets.items()):
        shared = np.fromiter((n in nodes for n in step_node), dtype=bool, count=len(step_node))
        covered = np.zeros(len(width))
        for lo, hi in zip(a[shared], b[shared], strict=True):
            first = max(0, int(np.searchsorted(edges, lo, side="right")) - 1)
            for k in range(first, len(width)):
                overlap = min(hi, edges[k + 1]) - max(lo, edges[k])
                if overlap <= 0 and edges[k] >= hi:
                    break
                covered[k] += max(0.0, overlap)
        coverage[sample] = [round(float(v), 4) for v in np.minimum(1.0, covered / width)]
    return coverage


def render_full_graph(
    path: Path,
    skeleton: Mapping[str, Any],
    layout: Mapping[str, Sequence[Sequence[float]]],
    *,
    header: str = "",
    dpi: int = 300,
) -> Path:
    """The complete skeleton as its own 183 mm figure, every block named and sized."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.patches import Patch

    from .ogdraw import _inches_per_unit
    from .style import FONT_PT, MM_PER_INCH, publication_style, width_inches

    width = width_inches("double")
    height = 150.0 / MM_PER_INCH
    with publication_style():
        fig = plt.figure(figsize=(width, height))
        try:
            ax = fig.add_axes((0.0, 0.06, 1.0, 0.9))
            ax.set_xlim(-1.45, 1.45)
            ax.set_ylim(-1.02, 1.02)
            ax.set_aspect("equal")
            ax.axis("off")
            labelled = draw_pangenome_graph(
                ax,
                skeleton,
                radius=1.0,
                unit_in=_inches_per_unit(ax),
                layout={k: np.asarray(v) for k, v in layout.items()},
                label_blocks=len(skeleton["blocks"]),
                font_pt=FONT_PT["label"] + 0.5,
                named_labels=True,
                bounds=(-1.42, -1.0, 1.42, 1.0),
            )
            missing = len(skeleton["blocks"]) - labelled
            if missing:
                header += (
                    f"; {missing} short blocks unlabelled for space (all in pan_graph_blocks.tsv)"
                )
            classes = {b["class"] for b in skeleton["blocks"]}
            handles = [
                Patch(facecolor=CLASS_COLORS[c], label=CLASS_LABELS[c])
                for c in CLASS_COLORS
                if c in classes
            ]
            fig.legend(
                handles=handles,
                loc="lower left",
                ncol=3,
                frameon=False,
                fontsize=FONT_PT["body"],
                bbox_to_anchor=(0.01, 0.0),
            )
            if header:
                fig.text(0.01, 0.985, header, ha="left", va="top", fontsize=FONT_PT["body"])
            fig.savefig(str(path), dpi=dpi, facecolor="white")
        finally:
            plt.close(fig)
    return path


__all__ = [
    "CLASS_COLORS",
    "CLASS_LABELS",
    "draw_pangenome_graph",
    "pangenome_skeleton",
    "path_window_coverage",
    "read_gfa_paths",
    "render_full_graph",
    "skeleton_layout",
]
