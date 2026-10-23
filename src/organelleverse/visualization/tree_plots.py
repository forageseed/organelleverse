"""Annotated phylogenetic trees and tree-conflict (tanglegram) figures.

``plot_phylo_tree`` draws one tree, rectangular or circular, with support
symbols, coloured tip groups, clade brackets and tip-data strips (rings in the
circular layout). ``plot_tree_conflict`` sets trees of the same taxa from
different genomes (for example nuclear, mitochondrial and plastid) face to
face and classes every clade against the other tree:

* shared - the bipartition is in both trees;
* conflicting - it is incompatible with at least one bipartition of the other
  tree (the two topologies cannot both be true);
* supported conflict - conflicting, with support at or above the threshold in
  this tree and on an incompatible bipartition of the other tree;
* unresolved - compatible with the other tree but absent from it (a polytomy
  or a collapsed branch there), which is not a conflict.

Bipartitions are compared unrooted after pruning every tree to the shared
taxa, so rooting only changes the drawing. Everything drawn is computed at call
time into ``metrics``; drawing reads ``metrics`` only, so a serialized Result
redraws through ``ov.write`` and the figures compose with ``compose_figure``.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from organelleverse._public_validation import validated_public

from ..core.errors import OrganelleInputError
from ..inputs import Tree
from ..sequences import as_path
from .plot_object import OrganellePlot, PanelDrawer, plot_result
from .style import FONT_PT, JOURNAL_WIDTHS_MM, MAX_HEIGHT_MM, publication_rc, text_width_pt

_PT_PER_MM = 72.0 / 25.4
_INK = "#1F2328"
_GREY = "#8C939D"
_LEADER = "#C3C8CF"
#: Okabe-Ito without yellow, for tip groups.
_GROUP_COLORS = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#8C6D31", "#5D6D7E",
)
#: Clade brackets and shading.
_CLADE_COLORS = ("#4E79A7", "#59A14F", "#B07AA1", "#F28E2B", "#76B7B2", "#9C755F")
#: Categorical tip-data cells.
_CATEGORY_COLORS = (
    "#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#9C755F", "#76B7B2", "#EDC948", "#BAB0AC",
)
_NUM_LOW, _NUM_HIGH = (0.91, 0.94, 0.97), (0.03, 0.19, 0.42)
_PRESENT, _ABSENT_EDGE = "#3B4252", "#C9CED6"
_SHARED, _CONFLICT, _STRONG, _UNRESOLVED = _INK, "#E69F00", "#D55E00", "#A7ADB5"
_LINK = "#B9BEC6"

_ARC_BAND = 12.0  # radial room per clade level in the circular layout

_BINARY_TRUE = {"1", "true", "yes", "present", "+", "y"}
_BINARY_FALSE = {"0", "false", "no", "absent", "-", "n"}
_SUPPORT_DEFAULTS = {"bootstrap": (70.0, 95.0), "posterior": (0.90, 0.95)}

TREE_PLOT_OPERATIONS = frozenset(
    {"visualization.plot_phylo_tree", "visualization.plot_tree_conflict"}
)


# ── reading and normalising trees ─────────────────────────────────────────────


def _tip_text(name: str | None) -> str:
    return (name or "").strip().strip("'\"").replace("_", " ").strip()


def _key(name: str) -> str:
    return " ".join(_tip_text(name).casefold().split())


def _load_tree(source: str | Path | Tree) -> tuple[Any, str]:
    """A Bio.Phylo tree and a short origin label (file name or ``newick``)."""
    from Bio import Phylo  # type: ignore

    from ..phylogeny.treecompare import read_tree

    if isinstance(source, str) and source.lstrip().startswith("("):
        newick, origin = source.strip(), "newick"
    else:
        path = Path(as_path(source))
        if not path.is_file():
            raise OrganelleInputError(
                code="visualization.tree.missing", message=f"tree file not found: {path}"
            )
        newick, origin = read_tree(path), path.name
    try:
        tree = Phylo.read(io.StringIO(newick), "newick")
    except Exception as exc:  # Bio raises bare NewickError / ValueError
        raise OrganelleInputError(
            code="visualization.tree.unreadable", message=f"cannot parse tree {origin}: {exc}"
        ) from exc
    return tree, origin


def _clade_support(clade: Any) -> float | None:
    value = getattr(clade, "confidence", None)
    if value is None and clade.name:
        # "95/0.99"-style internal labels: the first number is the support
        head = str(clade.name).replace("\\", "/").split("/")[0]
        try:
            value = float(head)
        except ValueError:
            value = None
    return None if value is None else float(value)


def _display_names(tree: Any, rename: Mapping[str, str] | None) -> dict[Any, str]:
    lookup = {_key(k): v for k, v in (rename or {}).items()}
    names: dict[Any, str] = {}
    seen: dict[str, Any] = {}
    for tip in tree.get_terminals():
        text = _tip_text(tip.name)
        text = lookup.get(_key(text), text)
        if not text:
            raise OrganelleInputError(
                code="visualization.tree.unnamed_tip", message="every tip must have a name"
            )
        if _key(text) in seen:
            raise OrganelleInputError(
                code="visualization.tree.duplicate_tip",
                message=f"tip name {text!r} occurs twice in one tree",
            )
        seen[_key(text)] = tip
        names[tip] = text
    return names


def _root_tree(
    tree: Any, names: dict[Any, str], outgroup: Sequence[str] | None, midpoint: bool
) -> dict[str, Any]:
    """Root in place; return what was done (for metrics and flags)."""
    info: dict[str, Any] = {"method": "as_given"}
    if outgroup:
        wanted = {_key(n) for n in outgroup}
        clades = [tip for tip, text in names.items() if _key(text) in wanted]
        info.update(method="outgroup", outgroup=sorted(names[c] for c in clades))
        if not clades:
            info["method"] = "as_given"
            info["outgroup_missing"] = True
            return info
        try:
            tree.root_with_outgroup(*clades)
        except Exception:  # outgroup spans the root of an unrooted tree, etc.
            info["method"] = "as_given"
            info["outgroup_failed"] = True
            return info
        mrca = tree.common_ancestor(*clades) if len(clades) > 1 else clades[0]
        info["outgroup_monophyletic"] = {
            names[t] for t in mrca.get_terminals()
        } == {names[c] for c in clades}
    elif midpoint:
        if any((c.branch_length or 0) > 0 for c in tree.find_clades()):
            tree.root_at_midpoint()
            info["method"] = "midpoint"
        else:
            info["midpoint_without_lengths"] = True
    return info


def _serialize(tree: Any, names: dict[Any, str]) -> list[dict[str, Any]]:
    """Preorder node list: id, parent, length, children, and name or support."""
    nodes: list[dict[str, Any]] = []
    stack = [(tree.root, -1)]
    while stack:
        clade, parent = stack.pop()
        node: dict[str, Any] = {
            "id": len(nodes),
            "parent": parent,
            "length": 0.0 if parent < 0 else float(clade.branch_length or 0.0),
            "children": [],
        }
        nodes.append(node)
        if parent >= 0:
            nodes[parent]["children"].append(node["id"])
        if clade.clades:
            node["support"] = _clade_support(clade)
            for child in reversed(clade.clades):
                stack.append((child, node["id"]))
        else:
            node["name"] = names[clade]
    # children were appended in pop order (left to right); keep them that way
    return nodes


def _postorder(nodes: list[dict[str, Any]]) -> list[int]:
    order, stack = [], [0]
    while stack:
        nid = stack.pop()
        order.append(nid)
        stack.extend(nodes[nid]["children"])
    return order[::-1]


def _tips_under(nodes: list[dict[str, Any]]) -> dict[int, list[str]]:
    under: dict[int, list[str]] = {}
    for nid in _postorder(nodes):
        node = nodes[nid]
        if node["children"]:
            under[nid] = [t for c in node["children"] for t in under[c]]
        else:
            under[nid] = [node["name"]]
    return under


def _ladderize(nodes: list[dict[str, Any]]) -> None:
    counts: dict[int, int] = {}
    for nid in _postorder(nodes):
        kids = nodes[nid]["children"]
        counts[nid] = sum(counts[c] for c in kids) if kids else 1
        kids.sort(key=lambda c: counts[c])


def _tip_order(nodes: list[dict[str, Any]]) -> list[str]:
    order, stack = [], [0]
    while stack:
        node = nodes[stack.pop()]
        if node["children"]:
            stack.extend(reversed(node["children"]))
        else:
            order.append(node["name"])
    return order


def _has_lengths(nodes: list[dict[str, Any]]) -> bool:
    return any(n["length"] > 0 for n in nodes[1:])


def _coordinates(nodes: list[dict[str, Any]]) -> tuple[dict[int, float], dict[int, float], float]:
    """x (root-to-node distance, or cladogram depth) and y (tip rank) per node."""
    x: dict[int, float] = {0: 0.0}
    lengths = _has_lengths(nodes)
    if lengths:
        for node in nodes[1:]:  # preorder: parents come first
            x[node["id"]] = x[node["parent"]] + max(node["length"], 0.0)
    else:
        height: dict[int, int] = {}
        for nid in _postorder(nodes):
            kids = nodes[nid]["children"]
            height[nid] = 1 + max(height[c] for c in kids) if kids else 0
        top = height[0]
        for node in nodes:
            x[node["id"]] = float(top - height[node["id"]])
    y: dict[int, float] = {}
    rank = {name: i for i, name in enumerate(_tip_order(nodes))}
    for nid in _postorder(nodes):
        node = nodes[nid]
        if node["children"]:
            y[nid] = (y[node["children"][0]] + y[node["children"][-1]]) / 2.0
        else:
            y[nid] = float(rank[node["name"]])
    span = max(x.values()) or 1.0
    return x, y, span


def _support_scale(nodes: list[dict[str, Any]]) -> str | None:
    values = [n["support"] for n in nodes if n.get("support") is not None]
    if not values:
        return None
    return "posterior" if max(values) <= 1.0 else "bootstrap"


def _thresholds(scale: str | None, override: Sequence[float] | None) -> tuple[float, float] | None:
    if scale is None:
        return None
    if override is not None:
        low, high = (float(v) for v in override)
        return (min(low, high), max(low, high))
    return _SUPPORT_DEFAULTS[scale]


# ── annotation inputs ─────────────────────────────────────────────────────────


def _read_table(path: Path) -> list[list[str]]:
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.reader(handle, delimiter=delimiter) if any(c.strip() for c in row)]


def _groups_input(groups: Mapping[str, str] | str | Path | None) -> dict[str, str]:
    if groups is None:
        return {}
    if isinstance(groups, Mapping):
        return {str(k): str(v) for k, v in groups.items()}
    rows = _read_table(Path(as_path(groups)))
    if rows and rows[0][1:2] and rows[0][1].strip().casefold() in {"group", "clade", "category"}:
        rows = rows[1:]
    return {r[0].strip(): r[1].strip() for r in rows if len(r) >= 2 and r[1].strip()}


def _data_input(data: Mapping[str, Mapping[str, Any]] | str | Path | None) -> dict[str, dict[str, Any]]:
    if data is None:
        return {}
    if isinstance(data, Mapping):
        return {str(col): {str(k): v for k, v in values.items()} for col, values in data.items()}
    rows = _read_table(Path(as_path(data)))
    if len(rows) < 2:
        return {}
    header = rows[0]
    table: dict[str, dict[str, Any]] = {col.strip(): {} for col in header[1:]}
    for row in rows[1:]:
        for col, cell in zip(header[1:], row[1:], strict=False):
            if cell.strip() not in {"", "NA", "nan"}:
                table[col.strip()][row[0].strip()] = cell.strip()
    return table


def _strip(name: str, values: Mapping[str, Any]) -> dict[str, Any]:
    texts = [str(v).strip() for v in values.values()]
    lowered = [t.casefold() for t in texts]
    if texts and all(t in _BINARY_TRUE | _BINARY_FALSE for t in lowered):
        return {
            "name": name, "kind": "binary",
            "values": {k: str(v).strip().casefold() in _BINARY_TRUE for k, v in values.items()},
        }
    try:
        numbers = {k: float(v) for k, v in values.items()}
    except (TypeError, ValueError):
        numbers = None
    if numbers is not None and numbers:
        finite = [v for v in numbers.values() if math.isfinite(v)]
        return {
            "name": name, "kind": "numeric", "values": numbers,
            "min": min(finite) if finite else 0.0, "max": max(finite) if finite else 0.0,
        }
    categories = sorted({str(v).strip() for v in values.values()})
    return {
        "name": name, "kind": "categorical",
        "values": {k: str(v).strip() for k, v in values.items()}, "categories": categories,
    }


def _match(names: Sequence[str], wanted: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Map annotation keys onto displayed tip names; return the matches and the misses."""
    by_key = {_key(n): n for n in names}
    matched, missing = {}, []
    for raw, value in wanted.items():
        tip = by_key.get(_key(raw))
        if tip is None:
            missing.append(str(raw))
        else:
            matched[tip] = value
    return matched, sorted(missing)


# ── plot_phylo_tree ───────────────────────────────────────────────────────────


@validated_public('visualization.plot_phylo_tree')
def plot_phylo_tree(
    tree: str | Path | Tree,
    *,
    layout: str = "rectangular",
    outgroup: Sequence[str] | None = None,
    midpoint_root: bool = False,
    ladderize: bool = True,
    support: str = "symbols",
    support_thresholds: Sequence[float] | None = None,
    tip_groups: Mapping[str, str] | str | Path | None = None,
    clades: Mapping[str, Sequence[str]] | None = None,
    tip_data: Mapping[str, Mapping[str, Any]] | str | Path | None = None,
    rename: Mapping[str, str] | None = None,
    tip_labels: bool = True,
    italic: bool = True,
    width: str | float | None = None,
    dpi: int = 600,
) -> OrganellePlot:
    """Prepare an annotated phylogenetic tree; render with ``ov.write``.

    Parameters
    ----------
    tree : Newick or NEXUS file (or a Newick string)
    layout : ``"rectangular"`` or ``"circular"``
    outgroup : tip names to root on; ``midpoint_root`` roots at the midpoint
        instead. Without either the tree is drawn as given (flagged when its root
        is a polytomy, i.e. an unrooted tree).
    support : ``"symbols"`` (filled dots for strong, grey for moderate support),
        ``"values"`` or ``"none"``. Bootstrap (0-100) or posterior (0-1) is
        detected; default thresholds are 70/95 and 0.90/0.95, or
        ``support_thresholds=(moderate, strong)``.
    tip_groups : ``{tip: group}`` or a two-column TSV/CSV; colours the tip names
        and every branch whose descendants all share one group.
    clades : ``{label: [tips...]}``; a bracket and light shading over the most
        recent common ancestor. Two tips name the clade by their MRCA; three or
        more are its full membership, and a set that is not monophyletic gets
        ``*`` (its bracket covers the whole MRCA).
    tip_data : ``{column: {tip: value}}`` or a TSV/CSV with a header (first
        column = tip); drawn as aligned cells (rings when circular). Columns are
        typed as presence/absence, numeric (light to dark) or categorical.
    rename : ``{tip in file: name to show}`` (e.g. accession to species).
    width : ``"single"``/``"onehalf"``/``"double"`` or millimetres; by default
        the narrowest journal width that fits the labels and annotations.
    """
    if layout not in {"rectangular", "circular"}:
        raise OrganelleInputError(
            code="visualization.tree.layout", message="layout must be 'rectangular' or 'circular'"
        )
    if support not in {"symbols", "values", "none"}:
        raise OrganelleInputError(
            code="visualization.tree.support", message="support must be 'symbols', 'values' or 'none'"
        )
    bio, origin = _load_tree(tree)
    names = _display_names(bio, rename)
    rooting = _root_tree(bio, names, outgroup, midpoint_root)
    names = {tip: names[tip] for tip in bio.get_terminals()}
    nodes = _serialize(bio, names)
    if ladderize:
        _ladderize(nodes)
    tips = _tip_order(nodes)
    under = _tips_under(nodes)
    scale = _support_scale(nodes)

    groups, groups_missing = _match(tips, _groups_input(tip_groups))
    group_order = sorted(set(groups.values()), key=lambda g: min(tips.index(t) for t in groups if groups[t] == g))

    clade_rows, clades_missing = [], []
    for label, members in (clades or {}).items():
        found, missing = _match(tips, {m: True for m in members})
        clades_missing.extend(missing)
        if not found:
            continue
        mrca = _mrca(nodes, under, set(found))
        clade_rows.append({
            "label": str(label),
            "requested": sorted(found),
            "definition": "mrca" if len(found) <= 2 else "members",
            "node": mrca,
            "members": under[mrca],
            "monophyletic": len(found) <= 2 or set(under[mrca]) == set(found),
        })

    _clade_levels(clade_rows, tips)

    strips, data_missing = [], []
    for column, values in _data_input(tip_data).items():
        found, missing = _match(tips, values)
        data_missing.extend(missing)
        if found:
            strips.append(_strip(column, found))

    metrics: dict[str, Any] = {
        "source": origin,
        "layout": layout,
        "n_tips": len(tips),
        "nodes": nodes,
        "has_branch_lengths": _has_lengths(nodes),
        "rooting": rooting,
        "root_is_polytomy": len(nodes[0]["children"]) > 2,
        "support_mode": support if scale else "none",
        "support_scale": scale,
        "support_thresholds": _thresholds(scale, support_thresholds),
        "groups": groups,
        "group_order": group_order,
        "clades": clade_rows,
        "strips": strips,
        "unmatched": {
            "tip_groups": groups_missing,
            "clades": sorted(set(clades_missing)),
            "tip_data": sorted(set(data_missing)),
        },
        "tip_labels": bool(tip_labels),
        "italic": bool(italic),
        "dpi": int(dpi),
    }
    _size_phylo(metrics, width)
    flags = []
    if metrics["root_is_polytomy"] and rooting["method"] == "as_given":
        flags.append("unrooted_tree_drawn_as_given")
    if rooting.get("outgroup_missing") or rooting.get("outgroup_failed"):
        flags.append("outgroup_not_applied")
    if rooting.get("outgroup_monophyletic") is False:
        flags.append("outgroup_not_monophyletic")
    if any(not c["monophyletic"] for c in clade_rows):
        flags.append("clade_not_monophyletic")
    if any(metrics["unmatched"].values()):
        flags.append("annotation_names_unmatched")
    if metrics["tip_labels"] and not metrics["labels_fit"]:
        flags.append("tip_labels_hidden")
    if metrics["height_mm"] > MAX_HEIGHT_MM:
        flags.append("taller_than_page")
    return _with_panel(plot_result(
        "plot_phylo_tree",
        lambda path: _render(Path(path), metrics),
        organelle="mixed",
        method=f"ov_tree_{layout}",
        metrics=metrics,
        flags=tuple(flags),
        summary=f"{layout.capitalize()} tree of {len(tips)} tips prepared.",
    ), metrics)


def _clade_levels(rows: list[dict[str, Any]], tips: Sequence[str]) -> None:
    """Give nested or overlapping clades separate bracket columns (level 0 innermost)."""
    rank = {t: i for i, t in enumerate(tips)}
    taken: list[list[tuple[int, int]]] = []
    for row in sorted(rows, key=lambda r: (len(r["members"]), r["label"])):
        span = (min(rank[t] for t in row["members"]), max(rank[t] for t in row["members"]))
        row["rows"] = list(span)
        level = 0
        while level < len(taken) and any(not (span[1] < a or b < span[0]) for a, b in taken[level]):
            level += 1
        if level == len(taken):
            taken.append([])
        taken[level].append(span)
        row["level"] = level


def _clade_text(clade: Mapping[str, Any]) -> str:
    return clade["label"] + ("*" if not clade["monophyletic"] else "")


def _clade_columns(m: Mapping[str, Any], pad: float) -> list[float]:
    """Offset of each bracket level from the first bracket, plus the total width last."""
    levels = max((c["level"] for c in m["clades"]), default=-1) + 1
    offsets, at = [], 0.0
    for level in range(levels):
        offsets.append(at)
        at += pad + _plain_width([_clade_text(c) for c in m["clades"] if c["level"] == level])
    return offsets + [at]


def _mrca(nodes: list[dict[str, Any]], under: dict[int, list[str]], members: set[str]) -> int:
    best = 0
    for node in nodes:  # preorder: the last node whose tips cover the set is the deepest
        if members <= set(under[node["id"]]):
            best = node["id"]
    return best


# ── sizing ────────────────────────────────────────────────────────────────────


def _label_width(names: Sequence[str], italic: bool) -> float:
    size = FONT_PT["label"]
    return max((text_width_pt(n, size, italic=italic) for n in names), default=0.0)


def _plain_width(texts: Sequence[str], size: float | None = None) -> float:
    size = FONT_PT["label"] if size is None else size
    return max((text_width_pt(t, size) for t in texts), default=0.0)


def _legend_entries(m: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """``(kind, colour, label)`` triples in display order."""
    entries: list[tuple[str, str, str]] = []
    if m.get("support_mode") == "symbols" and m.get("support_thresholds"):
        low, high = m["support_thresholds"]
        name = "PP" if m["support_scale"] == "posterior" else "BS"
        entries.append(("dot", _INK, f"{name} ≥ {high:g}"))
        entries.append(("dot", _GREY, f"{name} {low:g}–{high:g}"))
    for i, group in enumerate(m.get("group_order", ())):
        entries.append(("patch", _GROUP_COLORS[i % len(_GROUP_COLORS)], group))
    rings = m.get("layout") == "circular"
    for j, strip in enumerate(m.get("strips", ())):
        prefix = f"{j + 1} {strip['name']}" if rings else strip["name"]
        if strip["kind"] == "binary":
            entries.append(("patch", _PRESENT, f"{prefix}: present"))
        elif strip["kind"] == "numeric":
            entries.append(("patch", _rgb_hex(_NUM_LOW), f"{prefix}: {_number(strip['min'])}"))
            entries.append(("patch", _rgb_hex(_NUM_HIGH), _number(strip["max"])))
        else:
            for k, cat in enumerate(strip["categories"]):
                label = f"{prefix}: {cat}" if k == 0 else cat
                entries.append(("patch", _CATEGORY_COLORS[k % len(_CATEGORY_COLORS)], label))
    if any(not c["monophyletic"] for c in m.get("clades", ())):
        entries.append(("text", _INK, "* not monophyletic"))
    for kind, colour, label in m.get("conflict_legend", ()):
        entries.append((kind, colour, label))
    return entries


def _number(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.3g}"


def _legend_band(entries: Sequence[tuple[str, str, str]], width_pt: float) -> tuple[float, int]:
    if not entries:
        return 0.0, 1
    item = max(text_width_pt(e[2], FONT_PT["tick"]) for e in entries) + 14.0
    ncol = max(1, min(len(entries), int(width_pt // item)))
    rows = math.ceil(len(entries) / ncol)
    return rows * 8.5 + 5.0, ncol


def _rect_parts(m: Mapping[str, Any]) -> dict[str, float]:
    tips = [n["name"] for n in m["nodes"] if not n["children"]]
    label_w = _label_width(tips, m["italic"]) + 3.0 if m["tip_labels"] else 0.0
    strip_w = len(m["strips"]) * 7.0 + (3.0 if m["strips"] else 0.0)
    clade_w = _clade_columns(m, 8.0)[-1] if m["clades"] else 0.0
    head = (_plain_width([s["name"] for s in m["strips"]]) + 4.0) if m["strips"] else 4.0
    bottom = 16.0 if m["has_branch_lengths"] else 4.0
    return {"label": label_w, "strip": strip_w, "clade": clade_w, "head": head, "bottom": bottom}


def _circular_parts(m: Mapping[str, Any]) -> dict[str, float]:
    parts = _rect_parts(m)
    parts["ring"] = len(m["strips"]) * 6.0 + (3.0 if m["strips"] else 0.0)
    levels = max((c["level"] for c in m["clades"]), default=-1) + 1
    parts["clade"] = levels * _ARC_BAND
    return parts


def _circular_fit(m: Mapping[str, Any], width_pt: float) -> tuple[bool, float]:
    """Whether tip labels fit around a circle this wide, and the tree radius."""
    parts = _circular_parts(m)
    radius = width_pt / 2.0 - 3.0
    n = max(m["n_tips"], 1)
    step = math.radians(_circular_step(m))
    with_labels = radius - parts["clade"] - parts["label"] - parts["ring"] - 4.0
    label_r = with_labels + parts["ring"] + 3.0
    fits = with_labels >= 0.22 * radius and label_r * step >= FONT_PT["label"] * 1.12
    if fits and m["tip_labels"]:
        return True, with_labels
    return False, radius - parts["clade"] - parts["ring"] - 4.0 if n else radius


def _circular_gap(m: Mapping[str, Any]) -> float:
    """Degrees left open at the top for the ring numbers (none without rings)."""
    return 14.0 if m["strips"] else 0.0


def _circular_step(m: Mapping[str, Any]) -> float:
    return (360.0 - _circular_gap(m)) / max(m["n_tips"], 1)


def _size_phylo(m: dict[str, Any], width: str | float | None) -> None:
    candidates = [JOURNAL_WIDTHS_MM[k] for k in ("single", "onehalf", "double")]
    if width is not None:
        candidates = [JOURNAL_WIDTHS_MM[width] if isinstance(width, str) else float(width)]
    chosen = candidates[-1]
    if m["layout"] == "circular":
        for mm in candidates:
            if _circular_fit(m, mm * _PT_PER_MM)[0] or not m["tip_labels"]:
                chosen = mm
                break
        m["labels_fit"] = bool(m["tip_labels"]) and _circular_fit(m, chosen * _PT_PER_MM)[0]
        body = chosen * _PT_PER_MM
    else:
        parts = _rect_parts(m)
        need = parts["label"] + parts["strip"] + parts["clade"] + 40.0 * _PT_PER_MM
        for mm in candidates:
            if mm * _PT_PER_MM >= need:
                chosen = mm
                break
        m["labels_fit"] = bool(m["tip_labels"])
        pitch = 8.0
        room = MAX_HEIGHT_MM * _PT_PER_MM - parts["head"] - parts["bottom"] - 30.0
        if m["n_tips"] * pitch > room:
            pitch = max(FONT_PT["label"] * 1.12, room / max(m["n_tips"], 1))
        m["row_pitch_pt"] = round(pitch, 3)
        body = parts["head"] + m["n_tips"] * pitch + parts["bottom"]
    band, _ = _legend_band(_legend_entries(m), chosen * _PT_PER_MM)
    m["width_mm"] = round(chosen, 2)
    m["height_mm"] = round((body + band) / _PT_PER_MM, 2)


# ── drawing helpers ───────────────────────────────────────────────────────────


def _rgb_hex(rgb: Sequence[float]) -> str:
    return "#" + "".join(f"{round(c * 255):02X}" for c in rgb)


def _numeric_colour(value: float, low: float, high: float) -> str:
    t = 0.5 if high <= low else min(max((value - low) / (high - low), 0.0), 1.0)
    return _rgb_hex([a + (b - a) * t for a, b in zip(_NUM_LOW, _NUM_HIGH, strict=True)])


def _cell_colour(strip: Mapping[str, Any], tip: str) -> tuple[str, str] | None:
    """(face, edge) of a tip-data cell, or ``None`` when the tip has no value."""
    if tip not in strip["values"]:
        return None
    value = strip["values"][tip]
    if strip["kind"] == "binary":
        return (_PRESENT, _PRESENT) if value else ("white", _ABSENT_EDGE)
    if strip["kind"] == "numeric":
        colour = _numeric_colour(float(value), strip["min"], strip["max"])
        return colour, colour
    k = strip["categories"].index(value)
    colour = _CATEGORY_COLORS[k % len(_CATEGORY_COLORS)]
    return colour, colour


def _branch_colours(m: Mapping[str, Any], under: dict[int, list[str]]) -> dict[int, str]:
    groups = m.get("groups") or {}
    order = list(m.get("group_order") or ())
    colours: dict[int, str] = {}
    for node in m["nodes"]:
        tips_ = under[node["id"]]
        shared = {groups.get(t) for t in tips_}
        if groups and len(shared) == 1 and None not in shared:
            colours[node["id"]] = _GROUP_COLORS[order.index(shared.pop()) % len(_GROUP_COLORS)]
        else:
            colours[node["id"]] = _INK
    return colours


def _support_class(m: Mapping[str, Any], value: float | None) -> str | None:
    if value is None or not m.get("support_thresholds"):
        return None
    low, high = m["support_thresholds"]
    if value >= high:
        return _INK
    if value >= low:
        return _GREY
    return None


def _support_text(value: float) -> str:
    return f"{value:.0f}" if value > 1 else f"{value:.2f}"


def _scale_length(span: float) -> float:
    limit = span * 0.25
    base = 10 ** math.floor(math.log10(limit))
    return max(k * base for k in (1, 2, 5) if k * base <= limit)


def _ax_pt(ax: Any) -> tuple[float, float]:
    fig = ax.figure
    box = ax.get_position()
    return box.width * fig.get_figwidth() * 72.0, box.height * fig.get_figheight() * 72.0


def _legend_handles(entries: Sequence[tuple[str, str, str]]) -> list[Any]:
    from matplotlib.lines import Line2D  # type: ignore
    from matplotlib.patches import Patch  # type: ignore

    handles = []
    for kind, colour, label in entries:
        if kind == "dot":
            handles.append(Line2D([], [], ls="", marker="o", ms=3.2, mfc=colour, mec=colour, label=label))
        elif kind == "line":
            handles.append(Line2D([], [], color=colour, lw=0.9, label=label))
        elif kind == "thick":
            handles.append(Line2D([], [], color=colour, lw=1.6, label=label))
        elif kind == "dashed":
            handles.append(Line2D([], [], color=colour, lw=0.9, ls=(0, (2, 1.2)), label=label))
        elif kind == "text":
            handles.append(Line2D([], [], ls="", marker="", label=label))
        else:
            handles.append(Patch(facecolor=colour, edgecolor="none", label=label))
    return handles


def _draw_legend(ax: Any, entries, width_pt: float) -> None:
    if not entries:
        return
    _, ncol = _legend_band(entries, width_pt)
    ax.legend(
        handles=_legend_handles(entries), loc="lower left", bbox_to_anchor=(0.0, 0.0),
        ncol=ncol, frameon=False, fontsize=FONT_PT["tick"], handlelength=1.1,
        handleheight=0.8, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.2,
        labelspacing=0.35,
    )


# ── rectangular tree ──────────────────────────────────────────────────────────


def _draw_rectangular(ax: Any, m: Mapping[str, Any], box: tuple[float, float, float, float]) -> None:
    """Draw into the points-box ``(left, top, width, height)`` of a y-down axes."""
    from matplotlib.collections import LineCollection  # type: ignore
    from matplotlib.patches import Rectangle  # type: ignore

    nodes = m["nodes"]
    x, y, span = _coordinates(nodes)
    under = _tips_under(nodes)
    colours = _branch_colours(m, under)
    parts = _rect_parts(m)
    left, top, width, height = box
    aligned = bool(m["strips"] or m["clades"])
    tree_w = max(width - parts["label"] - parts["strip"] - parts["clade"] - 6.0, 0.22 * width)
    n = max(m["n_tips"], 1)
    pitch = (height - parts["head"] - parts["bottom"]) / n
    x0 = left + 4.0

    def px(nid: int) -> float:
        return x0 + x[nid] / span * tree_w

    def py(nid: int) -> float:
        return top + parts["head"] + (y[nid] + 0.5) * pitch

    label_x = x0 + tree_w + 3.0
    strip_x = label_x + parts["label"]
    clade_x = strip_x + parts["strip"] + 1.0

    offsets = _clade_columns(m, 8.0)
    for k, clade in enumerate(m["clades"]):
        colour = _CLADE_COLORS[k % len(_CLADE_COLORS)]
        y1 = top + parts["head"] + clade["rows"][0] * pitch
        y2 = top + parts["head"] + (clade["rows"][1] + 1) * pitch
        start = px(clade["node"]) - 3.0
        if clade["level"] == 0:
            ax.add_patch(Rectangle((start, y1), label_x + parts["label"] - 2.0 - start, y2 - y1,
                                   facecolor=colour, alpha=0.08, edgecolor="none", zorder=0))
        bx = clade_x + 2.0 + offsets[clade["level"]]
        ax.plot([bx] * 2, [y1 + 0.18 * pitch, y2 - 0.18 * pitch], color=colour,
                lw=1.3, solid_capstyle="butt")
        ax.text(bx + 3.0, (y1 + y2) / 2, _clade_text(clade),
                ha="left", va="center", fontsize=FONT_PT["label"], color=_INK)

    segments, seg_colours = [], []
    for node in nodes:
        nid = node["id"]
        if node["parent"] >= 0:
            segments.append([(px(node["parent"]), py(nid)), (px(nid), py(nid))])
            seg_colours.append(colours[nid])
        if node["children"]:
            first, last = node["children"][0], node["children"][-1]
            segments.append([(px(nid), py(first)), (px(nid), py(last))])
            seg_colours.append(colours[nid])
    segments.append([(x0 - 3.0, py(0)), (px(0), py(0))])
    seg_colours.append(colours[0])
    ax.add_collection(LineCollection(segments, colors=seg_colours, linewidths=0.65,
                                     capstyle="projecting", zorder=2))

    groups, order = m.get("groups") or {}, list(m.get("group_order") or ())
    for node in nodes:
        if node["children"]:
            continue
        nid, name = node["id"], node["name"]
        tip_x = px(nid)
        if m["tip_labels"]:
            at = label_x if aligned else tip_x + 2.5
            if aligned and at - tip_x > 4.0:
                ax.plot([tip_x + 1.5, at - 1.5], [py(nid)] * 2, color=_LEADER, lw=0.4,
                        ls=(0, (1, 1.5)), zorder=1)
            colour = _GROUP_COLORS[order.index(groups[name]) % len(_GROUP_COLORS)] if name in groups else _INK
            ax.text(at, py(nid), name, ha="left", va="center", fontsize=FONT_PT["label"],
                    fontstyle="italic" if m["italic"] else "normal", color=colour)
        for j, strip in enumerate(m["strips"]):
            cell = _cell_colour(strip, name)
            if cell is None:
                continue
            ax.add_patch(Rectangle((strip_x + j * 7.0 + 0.5, py(nid) - 0.42 * pitch), 6.0, 0.84 * pitch,
                                   facecolor=cell[0], edgecolor=cell[1], lw=0.4, zorder=2))
    for j, strip in enumerate(m["strips"]):
        ax.text(strip_x + j * 7.0 + 3.5, top + parts["head"] - 2.0, strip["name"], rotation=90,
                ha="center", va="bottom", fontsize=FONT_PT["label"], color=_INK)

    _draw_support(ax, m, nodes, lambda nid: (px(nid), py(nid)), rectangular=True)

    if m["has_branch_lengths"]:
        length = _scale_length(span)
        bar_y = top + height - 9.0
        ax.plot([x0, x0 + length / span * tree_w], [bar_y, bar_y], color=_INK, lw=0.65,
                solid_capstyle="butt")
        ax.text(x0 + length / span * tree_w / 2, bar_y + 1.5, f"{length:g}", ha="center", va="top",
                fontsize=FONT_PT["tick"], color=_INK)


def _draw_support(ax, m, nodes, at, *, rectangular: bool) -> None:
    mode = m.get("support_mode")
    if mode in (None, "none"):
        return
    for node in nodes:
        if not node["children"] or node["parent"] < 0 or node.get("support") is None:
            continue
        xx, yy = at(node["id"])
        if mode == "symbols":
            colour = _support_class(m, node["support"])
            if colour:
                ax.plot([xx], [yy], ls="", marker="o", ms=3.0, mfc=colour, mec="white",
                        mew=0.3, zorder=4)
        else:
            ax.text(xx - 1.2, yy - 1.0 if rectangular else yy, _support_text(node["support"]),
                    ha="right", va="bottom", fontsize=FONT_PT["label"], color=_GREY, zorder=4)


# ── circular tree ─────────────────────────────────────────────────────────────


def _draw_circular(ax: Any, m: Mapping[str, Any], box: tuple[float, float, float, float]) -> None:
    """Draw into the points-box ``(left, bottom, width, height)`` of a y-up axes."""
    from matplotlib.collections import LineCollection  # type: ignore
    from matplotlib.patches import Wedge  # type: ignore

    nodes = m["nodes"]
    x, y, span = _coordinates(nodes)
    under = _tips_under(nodes)
    colours = _branch_colours(m, under)
    parts = _circular_parts(m)
    left, bottom, width, height = box
    size = min(width, height)
    cx, cy = left + width / 2.0, bottom + height / 2.0
    radius = size / 2.0 - 3.0
    show_labels = m["tip_labels"] and m.get("labels_fit", True) and _circular_fit(m, size)[0]
    label_w = parts["label"] if show_labels else 0.0
    tree_r = max(radius - parts["clade"] - label_w - parts["ring"] - 4.0, 0.2 * radius)
    step = _circular_step(m)
    # tip i sits in the middle of its own sector, clockwise from the top gap
    start = 90.0 - _circular_gap(m) / 2.0 - step / 2.0
    aligned = bool(m["strips"] or m["clades"])

    def theta(nid: int) -> float:
        return start - y[nid] * step

    def r(nid: int) -> float:
        return x[nid] / span * tree_r

    def pos(radius_: float, angle: float) -> tuple[float, float]:
        rad = math.radians(angle)
        return cx + radius_ * math.cos(rad), cy + radius_ * math.sin(rad)

    def arc(radius_: float, a1: float, a2: float) -> list[tuple[float, float]]:
        steps = max(2, int(abs(a2 - a1) / 1.5) + 1)
        return [pos(radius_, a1 + (a2 - a1) * i / (steps - 1)) for i in range(steps)]

    ring_r0 = tree_r + 3.0
    label_r = ring_r0 + parts["ring"]
    clade_r = label_r + label_w + 2.0

    for k, clade in enumerate(m["clades"]):
        colour = _CLADE_COLORS[k % len(_CLADE_COLORS)]
        a_hi = start - clade["rows"][0] * step + step / 2.0
        a_lo = start - clade["rows"][1] * step - step / 2.0
        if clade["level"] == 0:
            inner = max(r(clade["node"]) - 3.0, 0.0)
            ax.add_patch(Wedge((cx, cy), ring_r0 - 1.0, a_lo, a_hi, width=ring_r0 - 1.0 - inner,
                               facecolor=colour, alpha=0.08, edgecolor="none", zorder=0))
        radius_ = clade_r + 1.5 + clade["level"] * _ARC_BAND
        ax.add_collection(LineCollection([arc(radius_, a_lo + step * 0.3, a_hi - step * 0.3)],
                                         colors=[colour], linewidths=1.3, capstyle="butt"))
        mid = (a_hi + a_lo) / 2.0
        upper = math.sin(math.radians(mid)) >= 0
        ax.text(*pos(radius_ + 5.0, mid), _clade_text(clade), rotation=mid - 90.0 if upper else mid + 90.0,
                rotation_mode="anchor", ha="center", va="center", fontsize=FONT_PT["label"], color=_INK)

    segments, seg_colours = [], []
    for node in nodes:
        nid = node["id"]
        if node["parent"] >= 0:
            segments.append([pos(r(node["parent"]), theta(nid)), pos(r(nid), theta(nid))])
            seg_colours.append(colours[nid])
        if node["children"]:
            first, last = node["children"][0], node["children"][-1]
            segments.append(arc(r(nid), theta(first), theta(last)))
            seg_colours.append(colours[nid])
    ax.add_collection(LineCollection(segments, colors=seg_colours, linewidths=0.6,
                                     capstyle="projecting", zorder=2))

    groups, order = m.get("groups") or {}, list(m.get("group_order") or ())
    for node in nodes:
        if node["children"]:
            continue
        nid, name, angle = node["id"], node["name"], theta(node["id"])
        if aligned and ring_r0 - r(nid) > 4.0:
            ax.plot(*zip(pos(r(nid) + 1.5, angle), pos(ring_r0 - 1.5, angle), strict=True),
                    color=_LEADER, lw=0.4, ls=(0, (1, 1.5)), zorder=1)
        for j, strip in enumerate(m["strips"]):
            cell = _cell_colour(strip, name)
            if cell is None:
                continue
            outer = ring_r0 + j * 6.0 + 5.0
            ax.add_patch(Wedge((cx, cy), outer, angle - step * 0.46, angle + step * 0.46, width=5.0,
                               facecolor=cell[0], edgecolor=cell[1], lw=0.3, zorder=2))
        if show_labels:
            at = label_r if aligned else r(nid) + 2.5
            colour = _GROUP_COLORS[order.index(groups[name]) % len(_GROUP_COLORS)] if name in groups else _INK
            _radial_text(ax, pos(at, angle), angle, name, italic=m["italic"], colour=colour)
    for j, _strip in enumerate(m["strips"]):
        ax.text(*pos(ring_r0 + j * 6.0 + 2.5, 90.0), str(j + 1), ha="center", va="center",
                fontsize=FONT_PT["label"], color=_INK)

    _draw_support(ax, m, nodes, lambda nid: pos(r(nid), theta(nid)), rectangular=False)

    if m["has_branch_lengths"]:
        length = _scale_length(span)
        bar_y = bottom + 8.0
        ax.plot([left + 4.0, left + 4.0 + length / span * tree_r], [bar_y, bar_y], color=_INK,
                lw=0.65, solid_capstyle="butt")
        ax.text(left + 4.0 + length / span * tree_r / 2, bar_y - 1.5, f"{length:g}", ha="center",
                va="top", fontsize=FONT_PT["tick"], color=_INK)


def _radial_text(ax, xy, angle: float, text: str, *, italic: bool, colour: str = _INK) -> None:
    angle = angle % 360.0
    flip = 90.0 < angle < 270.0
    ax.text(*xy, text, rotation=angle - 180.0 if flip else angle, rotation_mode="anchor",
            ha="right" if flip else "left", va="center", fontsize=FONT_PT["label"],
            fontstyle="italic" if italic else "normal", color=colour)


# ── plot_tree_conflict ────────────────────────────────────────────────────────


def _splits(nodes: list[dict[str, Any]], under: dict[int, list[str]], taxa: frozenset[str]):
    """``{node id: canonical bipartition}`` and ``{bipartition: best support}``."""
    anchor = min(taxa)
    by_node: dict[int, frozenset[str]] = {}
    support: dict[frozenset[str], float | None] = {}
    for node in nodes:
        if not node["children"] or node["parent"] < 0:
            continue
        side = frozenset(under[node["id"]])
        if len(side) < 2 or len(taxa) - len(side) < 2:
            continue
        split = taxa - side if anchor in side else side
        by_node[node["id"]] = split
        value = node.get("support")
        old = support.get(split)
        support[split] = value if old is None else (old if value is None else max(old, value))
    return by_node, support


def _compatible(a: frozenset[str], b: frozenset[str], taxa: frozenset[str]) -> bool:
    return not (a & b) or a <= b or b <= a or (a | b) == taxa


def _classify(
    mine: tuple[dict[int, frozenset[str]], dict[frozenset[str], float | None]],
    other: tuple[dict[int, frozenset[str]], dict[frozenset[str], float | None]],
    taxa: frozenset[str],
    my_threshold: float | None,
    other_threshold: float | None,
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    by_node, support = mine
    other_support = other[1]
    classes: dict[int, str] = {}
    conflicts: dict[frozenset[str], dict[str, Any]] = {}
    for nid, split in by_node.items():
        if split in other_support:
            classes[nid] = "shared"
            continue
        against = [t for t in other_support if not _compatible(split, t, taxa)]
        if not against:
            classes[nid] = "unresolved"
            continue
        opposing = [other_support[t] for t in against if other_support[t] is not None]
        mine_value = support[split]
        strong = (
            my_threshold is not None and other_threshold is not None and mine_value is not None
            and mine_value >= my_threshold and bool(opposing) and max(opposing) >= other_threshold
        )
        classes[nid] = "strong" if strong else "conflict"
        side = split if len(split) <= len(taxa) / 2 else taxa - split
        conflicts[split] = {
            "taxa": sorted(side),
            "support": mine_value,
            "opposing_support": max(opposing) if opposing else None,
            "strongly_supported": strong,
        }
    return classes, sorted(conflicts.values(), key=lambda c: (not c["strongly_supported"], len(c["taxa"]), c["taxa"]))


def _rotate_to(nodes: list[dict[str, Any]], rank: Mapping[str, int]) -> None:
    mean: dict[int, float] = {}
    for nid in _postorder(nodes):
        node = nodes[nid]
        if node["children"]:
            node["children"].sort(key=lambda c: mean[c])
            weights = [(mean[c], _count(nodes, c)) for c in node["children"]]
            mean[nid] = sum(v * w for v, w in weights) / sum(w for _, w in weights)
        else:
            mean[nid] = float(rank[node["name"]])


def _count(nodes: list[dict[str, Any]], nid: int) -> int:
    stack, total = [nid], 0
    while stack:
        node = nodes[stack.pop()]
        if node["children"]:
            stack.extend(node["children"])
        else:
            total += 1
    return total


def _crossings(left: Sequence[str], right: Sequence[str]) -> int:
    where = {name: i for i, name in enumerate(right)}
    seq = [where[name] for name in left]
    return sum(1 for i in range(len(seq)) for j in range(i + 1, len(seq)) if seq[i] > seq[j])


def _untangle(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    """Rotate child order (topology unchanged) to reduce connector crossings."""
    import copy

    best = (_crossings(_tip_order(left), _tip_order(right)), copy.deepcopy(left), copy.deepcopy(right))
    for _ in range(3):
        _rotate_to(right, {n: i for i, n in enumerate(_tip_order(left))})
        _rotate_to(left, {n: i for i, n in enumerate(_tip_order(right))})
        _rotate_to(right, {n: i for i, n in enumerate(_tip_order(left))})
        score = _crossings(_tip_order(left), _tip_order(right))
        if score < best[0]:
            best = (score, copy.deepcopy(left), copy.deepcopy(right))
    left[:], right[:] = best[1], best[2]
    return best[0]


@validated_public('visualization.plot_tree_conflict')
def plot_tree_conflict(
    trees: Sequence[str | Path | Tree],
    *,
    names: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
    outgroup: Sequence[str] | None = None,
    midpoint_root: bool = False,
    support_threshold: float | None = None,
    posterior_threshold: float = 0.95,
    italic: bool = True,
    width: str | float = "double",
    dpi: int = 600,
) -> OrganellePlot:
    """Prepare face-to-face trees of the same taxa with their conflicts marked.

    The first tree is the reference (e.g. the nuclear tree); each other tree
    (mitochondrial, plastid...) is drawn facing it, with connectors between the
    same taxon. Child order is rotated to untangle the connectors; topologies
    are not changed. Clades are coloured by their relation to the facing tree
    (shared, conflicting, supported conflict, unresolved; see the module
    docstring). Trees are pruned to the taxa they share; dropped taxa are
    listed in ``metrics["dropped"]``.

    Parameters
    ----------
    trees : two or three Newick/NEXUS files (a ``{name: file}`` mapping is also
        accepted from Python and names the panels)
    names : panel names (default: mapping keys, else "Tree 1", "Tree 2"...)
    rename : ``{tip in file: name to show}``, applied to every tree before
        matching (use it when the trees label taxa differently)
    outgroup, midpoint_root : rooting for the drawing (comparison is unrooted)
    support_threshold : bootstrap value a conflict needs on both sides to count
        as supported (default 70); ``posterior_threshold`` is used for trees with
        0-1 support
    """
    if isinstance(trees, Mapping):
        names = list(names or trees.keys())
        sources = list(trees.values())
    else:
        sources = list(trees)
        names = list(names or [f"Tree {i + 1}" for i in range(len(sources))])
    if not 2 <= len(sources) <= 3 or len(names) != len(sources):
        raise OrganelleInputError(
            code="visualization.tree_conflict.count",
            message="give two or three trees (reference first) and one name per tree",
        )
    loaded = []
    for source in sources:
        bio, origin = _load_tree(source)
        loaded.append((bio, origin, _display_names(bio, rename)))
    shared = set.intersection(*({_key(t) for t in d.values()} for _, _, d in loaded))
    if len(shared) < 4:
        raise OrganelleInputError(
            code="visualization.tree_conflict.too_few_taxa",
            message=f"the trees share {len(shared)} taxa; at least 4 are needed (use rename= to match names)",
        )
    dropped: dict[str, list[str]] = {}
    serialized, rootings, scales, thresholds = [], [], [], []
    for (bio, origin, display), name in zip(loaded, names, strict=True):
        extra = [tip for tip, text in display.items() if _key(text) not in shared]
        dropped[name] = sorted(display[t] for t in extra)
        for tip in extra:
            bio.prune(tip)
        display = {tip: display[tip] for tip in bio.get_terminals()}
        rootings.append(_root_tree(bio, display, outgroup, midpoint_root))
        nodes = _serialize(bio, {tip: display[tip] for tip in bio.get_terminals()})
        _ladderize(nodes)
        serialized.append(nodes)
        scale = _support_scale(nodes)
        scales.append(scale)
        thresholds.append(
            None if scale is None
            else posterior_threshold if scale == "posterior"
            else (70.0 if support_threshold is None else float(support_threshold))
        )
    # one display name per shared taxon: the reference tree's
    canonical = {_key(n["name"]): n["name"] for n in serialized[0] if not n["children"]}
    for nodes in serialized[1:]:
        for node in nodes:
            if not node["children"]:
                node["name"] = canonical[_key(node["name"])]
    taxa = frozenset(canonical.values())

    pairs = []
    for k in range(1, len(serialized)):
        left = [dict(n, children=list(n["children"])) for n in serialized[0]]
        right = [dict(n, children=list(n["children"])) for n in serialized[k]]
        crossings = _untangle(left, right)
        under_l, under_r = _tips_under(left), _tips_under(right)
        split_l, split_r = _splits(left, under_l, taxa), _splits(right, under_r, taxa)
        cls_l, conf_l = _classify(split_l, split_r, taxa, thresholds[0], thresholds[k])
        cls_r, conf_r = _classify(split_r, split_l, taxa, thresholds[k], thresholds[0])
        a, b = set(split_l[1]), set(split_r[1])
        rf = len(a ^ b)
        involved = sorted({t for c in conf_l + conf_r if c["strongly_supported"] for t in c["taxa"]})
        pairs.append({
            "left": 0, "right": k,
            "left_nodes": left, "right_nodes": right,
            "left_classes": {str(i): c for i, c in cls_l.items()},
            "right_classes": {str(i): c for i, c in cls_r.items()},
            "crossings": crossings,
            "robinson_foulds": rf,
            "robinson_foulds_max": len(a) + len(b),
            "normalized_rf": round(rf / (len(a) + len(b)), 4) if a or b else 0.0,
            "shared_clades": len(a & b),
            "conflicts": {names[0]: conf_l, names[k]: conf_r},
            "n_conflicting": {names[0]: len(conf_l), names[k]: len(conf_r)},
            "n_supported_conflicts": {
                names[0]: sum(c["strongly_supported"] for c in conf_l),
                names[k]: sum(c["strongly_supported"] for c in conf_r),
            },
            "taxa_in_supported_conflict": involved,
        })

    metrics: dict[str, Any] = {
        "names": names,
        "sources": [origin for _, origin, _ in loaded],
        "n_taxa": len(taxa),
        "taxa": sorted(taxa),
        "dropped": dropped,
        "rooting": rootings,
        "support_scale": scales,
        "support_threshold": thresholds,
        "has_branch_lengths": [_has_lengths(nodes) for nodes in serialized],
        "pairs": pairs,
        "italic": bool(italic),
        "dpi": int(dpi),
        "comparison": "unrooted bipartitions after pruning to shared taxa",
        "conflict_legend": [
            ("line", _SHARED, "Shared clade"),
            ("line", _CONFLICT, "Conflicting clade"),
            ("thick", _STRONG, "Supported conflict"),
            ("dashed", _UNRESOLVED, "Not resolved in the other tree"),
        ],
    }
    _size_conflict(metrics, width)
    flags = []
    if any(dropped.values()):
        flags.append("taxa_pruned_to_shared")
    if any(p["n_supported_conflicts"][names[p["right"]]] or p["n_supported_conflicts"][names[0]]
           for p in pairs):
        flags.append("supported_conflict")
    if None in scales:
        flags.append("support_missing")
    if metrics["height_mm"] > MAX_HEIGHT_MM:
        flags.append("taller_than_page")
    summary = "; ".join(
        f"{names[0]} vs {names[p['right']]}: RF {p['robinson_foulds']}/{p['robinson_foulds_max']}, "
        f"{p['n_supported_conflicts'][names[0]]} supported conflicting clade(s)"
        for p in pairs
    )
    return _with_panel(plot_result(
        "plot_tree_conflict",
        lambda path: _render(Path(path), metrics),
        organelle="mixed",
        method="bipartition_conflict",
        metrics=metrics,
        flags=tuple(flags),
        summary=f"Tree conflict over {len(taxa)} shared taxa prepared ({summary}).",
    ), metrics)


def _pair_parts(m: Mapping[str, Any], pair: Mapping[str, Any]) -> dict[str, float]:
    tips = m["taxa"]
    label = _label_width(tips, m["italic"]) + 3.0
    return {"label": label, "link": 16.0}


def _size_conflict(m: dict[str, Any], width: str | float) -> None:
    total = (JOURNAL_WIDTHS_MM[width] if isinstance(width, str) else float(width)) * _PT_PER_MM
    pairs = m["pairs"]
    label = _pair_parts(m, pairs[0])["label"]
    side_by_side = len(pairs) == 1 or ((total - 12.0) / 2 - 2 * label - 16.0) / 2 >= 12.0 * _PT_PER_MM
    m["arrangement"] = "side_by_side" if side_by_side else "stacked"
    pitch = 8.0
    head, foot = 14.0, 12.0
    rows = 1 if side_by_side else len(pairs)
    room = MAX_HEIGHT_MM * _PT_PER_MM - rows * (head + foot) - 30.0
    if rows * m["n_taxa"] * pitch > room:
        pitch = max(FONT_PT["label"] * 1.12, room / (rows * m["n_taxa"]))
    m["row_pitch_pt"] = round(pitch, 3)
    body = rows * (head + m["n_taxa"] * pitch + foot) + (rows - 1) * 6.0
    band, _ = _legend_band(_legend_entries(m), total)
    m["width_mm"] = round(total / _PT_PER_MM, 2)
    m["height_mm"] = round((body + band) / _PT_PER_MM, 2)


def _draw_conflict_pair(ax, m, pair, box) -> None:
    from matplotlib.collections import LineCollection  # type: ignore

    left_, top, width, height = box
    names = m["names"]
    label = _pair_parts(m, pair)["label"]
    link = max(16.0, 0.1 * width)
    tree_w = max((width - 2 * label - link - 4.0) / 2.0, 0.12 * width)
    head, foot = 14.0, 12.0
    n = m["n_taxa"]
    pitch = (height - head - foot) / n
    sides = (
        ("left", pair["left_nodes"], {int(k): v for k, v in pair["left_classes"].items()}),
        ("right", pair["right_nodes"], {int(k): v for k, v in pair["right_classes"].items()}),
    )
    tip_y: dict[str, dict[str, float]] = {}
    for side, nodes, classes in sides:
        x, y, span = _coordinates(nodes)
        mirror = side == "right"

        def px(nid: int, x=x, span=span, mirror=mirror) -> float:
            d = x[nid] / span * tree_w
            return left_ + width - 2.0 - d if mirror else left_ + 2.0 + d

        def py(nid: int, y=y) -> float:
            return top + head + (y[nid] + 0.5) * pitch

        styles = {
            "shared": (_SHARED, 0.65, "solid"), "conflict": (_CONFLICT, 0.9, "solid"),
            "strong": (_STRONG, 1.5, "solid"), "unresolved": (_UNRESOLVED, 0.65, (0, (2, 1.2))),
        }
        groups: dict[str, list] = {key: [] for key in styles}
        for node in nodes:
            nid = node["id"]
            cls = classes.get(nid, "shared")
            if node["parent"] >= 0:
                groups[cls].append([(px(node["parent"]), py(nid)), (px(nid), py(nid))])
            if node["children"]:
                first, last = node["children"][0], node["children"][-1]
                groups["shared"].append([(px(nid), py(first)), (px(nid), py(last))])
        stub = 3.0 if mirror else -3.0
        groups["shared"].append([(px(0) - stub, py(0)), (px(0), py(0))])
        for cls, segs in groups.items():
            if segs:
                colour, lw, ls = styles[cls]
                ax.add_collection(LineCollection(segs, colors=colour, linewidths=lw, linestyles=ls,
                                                 capstyle="butt", zorder=3 if cls == "strong" else 2))
        tip_y[side] = {}
        label_x = left_ + width - 2.0 - tree_w - 3.0 if mirror else left_ + 2.0 + tree_w + 3.0
        for node in nodes:
            if node["children"]:
                continue
            nid, name = node["id"], node["name"]
            tip_y[side][name] = py(nid)
            if abs(label_x - px(nid)) > 4.0:
                start = px(nid) + (-1.5 if mirror else 1.5)
                ax.plot([start, label_x + (1.5 if mirror else -1.5)], [py(nid)] * 2, color=_LEADER,
                        lw=0.4, ls=(0, (1, 1.5)), zorder=1)
            ax.text(label_x, py(nid), name, ha="right" if mirror else "left", va="center",
                    fontsize=FONT_PT["label"], fontstyle="italic" if m["italic"] else "normal",
                    color=_INK)
        # a node marker keeps a conflict visible on a short branch
        for nid, cls in classes.items():
            if cls in {"strong", "conflict"}:
                colour = _STRONG if cls == "strong" else _CONFLICT
                ax.plot([px(nid)], [py(nid)], ls="", marker="s", ms=3.0 if cls == "strong" else 2.4,
                        mfc=colour, mec="white", mew=0.3, zorder=4)
        title_x = left_ + width - 2.0 if mirror else left_ + 2.0
        ax.text(title_x, top + 6.0, names[pair[side]], ha="right" if mirror else "left", va="center",
                fontsize=FONT_PT["body"], fontweight="bold", color=_INK)
    x_from = left_ + 2.0 + tree_w + 3.0 + label - 2.0
    x_to = left_ + width - 2.0 - tree_w - 3.0 - label + 2.0
    links = [[(x_from, yl), (x_to, tip_y["right"][name])] for name, yl in tip_y["left"].items()]
    ax.add_collection(LineCollection(links, colors=_LINK, linewidths=0.45, zorder=1))
    right_name = names[pair["right"]]
    text = (
        f"RF {pair['robinson_foulds']}/{pair['robinson_foulds_max']} "
        f"({pair['normalized_rf']:.2f}). Conflicting clades: "
        + ", ".join(
            f"{tree} {pair['n_conflicting'][tree]} ({pair['n_supported_conflicts'][tree]} supported)"
            for tree in (names[0], right_name)
        )
    )
    ax.text(left_ + 2.0, top + height - 5.0, text, ha="left", va="center",
            fontsize=FONT_PT["tick"], color=_GREY)


def _draw_conflict(ax, m, box) -> None:
    left, top, width, height = box
    pairs = m["pairs"]
    if m["arrangement"] == "side_by_side":
        gap = 12.0 if len(pairs) > 1 else 0.0
        w = (width - gap * (len(pairs) - 1)) / len(pairs)
        for k, pair in enumerate(pairs):
            _draw_conflict_pair(ax, m, pair, (left + k * (w + gap), top, w, height))
    else:
        gap = 6.0
        h = (height - gap * (len(pairs) - 1)) / len(pairs)
        for k, pair in enumerate(pairs):
            _draw_conflict_pair(ax, m, pair, (left, top + k * (h + gap), width, h))


# ── rendering, panels and restore ─────────────────────────────────────────────


def _draw(ax: Any, m: Mapping[str, Any], legend: bool) -> list[Any]:
    width, height = _ax_pt(ax)
    entries = _legend_entries(m)
    band = _legend_band(entries, width)[0] if legend else 0.0
    ax.set_axis_off()
    ax.set_xlim(0, width)
    if m.get("pairs") is not None:
        ax.set_ylim(height, 0)
        _draw_conflict(ax, m, (0.0, 0.0, width, height - band))
    elif m["layout"] == "circular":
        ax.set_ylim(0, height)
        ax.set_aspect("equal", adjustable="box", anchor="N")
        _draw_circular(ax, m, (0.0, band, width, height - band))
    else:
        ax.set_ylim(height, 0)
        _draw_rectangular(ax, m, (0.0, 0.0, width, height - band))
    if legend:
        _draw_legend(ax, entries, width)
        return []
    return _legend_handles(entries)


def _render(output: Path, m: Mapping[str, Any]) -> Path:
    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(publication_rc()):
        fig = plt.figure(figsize=(m["width_mm"] / 25.4, m["height_mm"] / 25.4))
        try:
            ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
            _draw(ax, m, True)
            fig.savefig(str(output), dpi=int(m.get("dpi", 600)), facecolor="white")
        finally:
            plt.close(fig)
    return output


def _with_panel(plot: OrganellePlot, metrics: Mapping[str, Any]) -> OrganellePlot:
    header = "Tree conflict" if metrics.get("pairs") is not None else "Phylogeny"
    plot._panel = PanelDrawer(
        lambda ax, legend: _draw(ax, metrics, legend),
        float(metrics["height_mm"]) / float(metrics["width_mm"]),
        header,
    )
    return plot


def restore_tree_plot(result) -> OrganellePlot:
    """Reattach rendering to a serialized tree-plot Result; nothing is recomputed."""
    from ..core.frozen import thaw_json

    plot = OrganellePlot.model_validate(result.model_dump(mode="python"))
    metrics = thaw_json(plot.metrics)
    plot._renderer = lambda path: _render(Path(path), metrics)
    return _with_panel(plot, metrics)


def render_circular_tree(newick_path: str | Path, output: Path, *, italic: bool = True,
                         show_support: bool = True) -> Path:
    """Circular drawing behind ``plot_tree(layout="circular")``."""
    from ..core.frozen import thaw_json

    plot = plot_phylo_tree(newick_path, layout="circular", italic=italic,
                           support="symbols" if show_support else "none")
    return _render(Path(output), thaw_json(plot.metrics))


__all__ = [
    "TREE_PLOT_OPERATIONS",
    "plot_phylo_tree",
    "plot_tree_conflict",
    "render_circular_tree",
    "restore_tree_plot",
]
