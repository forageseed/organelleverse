"""Haplotype-network construction + visualisation (PopART-equivalent).

Three classical algorithms, all re-implemented from the primary literature in
the OrganelleVerse style (no GPL code, no external dependency beyond numpy /
matplotlib / networkx, all optional):

  - **TCS** — statistical-parsimony network of Clement, Posada & Crandall
    (2000, *Mol Ecol* 9:1657). Component-based connection at the Templeton
    et al. (1992) 95% parsimony limit, with intermediate-sequence inference
    and post-processing degree-2 vertex collapse (matches PopART's TCS).

  - **MJN** — median-joining network of Bandelt, Forster & Röhl (1999,
    *Mol Biol Evol* 16:37). Iteratively adds median vectors (unsampled
    ancestors) to minimise total network length.

  - **MSN** — minimum-spanning network (the foundation both TCS and MJN build
    on). Greedy Kruskal join at increasing distance levels.

The haplotype-collapse + p-distance helpers and the matplotlib renderer are
shared across all three. The renderer draws population-aware pie-chart nodes
(the signature PopART look) when population metadata is supplied.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .._bio import read_fasta
from ..core.result import OrganelleResult
from ..visualization.plot_object import OrganellePlot, plot_result
from ._results import failed_result, findings, ok_result, provenance

# ===========================================================================
# Step 1 — collapse identical sequences to unique haplotypes
# ===========================================================================


def collapse_haplotypes(
    seqs: list[tuple[str, str]],
    population_map: dict[str, str] | None = None,
) -> tuple[list[dict], dict[str, list[str]], dict[str, int], dict[str, dict[str, int]]]:
    """Collapse identical sequences to unique haplotypes.

    Returns ``(haplotypes, members, freq, freq_by_pop)`` where ``haplotypes``
    is a list of ``{"id", "sequence"}`` dicts (ids ``H1, H2, ...`` ordered by
    frequency desc), ``members[hap_id]`` lists the original sample ids,
    ``freq[hap_id]`` the count, and ``freq_by_pop[hap_id]`` the per-population
    counts (only populated when ``population_map`` is given).
    """
    groups: dict[str, list[str]] = {}
    for name, seq in seqs:
        groups.setdefault(seq.upper(), []).append(name)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    haplotypes: list[dict] = []
    members: dict[str, list[str]] = {}
    freq: dict[str, int] = {}
    freq_by_pop: dict[str, dict[str, int]] = {}
    for i, (seq_key, names) in enumerate(ordered, 1):
        hap_id = f"H{i}"
        haplotypes.append({"id": hap_id, "sequence": seq_key})
        members[hap_id] = sorted(names)
        freq[hap_id] = len(names)
        if population_map:
            pop_counts: dict[str, int] = {}
            for n in names:
                pop = population_map.get(n, "unknown")
                pop_counts[pop] = pop_counts.get(pop, 0) + 1
            freq_by_pop[hap_id] = pop_counts
        else:
            freq_by_pop[hap_id] = {}
    return haplotypes, members, freq, freq_by_pop


# ===========================================================================
# Step 2 — pairwise p-distance (uncorrected, gap/ambig aware)
# ===========================================================================


def pdistance(a: str, b: str) -> int:
    """Uncorrected p-distance (number of differing ACGT columns).

    Gaps and ambiguous bases (N/?) are ignored, matching PopART/TCS behaviour.
    """
    n = min(len(a), len(b))
    d = 0
    for i in range(n):
        x, y = a[i], b[i]
        if x in "ACGT" and y in "ACGT" and x != y:
            d += 1
    return d


def pairwise_distances(haplotypes: list[Mapping[str, str | int | float | bool | None]]) -> list[tuple[str, str, int]]:
    """All pairwise p-distances between haplotypes."""
    pairs: list[tuple[str, str, int]] = []
    for i in range(len(haplotypes)):
        for j in range(i + 1, len(haplotypes)):
            d = pdistance(haplotypes[i]["sequence"], haplotypes[j]["sequence"])
            pairs.append((haplotypes[i]["id"], haplotypes[j]["id"], d))
    return pairs


# ===========================================================================
# Step 3 — Templeton 1992 95% parsimony connection limit
# ===========================================================================


def tcs_connection_limit(
    n_haplotypes: int,
    sequence_length: int = 0,
    confidence: float = 0.95,
) -> int:
    """Maximum connection distance for the 95% parsimony criterion.

    Per Templeton, Crandall & Sing (1992, *Genetics* 130:163). With no sequence
    length we fall back to a simple floor of the Poisson-derived threshold.
    """
    if sequence_length > 0:
        # Poisson cumulative probability on k mutations; λ ≈ 2·ln(n_haplotypes).
        lam = 2.0 * math.log(n_haplotypes) if n_haplotypes > 1 else 1.0
        cumulative = 0.0
        for k in range(1, sequence_length + 1):
            try:
                cumulative += (lam**k) * math.exp(-lam) / math.factorial(k)
            except (OverflowError, ValueError):
                break
            if cumulative >= (1 - confidence):
                return max(1, k)
        return max(1, min(sequence_length, 10))
    # No length given: use the closed-form approx (same as the older helper).
    threshold = (1 - confidence) * 3 / 4
    if threshold <= 0:
        return max(1, n_haplotypes - 1)
    d = 1 + math.log(threshold) / math.log(2 / 3)
    return max(1, math.floor(d))


# ===========================================================================
# Algorithm 1 — MSN (minimum-spanning network; Kruskal greedy join)
# ===========================================================================


def build_msn(
    haplotypes: list[Mapping[str, str | int | float | bool | None]],
    connection_limit: int | None = None,
) -> tuple[list[tuple[str, str, int]], dict[str, list[str]]]:
    """Minimum-spanning network: Kruskal greedy join at ascending distances.

    Returns ``(edges, components)``. ``edges`` is a list of
    ``(hap_a, hap_b, distance)``; ``components`` maps a component id to its
    haplotype ids.
    """
    pairs = pairwise_distances(haplotypes)
    if connection_limit is not None:
        pairs = [p for p in pairs if p[2] <= connection_limit]
    pairs.sort(key=lambda p: p[2])

    parent = {h["id"]: h["id"] for h in haplotypes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges: list[tuple[str, str, int]] = []
    for d, a, b in ((p[2], p[0], p[1]) for p in pairs):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            edges.append((a, b, d))

    comps: dict[str, list[str]] = {}
    for h in haplotypes:
        root = find(h["id"])
        comps.setdefault(root, []).append(h["id"])
    components = {f"net{i + 1}": sorted(hids) for i, (_, hids) in enumerate(sorted(comps.items()))}
    return edges, components


# ===========================================================================
# Algorithm 2 — TCS (Clement 2000; component connection + intermediates)
# ===========================================================================


def build_tcs_network(
    haplotypes: list[Mapping[str, str | int | float | bool | None]],
    members: dict[str, list[str]],
    connection_limit: int | None = None,
    infer_intermediates: bool = True,
    collapse_degree2: bool = True,
) -> tuple[list[tuple[str, str, int]], dict[str, list[str]], list[Mapping[str, str | int | float | bool | None]]]:
    """TCS statistical-parsimony network (Clement, Posada & Crandall 2000).

    Component-based connection: haplotypes are joined at increasing distance
    levels up to ``connection_limit``; pairs in different components are
    connected (directly at distance 1, or via inferred intermediate sequences
    at distance > 1). After construction, degree-2 intermediate vertices are
    collapsed (matches PopART's TCS post-processing).

    Returns ``(edges, components, intermediates)`` where ``intermediates`` is a
    list of ``{"id", "between": (a, b)}`` dicts for the inferred (unsampled)
    nodes that remain after degree-2 collapse.
    """
    n = len(haplotypes)
    if n < 2:
        return [], {"net1": [h["id"] for h in haplotypes]}, []

    pairs = pairwise_distances(haplotypes)
    by_distance: dict[int, list[tuple[str, str]]] = {}
    for a, b, d in pairs:
        if d == 0:
            continue
        if connection_limit is None or d <= connection_limit:
            by_distance.setdefault(d, []).append((a, b))

    # adjacency: hap_id -> {neighbour: distance}
    adj: dict[str, dict[str, int]] = {h["id"]: {} for h in haplotypes}
    component_ids: dict[str, int] = {h["id"]: i for i, h in enumerate(haplotypes)}
    intermediates: list[dict] = []
    int_counter = 0

    for M in sorted(by_distance.keys()):
        pending = by_distance[M]
        while pending:
            comp_a = comp_b = -1
            deferred: list[tuple[str, str]] = []
            for u_id, v_id in pending:
                cu, cv = component_ids[u_id], component_ids[v_id]
                if cu == cv:
                    continue
                if cu > cv:
                    cu, cv = cv, cu
                    u_id, v_id = v_id, u_id
                if comp_a < 0:
                    comp_a, comp_b = cu, cv
                if cu == comp_a and cv == comp_b:
                    if M == 1 or not infer_intermediates:
                        adj.setdefault(u_id, {})[v_id] = M
                        adj.setdefault(v_id, {})[u_id] = M
                    else:
                        # insert a chain of (M-1) intermediates between u and v
                        int_counter = _insert_chain(adj, u_id, v_id, M, int_counter, intermediates)
                        # intermediates live in "no man's land" (comp -1)
                        for k in range(int_counter - (M - 1), int_counter):
                            component_ids[f"mv{k}"] = -1
                else:
                    deferred.append((u_id, v_id))
            # merge components comp_b into comp_a
            if comp_a >= 0:
                for hid in list(component_ids):
                    if component_ids[hid] == comp_b:
                        component_ids[hid] = comp_a
                    elif component_ids[hid] > comp_b:
                        component_ids[hid] -= 1
            pending = deferred

    # Convert adjacency to edge list.
    seen: set[tuple[str, str]] = set()
    edges: list[tuple[str, str, int]] = []
    for u, neigh in adj.items():
        for v, d in neigh.items():
            key = tuple(sorted((u, v)))
            if key not in seen:
                seen.add(key)
                edges.append((u, v, d))

    # Collapse degree-2 intermediates (PopART TCS post-processing).
    if collapse_degree2 and infer_intermediates:
        edges = _collapse_degree2(edges)

    # Components (on the post-collapse graph).
    comps = _connected_components(
        edges, [h["id"] for h in haplotypes] + [i["id"] for i in intermediates]
    )
    return edges, comps, intermediates


def _insert_chain(adj, u_id, v_id, distance, counter, intermediates):
    """Insert a chain of (distance-1) intermediate vertices between u and v."""
    current = u_id
    for _ in range(distance - 1):
        mv_id = f"mv{counter}"
        counter += 1
        adj.setdefault(current, {})[mv_id] = 1
        adj.setdefault(mv_id, {})[current] = 1
        intermediates.append({"id": mv_id, "between": (u_id, v_id)})
        current = mv_id
    adj.setdefault(current, {})[v_id] = 1
    adj.setdefault(v_id, {})[current] = 1
    return counter


def _collapse_degree2(edges):
    """Collapse intermediate (mv*) vertices of degree 2 into a single edge."""
    adj: dict[str, dict[str, int]] = {}
    for a, b, d in edges:
        adj.setdefault(a, {})[b] = d
        adj.setdefault(b, {})[a] = d
    changed = True
    while changed:
        changed = False
        for node in list(adj):
            if not node.startswith("mv"):
                continue
            neigh = adj[node]
            if len(neigh) != 2:
                continue
            n1, n2 = list(neigh)
            w = neigh[n1] + neigh[n2]
            del adj[n1][node]
            del adj[n2][node]
            adj[n1][n2] = w
            adj[n2][n1] = w
            del adj[node]
            changed = True
    out: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    for u, neigh in adj.items():
        for v, d in neigh.items():
            key = tuple(sorted((u, v)))
            if key not in seen:
                seen.add(key)
                out.append((u, v, d))
    return out


def _connected_components(edges, all_nodes):
    adj: dict[str, list[str]] = {n: [] for n in all_nodes}
    for a, b, _ in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen: set[str] = set()
    comps: dict[str, list[str]] = {}
    ci = 0
    for n in all_nodes:
        if n in seen:
            continue
        ci += 1
        stack = [n]
        comp: list[str] = []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj.get(x, []))
        comps[f"net{ci}"] = sorted(comp)
    return comps


# ===========================================================================
# Algorithm 3 — MJN (Bandelt, Forster & Röhl 1999, median-joining)
# ===========================================================================


def _three_way_median(a: str, b: str, c: str) -> str:
    """Quasi-median of three equal-length sequences (Bandelt 1999).

    At each column, take the majority base; if all three differ, pick 'N'
    (ambiguous — not a valid median site).
    """
    L = min(len(a), len(b), len(c))
    out = []
    for i in range(L):
        cols = (a[i], b[i], c[i])
        # majority vote
        if cols[0] == cols[1] or cols[0] == cols[2]:
            out.append(cols[0])
        elif cols[1] == cols[2]:
            out.append(cols[1])
        else:
            out.append("N")  # all three differ → not a real median site
    return "".join(out)


def build_mjn(
    haplotypes: list[Mapping[str, str | int | float | bool | None]],
    epsilon: int = 0,
    max_iterations: int = 50,
) -> tuple[list[tuple[str, str, int]], dict[str, list[str]], list[Mapping[str, str | int | float | bool | None]]]:
    """Median-joining network (Bandelt, Forster & Röhl 1999).

    Iteratively adds median vectors (unsampled ancestors) for connected
    triplets when doing so reduces the total network length (within ``epsilon``).
    Returns ``(edges, components, medians)`` where ``medians`` is the list of
    inferred ``{"id", "sequence"}`` dicts.
    """
    n = len(haplotypes)
    if n < 3:
        edges, comps = build_msn(haplotypes)
        return edges, comps, []

    # Current node set: observed haplotypes + inferred medians.
    nodes: dict[str, str] = {h["id"]: h["sequence"] for h in haplotypes}
    medians: list[dict] = []
    mv_counter = 0

    for _ in range(max_iterations):
        edges, _ = build_msn([{"id": k, "sequence": v} for k, v in nodes.items()])
        adj = _edge_list_to_adj(edges)
        # find all connected triplets in the current MSN
        candidate_medians: list[tuple[str, str]] = []  # (mv_id, sequence)
        seen_seqs = set(nodes.values())
        current_cost = sum(d for _, _, d in edges)

        for a in nodes:
            for b in adj.get(a, []):
                if b <= a:
                    continue
                for c in adj.get(a, []):
                    if c <= b or c == b:
                        continue
                    if c not in adj.get(b, []) and b not in adj.get(c, []):
                        continue
                    # a, b, c form a connected triplet
                    med = _three_way_median(nodes[a], nodes[b], nodes[c])
                    if "N" in med:
                        continue  # no valid quasi-median
                    if med in seen_seqs:
                        continue
                    # cost reduction check: add med and recompute MSN length
                    test_nodes = dict(nodes)
                    test_id = f"mv{mv_counter}"
                    test_nodes[test_id] = med
                    test_edges, _ = build_msn(
                        [{"id": k, "sequence": v} for k, v in test_nodes.items()]
                    )
                    new_cost = sum(d for _, _, d in test_edges)
                    if new_cost <= current_cost + epsilon:
                        candidate_medians.append((test_id, med))
                        mv_counter += 1
                        seen_seqs.add(med)
                        nodes[test_id] = med
                        medians.append({"id": test_id, "sequence": med})
                        current_cost = new_cost
                        break  # restart the triplet scan with the new node
                else:
                    continue
                break
            else:
                continue
            break
        else:
            break  # no candidate added this pass → converged

    edges, comps = build_msn([{"id": k, "sequence": v} for k, v in nodes.items()])
    components = _connected_components(edges, list(nodes.keys()))
    return edges, components, medians


def _edge_list_to_adj(edges):
    adj: dict[str, list[str]] = {}
    for a, b, _ in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    return adj


# ===========================================================================
# Visualisation — matplotlib, PopART-style population pie-chart nodes
# ===========================================================================

# A colour-blind-safe palette (Okabe-Ito subset) for population slices.
_POP_COLORS = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#999999",
]


def render_network(
    haplotypes: list[Mapping[str, str | int | float | bool | None]],
    members: dict[str, list[str]],
    freq: dict[str, int],
    edges: list[tuple[str, str, int]],
    *,
    freq_by_pop: dict[str, dict[str, int]] | None = None,
    title: str = "TCS haplotype network",
) -> OrganellePlot:
    """Prepare a haplotype-network figure for explicit materialization."""

    def _render(output: str | Path) -> Path:
        return _render_network(
            haplotypes,
            members,
            freq,
            edges,
            output,
            freq_by_pop=freq_by_pop,
            title=title,
        )

    return plot_result(
        "render_network",
        _render,
        suite="phylogeny",
        organelle="none",
        metrics={"haplotypes": len(haplotypes), "edges": len(edges)},
        summary="Haplotype network prepared.",
    )


def _render_network(
    haplotypes: list[Mapping[str, str | int | float | bool | None]],
    members: dict[str, list[str]],
    freq: dict[str, int],
    edges: list[tuple[str, str, int]],
    output: str | Path,
    *,
    freq_by_pop: dict[str, dict[str, int]] | None = None,
    title: str = "TCS haplotype network",
) -> Path:
    """Render the network as PNG/SVG with population-aware pie-chart nodes.

    Haplotype node size scales with frequency; when ``freq_by_pop`` is given
    each node is a pie chart coloured by population (the signature PopART
    look). Edges are labelled with mutation counts. Median/intermediate
    vectors (id starting with ``mv``) are drawn small and hollow.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return _write_dot(haplotypes, freq, edges, output)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    hap_ids = [h["id"] for h in haplotypes]
    # include median vectors present in edges
    extra = {n for e in edges for n in (e[0], e[1]) if n not in set(hap_ids)}
    all_nodes = hap_ids + sorted(extra)

    # spring layout via a tiny networkx graph (preferred) or circular fallback
    pos = _layout(all_nodes, edges)

    fig, ax = plt.subplots(figsize=(9, 9))
    max_freq = max(freq.values()) if freq else 1

    # edges
    for a, b, d in edges:
        xa, ya = pos[a]
        xb, yb = pos[b]
        ax.plot([xa, xb], [ya, yb], color="#888888", lw=1.2, zorder=1)
        if d > 0:
            ax.text(
                (xa + xb) / 2,
                (ya + yb) / 2,
                str(d),
                fontsize=8,
                color="#b22222",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            )

    # nodes
    pops = sorted({p for fbp in (freq_by_pop or {}).values() for p in fbp})
    pop_color = {p: _POP_COLORS[i % len(_POP_COLORS)] for i, p in enumerate(pops)}
    for nid in all_nodes:
        x, y = pos[nid]
        is_observed = nid in freq
        if is_observed:
            size = 80 + 520 * (freq.get(nid, 1) / max_freq)
            fbp = (freq_by_pop or {}).get(nid, {})
            if fbp and sum(fbp.values()) > 0 and len(fbp) > 1:
                # pie-chart node
                _draw_pie(ax, x, y, size, fbp, pop_color)
            else:
                col = pop_color.get(next(iter(fbp)), "#4DBBD5") if fbp else "#4DBBD5"
                ax.scatter([x], [y], s=size, c=col, edgecolors="black", linewidths=1.2, zorder=3)
            ax.text(x, y, str(freq.get(nid, 1)), fontsize=8, ha="center", va="center", zorder=4)
            ax.text(
                x * 1.16, y * 1.16, nid, fontsize=9, ha="center", va="center", fontweight="bold"
            )
        else:
            # median / intermediate vector: small hollow circle
            ax.scatter(
                [x], [y], s=60, facecolors="white", edgecolors="#666666", linewidths=1.0, zorder=3
            )

    # legend (populations)
    if pops:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=p)
            for p, c in pop_color.items()
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(str(output), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def _draw_pie(ax, x, y, size, fbp, pop_color):
    """Draw a small pie-chart scatter at (x, y) coloured by population."""
    import numpy as np  # type: ignore

    total = sum(fbp.values())
    if total == 0:
        return
    fracs = [v / total for v in fbp.values()]
    colors = [pop_color.get(p, "#999999") for p in fbp]
    radius = 0.04 + 0.06 * (size / 600.0)
    starts = np.cumsum([0.0, *fracs[:-1]]) * 2 * math.pi
    ends = np.cumsum(fracs) * 2 * math.pi
    for s, e, c in zip(starts, ends, colors, strict=False):
        theta = np.linspace(s, e, 30)
        xs = np.r_[x, x + radius * np.cos(theta)]
        ys = np.r_[y, y + radius * np.sin(theta)]
        ax.fill(xs, ys, color=c, ec="black", lw=0.8, zorder=3)


def _layout(nodes, edges):
    """Spring layout via networkx (preferred) or circular fallback."""
    try:
        import networkx as nx  # type: ignore

        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from([(a, b) for a, b, _ in edges])
        return nx.spring_layout(g, seed=42) if nodes else {}
    except ImportError:
        pos = {}
        for i, n in enumerate(nodes):
            angle = 2 * math.pi * i / max(1, len(nodes))
            pos[n] = (math.cos(angle), math.sin(angle))
        return pos


def _write_dot(haplotypes, freq, edges, output: Path) -> Path:
    """Emit a Graphviz DOT file (used when matplotlib is missing)."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dot = output.with_suffix(".dot")
    mx = max(freq.values()) if freq else 1
    lines = ["graph hapnet {", "  layout=neato;"]
    for h in haplotypes:
        w = 0.3 + 0.7 * freq.get(h["id"], 1) / mx
        lines.append(
            f'  "{h["id"]}" [shape=circle, width={w:.2f}, '
            f'label="{h["id"]}\\n({freq.get(h["id"], 1)})"];'
        )
    for a, b, d in edges:
        lines.append(f'  "{a}" -- "{b}" [label="{d}"];')
    lines.append("}")
    dot.write_text("\n".join(lines) + "\n")
    return dot


def _dot_text(haplotypes, freq, edges) -> str:
    out = Path("/tmp/_hapnet.dot")
    p = _write_dot(haplotypes, freq, edges, out)
    return p.read_text()


# ===========================================================================
# Public API
# ===========================================================================


def haplotype_network(
    alignment_fasta: str | Path,
    *,
    method: str = "auto",
    connection_limit: int | None = None,
    confidence: float = 0.95,
    epsilon: int = 0,
    population_map: dict[str, str] | None = None,
    title: str = "Haplotype network",
    backend: str = "auto",
) -> OrganelleResult:
    """Build a TCS / median-joining / minimum-spanning haplotype network.

    Algorithms (all PopART-equivalent):
      - ``method="tcs"`` (default for ``auto`` when ``connection_limit`` or
        ``confidence`` is given): Clement, Posada & Crandall (2000)
        statistical-parsimony network with the Templeton 1992 limit and
        intermediate-sequence inference.
      - ``method="mjn"``: Bandelt, Forster & Röhl (1999) median-joining network
        (infers unsampled ancestor medians).
      - ``method="msn"``: minimum-spanning network (no intermediates).

    ``backend="hapnet"`` routes through the external hapnet package when
    installed (population-aware MST with its own renderer); ``backend="auto"``
    uses the in-tree implementation unless hapnet is explicitly requested.

    ``population_map`` (sample_id → population) enables the PopART-style
    pie-chart node colouring in the rendered figure.

    This function only computes the network. Use
    :func:`write_haplotype_network` to write ``haplotypes.tsv``,
    ``network.dot``, and ``network.png``.
    """
    seqs = read_fasta(Path(alignment_fasta))
    if len(seqs) < 2:
        return failed_result(
            "haplotype_network",
            summary_text="haplotype_network needs ≥2 sequences.",
            code="phylogeny.haplotype_network.too_few_sequences",
            anomalies=["too_few_sequences"],
        )

    # external hapnet backend
    if backend == "hapnet":
        try:
            import hapnet  # type: ignore # noqa: F401
        except ImportError:
            return failed_result(
                "haplotype_network",
                summary_text="hapnet backend requested but not installed.",
                code="phylogeny.haplotype_network.backend_missing",
                anomalies=["hapnet_missing"],
            )
        return _run_hapnet_backend(seqs, population_map, title)

    # in-tree implementation
    haplotypes, members, freq, freq_by_pop = collapse_haplotypes(seqs, population_map)
    if len(haplotypes) < 2:
        return _single_haplotype_result(haplotypes, members, freq)

    # resolve method
    if method == "auto":
        method = "tcs" if (connection_limit is not None or confidence is not None) else "msn"
    seq_len = len(haplotypes[0]["sequence"])

    if method == "tcs":
        limit = (
            connection_limit
            if connection_limit is not None
            else tcs_connection_limit(len(haplotypes), seq_len, confidence)
        )
        edges, components, intermediates = build_tcs_network(
            haplotypes, members, limit, infer_intermediates=True
        )
        method_label = "tcs_clement2000"
    elif method == "mjn":
        edges, components, intermediates = build_mjn(haplotypes, epsilon=epsilon)
        limit = None
        method_label = "mjn_bandelt1999"
    else:  # msn
        limit = connection_limit
        edges, components = build_msn(haplotypes, limit)
        intermediates = []
        method_label = "msn"

    hap_info = [
        {
            "id": h["id"],
            "frequency": freq[h["id"]],
            "members": members[h["id"]],
            "counts_by_pop": freq_by_pop.get(h["id"], {}),
        }
        for h in haplotypes
    ]
    edge_info = [{"from": a, "to": b, "mutations": d} for a, b, d in edges]

    flags = ("network_built",)
    if population_map:
        flags += ("population_aware",)
    if intermediates:
        flags += ("intermediates_inferred",)

    summary = f"{len(haplotypes)} haplotypes, {len(edges)} edges, {len(components)} component(s)"
    if method == "tcs" and limit is not None:
        summary += f"; TCS connection limit = {limit}"
    if intermediates:
        summary += f"; {len(intermediates)} inferred median/intermediate vector(s)"

    return ok_result(
        "haplotype_network",
        metrics={
            "n_haplotypes": len(haplotypes),
            "n_edges": len(edges),
            "n_components": len(components),
            "connection_limit": limit,
            "method": method_label,
            "n_intermediates": len(intermediates),
            "haplotypes": hap_info,
            "edges": edge_info,
            "components": components,
            "n_samples": len(seqs),
            "intermediates": intermediates,
        },
        result_findings=findings(
            ("n_haplotypes", len(haplotypes)),
            ("n_edges", len(edges)),
            ("method", method_label),
        ),
        flags=flags,
        summary_text=summary + ".",
        result_provenance=provenance(
            "haplotype_network",
            method=method_label,
            parameters={
                "alignment_fasta": str(alignment_fasta),
                "method": method,
                "connection_limit": connection_limit,
                "confidence": confidence,
                "epsilon": epsilon,
                "backend": backend,
                "population_aware": bool(population_map),
            },
        ),
    )


def write_haplotype_network(
    result: OrganelleResult | Mapping[str, Any],
    output_dir: str | Path,
    *,
    title: str = "Haplotype network",
) -> dict[str, Path]:
    """Write haplotype-network TSV/DOT/PNG artifacts from a computed result."""
    metrics: Mapping[str, Any] = (
        result.metrics if isinstance(result, OrganelleResult) else dict(result)
    )
    hap_info = list(metrics.get("haplotypes", ()))
    edge_info = list(metrics.get("edges", ()))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tsv = out / "haplotypes.tsv"
    tsv.write_text(
        "haplotype\tfrequency\tmembers\n"
        + "\n".join(
            f"{h['id']}\t{h.get('frequency', 0)}\t{','.join(h.get('members', ()))}"
            for h in hap_info
        )
        + ("\n" if hap_info else "")
    )

    haplotypes = [{"id": str(h["id"]), "sequence": str(h.get("sequence", ""))} for h in hap_info]
    members = {str(h["id"]): list(h.get("members", ())) for h in hap_info}
    freq = {str(h["id"]): int(h.get("frequency", 0)) for h in hap_info}
    freq_by_pop = {str(h["id"]): dict(h.get("counts_by_pop", {})) for h in hap_info}
    edges = [
        (str(e.get("from", "")), str(e.get("to", "")), int(e.get("mutations", 0)))
        for e in edge_info
    ]

    dot = out / "network.dot"
    dot.write_text(_dot_text(haplotypes, freq, edges))
    img = _render_network(
        haplotypes,
        members,
        freq,
        edges,
        out / "network.png",
        freq_by_pop=freq_by_pop if any(freq_by_pop.values()) else None,
        title=title,
    )
    return {"haplotypes": tsv, "dot": dot, "image": img}


def _single_haplotype_result(haplotypes, members, freq):
    hid = haplotypes[0]["id"]
    return ok_result(
        "haplotype_network",
        metrics={
            "n_haplotypes": 1,
            "n_edges": 0,
            "edges": [],
            "n_components": 1,
            "components": {"net1": [hid]},
            "method": "tcs_clement2000",
            "n_intermediates": 0,
            "haplotypes": [{"id": hid, "frequency": freq[hid], "members": members[hid]}],
        },
        result_findings=findings(("n_haplotypes", 1), ("monomorphic", True)),
        flags=("monomorphic",),
        summary_text="Single haplotype (all sequences identical).",
        result_provenance=provenance("haplotype_network", method="tcs_clement2000"),
    )


def _run_hapnet_backend(seqs, population_map, title):
    """External hapnet (PyPI) backend — population-aware MST."""
    import tempfile

    import hapnet  # type: ignore

    pop = "all"
    prepared = [
        (
            f"{n}_{population_map.get(n, pop)}"
            if population_map and "_" not in n
            else (f"{n}_{pop}" if "_" not in n else n),
            s,
        )
        for n, s in seqs
    ]
    with tempfile.TemporaryDirectory(prefix="organelleverse_hapnet_") as tmp:
        tmp_fa = Path(tmp) / "input_pop.fa"
        tmp_fa.write_text("".join(f">{n}\n{s}\n" for n, s in prepared))
        records = hapnet.read_fasta(str(tmp_fa))
    haps, _ = hapnet.build_haplotypes(records)
    G = hapnet.build_mst_network(haps)
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append(
            {
                "from": _hap_id(u, haps),
                "to": _hap_id(v, haps),
                "mutations": int(data.get("weight", 0)),
            }
        )
    hap_info = [
        {
            "id": h.hap_id,
            "frequency": int(h.n_total),
            "members": list(h.members),
            "counts_by_pop": dict(h.counts_by_pop) if hasattr(h, "counts_by_pop") else {},
        }
        for h in haps
    ]

    import networkx as nx  # type: ignore

    comps = {
        f"net{i + 1}": sorted([_hap_id(n, haps) for n in c])
        for i, c in enumerate(nx.connected_components(G))
    }

    return ok_result(
        "haplotype_network",
        metrics={
            "n_haplotypes": len(haps),
            "n_edges": len(edges),
            "edges": edges,
            "haplotypes": hap_info,
            "components": comps,
            "n_components": len(comps),
            "n_samples": len(seqs),
            "method": "hapnet_mst",
        },
        result_findings=findings(
            ("n_haplotypes", len(haps)),
            ("n_edges", len(edges)),
            ("method", "hapnet_mst"),
        ),
        flags=("network_built", "population_aware") if population_map else ("network_built",),
        summary_text=(f"{len(haps)} haplotypes, {len(edges)} edges (hapnet MST)."),
        result_provenance=provenance(
            "haplotype_network",
            method="hapnet_mst",
            parameters={"backend": "hapnet", "population_aware": bool(population_map)},
        ),
    )


def _hap_id(node, haps):
    if isinstance(node, int) and 0 <= node < len(haps):
        return haps[node].hap_id
    return str(node)
