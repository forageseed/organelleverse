"""Exact minimal acyclic superbubbles in an oriented GFA segment digraph.

Algorithm: Onodera, Sadakane & Shibuya (WABI 2013), Fig. 5,
https://arxiv.org/abs/1307.7925. Worst case O(V(V+E)); this is not a
linear-time claim, a bounded diamond search, or a bidirected snarl detector.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..core.artifacts import ArtifactRef
from ._gfa import _flip, load_gfa

Vertex = tuple[str, str]
DEFINITION = (
    "Onodera2013 minimal acyclic superbubbles in the oriented-segment digraph: "
    "exit reachable from entrance; forward reachability stopping at exit equals "
    "reverse reachability stopping at entrance; induced region acyclic; no earlier "
    "exit satisfies these conditions. Reverse-complement regions are separate "
    "oriented records, not independent biological variants."
)


def _exit(
    entrance: Vertex, children: dict[Vertex, dict[Vertex, None]], parents: dict[Vertex, set[Vertex]]
) -> tuple[Vertex, set[Vertex]] | None:
    """Published per-entrance topological frontier algorithm, with indegree counters."""
    ready = [entrance]
    seen: set[Vertex] = set()
    visited: set[Vertex] = set()
    remaining: dict[Vertex, int] = {}
    while ready:
        vertex = ready.pop()
        visited.add(vertex)
        seen.discard(vertex)
        if not children[vertex]:
            return None
        for child in children[vertex]:
            if child == entrance:
                return None
            seen.add(child)
            remaining[child] = remaining.get(child, len(parents[child])) - 1
            if remaining[child] == 0:
                ready.append(child)
        if len(ready) == 1 and len(seen) == 1:
            exit_vertex = ready[0]
            if entrance in children[exit_vertex]:
                return None
            return exit_vertex, visited | {exit_vertex}
    return None


def superbubbles(gfa_path: str | Path, *, include_trivial: bool = False) -> dict[str, Any]:
    """Enumerate all oriented superbubbles, allowing arbitrary acyclic interiors.

    Each L record induces its oriented arc and the reverse-complement arc.
    Parallel arcs collapse for reachability; their multiplicity remains in source
    GFA statistics. Acyclic regions may occur inside otherwise cyclic graphs.
    Trivial two-vertex regions are excluded by default (a simple chain is not
    displayed as a sequence-variation bubble). A nontrivial minimal superbubble
    necessarily has entrance outdegree >= 2. No cyclic snarls or inversion
    genotypes are inferred. All matching paths are reported as touching paths,
    not asserted complete alternative alleles.
    """
    graph = load_gfa(gfa_path)
    vertices = [(node, orientation) for node in graph.segments for orientation in ("+", "-")]
    children: dict[Vertex, dict[Vertex, None]] = {v: {} for v in vertices}
    parents: dict[Vertex, set[Vertex]] = {v: set() for v in vertices}
    for link in graph.links:
        for source, target in (
            (link.source, link.target),
            (_flip(link.target), _flip(link.source)),
        ):
            children[source][target] = None
            parents[target].add(source)
    node_paths: dict[str, set[str]] = defaultdict(set)
    for name, path in graph.paths.items():
        for node, _ in path.steps:
            node_paths[node].add(name)
    source_order = {vertex: index for index, vertex in enumerate(vertices)}
    node_order = {node: index for index, node in enumerate(graph.segments)}
    path_order = {name: index for index, name in enumerate(graph.paths)}
    records = []
    by_pair = {}
    for index, entrance in enumerate(vertices):
        if not children[entrance] or (not include_trivial and len(children[entrance]) < 2):
            continue
        found = _exit(entrance, children, parents)
        if found is None:
            continue
        exit_vertex, region = found
        if not include_trivial and len(region) == 2:
            continue
        nodes = {node for node, _ in region}
        touching = set().union(*(node_paths[node] for node in nodes))
        record = {
            "id": f"sb{index + 1:08d}",
            "entrance": {"node_id": entrance[0], "orientation": entrance[1]},
            "exit": {"node_id": exit_vertex[0], "orientation": exit_vertex[1]},
            "node_ids": sorted(nodes, key=node_order.__getitem__),
            "oriented_nodes": [
                {"node_id": node, "orientation": orient}
                for node, orient in sorted(region, key=source_order.__getitem__)
            ],
            "path_names": sorted(touching, key=path_order.__getitem__),
            "sample_names": sorted({graph.paths[name].sample for name in touching}),
            "oriented_node_count": len(region),
            "interior_oriented_node_count": len(region) - 2,
            "trivial": len(region) == 2,
        }
        records.append(record)
        by_pair[(entrance, exit_vertex)] = record["id"]
    for record in records:
        entrance = (record["entrance"]["node_id"], record["entrance"]["orientation"])
        exit_vertex = (record["exit"]["node_id"], record["exit"]["orientation"])
        record["reverse_complement_id"] = by_pair.get((_flip(exit_vertex), _flip(entrance)))
    return {
        "schema": "organelleverse.pangenome.superbubbles.v1",
        "source_graph": ArtifactRef.from_path(
            gfa_path, kind="pangenome_graph", format="gfa"
        ).model_dump(mode="json"),
        "algorithm": "onodera2013",
        "definition": DEFINITION,
        "include_trivial": include_trivial,
        "graph_model": "oriented_segment_digraph",
        "total_bubbles": len(records),
        "oriented_vertices": len(vertices),
        "oriented_arcs": sum(map(len, children.values())),
        "path_membership_semantics": "paths touching any region node; not necessarily complete traversals",
        "bubbles": records,
    }
