"""Infer graph configurations supported by consecutive junctions in individual reads.

This is a conditional inference on the retained assembly graph, not a physical
chromosome certificate. Junction-spanning reads identify local sequence order;
they cannot distinguish a circular molecule from a linear tandem copy of it.
Marginal repeat pairings are deliberately never multiplied into global isoforms.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

from .graph_compare import parse_gfa


def _flip(label: str) -> str:
    return label[:-1] + ("-" if label[-1] == "+" else "+")


def _canonical(path: tuple[str, ...]) -> tuple[str, ...]:
    return min(path, tuple(_flip(x) for x in reversed(path)))


def _edge(a: str, b: str) -> tuple[str, str]:
    return min((a, b), (_flip(b), _flip(a)))


def _cycle(path: tuple[str, ...]) -> tuple[str, ...]:
    """Canonical cyclic order, including reverse complement and rotation."""
    rc = tuple(_flip(x) for x in reversed(path))
    return min(p[i:] + p[:i] for p in (path, rc) for i in range(len(p)))


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("configuration evidence needs nonnegative integer read counts")
    return value


def infer_configurations(gfa: Path, evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Describe local alternatives, observed linkage and unresolved graph topology.

    Uses the same filtered GFA and junction evidence as sequence export. Older
    evidence remains readable, but reports that joint read linkage is unavailable.
    Zero support means unobserved at the supplied sampling depth, never impossible.
    """
    segments, raw_links = parse_gfa(gfa)
    if not segments or any(sequence == "*" for sequence in segments.values()):
        raise ValueError("configuration inference requires a nonempty graph with sequences")
    labels = {name + orientation for name in segments for orientation in ("+", "-")}
    links: list[tuple[str, str, int]] = []
    seen: set[tuple[tuple[str, str], int]] = set()
    for link in raw_links:
        a, b = link.source + link.source_orient, link.target + link.target_orient
        if a not in labels or b not in labels:
            raise ValueError("configuration graph contains an unknown segment or orientation")
        key = (_edge(a, b), link.overlap)
        if key not in seen:
            seen.add(key)
            links.append((a, b, link.overlap))
    measured = evidence.get("links")
    if not isinstance(measured, list) or len(measured) != len(links):
        raise ValueError("configuration evidence does not match the graph's unique adjacencies")
    adj: dict[str, set[tuple[str, int]]] = defaultdict(set)
    incoming: dict[str, set[tuple[str, int]]] = defaultdict(set)
    neighbors: dict[str, set[str]] = defaultdict(set)
    counts: dict[tuple[str, str, int], int] = {}
    for (a, b, overlap), record in zip(links, measured, strict=True):
        if record["from"] not in labels or record["to"] not in labels:
            raise ValueError("configuration evidence adjacency names an unknown oriented segment")
        if _edge(record["from"], record["to"]) != _edge(a, b):
            raise ValueError("configuration evidence adjacency order differs from the graph")
        n = _count(record["reads"])
        for left, right in ((a, b), (_flip(b), _flip(a))):
            adj[left].add((right, overlap))
            incoming[right].add((left, overlap))
            counts[left, right, overlap] = n
        neighbors[a[:-1]].add(b[:-1])
        neighbors[b[:-1]].add(a[:-1])

    def paths(key: str) -> list[tuple[tuple[str, ...], int]]:
        records = evidence.get(key, [])
        if not isinstance(records, list):
            raise ValueError(f"configuration evidence {key} must be a list")
        result = []
        for record in records:
            walk = tuple(record["path"].split())
            if len(walk) < 3 or any(x not in labels for x in walk):
                raise ValueError(f"invalid {key} path in configuration evidence")
            if any(not any(to == b for to, _ in adj[a]) for a, b in pairwise(walk)):
                raise ValueError(f"{key} path traverses an adjacency absent from the graph")
            result.append((_canonical(walk), _count(record["reads"])))
        return result

    traversals = paths("traversals")
    walks = paths("observed_walks")
    phased = paths("phased_paths")
    repeat_names = set(evidence.get("repeat_segments", []))
    if not repeat_names <= segments.keys():
        raise ValueError("configuration evidence names a repeat absent from the graph")
    local = []
    for name in sorted(repeat_names):
        repeat = name + "+"
        entries = sorted(x for x, _ in incoming[repeat])
        exits = sorted(x for x, _ in adj[repeat])
        observed: dict[tuple[str, str], int] = defaultdict(int)
        for walk, n in traversals:
            if len(walk) != 3 or walk[1][:-1] != name:
                continue
            oriented = walk if walk[1] == repeat else tuple(_flip(x) for x in reversed(walk))
            observed[oriented[0], oriented[2]] += n
        alternatives = any(sum(observed[entry, exit] > 0 for exit in exits) > 1 for entry in entries)
        alternatives |= any(sum(observed[entry, exit] > 0 for entry in entries) > 1 for exit in exits)
        local.append({
            "repeat": name,
            "length": len(segments[name]),
            "status": "alternative_contexts_observed" if alternatives else (
                "contexts_observed" if any(observed.values()) else "unspanned"
            ),
            "entries": entries,
            "exits": exits,
            "contexts": [
                {"path": " ".join((entry, repeat, exit)), "reads": observed[entry, exit]}
                for entry in entries for exit in exits
            ],
        })

    repeat_linkage = []
    for a, b, overlap in links:
        if a[:-1] not in repeat_names or b[:-1] not in repeat_names:
            continue
        key = _edge(a, b)
        linked = [
            {"path": " ".join(walk), "reads": n}
            for walk, n in phased if n and any(_edge(x, y) == key for x, y in pairwise(walk))
        ]
        repeat_linkage.append({
            "adjacency": " ".join(key),
            "combined_repeat_length": len(segments[a[:-1]]) + len(segments[b[:-1]]) - overlap,
            "status": "anchor_linkage_observed" if linked else "anchor_linkage_unresolved",
            "joint_paths": linked,
        })

    remaining = set(segments)
    components = []
    while remaining:
        todo = [min(remaining)]
        names: set[str] = set()
        while todo:
            name = todo.pop()
            if name in names:
                continue
            names.add(name)
            todo.extend(neighbors[name] - names)
        remaining -= names
        oriented = {name + sign for name in names for sign in ("+", "-")}
        component_links = [(a, b, o) for a, b, o in links if a[:-1] in names]
        unsupported = [" ".join((a, b)) for a, b, o in component_links if counts[a, b, o] == 0]
        cycle: tuple[str, ...] | None = None
        if all(len(adj[x]) == len(incoming[x]) == 1 for x in oriented):
            start = min(names) + "+"
            walk = []
            current = start
            while current not in walk:
                walk.append(current)
                current = next(iter(adj[current]))[0]
            if current == start and len(walk) == len(names) and {x[:-1] for x in walk} == names:
                cycle = _cycle(tuple(walk))
        status = "simple_cycle" if cycle is not None else (
            "open_chain" if all(len(adj[x]) <= 1 and len(incoming[x]) <= 1 for x in oriented)
            else "branching_graph"
        )
        chain: tuple[str, ...] | None = None
        if status == "open_chain":
            orders = set()
            for start in sorted(x for x in oriented if not incoming[x]):
                walk = []
                current = start
                while current not in walk:
                    walk.append(current)
                    if not adj[current]:
                        if len(walk) == len(names) and {x[:-1] for x in walk} == names:
                            orders.add(_canonical(tuple(walk)))
                        break
                    current = next(iter(adj[current]))[0]
            if len(orders) == 1:
                chain = orders.pop()
        component_repeats = [x for x in local if x["repeat"] in names]
        components.append({
            "segments": sorted(names),
            "topology": status,
            "graph_order_identified": (cycle is not None or chain is not None) and not unsupported,
            "cyclic_order": " ".join(cycle) if cycle else None,
            "linear_order": " ".join(chain) if chain else None,
            "unobserved_junctions": unsupported,
            "unspanned_repeats": [x["repeat"] for x in component_repeats if x["status"] == "unspanned"],
            "alternative_repeat_contexts": [x["repeat"] for x in component_repeats
                                           if x["status"] == "alternative_contexts_observed"],
            "unresolved_repeat_links": [x["adjacency"] for x in repeat_linkage
                                        if x["adjacency"].split()[0][:-1] in names
                                        and x["status"] == "anchor_linkage_unresolved"],
        })

    # A whole observed chain links its repeat choices. Separate chains, even with
    # matching endpoints, are not joined: they may come from different molecules.
    observed_cycles = []
    for walk, n in walks:
        orders = set()
        for i, start in enumerate(walk[:-1]):
            # First return to this oriented segment is a witnessed cyclic order.
            # A longer read may link it to other cycles; keep that supporting walk.
            for j in range(i + 1, len(walk)):
                if walk[j] == start:
                    orders.add(_cycle(walk[i:j]))
                    break
        if n:
            observed_cycles.extend({
                "cyclic_order": " ".join(order), "reads": n, "supporting_walk": " ".join(walk),
            } for order in sorted(orders))
    return {
        "schema": "ovasm.configuration.v1",
        "scope": "sequence-order inference conditional on the retained graph and supplied reads",
        "joint_read_linkage_available": "observed_walks" in evidence,
        "read_length_summary": {
            "maximum_read_length": evidence.get("maximum_read_length"),
            "junction_min_anchor": evidence.get("params", {}).get("min_anchor"),
            "scope": "Length of supplied reads, including uninformative reads; length eligibility is not locus-spanning evidence",
        },
        "physical_molecule_count": None,
        "physical_topology_identified": False,
        "graph_order_identified": all(x["graph_order_identified"] for x in components),
        "components": components,
        "repeat_contexts": local,
        "repeat_linkage": repeat_linkage,
        "observed_walks": [{"path": " ".join(w), "reads": n} for w, n in walks],
        "phased_paths": [{"path": " ".join(w), "reads": n} for w, n in phased],
        "observed_cyclic_orders": observed_cycles,
        "interpretation": [
            "Read counts describe observed sequence contexts; no read count is a molecule count.",
            "Counts of different walks can share reads and must not be summed as independent samples.",
            "An unobserved context is unresolved, rather than excluded, without a detection model.",
            "Local repeat choices are linked only where one consecutive read walk spans them.",
            "Cyclic sequence order also fits a linear tandem molecule; molecular shape needs additional evidence.",
        ],
    }
