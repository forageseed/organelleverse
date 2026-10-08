"""Validated GFA1 path/node analytics and bounded local graph views.

S/L/P/W parsing follows https://gfa-spec.github.io/GFA-spec/GFA1.html.
Unknown sequence and ambiguous overlap are never converted into sample genomes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

from ._gfa import (
    _COMPLEMENT,
    _DNA,
    _STEP,
    GFAGraph,
    GraphPath,
    _exact_overlap,
    _flip,
    _read_lines,
    load_gfa,
)


def _overlaps(graph: GFAGraph, path: GraphPath) -> list[int]:
    result = []
    for index, pair in enumerate(pairwise(path.steps)):
        explicit = path.overlaps[index] if path.overlaps is not None else "*"
        if explicit != "*":
            result.append(_exact_overlap(explicit))
            continue
        values = graph.transitions[pair]
        exact = {_exact_overlap(c) for c in values}
        if len(exact) != 1:
            raise ValueError(f"ambiguous overlap for path {path.name}: {pair}")
        result.append(exact.pop())
    return result


def _steps(graph: GFAGraph, path: GraphPath) -> list[dict[str, Any]]:
    overlaps = _overlaps(graph, path)
    position = path.start or 0
    result = []
    for index, (node, orientation) in enumerate(path.steps):
        length = graph.segments[node].length
        if length is None:
            raise ValueError(f"unknown segment length: {node}")
        overlap = overlaps[index - 1] if index else 0
        position -= overlap
        result.append(
            {
                "node": node,
                "orientation": orientation,
                "start": position,
                "end": position + length,
                "node_length": length,
                "node_len": length,
                "overlap_previous": overlap,
                "coordinate_system": "molecule"
                if path.kind == "W" and path.start is not None
                else "path_local",
            }
        )
        position += length
    return result


def path_steps(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Full node spans on each path; overlapping nodes have overlapping spans.

    P coordinates start at zero; W starts at declared SeqStart, or zero when
    unspecified. Repeated node visits remain distinct rows. Spans are half-open.
    """
    graph = load_gfa(path)
    return {name: _steps(graph, record) for name, record in graph.paths.items()}


def _sequence(graph: GFAGraph, path: GraphPath) -> str:
    overlaps = _overlaps(graph, path)
    sequence = ""
    previous = ""
    for index, (node, orient) in enumerate(path.steps):
        seq = graph.segments[node].sequence
        if seq is None:
            raise ValueError(f"segment {node} has no stored sequence")
        if not _DNA.fullmatch(seq):
            raise ValueError(f"segment {node} sequence is not IUPAC DNA")
        seq = seq.upper()
        if orient == "-":
            seq = seq.translate(_COMPLEMENT)[::-1]
        overlap = overlaps[index - 1] if index else 0
        if overlap and previous[-overlap:] != seq[:overlap]:
            raise ValueError(f"overlap sequences disagree on path {path.name} at node {node}")
        sequence += seq[overlap:]
        previous = seq
    return sequence


def path_sequences(path: str | Path) -> dict[str, str]:
    """Spell every stored P/W independently; never join molecules or W intervals."""
    graph = load_gfa(path)
    if not graph.paths:
        raise ValueError("GFA has no paths to convert; segment sequences are not sample genomes")
    return {name: _sequence(graph, record) for name, record in graph.paths.items()}


def _classify(count: int, denominator: int, threshold: float, core_threshold: float = 1.0) -> str:
    if count == 0:
        return "unobserved"
    if count / denominator >= core_threshold:
        return "core"
    return "cloud" if count / denominator <= threshold else "shell"


def node_pav(
    path: str | Path, cloud_threshold: float = 0.05, *, core_threshold: float = 1.0
) -> dict[str, Any]:
    """Binary node x path presence, plus explicit OR aggregation by sample.

    Core=frequency>=core_threshold (strict 1.0 by default);
    cloud=0<frequency<=cloud_threshold; shell=cloud_threshold<frequency<core_threshold;
    unobserved=0. PanSN sample grouping only recognizes sample#integer#molecule;
    otherwise each P name is its own sample. W uses its explicit SampleId.
    """
    if not 0 <= cloud_threshold < core_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= cloud_threshold < core_threshold <= 1")
    graph = load_gfa(path)
    if not graph.paths:
        raise ValueError("node PAV requires P or W records; no paths are present")
    paths, nodes = list(graph.paths), list(graph.segments)
    samples = list(dict.fromkeys(p.sample for p in graph.paths.values()))
    path_samples = [p.sample for p in graph.paths.values()]
    presence = {name: {node for node, _ in record.steps} for name, record in graph.paths.items()}
    matrix = [[int(node in presence[name]) for name in paths] for node in nodes]
    sample_paths = {
        sample: [i for i, value in enumerate(path_samples) if value == sample] for sample in samples
    }
    sample_matrix = [
        [int(any(row[i] for i in sample_paths[sample])) for sample in samples] for row in matrix
    ]
    counts, sample_counts = [sum(row) for row in matrix], [sum(row) for row in sample_matrix]
    return {
        "paths": paths,
        "nodes": nodes,
        "matrix": matrix,
        "counts": counts,
        "frequencies": [n / len(paths) for n in counts],
        "classes": [_classify(n, len(paths), cloud_threshold, core_threshold) for n in counts],
        "path_samples": path_samples,
        "samples": samples,
        "sample_matrix": sample_matrix,
        "sample_counts": sample_counts,
        "sample_frequencies": [n / len(samples) for n in sample_counts],
        "sample_classes": [
            _classify(n, len(samples), cloud_threshold, core_threshold) for n in sample_counts
        ],
        "cloud_threshold": cloud_threshold,
        "core_threshold": core_threshold,
        "sample_aggregation": "OR across all paths/intervals assigned to a sample",
        "frequency_denominator": "paths",
        "classification": {
            "core": "core_threshold <= f <= 1",
            "shell": "cloud_threshold < f < core_threshold",
            "cloud": "0 < f <= cloud_threshold",
            "unobserved": "f = 0",
        },
    }


def graph_statistics(path: str | Path) -> dict[str, Any]:
    """Node multigraph metrics; degree counts L records (loops contribute two).

    Components ignore orientation. Path coverage is distinct-path incidence per
    node, not sequencing read depth. Length metrics are null if unavailable.
    """
    graph = load_gfa(path)
    adjacency: dict[str, set[str]] = {n: set() for n in graph.segments}
    degree = dict.fromkeys(graph.segments, 0)
    for link in graph.links:
        a, b = link.source[0], link.target[0]
        adjacency[a].add(b)
        adjacency[b].add(a)
        degree[a] += 1
        degree[b] += 1
    remaining = set(graph.segments)
    components = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack, component = [seed], []
        while stack:
            node = stack.pop()
            component.append(node)
            unseen = adjacency[node] & remaining
            remaining.difference_update(unseen)
            stack.extend(sorted(unseen, reverse=True))
        components.append(sorted(component))
    lengths = {n: s.length for n, s in graph.segments.items()}
    known = [value for value in lengths.values() if value is not None]
    complete = len(known) == len(lengths)
    total = sum(known) if complete else None
    n50, cumulative = None, 0
    if total:
        for length in sorted(known, reverse=True):
            cumulative += length
            if 2 * cumulative >= total:
                n50 = length
                break
    coverage = dict.fromkeys(graph.segments, 0)
    traversals = dict.fromkeys(graph.segments, 0)
    path_lengths, length_errors, composition = {}, {}, {}
    for name, record in graph.paths.items():
        counter = Counter(node for node, _ in record.steps)
        composition[name] = dict(counter)
        for node, count in counter.items():
            coverage[node] += 1
            traversals[node] += count
        try:
            spans = _steps(graph, record)
            path_lengths[name] = spans[-1]["end"] - spans[0]["start"]
        except ValueError as exc:
            path_lengths[name] = None
            length_errors[name] = str(exc)
    return {
        "nodes": len(graph.segments),
        "edges": len(graph.links),
        "paths": len(graph.paths),
        "p_records": sum(p.kind == "P" for p in graph.paths.values()),
        "w_records": sum(p.kind == "W" for p in graph.paths.values()),
        "components": len(components),
        "component_nodes": components,
        "total_bp": total,
        "known_total_bp": sum(known),
        "unknown_length_nodes": len(lengths) - len(known),
        "n50": n50,
        "degree": degree,
        "node_lengths": lengths,
        "degree_distribution": [
            {"degree": k, "nodes": v} for k, v in sorted(Counter(degree.values()).items())
        ],
        "node_length_distribution": [
            {"length": k, "nodes": v} for k, v in sorted(Counter(known).items())
        ],
        "path_coverage": coverage,
        "path_coverage_distribution": [
            {"paths": k, "nodes": v} for k, v in sorted(Counter(coverage.values()).items())
        ],
        "node_traversal_counts": traversals,
        "path_lengths": path_lengths,
        "path_length_errors": length_errors,
        "path_composition": composition,
        "coverage_definition": "number of distinct P/W records containing each segment",
        "edge_definition": "L record count; parallel and reciprocal records are retained",
    }


def extract_subgraph(
    path: str | Path,
    path_names: list[str] | None = None,
    *,
    node_names: list[str] | None = None,
    path_interval: tuple[str, int, int] | None = None,
) -> str:
    """Induced graph on selected paths/nodes or full nodes intersecting an interval.

    Explicit path selection preserves exactly those complete paths. Node/interval
    selection retains only complete paths lying entirely within selected nodes;
    partial traversals are omitted, never emitted with dangling references.
    Interval selection does not trim nodes or imply base-exact clipping.
    """
    graph = load_gfa(path)
    selected: set[str] = set()
    requested = set(path_names or [])
    unknown_paths = requested - graph.paths.keys()
    if unknown_paths:
        raise ValueError(f"unknown paths: {sorted(unknown_paths)}")
    for name in requested:
        selected.update(node for node, _ in graph.paths[name].steps)
    selected.update(node_names or [])
    if path_interval is not None:
        name, start, end = path_interval
        if name not in graph.paths:
            raise ValueError(f"unknown path: {name}")
        if start < 0 or end <= start:
            raise ValueError("interval must be a nonempty half-open nonnegative interval")
        spans = _steps(graph, graph.paths[name])
        if start < spans[0]["start"] or end > spans[-1]["end"]:
            raise ValueError("interval outside path coordinates")
        selected.update(s["node"] for s in spans if s["start"] < end and s["end"] > start)
    if not selected:
        raise ValueError("select at least one path, node or intersecting interval")
    unknown_nodes = selected - graph.segments.keys()
    if unknown_nodes:
        raise ValueError(f"unknown segments: {sorted(unknown_nodes)}")
    lines = list(graph.headers)
    lines.extend(record.raw for node, record in graph.segments.items() if node in selected)
    lines.extend(
        link.raw
        for link in graph.links
        if link.source[0] in selected and link.target[0] in selected
    )
    for name, record in graph.paths.items():
        if requested and name not in requested:
            continue
        if all(node in selected for node, _ in record.steps):
            lines.append(record.raw)
    _read_lines(lines)
    return "\n".join(lines) + "\n"


def inspect_topology(path: str | Path, *, max_nodes: int = 2000) -> dict[str, Any]:
    """Bounded local topology; detects only induced two-edge oriented diamonds.

    A diamond consists of source -> >=2 distinct intermediate nodes -> sink.
    Each intermediate has exactly one incoming and one outgoing oriented neighbor.
    Nodes are distinct and the source's complete outgoing neighbor set is these
    intermediates. This is not a general superbubble or biological variant caller.
    Reverse-complement duplicates are collapsed. Cyclic/rearranged structures
    outside this exact definition are not labelled bubbles.
    """
    graph = load_gfa(path)
    if max_nodes < 1 or len(graph.segments) > max_nodes:
        raise ValueError(
            f"local topology is limited to {max_nodes} nodes; extract a subgraph first"
        )
    outgoing: dict[_STEP, set[_STEP]] = defaultdict(set)
    incoming: dict[_STEP, set[_STEP]] = defaultdict(set)
    edges: Counter[tuple[_STEP, _STEP]] = Counter()
    for link in graph.links:
        pair = (link.source, link.target)
        reverse = (_flip(link.target), _flip(link.source))
        edges[min(pair, reverse)] += 1
    for source, target in graph.transitions:
        outgoing[source].add(target)
        incoming[target].add(source)
    bubbles, seen = [], set()
    for source, branches in sorted(outgoing.items()):
        if len(branches) < 2:
            continue
        if any(incoming[b] != {source} or len(outgoing[b]) != 1 for b in branches):
            continue
        sinks = {next(iter(outgoing[b])) for b in branches}
        if len(sinks) != 1:
            continue
        sink = sinks.pop()
        if len({source[0], sink[0], *(b[0] for b in branches)}) != len(branches) + 2:
            continue
        if incoming[sink] != branches:
            continue
        direct = (source, sink, tuple(sorted(branches)))
        reverse = (_flip(sink), _flip(source), tuple(sorted(_flip(b) for b in branches)))
        key = min(direct, reverse)
        if key in seen:
            continue
        seen.add(key)
        bubbles.append(
            {
                "source": "".join(key[0]),
                "sink": "".join(key[1]),
                "branches": ["".join(b) for b in key[2]],
                "kind": "two_edge_oriented_diamond",
            }
        )
    return {
        "nodes": len(graph.segments),
        "links": len(graph.links),
        "bubbles": bubbles,
        "bubble_definition": "induced two-edge oriented diamond; not general superbubbles or variant calls",
        "parallel_edges": [
            {"source": "".join(a), "target": "".join(b), "records": count}
            for (a, b), count in sorted(edges.items())
            if count > 1
        ],
        "parallel_edge_definition": "multiple L records for identical bidirected endpoints, including reciprocal records",
    }


def graph_slice(
    path: str | Path,
    offset: int = 0,
    limit: int = 100,
    *,
    node_ids: set[str] | list[str] | None = None,
) -> dict[str, Any]:
    """Read a bounded node page and its induced L edges for browser display.

    Whole graph is read only on the backend. Node order is file order, maximum
    500 nodes per response. Path summaries include only paths touching this page;
    step counts refer to complete source traversals, not sliced sequence. Optional
    node_ids restrict before pagination; an empty selection returns an empty page.
    """
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise ValueError("limit must be an integer in [1,500]")
    graph = load_gfa(path)
    all_names = list(graph.segments)
    if node_ids is not None:
        if isinstance(node_ids, str) or any(not isinstance(node, str) for node in node_ids):
            raise ValueError("node_ids must be a collection of node ID strings")
        requested = set(node_ids)
        unknown = requested - graph.segments.keys()
        if unknown:
            raise ValueError(f"Unknown graph node IDs: {sorted(unknown)}")
        matching = [name for name in all_names if name in requested]
    else:
        matching = all_names
    names = matching[offset : offset + limit]
    selected = set(names)
    return {
        "nodes": [{"id": name, "length": graph.segments[name].length} for name in names],
        "edges": [
            {
                "source": link.source[0],
                "target": link.target[0],
                "source_orientation": link.source[1],
                "target_orientation": link.target[1],
                "overlap": link.overlap,
            }
            for link in graph.links
            if link.source[0] in selected and link.target[0] in selected
        ],
        "paths": [
            {"name": name, "sample": record.sample, "steps": len(record.steps)}
            for name, record in graph.paths.items()
            if any(node in selected for node, _ in record.steps)
        ],
        "total_nodes": len(matching),
        "total_graph_nodes": len(all_names),
        "offset": offset,
        "limit": limit,
    }
