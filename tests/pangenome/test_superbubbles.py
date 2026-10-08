"""Compare the traversal algorithm with an independent literal-definition oracle."""

import itertools
import random

from organelleverse.pangenome.superbubbles import _exit, superbubbles


def oracle(vertices, edges):
    adjacency = {v: {b for a, b in edges if a == v} for v in vertices}
    reverse = {v: {a for a, b in edges if b == v} for v in vertices}

    def reach(start, stop, links):
        reached, pending = {start}, [start]
        while pending:
            node = pending.pop()
            if node == stop:
                continue
            for child in links[node] - reached:
                reached.add(child)
                pending.append(child)
        return reached

    candidates = {}
    for source, target in itertools.permutations(vertices, 2):
        region = reach(source, target, adjacency)
        if target not in region or region != reach(target, source, reverse):
            continue
        pending = set(region)
        while pending:
            roots = {v for v in pending if not reverse[v] & pending}
            if not roots:
                break
            pending -= roots
        if not pending:
            candidates[(source, target)] = region
    return {
        (source, target): region
        for (source, target), region in candidates.items()
        if not any((source, other) in candidates for other in region - {source, target})
    }


def check(vertices, edges):
    children = {v: {target: None for source, target in edges if source == v} for v in vertices}
    parents = {v: {source for source, target in edges if target == v} for v in vertices}
    actual = {}
    for source in vertices:
        found = _exit(source, children, parents)
        if found:
            target, nodes = found
            actual[(source, target)] = nodes
    assert actual == oracle(vertices, edges), edges


def test_all_four_vertex_directed_graphs_match_definition():
    vertices = [(str(i), "+") for i in range(4)]
    possible = list(itertools.permutations(vertices, 2))
    for mask in range(1 << len(possible)):
        check(vertices, [edge for index, edge in enumerate(possible) if mask & (1 << index)])


def test_random_cyclic_selfloop_and_external_entry_graphs_match_definition():
    rng = random.Random(71)
    for size in range(2, 9):
        vertices = [(str(i), "+") for i in range(size)]
        for _ in range(80):
            edges = [edge for edge in itertools.product(vertices, repeat=2) if rng.random() < 0.23]
            check(vertices, edges)


def write_graph(tmp_path, edges, paths=""):
    nodes = sorted({node for source, target in edges for node in (source, target)})
    path = tmp_path / "graph.gfa"
    path.write_text(
        "".join(f"S\t{node}\tA\n" for node in nodes)
        + "".join(f"L\t{s}\t+\t{t}\t+\t0M\n" for s, t in edges)
        + paths
    )
    return path


def test_nested_multi_edge_arms_and_mirror_pairs(tmp_path):
    edges = [
        ("0", "1"),
        ("0", "5"),
        ("1", "2"),
        ("1", "3"),
        ("2", "4"),
        ("3", "4"),
        ("4", "6"),
        ("5", "6"),
    ]
    graph = write_graph(tmp_path, edges, "P\tsample#1#chr\t0+,1+,2+,4+,6+\t*\n")
    result = superbubbles(graph)
    assert result["total_bubbles"] == 4
    forward = [row for row in result["bubbles"] if row["entrance"]["orientation"] == "+"]
    assert [(row["entrance"]["node_id"], row["exit"]["node_id"]) for row in forward] == [
        ("0", "6"),
        ("1", "4"),
    ]
    assert len(forward[0]["node_ids"]) == 7
    assert all(row["reverse_complement_id"] for row in result["bubbles"])
    assert all(row["sample_names"] == ["sample"] for row in result["bubbles"])


def test_parallel_links_do_not_invent_bubbles_and_trivial_scope_explicit(tmp_path):
    graph = write_graph(tmp_path, [("1", "2"), ("1", "2"), ("2", "3")])
    assert superbubbles(graph)["total_bubbles"] == 0
    result = superbubbles(graph, include_trivial=True)
    assert result["total_bubbles"] == 4
    assert all(row["trivial"] for row in result["bubbles"])


def test_tip_and_external_entry_destroy_matching(tmp_path):
    diamond = [("0", "1"), ("0", "2"), ("1", "3"), ("2", "3")]
    for extra in [[("1", "4")], [("4", "1")], [("1", "1")], [("3", "0")]]:
        assert superbubbles(write_graph(tmp_path, diamond + extra))["total_bubbles"] == 0


def test_inversion_orientations_are_retained(tmp_path):
    graph = tmp_path / "inversion.gfa"
    graph.write_text(
        "S\t1\tA\nS\t2\tC\nS\t3\tG\nS\t4\tT\nL\t1\t+\t2\t-\t0M\nL\t1\t+\t3\t+\t0M\nL\t2\t-\t4\t+\t0M\nL\t3\t+\t4\t+\t0M\n"
    )
    rows = superbubbles(graph)["bubbles"]
    assert len(rows) == 2
    assert {"node_id": "2", "orientation": "-"} in rows[0]["oriented_nodes"]
