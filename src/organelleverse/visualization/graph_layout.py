"""Stress-majorization layout for assembly graphs drawn as fixed-width ribbons."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..structure.graph_conformation import parse_node_label

_SEGMENT_SPACING = 0.8  # layout units between chain points
_LINK_LENGTH = 0.9  # layout units between joined segment ends
_MAX_LAYOUT_POINTS = 900


def graph_layout(
    segments: Sequence[Mapping[str, Any]],
    links: Sequence[Mapping[str, Any]],
    *,
    last: Sequence[str] = (),
    seed: int = 0,
) -> dict[str, list[list[float]]]:
    """Polyline per segment from stress majorization of a segment-chain graph.

    Each segment becomes a chain of points whose drawn length grows with the
    square root of its sequence length; links join chain ends. Several starts
    are run and the layout with the fewest crossings (then lowest stress) is
    kept; components are laid out separately and packed left to right,
    largest first, with components holding a segment named in ``last`` (e.g.
    the plastid graph beside a mitochondrial one) at the right end.
    """
    max_length = max((row["length"] for row in segments), default=1) or 1
    budget = max(1.0, sum(_drawn_length(row["length"], max_length) for row in segments))
    spacing = max(_SEGMENT_SPACING, _SEGMENT_SPACING * budget / _MAX_LAYOUT_POINTS * 1.25)
    chains: dict[str, list[int]] = {}
    edges: list[tuple[int, int, float]] = []
    count = 0
    for row in segments:
        drawn = _drawn_length(row["length"], max_length)
        points = max(2, round(drawn / spacing) + 1)
        chains[row["name"]] = list(range(count, count + points))
        step = drawn / (points - 1)
        edges.extend((i, i + 1, step) for i in range(count, count + points - 1))
        count += points
    link_pairs = []
    for link in links:
        source, target = parse_node_label(link["source"]), parse_node_label(link["target"])
        a = _end_point(chains, source, exit_end=True)
        b = _end_point(chains, target, exit_end=False)
        link_pairs.append((a, b))
        if a != b:
            edges.append((a, b, _LINK_LENGTH))
    components = _components(count, edges)
    rng = np.random.default_rng(seed)
    coords = np.zeros((count, 2))
    placed = []
    trailing = {node for name in last for node in chains[name]}
    for nodes in sorted(components, key=lambda c: (bool(trailing & set(c)), -len(c))):
        index = {node: i for i, node in enumerate(nodes)}
        local_edges = [(index[a], index[b], w) for a, b, w in edges if a in index]
        pieces = _pieces(chains, link_pairs, index)
        coords_local = _best_layout(len(nodes), local_edges, pieces, rng)
        placed.append((nodes, coords_local))
    offset = 0.0
    for nodes, local in placed:
        local = _principal_axes(local)
        local[:, 0] -= local[:, 0].min()
        local[:, 1] -= (local[:, 1].min() + local[:, 1].max()) / 2
        local[:, 0] += offset
        coords[nodes] = local
        offset = local[:, 0].max() + 2.5
    if len(coords):
        coords -= coords.mean(axis=0)
    return {
        name: [[_coordinate(x), _coordinate(y)] for x, y in coords[chain]]
        for name, chain in chains.items()
    }


def _coordinate(value: float) -> float:
    """Round one layout coordinate, folding -0.0 into 0.0.

    Centering leaves residues like -1e-17 that round to -0.0. It equals 0.0
    but serializes as "-0.0", which changes the content-addressed result id of
    an otherwise identical deterministic plot.
    """
    return round(float(value), 4) + 0.0


def _drawn_length(length: int, max_length: int) -> float:
    return 1.2 + 10.0 * math.sqrt(max(length, 1) / max_length)


def _end_point(chains: Mapping[str, list[int]], node: tuple[str, str], *, exit_end: bool) -> int:
    chain = chains[node[0]]
    forward_end = (node[1] == "+") == exit_end
    return chain[-1] if forward_end else chain[0]


def _components(count: int, edges: Sequence[tuple[int, int, float]]) -> list[list[int]]:
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in edges:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for node in range(count):
        groups.setdefault(find(node), []).append(node)
    return list(groups.values())


def _pieces(chains, link_pairs, index) -> list[tuple[int, int]]:
    pieces = []
    for chain in chains.values():
        if chain[0] in index:
            pieces.extend((index[a], index[b]) for a, b in itertools.pairwise(chain))
    pieces.extend((index[a], index[b]) for a, b in link_pairs if a in index and a != b)
    return pieces


def _best_layout(n: int, edges, pieces, rng) -> np.ndarray:
    if n == 1:
        return np.zeros((1, 2))
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path

    rows = [a for a, _, _ in edges] + [b for _, b, _ in edges]
    cols = [b for _, b, _ in edges] + [a for a, _, _ in edges]
    weights = [w for _, _, w in edges] * 2
    graph = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    distances = shortest_path(graph, directed=False)
    starts = [_classical_mds(distances)] + [
        rng.normal(scale=distances.max() / 3, size=(n, 2)) for _ in range(7 if n < 400 else 3)
    ]
    best = None
    for start in starts:
        layout, stress = _stress_majorization(distances, start)
        crossings = _crossings(layout, pieces)
        key = (crossings, stress)
        if best is None or key < best[0]:
            best = (key, layout)
    assert best is not None
    return best[1]


def _classical_mds(distances: np.ndarray) -> np.ndarray:
    n = len(distances)
    centering = np.eye(n) - 1.0 / n
    gram = -0.5 * centering @ (distances**2) @ centering
    values, vectors = np.linalg.eigh(gram)
    top = np.argsort(values)[::-1][:2]
    return vectors[:, top] * np.sqrt(np.maximum(values[top], 1e-9))


def _stress_majorization(distances: np.ndarray, start: np.ndarray, iterations: int = 400):
    weights = np.zeros_like(distances)
    mask = distances > 0
    weights[mask] = distances[mask] ** -2.0
    laplacian = -weights.copy()
    np.fill_diagonal(laplacian, weights.sum(axis=1))
    inverse = np.linalg.pinv(laplacian)
    layout = start - start.mean(axis=0)
    for _ in range(iterations):
        delta = layout[:, None, :] - layout[None, :, :]
        current = np.sqrt((delta**2).sum(axis=-1))
        np.maximum(current, 1e-9, out=current)
        b = -weights * distances / current
        np.fill_diagonal(b, 0.0)
        np.fill_diagonal(b, -b.sum(axis=1))
        updated = inverse @ (b @ layout)
        moved = np.abs(updated - layout).max()
        layout = updated
        if moved < 1e-4:
            break
    delta = layout[:, None, :] - layout[None, :, :]
    current = np.sqrt((delta**2).sum(axis=-1))
    stress = float((weights * (current - distances) ** 2).sum() / 2)
    return layout, stress


def _crossings(layout: np.ndarray, pieces: Sequence[tuple[int, int]]) -> int:
    if len(pieces) < 2:
        return 0
    a = np.array([p[0] for p in pieces])
    b = np.array([p[1] for p in pieces])
    p, q = layout[a], layout[b]

    def orient(o, x, y):
        return np.sign(
            (x[..., 0] - o[..., 0]) * (y[..., 1] - o[..., 1])
            - (x[..., 1] - o[..., 1]) * (y[..., 0] - o[..., 0])
        )

    p1, q1 = p[:, None, :], q[:, None, :]
    p2, q2 = p[None, :, :], q[None, :, :]
    cross = (orient(p1, q1, p2) * orient(p1, q1, q2) < 0) & (
        orient(p2, q2, p1) * orient(p2, q2, q1) < 0
    )
    shared = (
        (a[:, None] == a[None, :])
        | (a[:, None] == b[None, :])
        | (b[:, None] == a[None, :])
        | (b[:, None] == b[None, :])
    )
    return int(np.triu(cross & ~shared, 1).sum())


def _principal_axes(layout: np.ndarray) -> np.ndarray:
    centered = layout - layout.mean(axis=0)
    if len(layout) < 3:
        return centered
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rotated = centered @ vt.T
    return rotated


__all__ = ["graph_layout"]
