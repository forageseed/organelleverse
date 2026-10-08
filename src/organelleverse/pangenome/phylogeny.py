"""Node-presence Jaccard similarity trees with reproducible locus bootstrap.

The tree is a phenetic summary of graph-node presence, not a substitute for a
sequence evolution model. Linked nodes are not independent loci; column/node
bootstrap support can be optimistic when graph segmentation creates correlated
characters.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform


def _validated_pav(paths: Sequence[str], matrix: Sequence[Sequence[int]]) -> np.ndarray:
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("Node PAV requires at least two uniquely named paths")
    if any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("Path names must be nonempty strings")
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[1] != len(paths) or values.shape[0] == 0:
        raise ValueError("Node PAV must be a nonempty nodes-by-paths matrix")
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Node PAV values must be binary")
    return values.astype(np.int64)


def _distance(values: np.ndarray) -> np.ndarray:
    intersection = values.T @ values
    counts = values.sum(axis=0)
    union = counts[:, None] + counts[None, :] - intersection
    # Jaccard distance for two empty sets is zero.
    similarity = np.divide(
        intersection, union, out=np.ones(union.shape, dtype=float), where=union != 0
    )
    return 1.0 - similarity


def jaccard_distances(paths: Sequence[str], matrix: Sequence[Sequence[int]]) -> list[list[float]]:
    """Return path-by-path unweighted Jaccard distances from node-by-path PAV."""
    return _distance(_validated_pav(paths, matrix)).tolist()


def _tree(distances: np.ndarray) -> Any:
    return to_tree(linkage(squareform(distances, checks=False), method="average"))


def _members(node: Any) -> tuple[int, ...]:
    return tuple(sorted(node.pre_order(lambda leaf: leaf.id)))


def _internal_clades(node: Any) -> list[tuple[int, ...]]:
    if node.is_leaf():
        return []
    return [_members(node), *_internal_clades(node.left), *_internal_clades(node.right)]


def _quoted(label: str) -> str:
    return "'" + label.replace("'", "''") + "'"


def _newick(
    node: Any,
    paths: Sequence[str],
    support: dict[tuple[int, ...], float],
    parent_height: float | None,
) -> str:
    height = node.dist / 2.0
    if node.is_leaf():
        body = _quoted(paths[node.id])
    else:
        body = (
            "("
            + _newick(node.left, paths, support, height)
            + ","
            + _newick(node.right, paths, support, height)
            + ")"
        )
        if parent_height is not None and _members(node) in support:
            body += f"{100.0 * support[_members(node)]:.6g}"
    if parent_height is not None:
        body += f":{parent_height - height:.10g}"
    return body


def node_pav_tree(
    paths: Sequence[str],
    matrix: Sequence[Sequence[int]],
    *,
    bootstrap_replicates: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """UPGMA with Felsenstein resampling of node rows, preserving path columns.

    Internal support measures the fraction of bootstrap trees containing the
    exact rooted clade. Deterministic average-linkage tie breaking uses input
    path order. Every replicate resamples exactly the original number of nodes.
    """
    values = _validated_pav(paths, matrix)
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int):
        raise ValueError("bootstrap_replicates must be a nonnegative integer")
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    distances = _distance(values)
    root = _tree(distances)
    counts: Counter[tuple[int, ...]] = Counter()
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap_replicates):
        indices = rng.integers(0, values.shape[0], size=values.shape[0])
        counts.update(_internal_clades(_tree(_distance(values[indices]))))
    clades = [clade for clade in _internal_clades(root) if len(clade) < len(paths)]
    supports = (
        {clade: counts[clade] / bootstrap_replicates for clade in clades}
        if bootstrap_replicates
        else {}
    )
    return {
        "method": "Jaccard distance; UPGMA (average linkage)",
        "paths": list(paths),
        "distances": distances.tolist(),
        "newick": _newick(root, paths, supports, None) + ";",
        "clades": [
            {
                "paths": [paths[index] for index in clade],
                "support": supports.get(clade),
                "replicate_count": counts[clade] if bootstrap_replicates else None,
            }
            for clade in clades
        ],
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "resampling_unit": "graph_node",
        "node_count": int(values.shape[0]),
        "interpretation": (
            "Presence-based similarity tree, not a model-based evolutionary phylogeny. "
            "Linked graph nodes violate independent-character bootstrap assumptions; "
            "support depends on graph segmentation. Ties follow input path order."
        ),
    }
