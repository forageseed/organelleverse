"""Chunked Jaccard trees and adaptive-stopping node-PAV bootstrap.

Multiplicity-weighted Jaccard is exactly ordinary node resampling; this module
is not IQ-TREE UFBoot2's likelihood-based RELL candidate-tree approximation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

from .phylogeny import _internal_clades, _newick, _tree

BatchSource = Callable[[], Iterable[np.ndarray]]


def _distance(source: BatchSource, n_nodes: int, n_samples: int, rng=None) -> np.ndarray:
    intersection = np.zeros((n_samples, n_samples), dtype=np.float64)
    counts = np.zeros(n_samples, dtype=np.float64)
    seen, remaining_draws = 0, n_nodes
    for raw in source():
        values = np.asarray(raw)
        if values.ndim != 2 or values.shape[1] != n_samples or not np.isin(values, [0, 1]).all():
            raise ValueError("PAV batches must be binary nodes-by-samples arrays")
        size = values.shape[0]
        if size == 0 or seen + size > n_nodes:
            raise ValueError("PAV batch row count differs from declared node count")
        numeric = values.astype(np.float64)
        if rng is None:
            weighted = numeric
        else:
            # Conditional multinomial allocation gives exactly n_nodes draws
            # from uniform nodes, with O(batch_rows) rather than O(n_nodes) RAM.
            remaining_nodes = n_nodes - seen
            allocated = (
                remaining_draws
                if size == remaining_nodes
                else rng.binomial(remaining_draws, size / remaining_nodes)
            )
            weights = rng.multinomial(allocated, np.full(size, 1 / size))
            remaining_draws -= allocated
            weighted = numeric * weights[:, None]
        intersection += numeric.T @ weighted
        counts += weighted.sum(axis=0)
        seen += size
    if seen != n_nodes:
        raise ValueError("PAV batches do not contain the declared node count")
    union = counts[:, None] + counts[None, :] - intersection
    similarity = np.divide(intersection, union, out=np.ones_like(union), where=union != 0)
    return 1.0 - similarity


class _PatternStore:
    """One pass over the PAV batches, collapsed to distinct presence patterns.

    Replicates repeat ``_distance``'s random calls batch by batch, so a seed
    gives the same node draws; the draws are then summed per pattern. All
    products are of integer counts, so distances equal ``_distance`` exactly.
    Memory is O(distinct patterns x samples), not O(replicates x nodes).
    """

    def __init__(
        self,
        source: BatchSource,
        n_nodes: int,
        n_samples: int,
        node_blocks: np.ndarray | None = None,
    ):
        index: dict[bytes, int] = {}
        rows: list[np.ndarray] = []
        node_pattern = np.empty(n_nodes, dtype=np.int64)
        self.batch_sizes: list[int] = []
        seen = 0
        for raw in source():
            values = np.asarray(raw)
            if values.ndim != 2 or values.shape[1] != n_samples or not np.isin(values, [0, 1]).all():
                raise ValueError("PAV batches must be binary nodes-by-samples arrays")
            size = values.shape[0]
            if size == 0 or seen + size > n_nodes:
                raise ValueError("PAV batch row count differs from declared node count")
            for offset, row in enumerate(values.astype(np.uint8)):
                key = row.tobytes()
                number = index.get(key)
                if number is None:
                    number = index[key] = len(rows)
                    rows.append(row)
                node_pattern[seen + offset] = number
            self.batch_sizes.append(size)
            seen += size
        if seen != n_nodes:
            raise ValueError("PAV batches do not contain the declared node count")
        if node_blocks is not None:
            node_blocks = np.asarray(node_blocks, dtype=np.int64)
            if node_blocks.shape != (n_nodes,) or (n_nodes and node_blocks.min() < 0):
                raise ValueError("Block labels must give one nonnegative block per PAV node")
        # float32 sums integers exactly below 2**24. Block draws can give one
        # node a large weight, so block replicates always use float64.
        exact32 = n_nodes < 2**24 and node_blocks is None
        self.patterns = np.asarray(rows, dtype=np.float32 if exact32 else np.float64)
        self.node_pattern = node_pattern
        self.n_nodes = n_nodes
        self.node_blocks = node_blocks
        self.n_blocks = int(node_blocks.max()) + 1 if node_blocks is not None and n_nodes else 0

    def _node_weights(self, rng) -> np.ndarray:
        if self.node_blocks is not None:
            # Resample whole blocks; every node of a drawn block gets its count.
            draws = np.bincount(
                rng.integers(0, self.n_blocks, self.n_blocks), minlength=self.n_blocks
            )
            return draws[self.node_blocks]
        weights = np.empty(self.n_nodes, dtype=np.int64)
        seen, remaining_draws = 0, self.n_nodes
        for size in self.batch_sizes:
            remaining_nodes = self.n_nodes - seen
            allocated = (
                remaining_draws
                if size == remaining_nodes
                else rng.binomial(remaining_draws, size / remaining_nodes)
            )
            weights[seen : seen + size] = rng.multinomial(allocated, np.full(size, 1 / size))
            remaining_draws -= allocated
            seen += size
        return weights

    def distance(self, rng=None) -> np.ndarray:
        if rng is None:
            counts = np.bincount(self.node_pattern, minlength=len(self.patterns))
        else:
            counts = np.bincount(
                self.node_pattern, weights=self._node_weights(rng), minlength=len(self.patterns)
            )
        weighted = self.patterns * counts.astype(self.patterns.dtype)[:, None]
        intersection = (self.patterns.T @ weighted).astype(np.float64)
        sizes = weighted.sum(axis=0, dtype=np.float64)
        union = sizes[:, None] + sizes[None, :] - intersection
        similarity = np.divide(intersection, union, out=np.ones_like(union), where=union != 0)
        return 1.0 - similarity


def superbubble_blocks(node_ids: Sequence[str], bubbles: Iterable[dict]) -> np.ndarray:
    """Block label per node: interiors of overlapping or nested superbubbles merge.

    Entrance and exit nodes are left out, otherwise consecutive bubbles would
    chain through their shared boundary nodes into one block. Nodes outside
    every bubble interior are their own block.
    """
    index = {node: i for i, node in enumerate(node_ids)}
    parent = list(range(len(index)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for bubble in bubbles:
        ends = {bubble["entrance"]["node_id"], bubble["exit"]["node_id"]}
        interior = [index[node] for node in bubble["node_ids"] if node not in ends]
        for other in interior[1:]:
            left, right = find(interior[0]), find(other)
            if left != right:
                parent[left] = right
    roots = np.fromiter((find(i) for i in range(len(parent))), dtype=np.int64, count=len(parent))
    return np.unique(roots, return_inverse=True)[1]


def _split(clade, n_samples):
    left = tuple(sorted(clade))
    right = tuple(sorted(set(range(n_samples)) - set(left)))
    return min((left, right), key=lambda item: (len(item), item))


def _convergence(previous, current):
    if np.array_equal(previous, current):
        return 1.0, "identical_support_vector"
    if len(current) < 2 or np.std(previous) == 0 or np.std(current) == 0:
        return None, "pearson_undefined"
    return float(np.corrcoef(previous, current)[0, 1]), "pearson"


def batch_node_pav_tree(
    paths: Sequence[str],
    n_nodes: int,
    batches: BatchSource,
    *,
    bootstrap_replicates: int = 0,
    seed: int = 0,
    bootstrap_method: str = "felsenstein",
    min_replicates: int = 1000,
    max_replicates: int = 10000,
    convergence_threshold: float = 0.99,
    batch_size: int = 100,
    consecutive_converged: int = 2,
    node_blocks: np.ndarray | None = None,
) -> dict:
    """Read the batches once; keep distinct presence patterns, not every node.

    Adaptive mode draws replicates in batches and stops once the support
    vector of the original tree's splits correlates above the threshold for
    consecutive batches. The maximum bound is exact, equal-sized splits have
    canonical ties, and undefined Pearson values never count as convergence.
    Fixed mode supports rooted clades, matching the existing node_pav_tree;
    adaptive mode supports nontrivial unrooted splits. ``superbubble_block``
    resamples whole blocks of linked nodes (``node_blocks``, one label per PAV
    row) instead of single nodes, so nodes of one bubble enter a replicate
    together; it otherwise follows fixed mode.
    """
    if len(paths) < 2 or len(set(paths)) != len(paths) or any(not x for x in paths):
        raise ValueError("Tree requires at least two uniquely named samples")
    for value in (
        n_nodes,
        bootstrap_replicates,
        seed,
        min_replicates,
        max_replicates,
        batch_size,
        consecutive_converged,
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "Node count, seed and replicate parameters must be nonnegative integers"
            )
    if n_nodes == 0 or bootstrap_method not in {"felsenstein", "adaptive_pav", "superbubble_block"}:
        raise ValueError("Nonempty PAV and a supported bootstrap method are required")
    blocked = bootstrap_method == "superbubble_block"
    if blocked != (node_blocks is not None):
        raise ValueError("Block labels are required for, and only for, superbubble_block")
    adaptive = bootstrap_method == "adaptive_pav"
    if adaptive and (
        min_replicates < 1
        or max_replicates < min_replicates
        or batch_size < 1
        or consecutive_converged < 1
        or not 0 < convergence_threshold <= 1
    ):
        raise ValueError("Adaptive bootstrap stopping parameters are invalid")
    n_samples = len(paths)
    store = _PatternStore(batches, n_nodes, n_samples, node_blocks)
    distances = store.distance()
    root = _tree(distances)
    clades = [clade for clade in _internal_clades(root) if len(clade) < n_samples]
    def keys(tree):
        return (
            {
                _split(c, n_samples)
                for c in _internal_clades(tree)
                if 1 < len(_split(c, n_samples)) < n_samples - 1
            }
            if adaptive
            else set(_internal_clades(tree))
        )
    original_keys = sorted(keys(root))
    counter = Counter()
    rng = np.random.default_rng(seed)
    limit = max_replicates if adaptive else bootstrap_replicates
    previous, rounds, converged, history = None, 0, False, []
    total = 0
    while total < limit:
        size = min(batch_size if adaptive else limit, limit - total)
        for _ in range(size):
            counter.update(keys(_tree(store.distance(rng))))
        total += size
        if adaptive:
            current = np.array([counter[key] / total for key in original_keys])
            corr, criterion = (None, "insufficient_replicates")
            if total >= min_replicates and previous is not None and original_keys:
                corr, criterion = _convergence(previous, current)
                rounds = rounds + 1 if corr is not None and corr >= convergence_threshold else 0
            history.append({"replicates": total, "correlation": corr, "criterion": criterion})
            previous = current
            if rounds >= consecutive_converged:
                converged = True
                break
    supports = {}
    for clade in clades:
        key = _split(clade, n_samples) if adaptive else clade
        if total and (not adaptive or key in original_keys):
            supports[clade] = counter[key] / total
    return {
        "method": "Jaccard distance; UPGMA (average linkage)",
        "paths": list(paths),
        "distances": distances.tolist(),
        "newick": _newick(root, paths, supports, None) + ";",
        "clades": [
            {"paths": [paths[i] for i in clade], "support": supports.get(clade)} for clade in clades
        ],
        "bootstrap_method": bootstrap_method,
        "bootstrap_replicates": total,
        "resampling_unit": "superbubble_block" if blocked else "graph_node",
        "seed": seed,
        "node_count": n_nodes,
        **({"block_count": store.n_blocks} if blocked else {}),
        "support_definition": "nontrivial_unrooted_split" if adaptive else "rooted_clade",
        "converged": converged if adaptive else None,
        "convergence_history": history,
        "stopping_parameters": {
            "min_replicates": min_replicates,
            "max_replicates": max_replicates,
            "batch_size": batch_size,
            "threshold": convergence_threshold,
            "consecutive_converged": consecutive_converged,
        }
        if adaptive
        else None,
        "interpretation": "Presence-based similarity tree; linked nodes and graph segmentation affect support. Adaptive PAV uses ordinary multinomial node resampling with adaptive stopping, not IQ-TREE likelihood UFBoot2. Ties follow input sample order.",
    }


def stored_node_pav_tree(directory: str | Path, **kwargs) -> dict:
    """Tree adapter for the versioned bounded Parquet PAV store.

    Block resampling reads the run's superbubbles.json next to the store.
    """
    import json

    from .pav_store import iter_pav_batches, iter_pav_rows, load_pav_metadata

    metadata = load_pav_metadata(directory)
    if kwargs.get("bootstrap_method") == "superbubble_block":
        bubbles_path = Path(directory) / "superbubbles.json"
        if not bubbles_path.is_file():
            raise ValueError("Superbubble block bootstrap needs the run's superbubbles.json")
        node_ids = [row["node_id"] for row in iter_pav_rows(directory, unit="sample")]
        bubbles = json.loads(bubbles_path.read_text())["bubbles"]
        kwargs["node_blocks"] = superbubble_blocks(node_ids, bubbles)
    return batch_node_pav_tree(
        metadata.samples,
        metadata.node_count,
        lambda: iter_pav_batches(directory, unit="sample"),
        **kwargs,
    )
