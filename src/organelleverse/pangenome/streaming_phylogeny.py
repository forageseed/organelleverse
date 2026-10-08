"""Chunked Jaccard trees and MineGraph-referenced adaptive PAV bootstrap.

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
) -> dict:
    """Read replayable bounded batches; retain only sample-square distances.

    Adaptive stopping follows MineGraph b8c9ebf's multiplicity Jaccard idea.
    Improvements: exact maximum bound, canonical ties for equal-sized splits,
    and undefined Pearson values never automatically count as convergence.
    Fixed mode supports rooted clades, matching the existing node_pav_tree.
    Adaptive mode supports nontrivial unrooted splits, matching MineGraph's
    intended bipartition semantics. Neither mode treats linked nodes as loci.
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
    if n_nodes == 0 or bootstrap_method not in {"felsenstein", "adaptive_pav"}:
        raise ValueError("Nonempty PAV and a supported bootstrap method are required")
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
    distances = _distance(batches, n_nodes, n_samples)
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
            counter.update(keys(_tree(_distance(batches, n_nodes, n_samples, rng))))
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
        "resampling_unit": "graph_node",
        "seed": seed,
        "node_count": n_nodes,
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
        "algorithm_reference": "https://github.com/hubner-lab/MineGraph/blob/b8c9ebfd7511d968d8f2ce763a39a83f2242d473/src/utils.py#L2053"
        if adaptive
        else None,
        "interpretation": "Presence-based similarity tree; linked nodes and graph segmentation affect support. Adaptive PAV uses ordinary multinomial node resampling with adaptive stopping, not IQ-TREE likelihood UFBoot2. Ties follow input sample order.",
    }


def stored_node_pav_tree(directory: str | Path, **kwargs) -> dict:
    """Tree adapter for the versioned bounded Parquet PAV store."""
    from .pav_store import iter_pav_batches, load_pav_metadata

    metadata = load_pav_metadata(directory)
    return batch_node_pav_tree(
        metadata.samples,
        metadata.node_count,
        lambda: iter_pav_batches(directory, unit="sample"),
        **kwargs,
    )
