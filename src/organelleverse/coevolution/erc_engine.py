"""ERC 2.0 engine — the nclark-lab method (Clark 2012; ERC 2.0 = Little 2025).

Composable typed cores (design 2026-06-30): every compute stage is a pure
**data-in → data-out** function (no ``OrganelleResult`` wrapper), so an LLM or
human can call and chain them directly — exactly like OmicVerse's
``ov.pp.*`` / ``ov.single.*`` pattern.

Typed cores:
    project_paths(gene_trees, species_tree) → {gene: [path_per_edge]}
    residual_matrix(path_matrix, *, transform) → {gene: [residual]}
    correlation_matrix(residuals, *, method, min_overlap) → (cor, count)
    fisher(cor, count) → {gene: {gene: z}}
    perm_test(gene_set, cor, *, n_perms) → dict

One public entry point (dual-readable ``OrganelleResult``):
    run_erc(gene_trees, species_tree, ...) → OrganelleResult

The entry point composes the typed stages (local variables, no recompute).
"""

from __future__ import annotations

import math
import random
from itertools import combinations
from typing import Any

from ..core import OrganelleResult
from .coevolution import (
    TreeNode,
    parse_newick,
    _failed,
    _finding,
    _is_missing,
    _ok,
    _provenance,
    _rank,
)

__all__ = [
    # Typed cores (data-in → data-out)
    "project_paths",
    "residual_matrix",
    "correlation_matrix",
    "fisher",
    "perm_test",
    # Entry point (OrganelleResult)
    "run_erc",
    # Back-compat aliases (old names, now thin wrappers)
    "erc_master_paths",
    "erc_residual_matrix",
    "erc_correlation_matrix",
    "fisher_transform",
    "perm_test_matrix",
]


# =========================================================================
# Tree helpers
# =========================================================================


def _path_lengths(tree: TreeNode) -> dict[int, float]:
    """Root-to-node path lengths for every node."""
    paths: dict[int, float] = {}
    _walk_paths(tree, 0.0, paths)
    return paths


def _walk_paths(node: TreeNode, cum: float, paths: dict[int, float]) -> None:
    dist = cum + node.branch_length
    paths[id(node)] = dist
    for child in node.children:
        _walk_paths(child, dist, paths)


def _edge_list(tree: TreeNode) -> list[tuple[int, int, float]]:
    """(parent_id, child_id, child_branch_length) for every edge."""
    edges: list[tuple[int, int, float]] = []
    for node in tree.traverse():
        for child in node.children:
            edges.append((id(node), id(child), child.branch_length))
    return edges


def _descendant_species_set(node: TreeNode) -> set[str]:
    """Species names under a node."""
    return {leaf.name for leaf in node.leaves() if leaf.name}


# =========================================================================
# NaN-safe transform
# =========================================================================


def _nan_safe_transform(vec: list[float], method: str) -> list[float]:
    """Transform branch lengths, preserving NaN (missing data).

    sqrt: sqrt(v) for v>=0, NaN for NaN.
    log:  log(v + min_pos) for v>=0, NaN for NaN.
    none: identity.
    """
    if method == "sqrt":
        return [
            math.sqrt(v)
            if not _is_missing(v) and v >= 0
            else (float("nan") if _is_missing(v) else 0.0)
            for v in vec
        ]
    if method == "log":
        positives = [v for v in vec if not _is_missing(v) and v > 0]
        min_pos = min(positives) if positives else 1e-6
        return [
            math.log(v + min_pos) if not _is_missing(v) and v >= 0 else float("nan") for v in vec
        ]
    return list(vec)


# =========================================================================
# Typed Core 1: project_paths
# =========================================================================


def _species_token(name: str) -> str:
    """One normalization for both master and gene-tree leaf names.

    Accepts ``species__gene``, ``species_gene`` and ``species|gene`` label
    styles. Applying this to only one side (as before) made the descendant
    set comparison fail for every edge on real two-word species names.
    """
    return name.split("__")[0].split("_")[0].split("|")[0]


def project_paths(
    gene_trees: dict[str, str],
    species_tree_newick: str,
) -> dict[str, list[float]]:
    """Project gene-tree branch lengths onto master-tree edge space (rooted trees).

    Input contract: both trees must be rooted (root-to-node path lengths),
    and leaf labels are tokenized identically on the two sides
    (``species__gene`` / ``species_gene`` / ``species|gene`` styles all
    collapse to the species token) - normalizing one side alone silently
    voids every edge, which is what happened on real two-word species
    names. Edges whose clade is absent from a gene tree (topology
    discordance) are NaN.
    """
    master = parse_newick(species_tree_newick)
    master_edges = _edge_list(master)

    # Precompute descendant species for each master edge's child node
    edge_species: list[set[str]] = []
    for pid, cid, bl in master_edges:
        for node in master.traverse():
            if id(node) == cid:
                edge_species.append(
                    {_species_token(leaf.name) for leaf in node.leaves() if leaf.name}
                )
                break
        else:
            edge_species.append(set())

    path_matrix: dict[str, list[float]] = {}
    for gene_name, newick in gene_trees.items():
        try:
            gtree = parse_newick(newick)
        except Exception:
            continue
        gpaths = _path_lengths(gtree)
        gene_leaf_species: dict[int, str] = {}
        for leaf in gtree.leaves():
            if leaf.name:
                gene_leaf_species[id(leaf)] = _species_token(leaf.name)

        row: list[float] = []
        for idx, (pid, cid, bl) in enumerate(master_edges):
            target = edge_species[idx]
            if not target:
                row.append(float("nan"))
                continue
            # find the gene-tree node whose descendant species matches target
            found = False
            for node in gtree.traverse():
                node_sp = {
                    gene_leaf_species.get(id(l), "")
                    for l in node.leaves()
                    if id(l) in gene_leaf_species
                }
                if node_sp == target:
                    row.append(gpaths.get(id(node), float("nan")))
                    found = True
                    break
            if not found:
                row.append(float("nan"))
        path_matrix[gene_name] = row

    return path_matrix


# =========================================================================
# Typed Core 2: residual_matrix
# =========================================================================


def residual_matrix(
    path_matrix: dict[str, list[float]],
    *,
    transform: str = "sqrt",
    trim_prop: float = 0.05,
) -> dict[str, list[float]]:
    """Master-relative residuals (pure data function).

    Steps: transform → trimmed-mean master → OLS residuals.
    NaN (missing) edges survive: excluded from fit, residual = NaN.
    """
    genes = list(path_matrix)
    if not genes:
        return {}
    transformed = {g: _nan_safe_transform(path_matrix[g], transform) for g in genes}
    n_edges = len(next(iter(transformed.values())))

    # master (trimmed mean per edge, NaN-aware)
    master_vec: list[float] = []
    for j in range(n_edges):
        col = sorted(
            transformed[g][j]
            for g in transformed
            if j < len(transformed[g]) and not _is_missing(transformed[g][j])
        )
        if not col:
            master_vec.append(float("nan"))
            continue
        n = len(col)
        trim = int(n * trim_prop)
        trimmed = col[trim : n - trim] if n > 2 * trim else col
        master_vec.append(sum(trimmed) / len(trimmed) if trimmed else float("nan"))

    # residuals
    out: dict[str, list[float]] = {}
    for g in transformed:
        vec = transformed[g]
        n = min(len(vec), len(master_vec))
        valid = [i for i in range(n) if not _is_missing(master_vec[i]) and not _is_missing(vec[i])]
        res = [float("nan")] * n
        if len(valid) >= 2:
            pairs = [(master_vec[i], vec[i]) for i in valid]
            mx = sum(p[0] for p in pairs) / len(pairs)
            my = sum(p[1] for p in pairs) / len(pairs)
            num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
            den = sum((p[0] - mx) ** 2 for p in pairs)
            slope = num / den if den else 0.0
            intercept = my - slope * mx
            for i in valid:
                res[i] = vec[i] - (slope * master_vec[i] + intercept)
        out[g] = res
    return out


# =========================================================================
# Typed Core 3: correlation_matrix
# =========================================================================


def correlation_matrix(
    residuals: dict[str, list[float]],
    *,
    method: str = "pearson",
    min_overlap: int = 5,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    """All-by-all ERC correlation (pure data function).

    Returns (cor_matrix, count_matrix) where:
      cor_matrix[gene_a][gene_b] = r
      count_matrix[gene_a][gene_b] = n_overlap (non-NaN edges)
    """
    genes = list(residuals)
    cor: dict[str, dict[str, float]] = {g: {} for g in genes}
    count: dict[str, dict[str, int]] = {g: {} for g in genes}
    for ga, gb in combinations(genes, 2):
        r, n_ov = _erc2_correlate(residuals[ga], residuals[gb], method)
        cor[ga][gb] = r
        cor[gb][ga] = r
        count[ga][gb] = n_ov
        count[gb][ga] = n_ov
    return cor, count


def _erc2_correlate(a: list[float], b: list[float], method: str) -> tuple[float, int]:
    """Correlate over non-NaN overlap."""
    n = min(len(a), len(b))
    pairs = [(a[i], b[i]) for i in range(n) if not _is_missing(a[i]) and not _is_missing(b[i])]
    if len(pairs) < 2:
        return 0.0, len(pairs)
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    if method == "spearman":
        x = _rank(x)
        y = _rank(y)
    elif method == "kendall":
        return _kendall_tau_simple(x, y), len(pairs)
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    da = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    db = math.sqrt(sum((yi - my) ** 2 for yi in y))
    return (num / (da * db) if da and db else 0.0), len(pairs)


def _kendall_tau_simple(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    conc = disc = tie_x = tie_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 or dy == 0:
                if dx == 0:
                    tie_x += 1
                if dy == 0:
                    tie_y += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) / 2
    v0 = n0 - tie_x
    v1 = n0 - tie_y
    denom = math.sqrt(v0 * v1) if v0 > 0 and v1 > 0 else 0.0
    return (conc - disc) / denom if denom else 0.0


# =========================================================================
# Typed Core 4: fisher
# =========================================================================


def fisher(
    cor: dict[str, dict[str, float]],
    count: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    """Fisher z-transform: z = atanh(r) · √(n − 3). Pure data function."""
    ft: dict[str, dict[str, float]] = {}
    for ga, row in cor.items():
        ft[ga] = {}
        for gb, r in row.items():
            n = count.get(ga, {}).get(gb, 0)
            ft[ga][gb] = math.atanh(r) * math.sqrt(n - 3) if n > 3 and abs(r) < 1.0 else 0.0
    return ft


# =========================================================================
# Typed Core 5: perm_test
# =========================================================================


def perm_test(
    gene_set: list[str],
    cor: dict[str, dict[str, float]],
    *,
    n_perms: int = 10000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Permutation test: is within-set mean ERC higher than random?

    Pure data function: returns {"observed", "p_value", "null_mean"}.
    """
    if seed is not None:
        random.seed(seed)
    all_genes = list(cor.keys())
    clean = [g for g in gene_set if g in cor]
    if len(clean) < 2:
        return {"observed": float("nan"), "p_value": 1.0, "null_mean": 0.0}

    def mean_erc(genes: list[str]) -> float:
        total = 0.0
        count = 0
        for i, ga in enumerate(genes):
            for gb in genes[i + 1 :]:
                r = cor.get(ga, {}).get(gb)
                if r is not None and not _is_missing(r):
                    total += r
                    count += 1
        return total / count if count else float("nan")

    observed = mean_erc(clean)
    if _is_missing(observed):
        return {"observed": float("nan"), "p_value": 1.0, "null_mean": 0.0}

    null_values: list[float] = []
    for _ in range(n_perms):
        sampled = random.sample(all_genes, len(clean))
        m = mean_erc(sampled)
        if not _is_missing(m):
            null_values.append(m)
    if not null_values:
        return {"observed": observed, "p_value": 1.0, "null_mean": 0.0}
    p_value = sum(1 for v in null_values if v >= observed) / len(null_values)
    return {
        "observed": round(observed, 4),
        "p_value": round(p_value, 6),
        "null_mean": round(sum(null_values) / len(null_values), 4),
    }


# =========================================================================
# Entry point: run_erc (composes typed stages, no recompute)
# =========================================================================


def run_erc(
    gene_trees: dict[str, str],
    species_tree_newick: str,
    *,
    transform: str = "sqrt",
    method: str = "pearson",
    min_overlap: int = 5,
    gene_sets: dict[str, list[str]] | None = None,
    n_perms: int = 10000,
) -> OrganelleResult:
    """Full ERC 2.0 pipeline: project → residual → correlate → Fisher → permute (rooted trees in).

    Inputs are rooted Newick trees per gene plus a rooted master tree;
    topology-discordant edges stay NaN and correlations use the surviving
    overlap. Comparing against the official nclark-lab R package demands
    extra input discipline on THEIR side: tab-separated name/newick pairs,
    one uniform species set, no polytomies, and every gene split present in
    the master (otherwise the official voids the whole gene row) - refit
    branch lengths on a fixed topology (e.g. iqtree ``-te``) for
    cross-engine parity.

    Composes the typed cores (no inline recompute). Returns dual-readable
    ``OrganelleResult`` for the whole-pipeline convenience call.
    """
    if len(gene_trees) < 2:
        return _failed(
            "run_erc",
            summary_text="Need >=2 gene trees.",
            anomalies=["too_few_genes"],
        )

    # Stage 1: project paths
    path_mat = project_paths(gene_trees, species_tree_newick)

    # Stage 2: residual matrix
    resid = residual_matrix(path_mat, transform=transform)

    # Stage 3: correlation matrix
    cor, count = correlation_matrix(resid, method=method, min_overlap=min_overlap)

    # Stage 4: Fisher z
    ft = fisher(cor, count)

    # Significance via Fisher z (|z| > 1.96 ≈ p < 0.05 two-tailed)
    z_threshold = 1.96
    genes = list(cor)
    n_significant = 0
    for ga in genes:
        for gb in genes:
            if ga < gb and count.get(ga, {}).get(gb, 0) >= min_overlap:
                z = ft.get(ga, {}).get(gb, 0.0)
                if abs(z) > z_threshold:
                    n_significant += 1

    # Stage 5: permutation tests (optional)
    perm_results: dict[str, Any] = {}
    if gene_sets:
        for set_name, gene_list in gene_sets.items():
            perm_results[set_name] = perm_test(gene_list, cor, n_perms=n_perms)

    n_genes = len(genes)
    n_edges = len(path_mat.get(genes[0], [])) if genes else 0

    return _ok(
        "run_erc",
        artifacts=(),
        metrics={
            "method": "erc2",
            "n_genes": n_genes,
            "n_edges": n_edges,
            "n_significant_pairs": n_significant,
            "transform": transform,
            "correlation_method": method,
            "n_gene_sets_tested": len(perm_results),
            "z_threshold": z_threshold,
            # The full permutation payload used to live only in key_findings;
            # canonical Findings hold one scalar each, so it is kept here too.
            "gene_set_permutation_tests": perm_results,
        },
        findings=(
            _finding("n_genes", value=n_genes),
            _finding("n_significant", value=n_significant),
            *[_finding("perm_test", metric=k, value=v["p_value"]) for k, v in perm_results.items()],
        ),
        flags=("erc2_complete",) + (("covariation_detected",) if n_significant else ()),
        summary_text=(
            f"ERC 2.0: {n_genes} genes × {n_edges} edges; "
            f"{n_significant} significant pairs (Fisher |z|>{z_threshold})"
            + (f"; {len(perm_results)} gene set(s) tested." if perm_results else ".")
        ),
        provenance=_provenance(
            "run_erc",
            method="erc2",
            parameters={
                "transform": transform,
                "method": method,
                "min_overlap": min_overlap,
                "n_perms": n_perms,
                "gene_sets": sorted(gene_sets) if gene_sets else [],
            },
        ),
    )


# =========================================================================
# Back-compat aliases (old names → typed cores)
# =========================================================================


def erc_master_paths(gene_trees, species_tree_newick, **kw):
    """Back-compat: project_paths (typed data, no OrganelleResult)."""
    return project_paths(gene_trees, species_tree_newick)


def erc_residual_matrix(path_matrix, **kw):
    """Back-compat: residual_matrix (typed data)."""
    return residual_matrix(path_matrix, **kw)


def erc_correlation_matrix(residuals, **kw):
    """Back-compat: correlation_matrix (typed data)."""
    cor, count = correlation_matrix(residuals, **kw)
    return cor, count  # callers that expect tuple get it


def fisher_transform(cor, count):
    """Back-compat alias for fisher()."""
    return fisher(cor, count)


def perm_test_matrix(gene_set, cor, **kw):
    """Back-compat alias for perm_test()."""
    return perm_test(gene_set, cor, **kw)
