"""ERC 2.0 engine — the nclark-lab method (Clark 2012; ERC 2.0 = Little 2025).

Composable typed cores (design 2026-06-30): every compute stage is a pure
**data-in → data-out** function (no ``OrganelleResult`` wrapper), so an LLM or
human can call and chain them directly — exactly like OmicVerse's
``ov.pp.*`` / ``ov.single.*`` pattern.

Typed cores:
    project_paths(gene_trees, species_tree) → {gene: [node_pair_path_or_None]}
    residual_matrix(path_matrix, *, transform) → {gene: [residual]}
    correlation_matrix(residuals, *, method, min_overlap) → (cor, count)
    fisher(cor, count) → {gene: {gene: {z, p_value, status}}}
    perm_test(gene_set, cor, *, n_perms) → dict

One public entry point (dual-readable ``OrganelleResult``):
    run_erc(gene_trees, species_tree, ...) → OrganelleResult

The entry point supplies tree context to the composable scientific stages.
"""

from __future__ import annotations

import math
import random
from itertools import combinations
from typing import Any
from collections.abc import Mapping, Sequence

from scipy import stats
import numpy as np

from organelleverse._public_validation import validated_public

from ..core import OrganelleResult
from .coevolution import (
    _failed,
    _finding,
    _is_missing,
    _ok,
    _provenance,
)
from ._erc_paths import project as _project, lowess as _lowess

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
# NaN-safe transform
# =========================================================================


def _nan_safe_transform(vec: list[float], method: str, log_offset: float | None = None) -> list[float]:
    """Transform branch lengths, preserving NaN (missing data).

    sqrt: sqrt(v) for nonnegative values; missing data stay missing.
    log: log(v + matrix-wide min_pos) for nonnegative values.
    none: identity.
    """
    if method not in {'sqrt', 'log', 'none'}:
        raise ValueError('transform must be sqrt, log, or none')
    if any(v < 0 for v in vec if not _is_missing(v)):
        raise ValueError('Evolutionary path lengths must be nonnegative')
    if method == "sqrt":
        return [math.sqrt(v) if not _is_missing(v) else v for v in vec]
    if method == "log":
        positives = [v for v in vec if not _is_missing(v) and v > 0]
        min_pos = log_offset if log_offset is not None else (min(positives) if positives else None)
        if min_pos is None:
            raise ValueError('Log transformation requires at least one positive path length')
        return [
            math.log(v + min_pos) if not _is_missing(v) and v >= 0 else float("nan") for v in vec
        ]
    return list(vec)


# =========================================================================
# Typed Core 1: project_paths
# =========================================================================


@validated_public('coevolution.project_paths')
def project_paths(
    gene_trees: dict[str, str],
    species_tree_newick: str,
) -> dict[str, list[float | None]]:
    """Map every ancestor-to-descendant node-pair path onto the master tree.

    Input contract: both trees must be rooted (root-to-node path lengths).
    The master tree's leaf names are the species set; every gene-tree leaf label
    (``species__gene`` / ``species_gene`` / ``species|gene``) is resolved to one
    of them by longest-name prefix, so two-word species names and congeners
    (``Arabidopsis_thaliana`` / ``Arabidopsis_lyrata``) stay distinct. Missing
    paths are ``None`` (JSON null). Gene trees must be single-copy and agree
    with the induced master topology and root; incompatible trees fail clearly.
    Paths follow node preorder, nearest ancestor first for each descendant.
    Unlike root-to-node distances, each value records both endpoints.
    """
    return _project(gene_trees, species_tree_newick)[0]


# =========================================================================
# Typed Core 2: residual_matrix
# =========================================================================


@validated_public('coevolution.residual_matrix')
def residual_matrix(
    path_matrix: Mapping[str, Sequence[float | None]],
    *,
    transform: str = "sqrt",
    trim_prop: float = 0.05,
    gene_trees: dict[str, str] | None = None,
    species_tree_newick: str | None = None,
    variance_weight: bool = False,
) -> dict[str, list[float | None]]:
    """Master-relative residuals (pure data function).

    Steps: transform → trimmed-mean master → OLS/WLS residuals.
    ``None`` marks missing data. With tree context, fit only independent
    gene-tree edges and predict all node-pair paths, as ERC 2.0 does.
    ``variance_weight=True`` estimates the LOWESS mean/variance trend used by
    the official n.pcs=0, impute=False workflow. Without tree context, each
    input column is treated as an independent observation.
    """
    genes = list(path_matrix)
    if transform not in {'sqrt', 'log', 'none'}:
        raise ValueError('transform must be sqrt, log, or none')
    if not genes:
        return {}
    if len({len(row) for row in path_matrix.values()}) != 1:
        raise ValueError('All path-matrix rows must have the same path layout and length')
    positives = [v for row in path_matrix.values() for v in row if v is not None and v > 0]
    log_offset = min(positives) if positives else None
    transformed = {g: _nan_safe_transform([float('nan') if v is None else v for v in path_matrix[g]], transform, log_offset) for g in genes}
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

    fit_indices = {g: list(range(n_edges)) for g in genes}
    if (gene_trees is None) != (species_tree_newick is None):
        raise ValueError('gene_trees and species_tree_newick must be provided together')
    if gene_trees is not None:
        _, paths, species = _project(gene_trees, species_tree_newick)
        if len(paths.paths) != n_edges or set(gene_trees) != set(genes):
            raise ValueError('Path matrix does not match the supplied gene/master tree layout')
        fit_indices = {g: paths.edge_indices(species[g]) for g in genes}
    data = np.asarray([transformed[g] for g in genes], dtype=float)
    master = np.asarray(master_vec, dtype=float)
    weights = np.ones_like(data)
    if variance_weight:
        # Official computeWeightsAllVar: unweighted predictions over all paths,
        # LOWESS(log(residual^2) ~ transformed path), then evaluate at predictions.
        prediction = np.full_like(data, np.nan)
        for i, row in enumerate(data):
            valid = np.isfinite(row) & np.isfinite(master)
            if valid.sum() > 2:
                design = np.column_stack((np.ones(valid.sum()), master[valid]))
                coefficients = np.linalg.lstsq(design, row[valid], rcond=None)[0]
                prediction[i, valid] = coefficients[0] + coefficients[1] * master[valid]
        squared = (data - prediction) ** 2
        finite = np.isfinite(data) & np.isfinite(squared) & (squared > 0)
        if finite.sum() < 2:
            raise ValueError('ERC variance trend is undefined: insufficient nonzero residual variation')
        xx, yy = _lowess(data[finite].tolist(), np.log(squared[finite]).tolist())
        weights = np.exp(-np.interp(prediction, xx, yy))
    finite_values = data[np.isfinite(data)]
    cutoff = float(np.quantile(finite_values, .05)) if finite_values.size else 0.0
    out: dict[str, list[float | None]] = {}
    for g in transformed:
        vec = transformed[g]
        n = min(len(vec), len(master_vec))
        valid = [i for i in fit_indices[g] if i < n and math.isfinite(master_vec[i]) and math.isfinite(vec[i]) and (not variance_weight or vec[i] >= cutoff)]
        res: list[float | None] = [None] * n
        if len(valid) > 2:
            design = np.column_stack((np.ones(len(valid)), master[valid]))
            w = weights[genes.index(g), valid]
            coefficients = np.linalg.lstsq(design * np.sqrt(w[:, None]), np.asarray(vec)[valid] * np.sqrt(w), rcond=None)[0]
            for i in range(n):
                if math.isfinite(vec[i]) and math.isfinite(master_vec[i]):
                    value = vec[i] - (coefficients[0] + coefficients[1] * master_vec[i])
                    # getRMat(..., rmatweights=weights) whitens final residuals.
                    res[i] = float(value * math.sqrt(weights[genes.index(g), i]))
        out[g] = res
    return out


# =========================================================================
# Typed Core 3: correlation_matrix
# =========================================================================


@validated_public('coevolution.correlation_matrix')
def correlation_matrix(
    residuals: Mapping[str, Sequence[float | None]],
    *,
    method: str = "pearson",
    min_overlap: int = 5,
    gene_trees: dict[str, str] | None = None,
    species_tree_newick: str | None = None,
    winsorize: int = 0,
) -> tuple[dict[str, dict[str, float | None]], dict[str, dict[str, int]]]:
    """All-by-all ERC correlation (pure data function).

    Returns (cor_matrix, count_matrix) where:
      cor_matrix[gene_a][gene_b] = r
      count_matrix[gene_a][gene_b] = n_overlap. Untestable pairs are ``None``.
    Supply both tree arguments to select independent edges of each pair's
    common-species tree. The full ERC 2.0 workflow uses winsorize=3, pooled
    over each presence cluster's residual matrix as official ERC.R does.
    """
    genes = list(residuals)
    if method not in {'pearson', 'spearman', 'kendall'}:
        raise ValueError('method must be pearson, spearman, or kendall')
    if min_overlap < 2 or winsorize < 0:
        raise ValueError('min_overlap must be >=2 and winsorize must be >=0')
    if len({len(row) for row in residuals.values()}) > 1:
        raise ValueError('All residual rows must have the same path layout and length')
    cor: dict[str, dict[str, float | None]] = {g: {} for g in genes}
    count: dict[str, dict[str, int]] = {g: {} for g in genes}
    if (gene_trees is None) != (species_tree_newick is None):
        raise ValueError('gene_trees and species_tree_newick must be provided together')
    paths = species = None
    if gene_trees is not None:
        _, paths, species = _project(gene_trees, species_tree_newick)
        if set(gene_trees) != set(genes) or any(len(row) != len(paths.paths) for row in residuals.values()):
            raise ValueError('Residual matrix does not match the supplied gene/master tree layout')
    clusters = {}
    if species is not None:
        for gene in genes:
            clusters.setdefault(frozenset(species[gene]), []).append(gene)
    for ga, gb in combinations(genes, 2):
        a, b = residuals[ga], residuals[gb]
        if paths is not None:
            indices = paths.edge_indices(species[ga] & species[gb])
            a, b = [a[i] for i in indices], [b[i] for i in indices]
        pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
        n_ov = len(pairs)
        r = None
        if n_ov >= max(2, min_overlap) and winsorize <= n_ov:
            x, y = [p[0] for p in pairs], [p[1] for p in pairs]
            if winsorize:
                def win(values, gene):
                    # ERC.R calls win(t(rri), w) on the whole presence-cluster
                    # matrix, not independently on each gene's column.
                    pool = values
                    if species is not None:
                        pool = [residuals[g][i] for g in clusters[frozenset(species[gene])]
                                for i in indices if residuals[g][i] is not None]
                    ordered = sorted(pool)
                    lo, hi = ordered[winsorize - 1], ordered[-winsorize]
                    return [max(lo, min(hi, v)) for v in values]
                x, y = win(x, ga), win(y, gb)
            r, _ = _erc2_correlate(x, y, method)
        cor[ga][gb] = r
        cor[gb][ga] = r
        count[ga][gb] = n_ov
        count[gb][ga] = n_ov
    return cor, count


def _erc2_correlate(a: Sequence[float], b: Sequence[float], method: str) -> tuple[float | None, int]:
    """Correlate over non-NaN overlap."""
    if len(a) != len(b):
        raise ValueError('Correlation vectors must have the same path layout and length')
    n = len(a)
    pairs = [(a[i], b[i]) for i in range(n) if not _is_missing(a[i]) and not _is_missing(b[i])]
    if len(pairs) < 2:
        return None, len(pairs)
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None, len(pairs)
    methods = {'pearson': stats.pearsonr, 'spearman': stats.spearmanr, 'kendall': stats.kendalltau}
    if method not in methods:
        raise ValueError('method must be pearson, spearman, or kendall')
    return float(methods[method](x, y).statistic), len(pairs)


# =========================================================================
# Typed Core 4: fisher
# =========================================================================


@validated_public('coevolution.fisher')
def fisher(
    cor: Mapping[str, Mapping[str, float | None]],
    count: dict[str, dict[str, int]],
) -> dict[str, dict[str, dict[str, float | str | None]]]:
    """Fisher estimates with finite JSON and explicit perfect/undefined states.

    Each pair has ``z``, ``p_value`` and ``status``. Perfect correlations have
    z=null, p_value=0 and status=perfect_positive/perfect_negative; z is never
    fabricated as zero. Insufficient or undefined pairs have p_value=null.
    """
    ft: dict[str, dict[str, dict[str, float | str | None]]] = {}
    for ga, row in cor.items():
        ft[ga] = {}
        for gb, r in row.items():
            n = count.get(ga, {}).get(gb, 0)
            if r is None:
                value = {'z': None, 'p_value': None, 'status': 'undefined_correlation'}
            elif abs(r) > 1:
                raise ValueError('correlation must be between -1 and 1')
            elif n <= 3:
                value = {'z': None, 'p_value': None, 'status': 'insufficient_overlap'}
            elif abs(r) == 1:
                value = {'z': None, 'p_value': 0.0, 'status': 'perfect_positive' if r > 0 else 'perfect_negative'}
            else:
                z = math.atanh(r) * math.sqrt(n - 3)
                value = {'z': z, 'p_value': float(2 * stats.norm.sf(abs(z))), 'status': 'finite'}
            ft[ga][gb] = value
    return ft


# =========================================================================
# Typed Core 5: perm_test
# =========================================================================


@validated_public('coevolution.perm_test')
def perm_test(
    gene_set: list[str],
    cor: Mapping[str, Mapping[str, float | None]],
    *,
    n_perms: int = 10000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Permutation test: is within-set mean ERC higher than random?

    Pure data function: returns {"observed", "p_value", "null_mean"}.
    """
    rng = random.Random(seed)
    if n_perms < 1:
        raise ValueError('n_perms must be >=1')
    all_genes = list(cor.keys())
    clean = list(dict.fromkeys(g for g in gene_set if g in cor))
    if len(clean) < 2:
        return {"observed": None, "p_value": None, "null_mean": None, 'status': 'insufficient_genes'}

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
        return {"observed": None, "p_value": None, "null_mean": None, 'status': 'no_testable_pairs'}

    null_values: list[float] = []
    for _ in range(n_perms):
        sampled = rng.sample(all_genes, len(clean))
        m = mean_erc(sampled)
        if not _is_missing(m):
            null_values.append(m)
    if not null_values:
        return {"observed": observed, "p_value": None, "null_mean": None, 'status': 'no_testable_permutations'}
    p_value = (1 + sum(1 for v in null_values if v >= observed)) / (1 + len(null_values))
    return {
        "observed": round(observed, 4),
        "p_value": round(p_value, 6),
        "null_mean": round(sum(null_values) / len(null_values), 4),
        'status': 'tested',
    }


# =========================================================================
# Entry point: run_erc (composes typed stages, no recompute)
# =========================================================================


@validated_public('coevolution.run_erc')
def run_erc(
    gene_trees: dict[str, str],
    species_tree_newick: str,
    *,
    transform: str = "sqrt",
    method: str = "pearson",
    min_overlap: int = 5,
    gene_sets: dict[str, list[str]] | None = None,
    n_perms: int = 10000,
    fdr_threshold: float = 0.05,
    seed: int = 0,
) -> OrganelleResult:
    """ERC 2.0 with node-pair paths, weighted residuals and independent edges.

    Gene and master trees must use consistent roots and induced topologies,
    with one gene copy per species. Taxa can be missing; absent node-pair
    paths are None. Discordant trees fail explicitly. For each pair, only
    independent branches of the common-species tree contribute observations.
    The implemented official parameter combination is n.pcs=0, impute=False,
    LOWESS variance weighting, WLS residuals, winsorization=3 and Pearson.
    Pairwise Fisher tests and optional gene-set tests receive separate BH
    corrections. Full R-package numerical parity is a distinct validation.
    """
    if not 0 < fdr_threshold <= 1:
        raise ValueError('fdr_threshold must be >0 and <=1')
    if len(gene_trees) < 2:
        return _failed(
            "run_erc",
            summary_text="Need >=2 gene trees.",
            anomalies=["too_few_genes"],
        )

    # The official ERC 2.0 workflow uses Pearson correlation of winsorized
    # variance-weighted residuals. Rank methods remain available as standalone
    # correlation_matrix operations; Fisher's Pearson null is not reused for them.
    if method != 'pearson':
        return _failed('run_erc', summary_text='ERC 2.0 uses Pearson correlation; use correlation_matrix for rank correlation.', anomalies=['unsupported_erc2_correlation'])
    try:
        path_mat, paths, _species = _project(gene_trees, species_tree_newick)
    except ValueError as exc:
        return _failed('run_erc', summary_text=str(exc), anomalies=['incompatible_erc2_trees'])

    # Stage 2: residual matrix
    try:
        resid = residual_matrix(path_mat, transform=transform, gene_trees=gene_trees,
                                species_tree_newick=species_tree_newick, variance_weight=True)
    except ValueError as exc:
        return _failed('run_erc', summary_text=str(exc), anomalies=['erc2_residual_fit_failed'])

    # Stage 3: correlation matrix
    cor, count = correlation_matrix(resid, method=method, min_overlap=min_overlap,
                                    gene_trees=gene_trees, species_tree_newick=species_tree_newick,
                                    winsorize=3)

    # Stage 4: Fisher z
    ft = fisher(cor, count)

    genes = list(cor)
    pairs = []
    for ga, gb in combinations(genes, 2):
        estimate = ft[ga][gb]
        if estimate['p_value'] is not None:
            pairs.append({'gene_a': ga, 'gene_b': gb, 'r': cor[ga][gb],
                          'n_branches': count[ga][gb], 'fisher_z': estimate['z'],
                          'fisher_status': estimate['status'], 'pval': estimate['p_value']})
    if pairs:
        adjusted = stats.false_discovery_control([p['pval'] for p in pairs], method='bh')
        for p, q in zip(pairs, adjusted, strict=True):
            p['fdr_pval'] = float(q)
            p['significant'] = bool(q < fdr_threshold)
    n_significant = sum(p['significant'] for p in pairs)

    # Stage 5: permutation tests (optional)
    perm_results: dict[str, Any] = {}
    if gene_sets:
        for set_name, gene_list in gene_sets.items():
            perm_results[set_name] = perm_test(gene_list, cor, n_perms=n_perms, seed=seed)
    tested_sets = [v for v in perm_results.values() if v['p_value'] is not None]
    if tested_sets:
        adjusted = stats.false_discovery_control([v['p_value'] for v in tested_sets], method='bh')
        for value, q in zip(tested_sets, adjusted, strict=True):
            value['fdr_p_value'] = float(q)

    n_genes = len(genes)
    n_edges = len(path_mat.get(genes[0], [])) if genes else 0

    return _ok(
        "run_erc",
        artifacts=(),
        metrics={
            "method": "erc2",
            "n_genes": n_genes,
            "n_edges": len(paths.edge_indices(set(paths.clades[paths.tree]))),
            "n_paths": n_edges,
            'pairs_tested': len(pairs),
            'pairs': pairs,
            "n_significant_pairs": n_significant,
            "transform": transform,
            "correlation_method": method,
            "n_gene_sets_tested": len(perm_results),
            'fdr_threshold': fdr_threshold,
            'multiple_testing': 'Benjamini-Hochberg',
            'path_model': 'ancestor-descendant; independent common-species edges per pair',
            'residual_model': 'LOWESS variance weights; WLS; n.pcs=0; impute=False',
            'winsorize': 3,
            'winsorization_scope': 'presence-cluster residual matrix on common-species edges',
            # The full permutation payload used to live only in key_findings;
            # canonical Findings hold one scalar each, so it is kept here too.
            "gene_set_permutation_tests": perm_results,
        },
        findings=(
            _finding("n_genes", value=n_genes),
            _finding("n_significant", value=n_significant),
            *[_finding("perm_test", metric=k, value=v["p_value"]) for k, v in perm_results.items() if v['p_value'] is not None],
        ),
        flags=("erc2_complete",) + (("covariation_detected",) if n_significant else ()),
        summary_text=(
            f"ERC 2.0: {n_genes} genes × {n_edges} node-pair paths; "
            f"{n_significant} significant pairs (Fisher test, BH FDR<{fdr_threshold})"
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
                'fdr_threshold': fdr_threshold,
                'seed': seed,
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
