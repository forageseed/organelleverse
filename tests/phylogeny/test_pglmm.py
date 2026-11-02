"""Bayesian PGLMM: recovers simulated effects and heritability; validates its inputs."""

from __future__ import annotations

import numpy as np
import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.phylogeny.pglmm import phylogenetic_mixed_model
from organelleverse.phylogeny.treecompare import parse_rooted_tree


def _random_tree(n: int, rng: np.random.Generator) -> str:
    """Ultrametric-ish random tree by repeatedly joining random pairs (coalescent order)."""
    nodes = [(f"sp{i}", 0.0) for i in range(n)]  # (newick, height)
    height = 0.0
    while len(nodes) > 1:
        height += rng.exponential(1.0 / len(nodes))
        i, j = sorted(rng.choice(len(nodes), 2, replace=False))
        (a, ha), (b, hb) = nodes[i], nodes[j]
        joined = (f"({a}:{height - ha:.6f},{b}:{height - hb:.6f})", height)
        nodes = [node for k, node in enumerate(nodes) if k not in (i, j)] + [joined]
    return nodes[0][0] + ";"


def _simulate(
    tmp_path, *, n: int, binary: bool, seed: int, slope: float | None = None, s2_phylo: float | None = None
):
    rng = np.random.default_rng(seed)
    newick = _random_tree(n, rng)
    taxa = [f"sp{i}" for i in range(n)]
    A = parse_rooted_tree(newick).shared_path_matrix(taxa)
    A /= np.mean(np.diag(A))
    s2_resid = 1.0 if binary else 0.5
    if s2_phylo is None:
        s2_phylo = 0.5 if binary else 1.0
    if slope is None:
        slope = 1.0 if binary else 0.8
    u = np.linalg.cholesky(A + 1e-10 * np.eye(n)) @ rng.standard_normal(n) * np.sqrt(s2_phylo)
    x = rng.standard_normal(n) * 3 + 10
    noise = rng.standard_normal(n)  # an unrelated predictor
    z = (x - x.mean()) / x.std(ddof=1)
    liability = 1.0 + slope * z + u + rng.standard_normal(n) * np.sqrt(s2_resid)
    y = (liability > 0).astype(int) if binary else liability
    tree = tmp_path / "tree.nwk"
    tree.write_text(newick)
    table = tmp_path / "traits.tsv"
    lines = ["species\ty\tx\tnoise\tclade"]
    for k in range(n):
        lines.append(f"sp{k}\t{y[k]}\t{x[k]:.6f}\t{noise[k]:.6f}\t{'ab'[k % 2]}")
    lines.append("not_in_tree\t1\t1\t1\ta")
    table.write_text("\n".join(lines) + "\n")
    return tree, table, A, y, z, s2_phylo, s2_resid


def _term(result, name):
    return next(row for row in result.metrics["fixed_effects"] if row["term"] == name)


def test_gaussian_recovers_slope_and_heritability(tmp_path) -> None:
    tree, table, A, y, z, s2_phylo, s2_resid = _simulate(tmp_path, n=120, binary=False, seed=3)
    result = phylogenetic_mixed_model(
        tree, table, "y", ["x", "noise"], standardize=["x"],
        iterations=12_000, burnin=2_000, thin=10, seed=7,
    )
    assert result.status == "ok", result.flags
    assert result.metrics["n_species"] == 120
    assert list(result.metrics["dropped_not_in_tree"]) == ["not_in_tree"]
    slope = _term(result, "x")
    assert slope["hpd95_low"] < 0.8 < slope["hpd95_high"]
    assert slope["pmcmc"] < 0.01
    # Both slopes sit on the GLS estimate under the true covariance (the
    # "noise" column can correlate with y by chance in one sample, so it is
    # checked against GLS, not against zero).
    rows = [line.split("\t") for line in table.read_text().splitlines()[1:121]]
    noise_column = np.array([float(row[3]) for row in rows])
    V = s2_phylo * A + s2_resid * np.eye(len(y))
    X = np.column_stack([np.ones(len(y)), z, noise_column])
    Vi = np.linalg.inv(V)
    gls = np.linalg.solve(X.T @ Vi @ X, X.T @ Vi @ y)
    assert slope["posterior_mean"] == pytest.approx(gls[1], abs=0.12)
    assert _term(result, "noise")["posterior_mean"] == pytest.approx(gls[2], abs=0.12)
    h2 = result.metrics["phylogenetic_heritability"]
    assert h2["hpd95_low"] < s2_phylo / (s2_phylo + s2_resid) < h2["hpd95_high"]
    assert h2["posterior_mean"] > 0.3


def test_factor_predictor_is_treatment_coded(tmp_path) -> None:
    tree, table, *_ = _simulate(tmp_path, n=40, binary=False, seed=5)
    result = phylogenetic_mixed_model(
        tree, table, "y", ["clade"], iterations=3_000, burnin=500, thin=5, seed=1
    )
    assert [row["term"] for row in result.metrics["fixed_effects"]] == ["(Intercept)", "cladeb"]
    assert result.metrics["factors"]["clade"]["reference"] == "a"


def test_binary_probit_recovers_sign(tmp_path) -> None:
    tree, table, *_ = _simulate(tmp_path, n=200, binary=True, seed=11)
    result = phylogenetic_mixed_model(
        tree, table, "y", ["x"], family="binary", standardize=["x"],
        iterations=12_000, burnin=2_000, thin=10, seed=2,
    )
    slope = _term(result, "x")
    assert slope["pmcmc"] < 0.05
    assert 0.4 < slope["posterior_mean"] < 2.5
    assert result.metrics["variance_components"]["residual"] is None


def test_binary_mixes_when_nearly_separable(tmp_path) -> None:
    # A strong predictor plus a large phylogenetic variance leaves few
    # "wrong-side" species: the liability scale is then weakly identified and,
    # before the scale-group move and the chi-square default prior, the chain
    # crawled (ESS < 20 on the 192-species DToL complexity model).
    tree, table, *_ = _simulate(tmp_path, n=150, binary=True, seed=4, slope=3.0, s2_phylo=3.0)
    result = phylogenetic_mixed_model(
        tree, table, "y", ["x"], family="binary", standardize=["x"],
        iterations=20_000, burnin=2_000, thin=10, seed=3,
    )
    assert result.metrics["mcmc"]["prior"]["nu"] == 1000.0
    assert result.metrics["mcmc"]["prior"]["alpha_v"] == 1.0
    slope = _term(result, "x")
    assert slope["effective_sample_size"] > 300
    assert result.metrics["phylogenetic_heritability"]["effective_sample_size"] > 300
    assert slope["pmcmc"] < 0.01 and slope["posterior_mean"] > 1.0


def test_input_errors(tmp_path) -> None:
    tree, table, *_ = _simulate(tmp_path, n=20, binary=False, seed=1)
    with pytest.raises(OrganelleInputError):
        phylogenetic_mixed_model(tree, table, "missing_column", ["x"])
    text = table.read_text()
    table.write_text(text + text.splitlines()[1] + "\n")
    with pytest.raises(OrganelleInputError, match="twice"):
        phylogenetic_mixed_model(tree, table, "y", ["x"], iterations=100, burnin=10)
    bare = tmp_path / "bare.nwk"
    bare.write_text("((sp0,sp1),(sp2,sp3));")
    with pytest.raises(OrganelleInputError):
        phylogenetic_mixed_model(bare, table, "y", ["x"])
