"""Bayesian phylogenetic mixed model (PGLMM) for species-level comparative data.

The model, one row per species::

    y = X b + u + e,    u ~ N(0, s2_phylo * A),    e ~ N(0, s2_resid * I)

``A`` is the Brownian-motion covariance of the tree (shared root-to-MRCA path
length, scaled so the mean root-to-tip depth is 1, like ``MCMCglmm::inverseA``
with ``scale=TRUE``). ``family="binary"`` is the threshold (probit) model: a
latent liability follows the equation above with ``s2_resid`` fixed at 1 and
the observed state is ``liability > 0``.

Sampling is Gibbs throughout, in the eigenbasis of ``A`` where every
conditional is diagonal; the fixed effects are drawn with u integrated out (a
blocked step: the intercept and u are strongly correlated), and the
phylogenetic variance is parameter-expanded (``u = alpha * w``, as in
MCMCglmm) so the chain does not stick near zero. Fixed effects have a flat
prior (``alpha`` a N(0, 1000) working prior); variances an inverse-gamma prior
(``V``, ``nu``; default V=1, nu=0.002). Reported per fixed effect, as MCMCglmm
summaries do: posterior mean, 95% HPD interval, pMCMC (twice the smaller tail
probability of 0) and effective sample size; plus the phylogenetic
heritability ``H2 = s2_phylo / (s2_phylo + s2_resid)``.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.special import ndtr, ndtri

from organelleverse._public_validation import validated_public

from ..core.errors import OrganelleInputError
from ..core.result import OrganelleResult
from ..inputs import Tree
from ..sequences import as_path
from ._results import findings, ok_result, provenance
from .treecompare import parse_rooted_tree, read_tree

__all__ = ["fit_pglmm", "phylogenetic_mixed_model"]

_LOW_ESS = 200


def _read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    text = path.read_text(encoding="utf-8-sig")
    first = text.splitlines()[0] if text else ""
    delimiter = "\t" if "\t" in first else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    rows = [dict(row) for row in reader]
    return list(reader.fieldnames or []), rows


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text == "" or text.upper() in {"NA", "NAN", "NULL", "NONE"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _missing(value: str | None) -> bool:
    return value is None or value.strip().upper() in {"", "NA", "NAN", "NULL", "NONE"}


def _hpd(samples: np.ndarray, mass: float = 0.95) -> tuple[float, float]:
    """Shortest interval holding ``mass`` of the samples (coda::HPDinterval)."""
    ordered = np.sort(samples)
    n = len(ordered)
    width = max(1, int(math.floor(mass * n)))
    if width >= n:
        return float(ordered[0]), float(ordered[-1])
    spans = ordered[width:] - ordered[: n - width]
    start = int(np.argmin(spans))
    return float(ordered[start]), float(ordered[start + width])


def _ess(samples: np.ndarray) -> float:
    """Effective sample size from the initial positive autocorrelation sequence."""
    n = len(samples)
    if n < 4:
        return float(n)
    x = samples - samples.mean()
    variance = float(np.dot(x, x) / n)
    if variance == 0:
        return float(n)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, size)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n] / (n * variance)
    total = 0.0
    for lag in range(1, n - 1, 2):
        pair = acf[lag] + acf[lag + 1]
        if pair <= 0:
            break
        total += pair
    tau = -1.0 + 2.0 * (acf[0] + total) if total else 1.0
    return float(n / max(tau, 1.0))


def _summary(samples: np.ndarray, *, test_zero: bool) -> dict[str, Any]:
    low, high = _hpd(samples)
    summary: dict[str, Any] = {
        "posterior_mean": float(samples.mean()),
        "hpd95_low": low,
        "hpd95_high": high,
        "effective_sample_size": round(_ess(samples), 1),
    }
    if test_zero:
        above = float(np.mean(samples > 0))
        summary["pmcmc"] = float(min(1.0, 2.0 * min(above, 1.0 - above)))
    return summary


def _truncated_normal(
    mean: np.ndarray, positive: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw N(mean, 1) truncated to (0, inf) where ``positive`` else (-inf, 0]."""
    cut = ndtr(-mean)  # P(liability <= 0)
    uniform = rng.random(len(mean))
    probability = np.where(positive, cut + uniform * (1.0 - cut), uniform * cut)
    probability = np.clip(probability, 1e-15, 1.0 - 1e-15)
    return mean + ndtri(probability)


def fit_pglmm(
    y: np.ndarray,
    X: np.ndarray,
    A: np.ndarray,
    *,
    family: Literal["gaussian", "binary"] = "gaussian",
    iterations: int = 130_000,
    burnin: int = 30_000,
    thin: int = 100,
    prior_v: float = 1.0,
    prior_nu: float = 0.002,
    alpha_v: float = 1000.0,
    seed: int = 1,
) -> dict[str, np.ndarray]:
    """Gibbs-sample the PGLMM; returns posterior draws of ``beta``, ``s2_phylo``, ``s2_resid``.

    ``y`` is the response (0/1 for ``binary``), ``X`` the design matrix with
    its intercept column, ``A`` the phylogenetic covariance (positive definite).
    """
    n, p = X.shape
    if not 0 <= burnin < iterations or thin < 1:
        raise OrganelleInputError(
            code="phylogeny.pglmm.bad_mcmc", message="need 0 <= burnin < iterations and thin >= 1"
        )
    eigenvalues, Q = np.linalg.eigh(A)
    floor = 1e-8 * float(eigenvalues.max())
    eigenvalues = np.maximum(eigenvalues, floor)
    rng = np.random.default_rng(seed)
    Xt = Q.T @ X
    xtx_inv = np.linalg.inv(Xt.T @ Xt)
    binary = family == "binary"
    positive = y > 0.5
    liability = np.where(positive, 0.5, -0.5) if binary else y.astype(float)
    yt = Q.T @ liability
    beta = xtx_inv @ (Xt.T @ yt)
    w = np.zeros(n)
    alpha = 1.0
    s2_w = 1.0
    s2_e = 1.0 if binary else max(float(np.var(liability)) / 2.0, 1e-6)
    kept = (iterations - burnin) // thin
    draws_beta = np.empty((kept, p))
    draws_phylo = np.empty(kept)
    draws_resid = np.empty(kept)
    shape = (prior_nu + n) / 2.0
    k = 0
    for it in range(iterations):
        if binary:
            u = Q @ (alpha * w)
            liability = _truncated_normal(X @ beta + u, positive, rng)
            yt = Q.T @ liability
        # beta | variances, with u integrated out: the intercept and the
        # root-level part of u are strongly correlated, so drawing (beta, u)
        # as one block is what keeps the intercept chain mixing.
        marginal = 1.0 / (alpha * alpha * s2_w * eigenvalues + s2_e)
        weighted = Xt * marginal[:, None]
        b_cov = np.linalg.inv(Xt.T @ weighted)
        beta = b_cov @ (weighted.T @ yt) + np.linalg.cholesky(b_cov) @ rng.standard_normal(p)
        # w | beta, rest (eigenbasis, diagonal)
        resid = yt - Xt @ beta
        precision = 1.0 / (s2_w * eigenvalues) + alpha * alpha / s2_e
        variance = 1.0 / precision
        w = variance * alpha * resid / s2_e + np.sqrt(variance) * rng.standard_normal(n)
        # alpha | rest (working parameter of the expansion)
        ww = float(w @ w)
        a_precision = ww / s2_e + 1.0 / alpha_v
        alpha = (float(w @ resid) / s2_e) / a_precision + rng.standard_normal() / math.sqrt(
            a_precision
        )
        # s2_w | rest
        s2_w = (prior_nu * prior_v + float(np.sum(w * w / eigenvalues))) / 2.0 / rng.gamma(shape)
        if binary:
            # Scale-group move (Liu & Sabatti 2000): with the residual variance
            # fixed at 1 the liability scale is weakly identified, and the
            # liabilities, beta and u drift together far more slowly than any
            # one-block Gibbs step can follow. Rescaling all three by a common
            # c is an exact Gibbs step: c^2 ~ Gamma((n + p + 1) / 2, rate S / 2)
            # under the flat beta prior and the N(0, alpha_v) prior on alpha.
            r = yt - Xt @ beta - alpha * w
            scale_rate = (float(r @ r) + alpha * alpha / alpha_v) / 2.0
            c = math.sqrt(rng.gamma((n + p + 1) / 2.0) / scale_rate)
            yt = yt * c
            beta = beta * c
            alpha *= c
        v = alpha * w
        # s2_resid | rest
        if not binary:
            e = yt - Xt @ beta - v
            s2_e = (prior_nu * prior_v + float(e @ e)) / 2.0 / rng.gamma(shape)
        if it >= burnin and (it - burnin) % thin == 0 and k < kept:
            draws_beta[k] = beta
            draws_phylo[k] = alpha * alpha * s2_w
            draws_resid[k] = s2_e
            k += 1
    return {"beta": draws_beta[:k], "s2_phylo": draws_phylo[:k], "s2_resid": draws_resid[:k]}


@validated_public("phylogeny.phylogenetic_mixed_model")
def phylogenetic_mixed_model(
    tree: str | Path | Tree,
    table: str | Path,
    response: str,
    predictors: Sequence[str] = (),
    *,
    species_column: str = "species",
    family: Literal["gaussian", "binary"] = "gaussian",
    response_level: str | None = None,
    log_response: bool = False,
    standardize: Sequence[str] = (),
    iterations: int = 130_000,
    burnin: int = 30_000,
    thin: int = 100,
    prior_v: float = 1.0,
    prior_nu: float | None = None,
    prior_alpha_v: float | None = None,
    seed: int = 1,
) -> OrganelleResult:
    """Fit a Bayesian phylogenetic mixed model of a species trait on predictors.

    ``table`` is a CSV/TSV with one row per species (``species_column`` must
    match the tree's tip names); species missing from the tree or with a
    missing value are dropped and listed. Numeric predictors enter as is (or
    z-standardized when named in ``standardize``); text predictors become
    treatment-coded factors with the first level (sorted) as reference.

    ``family="gaussian"`` for continuous responses (``log_response`` takes the
    natural log first, e.g. genome length or node count). ``family="binary"``
    is the probit threshold model for a two-state trait; ``response_level``
    names the state coded 1 (default: the column is already 0/1). For three or
    more states, fit one binary contrast per state against the reference.

    Priors: the variances take an inverse-gamma ``(prior_v, prior_nu)`` prior
    and the expansion parameter a ``N(0, prior_alpha_v)`` working prior. The
    defaults follow MCMCglmm practice: ``nu=0.002, alpha_v=1000`` (nearly
    flat) for ``gaussian``; for ``binary``, whose residual variance is fixed,
    ``nu=1000, alpha_v=1``, a chi-square(1) prior on the phylogenetic variance
    that keeps it finite when the trait is close to separable.

    Returns posterior mean, 95% HPD interval, pMCMC and effective sample size
    for every fixed effect, the phylogenetic and residual variances, and the
    phylogenetic heritability H2. Flags ``low_effective_sample_size`` when any
    reported quantity has ESS < 200: run longer before reading the result.
    """
    tree = as_path(tree)
    table = as_path(table)
    predictors = list(predictors)
    if prior_nu is None:
        prior_nu = 1000.0 if family == "binary" else 0.002
    if prior_alpha_v is None:
        prior_alpha_v = 1.0 if family == "binary" else 1000.0
    rooted = parse_rooted_tree(read_tree(tree))
    if not rooted.has_lengths:
        raise OrganelleInputError(
            code="phylogeny.pglmm.no_branch_lengths",
            message="the tree needs branch lengths (e.g. a dated or ML tree)",
        )
    columns, rows = _read_table(Path(table))
    for name in (species_column, response, *predictors):
        if name not in columns:
            raise OrganelleInputError(
                code="phylogeny.pglmm.missing_column",
                message=f"column {name!r} is not in the table",
                details={"columns": columns},
            )
    unknown = [name for name in standardize if name not in predictors]
    if unknown:
        raise OrganelleInputError(
            code="phylogeny.pglmm.bad_standardize",
            message="standardize names columns that are not predictors",
            details={"unknown": unknown},
        )
    tips = set(rooted.leaf_names)
    seen: set[str] = set()
    not_in_tree: list[str] = []
    incomplete: list[str] = []
    kept_rows: list[dict[str, str]] = []
    for row in rows:
        species = (row.get(species_column) or "").strip()
        if species in seen:
            raise OrganelleInputError(
                code="phylogeny.pglmm.duplicate_species",
                message=f"species {species!r} appears twice; give one row per species",
            )
        seen.add(species)
        if species not in tips:
            not_in_tree.append(species)
        elif any(_missing(row.get(name)) for name in (response, *predictors)):
            incomplete.append(species)
        else:
            kept_rows.append(row)
    n = len(kept_rows)
    species_names = [row[species_column].strip() for row in kept_rows]

    raw_response = [row[response].strip() for row in kept_rows]
    if family == "binary":
        if response_level is not None:
            y = np.array([value == response_level for value in raw_response], dtype=float)
        else:
            values = [_number(value) for value in raw_response]
            if any(value not in (0.0, 1.0) for value in values):
                raise OrganelleInputError(
                    code="phylogeny.pglmm.binary_response",
                    message="binary response must be 0/1, or name the state coded 1 in response_level",
                    details={"levels": sorted(set(raw_response))[:10]},
                )
            y = np.array(values, dtype=float)
        if y.min() == y.max():
            raise OrganelleInputError(
                code="phylogeny.pglmm.binary_response", message="binary response has one state only"
            )
    else:
        values = [_number(value) for value in raw_response]
        if any(value is None for value in values):
            raise OrganelleInputError(
                code="phylogeny.pglmm.non_numeric_response",
                message=f"response {response!r} is not numeric",
            )
        y = np.array(values, dtype=float)
        if log_response:
            if (y <= 0).any():
                raise OrganelleInputError(
                    code="phylogeny.pglmm.log_nonpositive",
                    message="log_response needs a positive response",
                )
            y = np.log(y)

    design = [np.ones(n)]
    terms = ["(Intercept)"]
    factors: dict[str, dict[str, Any]] = {}
    for name in predictors:
        raw = [row[name].strip() for row in kept_rows]
        numeric = [_number(value) for value in raw]
        if all(value is not None for value in numeric):
            column = np.array(numeric, dtype=float)
            if name in standardize:
                sd = float(column.std(ddof=1)) if n > 1 else 0.0
                if sd == 0:
                    raise OrganelleInputError(
                        code="phylogeny.pglmm.constant_predictor",
                        message=f"predictor {name!r} is constant",
                    )
                column = (column - column.mean()) / sd
            design.append(column)
            terms.append(name)
        else:
            levels = sorted(set(raw))
            factors[name] = {"reference": levels[0], "levels": levels}
            for level in levels[1:]:
                design.append(np.array([value == level for value in raw], dtype=float))
                terms.append(f"{name}{level}")
    X = np.column_stack(design)
    if n < X.shape[1] + 3:
        raise OrganelleInputError(
            code="phylogeny.pglmm.too_few_species",
            message=f"{n} usable species for {X.shape[1]} fixed effects",
        )
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise OrganelleInputError(
            code="phylogeny.pglmm.collinear_predictors",
            message="predictors are collinear (the design matrix is rank deficient)",
        )

    A = rooted.shared_path_matrix(species_names)
    scale = float(np.mean(np.diag(A)))
    if scale <= 0:
        raise OrganelleInputError(
            code="phylogeny.pglmm.no_branch_lengths", message="the retained taxa have zero depth"
        )
    A = A / scale
    draws = fit_pglmm(
        y,
        X,
        A,
        family=family,
        iterations=iterations,
        burnin=burnin,
        thin=thin,
        prior_v=prior_v,
        prior_nu=prior_nu,
        alpha_v=prior_alpha_v,
        seed=seed,
    )
    fixed = [
        {"term": term, **_summary(draws["beta"][:, j], test_zero=True)}
        for j, term in enumerate(terms)
    ]
    heritability = draws["s2_phylo"] / (draws["s2_phylo"] + draws["s2_resid"])
    variance = {
        "phylogeny": _summary(draws["s2_phylo"], test_zero=False),
        "residual": None if family == "binary" else _summary(draws["s2_resid"], test_zero=False),
    }
    h2 = _summary(heritability, test_zero=False)
    ess = [row["effective_sample_size"] for row in fixed] + [
        variance["phylogeny"]["effective_sample_size"],
        h2["effective_sample_size"],
    ]
    flags = []
    if min(ess) < _LOW_ESS:
        flags.append("low_effective_sample_size")
    if not_in_tree:
        flags.append("species_not_in_tree")
    if incomplete:
        flags.append("rows_with_missing_values")
    significant = [row["term"] for row in fixed[1:] if row["pmcmc"] < 0.05]
    metrics = {
        "family": family,
        "response": response,
        "log_response": log_response,
        "n_species": n,
        "fixed_effects": fixed,
        "factors": factors,
        "standardized": list(standardize),
        "variance_components": variance,
        "phylogenetic_heritability": h2,
        "mcmc": {
            "iterations": iterations,
            "burnin": burnin,
            "thin": thin,
            "samples": int(len(heritability)),
            "seed": seed,
            "prior": {"V": prior_v, "nu": prior_nu, "alpha_v": prior_alpha_v, "fixed": "flat"},
        },
        "dropped_not_in_tree": not_in_tree,
        "dropped_missing_values": incomplete,
        "tree_depth_scale": scale,
    }
    return ok_result(
        "phylogenetic_mixed_model",
        status="warning" if "low_effective_sample_size" in flags else "ok",
        metrics=metrics,
        result_findings=findings(
            ("n_species", n),
            ("phylogenetic_heritability", round(h2["posterior_mean"], 4)),
            ("significant_terms", ", ".join(significant) or "none"),
        ),
        flags=tuple(flags),
        summary_text=(
            f"{family} PGLMM of {response} on {n} species: H2 = {h2['posterior_mean']:.2f} "
            f"(95% HPD {h2['hpd95_low']:.2f}-{h2['hpd95_high']:.2f}); "
            f"pMCMC < 0.05: {', '.join(significant) or 'none'}."
        ),
        result_provenance=provenance(
            "phylogenetic_mixed_model",
            method="ov_gibbs_pglmm",
            parameters={
                "tree": str(tree),
                "table": str(table),
                "response": response,
                "predictors": predictors,
                "species_column": species_column,
                "family": family,
                "response_level": response_level,
                "log_response": log_response,
                "standardize": list(standardize),
                "iterations": iterations,
                "burnin": burnin,
                "thin": thin,
                "prior_v": prior_v,
                "prior_nu": prior_nu,
                "prior_alpha_v": prior_alpha_v,
            },
            random_seed=seed,
        ),
    )
