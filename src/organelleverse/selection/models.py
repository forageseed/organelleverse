"""PAML codeml selection-pressure models — control file generation + execution.

Four model classes (the core of molecular evolution analysis):

1. Branch models    — detect lineage-specific ω (branch_model)
2. Site models      — detect positive-selection sites (site_model: M0/M1a/M2a/M7/M8)
3. Branch-site models — detect lineage-specific positive-selection sites
                       (branch_site_model: Test 1 null vs Test 2 alt + BEB)
4. Clade models     — detect divergent selection between clades (clade_model: CmC)

Each function:
  - Generates the codeml control file (.ctl) with correct parameters
  - Optionally runs codeml via subprocess (if installed)
  - Parses the output (lnL, ω estimates, BEB sites)
  - Computes likelihood ratio tests (LRT) between null/alternative

Self-contained .ctl generation + output parsing in pure Python; codeml
execution itself uses the standard PAML codeml binary as a subprocess.
"""

from __future__ import annotations

import re
import math
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from organelleverse._public_validation import validated_public
from scipy import stats as _scipy_stats

from ..core.result import OrganelleResult
from ..core.errors import OrganelleInputError
from . import _contract

__all__ = [
    "branch_model",
    "branch_site_model",
    "clade_model",
    "likelihood_ratio_test",
    "parse_codeml_output",
    "run_codeml",
    "site_model",
    "write_branch_model",
    "write_branch_site_model",
    "write_clade_model",
    "write_ctl_file",
    "write_site_model",
]


# =========================================================================
# Control-file writer (shared)
# =========================================================================


def write_ctl_file(
    path: str | Path,
    *,
    seqfile: str,
    treefile: str,
    outfile: str,
    model: int,  # 0=one-ratio, 2=branch, 1/3=site/branch-site
    NSsites: str,  # site model class (e.g. "0", "1 2 7 8", "2")
    fix_omega: int = 0,
    omega: float = 1.0,
    fix_kappa: int = 0,
    kappa: float = 2.0,
    codon_freq: int = 2,  # F3x4
    cleandata: int = 0,
    method: int = 0,  # 0=relative, 1=one branch
    Small_Diff: float = 0.5e-6,
) -> Path:
    """Write a codeml control file (.ctl) with the given parameters.

    Returns the path to the written file.
    """
    lines = [
        f"      seqfile  = {seqfile}",
        f"      treefile = {treefile}",
        f"      outfile  = {outfile}",
        "      noisy = 9",
        "      verbose = 1",
        "      runmode = 0",
        "      seqtype = 1",
        f"      CodonFreq = {codon_freq}",
        "      clock = 0",
        "      aaDist = 0",
        f"      model = {model}",
        f"      NSsites = {NSsites}",
        "      icode = 0",
        f"      fix_kappa = {fix_kappa}",
        f"      kappa = {kappa}",
        f"      fix_omega = {fix_omega}",
        f"      omega = {omega}",
        "      fix_alpha = 1",
        "      alpha = 0.",
        "      Malpha = 0",
        "      ncatG = 4",
        "      getSE = 0",
        "      RateAncestor = 0",
        f"      Small_Diff = {Small_Diff}",
        f"      cleandata = {cleandata}",
        f"      method = {method}",
    ]
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path


# =========================================================================
# 1. BRANCH MODELS (lineage-specific ω)
#    Model A (one-ratio, M0) vs Model B (free-ratio / two-ratio)
#    Tests if foreground branches have different ω from background.
# =========================================================================


@validated_public('selection.branch_model')
def branch_model(
    *,
    alignment: str | Path,
    tree: str | Path,
    foreground_labels: list[str] | None = None,
    run: bool = True,
    seqfile: str = "codon_aln.paml",
    treefile: str = "tree.nwk",
) -> OrganelleResult:
    """Plan branch-model codeml inputs without writing user-facing files."""
    with tempfile.TemporaryDirectory() as tmp:
        result = write_branch_model(
            alignment=alignment,
            tree=tree,
            output_dir=tmp,
            foreground_labels=foreground_labels,
            run=run,
            seqfile=seqfile,
            treefile=treefile,
        )
    if result.status != "ok":
        return result
    return result.model_copy(
        update={"artifacts": (), "flags": (*result.flags, "output_dir_unbound")}
    )


def write_branch_model(
    *,
    alignment: str | Path,
    tree: str | Path,
    output_dir: str | Path,
    foreground_labels: list[str] | None = None,
    run: bool = True,
    seqfile: str = "codon_aln.paml",
    treefile: str = "tree.nwk",
) -> OrganelleResult:
    """Branch model: test lineage-specific selection pressure.

    Generates two .ctl files:
      - null:   model=0 (one-ratio, all branches share ω)
      - alt:    model=2 (two-ratio, foreground #1 vs background)

    Runs both if codeml is available; computes LRT (df=1) for omega
    heterogeneity. Foreground omega > 1 is descriptive, not an independent
    positive-selection test; use a branch-site test or an appropriate null
    that constrains the foreground omega when testing positive selection.

    Parameters
    ----------
    foreground_labels : list[str], optional
        Species names to mark as foreground (with #1 in tree). If None,
        uses the tree as-is.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # prepare tree with foreground marks
    _prepare_tree(tree, foreground_labels, out / treefile)
    seq_path = Path(alignment)
    _ensure_paml_format(seq_path, out / seqfile)

    branch_types = _branch_types(out / treefile)
    if branch_types != {0, 1}:
        raise OrganelleInputError(code="selection.branch_model_labels",
                                  message="two-ratio branch model requires background #0 and foreground #1 branches")

    # null model: one ratio (model=0)
    null_ctl = write_ctl_file(
        out / "branch_null.ctl",
        seqfile=seqfile,
        treefile=treefile,
        outfile="branch_null.out",
        model=0,
        NSsites="0",
    )
    # alt model: two-ratio (model=2)
    alt_ctl = write_ctl_file(
        out / "branch_alt.ctl",
        seqfile=seqfile,
        treefile=treefile,
        outfile="branch_alt.out",
        model=2,
        NSsites="0",
    )

    null_result = run_codeml(null_ctl, out) if run else None
    alt_result = run_codeml(alt_ctl, out) if run else None

    metrics: dict[str, Any] = {"model_type": "branch"}
    if run and not (_parsed_likelihood(null_result) and _parsed_likelihood(alt_result)):
        return _codeml_failed("branch_model", {"run": run})
    if null_result and alt_result:
        lrt = likelihood_ratio_test(null_result["lnL"], alt_result["lnL"], df=1)
        metrics.update(
            {
                "lnL_null": null_result["lnL"],
                "lnL_alt": alt_result["lnL"],
                "execution": {"null": null_result.get("execution"), "alternative": alt_result.get("execution")},
                "lrt_stat": round(lrt["lrt_stat"], 4),
                "p_value": round(lrt["p_value"], 6),
                "omega_foreground": alt_result.get("omega_foreground"),
                "omega_background": alt_result.get("omega_background"),
                "significant": lrt["p_value"] < 0.05,
                "test": "M0_vs_two_ratio_heterogeneity",
                "foreground_omega_above_one": (alt_result.get("omega_foreground") is not None
                                                and alt_result["omega_foreground"] > 1),
                "inference_note": "This LRT tests branch omega heterogeneity; foreground omega above one is descriptive. Use branch-site or an appropriate foreground-constrained null to test positive selection.",
            }
        )

    return _contract.ok(
        "branch_model",
        artifacts=_contract.artifacts((null_ctl, alt_ctl)),
        metrics=metrics,
        findings=_contract.findings("branch_model", _branch_key_findings(metrics)),
        flags=("branch_rate_heterogeneity",) if metrics.get("significant") else (),
        summary_text=_branch_summary(metrics),
        method="codeml_branch",
        parameters={
            "foreground_labels": list(foreground_labels or ()),
            "run": run,
            "seqfile": seqfile,
            "treefile": treefile,
        },
    )


# =========================================================================
# 2. SITE MODELS (positive-selection sites)
#    M0 (one-ratio) / M1a (nearly-neutral) / M2a (positive selection) /
#    M7 (beta) / M8 (beta + ω>1)
# =========================================================================


@validated_public('selection.site_model')
def site_model(
    *,
    alignment: str | Path,
    tree: str | Path,
    models: str = "0 1 2 7 8",
    run: bool = True,
    seqfile: str = "codon_aln.paml",
    treefile: str = "tree.nwk",
) -> OrganelleResult:
    """Plan site-model codeml inputs without writing user-facing files."""
    with tempfile.TemporaryDirectory() as tmp:
        result = write_site_model(
            alignment=alignment,
            tree=tree,
            output_dir=tmp,
            models=models,
            run=run,
            seqfile=seqfile,
            treefile=treefile,
        )
    if result.status != "ok":
        return result
    return result.model_copy(
        update={"artifacts": (), "flags": (*result.flags, "output_dir_unbound")}
    )


def write_site_model(
    *,
    alignment: str | Path,
    tree: str | Path,
    output_dir: str | Path,
    models: str = "0 1 2 7 8",
    run: bool = True,
    seqfile: str = "codon_aln.paml",
    treefile: str = "tree.nwk",
) -> OrganelleResult:
    """Site model: detect positive-selection sites.

    Runs the specified NSsites models. Key LRT comparisons:
      - M0 vs M3 (test for site variation)
      - M1a vs M2a (test for positive selection, df=2)
      - M7 vs M8  (test for positive selection, df=2)

    BEB sites with Pr(ω>1) > 0.95 are reported only after the corresponding LRT is significant.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _prepare_tree(tree, None, out / treefile)
    _ensure_paml_format(Path(alignment), out / seqfile)

    ctl = write_ctl_file(
        out / "site_model.ctl",
        seqfile=seqfile,
        treefile=treefile,
        outfile="site_model.out",
        model=0,
        NSsites=models,
    )

    result = run_codeml(ctl, out) if run else None
    if run and not result:
        return _contract.failed(
            "site_model",
            summary_text=(
                "codeml produced no parseable output (binary missing, failed, "
                "or exceeded the run timeout). The .ctl is written; rerun it "
                "manually or pass a longer timeout to run_codeml."
            ),
            anomalies=["codeml_no_output"],
            parameters={"models": models, "run": run,
                        "seqfile": seqfile, "treefile": treefile},
        )

    metrics: dict[str, Any] = {"model_type": "site", "models": models}
    if result:
        per_model = result.get("model_results")
        if per_model is None:
            requested = models.split()
            per_model = {requested[0]: result} if len(requested) == 1 else {}
        if (set(per_model) != set(models.split())
                or any(not _parsed_likelihood(value) for value in per_model.values())):
            return _codeml_failed("site_model", {"models": models, "run": run})
        tests = {}
        positive = []
        for null_id, alt_id in (("1", "2"), ("7", "8")):
            if null_id in per_model and alt_id in per_model:
                test = likelihood_ratio_test(per_model[null_id]["lnL"], per_model[alt_id]["lnL"], 2)
                test["significant"] = test["p_value"] < 0.05
                tests[f"M{null_id}_vs_M{alt_id}"] = test
                if test["significant"]:
                    positive.extend({**site, "model": alt_id}
                                    for site in per_model[alt_id].get("beb_sites", [])
                                    if site["prob"] > 0.95)
        metrics.update(model_results=per_model, likelihood_ratio_tests=tests,
                       execution=result.get("execution"),
                       positive_sites=positive, n_positive_sites=len({s["site"] for s in positive}),
                       n_positive_site_model_calls=len(positive))

    return _contract.ok(
        "site_model",
        artifacts=_contract.artifacts((ctl,)),
        metrics=metrics,
        findings=_contract.findings("site_model", _site_key_findings(metrics)),
        flags=("positive_selection_sites",) if metrics.get("n_positive_sites", 0) > 0 else (),
        summary_text=_site_summary(metrics),
        method="codeml_site",
        parameters={
            "models": models,
            "run": run,
            "seqfile": seqfile,
            "treefile": treefile,
        },
    )


# =========================================================================
# 3. BRANCH-SITE MODELS (lineage-specific positive-selection sites)
#    Test 1 (null: ω fixed=1) vs Test 2 (alt: ω free) + BEB
#    This is the "branch-site test of positive selection" (Zhang et al. 2005).
# =========================================================================


@validated_public('selection.branch_site_model')
def branch_site_model(
    *,
    alignment: str | Path,
    tree: str | Path,
    foreground_labels: list[str] | None = None,
    run: bool = True,
    seqfile: str = "codon_aln.paml",
    treefile: str = "tree.nwk",
) -> OrganelleResult:
    """Plan branch-site codeml inputs without writing user-facing files."""
    with tempfile.TemporaryDirectory() as tmp:
        result = write_branch_site_model(
            alignment=alignment,
            tree=tree,
            output_dir=tmp,
            foreground_labels=foreground_labels,
            run=run,
            seqfile=seqfile,
            treefile=treefile,
        )
    if result.status != "ok":
        return result
    return result.model_copy(
        update={"artifacts": (), "flags": (*result.flags, "output_dir_unbound")}
    )


def write_branch_site_model(
    *,
    alignment: str | Path,
    tree: str | Path,
    output_dir: str | Path,
    foreground_labels: list[str] | None = None,
    run: bool = True,
    seqfile: str = "codon_aln.paml",
    treefile: str = "tree.nwk",
) -> OrganelleResult:
    """Branch-site model: detect lineage-specific positive-selection sites.

    Generates two .ctl files:
      - null  (Test 1): NSsites=2, model=2, fix_omega=1, omega=1
      - alt   (Test 2): NSsites=2, model=2, fix_omega=0, omega=1.5

    LRT (df=1) tests if foreground has sites with ω>1. BEB identifies which sites.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _prepare_tree(tree, foreground_labels, out / treefile)
    _ensure_paml_format(Path(alignment), out / seqfile)

    null_ctl = write_ctl_file(
        out / "branchsite_null.ctl",
        seqfile=seqfile,
        treefile=treefile,
        outfile="branchsite_null.out",
        model=2,
        NSsites="2",
        fix_omega=1,
        omega=1.0,
    )
    alt_ctl = write_ctl_file(
        out / "branchsite_alt.ctl",
        seqfile=seqfile,
        treefile=treefile,
        outfile="branchsite_alt.out",
        model=2,
        NSsites="2",
        fix_omega=0,
        omega=1.5,
    )

    null_result = run_codeml(null_ctl, out) if run else None
    alt_result = run_codeml(alt_ctl, out) if run else None

    metrics: dict[str, Any] = {"model_type": "branch_site"}
    beb_sites: list[dict] = []
    if run and not (_parsed_likelihood(null_result) and _parsed_likelihood(alt_result)):
        return _codeml_failed("branch_site_model", {"run": run})
    if null_result and alt_result:
        lrt = likelihood_ratio_test(null_result["lnL"], alt_result["lnL"], df=1)
        # The null distribution is a 50:50 point mass at 0 and chi-square(1).
        conservative_p = lrt["p_value"]
        lrt["p_value"] = 0.5 * lrt["p_value"] if lrt["lrt_stat"] > 0 else 1.0
        beb_sites = [site for site in alt_result.get("beb_sites", [])
                     if site["prob"] > 0.95 and lrt["p_value"] < 0.05]
        metrics.update(
            {
                "lnL_null": null_result["lnL"],
                "lnL_alt": alt_result["lnL"],
                "execution": {"null": null_result.get("execution"), "alternative": alt_result.get("execution")},
                "lrt_stat": round(lrt["lrt_stat"], 4),
                "p_value": round(lrt["p_value"], 6),
                "significant": lrt["p_value"] < 0.05,
                "omega_foreground": alt_result.get("omega_foreground"),
                "null_distribution": "0.5*point_mass_0 + 0.5*chi2(1)",
                "p_value_chi2_conservative": conservative_p,
                "distribution_note": "The boundary mixture is used; the conservative chi-square(1) probability is reported separately.",
                "beb_sites": beb_sites,
                "site_classes": alt_result.get("site_classes", {}),
                "omega_foreground_classes": alt_result.get("omega_foreground_classes", []),
                "omega_background_classes": alt_result.get("omega_background_classes", []),
                "n_positive_sites": len(beb_sites),
            }
        )

    return _contract.ok(
        "branch_site_model",
        artifacts=_contract.artifacts((null_ctl, alt_ctl)),
        metrics=metrics,
        # The full BEB site table stays machine-readable in
        # ``metrics["beb_sites"]``; a canonical Finding holds one scalar, so
        # each reported site contributes its posterior probability.
        findings=_contract.findings(
            "branch_site_model",
            (
                ("lrt_pvalue", metrics.get("p_value", 1.0)),
                ("n_beb_sites", len(beb_sites)),
                *[(f"beb_site:{s.get('site')}", s.get("prob")) for s in beb_sites[:5]],
            ),
        ),
        flags=("positive_selection_branch_site",) if metrics.get("significant") else (),
        summary_text=_branchsite_summary(metrics),
        method="codeml_branchsite",
        parameters={
            "foreground_labels": list(foreground_labels or ()),
            "run": run,
            "seqfile": seqfile,
            "treefile": treefile,
        },
    )


# =========================================================================
# 4. CLADE MODELS (divergent selection between clades, CmC)
#    model=3, NSsites=2 — compares site-level ω between clades.
# =========================================================================


@validated_public('selection.clade_model')
def clade_model(
    *,
    alignment: str | Path,
    tree: str | Path,
    clade1_labels: list[str] | None = None,
    clade2_labels: list[str] | None = None,
    run: bool = True,
    seqfile: str = "codon_aln.paml",
    treefile: str = "tree.nwk",
) -> OrganelleResult:
    """Plan clade-model codeml inputs without writing user-facing files."""
    with tempfile.TemporaryDirectory() as tmp:
        result = write_clade_model(
            alignment=alignment,
            tree=tree,
            output_dir=tmp,
            clade1_labels=clade1_labels,
            clade2_labels=clade2_labels,
            run=run,
            seqfile=seqfile,
            treefile=treefile,
        )
    if result.status != "ok":
        return result
    return result.model_copy(
        update={"artifacts": (), "flags": (*result.flags, "output_dir_unbound")}
    )


def write_clade_model(
    *,
    alignment: str | Path,
    tree: str | Path,
    output_dir: str | Path,
    clade1_labels: list[str] | None = None,
    clade2_labels: list[str] | None = None,
    run: bool = True,
    seqfile: str = "codon_aln.paml",
    treefile: str = "tree.nwk",
) -> OrganelleResult:
    """Clade model C (CmC): detect divergent selection between clades.

    Generates:
      - null:  model=0, NSsites=22 (M2a_rel)
      - alt:   model=3, NSsites=2 (clade model, two clade partitions)

    LRT df is the number of effective branch types minus one.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # mark clade1 as #1, clade2 as #2 in tree
    clade_labels = {}
    if set(clade1_labels or ()) & set(clade2_labels or ()):
        raise OrganelleInputError(code="selection.clade_labels_overlap", message="clade label sets must be disjoint")
    if clade1_labels:
        for s in clade1_labels:
            clade_labels[s] = "#1"
    if clade2_labels:
        for s in clade2_labels:
            clade_labels[s] = "#2"
    _prepare_tree(tree, None, out / treefile, label_map=clade_labels)
    _ensure_paml_format(Path(alignment), out / seqfile)
    branch_types = _branch_types(out / treefile)
    if len(branch_types) < 2:
        raise OrganelleInputError(code="selection.clade_model_labels",
                                  message="clade model requires at least two effective branch types")
    if branch_types != set(range(len(branch_types))):
        raise OrganelleInputError(code="selection.clade_model_label_order",
                                  message="clade branch labels must be consecutive from background #0")
    df = len(branch_types) - 1

    null_ctl = write_ctl_file(
        out / "clade_null.ctl",
        seqfile=seqfile,
        treefile=treefile,
        outfile="clade_null.out",
        model=0,
        NSsites="22",
    )
    alt_ctl = write_ctl_file(
        out / "clade_alt.ctl",
        seqfile=seqfile,
        treefile=treefile,
        outfile="clade_alt.out",
        model=3,
        NSsites="2",
    )

    null_result = run_codeml(null_ctl, out) if run else None
    alt_result = run_codeml(alt_ctl, out) if run else None

    metrics: dict[str, Any] = {"model_type": "clade", "null_model": "M2a_rel",
                               "branch_types": sorted(branch_types), "df": df}
    if run and not (_parsed_likelihood(null_result) and _parsed_likelihood(alt_result)):
        return _codeml_failed("clade_model", {"run": run})
    if null_result and alt_result:
        if ("n_params" in null_result and "n_params" in alt_result
                and alt_result["n_params"] - null_result["n_params"] != df):
            return _codeml_failed("clade_model", {"run": run, "expected_df": df})
        lrt = likelihood_ratio_test(null_result["lnL"], alt_result["lnL"], df=df)
        metrics.update(
            {
                "lnL_null": null_result["lnL"],
                "lnL_alt": alt_result["lnL"],
                "execution": {"null": null_result.get("execution"), "alternative": alt_result.get("execution")},
                "lrt_stat": round(lrt["lrt_stat"], 4),
                "p_value": round(lrt["p_value"], 6),
                "significant": lrt["p_value"] < 0.05,
                "site_classes": alt_result.get("site_classes", {}),
            }
        )

    return _contract.ok(
        "clade_model",
        artifacts=_contract.artifacts((null_ctl, alt_ctl)),
        metrics=metrics,
        findings=_contract.findings("clade_model", (("cmc_pvalue", metrics.get("p_value", 1.0)),)),
        flags=("divergent_selection",) if metrics.get("significant") else (),
        summary_text=_clade_summary(metrics),
        method="codeml_clade",
        parameters={
            "clade1_labels": list(clade1_labels or ()),
            "clade2_labels": list(clade2_labels or ()),
            "run": run,
            "seqfile": seqfile,
            "treefile": treefile,
        },
    )


# =========================================================================
# codeml execution + output parsing
# =========================================================================


def run_codeml(ctl_path: str | Path, work_dir: str | Path, *, timeout: float | None = 3600.0) -> dict[str, Any] | None:
    """Run PAML once, respecting timeout and refusing stale output files."""
    codeml_bin = shutil.which("codeml")
    if not codeml_bin:
        return None
    ctl_path, work_dir = Path(ctl_path), Path(work_dir)
    match = re.search(r"(?m)^\s*outfile\s*=\s*(\S+)", ctl_path.read_text())
    if match is None:
        return None
    out_path = work_dir / match.group(1)
    if out_path.exists():
        raise OrganelleInputError(code="selection.codeml_output_exists",
                                  message=f"refusing to reuse existing codeml output: {out_path}")
    try:
        completed = subprocess.run([codeml_bin, str(ctl_path.resolve())], cwd=work_dir,
                                   capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if not out_path.exists():
        return None
    # CODEML writes this footer only after Forestry has completed all models.
    # An lnL is printed earlier, before detailed estimates/BEB, and is insufficient.
    text = out_path.read_text(errors="replace")
    if completed.returncode < 0 or not re.search(r"(?m)^\s*Time used:\s*\S", text):
        return None
    if "check convergence.." in text:
        return None
    parsed = parse_codeml_output(out_path)
    parsed["execution"] = {"returncode": completed.returncode, "complete": True,
                           "completion_marker": "Time used", "timeout_seconds": timeout}
    if _parsed_likelihood(parsed) or (parsed.get("model_results") and
            all(_parsed_likelihood(value) for value in parsed["model_results"].values())):
        return parsed
    return None


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@validated_public('selection.parse_codeml_output')
def parse_codeml_output(out_path: str | Path) -> dict[str, Any]:
    """Parse actual PAML fields, preserving each NSsites model separately.

    Multi-model output has ``model_results`` keyed by the NSsites number.
    No aggregate lnL or BEB table is invented across different models.
    """
    text = Path(out_path).read_text(errors="replace")
    starts = list(re.finditer(r"(?m)^\s*(?:NSsites\s+)?Model\s+(\d+)\s*:", text))
    if not starts:
        return _parse_codeml_model(text)
    model_results = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        model_results[match.group(1)] = _parse_codeml_model(text[match.end():end])
    if len(model_results) == 1:
        return {**next(iter(model_results.values())), "model_results": model_results}
    return {"model_results": model_results}


def _parse_codeml_model(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    likelihood = re.search(r"lnL\([^)]*\)\s*:\s*(" + _NUMBER + r")", text)
    if likelihood:
        result["lnL"] = float(likelihood.group(1))
        count = re.search(r"\bnp\s*:\s*(\d+)", likelihood.group(0))
        if count:
            result["n_params"] = int(count.group(1))
    kappa = re.search(r"kappa\s*\(ts/tv\)\s*=\s*(" + _NUMBER + r")", text)
    if kappa:
        result["kappa"] = float(kappa.group(1))
    omega = re.search(r"(?m)^\s*omega\s*(?:\(dN/dS\))?\s*=\s*(" + _NUMBER + r")", text)
    if omega:
        result["omega"] = float(omega.group(1))
    branch = re.search(r"w\s*\(dN/dS\)\s+for branches:\s*([^\n]+)", text)
    if branch:
        ratios = [float(value) for value in re.findall(_NUMBER, branch.group(1))]
        result["omega_by_branch_type"] = ratios
        if len(ratios) == 2:
            result.update(omega_background=ratios[0], omega_foreground=ratios[1])
    # PAML prints site classes as aligned rows, not repeated omega= blocks.
    props = re.search(r"(?m)^\s*(?:p:|proportion)\s+([^\n]+)", text)
    omegas = re.search(r"(?m)^\s*w:\s*([^\n]+)", text)
    if props and omegas:
        pv = [float(x) for x in re.findall(_NUMBER, props.group(1))]
        wv = [float(x) for x in re.findall(_NUMBER, omegas.group(1))]
        if len(pv) == len(wv):
            result["site_classes"] = {str(i): {"proportion": v, "omega": wv[i]}
                                      for i, v in enumerate(pv)}
    fg = re.search(r"(?m)^\s*foreground w\s+([^\n]+)", text)
    bg = re.search(r"(?m)^\s*background w\s+([^\n]+)", text)
    if fg:
        result["omega_foreground_classes"] = [float(x) for x in re.findall(_NUMBER, fg.group(1))]
    if bg:
        result["omega_background_classes"] = [float(x) for x in re.findall(_NUMBER, bg.group(1))]
    # Clade models print one omega row per branch type; branch-site models
    # print background/foreground rows. Preserve the separate site classes.
    branch_rows = list(re.finditer(r"(?m)^\s*branch type\s+(\d+):\s*([^\n]+)", text))
    if props and branch_rows:
        pv = [float(x) for x in re.findall(_NUMBER, props.group(1))]
        by_type = {row.group(1): [float(x) for x in re.findall(_NUMBER, row.group(2))]
                   for row in branch_rows}
        if all(len(values) == len(pv) for values in by_type.values()):
            result["site_classes"] = {str(i): {"proportion": value,
                "omega_by_branch_type": {kind: values[i] for kind, values in by_type.items()}}
                for i, value in enumerate(pv)}
    elif props and fg and bg:
        pv = [float(x) for x in re.findall(_NUMBER, props.group(1))]
        fw, bw = result["omega_foreground_classes"], result["omega_background_classes"]
        classes = re.search(r"(?m)^\s*site class\s+([^\n]+)", text)
        names = classes.group(1).split() if classes else []
        if len(names) == len(pv) == len(fw) == len(bw):
            result["site_classes"] = {name: {"proportion": pv[i],
                "omega_foreground": fw[i], "omega_background": bw[i]}
                for i, name in enumerate(names)}
    beb = re.search(r"Bayes Empirical Bayes.*?(?:\n|$)", text)
    if beb:
        rows = []
        started = False
        for line in text[beb.end():].splitlines():
            match = re.match(r"\s*(\d+)\s+([A-Z*])\s+(" + _NUMBER + r")(\*{0,2})(?:\s|$)", line)
            if match:
                started = True
                probability = float(match.group(3))
                rows.append({"site": int(match.group(1)), "aa": match.group(2),
                             "prob": probability, "significant": probability > 0.95})
            elif started:
                break
        result["beb_sites"] = rows
    return result


def _parsed_likelihood(result: dict[str, Any] | None) -> bool:
    return bool(result and isinstance(result.get("lnL"), (int, float))
                and math.isfinite(result["lnL"]))


def _codeml_failed(op: str, parameters: dict[str, Any]) -> OrganelleResult:
    return _contract.failed(op, summary_text="codeml did not produce a complete, parseable likelihood result.",
                            anomalies=["codeml_no_output"], parameters=parameters)


def _branch_types(treefile: Path) -> set[int]:
    """Effective PAML branch labels, including inherited $ clade labels."""
    from ._tree import _parse_newick
    root = _parse_newick(re.sub(r"^\s*\d+\s+1\s*\n", "", treefile.read_text()))
    types: set[int] = set()
    def number(marks: set[str], default: int) -> int:
        if not marks:
            return default
        if len(marks) != 1:
            raise OrganelleInputError(code="selection.codeml_branch_labels",
                                      message="each branch must have one effective numeric label")
        mark = next(iter(marks))
        return 1 if mark == "Foreground" else int(mark.removeprefix("Group"))
    def visit(node: Any, inherited: int, is_root: bool = False) -> None:
        clade = number(node.clade_marks, inherited)
        if not is_root:
            types.add(number(node.marks, clade))
        for child in node.children:
            visit(child, clade)
    visit(root, 0, True)
    return types


@validated_public('selection.likelihood_ratio_test')
def likelihood_ratio_test(lnL_null: float, lnL_alt: float, df: int) -> dict[str, float]:
    """Likelihood ratio test: 2*(lnL_alt - lnL_null) ~ χ²(df).

    Returns the LRT statistic and scipy's chi-square survival probability.
    Model-specific boundary mixtures are applied by the calling analysis.
    """
    if df <= 0 or not all(math.isfinite(value) for value in (lnL_null, lnL_alt)):
        raise OrganelleInputError(code="selection.invalid_lrt", message="LRT needs finite log likelihoods and positive degrees of freedom")
    lrt_stat = 2.0 * (lnL_alt - lnL_null)
    if lrt_stat < 0:
        raise OrganelleInputError(code="selection.lrt_optimization_failed",
                                  message="alternative likelihood is below its nested null; check optimization before inference")
    p_value = _chi2_sf(lrt_stat, df)
    return {"lrt_stat": lrt_stat, "p_value": p_value, "df": df}


def _chi2_sf(x: float, df: int) -> float:
    """Chi-square survival function (upper tail) via scipy."""
    return float(_scipy_stats.chi2.sf(x, df))


# =========================================================================
# Helpers: tree preparation, PAML format, summaries
# =========================================================================


def _prepare_tree(
    tree: str | Path,
    foreground_labels: list[str] | None,
    out_path: Path,
    label_map: dict[str, str] | None = None,
) -> str:
    """Read a tree file and optionally mark foreground/clade branches."""
    tree_text = (
        Path(tree).read_text().strip()
        if isinstance(tree, (str, Path)) and Path(str(tree)).exists()
        else str(tree)
    )
    tree_text = re.sub(r"^\s*\d+\s+1\s*\n", "", tree_text)
    if foreground_labels or label_map:
        from ._tree import _label_tips, _parse_newick, _serialize_codeml, _tips

        root = _parse_newick(tree_text)
        markers = (
            {label: "#1" for label in foreground_labels} if foreground_labels else label_map or {}
        )
        # Parse marker syntax with the same parser as existing tree annotations.
        annotations = {name: _parse_newick(f"tip {mark};") for name, mark in markers.items()}
        _label_tips(
            root, {name: node.marks for name, node in annotations.items()}, error_prefix="codeml"
        )

        for tip in _tips(root):
            if tip.name in annotations:
                tip.clade_marks.update(annotations[tip.name].clade_marks)
        tree_text = _serialize_codeml(root) + ";"
    # Bare Newick is read as an unspecified tree count by PAML 4.10.9;
    # it then errors at EOF after fitting the first tree. Declare one tree.
    from ._tree import _parse_newick, _tips
    n_taxa = len(_tips(_parse_newick(tree_text)))
    out_path.write_text(f"{n_taxa} 1\n{tree_text}\n")
    return tree_text


def _ensure_paml_format(fasta_path: Path, out_path: Path) -> None:
    """Ensure the alignment is in PAML sequential format (copy or convert)."""
    if fasta_path.suffix in (".paml", ".pml"):
        out_path.write_text(fasta_path.read_text())
    else:
        from .codeml import _materialize_paml, to_paml

        paml_result = to_paml(fasta_path, convert_t_to_u=True)
        _materialize_paml(paml_result, out_path)


def _branch_key_findings(metrics: dict) -> tuple[tuple[str, Any], ...]:
    findings: list[tuple[str, Any]] = []
    if "omega_foreground" in metrics:
        findings.append(("omega_foreground", metrics["omega_foreground"]))
    if "p_value" in metrics:
        findings.append(("lrt_pvalue", metrics["p_value"]))
    return tuple(findings) or (("status", "ctl_generated"),)


def _branch_summary(metrics: dict) -> str:
    if "lnL_null" not in metrics:
        return "Branch model .ctl files generated (codeml not run)."
    sig = "significant" if metrics.get("significant") else "not significant"
    w = metrics.get("omega_foreground", "?")
    return f"Branch omega heterogeneity: ω_fg={w}, LRT p={metrics.get('p_value', '?'):.4f} ({sig})."


def _site_key_findings(metrics: dict) -> tuple[tuple[str, Any], ...]:
    findings: list[tuple[str, Any]] = [("n_positive_sites", metrics.get("n_positive_sites", 0))]
    if "lnL" in metrics:
        findings.append(("lnL", metrics["lnL"]))
    return tuple(findings)


def _site_summary(metrics: dict) -> str:
    n = metrics.get("n_positive_sites", 0)
    if "model_results" not in metrics:
        return f"Site model .ctl generated (models: {metrics.get('models', '?')})."
    return f"Site model: {n} BEB positive-selection sites."


def _branchsite_summary(metrics: dict) -> str:
    if "lnL_null" not in metrics:
        return "Branch-site model .ctl files generated (codeml not run)."
    n = metrics.get("n_positive_sites", 0)
    sig = "significant" if metrics.get("significant") else "not significant"
    return f"Branch-site: LRT p={metrics.get('p_value', '?'):.4f} ({sig}), {n} BEB sites."


def _clade_summary(metrics: dict) -> str:
    if "lnL_null" not in metrics:
        return "Clade model C .ctl files generated (codeml not run)."
    sig = "significant" if metrics.get("significant") else "not significant"
    return f"Clade model C: LRT p={metrics.get('p_value', '?'):.4f} ({sig})."


# =========================================================================
# Batch parallel codeml — run multiple genes/models in parallel
# =========================================================================


@validated_public('selection.batch_codeml')
def batch_codeml(
    jobs: list[Mapping[str, str]],
    *,
    n_workers: int = 0,
    timeout: int = 600,
    codeml_path: str | Path | None = None,
) -> OrganelleResult:
    """Run multiple codeml jobs in parallel using multiprocessing.

    Each job is a dict with keys:
        ctl_path  — path to the .ctl file (required)
        work_dir  — working directory for codeml (required)

    codeml is single-threaded, but genes/models are independent, so running
    N jobs on N cores gives ~N× speedup.

    Parameters
    ----------
    jobs : list[dict]
        Each dict: ``{"ctl_path": Path, "work_dir": Path, "label": str}``.
    n_workers : int
        Number of parallel processes. 0 = min(n_jobs, cpu_count).
    timeout : int
        Per-job timeout in seconds.
    codeml_path : str, optional
        Path to codeml binary. If None, uses ``shutil.which("codeml")``.

    Returns
    -------
    OrganelleResult
        ``metrics`` contains: n_jobs, n_succeeded, n_failed, n_workers,
        wall_time, total_cpu_time, speedup, and the per-job
        ``job_results`` rows. ``findings`` carries one lnL per job.
    """
    import os
    import time as _time
    from multiprocessing import Pool

    parameters = {"n_workers": n_workers, "timeout": timeout, "n_jobs": len(jobs)}
    codeml_bin = str(codeml_path) if codeml_path else shutil.which("codeml")
    if not codeml_bin or not Path(codeml_bin).exists():
        return _contract.failed(
            "batch_codeml",
            summary_text="codeml not found. Install PAML or pass codeml_path.",
            anomalies=["codeml_not_found"],
            parameters=parameters,
        )

    if not jobs:
        return _contract.failed(
            "batch_codeml",
            summary_text="batch_codeml() requires >=1 job.",
            anomalies=["no_jobs"],
            parameters=parameters,
        )

    if n_workers <= 0:
        n_workers = min(len(jobs), os.cpu_count() or 1)

    # prepare picklable job args for the worker pool
    worker_args = [
        (codeml_bin, str(j["ctl_path"]), str(j["work_dir"]), timeout, j.get("label", ""))
        for j in jobs
    ]

    t0 = _time.perf_counter()
    if n_workers == 1:
        results = [_run_codeml_worker(args) for args in worker_args]
    else:
        with Pool(processes=n_workers) as pool:
            results = pool.map(_run_codeml_worker, worker_args)
    wall_time = _time.perf_counter() - t0

    n_succeeded = sum(1 for r in results if r.get("status") == "ok")
    n_failed = len(results) - n_succeeded
    # estimate speedup: sum of per-job times / wall time
    total_cpu_time = sum(r.get("elapsed", 0) for r in results)
    speedup = total_cpu_time / wall_time if wall_time > 0 else 0.0

    return _contract.ok(
        "batch_codeml",
        artifacts=(),
        metrics={
            "n_jobs": len(jobs),
            "n_succeeded": n_succeeded,
            "n_failed": n_failed,
            "n_workers": n_workers,
            "wall_time": round(wall_time, 2),
            "total_cpu_time": round(total_cpu_time, 2),
            "speedup": round(speedup, 2),
            # The per-job rows the pre-v1 contract carried in ``key_findings``
            # live here now (first 20, as before): a canonical Finding holds
            # one scalar, so the structured job table belongs in ``metrics``.
            "job_results": [
                {
                    "label": r.get("label", ""),
                    "status": r.get("status"),
                    "lnL": r.get("lnL"),
                }
                for r in results[:20]
            ],
        },
        findings=_contract.findings(
            "batch_codeml",
            tuple(
                (f"job_result:{r.get('label', '')}", r.get("lnL")) for r in results[:20]
            ),  # first 20 in findings
        ),
        flags=("all_succeeded",) if n_failed == 0 else (),
        summary_text=(
            f"Batch codeml: {n_succeeded}/{len(jobs)} succeeded in "
            f"{wall_time:.1f}s ({n_workers} workers, ~{speedup:.1f}× speedup)."
        ),
        method="parallel_multiprocessing",
        parameters=parameters,
    )


def _run_codeml_worker(args: tuple) -> dict:
    """Worker function for batch_codeml — runs one codeml job.

    Must be top-level (not nested) for multiprocessing pickling.
    """
    codeml_bin, ctl_path_str, work_dir_str, timeout, label = args
    import time as _time

    ctl_path = Path(ctl_path_str)
    work_dir = Path(work_dir_str)
    result: dict[str, Any] = {"label": label, "status": "failed", "lnL": None, "elapsed": 0.0}

    t0 = _time.perf_counter()
    try:
        r = subprocess.run(
            [codeml_bin, ctl_path.name],
            cwd=work_dir_str,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result["elapsed"] = _time.perf_counter() - t0
        result["returncode"] = r.returncode
        # find outfile
        ctl_text = ctl_path.read_text()
        m = re.search(r"outfile\s*=\s*(\S+)", ctl_text)
        if m:
            out_path = work_dir / m.group(1)
            if out_path.exists():
                parsed = parse_codeml_output(out_path)
                if parsed and "lnL" in parsed:
                    result["status"] = "ok"
                    result["lnL"] = parsed["lnL"]
                    result["parsed"] = parsed
    except subprocess.TimeoutExpired:
        result["elapsed"] = timeout
        result["error"] = "timeout"
    except Exception as exc:
        result["elapsed"] = _time.perf_counter() - t0
        result["error"] = str(exc)[:200]

    return result
