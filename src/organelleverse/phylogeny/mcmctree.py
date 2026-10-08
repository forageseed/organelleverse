"""PAML approximate-likelihood relaxed-clock dating, with two independent chains.

PAML's time unit is 100 Ma. Bounds become soft calibration densities (2.5%
per specified tail); L additionally uses PAML's offset=0.1, scale=1 defaults.
Input node bounds are not the marginal effective priors after tree ordering.
"""

from __future__ import annotations

import io
import math
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
from Bio import Phylo
from pydantic import ValidationError
from scipy.signal import fftconvolve

from .._bio import read_fasta
from ..core.errors import OrganelleDependencyError
from ..core.external import run_external
from ._results import artifact_for, ok_result, provenance
from .dating import Calibration, MCMCTreeOptions, _invalid, _output_error
from .partition import _charset_sites, parse_nexus_sets

TIME_UNIT_MA = 100.0


def _partition_alignment(sequences, aliases, partition_nexus):
    """Write consecutive PAML DNA blocks; every input site belongs to one block."""
    nchar = len(sequences[0][1])
    scheme = {"charsets": [("all", f"1-{nchar}")], "models": {}}
    if partition_nexus is not None:
        try:
            scheme = parse_nexus_sets(partition_nexus)
        except OSError as error:
            raise _invalid(f"Cannot read partition file: {error}") from error
    if any(model != "GTR+G4" for model in scheme["models"].values()):
        raise _invalid("MCMCTree partitions require GTR+G4; provide charsets without other models.")
    blocks, partitions, covered = [], [], set()
    names = set()
    for name, spec in scheme["charsets"]:
        try:
            sites = _charset_sites(spec, nchar)
        except ValueError as error:
            raise _invalid(f"Invalid partition {name}: {error}") from error
        if name in names or not sites:
            raise _invalid("Partition names must be unique and each partition nonempty.")
        names.add(name)
        if min(sites) < 1 or max(sites) > nchar:
            raise _invalid("Partition sites are outside the alignment.")
        if len(set(sites)) != len(sites) or covered.intersection(sites):
            raise _invalid("Partitions must cover every alignment site exactly once (overlap).")
        covered.update(sites)
        sites.sort()
        blocks.append(
            f"{len(aliases)} {len(sites)}\n"
            + "".join(
                f"{aliases[taxon]}  {''.join(seq[i - 1] for i in sites).upper()}\n"
                for taxon, seq in sequences
            )
        )
        partitions.append({"name": name, "charset": spec, "n_sites": len(sites), "model": "GTR+G4"})
    if len(covered) != nchar:
        raise _invalid("Partitions must cover every alignment site exactly once (missing sites).")
    return "\n".join(blocks), partitions


def _bound(c: Calibration) -> str:
    lo, hi = c.min_age_ma, c.max_age_ma
    if lo is not None and hi is not None:
        if lo == hi:
            raise _invalid(
                "MCMCTree soft bounds require min_age_ma < max_age_ma; fixed ages are unsupported."
            )
        return f"B({lo / TIME_UNIT_MA:.17g},{hi / TIME_UNIT_MA:.17g})"
    return f"L({lo / TIME_UNIT_MA:.17g})" if hi is None else f"U({hi / TIME_UNIT_MA:.17g})"


def _tree_text(tree, rows, aliases):
    labels = {tree.root if c.root else tree.common_ancestor(c.mrca): _bound(c) for c in rows}

    def newick(node):
        if node.is_terminal():
            return aliases[node.name]
        return (
            "("
            + ",".join(newick(c) for c in node.clades)
            + ")"
            + (f"'{labels[node]}'" if node in labels else "")
        )

    return f"{len(aliases)} 1\n{newick(tree.root)};\n"


def effective_sample_size(values: np.ndarray) -> float:
    """Geyer initial-positive, initial-monotone paired autocorrelation ESS.

    FFT computes biased autocovariances. Constant traces have ESS=0, since
    a stuck chain is not evidence of posterior exploration.
    """
    if np.ptp(values) == 0:
        return 0.0
    centered = np.asarray(values, dtype=float) - np.mean(values)
    n = len(centered)
    cov = fftconvolve(centered, centered[::-1], mode="full")[n - 1 :] / n
    if cov[0] <= 0:
        return 0.0
    rho = cov / cov[0]
    pairs = rho[: 2 * (n // 2)].reshape(-1, 2).sum(axis=1)
    nonpositive = np.flatnonzero(pairs <= 0)
    if len(nonpositive):
        pairs = pairs[: nonpositive[0]]
    pairs = np.minimum.accumulate(pairs)
    tau = -1 + 2 * float(pairs.sum())
    return float(n / tau) if tau > 0 else 0.0


def _split_rhat(chains: list[np.ndarray]) -> float | None:
    n = min(len(c) for c in chains) // 2
    split = np.array([part for c in chains for part in (c[:n], c[-n:])])
    within = float(np.var(split, axis=1, ddof=1).mean())
    if within == 0:
        return None
    between = n * float(np.var(split.mean(axis=1), ddof=1))
    return math.sqrt(((n - 1) / n * within + between / n) / within)


def _hpd(values):
    ordered = np.sort(values)
    count = math.ceil(0.95 * len(ordered))
    widths = ordered[count - 1 :] - ordered[: len(ordered) - count + 1]
    start = int(np.argmin(widths))
    return float(ordered[start]), float(ordered[start + count - 1])


def _read_trace(path: Path, nsample: int, sampfreq: int):
    try:
        with path.open() as stream:
            header = next((line.split() for line in stream if line.strip()), [])
            data = np.loadtxt(stream, ndmin=2)
    except (OSError, ValueError) as error:
        raise _output_error(f"Cannot read MCMCTree samples: {error}") from error
    expected = np.arange(sampfreq, nsample * sampfreq + 1, sampfreq)
    # PAML writes Gen=1 in addition to the scheduled samples. Remove that
    # single unscheduled post-burnin draw so autocorrelations have uniform lags.
    generations = np.r_[1, expected] if sampfreq > 1 else expected
    if data.shape != (len(generations), len(header)) or not np.isfinite(data).all():
        raise _output_error("MCMCTree samples are incomplete, malformed or nonfinite.")
    if not header or header[0] != "Gen" or len(header) != len(set(header)):
        raise _output_error("Invalid MCMCTree sample header.")
    if not np.array_equal(data[:, 0], generations):
        raise _output_error("MCMCTree sample generations do not match the requested schedule.")
    return header[1:], data[int(sampfreq > 1) :, 1:]


def summarize_chains(tree, paths, nsample, sampfreq):
    """Map PAML preorder internal IDs to clades and summarize saved draws."""
    headers, traces = zip(*[_read_trace(p, nsample, sampfreq) for p in paths], strict=True)
    if headers[0] != headers[1]:
        raise _output_error("MCMCTree chains have different parameter columns.")
    header = headers[0]
    nodes = tree.get_nonterminals(order="preorder")
    # PAML ReadTreeN numbers internal nodes in preorder, after the n tips.
    names = [f"t_n{i}" for i in range(len(tree.get_terminals()) + 1, 2 * len(tree.get_terminals()))]
    if {h for h in header if h.startswith("t_n")} != set(names):
        raise _output_error("MCMCTree age columns do not match the rooted binary input tree.")
    diagnostics = []
    for j, name in enumerate(header):
        chains = [t[:, j] for t in traces]
        ess = [effective_sample_size(c) for c in chains]
        rhat = _split_rhat(chains)
        means = [float(c.mean()) for c in chains]
        scale = TIME_UNIT_MA if name in names else 1.0
        diagnostics.append(
            {
                "parameter": name,
                "ess": ess,
                "split_rhat": rhat,
                "chain_means": [m * scale for m in means],
                "absolute_mean_difference": abs(means[0] - means[1]) * scale,
                "unit": "Ma" if name in names else "PAML native",
                "passed": min(ess) >= 200 and rhat is not None and rhat <= 1.01,
            }
        )
    rows = []
    ages = {}
    for node, name in zip(nodes, names, strict=True):
        j = header.index(name)
        draws = np.concatenate([t[:, j] for t in traces]) * TIME_UNIT_MA
        if np.any(draws <= 0):
            raise _output_error("MCMCTree returned nonpositive internal ages.")
        ages[node] = float(draws.mean())
        low, high = _hpd(draws)
        rows.append(
            {
                "node_id": name,
                "is_root": node is tree.root,
                "descendant_taxa": sorted(t.name for t in node.get_terminals()),
                "age_ma": ages[node],
                "hpd_lower_ma": low,
                "hpd_upper_ma": high,
                "ci_lower_ma": low,
                "ci_upper_ma": high,
            }
        )
    for parent in nodes:
        for child in parent.clades:
            child.branch_length = ages[parent] - ages.get(child, 0.0)
            if child.branch_length < 0:
                raise _output_error("MCMCTree posterior ages violate ancestry ordering.")
    for node in tree.find_clades():
        node.confidence = None
        node.comment = None
        if not node.is_terminal():
            node.name = None
    tree.root.branch_length = 0.0
    stream = io.StringIO()
    Phylo.write(tree, stream, "newick", format_branch_length="%1.12g")
    failed = [d["parameter"] for d in diagnostics if not d["passed"]]
    return {
        "node_ages": rows,
        "timetree_newick": stream.getvalue().strip(),
        "taxa": sorted(t.name for t in tree.get_terminals()),
        "age_source": "MCMCTree two-chain posterior samples",
        "ci_method": "shortest empirical 95% HPD interval (pooled chains)",
        "ci_level": 0.95,
        "convergence": {
            "passed": not failed,
            "failed_parameters": failed,
            "min_ess": 200,
            "max_split_rhat": 1.01,
            "ess_method": "Geyer initial positive monotone sequence (FFT)",
            "parameters": diagnostics,
        },
    }


def run_mcmctree(
    tree,
    alignment_fasta,
    rows,
    *,
    output_dir,
    options,
    model,
    partition_nexus,
    seed,
    binary,
    dry_run,
):
    try:
        opts = MCMCTreeOptions.model_validate(options or {})
    except ValidationError as error:
        raise _invalid(f"Invalid MCMCTree options: {error}") from error
    for prior in (opts.rgene_gamma, opts.sigma2_gamma):
        if len(prior) != 2 or any(not math.isfinite(p) or p <= 0 for p in prior):
            raise _invalid("MCMCTree Gamma shape/rate must contain two positive finite parameters.")
    if model != "GTR+G4":
        raise _invalid("MCMCTree requires model='GTR+G4' for each DNA partition.")
    if any(len(n.clades) != 2 for n in tree.get_nonterminals()):
        raise _invalid("MCMCTree requires a fully bifurcating rooted tree.")
    if not any(
        (c.root or tree.common_ancestor(c.mrca) is tree.root) and c.max_age_ma is not None
        for c in rows
    ):
        raise _invalid("MCMCTree requires an explicit upper age bound on the root.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 1 <= seed <= 2147483644:
        raise _invalid("MCMCTree seed must be an integer in 1..2147483644.")
    aliases = {tip.name: f"T{i:04d}" for i, tip in enumerate(tree.get_terminals(), 1)}
    tree_text = _tree_text(tree, rows, aliases)
    sequences = list(read_fasta(alignment_fasta))
    if any(set(seq.upper()) - set("ACGTRYMKSWBDHVN?-") for _, seq in sequences):
        raise _invalid("MCMCTree requires IUPAC DNA alignment symbols.")
    alignment, partitions = _partition_alignment(sequences, aliases, partition_nexus)
    selected = binary or os.environ.get("ORGANELLEVERSE_MCMCTREE_BIN") or "mcmctree"
    executable = shutil.which(selected)
    tool_dir = str(Path(executable).resolve().parent) if executable else ""
    env = dict(os.environ, PATH=tool_dir + os.pathsep + os.environ.get("PATH", ""))
    if not dry_run and (not executable or not shutil.which("baseml", path=env["PATH"])):
        raise OrganelleDependencyError(
            code="phylogeny.dating.missing_paml",
            message="MCMCTree and BASEML are required. Install: micromamba create -n ov-mcmctree -c conda-forge -c bioconda paml; set ORGANELLEVERSE_MCMCTREE_BIN.",
        )
    output = Path(output_dir).resolve()
    metrics = {
        "backend": "mcmctree",
        "age_unit": "Ma",
        "tip_age_ma": 0,
        "time_unit_ma": TIME_UNIT_MA,
        "model": model,
        "partition_nexus": str(partition_nexus) if partition_nexus is not None else None,
        "n_partitions": len(partitions),
        "partitions": partitions,
        "rate_prior": "gamma-Dirichlet; rgene_gamma and sigma2_gamma shape/rate; concentration=1; prior=0",
        "n_taxa": len(aliases),
        "alignment_length": len(sequences[0][1]),
        "calibrations": [c.model_dump(mode="json") for c in rows],
        "options": opts.model_dump(mode="json"),
        "planned": dry_run,
        "calibration_semantics": "soft B/L/U; tail probabilities 0.025; L offset=0.1, scale=1",
        "chain_seeds": [seed, seed + 2],
        "taxon_aliases": aliases,
        "commands": [[executable or selected, "run.ctl"]] * 3,
    }
    prov = provenance(
        "date_tree",
        method="mcmctree",
        argv=metrics["commands"][0],
        parameters=metrics,
        random_seed=seed,
    )
    if dry_run:
        return ok_result(
            "date_tree",
            status="warning",
            summary_text="MCMCTree two-chain dating plan; no ages estimated.",
            flags=("planned_only",),
            metrics=metrics,
            result_provenance=prov,
        )
    output.mkdir(parents=True, exist_ok=False)
    common = {
        "seqfile": "aln.phy",
        "treefile": "tree.tre",
        "outfile": "out.txt",
        "mcmcfile": "mcmc.txt",
        "ndata": len(partitions),
        "seqtype": 0,
        "clock": opts.clock,
        "model": 7,
        "alpha": 0.5,
        "ncatG": 4,
        "cleandata": 0,
        "BDparas": "1 1 0.1 c",
        "rgene_gamma": " ".join(map(str, opts.rgene_gamma)) + " 1 0",
        "sigma2_gamma": " ".join(map(str, opts.sigma2_gamma)) + " 1",
        "print": 1,
        "burnin": opts.burnin,
        "sampfreq": opts.sampfreq,
        "nsample": opts.nsample,
    }
    timings = {}
    for label, usedata, run_seed in [
        ("likelihood", "3", seed),
        ("chain1", "2 in.BV", seed),
        ("chain2", "2 in.BV", seed + 2),
    ]:
        work = output / label
        work.mkdir()
        (work / "aln.phy").write_text(alignment)
        (work / "tree.tre").write_text(tree_text)
        if usedata.startswith("2"):
            shutil.copyfile(output / "likelihood/out.BV", work / "in.BV")
        settings = dict(common, usedata=usedata, seed=run_seed)
        (work / "run.ctl").write_text("".join(f"{k} = {v}\n" for k, v in settings.items()))
        started = time.monotonic()
        completed = run_external(
            [executable, "run.ctl"],
            cwd=work,
            env=env,
            tool="MCMCTree",
            code="phylogeny.dating.mcmctree_failed",
        )
        timings[label] = time.monotonic() - started
        (work / "stdout.txt").write_text(completed.stdout)
        (work / "stderr.txt").write_text(completed.stderr)
        if label == "likelihood":
            bv = work / "out.BV"
            if not bv.is_file() or not bv.stat().st_size:
                raise _output_error(
                    "MCMCTree usedata=3 produced no branch likelihood (out.BV); inspect BASEML logs."
                )
    version = re.search(r"MCMCTREE[^\n]+", completed.stdout)
    if version is None:
        raise _output_error("MCMCTree version missing from output.")
    metrics.update(
        summarize_chains(
            tree,
            [output / "chain1/mcmc.txt", output / "chain2/mcmc.txt"],
            opts.nsample,
            opts.sampfreq,
        )
    )
    metrics.update(software_version=version.group(0), elapsed_seconds=timings)
    dated = output / "dated_tree.nwk"
    dated.write_text(metrics["timetree_newick"] + "\n")
    metrics["time_newick_path"] = str(dated)
    passed = metrics["convergence"]["passed"]
    return ok_result(
        "date_tree",
        status="ok" if passed else "warning",
        metrics=metrics,
        summary_text=f"MCMCTree dated {len(metrics['node_ages'])} internal nodes (Ma); "
        + (
            "convergence checks passed."
            if passed
            else "MCMC convergence checks FAILED; posterior estimates are provisional."
        ),
        flags=() if passed else ("mcmc_not_converged",),
        artifacts=(artifact_for(dated, kind="tree", format="newick"),),
        result_provenance=prov,
    )
