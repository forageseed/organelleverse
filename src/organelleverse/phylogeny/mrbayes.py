"""MrBayes Bayesian tree inference on a partitioned alignment, with convergence diagnostics.

MrBayes is a complete MCMC inference program, so it is called as an external
tool (``mb`` or the MPI build ``mb-mpi`` under ``mpirun``) through
:func:`organelleverse.core.external.run_external`. Everything around it is
native Python:

* :func:`write_mrbayes_nexus` turns a FASTA alignment plus a NEXUS partition
  scheme (typically IQ-TREE's ``*.best_scheme.nex`` from
  :func:`.partition.select_partition_scheme`, whose ``charpartition`` carries
  one model per partition) into a runnable MrBayes file: data block,
  ``charset``/``partition``, per-partition ``lset``/``prset`` from
  :func:`.partition.iqtree_to_mrbayes_model`, ``unlink`` of the substitution
  parameters, ``prset ratepr=variable`` (partition rate multipliers, linked
  topology and branch lengths - the MrBayes analogue of IQ-TREE ``-p``),
  ``mcmcp``/``mcmc``/``sump``/``sumt``.
* :func:`run_mrbayes` runs it and reports convergence honestly: the average
  standard deviation of split frequencies (ASDSF), PSRF and ESS as reported by
  MrBayes (``sumt``/``sump`` output, ``.mcmc``, ``.pstat``), plus an
  independent recomputation from the raw ``.run*.t``/``.run*.p`` samples. A run
  that misses a threshold is returned with ``status="warning"`` and the flag
  ``mcmc_not_converged``; it is never reported as converged.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .._bio import read_fasta
from ..core.errors import (
    OrganelleDependencyError,
    OrganelleExecutionError,
    OrganelleInputError,
)
from ..core.external import run_external
from ..core.result import OrganelleResult
from ._results import artifact_for, failed_result, findings, ok_result, provenance
from .partition import _charset_sites, iqtree_to_mrbayes_model, parse_nexus_sets
from .treecompare import newick_clades, read_tree, tree_splits

__all__ = [
    "MRBAYES_INSTALL_HINT",
    "effective_sample_size",
    "psrf",
    "recompute_asdsf",
    "resolve_mrbayes",
    "run_mrbayes",
    "write_mrbayes_nexus",
]

MRBAYES_INSTALL_HINT = (
    "MrBayes not found (tried mb_bin, $ORGANELLEVERSE_MRBAYES_BIN, mb-mpi/mb on PATH and "
    "conda/micromamba envs). Install into a user env: micromamba create -n ov-mrbayes "
    "-c conda-forge -c bioconda mrbayes   (the bioconda 3.2.8 build ships the MPI "
    "binary mb-mpi and mpirun)."
)

DEFAULT_MODEL = "GTR+F+I+G4"


# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------


def _env_bin_dirs() -> list[Path]:
    dirs: list[Path] = []
    roots = [os.environ.get("MAMBA_ROOT_PREFIX"), os.environ.get("CONDA_PREFIX")]
    home = Path.home()
    roots += [str(home / "micromamba"), str(home / "software" / "micromamba" / "root")]
    for root in roots:
        if not root:
            continue
        envs = Path(root) / "envs"
        if envs.is_dir():
            preferred = envs / "ov-mrbayes" / "bin"
            if preferred.is_dir():
                dirs.append(preferred)
            dirs.extend(p / "bin" for p in sorted(envs.iterdir()) if (p / "bin").is_dir())
    try:
        from ..assembly.install import _list_conda_envs

        dirs.extend(_list_conda_envs())
    except Exception:  # pragma: no cover - env scanning is best effort
        pass
    return list(dict.fromkeys(dirs))


def resolve_mrbayes(mb_bin: str | None = None, *, mpi: bool = True) -> dict[str, Any] | None:
    """Locate MrBayes. Returns ``{"mb", "mpirun", "mpi"}`` or ``None``.

    Order: explicit ``mb_bin``; ``$ORGANELLEVERSE_MRBAYES_BIN``; ``mb-mpi``
    (when ``mpi``) then ``mb`` on PATH; the same names in conda/micromamba env
    ``bin`` directories (an env named ``ov-mrbayes`` first). ``mpirun`` is
    taken from the same directory as the binary, else PATH. A binary counts
    as MPI when its name contains ``mpi`` and an ``mpirun`` was found.
    """
    names = ("mb-mpi", "mb") if mpi else ("mb",)
    found: str | None = None
    for candidate in (mb_bin, os.environ.get("ORGANELLEVERSE_MRBAYES_BIN")):
        if candidate:
            path = shutil.which(candidate) or (
                candidate if Path(candidate).is_file() and os.access(candidate, os.X_OK) else None
            )
            if path:
                found = path
                break
    if found is None:
        for name in names:
            found = shutil.which(name)
            if found:
                break
    if found is None:
        for directory in _env_bin_dirs():
            for name in names:
                candidate_path = directory / name
                if candidate_path.is_file() and os.access(candidate_path, os.X_OK):
                    found = str(candidate_path)
                    break
            if found:
                break
    if found is None:
        return None
    sibling = Path(found).parent / "mpirun"
    mpirun = str(sibling) if sibling.is_file() else shutil.which("mpirun")
    is_mpi = mpi and "mpi" in Path(found).name.lower() and mpirun is not None
    return {"mb": found, "mpirun": mpirun if is_mpi else None, "mpi": is_mpi}


# ---------------------------------------------------------------------------
# NEXUS writer
# ---------------------------------------------------------------------------


def _scheme_partitions(
    scheme_nexus: str | Path, nchar: int, models: Mapping[str, str] | None, default_model: str
) -> list[dict[str, Any]]:
    scheme = parse_nexus_sets(scheme_nexus)
    chosen = dict(scheme["models"])
    chosen.update(dict(models or {}))
    seen: dict[int, str] = {}
    rows: list[dict[str, Any]] = []
    for index, (name, spec) in enumerate(scheme["charsets"], start=1):
        sites = _charset_sites(spec, nchar)
        for site in sites:
            if site < 1 or site > nchar:
                raise OrganelleInputError(
                    code="phylogeny.mrbayes.charset_out_of_range",
                    message=f"charset {name} refers to site {site} beyond nchar={nchar}",
                )
            if site in seen:
                raise OrganelleInputError(
                    code="phylogeny.mrbayes.overlapping_charsets",
                    message=f"site {site} is in both {seen[site]} and {name}",
                )
            seen[site] = name
        model = chosen.get(name) or default_model
        rows.append(
            {
                "id": f"P{index}",
                "source_name": name,
                "spec": " ".join(spec.split()),
                "n_sites": len(sites),
                "model_source": "scheme" if name in chosen else "default",
                "mrbayes": iqtree_to_mrbayes_model(model),
            }
        )
    missing = nchar - len(seen)
    if missing:
        raise OrganelleInputError(
            code="phylogeny.mrbayes.incomplete_partition",
            message=f"{missing} of {nchar} alignment sites are not in any charset",
        )
    return rows


def write_mrbayes_nexus(
    alignment_fasta: str | Path,
    scheme_nexus: str | Path,
    output_path: str | Path,
    *,
    models: Mapping[str, str] | None = None,
    default_model: str = DEFAULT_MODEL,
    ngen: int = 1_000_000,
    samplefreq: int = 1000,
    printfreq: int = 10_000,
    diagnfreq: int = 10_000,
    nruns: int = 2,
    nchains: int = 4,
    temp: float = 0.1,
    burninfrac: float = 0.25,
    seed: int = 42,
    stoprule: bool = False,
    stopval: float = 0.01,
) -> dict[str, Any]:
    """Write a runnable MrBayes NEXUS file; returns the per-partition settings.

    Charsets of ``scheme_nexus`` are renamed ``P1..Pk`` (MrBayes names may
    not contain ``+``); the original names stay in a comment and in the
    returned table. Models come from the scheme's ``charpartition`` (IQ-TREE
    ``best_scheme.nex``), overridden by ``models`` ({charset: IQ-TREE model}),
    else ``default_model``. Every site must be in exactly one charset.
    """
    if nruns < 2:
        raise OrganelleInputError(
            code="phylogeny.mrbayes.nruns",
            message="nruns must be >= 2 so split-frequency convergence can be assessed",
        )
    if not 0 <= burninfrac < 1:
        raise OrganelleInputError(
            code="phylogeny.mrbayes.burnin", message="burninfrac must be in [0, 1)"
        )
    records = read_fasta(Path(alignment_fasta))
    lengths = {len(seq) for _, seq in records}
    if len(lengths) != 1:
        raise OrganelleInputError(
            code="phylogeny.mrbayes.unaligned", message="alignment sequences differ in length"
        )
    nchar = lengths.pop()
    for name, _ in records:
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
            raise OrganelleInputError(
                code="phylogeny.mrbayes.bad_taxon_name",
                message=f"taxon name {name!r} is not a plain NEXUS token",
            )
    parts = _scheme_partitions(scheme_nexus, nchar, models, default_model)
    k = len(parts)

    lines = [
        "#NEXUS",
        "[Written by organelleverse.phylogeny.write_mrbayes_nexus]",
        "begin data;",
        f"  dimensions ntax={len(records)} nchar={nchar};",
        "  format datatype=dna missing=? gap=-;",
        "  matrix",
    ]
    width = max(len(name) for name, _ in records) + 2
    lines += [f"  {name.ljust(width)}{seq.upper()}" for name, seq in records]
    lines += ["  ;", "end;", "", "begin mrbayes;"]
    lines.append(f"  set autoclose=yes nowarn=yes seed={seed} swapseed={seed};")
    for part in parts:
        lines.append(
            f"  [{part['id']} = {part['source_name']} : {part['mrbayes']['iqtree_model']}]"
        )
        lines.append(f"  charset {part['id']} = {part['spec']};")
    settings = [p["mrbayes"] for p in parts]
    if k > 1:
        lines.append(f"  partition ovscheme = {k}: {', '.join(p['id'] for p in parts)};")
        lines.append("  set partition = ovscheme;")
    for index, cfg in enumerate(settings, start=1):
        applyto = f" applyto=({index})" if k > 1 else ""
        rates = f" rates={cfg['rates']}"
        if cfg["rates"] in {"gamma", "invgamma"}:
            rates += f" ngammacat={cfg['ngammacat']}"
        lines.append(f"  lset{applyto} nst={cfg['nst']}{rates};")
        lines.append(f"  prset{applyto} statefreqpr={cfg['statefreqpr']};")
    if k > 1:
        unlink = []
        if any(c["statefreqpr"] != "fixed(equal)" for c in settings):
            unlink.append("statefreq=(all)")
        if any(c["nst"] == 6 for c in settings):
            unlink.append("revmat=(all)")
        if any(c["nst"] == 2 for c in settings):
            unlink.append("tratio=(all)")
        if any(c["rates"] in {"gamma", "invgamma"} for c in settings):
            unlink.append("shape=(all)")
        if any(c["rates"] in {"propinv", "invgamma"} for c in settings):
            unlink.append("pinvar=(all)")
        if unlink:
            lines.append(f"  unlink {' '.join(unlink)};")
        lines.append("  prset applyto=(all) ratepr=variable;")
    lines.append(
        f"  mcmcp ngen={ngen} nruns={nruns} nchains={nchains} temp={temp} "
        f"samplefreq={samplefreq} printfreq={printfreq} diagnfreq={diagnfreq} "
        f"burninfrac={burninfrac} relburnin=yes stoprule={'yes' if stoprule else 'no'} "
        f"stopval={stopval} checkpoint=yes checkfreq={max(diagnfreq, samplefreq)};"
    )
    lines += [
        "  mcmc;",
        "  sump;",
        "  sumt contype=halfcompat conformat=simple;",
        "end;",
        "",
    ]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return {
        "nexus": str(out),
        "ntax": len(records),
        "nchar": nchar,
        "n_partitions": k,
        "partitions": parts,
        "mcmc": {
            "ngen": ngen,
            "nruns": nruns,
            "nchains": nchains,
            "temp": temp,
            "samplefreq": samplefreq,
            "printfreq": printfreq,
            "diagnfreq": diagnfreq,
            "burninfrac": burninfrac,
            "stoprule": stoprule,
            "stopval": stopval,
            "seed": seed,
        },
    }


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _read_table(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("["):
            continue
        fields = [f for f in line.split("\t")]
        if header is None:
            header = [h.strip() for h in fields]
            continue
        rows.append({h: v.strip() for h, v in zip(header, fields, strict=False)})
    return rows


def _float(value: str | None) -> float | None:
    try:
        number = float(value) if value not in (None, "", "NA") else None
    except ValueError:
        return None
    return number if number is not None and math.isfinite(number) else None


def _parse_stdout(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, pattern in (
        ("asdsf", r"Average standard deviation of split frequencies = ([\d.eE+-]+)"),
        ("max_sdsf", r"Maximum standard deviation of split frequencies = ([\d.eE+-]+)"),
        ("avg_psrf", r"Average PSRF for parameter values.*?= ([\d.eE+-]+)"),
        ("max_psrf", r"Maximum PSRF for parameter values = ([\d.eE+-]+)"),
    ):
        matches = re.findall(pattern, text)
        if matches:
            out[key] = _float(matches[-1])
    match = re.search(r"MrBayes (\d+\.\d+\.\d+\S*)", text)
    if match:
        out["mrbayes_version"] = match.group(1)
    return out


def _parse_mcmc_trace(path: Path) -> list[tuple[int, float]]:
    trace: list[tuple[int, float]] = []
    if not path.is_file():
        return trace
    rows = _read_table(path)
    for row in rows:
        value = _float(row.get("AvgStdDev(s)"))
        gen = row.get("Gen")
        if value is not None and gen:
            trace.append((int(float(gen)), value))
    return trace


def _posterior_summary(labels: Sequence[float]) -> dict[str, Any]:
    if not labels:
        return {"n_internal_splits": 0}
    bins = {"1.00": 0, "0.95-0.99": 0, "0.90-0.95": 0, "0.70-0.90": 0, "0.50-0.70": 0}
    for p in labels:
        if p >= 0.9995:
            bins["1.00"] += 1
        elif p >= 0.95:
            bins["0.95-0.99"] += 1
        elif p >= 0.90:
            bins["0.90-0.95"] += 1
        elif p >= 0.70:
            bins["0.70-0.90"] += 1
        else:
            bins["0.50-0.70"] += 1
    return {
        "n_internal_splits": len(labels),
        "mean": round(statistics.fmean(labels), 4),
        "median": round(statistics.median(labels), 4),
        "min": round(min(labels), 4),
        "n_pp_ge_0_95": sum(1 for p in labels if p >= 0.95),
        "n_pp_eq_1": bins["1.00"],
        "histogram": bins,
        "values": sorted(round(p, 4) for p in labels),
    }


# ---------------------------------------------------------------------------
# Independent diagnostics (recomputed from raw samples)
# ---------------------------------------------------------------------------


def _read_t_file(path: Path) -> list[str]:
    text = path.read_text()
    translate: dict[str, str] = {}
    match = re.search(r"(?is)\btranslate\b(.*?);", text)
    if match:
        for item in match.group(1).split(","):
            parts = item.split()
            if len(parts) >= 2:
                translate[parts[0]] = parts[1]
    trees = re.findall(r"(?im)^\s*tree\s+\S+\s*=\s*(?:\[&[UR]\]\s*)?(\(.*;)\s*$", text)
    if translate:
        pattern = re.compile(r"(?<=[(,])\s*([^(),:;\s]+)")
        trees = [pattern.sub(lambda m: translate.get(m.group(1), m.group(1)), t) for t in trees]
    return trees


def recompute_asdsf(
    tree_files: Sequence[str | Path], *, burninfrac: float = 0.25, minpartfreq: float = 0.10
) -> dict[str, Any]:
    """Recompute the ASDSF from the per-run ``.t`` tree samples.

    Mirrors MrBayes: discard the first ``int(burninfrac * n_samples)`` trees
    of each run, compute each non-trivial split's frequency per run, keep
    splits whose frequency reaches ``minpartfreq`` in at least one run, and
    average their across-run sample standard deviation (n-1 denominator).
    """
    per_run: list[dict[frozenset[str], float]] = []
    samples_used: list[int] = []
    for path in tree_files:
        trees = _read_t_file(Path(path))
        burn = int(burninfrac * len(trees))
        kept = trees[burn:]
        counts: dict[frozenset[str], int] = {}
        for newick in kept:
            _, splits = tree_splits(newick)
            for split in splits:
                counts[split] = counts.get(split, 0) + 1
        samples_used.append(len(kept))
        per_run.append({s: c / len(kept) for s, c in counts.items()} if kept else {})
    if len(per_run) < 2 or not all(samples_used):
        return {"asdsf": None, "max_sdsf": None, "n_splits": 0, "samples_per_run": samples_used}
    all_splits = set().union(*per_run)
    sds: list[float] = []
    for split in all_splits:
        freqs = [run.get(split, 0.0) for run in per_run]
        if max(freqs) < minpartfreq:
            continue
        sds.append(statistics.stdev(freqs))
    return {
        "asdsf": statistics.fmean(sds) if sds else 0.0,
        "max_sdsf": max(sds) if sds else 0.0,
        "n_splits": len(sds),
        "samples_per_run": samples_used,
    }


def effective_sample_size(values: Sequence[float]) -> float:
    """ESS by Geyer's initial positive sequence estimator of autocorrelation time."""
    n = len(values)
    if n < 4:
        return float(n)
    mean = statistics.fmean(values)
    centered = [v - mean for v in values]
    var = sum(c * c for c in centered) / n
    if var == 0:
        return float(n)

    def rho(lag: int) -> float:
        return sum(centered[i] * centered[i + lag] for i in range(n - lag)) / (n * var)

    tau = 1.0
    lag = 1
    while lag + 1 < n:
        pair = rho(lag) + rho(lag + 1)
        if pair <= 0:
            break
        tau += 2 * pair
        lag += 2
    return n / tau


def psrf(chains: Sequence[Sequence[float]]) -> float | None:
    """Gelman-Rubin PSRF as in MrBayes: sqrt(((n-1)/n W + (m+1)/(m n) B) / W)."""
    m = len(chains)
    n = min(len(c) for c in chains) if chains else 0
    if m < 2 or n < 2:
        return None
    trimmed = [list(c)[:n] for c in chains]
    means = [statistics.fmean(c) for c in trimmed]
    grand = statistics.fmean(means)
    b = n * sum((mu - grand) ** 2 for mu in means) / (m - 1)
    w = statistics.fmean(statistics.variance(c) for c in trimmed)
    if w == 0:
        return None
    return math.sqrt((((n - 1) / n) * w + ((m + 1) / (m * n)) * b) / w)


def _recompute_parameters(
    p_files: Sequence[Path], burninfrac: float, columns: Sequence[str] = ("LnL", "TL{all}")
) -> dict[str, Any]:
    runs: list[dict[str, list[float]]] = []
    for path in p_files:
        rows = _read_table(path)
        burn = int(burninfrac * len(rows))
        kept = rows[burn:]
        data: dict[str, list[float]] = {}
        for column in columns:
            key = "lnLike" if column == "LnL" else column
            data[column] = [v for v in (_float(r.get(key)) for r in kept) if v is not None]
        runs.append(data)
    out: dict[str, Any] = {}
    for column in columns:
        chains = [run[column] for run in runs if run.get(column)]
        if not chains:
            continue
        ess = [effective_sample_size(c) for c in chains]
        value = psrf(chains)
        out[column] = {
            "min_ess": round(min(ess), 1),
            "avg_ess": round(statistics.fmean(ess), 1),
            "psrf": round(value, 4) if value is not None else None,
        }
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_mrbayes(
    alignment_fasta: str | Path,
    scheme_nexus: str | Path,
    *,
    output_dir: str | Path,
    models: Mapping[str, str] | None = None,
    default_model: str = DEFAULT_MODEL,
    ngen: int = 1_000_000,
    samplefreq: int = 1000,
    printfreq: int = 10_000,
    diagnfreq: int = 10_000,
    nruns: int = 2,
    nchains: int = 4,
    temp: float = 0.1,
    burninfrac: float = 0.25,
    seed: int = 42,
    stoprule: bool = False,
    stopval: float = 0.01,
    mpi: bool = True,
    mpi_processes: int | None = None,
    asdsf_threshold: float = 0.01,
    psrf_tolerance: float = 0.02,
    min_ess: float = 200.0,
    mb_bin: str | None = None,
    timeout: float | None = None,
    dry_run: bool = False,
) -> OrganelleResult:
    """Run MrBayes on a partitioned alignment and diagnose convergence.

    Convergence requires all three of: final ASDSF (MrBayes ``sumt``) <
    ``asdsf_threshold`` (0.01); every parameter's PSRF in ``.pstat`` within
    ``1 +/- psrf_tolerance``; every parameter's ``minESS`` >= ``min_ess``
    (200). Otherwise the result has ``status="warning"``, the flag
    ``mcmc_not_converged`` and the failed criteria in
    ``metrics["convergence"]["failed_criteria"]`` - increase ``ngen``.

    With the MPI build (``mb-mpi``), MrBayes is launched as ``mpirun -np
    <nruns*nchains> mb-mpi file.nex`` (one chain per process). Outputs under
    ``output_dir``: ``mrbayes.nex`` (+ MrBayes' ``.run*.p/.t``, ``.mcmc``,
    ``.pstat``, ``.tstat``, ``.con.tre``), ``mrbayes.log`` (stdout),
    ``consensus.newick`` (majority-rule consensus, posterior probabilities as
    internal labels) and ``diagnostics.tsv``.
    """
    out = Path(output_dir)
    params = {
        "alignment_fasta": str(alignment_fasta),
        "scheme_nexus": str(scheme_nexus),
        "models": dict(models or {}),
        "default_model": default_model,
        "ngen": ngen,
        "samplefreq": samplefreq,
        "printfreq": printfreq,
        "diagnfreq": diagnfreq,
        "nruns": nruns,
        "nchains": nchains,
        "temp": temp,
        "burninfrac": burninfrac,
        "seed": seed,
        "stoprule": stoprule,
        "stopval": stopval,
        "mpi": mpi,
        "mpi_processes": mpi_processes,
        "asdsf_threshold": asdsf_threshold,
        "psrf_tolerance": psrf_tolerance,
        "min_ess": min_ess,
    }
    resolved = resolve_mrbayes(mb_bin, mpi=mpi)
    nexus_path = out / "mrbayes.nex"
    if resolved is None:
        argv = ["mb", str(nexus_path)]
    elif resolved["mpi"]:
        processes = mpi_processes or nruns * nchains
        argv = [resolved["mpirun"], "-np", str(processes)]
        if processes > (os.cpu_count() or 1):
            argv.append("--oversubscribe")
        argv += [resolved["mb"], nexus_path.name]
    else:
        argv = [resolved["mb"], nexus_path.name]
    prov = provenance(
        "run_mrbayes",
        method="mrbayes-mpi" if resolved and resolved["mpi"] else "mrbayes",
        argv=argv,
        parameters=params,
        random_seed=seed,
    )
    if dry_run:
        return ok_result(
            "run_mrbayes",
            metrics={"argv": argv, "backend_found": resolved is not None, "planned": True},
            flags=("mrbayes_planned",) + (() if resolved else ("backend_missing",)),
            summary_text="MrBayes run planned." + ("" if resolved else " " + MRBAYES_INSTALL_HINT),
            result_provenance=prov,
        )
    if resolved is None:
        raise OrganelleDependencyError(
            code="phylogeny.mrbayes.missing", message=MRBAYES_INSTALL_HINT
        )

    setup = write_mrbayes_nexus(
        alignment_fasta,
        scheme_nexus,
        nexus_path,
        models=models,
        default_model=default_model,
        ngen=ngen,
        samplefreq=samplefreq,
        printfreq=printfreq,
        diagnfreq=diagnfreq,
        nruns=nruns,
        nchains=nchains,
        temp=temp,
        burninfrac=burninfrac,
        seed=seed,
        stoprule=stoprule,
        stopval=stopval,
    )
    env = dict(os.environ)
    # Open MPI's vader CMA single-copy path is denied in many containers.
    env.setdefault("OMPI_MCA_btl_vader_single_copy_mechanism", "none")
    started = time.monotonic()
    try:
        completed = run_external(
            argv,
            cwd=out,
            env=env,
            timeout=timeout,
            tool="mrbayes",
            code="phylogeny.mrbayes.failed",
        )
    except OrganelleExecutionError as error:
        return failed_result(
            "run_mrbayes",
            summary_text=error.message,
            code=error.code,
            anomalies=["mrbayes_failed"],
            details={
                "stderr_tail": str(error.details.get("stderr_tail", ""))[-4000:],
                "stdout_tail": str(error.details.get("stdout_tail", ""))[-4000:],
            },
            result_provenance=prov,
        )
    runtime = time.monotonic() - started
    log_path = out / "mrbayes.log"
    log_path.write_text(completed.stdout + ("\n" + completed.stderr if completed.stderr else ""))
    reported = _parse_stdout(completed.stdout)

    stem = nexus_path.name
    pstat_rows = _read_table(out / f"{stem}.pstat") if (out / f"{stem}.pstat").is_file() else []
    parameters = []
    for row in pstat_rows:
        parameters.append(
            {
                "parameter": row.get("Parameter", ""),
                "mean": _float(row.get("Mean")),
                "median": _float(row.get("Median")),
                "hpd95_lower": _float(row.get("Lower")),
                "hpd95_upper": _float(row.get("Upper")),
                "min_ess": _float(row.get("minESS")),
                "avg_ess": _float(row.get("avgESS")),
                "psrf": _float(row.get("PSRF")),
            }
        )
    trace = _parse_mcmc_trace(out / f"{stem}.mcmc")
    con_path = out / f"{stem}.con.tre"
    if not con_path.is_file():
        return failed_result(
            "run_mrbayes",
            summary_text="MrBayes finished without a consensus tree (.con.tre).",
            code="phylogeny.mrbayes.no_consensus",
            anomalies=["missing_consensus"],
            details={"stdout_tail": completed.stdout[-4000:]},
            result_provenance=prov,
        )
    consensus = read_tree(con_path)
    consensus_newick = out / "consensus.newick"
    consensus_newick.write_text(consensus.strip() + "\n")
    _, clades = newick_clades(consensus)
    _, splits = tree_splits(consensus)
    posteriors = [p for p in (_float(label) for label in splits.values()) if p is not None]
    del clades

    t_files = [out / f"{stem}.run{i}.t" for i in range(1, nruns + 1)]
    p_files = [out / f"{stem}.run{i}.p" for i in range(1, nruns + 1)]
    recomputed: dict[str, Any] = {}
    if all(p.is_file() for p in t_files):
        recomputed["split_frequencies"] = recompute_asdsf(t_files, burninfrac=burninfrac)
    if all(p.is_file() for p in p_files):
        recomputed["parameters"] = _recompute_parameters(p_files, burninfrac)

    asdsf = reported.get("asdsf")
    if asdsf is None and trace:
        asdsf = trace[-1][1]
    psrf_values = [p["psrf"] for p in parameters if p["psrf"] is not None]
    ess_values = [p["min_ess"] for p in parameters if p["min_ess"] is not None]
    max_psrf_dev = max((abs(v - 1.0) for v in psrf_values), default=None)
    min_ess_value = min(ess_values, default=None)
    failed: list[str] = []
    if asdsf is None or asdsf >= asdsf_threshold:
        failed.append(f"asdsf={asdsf} (threshold < {asdsf_threshold})")
    if max_psrf_dev is None or max_psrf_dev > psrf_tolerance:
        failed.append(f"max |PSRF-1|={max_psrf_dev} (tolerance {psrf_tolerance})")
    if min_ess_value is None or min_ess_value < min_ess:
        failed.append(f"min ESS={min_ess_value} (threshold {min_ess})")
    converged = not failed

    diagnostics_tsv = out / "diagnostics.tsv"
    diagnostics_tsv.write_text(
        "parameter\tmean\tmedian\thpd95_lower\thpd95_upper\tmin_ess\tavg_ess\tpsrf\n"
        + "".join(
            "\t".join(
                str(p[k])
                for k in (
                    "parameter",
                    "mean",
                    "median",
                    "hpd95_lower",
                    "hpd95_upper",
                    "min_ess",
                    "avg_ess",
                    "psrf",
                )
            )
            + "\n"
            for p in parameters
        )
    )
    convergence = {
        "converged": converged,
        "failed_criteria": failed,
        "asdsf": asdsf,
        "max_sdsf": reported.get("max_sdsf"),
        "asdsf_threshold": asdsf_threshold,
        "avg_psrf": reported.get("avg_psrf"),
        "max_psrf": reported.get("max_psrf"),
        "max_abs_psrf_minus_1": max_psrf_dev,
        "psrf_tolerance": psrf_tolerance,
        "min_ess": min_ess_value,
        "min_ess_threshold": min_ess,
        "asdsf_trace": [[g, v] for g, v in trace],
    }
    artifacts = tuple(
        a
        for a in (
            artifact_for(consensus_newick, kind="phylogenetic_tree", format="newick"),
            artifact_for(con_path, kind="phylogenetic_tree", format="nexus"),
            artifact_for(diagnostics_tsv, kind="table", format="tsv"),
            artifact_for(nexus_path, kind="analysis_input", format="nexus"),
            artifact_for(log_path, kind="log", format="text"),
        )
        if a is not None
    )
    summary = (
        f"MrBayes {ngen} gen x {nruns} runs x {nchains} chains on {setup['n_partitions']} "
        f"partitions in {runtime:.0f} s: ASDSF={asdsf}, max|PSRF-1|={max_psrf_dev}, "
        f"min ESS={min_ess_value} -> "
        + ("converged." if converged else "NOT converged (" + "; ".join(failed) + ").")
    )
    return ok_result(
        "run_mrbayes",
        status="ok" if converged else "warning",
        metrics={
            "convergence": convergence,
            "parameters": parameters,
            "posterior_probabilities": _posterior_summary(posteriors),
            "recomputed_diagnostics": recomputed,
            "setup": setup,
            "runtime_seconds": round(runtime, 2),
            "mpi": resolved["mpi"],
            "argv": argv,
            "mrbayes_version": reported.get("mrbayes_version"),
            "consensus_newick": str(consensus_newick),
            "consensus_nexus": str(con_path),
        },
        result_findings=findings(
            ("converged", converged),
            ("asdsf", asdsf),
            ("min_ess", min_ess_value),
            ("runtime_seconds", round(runtime, 2)),
        ),
        flags=("tree_built", "mcmc_converged" if converged else "mcmc_not_converged"),
        artifacts=artifacts,
        summary_text=summary,
        result_provenance=prov,
    )
