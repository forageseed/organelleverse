"""Phylogenetic multiple-sequence alignment, trimming, and tree building.

align(): multiple-sequence alignment via the self-contained Rust MAFFT port
  (preferred) or the local ``mafft`` CLI. Missing or failed backends fail clearly.
trim_alignment(): plan/run trimal to trim poorly-aligned columns
  (``-automated1`` heuristic is the ML-optimised default per the trimAl paper).
build_tree(): plan a publication-grade ML tree search with each backend's
  real CLI — IQ-TREE 3/2 (default; -m MFP model selection + ultrafast bootstrap
  + SH-aLRT), RAxML-NG (--all bootstrap), or FastTree (-nt -gtr). The argv
  follows each tool's documented interface and includes reproducibility
  seeds, bootstrap support, and proper output redirection.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .._bio import read_fasta
from ..core.errors import OrganelleExecutionError
from ..core.external import run_external
from ..core.result import OrganelleResult
from ._results import artifact_for, failed_result, findings, ok_result, provenance


def _which(tool: str) -> str | None:
    """Return the path of ``tool`` on PATH, else None."""
    import shutil

    return shutil.which(tool)


def align(
    input_fasta: str | Path,
    *,
    method: str = "auto",
) -> OrganelleResult:
    """Align orthologous sequences using one available, real backend.

    ``auto`` selects the Rust MAFFT kernel when available, otherwise the MAFFT
    CLI. ``rust`` and ``mafft`` require that specific backend. Backend failure
    is reported with its cause; raw input is never returned as an alignment.
    """
    params = {"method": method, "input_fasta": str(input_fasta)}
    if method not in {"auto", "rust", "mafft"}:
        return failed_result(
            "align",
            summary_text="method must be auto, rust, or mafft.",
            code="phylogeny.align.invalid_method",
            result_provenance=provenance("align", parameters=params),
        )
    try:
        seqs = read_fasta(Path(input_fasta))
    except ValueError as exc:
        if "FASTA contains no records:" not in str(exc):
            raise
        seqs = []
    if not seqs:
        return failed_result(
            "align",
            summary_text="No sequences to align.",
            code="phylogeny.align.no_sequences",
            anomalies=["empty"],
        )

    from .. import accel

    use_rust = method in {"auto", "rust"} and accel.mafft_align is not None
    mafft_bin = _which("mafft") if not use_rust and method != "rust" else None
    chosen = "rust_mafft" if use_rust else "mafft_cli"
    if not use_rust and mafft_bin is None:
        return failed_result(
            "align",
            summary_text=f"No alignment backend available for method={method!r}; install MAFFT or the Rust MAFFT kernel.",
            code="phylogeny.align.backend_missing",
            result_provenance=provenance("align", parameters=params),
        )
    argv = [] if use_rust else [mafft_bin, "--auto", "-"]
    prov = provenance("align", method=chosen, argv=argv, parameters=params)
    try:
        if use_rust:
            aligned = accel.mafft_align(seqs, "auto")
        else:
            proc = run_external(
                argv,
                input_data="".join(f">{n}\n{s}\n" for n, s in seqs),
                tool="mafft",
                code="phylogeny.align.execution_failed",
            )
            aligned = _parse_fasta_text(proc.stdout)
    except OrganelleExecutionError as exc:
        return failed_result(
            "align",
            summary_text=str(exc),
            code="phylogeny.align.execution_failed",
            details=exc.details,
            result_provenance=prov,
        )
    except (ValueError, RuntimeError) as exc:
        return failed_result(
            "align",
            summary_text=f"{chosen} failed: {exc}",
            code="phylogeny.align.execution_failed",
            result_provenance=prov,
        )
    if (
        not aligned
        or len(aligned) != len(seqs)
        or sorted(n for n, _ in aligned) != sorted(n for n, _ in seqs)
        or len({len(s) for _, s in aligned}) != 1
        or not aligned[0][1]
        or sorted((n, s.upper().replace("-", "")) for n, s in aligned)
        != sorted((n, s.upper().replace("-", "")) for n, s in seqs)
    ):
        return failed_result(
            "align",
            summary_text=f"{chosen} did not return an equal-length alignment preserving the input records.",
            code="phylogeny.align.invalid_output",
            result_provenance=prov,
        )

    return ok_result(
        "align",
        metrics={
            "n_sequences": len(aligned),
            "method": chosen,
            "aligned_length": (len(aligned[0][1]) if aligned else 0),
            "alignment": aligned,
        },
        result_findings=findings(("n_sequences", len(aligned)), ("method", chosen)),
        flags=("alignment_ready",),
        summary_text=f"Aligned {len(aligned)} sequences ({chosen}).",
        result_provenance=prov,
    )


def write_alignment(result: OrganelleResult, output_path: str | Path) -> Path:
    """Write an alignment produced by ``align()`` to FASTA."""
    if result.status != "ok" or result.operation_id != "phylogeny.align":
        raise ValueError("write_alignment() requires an ok result produced by align().")
    aligned = result.metrics.get("alignment")
    if not isinstance(aligned, (list, tuple)):
        raise ValueError("write_alignment() requires alignment records in metrics.")
    p = Path(output_path)
    if p.suffix == "":
        p.mkdir(parents=True, exist_ok=True)
        p = p / "aligned.fasta"
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f">{name}\n{seq}\n" for name, seq in aligned))
    return p


def _parse_fasta_text(text: str) -> list[tuple[str, str]]:
    """Parse FASTA text into [(name, seq), ...]."""
    name = None
    chunks: list[str] = []
    out = []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(chunks)))
            name = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if name is not None:
        out.append((name, "".join(chunks)))
    return out


def trim_alignment(
    alignment_fasta: str | Path,
    *,
    method: str = "automated1",
    executor: Callable[[list[str]], Any] | None = None,
) -> OrganelleResult:
    """Trim an alignment to remove poorly-aligned columns (trimAl).

    Per the trimAl paper (Capella-Gutiérrez et al. 2009) ``-automated1`` is the
    heuristic recommended for downstream Maximum Likelihood phylogenetics
    (it picks the best of strict/gappyout/strictplus). ``-gappyout`` is faster
    and only uses gap-distribution; ``-nogaps`` removes every column with any
    gap.

    Plans the real ``trimal -in fa -out out.fa -automated1 -fasta`` command
    and runs it via ``executor`` when trimal is installed.
    """
    return _trim_alignment_impl(
        alignment_fasta,
        method=method,
        executor=executor,
    )


def run_trim_alignment(
    alignment_fasta: str | Path,
    *,
    output_dir: str | Path,
    method: str = "automated1",
    executor: Callable[[list[str]], Any] | None = None,
) -> OrganelleResult:
    """Plan or run trimAl with an explicit workflow output directory."""
    return _trim_alignment_impl(
        alignment_fasta,
        output_dir=output_dir,
        method=method,
        executor=executor,
    )


def _trim_alignment_impl(
    alignment_fasta: str | Path,
    *,
    output_dir: str | Path | None = None,
    method: str,
    executor: Callable[[list[str]], Any] | None,
) -> OrganelleResult:
    trimal_bin = _which("trimal")
    found = trimal_bin is not None
    if not found:
        trimal_bin = "trimal"

    out = Path(output_dir) if output_dir else Path(".")
    if output_dir and executor is not None:
        out.mkdir(parents=True, exist_ok=True)
    trimmed = str(out / "trimmed.fa")

    method_flag = {
        "automated1": "-automated1",
        "gappyout": "-gappyout",
        "strict": "-strict",
        "strictplus": "-strictplus",
        "nogaps": "-nogaps",
        "noallgaps": "-noallgaps",
    }.get(method, "-automated1")
    argv = [trimal_bin, "-in", str(alignment_fasta), "-out", trimmed, method_flag, "-fasta"]

    ran = False
    if executor is not None and found:
        try:
            executor(list(argv))
            ran = True
        except Exception as exc:
            return failed_result(
                "trim_alignment",
                summary_text=f"executor failed: {exc}",
                code="phylogeny.trim_alignment.executor_failed",
                anomalies=[f"executor_error:{type(exc).__name__}"],
                details={"exception_type": type(exc).__name__},
                result_provenance=provenance(
                    "trim_alignment",
                    method=method,
                    argv=argv,
                    parameters={"alignment_fasta": str(alignment_fasta), "method": method},
                ),
            )

    artifact = artifact_for(trimmed, kind="alignment", format="fasta") if ran else None
    hint = "" if found else "\n  ⚠ trimal not on PATH; install: conda install -c bioconda trimal"
    return ok_result(
        "trim_alignment",
        metrics={
            "method": method,
            "backend_found": found,
            "output": trimmed,
            # ``argv`` was a pre-v1 key_finding; canonical Finding values are
            # scalars, so the planned command line lives here and in provenance.
            "argv": list(argv),
        },
        result_findings=findings(("method", method)),
        flags=("alignment_trimmed",) if ran else ("trim_planned",),
        artifacts=(artifact,) if artifact is not None else (),
        summary_text=(f"trimAl ({method}); {'ran' if ran else 'planned'}.") + hint,
        result_provenance=provenance(
            "trim_alignment",
            method=method,
            argv=argv,
            parameters={"alignment_fasta": str(alignment_fasta), "method": method},
        ),
    )


def _build_iqtree_argv(
    bin_path: str,
    alignment: str,
    out_prefix: str,
    *,
    bootstrap: int = 1000,
    alrt: int = 1000,
    seed: int = 42,
    threads: int = 1,
    model: str = "MFP",
    v3: bool = False,
) -> list[str]:
    """IQ-TREE publication-grade command (v2 or v3).

    ``-m MFP`` runs ModelFinder Plus (auto-selects best-fit substitution model);
    ``-B 1000`` ultrafast bootstrap (UFBoot2; replaces v1's -bb); ``-alrt 1000``
    SH-aLRT branch support; ``-seed`` for reproducibility; ``-T`` threads.
    IQ-TREE 3 uses ``--prefix`` and ``--seed`` (long forms); v2 uses ``-pre``
    and ``-seed``.
    """
    prefix_flag = "--prefix" if v3 else "-pre"
    seed_flag = "--seed" if v3 else "-seed"
    return [
        bin_path,
        "-s",
        alignment,
        prefix_flag,
        out_prefix,
        "-m",
        model,
        "-B",
        str(bootstrap),
        "-alrt",
        str(alrt),
        seed_flag,
        str(seed),
        "-T",
        str(threads),
    ]


def _build_raxml_argv(
    bin_path: str,
    alignment: str,
    out_prefix: str,
    *,
    bootstrap: int = 1000,
    seed: int = 42,
    threads: int = 1,
    model: str = "GTR+G",
) -> list[str]:
    """RAxML-NG all-in-one (ML search + transfer bootstrap) command.

    ``--all`` runs ML search + bootstrap + support computation in one go;
    ``--model GTR+G`` (general time-reversible + gamma) is the standard DNA
    model; ``--seed`` + ``--threads`` for reproducibility.
    """
    return [
        bin_path,
        "--all",
        "--msa",
        alignment,
        "--model",
        model,
        "--prefix",
        out_prefix,
        "--seed",
        str(seed),
        "--threads",
        str(threads),
        "--bootstrap",
        str(bootstrap),
    ]


def _build_fasttree_argv(
    bin_path: str,
    alignment: str,
    out_newick: str,
    *,
    gtr: bool = True,
) -> list[str]:
    """FastTree command. Input on stdin, Newick tree on stdout (shell redirect).

    ``-nt`` nucleotide mode; ``-gtr`` generalised time-reversible (recommended
    for DNA, replaces default JC); FastTree reads the alignment from stdin and
    writes the tree to stdout, so a shell ``>`` redirect captures the Newick.
    """
    flags = ["-nt"]
    if gtr:
        flags.append("-gtr")
    return [bin_path, *flags, "<", alignment, ">", out_newick]


def extract_shared_genes(
    genbank_paths: list[str | Path],
    *,
    feature_type: str = "CDS",
) -> OrganelleResult:
    """Extract genes shared across multiple annotated genomes (PhyloSuite-style).

    For each GenBank input, collects gene names of the given ``feature_type``
    (default ``CDS``). The shared set is the intersection across all genomes.
    Used to build a common-PCG matrix for phylogenomics (e.g. the 22 conserved
    mito PCGs in Rosaceae comparisons).

    Writes one FASTA per shared gene (concatenating one sequence per genome)
    via :func:`write_shared_genes`.
    """
    try:
        from Bio import SeqIO  # type: ignore
    except ImportError:
        return failed_result(
            "extract_shared_genes",
            summary_text="biopython required for extract_shared_genes.",
            code="phylogeny.extract_shared_genes.missing_biopython",
            anomalies=["missing_biopython"],
        )

    per_genome: list[set[str]] = []
    genome_records: list[dict[str, tuple[str, str]]] = []  # {gene: (seqid, protein/nt)}
    names: list[str] = []
    for gbk in genbank_paths:
        rec = next(SeqIO.parse(str(gbk), "genbank"))
        genes: set[str] = set()
        rec_map: dict[str, tuple[str, str]] = {}
        for feat in rec.features:
            if feat.type != feature_type:
                continue
            gene = (feat.qualifiers.get("gene") or feat.qualifiers.get("product", ["unknown"]))[0]
            if not gene or gene == "unknown":
                continue
            seq = (
                feat.qualifiers.get("translation", [str(feat.extract(rec.seq))])[0]
                if feature_type == "CDS"
                else str(feat.extract(rec.seq))
            )
            genes.add(gene)
            rec_map[gene] = (rec.id, seq)
        per_genome.append(genes)
        genome_records.append(rec_map)
        names.append(rec.id)

    if not per_genome:
        return failed_result(
            "extract_shared_genes",
            summary_text="No GenBank records parsed.",
            code="phylogeny.extract_shared_genes.no_records",
            anomalies=["empty"],
        )
    shared = set.intersection(*per_genome) if per_genome else set()

    shared_records: dict[str, list[tuple[str, str]]] = {}
    for gene in sorted(shared):
        shared_records[gene] = []
        for gmap in genome_records:
            if gene in gmap:
                shared_records[gene].append(gmap[gene])

    return ok_result(
        "extract_shared_genes",
        metrics={
            "n_genomes": len(genbank_paths),
            # ``shared_genes`` was also a pre-v1 key_finding; canonical Finding
            # values are scalars, so the gene list stays here only.
            "shared_genes": sorted(shared),
            "n_shared": len(shared),
            "per_genome_counts": [len(s) for s in per_genome],
            "shared_records": shared_records,
        },
        result_findings=findings(("n_shared_genes", len(shared))),
        flags=("shared_genes_extracted",) if shared else ("no_shared_genes",),
        summary_text=(
            f"{len(shared)} shared {feature_type} genes across {len(genbank_paths)} genomes."
        ),
        result_provenance=provenance(
            "extract_shared_genes",
            method="biopython",
            parameters={
                "genbank_paths": [str(p) for p in genbank_paths],
                "feature_type": feature_type,
            },
        ),
    )


def write_shared_genes(result: OrganelleResult, output_dir: str | Path) -> dict[str, Path]:
    """Write one FASTA per shared gene from ``extract_shared_genes()``."""
    if result.status != "ok" or result.operation_id != "phylogeny.extract_shared_genes":
        raise ValueError(
            "write_shared_genes() requires an ok result produced by extract_shared_genes()."
        )
    shared_records = result.metrics.get("shared_records")
    if not isinstance(shared_records, Mapping):
        raise ValueError("write_shared_genes() requires shared_records in metrics.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for gene, records in shared_records.items():
        fa = out / f"{gene}.fa"
        fa.write_text("\n".join(f">{sid}\n{seq}" for sid, seq in records) + "\n")
        paths[str(gene)] = fa
    return paths


def build_tree(
    alignment_fasta: str | Path,
    *,
    method: str = "iqtree",
    bootstrap: int = 1000,
    seed: int = 42,
    threads: int = 1,
    model: str | None = None,
    executor: Callable[[list[str]], Any] | None = None,
) -> OrganelleResult:
    """Build a publication-grade ML phylogenetic tree, or return its plan.

    Without an ``executor`` this returns the planned argv (status ok,
    ``backend_found`` tells whether the CLI was located) without running
    anything; pass a subprocess-style ``executor`` to build for real. The
    fasttree plan uses shell redirection and resolves ``tree.newick``
    against the process CWD, not the caller's intent.

    Per-method CLI (each backend's real interface):

    - **iqtree** (default): ``iqtree2 -s fa -pre out -m MFP -B 1000 -alrt 1000
      -seed 42 -T 1``. ModelFinder Plus + ultrafast bootstrap + SH-aLRT.
    - **raxml**: ``raxml-ng --all --msa fa --model GTR+G --prefix out --seed 42
      --threads 1 --bootstrap 1000``.
    - **fasttree**: ``FastTree -nt -gtr < fa > out.newick`` (stdin→stdout).

    ``model=None`` lets each tool pick its default (MFP for IQ-TREE, GTR+G for
    RAxML-NG). Auto-locates the backend; if not found, returns the plan with an
    install hint.

    ``method`` accepts ``iqtree3`` (preferred when installed, IQ-TREE 3 with
    ModelFinder Plus + ultrafast bootstrap + SH-aLRT) or ``iqtree``/``iqtree2``
    (IQ-TREE 2), ``raxml``, or ``fasttree``.
    """
    return _build_tree_impl(
        alignment_fasta,
        method=method,
        bootstrap=bootstrap,
        seed=seed,
        threads=threads,
        model=model,
        executor=executor,
    )


def run_build_tree(
    alignment_fasta: str | Path,
    *,
    method: str = "iqtree",
    output_dir: str | Path,
    bootstrap: int = 1000,
    seed: int = 42,
    threads: int = 1,
    model: str | None = None,
    executor: Callable[[list[str]], Any] | None = None,
) -> OrganelleResult:
    """Plan or run a tree-building workflow with an explicit output directory."""
    return _build_tree_impl(
        alignment_fasta,
        method=method,
        output_dir=output_dir,
        bootstrap=bootstrap,
        seed=seed,
        threads=threads,
        model=model,
        executor=executor,
    )


def _build_tree_impl(
    alignment_fasta: str | Path,
    *,
    method: str,
    output_dir: str | Path | None = None,
    bootstrap: int,
    seed: int,
    threads: int,
    model: str | None,
    executor: Callable[[list[str]], Any] | None,
) -> OrganelleResult:
    bin_map = {
        "iqtree": "iqtree2",
        "iqtree2": "iqtree2",
        "iqtree3": "iqtree3",
        "raxml": "raxml-ng",
        "fasttree": "FastTree",
    }
    bin_name = bin_map.get(method, "iqtree2")
    # Prefer iqtree3 if user asked for it; auto-fall back to iqtree2 if missing.
    bin_path = _which(bin_name)
    found = bin_path is not None
    if not found and bin_name == "iqtree3":
        # fall back to iqtree2 if iqtree3 not installed
        bin_name = "iqtree2"
        bin_path = _which(bin_name)
        found = bin_path is not None
        if method == "iqtree3":
            method = "iqtree2"
    bin_path = bin_path or bin_name

    out = Path(output_dir) if output_dir else Path(".")
    if output_dir and executor is not None:
        out.mkdir(parents=True, exist_ok=True)
    stem = str(out / "tree")

    if method == "raxml":
        argv = _build_raxml_argv(
            bin_path,
            str(alignment_fasta),
            stem,
            bootstrap=bootstrap,
            seed=seed,
            threads=threads,
            model=model or "GTR+G",
        )
        out_tree = stem + ".raxml.bestTree"
    elif method == "fasttree":
        argv = _build_fasttree_argv(bin_path, str(alignment_fasta), stem + ".newick")
        out_tree = stem + ".newick"
    else:  # iqtree2 or iqtree3
        argv = _build_iqtree_argv(
            bin_path,
            str(alignment_fasta),
            stem,
            bootstrap=bootstrap,
            seed=seed,
            threads=threads,
            model=model or "MFP",
            v3=(bin_name == "iqtree3"),
        )
        out_tree = stem + ".treefile"

    tree_parameters = {
        "alignment_fasta": str(alignment_fasta),
        "method": method,
        "bootstrap": bootstrap,
        "seed": seed,
        "threads": threads,
        "model": model,
    }

    ran = False
    if executor is not None and found:
        try:
            executor(list(argv))
            ran = True
        except Exception as exc:
            return failed_result(
                "build_tree",
                summary_text=f"executor failed: {exc}",
                code="phylogeny.build_tree.executor_failed",
                anomalies=[f"executor_error:{type(exc).__name__}"],
                details={"exception_type": type(exc).__name__},
                result_provenance=provenance(
                    "build_tree",
                    method=method,
                    argv=argv,
                    parameters=tree_parameters,
                    random_seed=seed,
                ),
            )

    tree_artifact = (
        artifact_for(out_tree, kind="phylogenetic_tree", format="newick")
        if (ran and output_dir)
        else None
    )
    flags = ("tree_built",) if ran else ("tree_planned",)
    if not found:
        flags += ("backend_missing",)
    hint = (
        ""
        if found
        else f"\n  ⚠ {bin_name} not on PATH; install via conda: conda install -c bioconda {bin_name}"
    )

    return ok_result(
        "build_tree",
        metrics={
            "method": method,
            "backend_found": found,
            "model": model or ("MFP" if method == "iqtree" else "GTR+G"),
            "bootstrap": bootstrap,
            "seed": seed,
            "threads": threads,
            # ``argv`` was a pre-v1 key_finding; canonical Finding values are
            # scalars, so the planned command line lives here and in provenance.
            "argv": list(argv),
            "output_tree": out_tree,
        },
        result_findings=findings(("method", method)),
        flags=flags,
        artifacts=(tree_artifact,) if tree_artifact is not None else (),
        summary_text=(f"Phylogenetic tree ({method}); {'built' if ran else 'planned'}.") + hint,
        result_provenance=provenance(
            "build_tree",
            method=method,
            argv=argv,
            parameters=tree_parameters,
            random_seed=seed,
        ),
    )
