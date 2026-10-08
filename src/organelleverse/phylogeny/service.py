"""Managed entry points for tree-building tools."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..core.result import OrganelleResult
from ..runtime import managed_run_path
from .dating import Calibration, MCMCTreeOptions


def run_trim_alignment(
    alignment_fasta: str | Path,
    *,
    method: str = "automated1",
    executor: Callable[[list[str]], Any] | None = None,
) -> OrganelleResult:
    """Plan or run trimAl with intermediate files in managed storage."""
    from .phylo import run_trim_alignment as _run

    return _run(
        alignment_fasta,
        output_dir=managed_run_path("phylogeny.run_trim_alignment", uuid4().hex),
        method=method,
        executor=executor,
    )


def run_build_tree(
    alignment_fasta: str | Path,
    *,
    method: str = "iqtree",
    bootstrap: int = 1000,
    seed: int = 42,
    threads: int = 1,
    model: str | None = None,
    executor: Callable[[list[str]], Any] | None = None,
) -> OrganelleResult:
    """Plan or run tree inference with intermediate files in managed storage."""
    from .phylo import run_build_tree as _run

    return _run(
        alignment_fasta,
        method=method,
        output_dir=managed_run_path("phylogeny.run_build_tree", uuid4().hex),
        bootstrap=bootstrap,
        seed=seed,
        threads=threads,
        model=model,
        executor=executor,
    )


def build_partitioned_supermatrix(
    genbank_paths: list[str | Path],
    *,
    genes: list[str] | None = None,
    taxon_names: list[str] | None = None,
    codon_positions: bool = True,
    min_taxa_fraction: float = 1.0,
    min_codons: int = 30,
    max_length_deviation: float | None = 0.2,
    genetic_code: int = 11,
) -> OrganelleResult:
    """Build a codon-aligned, gene-partitioned CDS supermatrix in managed storage."""
    from .partition import build_partitioned_supermatrix as _run

    return _run(
        genbank_paths,
        output_dir=managed_run_path("phylogeny.build_partitioned_supermatrix", uuid4().hex),
        genes=genes,
        taxon_names=taxon_names,
        codon_positions=codon_positions,
        min_taxa_fraction=min_taxa_fraction,
        min_codons=min_codons,
        max_length_deviation=max_length_deviation,
        genetic_code=genetic_code,
    )


def select_partition_scheme(
    alignment_fasta: str | Path,
    partition_nexus: str | Path,
    *,
    merge: bool = True,
    model_set: str | None = "mrbayes",
    rcluster: int = 100,
    rcluster_fast: bool = True,
    rcluster_max: int | None = None,
    compare_codon_positions: bool = False,
    bootstrap: int = 0,
    seed: int = 42,
    threads: int = 1,
    dry_run: bool = False,
) -> OrganelleResult:
    """Select a scheme with IQ-TREE MFP+MERGE and fast relaxed clustering.

    For CDS alignments in frame from column 1, ``compare_codon_positions=True``
    additionally reports BIC against three fixed codon-position partitions.
    See :func:`organelleverse.phylogeny.partition.select_partition_scheme`.
    """
    from .partition import select_partition_scheme as _run

    return _run(
        alignment_fasta,
        partition_nexus,
        output_dir=managed_run_path("phylogeny.select_partition_scheme", uuid4().hex),
        merge=merge,
        model_set=model_set,
        rcluster=rcluster,
        rcluster_fast=rcluster_fast,
        rcluster_max=rcluster_max,
        compare_codon_positions=compare_codon_positions,
        bootstrap=bootstrap,
        seed=seed,
        threads=threads,
        dry_run=dry_run,
    )


def run_mrbayes(
    alignment_fasta: str | Path,
    scheme_nexus: str | Path,
    *,
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
    asdsf_threshold: float = 0.01,
    psrf_tolerance: float = 0.02,
    min_ess: float = 200.0,
    dry_run: bool = False,
) -> OrganelleResult:
    """Run MrBayes with convergence diagnostics; files go to managed storage."""
    from .mrbayes import run_mrbayes as _run

    return _run(
        alignment_fasta,
        scheme_nexus,
        output_dir=managed_run_path("phylogeny.run_mrbayes", uuid4().hex),
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
        mpi=mpi,
        asdsf_threshold=asdsf_threshold,
        psrf_tolerance=psrf_tolerance,
        min_ess=min_ess,
        dry_run=dry_run,
    )


def date_tree(
    tree_newick: str | Path,
    alignment_fasta: str | Path,
    calibrations: list[Calibration],
    *,
    backend: str = "iqtree_lsd2",
    mcmctree_options: MCMCTreeOptions | None = None,
    outgroup: list[str] | None = None,
    model: str = "GTR+G4",
    partition_nexus: str | Path | None = None,
    ci_replicates: int = 100,
    clock_sd: float = 0.2,
    seed: int = 42,
    threads: int = 1,
    dry_run: bool = False,
) -> OrganelleResult:
    """Estimate ages with LSD2 or MCMCTree in managed storage."""
    from .dating import date_tree as _run

    return _run(
        tree_newick,
        alignment_fasta,
        calibrations,
        output_dir=managed_run_path("phylogeny.date_tree", uuid4().hex),
        backend=backend,
        mcmctree_options=mcmctree_options,
        outgroup=outgroup,
        model=model,
        partition_nexus=partition_nexus,
        ci_replicates=ci_replicates,
        clock_sd=clock_sd,
        seed=seed,
        threads=threads,
        dry_run=dry_run,
    )
