"""Lazy public entry points for annotation and assembly quality control."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from annotated_types import Ge, Le

from organelleverse.core.result import OrganelleResult

__all__ = [
    "annotation",
    "assembly",
    "compare_assembly_to_reference",
    "filter_long_reads",
    "filter_short_reads",
    "read_statistics",
    "write",
]


def annotation(result: OrganelleResult) -> OrganelleResult:
    """Audit one canonical annotation Result."""

    from .annotation_service import run_annotation_qc

    return run_annotation_qc(result)


def assembly(
    result: OrganelleResult,
    *,
    threads: Annotated[int, Ge(1), Le(256)] = 4,
    timeout_seconds: Annotated[int, Ge(1)] | None = None,
) -> OrganelleResult:
    """Run assembly quality control over one canonical assembly Result."""

    from .service import run_assembly_qc

    return run_assembly_qc(
        result,
        threads=threads,
        timeout_seconds=timeout_seconds,
    )


def write(result: OrganelleResult, *, output: Path) -> OrganelleResult:
    """Materialize a canonical QC Result to a user-facing bundle or file."""

    from .writer import materialize_result

    return materialize_result(result, output)


def read_statistics(reads: Path) -> OrganelleResult:
    """Summarize plain/gzip FASTQ lengths and base-weighted Phred+33 quality."""
    from .reads import read_statistics as run

    return run(reads)


def filter_long_reads(
    reads: Path,
    *,
    min_length: int = 1000,
    min_mean_quality: float = 0.0,
    target_bases: int | None = None,
) -> OrganelleResult:
    """Filter ONT/HiFi; keep quality-ranked whole reads up to/through a base target."""
    from .reads import filter_long_reads as run

    return run(
        reads, min_length=min_length, min_mean_quality=min_mean_quality, target_bases=target_bases
    )


def filter_short_reads(
    reads: Path,
    *,
    reads2: Path | None = None,
    min_length: int = 15,
    min_mean_quality: int = 0,
    threads: int = 2,
    adapter_sequence: str | None = None,
    adapter_sequence_r2: str | None = None,
) -> OrganelleResult:
    """Run fastp SE/PE QC; return parsed metrics and managed FASTQ/report paths."""
    from .reads import filter_short_reads as run

    return run(
        reads,
        reads2=reads2,
        min_length=min_length,
        min_mean_quality=min_mean_quality,
        threads=threads,
        adapter_sequence=adapter_sequence,
        adapter_sequence_r2=adapter_sequence_r2,
    )


def compare_assembly_to_reference(
    input_fasta: Path,
    reference_fasta: Path,
    *,
    organelle: Literal["plastid", "generic"] = "generic",
) -> OrganelleResult:
    """Compare each candidate record with one reference, retaining both strands.

    Plastids receive quadrant comparisons before/after orientation normalization.
    Generic sequences receive whole-sequence comparisons without normalization.
    """
    from .reference import compare_assembly_to_reference as run

    return run(input_fasta, reference_fasta, organelle=organelle)
