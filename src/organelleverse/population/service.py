"""Managed public entry points for population-genetics tools."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..core.result import OrganelleResult
from ..runtime import managed_run_path


def call_variants(
    bam_dir: str | Path,
    *,
    ref_path: str | Path | None = None,
    method: str = "deepvariant",
    executor: Callable[[list[str]], Any] | None = None,
    ploidy: int = 1,
) -> OrganelleResult:
    """Plan or run variant calling with output in a managed run directory.

    ``ploidy`` defaults to 1 (organelles are haploid) and is forwarded to the
    backend's ploidy switch.
    """
    from .population import call_variants as _run

    output = managed_run_path("population.call_variants", uuid4().hex)
    if executor is not None:
        output.mkdir(parents=True, exist_ok=True)
    return _run(
        bam_dir,
        ref_path=ref_path,
        method=method,
        output_dir=output,
        executor=executor,
        ploidy=ploidy,
    )


def prepare_gemma_input(
    vcf_path: str | Path,
    *,
    phenotype: str | Path,
    executor: Callable[[list[str]], Any] | None = None,
) -> OrganelleResult:
    """Plan or run GEMMA input preparation in a managed run directory."""
    from .population import prepare_gemma_input as _run

    return _run(
        vcf_path,
        phenotype=phenotype,
        out_dir=managed_run_path("population.prepare_gemma_input", uuid4().hex),
        executor=executor,
    )


def cytonuclear_gwas(
    vcf_path: str | Path,
    *,
    phenotype: str | Path,
    method: str = "gemma",
    executor: Callable[[list[str]], Any] | None = None,
    prepare_input: bool = True,
) -> OrganelleResult:
    """Plan or run cytonuclear association in a managed run directory."""
    from .population import cytonuclear_gwas as _run

    return _run(
        vcf_path,
        phenotype=phenotype,
        method=method,
        output_prefix="cyto_gwas",
        out_dir=managed_run_path("population.cytonuclear_gwas", uuid4().hex),
        executor=executor,
        prepare_input=prepare_input,
    )
