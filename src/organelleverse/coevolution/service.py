"""Managed public entry points for external coevolution workflows."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..core.result import OrganelleResult
from ..runtime import managed_run_path


def run_orthofinder(
    proteome_dir: str | Path,
    *,
    threads: int = 8,
    msa: bool = True,
    search: str = "diamond",
    executor: Callable[[list[str]], Any] | None = None,
) -> OrganelleResult:
    """Plan or run OrthoFinder with generated files in a managed run directory."""
    from .coevolution import run_orthofinder as _run

    output = managed_run_path("coevolution.run_orthofinder", uuid4().hex)
    if executor is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
    return _run(
        proteome_dir,
        output_dir=output,
        threads=threads,
        msa=msa,
        search=search,
        executor=executor,
    )


def run_coevolution(
    proteome_dir: str | Path,
    *,
    method: str = "ercnet",
    threads: int = 8,
    executor: Callable[[list[str]], Any] | None = None,
    results_dir: str | Path | None = None,
    transform: str = "sqrt",
    fdr_threshold: float = 0.05,
    r_threshold: float = 0.5,
) -> OrganelleResult:
    """Plan or run the coevolution workflow in a managed run directory."""
    from .coevolution import run_coevolution as _run

    return _run(
        proteome_dir,
        output_dir=managed_run_path("coevolution.run_coevolution", uuid4().hex),
        method=method,
        threads=threads,
        executor=executor,
        results_dir=results_dir,
        transform=transform,
        fdr_threshold=fdr_threshold,
        r_threshold=r_threshold,
    )
