"""Typed cores for phylogeny (OmicVerse-style data-in/data-out)."""

from __future__ import annotations

from pathlib import Path

from ..core.errors import OrganelleExecutionError
from .phylo import (
    _build_fasttree_argv,
    _build_iqtree_argv,
    _build_raxml_argv,
    align,
)


def compute_alignment(input_fasta: str | Path) -> dict:
    """Compute a real alignment with the same backend contract as align()."""
    result = align(input_fasta)
    if result.status != "ok":
        error = result.errors[0]
        raise OrganelleExecutionError(
            code=error.code,
            message=error.message,
            details=dict(error.details),
        )
    return {
        "sequences": result.metrics["alignment"],
        "n_sequences": result.metrics["n_sequences"],
        "aligned_length": result.metrics["aligned_length"],
        "method": result.metrics["method"],
    }


def plan_tree_build(
    alignment_fasta: str | Path,
    method: str = "iqtree",
    bootstrap: int = 1000,
    seed: int = 42,
    threads: int = 1,
) -> dict:
    """Plan a tree-building command with each tool's real interface. Returns dict."""
    bin_map = {"iqtree": "iqtree2", "raxml": "raxml-ng", "fasttree": "FastTree"}
    bin_name = bin_map.get(method, "iqtree2")
    if method == "raxml":
        argv = _build_raxml_argv(
            bin_name, str(alignment_fasta), "tree", bootstrap=bootstrap, seed=seed, threads=threads
        )
    elif method == "fasttree":
        argv = _build_fasttree_argv(bin_name, str(alignment_fasta), "tree.newick")
    else:
        argv = _build_iqtree_argv(
            bin_name, str(alignment_fasta), "tree", bootstrap=bootstrap, seed=seed, threads=threads
        )
    return {"argv": argv, "method": method}
