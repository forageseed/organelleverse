"""Coevolution suite (ERC — evolutionary rate covariation). Self-contained.

Two independent ERC methods:
  method="ercnet" — ERCnet BXB with explicit LCA/DLCpar reconciliation and edge-mean normalization
  method="erc"    — ERC 2.0 (nclark-lab, master node-pair paths, WLS, independent edges, Fisher and BH)

Both run through run_coevolution() or their direct entry points.
"""

from __future__ import annotations
from .coevolution import (
    reconcile_trees,
    extract_branch_lengths,
    rate_covariation,
    coevolution_network,
    phylogenomics,
    compartment_map_from_prefix,
)
from .service import run_orthofinder, run_coevolution
from .erc_engine import (
    # Typed cores (data-in → data-out, OmicVerse-style composability)
    project_paths,
    residual_matrix,
    correlation_matrix,
    fisher,
    perm_test,
    # Entry point
    run_erc,
    # Back-compat aliases
    erc_master_paths,
    erc_residual_matrix,
    erc_correlation_matrix,
    fisher_transform,
    perm_test_matrix,
)
from .writer import materialize_result, write

__all__ = [
    # ERCnet (method="ercnet")
    "reconcile_trees",
    "extract_branch_lengths",
    "rate_covariation",
    "coevolution_network",
    "phylogenomics",
    "run_orthofinder",
    "run_coevolution",
    "compartment_map_from_prefix",
    # ERC 2.0 typed cores (method="erc")
    "project_paths",
    "residual_matrix",
    "correlation_matrix",
    "fisher",
    "perm_test",
    "run_erc",
    # Back-compat
    "erc_master_paths",
    "erc_residual_matrix",
    "erc_correlation_matrix",
    "fisher_transform",
    "perm_test_matrix",
    # Canonical publication boundary
    "materialize_result",
    "write",
]
