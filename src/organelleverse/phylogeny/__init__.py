"""Phylogeny suite: alignment, trimming, tree building, haplotype networks.

Also partitioned supermatrices, IQ-TREE partition-scheme/model selection
(``-m MFP+MERGE``) and MrBayes Bayesian inference with convergence
diagnostics. Tool runs use managed storage; figures are written explicitly.
"""

from __future__ import annotations

from .ancestral import gene_presence_traits, reconstruct_ancestral_states
from .dating import Calibration, MCMCTreeOptions
from .hapnet import (
    build_mjn,
    build_msn,
    build_tcs_network,
    collapse_haplotypes,
    haplotype_network,
    pdistance,
    render_network,
    tcs_connection_limit,
    write_haplotype_network,
)
from .mrbayes import write_mrbayes_nexus
from .partition import iqtree_to_mrbayes_model
from .phylo import (
    align,
    build_tree,
    extract_shared_genes,
    trim_alignment,
    write_alignment,
    write_shared_genes,
)
from .phylo_core import compute_alignment, plan_tree_build
from .service import (
    build_partitioned_supermatrix,
    date_tree,
    run_build_tree,
    run_mrbayes,
    run_trim_alignment,
    select_partition_scheme,
)
from .topology import test_topologies
from .treecompare import compare_trees, robinson_foulds

__all__ = [
    "Calibration",
    "MCMCTreeOptions",
    "align",
    "build_mjn",
    "build_msn",
    "build_partitioned_supermatrix",
    "build_tcs_network",
    "build_tree",
    "collapse_haplotypes",
    "compare_trees",
    "compute_alignment",
    "date_tree",
    "extract_shared_genes",
    "gene_presence_traits",
    "haplotype_network",
    "iqtree_to_mrbayes_model",
    "pdistance",
    "plan_tree_build",
    "reconstruct_ancestral_states",
    "render_network",
    "robinson_foulds",
    "run_build_tree",
    "run_mrbayes",
    "run_trim_alignment",
    "select_partition_scheme",
    "tcs_connection_limit",
    "test_topologies",
    "trim_alignment",
    "write_alignment",
    "write_haplotype_network",
    "write_mrbayes_nexus",
    "write_shared_genes",
]
