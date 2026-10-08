"""Comparative genomics suite.

Self-contained: synteny (gene-order comparison), gene presence comparison,
gene content tables, genome structure comparison, gene-name normalization,
and mVISTA-style whole-genome sequence identity against one reference
(mappy anchoring + edlib global block alignment). All pure Python over
GenBank-parsed features.
"""

from __future__ import annotations

from .compare import (
    synteny,
    compare_genes,
    gene_table,
    compare_genomes,
    normalize_genes,
    write_gene_comparison,
    write_gene_table,
    write_genome_comparison,
    write_synteny,
)

from .compare_core import (
    compute_gene_sets,
    compute_gene_intersection,
    normalize_gene_names,
    compute_synteny_score,
)

from .identity import (
    compute_genome_identity,
    load_reference_features,
)

from .orientation import normalize_plastome_orientation
from .structural import detect_structural_variants

__all__ = [
    "normalize_plastome_orientation",
    "detect_structural_variants",
    "synteny",
    "compare_genes",
    "gene_table",
    "compare_genomes",
    "normalize_genes",
    "write_gene_table",
    "write_synteny",
    "write_gene_comparison",
    "write_genome_comparison",
    "compute_gene_sets",
    "compute_gene_intersection",
    "normalize_gene_names",
    "compute_synteny_score",
    "compute_genome_identity",
    "load_reference_features",
]
