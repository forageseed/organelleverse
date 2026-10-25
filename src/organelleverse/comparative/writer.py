"""Materialization of ``comparative`` results: each operation writes with its own ``write_*``."""

from __future__ import annotations

from ..materialize import make_materializer


def _writers():
    from .compare import write_gene_comparison, write_gene_table, write_genome_comparison, write_synteny

    return {
        "comparative.gene_table": write_gene_table,
        "comparative.synteny": write_synteny,
        "comparative.compare_genes": write_gene_comparison,
        "comparative.compare_genomes": write_genome_comparison,
    }


materialize_result = make_materializer("comparative", _writers)

__all__ = ["materialize_result"]
