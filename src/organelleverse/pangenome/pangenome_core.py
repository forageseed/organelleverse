"""Typed cores for pangenome (OmicVerse-style data-in/data-out)."""

from __future__ import annotations

from .pangenome import _gene_pav_metrics


def compute_gene_pav(genomes) -> dict:
    """Named CDS-annotation PAV with mutually exclusive core/shell/cloud.

    Identical scientific policy to gene_pav: GenBank CDS /gene identifiers,
    missing annotations rejected, zeros do not prove biological absence.
    """
    return _gene_pav_metrics(genomes)
