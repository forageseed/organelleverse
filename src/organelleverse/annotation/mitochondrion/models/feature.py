"""Non-coding RNA feature models."""

from __future__ import annotations

from .gene import GeneAnnotation


class tRNAAnnotation(GeneAnnotation):
    """tRNA gene annotation."""

    gene_type: str = "tRNA"
    anticodon: str  # e.g. "CAU"
    amino_acid: str  # e.g. "Met"
    is_cp_derived: bool = False  # Chloroplast origin
    trnascan_score: float | None = None
    aragorn_score: float | None = None
    source_tool: str = ""  # "tRNAscan-SE" | "ARAGORN" | "merged"
    has_intron: bool = False


class rRNAAnnotation(GeneAnnotation):
    """rRNA gene annotation."""

    gene_type: str = "rRNA"
    rrna_type: str  # "5S" | "18S" | "26S"
    source_tool: str = ""
