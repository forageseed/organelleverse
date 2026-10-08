"""Shared mitochondrial CDS terminal-codon contract."""

from __future__ import annotations

STANDARD_START_CODONS = frozenset({"ATG"})
GENE_SPECIFIC_START_CODONS: dict[str, frozenset[str]] = {
    "mttb": frozenset({"ATA", "ATG", "GTG", "TTG"}),
    "rpl16": frozenset({"ATG", "GTG"}),
    "rps4": frozenset({"ACG", "ATG"}),
}


def allowed_start_codons(
    gene_name: str,
    *,
    allow_rna_editing: bool = False,
) -> set[str]:
    """Return the starts accepted consistently by repair and validation."""
    allowed = set(GENE_SPECIFIC_START_CODONS.get(gene_name.casefold(), STANDARD_START_CODONS))
    if allow_rna_editing:
        allowed.add("ACG")
    return allowed
