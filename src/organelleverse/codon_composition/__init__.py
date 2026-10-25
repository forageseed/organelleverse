"""Codon composition suite: codon_usage(), RSCU, ENC, and amino-acid usage."""

from __future__ import annotations

from ..inputs import make_reader
from .codon import amino_acid, codon_usage, write_amino_acid, write_usage
from .codon_core import compute_codon_usage, compute_amino_acid_composition

read = make_reader("codon_composition", "fasta")

__all__ = [
    "read",
    "codon_usage",
    "amino_acid",
    "write_usage",
    "write_amino_acid",
    "compute_codon_usage",
    "compute_amino_acid_composition",
]
