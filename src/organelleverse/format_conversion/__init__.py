"""Format conversion suite: convert() — GenBank format conversion. Self-contained."""

from __future__ import annotations
from .convert import convert, write_conversion
from .convert_core import convert_genbank_to_gff3, convert_genbank_to_fasta

__all__ = ["convert", "write_conversion"]
