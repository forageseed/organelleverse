"""Format conversion suite: convert() — GenBank format conversion. Self-contained."""

from __future__ import annotations

from ..inputs import make_reader
from .convert import convert, write_conversion
from .convert_core import convert_genbank_to_gff3, convert_genbank_to_fasta

read = make_reader("format_conversion", genome=True)

__all__ = ["read", "convert", "write_conversion"]
