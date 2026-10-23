"""HGT suite: horizontal gene transfer detection. Self-contained."""

from __future__ import annotations

from ..writer import deprecated_alias

from ..inputs import make_reader
from .hgt import detect_hgt
from ..writer import write

detect = detect_hgt
save = deprecated_alias(write, "save")

from .hgt_core import compute_hgt

read = make_reader("hgt", "fasta")

__all__ = ["read", "detect", "detect_hgt", "compute_hgt", "write", "save"]
