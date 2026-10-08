"""HGT suite: horizontal gene transfer detection. Self-contained."""

from __future__ import annotations
from .hgt import detect_hgt
from ..writer import write

detect = detect_hgt
save = write

from .hgt_core import compute_hgt

__all__ = ["detect", "detect_hgt", "compute_hgt", "write", "save"]
