"""Trans-splicing suite: detect trans-spliced genes. Self-contained."""

from __future__ import annotations

from .trans_splicing import detect_trans_splicing
from .trans_splicing_core import compute_trans_splicing

__all__ = ["compute_trans_splicing", "detect_trans_splicing"]
