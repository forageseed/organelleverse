"""Trans-splicing suite: detect trans-spliced genes. Self-contained."""

from __future__ import annotations

from ..inputs import make_reader

from .trans_splicing import detect_trans_splicing
from .trans_splicing_core import compute_trans_splicing

read = make_reader("trans_splicing", genome=True)

__all__ = ["read", "compute_trans_splicing", "detect_trans_splicing"]
