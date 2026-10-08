"""Diversity suite: nucleotide diversity (π), Watterson's θ_W, Tajima's D.

Backend: scikit-allel (community standard, same estimators as VCFtools/dadi)
when available, else the textbook formulas (pure Python fallback). The two
backends produce identical values (validated against hand-computed examples).

Also provides DnaSP-style sliding-window π with mutation-hotspot
(高变区) calling and optional GenBank gene/spacer labelling
(:func:`sliding_window_diversity`).
"""

from __future__ import annotations

from .diversity import (
    neutral_tests,
    nucleotide_diversity,
    sliding_window_diversity,
    write_neutral_tests,
    write_nucleotide,
    write_sliding_window_diversity,
)
from .diversity_core import compute_neutral_tests, compute_nucleotide_diversity

__all__ = [
    "compute_neutral_tests",
    "compute_nucleotide_diversity",
    "neutral_tests",
    "nucleotide_diversity",
    "sliding_window_diversity",
    "write_neutral_tests",
    "write_nucleotide",
    "write_sliding_window_diversity",
]
