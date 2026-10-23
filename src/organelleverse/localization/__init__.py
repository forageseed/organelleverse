"""Localization suite: predict() — subcellular-localization prediction.

The DeepLoc/TargetP execution bridge (``predict``, aliased as ``localize``)
drives external CLIs into a caller ``output_dir``: this is restored, reviewed
historical Python surface (see ``FROZEN_PUBLIC_SURFACE_VIOLATIONS`` in
``tests/operations/test_output_boundary_public_surface.py``), not a new
compute-first design.
"""

from __future__ import annotations

from ..inputs import make_reader

from .localize import localize, predict
from .localize_core import COMPARTMENTS, LocalizationScores, predict_heuristic

read = make_reader("localization", "fasta")

__all__ = ["read", "predict", "localize", "COMPARTMENTS", "LocalizationScores", "predict_heuristic"]
