"""Morphology suite with managed segmentation and training runs."""

from __future__ import annotations

from .correction import correct
from .measure import ORGSEG_CLASSES, ORGSEG_PALETTE, measure, overlay, summarize, write_csv
from .train import training_code_snippet, validate_dataset
from .verify import verify
from .webapp import build_app, serve
from .service import segment, train

__all__ = [
    "ORGSEG_CLASSES",
    "ORGSEG_PALETTE",
    "build_app",
    "correct",
    "measure",
    "overlay",
    "segment",
    "serve",
    "summarize",
    "train",
    "training_code_snippet",
    "validate_dataset",
    "verify",
    "write_csv",
]
