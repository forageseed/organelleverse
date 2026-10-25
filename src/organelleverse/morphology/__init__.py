"""Morphology suite with managed segmentation and training runs."""

from __future__ import annotations

from ..inputs import make_reader

from .batch import batch_measure_dir
from .correction import correct
from .measure import ORGSEG_CLASSES, ORGSEG_PALETTE, measure, overlay, summarize, write_csv
from .train import training_code_snippet, validate_dataset
from .verify import verify
from .webapp import build_app, serve
from .service import segment, train

read = make_reader("morphology", "image")

__all__ = [
    "read",
    "ORGSEG_CLASSES",
    "ORGSEG_PALETTE",
    "batch_measure_dir",
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
