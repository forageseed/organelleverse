"""RNA editing suite: predict_edits(), neural backends, validation, and writers."""

from __future__ import annotations
from .detection import detect_editing_sites
from .rna_edit import (
    predict_edits,
    predict_edits_deepred,
    predict_edits_plantc2u,
    validate_edits,
    predict_edits_prep,
    write_sites,
)

__all__ = [
    "detect_editing_sites",
    "predict_edits",
    "predict_edits_deepred",
    "predict_edits_plantc2u",
    "validate_edits",
    "predict_edits_prep",
    "write_sites",
]
