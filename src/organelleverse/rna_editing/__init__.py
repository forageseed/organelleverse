"""RNA editing suite: predict_edits(), neural backends, validation, and writers."""

from __future__ import annotations

from ..inputs import make_reader
from .detection import detect_editing_sites
from .efficiency import quantify_known_site_efficiency
from .rna_edit import (
    predict_edits,
    predict_edits_deepred,
    predict_edits_plantc2u,
    validate_edits,
    predict_edits_prep,
    write_sites,
)

read = make_reader("rna_editing", "bam", "fasta", genome=True)

__all__ = [
    "read",
    "detect_editing_sites",
    "predict_edits",
    "predict_edits_deepred",
    "predict_edits_plantc2u",
    "validate_edits",
    "predict_edits_prep",
    "quantify_known_site_efficiency",
    "write_sites",
]
