"""Typed cores for selection/CodeML (OmicVerse-style data-in/data-out)."""

from __future__ import annotations

from pathlib import Path

from organelleverse._public_validation import validated_public

from ..core.frozen import thaw_json
from ..sequences import Sequences, as_records
from .kaks import _pairwise_kaks


@validated_public('selection.compute_kaks')
def compute_kaks(cds_fasta: str | Path | Sequences) -> dict:
    """Compute pairwise Ka/Ks from CDS. Returns dict."""
    return _pairwise_kaks(as_records(cds_fasta))


@validated_public('selection.validate_cds_sequences')
def validate_cds_sequences(cds_fasta: str | Path | Sequences, require_start: bool = True) -> dict:
    """Validate CDS sequences. Returns dict with per-seq results."""
    from .codeml import validate_cds

    r = validate_cds(cds_fasta, require_start=require_start)
    return {key: thaw_json(value) for key, value in r.metrics.items()}
