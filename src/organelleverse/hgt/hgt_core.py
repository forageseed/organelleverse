"""Typed cores for HGT (OmicVerse-style data-in/data-out)."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
from .align import HGTBackendError, run_hgt_alignments
from .hgt import _build_hgt_metrics


def compute_hgt(
    donor_fasta: str | Path,
    recipient_fasta: str | Path,
    min_len: int = 200,
    *,
    min_identity: float = 90.0,
    backend: str = "auto",
    preset: str = "asm5",
    gc_threshold: float = 0.05,
    confirm_blast: bool = True,
    phylogeny_evidence: Mapping[str, Any] | str | Path | None = None,
) -> dict:
    """Detect HGT candidates with minimap2/mappy-style local alignment."""
    try:
        selected_backend, alignments = run_hgt_alignments(
            donor_fasta,
            recipient_fasta,
            backend=backend,
            min_len=min_len,
            min_identity=min_identity,
            preset=preset,
        )
    except HGTBackendError as exc:
        return {
            "status": "failed",
            "backend": backend,
            "error": str(exc),
            "anomaly": exc.code,
        }
    return _build_hgt_metrics(
        alignments,
        donor_fasta=donor_fasta,
        recipient_fasta=recipient_fasta,
        backend=selected_backend,
        min_len=min_len,
        min_identity=min_identity,
        gc_threshold=gc_threshold,
        confirm_blast=confirm_blast,
        phylogeny_evidence=phylogeny_evidence,
    )
