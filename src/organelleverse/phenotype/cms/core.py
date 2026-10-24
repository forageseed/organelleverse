"""Typed cores for phenotype/CMS (OmicVerse-style data-in/data-out)."""

from __future__ import annotations

from pathlib import Path

from organelleverse._public_validation import validated_public

from ..._bio import read_fasta
from ...sequences import Sequences, as_path
from .pipeline import _screen_candidates


@validated_public('phenotype.compute_cms_candidates')
def compute_cms_candidates(
    fasta_path: str | Path | Sequences, min_orf_aa: int = 50, tm_threshold: float = 0.5
) -> dict:
    """Predict CMS candidate ORFs. Returns dict."""
    fasta_path = as_path(fasta_path)
    seqs = read_fasta(Path(fasta_path))
    candidates = _screen_candidates(seqs, min_orf_aa, 21, tm_threshold)
    return {"candidates": candidates, "count": len(candidates)}
