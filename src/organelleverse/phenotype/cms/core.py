"""Typed cores for phenotype/CMS (OmicVerse-style data-in/data-out)."""

from __future__ import annotations

from pathlib import Path

from ..._bio import read_fasta
from .pipeline import _screen_candidates


def compute_cms_candidates(
    fasta_path: str | Path, min_orf_aa: int = 50, tm_threshold: float = 0.5
) -> dict:
    """Predict CMS candidate ORFs. Returns dict."""
    seqs = read_fasta(Path(fasta_path))
    candidates = _screen_candidates(seqs, min_orf_aa, 21, tm_threshold)
    return {"candidates": candidates, "count": len(candidates)}
