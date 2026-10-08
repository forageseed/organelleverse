"""Typed cores for trans-splicing (OmicVerse-style data-in/data-out)."""

from __future__ import annotations

from typing import Any, cast

from ..core.frozen import thaw_json
from .trans_splicing import detect_trans_splicing


def compute_trans_splicing(genome, min_exon_gap: int = 5000) -> dict:
    """Classify cis/trans-splicing genes. Returns dict."""
    r = detect_trans_splicing(genome, min_exon_gap=min_exon_gap)
    return cast(dict[str, Any], thaw_json(r.metrics))
