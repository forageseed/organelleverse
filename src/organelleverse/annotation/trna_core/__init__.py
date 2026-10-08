"""Clean-room tRNA annotation core."""

from __future__ import annotations

from .anticodon import isotypes_for_anticodon, known_anticodons, normalize_anticodon
from .models import (
    StemFeature,
    StemPair,
    TRNACall,
    TRNACandidate,
    TRNAFeatureSet,
    TRNAFilterDecision,
    TRNAScore,
)

__all__ = [
    "StemFeature",
    "StemPair",
    "TRNACall",
    "TRNACandidate",
    "TRNAFeatureSet",
    "TRNAFilterDecision",
    "TRNAScore",
    "isotypes_for_anticodon",
    "known_anticodons",
    "normalize_anticodon",
]
