"""Typed cores for structure (OmicVerse-style data-in/data-out)."""

from __future__ import annotations

from pathlib import Path

from .._bio import read_fasta
from .structure import _find_record_repeats, _find_ssrs


def compute_multiconf(
    fasta_path: str | Path, min_repeat_len: int = 50, max_mismatches: int = 0
) -> dict:
    """Detect repeat-mediated multi-configurations. Returns dict.

    Scans each FASTA record independently with record-local coordinates.
    Legacy configuration counts are candidate outcomes (2 per repeat pair),
    not distinct configurations validated by graph or read evidence.
    ``max_mismatches`` bridges substitutions between exact repeat seeds.
    """
    seqs = read_fasta(Path(fasta_path))
    repeats = _find_record_repeats(seqs, min_repeat_len, max_mismatches=max_mismatches)
    direct = [r for r in repeats if r["type"] == "direct"]
    inverted = [r for r in repeats if r["type"] == "inverted"]
    direct_pairs = len(direct)
    inverted_pairs = len(inverted)
    predicted = 2 * direct_pairs + 2 * inverted_pairs
    return {
        "direct_repeat_pairs": direct_pairs,
        "inverted_repeat_pairs": inverted_pairs,
        "candidate_configs": predicted,
        "predicted_configs": predicted,
        "repeats": repeats,
        "sequence_count": len(seqs),
        "interpretation": "repeat-pair candidate outcomes, not validated configurations",
    }


def compute_repeats(
    fasta_path: str | Path, min_ssr_unit: int = 1, max_ssr_unit: int = 6, min_copy: int = 3
) -> dict:
    """Detect exact SSRs with MISA's 10/5/4/3/3/3 copy thresholds."""
    seqs = read_fasta(Path(fasta_path))
    ssrs = [
        {"sequence_id": seq_id, **ssr}
        for seq_id, seq in seqs
        for ssr in _find_ssrs(seq.upper(), min_ssr_unit, max_ssr_unit, min_copy)
    ]
    return {"ssr_count": len(ssrs), "ssrs": ssrs}
