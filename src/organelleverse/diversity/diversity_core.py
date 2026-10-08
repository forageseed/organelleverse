"""Typed cores for diversity (OmicVerse-style data-in/data-out).

Backend: scikit-allel (preferred) with pure-Python fallback. The two paths
produce identical values (validated against hand-computed examples).
"""

from __future__ import annotations

from pathlib import Path

from .._bio import read_fasta
from .diversity import (
    _alignment_length,
    _alignment_stats,
    _comparable_mask,
    _pi_pure,
    _pi_skitallel,
    _theta_w_and_tajima_d,
)


def compute_nucleotide_diversity(
    alignment_fasta: str | Path, window_size: int = 1000, step: int = 500
) -> dict:
    """Compute π from a multi-sequence alignment. Returns dict.

    Uses scikit-allel when available, else the pairwise formula
    (differing_pairs / C(n,2)) averaged over comparable ACGT columns.
    """
    seqs = [s.upper() for _, s in read_fasta(Path(alignment_fasta))]
    if len(seqs) < 2:
        return {"pi": 0.0, "n_sequences": len(seqs), "windows": []}
    L = _alignment_length(seqs)
    seg_pos, alt_counts, n_comp = _alignment_stats(seqs)
    pi = _pi_skitallel(alt_counts, seg_pos, n_comp) if seg_pos else 0.0

    windows = []
    for start in range(0, max(1, L - window_size + 1), step):
        end = min(start + window_size, L)
        w_pi = _pi_pure([s[start:end] for s in seqs])
        windows.append({"start": start + 1, "end": end, "pi": round(w_pi, 6)})
    return {"pi": round(pi, 6), "n_sequences": len(seqs), "windows": windows}


def compute_neutral_tests(alignment_fasta: str | Path) -> dict:
    """Compute π, θ_W, Tajima's D, segregating-site count S from an alignment."""
    seqs = [s.upper() for _, s in read_fasta(Path(alignment_fasta))]
    if len(seqs) < 2:
        return {
            "pi": 0.0,
            "theta_w": 0.0,
            "tajima_d": float("nan"),
            "segregating_sites": 0,
            "n_sequences": len(seqs),
        }
    n = len(seqs)
    L = _alignment_length(seqs)
    try:
        import allel  # type: ignore
        import numpy as np  # type: ignore

        seg_pos, alt_counts, n_comp = _alignment_stats(seqs)
        S = len(seg_pos)
        if S == 0:
            return {
                "pi": 0.0,
                "theta_w": 0.0,
                "tajima_d": float("nan"),
                "segregating_sites": 0,
                "n_sequences": n,
            }
        width = max(len(c) for c in alt_counts)
        ac = allel.AlleleCountsArray(
            np.array([tuple(c) + (0,) * (width - len(c)) for c in alt_counts])
        )
        pos = np.array(seg_pos)
        pi = _pi_skitallel(alt_counts, seg_pos, n_comp)
        theta_w = (
            float(
                allel.watterson_theta(
                    pos, ac, start=1, stop=L, is_accessible=np.array(_comparable_mask(seqs))
                )
            )
            if n_comp
            else 0.0
        )
        tajima = float(allel.tajima_d(ac, pos=pos)) if S >= 3 else float("nan")
    except ImportError:
        pi = _pi_pure(seqs)
        seg_pos, alt_counts, n_comp = _alignment_stats(seqs)
        S = len(seg_pos)
        theta_w, tajima = _theta_w_and_tajima_d(S, n, n_comp, pi)
    return {
        "pi": round(pi, 6),
        "theta_w": round(theta_w, 6) if theta_w == theta_w else float("nan"),
        "tajima_d": round(tajima, 6) if tajima == tajima else float("nan"),
        "segregating_sites": S,
        "n_sequences": n,
        "comparable_sites": n_comp,
    }
