"""Typed cores for barcode (OmicVerse-style data-in/data-out)."""

from __future__ import annotations
from pathlib import Path
from .._bio import read_fasta


def compute_barcode_candidates(
    alignment_fasta: str | Path, min_window: int = 100, min_p_variability: float = 0.05
) -> dict:
    """Find highly variable regions suitable as barcodes. Returns dict."""
    from .barcode import _count_variable_sites

    if min_window < 2:
        min_window = 2
    seqs = [s.upper() for _, s in read_fasta(Path(alignment_fasta))]
    if len(seqs) < 2:
        return {"candidates": [], "n_sequences": len(seqs)}
    L = min(len(s) for s in seqs)
    step = max(1, min_window // 2)
    candidates = []
    for start in range(0, L - min_window + 1, step):
        end = start + min_window
        var = _count_variable_sites(seqs, start, end)
        p_var = var / (end - start)
        if p_var >= min_p_variability:
            candidates.append({"start": start + 1, "end": end, "p_variable": round(p_var, 4)})
    candidates.sort(key=lambda c: c["p_variable"], reverse=True)
    return {"candidates": candidates, "n_sequences": len(seqs), "alignment_length": L}


def compute_jaccard_identity(
    query_fasta: str | Path, reference_fasta: str | Path, k: int = 31
) -> dict:
    """Identify query against reference DB via k-mer Jaccard. Returns dict."""
    query = "".join(s.upper() for _, s in read_fasta(Path(query_fasta)))
    refs = read_fasta(Path(reference_fasta))
    qk = {query[i : i + k] for i in range(len(query) - k + 1) if "N" not in query[i : i + k]}
    scores = []
    for name, seq in refs:
        rk = {
            seq.upper()[i : i + k]
            for i in range(len(seq) - k + 1)
            if "N" not in seq.upper()[i : i + k]
        }
        if qk and rk:
            j = len(qk & rk) / len(qk | rk)
            scores.append((name, round(j, 4)))
    scores.sort(key=lambda x: x[1], reverse=True)
    best = scores[0] if scores else ("none", 0.0)
    return {
        "best_match": best[0],
        "jaccard": best[1],
        "n_references": len(refs),
        "all_scores": scores[:10],
    }
