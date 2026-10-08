"""Typed cores for composition (OmicVerse-style data-in/data-out)."""

from __future__ import annotations
from pathlib import Path
from .._bio import read_fasta


def compute_gc_content(fasta_path: str | Path, window_size: int = 500) -> dict:
    """Compute GC content + GC skew. Returns dict (not OrganelleResult)."""
    seqs = read_fasta(Path(fasta_path))
    all_seq = "".join(s.upper() for _, s in seqs)
    total = len(all_seq)
    gc = all_seq.count("G") + all_seq.count("C")
    gc_frac = gc / total if total else 0.0
    windows = []
    for start in range(0, max(1, total - window_size + 1), window_size):
        end = min(start + window_size, total)
        w = all_seq[start:end]
        g = w.count("G")
        c = w.count("C")
        skew = (g - c) / (g + c) if (g + c) else 0.0
        windows.append({"start": start + 1, "end": end, "gc_skew": round(skew, 4)})
    return {"gc_content": round(gc_frac, 4), "total_length": total, "windows": windows}
