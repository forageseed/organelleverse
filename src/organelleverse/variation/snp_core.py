"""Typed cores for variation (OmicVerse-style data-in/data-out)."""

from __future__ import annotations
from pathlib import Path
from .._bio import read_fasta


def compute_snp(alignment_fasta: str | Path, reference_index: int = 0) -> dict:
    """Compute SNP/Ti-Tv stats from an alignment. Returns dict."""
    seqs = read_fasta(Path(alignment_fasta))
    ref = seqs[reference_index][1].upper()
    L = len(ref)
    purines = {"A", "G"}
    pyrimidines = {"C", "T"}
    transition = transversion = total_snps = 0
    per_seq = []
    for name, seq in seqs:
        seq = seq.upper()[:L]
        count = 0
        for i in range(L):
            if seq[i] != ref[i] and seq[i] in "ACGT" and ref[i] in "ACGT":
                count += 1
                total_snps += 1
                if {seq[i], ref[i]} <= purines or {seq[i], ref[i]} <= pyrimidines:
                    transition += 1
                else:
                    transversion += 1
        per_seq.append({"name": name, "snp_count": count})
    ti_tv = transition / transversion if transversion else float(transition)
    return {
        "total_snps": total_snps,
        "transitions": transition,
        "transversions": transversion,
        "ti_tv": round(ti_tv, 4),
        "per_sequence": per_seq,
    }
