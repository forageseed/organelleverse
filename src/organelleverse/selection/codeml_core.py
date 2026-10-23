"""Typed cores for selection/CodeML (OmicVerse-style data-in/data-out)."""

from __future__ import annotations

from pathlib import Path

from organelleverse._public_validation import validated_public

from ..core.frozen import thaw_json
from ..sequences import Sequences, as_records
from .kaks import _nei_gojobori


@validated_public('selection.compute_kaks')
def compute_kaks(cds_fasta: str | Path | Sequences) -> dict:
    """Compute pairwise Ka/Ks from CDS. Returns dict."""
    seqs = as_records(cds_fasta)
    if len(seqs) < 2:
        return {"Ka": 0.0, "Ks": 0.0, "Ka_Ks": 0.0, "pairs": 0}
    results = []
    for a in range(len(seqs)):
        for b in range(a + 1, len(seqs)):
            r = _nei_gojobori(seqs[a][1].upper(), seqs[b][1].upper())
            if r:
                results.append(r)
    if not results:
        return {"Ka": 0.0, "Ks": 0.0, "Ka_Ks": 0.0, "pairs": 0}
    mean_ka = sum(r["Ka"] for r in results) / len(results)
    mean_ks = sum(r["Ks"] for r in results) / len(results)
    ratio = mean_ka / mean_ks if mean_ks else 0.0
    return {
        "Ka": round(mean_ka, 6),
        "Ks": round(mean_ks, 6),
        "Ka_Ks": round(ratio, 6),
        "pairs": len(results),
    }


@validated_public('selection.validate_cds_sequences')
def validate_cds_sequences(cds_fasta: str | Path | Sequences, require_start: bool = True) -> dict:
    """Validate CDS sequences. Returns dict with per-seq results."""
    from .codeml import validate_cds

    r = validate_cds(cds_fasta, require_start=require_start)
    return {key: thaw_json(value) for key, value in r.metrics.items()}
