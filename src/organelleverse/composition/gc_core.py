"""Typed cores for composition (OmicVerse-style data-in/data-out)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from annotated_types import Ge
from organelleverse._public_validation import validated_public

from .._bio import read_fasta
from ..sequences import Sequences, as_path


@validated_public('composition.compute_gc_content')
def compute_gc_content(
    fasta_path: str | Path | Sequences,
    window_size: Annotated[int, Ge(1)] = 500,
    *,
    step: Annotated[int, Ge(1)] | None = None,
) -> dict:
    """Compute global GC fraction and windowed GC fraction/GC skew.

    ``step=None`` preserves the historical nonoverlapping ``window_size`` step.
    Coordinates are 1-based inclusive over records concatenated in file order.
    Only full linear windows are emitted, except one short window when the
    entire input is shorter than ``window_size``; there is no circular wrapping.
    GC fractions include ambiguous sequence characters in the length denominator.
    A window without G or C has GC skew 0, preserving the existing definition.
    Returns a dict, not OrganelleResult.
    """
    fasta_path = as_path(fasta_path)
    seqs = read_fasta(Path(fasta_path))
    all_seq = "".join(s.upper() for _, s in seqs)
    total = len(all_seq)
    gc = all_seq.count("G") + all_seq.count("C")
    gc_frac = gc / total if total else 0.0
    windows = []
    stride = window_size if step is None else step
    for start in range(0, max(1, total - window_size + 1), stride):
        end = min(start + window_size, total)
        w = all_seq[start:end]
        g = w.count("G")
        c = w.count("C")
        skew = (g - c) / (g + c) if (g + c) else 0.0
        windows.append({
            "start": start + 1, "end": end, "gc_skew": round(skew, 4),
            "gc_content": round((g + c) / len(w), 4) if w else 0.0,
        })
    return {"gc_content": round(gc_frac, 4), "total_length": total, "windows": windows}
