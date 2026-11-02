"""Shared low-level sequence utilities.

Single source of truth for primitives that were previously duplicated under
several names across suites (``_revcomp`` / ``_reverse_complement`` + private
``_COMP`` tables). Pure Python, no heavy imports.
"""

from __future__ import annotations

__all__ = ["COMP", "reverse_complement"]

#: IUPAC complement table (case-preserving) for ``str.translate``. Ambiguity codes
#: complement too (K<->M, R<->Y, B<->V, D<->H; S, W and N are their own complement),
#: as in Biopython: a minus-strand M is a K, not an M.
COMP = str.maketrans("ACGTRYSWKMBDHVNacgtryswkmbdhvn", "TGCAYRSWMKVHDBNtgcayrswmkvhdbn")


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a nucleotide string.

    Complements every IUPAC nucleotide code (case-preserving); any other character
    (gaps, ``*``) is left as-is and only reversed.
    """
    return seq.translate(COMP)[::-1]
