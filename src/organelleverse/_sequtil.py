"""Shared low-level sequence utilities.

Single source of truth for primitives that were previously duplicated under
several names across suites (``_revcomp`` / ``_reverse_complement`` + private
``_COMP`` tables). Pure Python, no heavy imports.
"""

from __future__ import annotations

__all__ = ["COMP", "reverse_complement"]

#: IUPAC complement table (ACGTN, case-preserving) for ``str.translate``.
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a nucleotide string.

    Complements ACGTN (case-preserving); any other character is left as-is and
    only reversed, matching the previous per-module behaviour.
    """
    return seq.translate(COMP)[::-1]
