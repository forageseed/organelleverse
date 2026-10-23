"""Structural trans-splicing test shared by the GenBank writer and the genome maps."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

#: Longest gap treated as a cis intron, both when writing annotations and when a
#: map has to infer trans-splicing from a GenBank file without ``/trans_splicing``.
MAX_CIS_INTRON = 25_000


def trans_splicing_evidence(
    parts: Sequence[tuple[int, int, int]], length: int, max_cis_gap: int | None = None
) -> str | None:
    """Why a join is trans-spliced: ``"mixed_strand"``, ``"backward"``, ``"distance"`` or ``None``.

    ``parts`` are ``(start, end, strand)`` in biological (join) order, 0-based
    half-open. A cis intron is a forward gap; a junction that jumps back wraps
    round the circle, so its forward gap exceeds half the genome. A join across
    the circular origin is contiguous and has a tiny gap, so it stays cis.

    ``max_cis_gap`` adds the distance rule: a forward gap above it is trans
    (nad5 exons 2-3 sit 123 kb apart on one strand, which neither the strand nor
    the direction test can see).
    """
    if len(parts) < 2 or length <= 0:
        return None
    if len({strand for _, _, strand in parts}) > 1:
        return "mixed_strand"
    # Half the genome only separates a backward jump from a cis intron when it
    # is longer than any cis intron (a 40 kb toy genome cannot tell them apart).
    backward_is_visible = length // 2 > MAX_CIS_INTRON
    distance = False
    for (a_start, a_end, strand), (b_start, b_end, _) in pairwise(parts):
        gap = (b_start - a_end) % length if strand == 1 else (a_start - b_end) % length
        if backward_is_visible and gap > length // 2:
            return "backward"
        distance = distance or (max_cis_gap is not None and gap > max_cis_gap)
    return "distance" if distance else None


def parts_are_trans_spliced(
    parts: Sequence[tuple[int, int, int]], length: int, max_cis_gap: int | None = None
) -> bool:
    """True when a join mixes strands, runs backwards, or (with ``max_cis_gap``) spans too far."""
    return trans_splicing_evidence(parts, length, max_cis_gap) is not None
