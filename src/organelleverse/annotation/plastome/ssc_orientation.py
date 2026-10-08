"""Normalize the small single-copy (SSC) arc orientation of plastome inputs.

An assembler can linearize the circular plastome graph with the SSC in either
orientation: the inverted repeats give the graph two equivalent traversals and
which one appears is an implementation detail, not biology. Deposited records
overwhelmingly follow one convention, and pipelines that promise
submission-ready output flip the SSC when the input came in the other
orientation (PGA and CPGAVAS2 both ship this step).

The orientation of a single genome cannot be measured against anything by
itself; what is invariant is the SSC strand *relative to the LSC strand*.
This module computes that relative sign for every packaged reference from its
own annotation, takes the panel majority as the convention, and measures the
input's relative sign from the primary BLAST transfer itself (no extra
alignment, no new dependency). When the input deviates, the SSC arc is
reverse-complemented *before any feature is placed on the sequence* and the
search is rerun on the corrected sequence, so downstream stages never see
mixed orientations and junction-spanning genes (ycf1, ndhF) are annotated
fresh instead of being coordinate-surgered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from Bio.Seq import Seq

from ...ir_boundary.ir_boundary import _detect_ir_from_sequence
from .models import BlastHit, ReferenceQuery, ReferenceRecord

logger = logging.getLogger(__name__)

#: Votes (hit-versus-reference strand products) required per single-copy
#: region before the region's orientation counts as measured.
_MIN_REGION_VOTES = 3

#: References with a measurable relative sign required before the panel has a
#: convention at all.
_MIN_CONVENTION_REFS = 5

#: Shortest inverted repeat that makes a genome quadripartite. Conifer
#: plastomes keep a ~0.5 kb IR remnant (trnI-CAU) and the arc between the
#: copies is not an SSC; flipping it only moves LSC genes. Real IRs are
#: 10-76 kb.
_MIN_IR_LENGTH = 5000


@dataclass(frozen=True)
class SscDecision:
    """Whether and where to flip, with the evidence that produced the call."""

    flip: bool
    reason: str
    ssc_start: int | None = None
    ssc_length: int | None = None
    convention: int | None = None
    query_relative_sign: int | None = None
    votes: dict[str, int] | None = None


def _quadripartite(sequence: str) -> dict[str, tuple[int, int]] | None:
    """(start, length) of LSC/IRb/SSC/IRa on the circle; None without an IR.

    Same construction as the comparative orientation operation: the longer
    single-copy arc is the LSC, IRb follows LSC on the input strand, starts
    are modulo the input length. Boundary precision is that of the shared
    seed/extension IR detector.
    """
    n = len(sequence)
    pair = _detect_ir_from_sequence(sequence, circular=True)
    if pair is None:
        return None
    arms = sorted((r["start"] % n, r["end"] - r["start"]) for r in pair)
    (a, la), (b, lb) = arms
    if min(la, lb) < _MIN_IR_LENGTH:
        return None
    gap_ab, gap_ba = b - a - la, n + a - b - lb
    if min(gap_ab, gap_ba) <= 0 or gap_ab == gap_ba:
        return None
    if gap_ab > gap_ba:
        a, la, b, lb = b, lb, a, la
        gap_ab, gap_ba = gap_ba, gap_ab
    return {
        "LSC": ((b + lb) % n, gap_ba),
        "IRb": (a, la),
        "SSC": ((a + la) % n, gap_ab),
        "IRa": (b, lb),
    }


def _contains(region: tuple[int, int], position: int, length: int) -> bool:
    start, span = region
    return any(start <= position + shift < start + span for shift in (0, length))


def _region_of(regions: dict[str, tuple[int, int]], feature, length: int) -> str | None:
    """Single-copy region holding the feature's midpoint, if any."""
    location = feature.location
    midpoint = (int(location.start) + int(location.end)) // 2
    for name in ("LSC", "SSC"):
        if _contains(regions[name], midpoint, length):
            return name
    return None


def reference_relative_sign(reference: ReferenceRecord) -> int | None:
    """SSC-majority strand times LSC-majority strand in the reference's own coordinates.

    +1 is the dominant deposited arrangement (both single-copy regions read
    consistently); -1 means this reference was deposited with the SSC in the
    minority orientation. None when the record has no detectable IR or no
    strand majority in either single-copy region.
    """
    sequence = str(reference.record.seq).upper()
    regions = _quadripartite(sequence)
    if regions is None:
        return None
    votes = {"LSC": 0, "SSC": 0}
    for feat in reference.features:
        if feat.feature_type != "CDS" or not feat.gene:
            continue
        region = _region_of(regions, feat.feature, len(sequence))
        if region is not None:
            votes[region] += feat.feature.location.strand or 1
    if 0 in votes.values():
        return None
    return (1 if votes["SSC"] > 0 else -1) * (1 if votes["LSC"] > 0 else -1)


def panel_convention(
    references: tuple[ReferenceRecord, ...],
) -> tuple[int | None, dict[str, int]]:
    """Majority relative sign across the reference panel, with per-reference signs."""
    signs = {
        reference.path.name: sign
        for reference in references
        if (sign := reference_relative_sign(reference)) is not None
    }
    if len(signs) < _MIN_CONVENTION_REFS:
        return None, signs
    total = sum(signs.values())
    if total == 0:
        return None, signs
    return (1 if total > 0 else -1), signs


def _region_agreement(
    votes: dict[str, list[int]],
) -> dict[str, int | None]:
    """+1/-1 per region from hit-strand times reference-strand products."""
    agreement: dict[str, int | None] = {}
    for region, products in votes.items():
        if len(products) < _MIN_REGION_VOTES:
            agreement[region] = None
        else:
            total = sum(products)
            agreement[region] = (1 if total >= 0 else -1) if total != 0 else None
    return agreement


def evaluate_ssc_correction(
    target_seq: str,
    primary: ReferenceRecord,
    references: tuple[ReferenceRecord, ...],
    hits: dict[str, BlastHit],
    queries: tuple[ReferenceQuery, ...],
) -> SscDecision:
    """Decide whether the input's SSC arc should be reverse-complemented.

    The input's relative sign is inferred from the primary transfer: for each
    reference CDS lying in a single-copy region of the *primary* reference,
    the strand product of (hit strand on the input) x (gene strand in the
    reference) votes on whether that region presents the same way in both.
    Rerunning the search after a flip keeps junction-spanning genes honest;
    the decision itself never needs the input's own IR layout, only the flip
    does.
    """
    convention, _ = panel_convention(references)
    if convention is None:
        return SscDecision(flip=False, reason="reference_panel_has_no_convention")

    reference_seq = str(primary.record.seq).upper()
    regions = _quadripartite(reference_seq)
    if regions is None:
        return SscDecision(flip=False, reason="primary_reference_without_quadripartite_structure")
    primary_sign = reference_relative_sign(primary)
    if primary_sign is None:
        return SscDecision(flip=False, reason="primary_reference_sign_unmeasured")

    votes: dict[str, list[int]] = {"LSC": [], "SSC": []}
    for query in queries:
        feature = query.reference_feature
        if feature.feature_type != "CDS":
            continue
        region = _region_of(regions, feature.feature, len(reference_seq))
        if region is None:
            continue
        hit = hits.get(query.query_id)
        if hit is None:
            continue
        votes[region].append(hit.strand * (feature.feature.location.strand or 1))
    agreement = _region_agreement(votes)
    if agreement["LSC"] is None or agreement["SSC"] is None:
        return SscDecision(
            flip=False,
            reason="insufficient_transfer_evidence",
            convention=convention,
            votes={k: len(v) for k, v in votes.items()},
        )

    query_sign = primary_sign * agreement["LSC"] * agreement["SSC"]
    if query_sign == convention:
        return SscDecision(
            flip=False,
            reason="already_conventional",
            convention=convention,
            query_relative_sign=query_sign,
            votes={k: len(v) for k, v in votes.items()},
        )

    target_regions = _quadripartite(target_seq)
    if target_regions is None:
        return SscDecision(
            flip=False,
            reason="quadripartite_not_identified",
            convention=convention,
            query_relative_sign=query_sign,
        )
    start, length = target_regions["SSC"]
    return SscDecision(
        flip=True,
        reason="ssc_opposite_to_reference_panel_convention",
        ssc_start=start,
        ssc_length=length,
        convention=convention,
        query_relative_sign=query_sign,
        votes={k: len(v) for k, v in votes.items()},
    )


def flip_ssc_segment(sequence: str, start: int, length: int) -> str:
    """Reverse-complement the circular segment [start, start+length) in place.

    Positions outside the segment are byte-identical afterwards, including
    when the segment wraps the linear origin. Derived by rotating the segment
    to offset zero, reverse-complementing its head, and rotating back.
    """
    n = len(sequence)
    if not 0 < length < n:
        raise ValueError("SSC segment must be a proper subsegment of the circle")
    start %= n
    rotated = sequence[start:] + sequence[:start]
    flipped = str(Seq(rotated[:length]).reverse_complement()) + rotated[length:]
    if start == 0:
        return flipped
    return flipped[n - start :] + flipped[: n - start]
