"""Multi-chain genomic DNA domain correspondence, without model selection.

Physical parts are supplied in biological order. Part pairs are explicit:
origin-split parts and distinct IR copies must not be guessed from coordinates.
Correspondence is conditional on the requested physical target; all outside
alternatives are retained. M denotes aligned bases, not nucleotide identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

from Bio.SeqFeature import SeqFeature


@dataclass(frozen=True)
class GenomicSegment:
    chain_id: str
    source_interval: tuple[int, int]
    target_interval: tuple[int, int]
    orientation: int


@dataclass(frozen=True)
class GenomicAlignment:
    chain_id: str
    source_record: str
    target_record: str
    source_length: int
    target_length: int
    segments: tuple[GenomicSegment, ...]


def genomic_alignment_from_paf(line: str, chain_id: str) -> GenomicAlignment:
    """Decode base-aligned DNA PAF; retain every M/= /X block and chain ID."""
    fields = line.rstrip().split("\t")
    if len(fields) < 12 or not chain_id or fields[4] not in ("+", "-"):
        raise ValueError("DNA PAF requires identity, coordinates and strand")
    tags = {t.split(":", 2)[0]: t.split(":", 2)[2] for t in fields[12:]}
    if "cg" not in tags:
        raise ValueError("DNA PAF requires a CIGAR")
    qlen, qstart, qend = map(int, fields[1:4])
    tlen, tstart, tend = map(int, fields[6:9])
    if not (0 <= qstart < qend <= qlen and 0 <= tstart < tend <= tlen):
        raise ValueError("PAF coordinates must be within their records")
    orientation = 1 if fields[4] == "+" else -1
    q, t = (qstart if orientation == 1 else qend - 1), tstart
    operations = re.findall(r"([1-9]\d*)([MID=X])", tags["cg"])
    if not operations or "".join(n + op for n, op in operations) != tags["cg"]:
        raise ValueError("Unsupported DNA PAF CIGAR")
    segments = []
    for number, operation in operations:
        n = int(number)
        if operation in ("M", "=", "X"):
            start = q if orientation == 1 else q - n + 1
            segments.append(GenomicSegment(chain_id, (start, start + n), (t, t + n), orientation))
            q += orientation * n
            t += n
        elif operation == "I":
            q += orientation * n
        else:
            t += n
    if q != (qend if orientation == 1 else qstart - 1) or t != tend:
        raise ValueError("CIGAR does not consume the PAF intervals")
    return GenomicAlignment(chain_id, fields[0], fields[5], qlen, tlen, tuple(segments))


@dataclass(frozen=True)
class DomainMapping:
    chain_id: str
    source_interval: tuple[int, int]
    target_interval: tuple[int, int]
    orientation: int
    strand_agrees: bool
    inside_requested_part: bool


@dataclass(frozen=True)
class DomainState:
    source_interval: tuple[int, int]
    status: str
    mappings: tuple[DomainMapping, ...]


@dataclass(frozen=True)
class PhysicalPartCorrespondence:
    source_part: int
    target_part: int
    source_coding_interval: tuple[int, int]
    target_coding_interval: tuple[int, int]
    states: tuple[DomainState, ...]
    target_uncovered_intervals: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class GenomicProductCorrespondence:
    source_id: str
    target_id: str
    part_pairs: tuple[tuple[int, int], ...]
    parts: tuple[PhysicalPartCorrespondence, ...]
    unpaired_source_parts: tuple[int, ...]
    unpaired_target_parts: tuple[int, ...]


def _coding_intervals(feature: SeqFeature) -> tuple[tuple[int, int], ...]:
    phase = int(feature.qualifiers.get("codon_start", ["1"])[0])
    if phase not in (1, 2, 3):
        raise ValueError("codon_start must be 1, 2 or 3")
    offset = 1 - phase
    result = []
    for part in feature.location.parts:
        if part.strand not in (-1, 1):
            raise ValueError("Physical correspondence requires known part strands")
        result.append((offset, offset + len(part)))
        offset += len(part)
    return tuple(result)


def compare_genomic_products(
    source_id: str,
    source_feature: SeqFeature,
    target_id: str,
    target_feature: SeqFeature,
    alignments: tuple[GenomicAlignment, ...],
    *,
    source_record: str,
    target_record: str,
    part_pairs: tuple[tuple[int, int], ...],
) -> GenomicProductCorrespondence:
    """Exact piecewise coordinate relations across all provided DNA chains.

    Compatible pieces can come from different chains (cis/trans/origin parts).
    Distinct in-part coordinates or orientations remain ambiguous; identical
    duplicate mappings retain all chain IDs without creating extra votes.
    Outside-part placements stay in each state. No chain consistency, inferred
    homology, nucleotide identity, biological start or ranking is asserted.
    """
    if not source_id or not target_id or not source_record or not target_record:
        raise ValueError("Product and record identities are required")
    sp, tp = source_feature.location.parts, target_feature.location.parts
    sc, tc = _coding_intervals(source_feature), _coding_intervals(target_feature)
    if len(set(part_pairs)) != len(part_pairs) or any(
        i not in range(len(sp)) or j not in range(len(tp)) for i, j in part_pairs
    ):
        raise ValueError("Physical part pairs must be valid and distinct")
    segments = tuple(
        segment
        for alignment in alignments
        if alignment.source_record == source_record and alignment.target_record == target_record
        for segment in alignment.segments
    )
    relations = []
    for i, j in part_pairs:
        a, b, c, d = int(sp[i].start), int(sp[i].end), int(tp[j].start), int(tp[j].end)
        clipped = []
        cuts = {a, b}
        for segment in segments:
            lo, hi = segment.source_interval
            x, y = max(a, lo), min(b, hi)
            if x >= y:
                continue
            ta = segment.target_interval[0]
            start = ta + x - lo if segment.orientation == 1 else ta + hi - y
            clipped.append((segment, x, y, start))
            cuts.update((x, y))
            for boundary in (c, d):
                cut = x + boundary - start if segment.orientation == 1 else y - boundary + start
                if x < cut < y:
                    cuts.add(cut)
        ordered = sorted(cuts)
        states, covered = [], []
        for x, y in pairwise(ordered):
            mappings = []
            for segment, left, right, ta in clipped:
                if left <= x and y <= right:
                    start = ta + x - left if segment.orientation == 1 else ta + right - y
                    inside = c <= start and start + y - x <= d
                    agrees = sp[i].strand * segment.orientation == tp[j].strand
                    mappings.append(
                        DomainMapping(
                            segment.chain_id,
                            (x, y),
                            (start, start + y - x),
                            segment.orientation,
                            agrees,
                            inside,
                        )
                    )
                    if inside and agrees:
                        covered.append((start, start + y - x))
            inside = [m for m in mappings if m.inside_requested_part]
            distinct = {(m.target_interval, m.orientation) for m in inside}
            if not mappings:
                status = "unmapped"
            elif not inside:
                status = "outside_requested_part"
            elif len(distinct) > 1:
                status = "ambiguous_requested_part_mapping"
            elif inside[0].strand_agrees:
                status = "single_agreeing_requested_part_mapping"
            else:
                status = "opposed_requested_part_mapping"
            states.append(DomainState((x, y), status, tuple(mappings)))
        uncovered, cursor = [], c
        for start, end in sorted(covered):
            if cursor < start:
                uncovered.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < d:
            uncovered.append((cursor, d))
        relations.append(
            PhysicalPartCorrespondence(i, j, sc[i], tc[j], tuple(states), tuple(uncovered))
        )
    return GenomicProductCorrespondence(
        source_id,
        target_id,
        part_pairs,
        tuple(relations),
        tuple(i for i in range(len(sp)) if all(p[0] != i for p in part_pairs)),
        tuple(j for j in range(len(tp)) if all(p[1] != j for p in part_pairs)),
    )
