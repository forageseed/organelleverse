"""Complete inverted-repeat gene copies by mirroring across the plastid IR pair.

Reference transfer keeps one best hit per query, so a gene lying wholly inside
an inverted repeat (ycf2, rps7, rpl23, ndhB, ...) is usually annotated in only
one copy. The two IRs are reverse complements of each other, so the missing
copy is the mirror image of the annotated one. Each mirrored part is checked
against the sequence (and re-located nearby when the copies differ by an
indel). Boundary-crossing exons remain unchanged. Explicitly trans-spliced
CDS products can share exons outside both repeats while their coherent IR
exon block is mirrored into the other repeat.
"""

from __future__ import annotations

import copy
import logging

from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

logger = logging.getLogger(__name__)

IRPair = tuple[tuple[int, int, int], tuple[int, int, int]]

# How far a mirrored part may be re-located when the IR copies carry an indel.
_SEARCH_SLACK = 100


def _rc(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


def _offset_in(
    start: int, end: int, region: tuple[int, int], length: int
) -> tuple[int, int] | None:
    """Offsets of [start, end) inside ``region`` (which may run past the origin)."""
    r0, r1 = region
    for shift in (0, length):
        s, e = start + shift, end + shift
        if r0 <= s and e <= r1:
            return s - r0, e - r0
    return None


def _locate(seq: str, wanted: str, guess: int) -> int | None:
    """Start of ``wanted`` at ``guess``, else its unique occurrence within the slack."""
    if seq[guess : guess + len(wanted)] == wanted:
        return guess
    lo = max(0, guess - _SEARCH_SLACK)
    hi = min(len(seq), guess + len(wanted) + _SEARCH_SLACK)
    window = seq[lo:hi]
    first = window.find(wanted)
    if first < 0 or window.find(wanted, first + 1) >= 0:
        return None
    return lo + first


def _mirror_location(location, seq: str, ir_regions: IRPair):
    """Mirror image of ``location`` in the other IR, or None if not wholly inside one IR."""
    length = len(seq)
    regions = [(r0, r1) for r0, r1, _ in ir_regions]
    parts = list(location.parts)
    for source, target in ((regions[0], regions[1]), (regions[1], regions[0])):
        offsets = [_offset_in(int(p.start), int(p.end), source, length) for p in parts]
        if any(o is None for o in offsets):
            continue
        source_len = source[1] - source[0]
        mirrored = []
        for part, (_, oe) in zip(parts, offsets, strict=True):
            guess = (target[0] + source_len - oe) % length
            wanted = _rc(seq[int(part.start) : int(part.end)])
            start = _locate(seq, wanted, guess)
            if start is None:
                return None
            strand = -(part.strand or 1)
            mirrored.append(FeatureLocation(start, start + len(wanted), strand=strand))
        return mirrored[0] if len(mirrored) == 1 else CompoundLocation(mirrored)
    return None


def _mirror_shared_trans_location(location, seq, ir_regions, *, trans_splicing):
    """Mirror a coherent IR block while retaining disjoint shared exons.

    Ordinary contiguous boundary-crossing genes are outside this contract.
    """
    if not trans_splicing or ir_regions is None:
        return None
    length = len(seq)
    regions = [(start, end) for start, end, _ in ir_regions]
    memberships = []
    for part in location.parts:
        start, end = int(part.start), int(part.end)
        membership = next(
            (
                i
                for i, region in enumerate(regions)
                if _offset_in(start, end, region, length) is not None
            ),
            None,
        )
        # A shared exon must be wholly outside both repeats, not straddle an edge.
        if membership is None and any(
            max(start + shift, a) < min(end + shift, z)
            for a, z in regions
            for shift in (0, length)
        ):
            return None
        memberships.append(membership)
    copies = {i for i in memberships if i is not None}
    if len(copies) != 1 or None not in memberships:
        return None
    parts = []
    for part, membership in zip(location.parts, memberships, strict=True):
        if membership is None:
            parts.append(part)
        else:
            mapped = _mirror_location(part, seq, ir_regions)
            if mapped is None:
                return None
            parts.extend(mapped.parts)
    return CompoundLocation(parts)


def _positions(location) -> set[tuple[int, int]]:
    return {
        (p, part.strand) for part in location.parts for p in range(int(part.start), int(part.end))
    }


def cohere_ir_exons(features: list[SeqFeature], seq: str, ir_regions: IRPair | None) -> int:
    """Place transferred exons of an IR-contained CDS in one repeat copy.

    Independent best hits can put different exons of the same cis-spliced
    reference gene in opposite IRs. Map those exons to the first exon's copy
    using the sequence-verified IR correspondence, before CDS refinement.
    Genes with any part outside the IRs (including trans-spliced rps12) retain
    their locations and exon order.
    """
    if ir_regions is None:
        return 0
    seq = seq.upper()
    regions = [(s, e) for s, e, _ in ir_regions]
    changed = 0
    for feature in features:
        parts = list(feature.location.parts)
        if feature.type != "CDS" or len(parts) < 2:
            continue
        memberships = [
            next(
                (
                    i
                    for i, region in enumerate(regions)
                    if _offset_in(int(p.start), int(p.end), region, len(seq)) is not None
                ),
                None,
            )
            for p in parts
        ]
        if None in memberships or len(set(memberships)) != 2:
            continue
        target = memberships[0]
        coherent = []
        for part, source in zip(parts, memberships, strict=True):
            mapped = part if source == target else _mirror_location(part, seq, ir_regions)
            if mapped is None:
                break
            coherent.extend(mapped.parts)
        else:
            if all(p.strand == coherent[0].strand for p in coherent):
                feature.location = CompoundLocation(coherent)
                changed += 1
    return changed


def mirror_ir_copies(features: list[SeqFeature], seq: str, ir_regions: IRPair | None) -> int:
    """Add the missing IR copy of each CDS (and its gene feature); return how many.

    tRNA/rRNA come from native detection, which finds both copies itself.
    """
    if ir_regions is None:
        return 0
    seq = seq.upper()
    coding = {f.qualifiers.get("gene", [""])[0] for f in features if f.type == "CDS"}
    added = 0
    for feature in list(features):
        gene = feature.qualifiers.get("gene", [""])[0]
        if feature.type not in ("CDS", "gene") or not gene or gene not in coding:
            continue
        location = _mirror_location(feature.location, seq, ir_regions)
        shared = False
        if location is None and feature.type == "CDS":
            location = _mirror_shared_trans_location(
                feature.location,
                seq,
                ir_regions,
                trans_splicing="trans_splicing" in feature.qualifiers,
            )
            shared = location is not None
        if location is None:
            continue
        taken = _positions(location) if not shared else set()
        if any(
            other.type == feature.type
            and other.qualifiers.get("gene", [""])[0] == gene
            and (other.location == location if shared else taken & _positions(other.location))
            for other in features
        ):
            continue
        qualifiers = copy.deepcopy(feature.qualifiers)
        qualifiers.pop("translation", None)
        mirrored = SeqFeature(location, type=feature.type, qualifiers=qualifiers)
        if "transl_except" in qualifiers:
            # the edited codons are the mirror's own, at mirrored coordinates
            from .cds_refine import reannotate_edited_stops

            reannotate_edited_stops(mirrored, seq)
        features.append(mirrored)
        added += 1
    if added:
        logger.info("Added %d inverted-repeat copies by mirroring", added)
    return added
