"""Alignment-supported rearrangements; coordinates are zero-based, half-open.

The maximum-weight increasing chain of direct, nonoverlapping anchors defines
synteny. Reverse runs are inversions, direct anchors outside that chain are
translocations, and repeated reference coverage at disjoint query loci is
reported as duplication evidence. These are pairwise alignment descriptions,
not a minimum historical rearrangement scenario or a SyRI reimplementation.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from Bio.Seq import reverse_complement

from .._bio import read_fasta
from ..core.errors import OrganelleInputError
from ..core.result import OrganelleResult
from .compare import _provenance
from .identity import _anchor_blocks, _require_mappy
from .orientation import _quadripartite, _segment, _strand


def _input(path):
    records = [(name, seq.upper()) for name, seq in read_fasta(Path(path))]
    if len(records) != 1:
        raise OrganelleInputError(
            code="comparative.sv.sequence_count",
            message="Each FASTA must contain exactly one sequence.",
        )
    if not records[0][1] or set(records[0][1]) - set("ACGTRYSWKMBDHVN"):
        raise OrganelleInputError(
            code="comparative.sv.dna_required", message="Nonempty ungapped IUPAC DNA is required."
        )
    return records[0]


def _backbone(blocks):
    """Exact maximum reference-span chain in the 2D nonoverlap partial order."""
    direct = sorted(
        (b for b in blocks if b["strand"] == 1),
        key=lambda b: (b["reference_start"], b["query_start"], b["id"]),
    )
    scores, paths = [], []
    for i, b in enumerate(direct):
        choices = [
            j
            for j in range(i)
            if direct[j]["reference_end"] <= b["reference_start"]
            and direct[j]["query_end"] <= b["query_start"]
        ]
        prev = max(choices, key=lambda j: scores[j]) if choices else None
        scores.append(
            b["reference_end"] - b["reference_start"] + (scores[prev] if prev is not None else 0)
        )
        paths.append((paths[prev] if prev is not None else []) + [b["id"]])
    return set(paths[max(range(len(scores)), key=scores.__getitem__)]) if scores else set()


def _event(kind, blocks, rs, re, qs, qe, strand):
    return {
        "type": kind,
        "reference_start": rs,
        "reference_end": re,
        "query_start": qs,
        "query_end": qe,
        "reference_length": re - rs,
        "query_length": qe - qs,
        "length": max(re - rs, qe - qs),
        "strand": strand,
        "supporting_blocks": [b["id"] for b in blocks],
    }


def _query_bounds(b, start, end):
    r, q = b["reference_start"], b["query_start"] if b["strand"] == 1 else b["query_end"]
    positions = []
    for length, op in b["cigar"]:
        if op == "M":
            lo, hi = max(r, start), min(r + length, end)
            if lo < hi:
                positions.extend([q + b["strand"] * (lo - r), q + b["strand"] * (hi - r)])
        if op != "I":
            r += length
        if op != "D":
            q += b["strand"] * length
    return (min(positions), max(positions)) if positions else None


def _classify(blocks, minimum):
    backbone = _backbone(blocks)
    events, duplicated = [], set()
    # Intersections are exact repeated aligned reference intervals, not a
    # copy-number estimate for an entire genome or evidence of orthology.
    assigned = []
    for b in sorted(blocks, key=lambda b: (b["id"] not in backbone, b["query_start"])):
        overlaps = []
        for a in assigned:
            lo, hi = (
                max(a["reference_start"], b["reference_start"]),
                min(a["reference_end"], b["reference_end"]),
            )
            if hi - lo >= minimum:
                bounds = _query_bounds(b, lo, hi)
                if bounds is not None:
                    events.append(_event("duplication", [a, b], lo, hi, *bounds, b["strand"]))
                    overlaps.append((lo, hi))
        end = b["reference_start"]
        for lo, hi in sorted(overlaps):
            if lo > end:
                break
            end = max(end, hi)
        if end == b["reference_end"]:
            duplicated.add(b["id"])
        assigned.append(b)
    # Maximal inverted runs must be adjacent in BOTH reference and query
    # ordering. No distance threshold bridges intervening aligned blocks.
    rorder = sorted(blocks, key=lambda b: (b["reference_start"], b["query_start"]))
    qrank = {b["id"]: i for i, b in enumerate(sorted(blocks, key=lambda b: b["query_start"]))}
    runs = []
    for b in rorder:
        if b["strand"] != -1 or b["id"] in duplicated:
            runs.append([])
            continue
        if (
            runs
            and runs[-1]
            and qrank[runs[-1][-1]["id"]] == qrank[b["id"]] + 1
            and runs[-1][-1]["reference_end"] <= b["reference_start"]
        ):
            runs[-1].append(b)
        else:
            runs.append([b])
    for run in filter(None, runs):
        events.append(
            _event(
                "inversion",
                run,
                run[0]["reference_start"],
                run[-1]["reference_end"],
                min(b["query_start"] for b in run),
                max(b["query_end"] for b in run),
                -1,
            )
        )
    for b in blocks:
        if b["strand"] == 1 and b["id"] not in backbone | duplicated:
            events.append(
                _event(
                    "translocation",
                    [b],
                    b["reference_start"],
                    b["reference_end"],
                    b["query_start"],
                    b["query_end"],
                    1,
                )
            )
        r, q = b["reference_start"], b["query_start"] if b["strand"] == 1 else b["query_end"]
        for length, op in b["cigar"]:
            rn, qn = (
                r + (length if op != "I" else 0),
                q + (b["strand"] * length if op != "D" else 0),
            )
            if op in {"I", "D"} and length >= minimum:
                events.append(
                    _event(
                        "insertion" if op == "I" else "deletion",
                        [b],
                        r,
                        rn,
                        min(q, qn),
                        max(q, qn),
                        b["strand"],
                    )
                )
            r, q = rn, qn
    return sorted(
        events, key=lambda e: (e["reference_start"], e["query_start"], e["type"])
    ), sorted(backbone)


def detect_structural_variants(
    reference_fasta: str | Path,
    query_fasta: str | Path,
    *,
    preset: str = "asm20",
    min_size: int = 100,
    topology: str = "circular",
    plastome: bool = False,
) -> OrganelleResult:
    """Detect alignment-supported inversions, translocations, duplications and indels.

    One ungapped sequence per FASTA. Circular sequences are both cut at the
    longest shared anchor (ties: mapq then input coordinates); query strand
    follows that anchor. This fixes a coordinate convention, not an ancestral
    direction. Replayable operations and normalized sequences are returned.

    With ``plastome=True``, identify the existing detector's quadripartite
    regions and align SSC independently using the requested minimap2 preset.
    A resolved SSC isomer is normalized before compartment-wise anchoring. An
    unidentified IR or mixed/absent SSC anchors fails explicitly. Unlike the
    cohort orientation operation, rearranged LSC regions need not all agree.
    LSC strand maximizes the weight of its direct collinear backbone. The two
    IR copies are assigned by LSC/IRb/SSC/IRa position; rearrangements crossing
    these compartment boundaries are outside this plastome mode's contract.

    ``min_size`` filters anchor spans and CIGAR indels. Unaligned sequence is
    reported as uncovered bases, never equated to a deletion. Duplications
    are reference-relative alignment evidence; repeat-copy orthology and
    historical event counts are not identifiable from these alignments.
    """
    if (
        preset not in {"asm5", "asm10", "asm20"}
        or min_size < 1
        or topology not in {"linear", "circular"}
    ):
        raise OrganelleInputError(
            code="comparative.sv.parameters",
            message="Use asm5/asm10/asm20, min_size >= 1 and linear/circular topology.",
        )
    mp = _require_mappy()
    rn, reference = _input(reference_fasta)
    qn, query = _input(query_fasta)
    operations = {"reference": [], "query": []}
    ssc = {"policy": "not_requested"}
    if plastome:
        if topology != "circular":
            raise OrganelleInputError(
                code="comparative.sv.plastome_topology",
                message="Plastome normalization requires circular topology.",
            )
        regions = [_quadripartite(seq) for seq in (reference, query)]
        if any(r is None for r in regions):
            raise OrganelleInputError(
                code="comparative.sv.ir_unresolved",
                message="Plastome quadripartite regions could not be identified.",
            )
        # Cut at LSC so SSC is a contiguous interval even for rotated inputs.
        normalized = []
        for role, seq, reg in zip(("reference", "query"), (reference, query), regions, strict=True):
            offset = reg["LSC"]["start"]
            normalized.append(seq[offset:] + seq[:offset])
            operations[role].append({"operation": "rotate_left", "bases": offset})
        reference, query = normalized
        rr, qr = [_quadripartite(seq) for seq in normalized]
        # Compare maximum-weight chains in both directions without assuming
        # that the whole LSC is collinear.
        anchors = _anchor_blocks(
            _segment(reference, rr["LSC"]), _segment(query, qr["LSC"]), preset=preset
        )
        if not anchors:
            raise OrganelleInputError(
                code="comparative.sv.lsc_unresolved", message="No shared LSC anchors."
            )
        for i, b in enumerate(anchors):
            b["id"] = str(i)
        forward = _backbone(anchors)
        reverse_anchors = [
            dict(
                b,
                strand=-b["strand"],
                query_start=len(query) - b["query_end"],
                query_end=len(query) - b["query_start"],
            )
            for b in anchors
        ]
        reverse = _backbone(reverse_anchors)

        def score(ids):
            return sum(b["reference_end"] - b["reference_start"] for b in anchors if b["id"] in ids)

        if score(reverse) > score(forward):
            query = reverse_complement(query)
            operations["query"].append(
                {"operation": "reverse_complement", "start": 0, "end": len(query)}
            )
            qr = _quadripartite(query)
            offset = qr["LSC"]["start"]
            query = query[offset:] + query[:offset]
            operations["query"].append({"operation": "rotate_left", "bases": offset})
            qr = _quadripartite(query)
        strand = _strand(
            mp.Aligner(seq=_segment(reference, rr["SSC"]), preset=preset),
            _segment(query, qr["SSC"]),
        )
        if strand is None:
            raise OrganelleInputError(
                code="comparative.sv.ssc_unresolved",
                message="SSC anchors are absent or mixed; cannot distinguish an SSC isomer from rearrangement.",
            )
        ssc = {"policy": "reference_ssc", "anchor_strand": strand, "reversed": strand == -1}
        if strand == -1:
            start, length = qr["SSC"]["start"], qr["SSC"]["length"]
            query = (
                query[:start]
                + reverse_complement(query[start : start + length])
                + query[start + length :]
            )
            operations["query"].append(
                {
                    "operation": "reverse_complement",
                    "region": "SSC",
                    "start": start,
                    "end": start + length,
                }
            )
    elif topology == "circular":
        anchors = _anchor_blocks(reference, query, preset=preset)
        if not anchors:
            raise OrganelleInputError(
                code="comparative.sv.no_anchors",
                message="No shared anchors; structural variation is unresolvable.",
            )
        anchor = max(
            anchors,
            key=lambda b: (
                b["reference_end"] - b["reference_start"],
                b["mapq"],
                -b["reference_start"],
                -b["query_start"],
            ),
        )
        if anchor["strand"] == -1:
            query = reverse_complement(query)
            operations["query"].append(
                {"operation": "reverse_complement", "start": 0, "end": len(query)}
            )
        ro = anchor["reference_start"]
        qo = anchor["query_start"] if anchor["strand"] == 1 else len(query) - anchor["query_end"]
        reference, query = reference[ro:] + reference[:ro], query[qo:] + query[:qo]
        operations["reference"].append({"operation": "rotate_left", "bases": ro})
        operations["query"].append({"operation": "rotate_left", "bases": qo})
    if plastome:
        # IR copies are assigned by their quadripartite position, rather than
        # treating equally valid cross-mapping to the opposite IR as inversion.
        rr, qr = _quadripartite(reference), _quadripartite(query)
        anchors = []
        for region in ("LSC", "IRb", "SSC", "IRa"):
            pieces = _anchor_blocks(
                _segment(reference, rr[region]),
                _segment(query, qr[region]),
                preset=preset,
                reference_unique=False,
                include_cigar=True,
            )
            for b in pieces:
                for key in ("start", "end"):
                    b[f"reference_{key}"] += rr[region]["start"]
                    b[f"query_{key}"] += qr[region]["start"]
                b["region"] = region
            anchors.extend(pieces)
    else:
        anchors = _anchor_blocks(
            reference, query, preset=preset, reference_unique=False, include_cigar=True
        )
    blocks = [
        dict(b, id=f"block_{i + 1}")
        for i, b in enumerate(anchors)
        if min(b["reference_end"] - b["reference_start"], b["query_end"] - b["query_start"])
        >= min_size
    ]
    if not blocks:
        raise OrganelleInputError(
            code="comparative.sv.no_anchors",
            message="No retained anchors; structural variation is unresolvable.",
        )
    variants, backbone = _classify(blocks, min_size)
    covered = {}
    for role, seq in (("reference", reference), ("query", query)):
        end, total = 0, 0
        for start, stop in sorted((b[f"{role}_start"], b[f"{role}_end"]) for b in blocks):
            total += max(0, stop - max(start, end))
            end = max(end, stop)
        covered[role] = {
            "length": len(seq),
            "anchored_bases": total,
            "uncovered_bases": len(seq) - total,
        }
    plot_blocks = [
        {
            "genome_a": "reference:" + rn,
            "genome_b": "query:" + qn,
            "a_start": b["reference_start"],
            "a_end": b["reference_end"],
            "b_start": b["query_start"],
            "b_end": b["query_end"],
            "orientation": "direct" if b["strand"] == 1 else "inverted",
        }
        for b in blocks
    ]
    return OrganelleResult(
        operation_id="comparative.detect_structural_variants",
        scope="plastid" if plastome else "none",
        status="ok",
        summary_text=f"Detected {len(variants)} alignment-supported structural differences.",
        metrics={
            "variants": variants,
            "counts": dict(Counter(v["type"] for v in variants)),
            "alignment_blocks": blocks,
            "backbone": backbone,
            "blocks": plot_blocks,
            "coordinates": "zero-based half-open on normalized sequences",
            "orientation": {"operations": operations, "ssc": ssc, "topology": topology},
            "normalized_sequences": {"reference": reference, "query": query},
            "coverage": covered,
        },
        provenance=_provenance(
            "detect_structural_variants",
            {
                "reference_fasta": str(reference_fasta),
                "query_fasta": str(query_fasta),
                "preset": preset,
                "min_size": min_size,
                "topology": topology,
                "plastome": plastome,
            },
        ).evolve(actual_backend="mappy", attempted_backends=("mappy",)),
    )
