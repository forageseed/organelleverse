"""Compare complete original reads with nuclear context and a periodic graph sequence.

This is reference-assisted sequence evidence, not a certificate of physical origin.
No depth window or assembly truth enters the comparison. A periodic template is
tested by exact infix edit distance, so minimizer masking cannot manufacture a
nuclear-only call for a small circle. Equal costs and missing context are unresolved.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .graph_compare import parse_gfa, reverse_complement


def cigar_blocks(start: int, cigar: str) -> tuple[list[tuple[int, int]], int]:
    blocks = []
    pos = start
    if not re.fullmatch(r"(?:\d+[MIDN=X])+", cigar):
        raise ValueError("Invalid PAF CIGAR")
    for size, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar):
        size = int(size)
        if op in "M=X":
            blocks.append((pos, pos + size))
            pos += size
        elif op == "I":
            pos += size
        elif op not in "DN":
            raise ValueError("PAF CIGAR must not contain clipping or padding")
    return blocks, pos


def read_blocks(fields: list[str], cigar: str) -> list[tuple[int, int]]:
    length, start, end = map(int, (fields[1], fields[2], fields[3]))
    oriented = start if fields[4] == "+" else length - end
    blocks, consumed = cigar_blocks(oriented, cigar)
    if consumed != (end if fields[4] == "+" else length - start):
        raise ValueError("PAF query span disagrees with CIGAR")
    return blocks if fields[4] == "+" else [(length - b, length - a) for a, b in blocks]


def overlap(blocks: list[tuple[int, int]], start: int, end: int) -> int:
    return sum(max(0, min(b, end) - max(a, start)) for a, b in blocks)


def nuclear_context_costs(
    paf: Path, hits: dict[str, dict[str, Any]], roles: dict[str, str]
) -> dict[str, int]:
    """Observed edit-cost upper bounds with aligned bases on both candidate flanks.

    Unaligned query bases each cost one edit; a valid assigned-chromosome context
    must also overlap the candidate. A no-worse unplaced context vetoes the call.
    Secondary placements on assigned chromosomes are retained without a MAPQ cut.
    """
    assigned: dict[str, int] = {}
    unplaced: dict[str, int] = {}
    for line in paf.open():
        f = line.rstrip().split("\t")
        name = f[0]
        h = hits[name]
        tags = {x[:2]: x[5:] for x in f[12:]}
        cost = int(tags["NM"]) + int(f[1]) - (int(f[3]) - int(f[2]))
        role = roles[f[5]]
        if role != "assembled-molecule":
            unplaced[name] = min(unplaced.get(name, cost), cost)
            continue
        blocks = read_blocks(f, tags["cg"])
        a, b = h["query_start"], h["query_end"]
        if not (
            overlap(blocks, 0, a) and overlap(blocks, a, b) and overlap(blocks, b, h["read_length"])
        ):
            continue
        assigned[name] = min(assigned.get(name, cost), cost)
    return {name: cost for name, cost in assigned.items() if unplaced.get(name, cost + 1) > cost}


def extend_nuclear_context(
    read: str, reference: str, fields: list[str], edits: int
) -> dict[str, Any]:
    """Extend a retained PAF witness to both ends of the complete query.

    Keep its observed middle alignment and optimally align each clipped query
    end to the adjacent reference, anchored at the PAF boundary. SHW consumes
    the entire query with only the outer reference end free. A clip of length L
    costs at most L (insert it all), so an extension using at most L edits cannot
    consume more than 2L reference bases. This finite window is a mathematical
    bound, not a chosen flank size. The result is still an upper bound on the
    best nuclear alignment, and does not certify physical origin.
    """
    import edlib

    length, start, end = map(int, (fields[1], fields[2], fields[3]))
    ref_start, ref_end = map(int, fields[7:9])
    if len(read) != length or len(reference) != int(fields[6]) or edits < 0:
        raise ValueError("Read/reference lengths disagree with retained PAF")
    if fields[4] == "-":
        read = reverse_complement(read)
        start, end = length - end, length - start
    read = read.upper()
    left_query, right_query = read[:start], read[end:]
    left_reference = reference[max(0, ref_start - 2 * start) : ref_start][::-1].upper()
    right_reference = reference[ref_end : ref_end + 2 * len(right_query)].upper()

    def extension(query: str, target: str) -> tuple[int, int]:
        if not query:
            return 0, 0
        result = edlib.align(query, target, mode="SHW", task="locations", k=len(query))
        return result["editDistance"], result["locations"][0][1] + 1

    left_cost, left_span = extension(left_query[::-1], left_reference)
    right_cost, right_span = extension(right_query, right_reference)
    return {
        "cost_upper_bound": edits + left_cost + right_cost,
        "unextended_cost_upper_bound": edits + start + length - end,
        "reference": fields[5],
        "strand": fields[4],
        "reference_start": ref_start - left_span,
        "reference_end": ref_end + right_span,
        "middle_edits": edits,
        "left_extension_edits": left_cost,
        "right_extension_edits": right_cost,
        "left_extension_query_bp": start,
        "right_extension_query_bp": length - end,
    }


def extended_nuclear_contexts(
    paf: Path,
    hits: dict[str, dict[str, Any]],
    roles: dict[str, str],
    reads: dict[str, str],
    references: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Retain the best extended assigned and unplaced observed witnesses.

    Every observed placement remains eligible; no MAPQ cut or majority rule is
    used. Realign the complete query inside each extended witness interval.
    Assigned contexts must align the candidate and an existing query flank in
    the resulting path. A tied/no-worse unplaced witness remains unresolved.
    Mapper secondary limits and missing placements remain outside this scope.
    """
    result: dict[str, dict[str, Any]] = {}
    with paf.open() as src:
        for line in src:
            f = line.rstrip().split("\t")
            h = hits[f[0]]
            tags = {x[:2]: x[5:] for x in f[12:]}
            role = "assigned" if roles[f[5]] == "assembled-molecule" else "unplaced"
            contexts = result.setdefault(f[0], {})
            best = contexts.get(role)
            # Zero is the lower bound even after realigning the PAF middle.
            if best is not None and best["cost_upper_bound"] == 0:
                continue
            witness = realign_read_context(
                reads[f[0]], references[f[5]], f, int(tags["NM"]),
                max_cost=best["cost_upper_bound"] - 1 if best is not None else None,
            )
            if witness is None:
                continue
            if role == "assigned" and not candidate_and_flank(witness, h):
                continue
            if best is None or witness["cost_upper_bound"] < best["cost_upper_bound"]:
                contexts[role] = witness
    return result


def realign_read_context(
    read: str, reference: str, fields: list[str], edits: int, *, max_cost: int | None = None,
) -> dict[str, Any] | None:
    """Optimize the entire query within an explicitly observed nuclear window.

    The retained PAF plus insertion-only query clips supplies a valid upper bound.
    Keep the full 2L clip intervals, including tied optimal extensions with
    different spans. HW optimizes the middle and both ends in one search. It is
    exact within that interval, still an upper bound for the whole genome.
    The resulting path supplies candidate/flank bases; no fixed flank size.
    An explicit incumbent cost can bound the search; None then means there is
    no alignment at or below that cost, rather than a missing mapper placement.
    """
    import edlib

    length, start, end = map(int, (fields[1], fields[2], fields[3]))
    if len(read) != length or len(reference) != int(fields[6]) or edits < 0:
        raise ValueError("Read/reference lengths disagree with retained PAF")
    if fields[4] == "-":
        start, end = length - end, length - start
    lo = max(0, int(fields[7]) - 2 * start)
    hi = min(len(reference), int(fields[8]) + 2 * (length - end))
    query = read if fields[4] == "+" else reverse_complement(read)
    upper_bound = edits + start + length - end
    result = edlib.align(
        query.upper(), reference[lo:hi].upper(), mode="HW", task="path",
        k=min(upper_bound, max_cost) if max_cost is not None else upper_bound,
    )
    if result["editDistance"] < 0:
        return None
    start, end = result["locations"][0]
    return {
        "unextended_cost_upper_bound": upper_bound,
        "cost_upper_bound": result["editDistance"],
        "reference": fields[5],
        "strand": fields[4],
        "retained_middle_edits": edits,
        "retained_query_start": int(fields[2]),
        "retained_query_end": int(fields[3]),
        "window_start": lo,
        "window_end": hi,
        "reference_start": lo + start,
        "reference_end": lo + end + 1,
        "query_length": len(read),
        "cigar": result["cigar"],
    }


def candidate_and_flank(witness: dict[str, Any], hit: dict[str, Any]) -> bool:
    """Require reference-aligned candidate bases and at least one observed flank.

    A read ending in the candidate has only one available flank. An entirely
    candidate-contained read has none and cannot establish a source junction.
    Insertions, including a whole clipped flank inserted by an alignment, do
    not count as reference-aligned context.
    """
    blocks, length = cigar_blocks(0, witness["cigar"])
    if witness["strand"] == "-":
        blocks = [(length - b, length - a) for a, b in blocks]
    a, b = hit["query_start"], hit["query_end"]
    return bool(overlap(blocks, a, b) and (
        overlap(blocks, 0, a) or overlap(blocks, b, length)
    ))


def anchored_nuclear_read_contexts(
    paf: Path,
    hits: dict[str, dict[str, Any]],
    reads: dict[str, str],
    donor_reads: dict[str, str],
    donor_hits: dict[str, dict[str, Any]],
    donor_decisions: dict[str, dict[str, Any]],
    *,
    incumbent_costs: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare to observed nuclear-anchored read haplotypes, retaining junctions.

    Each donor must already have a strictly better assigned nuclear context
    than its graph period. A recipient alignment must include its own candidate
    and flank, and reference-aligned bases inside and outside the donor candidate.
    Matching only the donor's repeat does not inherit its nuclear flank. Donor
    provenance is retained; tied periodic/other contexts still veto exclusion.
    Existing eligible nuclear bounds can restrict this search to improvements.
    The caller must retain those witnesses when combining the result; a missing
    new witness does not invalidate an existing context.
    """
    result: dict[str, dict[str, Any]] = {}
    for line in paf.open():
        f = line.rstrip().split("\t")
        name, donor = f[0], f[5]
        proof = donor_decisions[donor]
        if proof["call"] != "nuclear_context_better":
            raise ValueError("Read context donor lacks a nuclear-better comparison")
        best = result.get(name)
        bound = best["cost_upper_bound"] if best is not None else None
        incumbent = (incumbent_costs or {}).get(name)
        if incumbent is not None:
            bound = incumbent if bound is None else min(bound, incumbent)
        if name == donor or bound == 0:
            continue
        tags = {x[:2]: x[5:] for x in f[12:]}
        if bound is not None and len(donor_reads[donor]) < len(reads[name]) - bound:
            continue
        witness = realign_read_context(
            reads[name], donor_reads[donor], f, int(tags["NM"]),
            max_cost=bound - 1 if bound is not None else None,
        )
        if witness is None:
            continue
        if not candidate_and_flank(witness, hits[name]):
            continue
        blocks, _ = cigar_blocks(
            witness["reference_start"], witness["cigar"].translate(str.maketrans("ID", "DI"))
        )
        a, b = donor_hits[donor]["query_start"], donor_hits[donor]["query_end"]
        if not (overlap(blocks, a, b) and (
            overlap(blocks, 0, a) or overlap(blocks, b, len(donor_reads[donor]))
        )):
            continue
        witness["donor_nuclear_contexts"] = proof["nuclear_contexts"]
        witness["donor_periodic_comparison"] = proof["periodic_cost_at_nuclear_bound"]
        if best is None or witness["cost_upper_bound"] < best["cost_upper_bound"]:
            result[name] = witness
    return result


def periodic_edit_cost(read: str, period: str, limit: int) -> int | None:
    """Exact minimum edit comparison; nuc cost is an observed upper bound.

    HW aligns the entire query, allowing free reference ends. A cost <= K uses
    at most len(query)+K reference bases. Adding one period includes every phase.
    Both strands are tested. If a periodic explanation is no worse, retain it.
    """
    if not read or not period or limit < 0:
        raise ValueError("Nonempty DNA and a nonnegative nuclear context cost are required")
    import edlib

    copies = (len(read) + limit + len(period) - 1) // len(period) + 1
    target = period * copies
    read = read.upper()
    target = target.upper()
    best = None
    for query in (read, reverse_complement(read)):
        distance = edlib.align(query, target, mode="HW", task="distance", k=limit)[
            "editDistance"
        ]
        if distance >= 0:
            best = distance if best is None else min(best, distance)
    return best


def compare_periodic_context(read: str, period: str, nuclear_cost: int | None) -> str:
    """Compare a complete read with an observed nuclear edit-cost upper bound."""
    if nuclear_cost is None:
        return "unresolved"
    return (
        "periodic_compatible"
        if periodic_edit_cost(read, period, nuclear_cost) is not None
        else "nuclear_context_better"
    )


def summarize_segment(
    name: str, sequence: str, classes: list[str], nuclear_intervals: list[tuple[int, int]]
) -> dict[str, Any]:
    counts = Counter(classes)
    if counts.keys() - {"nuclear_context_better", "periodic_compatible", "unresolved"}:
        raise ValueError("Unknown read-context comparison")
    return {
        "segment": name,
        "sequence": sequence,
        "associated_reads": len(classes),
        "nuclear_context_reads": counts["nuclear_context_better"],
        "periodic_compatible_reads": counts["periodic_compatible"],
        "unresolved_reads": counts["unresolved"],
        "nuclear_covered_intervals": nuclear_intervals,
    }


def collect_read_origin_evidence(
    graph: Path,
    candidate_sam: Path,
    decisions_jsonl: Path,
    *,
    original_reads: Path,
    scanned_reads: int,
    out_json: Path,
) -> dict[str, Any]:
    """Materialize retained exact comparisons without remapping or refitting.

    ``candidate_sam`` contains complete reads aligned to graph segments; each
    primary record must have exactly one retained decision with the same segment.
    Coverage counts only reference-aligned M/=X bases of nuclear-better reads.
    This evidence is scoped to the associated mapper hits, not all physical copies.
    Original reads must exist and the upstream full scan count must be supplied.
    """
    if not original_reads.is_file() or scanned_reads < 1:
        raise ValueError("Original reads and a positive full-scan count are required")
    sequences, _ = parse_gfa(graph)
    decisions = {}
    with decisions_jsonl.open() as src:
        for line in src:
            decision = json.loads(line)
            if decision["read"] in decisions:
                raise ValueError("Duplicate original-read decision")
            decisions[decision["read"]] = decision
    classes: dict[str, list[str]] = {}
    intervals: dict[str, list[tuple[int, int]]] = {}
    seen = set()
    opener = gzip.open if candidate_sam.suffix == ".gz" else open
    with opener(candidate_sam, "rt") as src:
        for line in src:
            if line.startswith("@"):
                continue
            f = line.rstrip().split("\t")
            if int(f[1]) & (4 | 256 | 2048):
                continue
            name, segment = f[0], f[2]
            if name in seen:
                raise ValueError("Duplicate primary candidate alignment")
            seen.add(name)
            decision = decisions[name]
            if decision["segment"] != segment:
                raise ValueError("Read decision disagrees with candidate segment")
            cigar = f[5]
            tokens = re.findall(r"(\d+)([MIDNSHP=X])", cigar)
            if "".join(n + op for n, op in tokens) != cigar or any(op == "H" for _, op in tokens):
                raise ValueError("Candidate SAM must contain complete, unclipped original reads")
            if f[9] == "*" or sum(int(n) for n, op in tokens if op in "MIS=X") != len(f[9]):
                raise ValueError("Candidate SAM does not contain the complete query sequence")
            classes.setdefault(segment, []).append(decision["call"])
            positions = intervals.setdefault(segment, [])
            if decision["call"] == "nuclear_context_better":
                pos = int(f[3]) - 1
                for n, op in tokens:
                    size = int(n)
                    if op in "M=X":
                        positions.append((pos, pos + size))
                    if op in "MDN=X":
                        pos += size
    if seen != decisions.keys() or len(seen) > scanned_reads:
        raise ValueError("Read decisions and full-scan candidate alignments disagree")
    result = {
        "schema": "ovasm.read-origins.v1",
        "original_reads": str(original_reads.resolve()),
        "scanned_reads": scanned_reads,
        "scope": "observed primary candidate hits; exact periodic competition; reference-assisted",
        "segments": [summarize_segment(s, sequences[s], classes[s], intervals[s]) for s in classes],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def assess_read_origins(
    graph: Path,
    candidate_sam: Path,
    nuclear_paf: Path,
    *,
    hits: dict[str, dict[str, Any]],
    nuclear_roles: dict[str, str],
    graph_periods: dict[str, str],
    original_reads: Path,
    scanned_reads: int,
    decisions_jsonl: Path,
    out_json: Path,
    nuclear_sequences: dict[str, str] | None = None,
    retained_nuclear_contexts: dict[str, dict[str, Any]] | None = None,
    nuclear_read_contexts: dict[str, dict[str, Any]] | None = None,
    organelle_contexts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare retained complete reads, then produce native component input.

    ``graph_periods`` are the proposed singleton cycles exported by OVASM primary,
    not mitochondrial truth. SAM sequence may be reverse-oriented; the comparison
    tests both strands. The caller retains the upstream mapper commands, whole-input
    scan receipt, reference roles and input run manifest. Unsupported graph paths
    must be assessed separately rather than assigned a fictitious periodic model.
    Explicit retained contexts can be consumed without repeating their alignment.
    Nuclear-anchored read haplotypes can improve assigned-context upper bounds;
    an observed organelle context at equal/lower cost vetoes a nuclear-only call.
    """
    opener = gzip.open if candidate_sam.suffix == ".gz" else open
    contexts = None
    if retained_nuclear_contexts is not None:
        contexts = {name: dict(c) for name, c in retained_nuclear_contexts.items()}
    elif nuclear_sequences is None:
        costs = nuclear_context_costs(nuclear_paf, hits, nuclear_roles)
    else:
        reads = {}
        with opener(candidate_sam, "rt") as src:
            for line in src:
                if line.startswith("@"):
                    continue
                f = line.rstrip().split("\t")
                if not int(f[1]) & (4 | 256 | 2048):
                    reads[f[0]] = f[9]
        contexts = extended_nuclear_contexts(
            nuclear_paf, hits, nuclear_roles, reads, nuclear_sequences
        )
    if contexts is not None:
        for name, witness in (nuclear_read_contexts or {}).items():
            c = contexts.setdefault(name, {})
            if "assigned" not in c or witness["cost_upper_bound"] < c["assigned"]["cost_upper_bound"]:
                c["assigned"] = witness
        costs = {
            name: c["assigned"]["cost_upper_bound"]
            for name, c in contexts.items()
            if "assigned" in c
            and (
                "unplaced" not in c
                or c["unplaced"]["cost_upper_bound"] > c["assigned"]["cost_upper_bound"]
            )
        }
    elif nuclear_read_contexts:
        raise ValueError("Nuclear read contexts require retained or sequence-backed nuclear contexts")
    decisions_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with opener(candidate_sam, "rt") as src, decisions_jsonl.open("w") as out:
        for line in src:
            if line.startswith("@"):
                continue
            f = line.rstrip().split("\t")
            if int(f[1]) & (4 | 256 | 2048):
                continue
            name, segment, read = f[0], f[2], f[9]
            if hits[name]["component"] != segment or len(read) != hits[name]["read_length"]:
                raise ValueError("Candidate SAM and original-read association receipt disagree")
            cost = costs.get(name)
            organelle = (organelle_contexts or {}).get(name)
            if organelle is not None and cost is not None and organelle["cost_upper_bound"] <= cost:
                cost = None
            periodic_cost = (
                periodic_edit_cost(read, graph_periods[segment], cost) if cost is not None else None
            )
            call = (
                "unresolved" if cost is None
                else "periodic_compatible" if periodic_cost is not None
                else "nuclear_context_better"
            )
            out.write(
                json.dumps(
                    {
                        "read": name,
                        "segment": segment,
                        "call": call,
                        "nuclear_cost_upper_bound": cost,
                        "periodic_cost_at_nuclear_bound": periodic_cost,
                        **({"nuclear_contexts": contexts.get(name, {})} if contexts is not None else {}),
                        **({"organelle_context": organelle} if organelle is not None else {}),
                    }
                )
                + "\n"
            )
    return collect_read_origin_evidence(
        graph,
        candidate_sam,
        decisions_jsonl,
        original_reads=original_reads,
        scanned_reads=scanned_reads,
        out_json=out_json,
    )
