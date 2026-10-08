"""Compare two organelle assembly graphs junction by junction (standard library only).

Segment names differ between assemblers, so junctions are matched by the sequence flanking
each link rather than by name: every GFA ``L`` record is one oriented junction, and an
oriented segment end with exactly two outgoing junctions is a two-way branch (the
definitions of the Nipponbare gold standard, Zhang et al. 2026, Plant Cell koag283). Used by
the gold mode (external assemblers against ovasm) and by
a graph-against-gold-graph scorer,
which loads this file by path so it runs without the package's dependencies.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


@dataclass(frozen=True)
class Link:
    source: str
    source_orient: str
    target: str
    target_orient: str
    overlap: int


def _run(argv: list[str]) -> str:
    """Run a command and return its standard output (tests replace this)."""
    return subprocess.run(argv, check=True, capture_output=True, text=True).stdout


def parse_gfa(path: Path) -> tuple[dict[str, str], list[Link]]:
    segments: dict[str, str] = {}
    links: list[Link] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if fields[0] == "S" and len(fields) >= 3:
            segments[fields[1]] = fields[2].upper()
        elif fields[0] == "L" and len(fields) >= 6:
            match = re.fullmatch(r"(\d+)M", fields[5])
            overlap = int(match.group(1)) if match else 0
            links.append(Link(fields[1], fields[2], fields[3], fields[4], overlap))
    return segments, links


def oriented(segments: dict[str, str], name: str, orient: str) -> str:
    seq = segments[name]
    return seq if orient == "+" else reverse_complement(seq)


def junction_flanks(segments: dict[str, str], link: Link, flank: int) -> tuple[str, str]:
    """Return (left, right): ``flank`` bp before and after the link point.

    The overlap is kept once, on the left side, so the pair spells the exact
    sequence a read traverses when it crosses this junction.
    """
    left = oriented(segments, link.source, link.source_orient)[-flank:]
    right = oriented(segments, link.target, link.target_orient)[link.overlap :][:flank]
    return left, right


def junction_signature(left: str, right: str) -> str:
    """Return an orientation-independent key for one junction.

    A read can cross the same junction from either strand: the link
    ``A+ -> B+`` is the same physical junction as ``B- -> A-``, whose flanks
    are ``reverse_complement(right)`` then ``reverse_complement(left)``.
    Two assemblies that store the junction on opposite strands must produce
    the same key here, and two different junctions must not collide.
    """
    forward = f"{left}|{right}"
    reverse = f"{reverse_complement(right)}|{reverse_complement(left)}"
    return min(forward, reverse)


def branch_points(links: list[Link]) -> dict[tuple[str, str], tuple[int, int]]:
    """Map each oriented endpoint with exactly two outgoing links to their indices."""
    flip = {"+": "-", "-": "+"}
    by_endpoint: dict[tuple[str, str], set[int]] = defaultdict(set)
    for index, link in enumerate(links):
        by_endpoint[(link.source, link.source_orient)].add(index)
        by_endpoint[(link.target, flip[link.target_orient])].add(index)
    return {
        endpoint: tuple(sorted(indices))  # type: ignore[misc]
        for endpoint, indices in by_endpoint.items()
        if len(indices) == 2
    }


def recovered_exact(gold_segments, gold_links, cand_segments, cand_links, flank: int) -> list[bool]:
    """Exact flank-sequence identity; requires identical breakpoints in both graphs."""
    gold_keys = [
        junction_signature(*junction_flanks(gold_segments, link, flank)) for link in gold_links
    ]
    cand_keys = {
        junction_signature(*junction_flanks(cand_segments, link, flank)) for link in cand_links
    }
    if len(set(gold_keys)) != len(gold_keys):
        raise ValueError(f"flank={flank} is too short: gold junction signatures collide")
    return [key in cand_keys for key in gold_keys]


def two_step_windows(
    segments: dict[str, str], links: list[Link], flank: int, max_middle: int = 4
) -> list[str]:
    """Sequences of walks ``X -> M1 -> ... -> Mj -> Y`` through up to ``max_middle`` middle
    segments that together (overlaps written once) are shorter than two flanks.

    Such middles split one physical junction over several links, so neither a link's flanks
    nor any single segment hold the gold junction's full ``2 * flank`` bp. With one middle this
    is the two-step walk; more arise where short repeats of different copy number sit at a
    repeat's edge (Nipponbare short reads, k 127: a 4.7 kb repeat ends in a 212 bp three-copy
    and a 222 bp two-copy piece). The window keeps ``flank`` bp of X, the middles and ``flank``
    bp of Y.
    """
    flip = {"+": "-", "-": "+"}
    out: dict[tuple[str, str], list[tuple[str, str, int]]] = defaultdict(list)
    for link in links:
        out[(link.source, link.source_orient)].append(
            (link.target, link.target_orient, link.overlap)
        )
        out[(link.target, flip[link.target_orient])].append(
            (link.source, flip[link.source_orient], link.overlap)
        )
    windows = []
    for name, seq in segments.items():
        if len(seq) >= 2 * flank:
            continue
        for orient in "+-":
            preds = [(x, flip[xo], ov) for x, xo, ov in out[(name, flip[orient])]]
            for x, xo, ov1 in preds:
                left = oriented(segments, x, xo)
                # chains of short middles starting at (name, orient)
                stack = [([(name, orient)], oriented(segments, name, orient)[ov1:])]
                while stack:
                    chain, middle = stack.pop()
                    last = chain[-1]
                    for y, yo, ov2 in out[last]:
                        right = oriented(segments, y, yo)
                        walk = left + middle + right[ov2:]
                        start = max(0, len(left) - flank)
                        windows.append(walk[start : len(left) + len(middle) + flank])
                        grown = middle + right[ov2:]
                        if (
                            len(chain) < max_middle
                            and len(segments[y]) < 2 * flank
                            and len(grown) < 2 * flank
                        ):
                            stack.append(([*chain, (y, yo)], grown))
    return windows


def recovered_by_alignment(
    gold_segments,
    gold_links,
    cand_segments,
    cand_links,
    flank: int,
    minimap2: str,
    min_cover: float = 0.9,
    min_identity: float = 0.98,
    tolerant: bool = False,
) -> list[bool]:
    """A gold junction is recovered when its spanning sequence (flank bp each side)
    aligns contiguously to one candidate segment or candidate junction sequence.

    With ``tolerant``, candidate junctions keep ``4 * flank`` bp per side (the gold side
    stays at ``flank``) and two-link walks through a short middle segment are added, so a
    breakpoint placed up to ``3 * flank`` bp away (e.g. at the other edge of a sub-repeat; on
    Nipponbare the ctg7/ctg9 starts share ~0.9-1.5 kb) or split over two links still matches.
    The gold sequence must still align contiguously, i.e. the candidate must hold that adjacency.

    Tolerates assemblers placing segment breakpoints at different repeat edges and
    a few base differences, which exact flank matching cannot.
    """
    cand_flank = 4 * flank if tolerant else flank
    with tempfile.TemporaryDirectory() as tmp:
        query, target = Path(tmp, "gold_junctions.fa"), Path(tmp, "candidate.fa")
        query.write_text(
            "".join(
                f">J{i:02d}\n{''.join(junction_flanks(gold_segments, link, flank))}\n"
                for i, link in enumerate(gold_links, 1)
            )
        )
        target.write_text(
            "".join(f">seg_{name}\n{seq}\n" for name, seq in cand_segments.items())
            + "".join(
                f">junc_{i}\n{''.join(junction_flanks(cand_segments, link, cand_flank))}\n"
                for i, link in enumerate(cand_links, 1)
            )
            + "".join(
                f">walk_{i}\n{seq}\n"
                for i, seq in enumerate(
                    two_step_windows(cand_segments, cand_links, cand_flank) if tolerant else [], 1
                )
            )
        )
        paf = _run([minimap2, "-x", "asm5", "-c", "--eqx", "-N", "50", str(target), str(query)])
    best: dict[str, bool] = defaultdict(bool)
    for row in paf.splitlines():
        f = row.split("\t")
        cover = (int(f[3]) - int(f[2])) / int(f[1])
        identity = int(f[9]) / int(f[10]) if int(f[10]) else 0.0
        best[f[0]] |= cover >= min_cover and identity >= min_identity
    return [best[f"J{i:02d}"] for i in range(1, len(gold_links) + 1)]


def score_graphs(
    gold_gfa: Path,
    candidate_gfa: Path,
    flank: int,
    match: str = "exact",
    minimap2: str = "minimap2",
) -> dict:
    gold_segments, gold_links = parse_gfa(gold_gfa)
    if not gold_links:
        raise ValueError(f"no GFA L records in the gold graph {gold_gfa}")
    # a fragmented candidate may have no links at all: its junctions are simply not recovered
    cand_segments, cand_links = parse_gfa(candidate_gfa)
    if match == "exact":
        recovered = recovered_exact(gold_segments, gold_links, cand_segments, cand_links, flank)
    else:
        recovered = recovered_by_alignment(
            gold_segments, gold_links, cand_segments, cand_links, flank, minimap2
        )
    junction_ids = [f"J{i:02d}" for i in range(1, len(gold_links) + 1)]
    walk_recovered = (
        recovered_by_alignment(
            gold_segments, gold_links, cand_segments, cand_links, flank, minimap2, tolerant=True
        )
        if match == "align"
        else recovered
    )
    branches = {}
    for index, (endpoint, (a, b)) in enumerate(sorted(branch_points(gold_links).items()), 1):
        branches[f"B{index:02d}"] = {
            "endpoint": "".join(endpoint),
            "junctions": [junction_ids[a], junction_ids[b]],
            "both_recovered": recovered[a] and recovered[b],
            "both_recovered_tolerant": walk_recovered[a] and walk_recovered[b],
        }
    true_positive = sum(recovered)
    return {
        "match": match,
        "flank_bp": flank,
        "gold_junctions": len(gold_links),
        "candidate_junctions": len(cand_links),
        "junctions_recovered": true_positive,
        "junction_recall": true_positive / len(gold_links),
        # alignment matching is gold-anchored; candidate-side precision is not defined there
        "junction_precision": (true_positive / len(cand_links) if cand_links else 0.0)
        if match == "exact"
        else None,
        "missing_junctions": [j for j, ok in zip(junction_ids, recovered, strict=True) if not ok],
        "branches_total": len(branches),
        "branches_fully_recovered": sum(b["both_recovered"] for b in branches.values()),
        # Breakpoint-tolerant matching (see recovered_by_alignment); the fields above keep the
        # original fixed-window definition for comparison with earlier reports.
        "junctions_recovered_tolerant": sum(walk_recovered),
        "missing_junctions_tolerant": [
            j for j, ok in zip(junction_ids, walk_recovered, strict=True) if not ok
        ],
        "branches_fully_recovered_tolerant": sum(
            b["both_recovered_tolerant"] for b in branches.values()
        ),
        "branches": branches,
    }


#: Indels at least this long are structural differences, not base errors.
LARGE_INDEL_BP = 1000


def compare_linear(gold_fasta: Path, candidate_fasta: Path, minimap2: str) -> dict:
    """Both-strand asm5 alignment of the candidate to the gold, per gold record.

    Either FASTA may hold several molecules (chromosomes); coverage is merged per gold record
    and the fraction is over all of them, unaligned records included. Reverse-strand blocks
    are kept as inversion evidence, and indels of at least ``LARGE_INDEL_BP`` are listed as
    structural differences rather than counted as base errors.
    """
    paf = _run([minimap2, "-x", "asm5", "-c", "--eqx", str(gold_fasta), str(candidate_fasta)])
    gold_lengths: dict[str, int] = {}
    name = None
    for line in gold_fasta.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            gold_lengths[name] = 0
        elif name is not None:
            gold_lengths[name] += len(line.strip())
    covered: dict[str, list[tuple[int, int]]] = {}
    matches = block = blocks = 0
    eq = mismatch = gap_opens = 0
    inverted: list[dict] = []
    large_indels: list[dict] = []
    for row in paf.splitlines():
        f = row.split("\t")
        covered.setdefault(f[5], []).append((int(f[7]), int(f[8])))
        matches += int(f[9])
        block += int(f[10])
        blocks += 1
        primary = "tp:A:P" in f[12:]
        # --eqx CIGAR: '=' match, 'X' mismatch; each I/D run counts once, so a 66 kb
        # deletion inside one block is one difference, not 66 kb of errors.
        cigar = next((tag[5:] for tag in f[12:] if tag.startswith("cg:Z:")), "")
        ref_pos = int(f[7])
        for length, op in re.findall(r"(\d+)([=XIDM])", cigar):
            length = int(length)
            if op == "=":
                eq += length
            elif op == "X":
                mismatch += length
            elif op in "ID":
                gap_opens += 1
                if length >= LARGE_INDEL_BP:
                    large_indels.append(
                        {
                            "type": "deletion_in_candidate"
                            if op == "D"
                            else "insertion_in_candidate",
                            "length": length,
                            "reference": f[5],
                            "reference_pos": ref_pos,
                            "query": f[0],
                            "is_primary": primary,
                        }
                    )
            if op in "=XMD":
                ref_pos += length
        if f[4] == "-":
            inverted.append(
                {
                    "query": f[0],
                    "query_start": int(f[2]),
                    "query_end": int(f[3]),
                    "reference": f[5],
                    "reference_start": int(f[7]),
                    "reference_end": int(f[8]),
                    "strand": "-",
                    "is_primary": primary,
                }
            )
    merged = 0
    for spans in covered.values():
        end = -1
        for start, stop in sorted(spans):
            if stop > end:
                merged += stop - max(start, end)
                end = stop
    gold_len = sum(gold_lengths.values())
    return {
        "gold_length": gold_len,
        "gold_records": len(gold_lengths),
        "gold_aligned_fraction": merged / gold_len if gold_len else 0.0,
        "alignment_identity": matches / block if block else None,
        # BLAST-style identity above counts every indel base; this one counts each gap once
        # and separates structural indels from base errors.
        "gap_compressed_identity": eq / (eq + mismatch + gap_opens) if eq else None,
        "large_indel_threshold_bp": LARGE_INDEL_BP,
        "large_indels": large_indels,
        "alignment_blocks": blocks,
        "inverted_blocks": inverted,
    }
