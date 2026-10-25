"""Repeat-mediated recombination conformations on an assembly graph.

A recombining repeat is a GFA segment ``R`` that can be entered from at least
two oriented predecessors and left towards at least two oriented successors.
Every predecessor/successor pair ``p -> R -> s`` is one way a molecule can
pass through the repeat; for a 2x2 repeat the two perfect matchings of
predecessors to successors are its two conformations (the assembled one and
the recombined one).

Read evidence is the published flank-spanning count: each pair is turned into
a probe ``tail(p) + R + head(s)``, long reads are aligned to all probes with
minimap2, and a read supports a pair only when one alignment covers the whole
repeat plus ``anchor`` bases of unique flank on both sides.  A conformation's
support is the sum over its two pairs; proportions are over the two
conformations.  Without reads, conformations are reported as candidates only.

Molecules for a conformation are spelled by walking the graph with the state
``(previous oriented segment, current oriented segment)``: at a resolved repeat
the chosen matching decides the successor, elsewhere the successor must be
unique.  Adjacent and inverted repeats need no node copying in this walk.
"""

from __future__ import annotations

import itertools
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import OrganelleDependencyError, OrganelleInputError, OrganelleParameterError

Oriented = tuple[str, str]

READ_PRESETS = {"hifi": "map-hifi", "ont": "map-ont", "clr": "map-pb"}
DEFAULT_MIN_IDENTITY = {"hifi": 0.95, "ont": 0.80, "clr": 0.75}
REPEAT_DEPTH_RATIO = 1.5

#: What a conformation ``proportion`` is, per support source. It is read
#: support for one pairing of a repeat's junctions, not the abundance of whole
#: circular molecules: no read covers a whole mitochondrial circle.
PROPORTION_DEFINITIONS = {
    "reads": (
        "numerator: reads whose alignment covers the whole repeat plus >= anchor bp of unique "
        "flank on both sides and whose best-scoring flank pair belongs to this conformation's two "
        "predecessor>repeat>successor pairings; denominator: all such reads assigned to either "
        "conformation of this repeat (ambiguous ties excluded). Read support for a junction "
        "pairing at this repeat, not whole-molecule abundance."
    ),
    "pairings": (
        "numerator: ovasm evidence.json pairing_reads, reads traversing entry>repeat>exit for this "
        "conformation's two pairings; denominator: pairing reads of both conformations. Read support "
        "for a junction pairing at this repeat, not whole-molecule abundance."
    ),
}


@dataclass(frozen=True)
class Segment:
    name: str
    sequence: str
    length: int
    depth: float


@dataclass(frozen=True)
class Link:
    source: str
    source_orient: str
    target: str
    target_orient: str
    overlap: int
    evidence: int | None = None


@dataclass
class AssemblyGraph:
    segments: dict[str, Segment]
    links: list[Link]
    paths: list[tuple[str, list[Oriented]]] = field(default_factory=list)
    successors: dict[Oriented, list[tuple[Oriented, int]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        succ: dict[Oriented, list[tuple[Oriented, int]]] = defaultdict(list)
        for link in self.links:
            a = (link.source, link.source_orient)
            b = (link.target, link.target_orient)
            _add_unique(succ[a], (b, link.overlap))
            _add_unique(succ[flip(b)], (flip(a), link.overlap))
        self.successors = dict(succ)

    def succ(self, node: Oriented) -> list[Oriented]:
        return [target for target, _ in self.successors.get(node, [])]

    def pred(self, node: Oriented) -> list[Oriented]:
        return [flip(source) for source in self.succ(flip(node))]

    def overlap(self, left: Oriented, right: Oriented) -> int:
        for target, overlap in self.successors.get(left, []):
            if target == right:
                return overlap
        raise KeyError(f"no link {left} -> {right}")

    def oriented_sequence(self, node: Oriented) -> str:
        sequence = self.segments[node[0]].sequence
        return sequence if node[1] == "+" else reverse_complement(sequence)


def _add_unique(items: list, item) -> None:
    if item not in items:
        items.append(item)


def flip(node: Oriented) -> Oriented:
    return (node[0], "-" if node[1] == "+" else "+")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def node_label(node: Oriented) -> str:
    return f"{node[0]}{node[1]}"


def parse_node_label(label: str) -> Oriented:
    return (label[:-1], label[-1])


# --------------------------------------------------------------------------- GFA


def read_assembly_graph(path: str | Path) -> AssemblyGraph:
    """Parse GFA1 S/L/P lines, keeping sequences, depth and link evidence."""
    segments: dict[str, Segment] = {}
    links: list[Link] = []
    paths: list[tuple[str, list[Oriented]]] = []
    for raw in Path(path).read_text().splitlines():
        fields = raw.split("\t")
        if fields[0] == "S":
            tags = _tags(fields[3:])
            sequence = "" if fields[2] == "*" else fields[2].upper()
            length = int(tags.get("LN", len(sequence)))
            segments[fields[1]] = Segment(fields[1], sequence, length, _depth(tags, length))
        elif fields[0] == "L":
            tags = _tags(fields[6:])
            evidence = tags.get("ev")
            links.append(
                Link(
                    fields[1],
                    fields[2],
                    fields[3],
                    fields[4],
                    _overlap(fields[5]),
                    int(evidence) if evidence is not None else None,
                )
            )
        elif fields[0] == "P":
            steps = [parse_node_label(step) for step in fields[2].split(",") if step]
            paths.append((fields[1], steps))
    if not segments:
        raise OrganelleInputError(
            code="structure.graph_empty",
            message="The GFA file declares no segments.",
            details={"path": str(path)},
        )
    for link in links:
        for name in (link.source, link.target):
            if name not in segments:
                raise OrganelleInputError(
                    code="structure.graph_dangling_link",
                    message="A GFA link names a segment that is not declared.",
                    details={"segment": name},
                )
    return AssemblyGraph(segments, links, paths)


def _tags(fields: Sequence[str]) -> dict[str, float]:
    tags: dict[str, float] = {}
    for item in fields:
        parts = item.split(":", 2)
        if len(parts) == 3 and parts[1] in {"i", "f"}:
            tags[parts[0]] = float(parts[2])
    return tags


def _depth(tags: Mapping[str, float], length: int) -> float:
    """Per-base depth: ``dp``/``DP`` as given, ``KC``/``RC``/``FC`` counts per base."""
    for key in ("dp", "DP"):
        if key in tags:
            return tags[key]
    for key in ("RC", "FC", "KC"):
        if key in tags and length:
            return tags[key] / length
    return 0.0


def _overlap(cigar: str) -> int:
    if cigar in {"*", ""}:
        return 0
    if not cigar.endswith("M") or not cigar[:-1].isdigit():
        raise OrganelleInputError(
            code="structure.graph_overlap",
            message="Only exact-match GFA link overlaps (<n>M) are supported.",
            details={"overlap": cigar},
        )
    return int(cigar[:-1])


# ----------------------------------------------------------------------- repeats


@dataclass
class Repeat:
    name: str
    predecessors: list[Oriented]
    successors: list[Oriented]
    copy_ratio: float

    @property
    def kind(self) -> str:
        return "repeat" if self.copy_ratio >= REPEAT_DEPTH_RATIO else "branch"

    def pairs(self) -> list[tuple[Oriented, Oriented]]:
        return [(p, s) for p in self.predecessors for s in self.successors]

    def configurations(self) -> list[list[tuple[Oriented, Oriented]]]:
        """Perfect matchings of a 2x2 repeat; other shapes have no defined pairing."""
        if len(self.predecessors) != 2 or len(self.successors) != 2:
            return []
        (p1, p2), (s1, s2) = self.predecessors, self.successors
        return [[(p1, s1), (p2, s2)], [(p1, s2), (p2, s1)]]


def single_copy_depth(graph: AssemblyGraph) -> float:
    """Length-weighted median depth over all segments."""
    rows = sorted((s.depth, s.length) for s in graph.segments.values() if s.length > 0)
    total = sum(length for _, length in rows)
    seen = 0
    for depth, length in rows:
        seen += length
        if seen * 2 >= total:
            return depth
    return 0.0


def find_repeats(graph: AssemblyGraph) -> list[Repeat]:
    """Segments that branch on both ends, in GFA order."""
    base = single_copy_depth(graph)
    repeats = []
    for name, segment in graph.segments.items():
        node = (name, "+")
        preds, succs = graph.pred(node), graph.succ(node)
        if len(preds) >= 2 and len(succs) >= 2:
            ratio = segment.depth / base if base > 0 else 0.0
            repeats.append(Repeat(name, preds, succs, ratio))
    return repeats


def assembled_pairs(graph: AssemblyGraph, repeat: Repeat) -> set[tuple[Oriented, Oriented]]:
    """Pairs the first declared GFA path through ``repeat`` walks.

    Assemblers that emit alternative paths (one per candidate conformation)
    list their reported one first; mixing paths would mix conformations.
    """
    node = (repeat.name, "+")
    for _, steps in graph.paths:
        found: set[tuple[Oriented, Oriented]] = set()
        circular = len(steps) > 1 and steps[0] in graph.succ(steps[-1])
        count = len(steps)
        for index, step in enumerate(steps):
            has_prev = index > 0 or circular
            has_next = index < count - 1 or circular
            if not (has_prev and has_next):
                continue
            prev, nxt = steps[index - 1], steps[(index + 1) % count]
            if step == node:
                found.add((prev, nxt))
            elif step == flip(node):
                found.add((flip(nxt), flip(prev)))
        if found:
            return found
    return set()


# ------------------------------------------------------------------ read support


@dataclass(frozen=True)
class Probe:
    repeat: str
    pair: tuple[Oriented, Oriented]
    sequence: str
    left_flank: int
    repeat_length: int
    right_flank: int


def build_probes(graph: AssemblyGraph, repeat: Repeat, flank: int) -> list[Probe]:
    node = (repeat.name, "+")
    core = graph.oriented_sequence(node)
    if not core:
        raise OrganelleInputError(
            code="structure.graph_sequence_missing",
            message="Read support needs segment sequences in the GFA.",
            details={"segment": repeat.name},
        )
    probes = []
    for left, right in repeat.pairs():
        left_seq = graph.oriented_sequence(left)
        right_seq = graph.oriented_sequence(right)
        left_tail = left_seq[: len(left_seq) - graph.overlap(left, node)][-flank:]
        right_head = right_seq[graph.overlap(node, right) :][:flank]
        probes.append(
            Probe(
                repeat.name,
                (left, right),
                left_tail + core + right_head,
                len(left_tail),
                len(core),
                len(right_head),
            )
        )
    return probes


@dataclass(frozen=True)
class Alignment:
    read: str
    read_length: int
    target: int
    target_start: int
    target_end: int
    matches: int
    block: int
    score: int


def parse_paf(lines: Iterable[str]) -> Iterable[Alignment]:
    """PAF rows from ``minimap2 -c``; the DP score ``AS`` ranks competing probes."""
    for line in lines:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 12:
            continue
        tags = dict(item.split(":", 2)[::2] for item in fields[12:])
        yield Alignment(
            fields[0],
            int(fields[1]),
            int(fields[5]),
            int(fields[7]),
            int(fields[8]),
            int(fields[9]),
            int(fields[10]),
            int(tags["AS"]),
        )


def count_spanning_reads(
    probes: Sequence[Probe],
    alignments: Iterable[Alignment],
    *,
    anchor: int,
    min_identity: float,
) -> dict[str, dict[str, Any]]:
    """Per repeat: reads spanning each pair, ambiguous reads, longest touching read.

    A read that spans several probes of one repeat is assigned to the probe
    with the highest alignment score; equal best scores make it ambiguous.
    """
    spans: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
    touching: dict[str, set[str]] = defaultdict(set)
    longest: dict[str, int] = defaultdict(int)
    for hit in alignments:
        probe = probes[hit.target]
        repeat_end = probe.left_flank + probe.repeat_length
        if hit.target_end <= probe.left_flank or hit.target_start >= repeat_end:
            continue
        touching[probe.repeat].add(hit.read)
        longest[probe.repeat] = max(longest[probe.repeat], hit.read_length)
        left = probe.left_flank - min(anchor, probe.left_flank)
        right = repeat_end + min(anchor, probe.right_flank)
        if (
            hit.target_start <= left
            and hit.target_end >= right
            and hit.matches >= min_identity * hit.block
        ):
            scores = spans[(probe.repeat, hit.read)]
            scores[hit.target] = max(hit.score, scores.get(hit.target, hit.score))
    out: dict[str, dict[str, Any]] = {
        name: {"pairs": defaultdict(int), "ambiguous": 0, "touching": 0, "longest": 0}
        for name in {probe.repeat for probe in probes}
    }
    for (repeat, _), scores in spans.items():
        best = max(scores.values())
        winners = [target for target, score in scores.items() if score == best]
        if len(winners) == 1:
            out[repeat]["pairs"][winners[0]] += 1
        else:
            out[repeat]["ambiguous"] += 1
    for repeat, reads in touching.items():
        out[repeat]["touching"] = len(reads)
        out[repeat]["longest"] = longest[repeat]
    return out


def run_minimap2(
    probes: Sequence[Probe],
    reads: Sequence[Path],
    *,
    preset: str,
    threads: int,
    minimap2: str,
) -> tuple[list[Alignment], str]:
    with tempfile.TemporaryDirectory(prefix="ov_conformation_") as tmp:
        reference = Path(tmp) / "probes.fa"
        reference.write_text(
            "".join(f">{index}\n{probe.sequence}\n" for index, probe in enumerate(probes))
        )
        # Probes of one repeat share its sequence, so MAPQ is low by design and
        # secondaries must be kept: -p 0 lets a short spanning hit survive next
        # to a longer hit on another repeat's probe.
        argv = [
            minimap2,
            "-x",
            preset,
            "-c",
            "--secondary=yes",
            "-p",
            "0",
            "-N",
            "100",
            "-t",
            str(threads),
            str(reference),
            *map(str, reads),
        ]
        process = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        assert process.stdout is not None
        alignments = list(parse_paf(process.stdout))
        stderr = process.stderr.read() if process.stderr else ""
        if process.wait() != 0:
            raise OrganelleInputError(
                code="structure.minimap2_failed",
                message=stderr.strip()[-1000:] or "minimap2 failed",
                details={"argv": argv},
            )
    version = subprocess.run([minimap2, "--version"], capture_output=True, text=True).stdout.strip()
    return alignments, version


# ------------------------------------------------------------------- molecules


def spell_molecules(
    graph: AssemblyGraph,
    choice: Mapping[str, Sequence[tuple[Oriented, Oriented]]],
) -> list[dict[str, Any]] | None:
    """Molecules implied by resolving each repeat with its chosen matching.

    Returns ``None`` when a segment outside ``choice`` still branches on
    either end, i.e. the graph does not spell a unique set of molecules.
    """
    rules: dict[Oriented, dict[Oriented, Oriented]] = {}
    for name, matching in choice.items():
        forward = dict(matching)
        rules[(name, "+")] = forward
        rules[(name, "-")] = {flip(s): flip(p) for p, s in forward.items()}
    for name in graph.segments:
        node = (name, "+")
        if node not in rules and (len(graph.succ(node)) > 1 or len(graph.pred(node)) > 1):
            return None

    def step(prev: Oriented, cur: Oriented) -> Oriented | None:
        if cur in rules:
            return rules[cur].get(prev)
        options = graph.succ(cur)
        return options[0] if options else None

    used: set[tuple[Oriented, Oriented]] = set()
    molecules = []
    for link in graph.links:
        a, b = (link.source, link.source_orient), (link.target, link.target_orient)
        if (a, b) in used:
            continue
        walk, circular = _walk(a, b, step)
        closing = [(walk[-1], walk[0])] if circular else []
        for left, right in [*itertools.pairwise(walk), *closing]:
            used.add((left, right))
            used.add((flip(right), flip(left)))
        molecules.append(_molecule(graph, walk, circular))
    visited = {parse_node_label(label)[0] for m in molecules for label in m["steps"]}
    for name in graph.segments:
        if name not in visited:
            molecules.append(_molecule(graph, [(name, "+")], False))
    return sorted(molecules, key=lambda m: -m["length"])


def _walk(a: Oriented, b: Oriented, step) -> tuple[list[Oriented], bool]:
    """Walk from link a->b until it closes on itself or ends on both sides.

    Transitions are injective (every unresolved segment has at most one
    predecessor and successor, and repeat matchings are bijections), so a walk
    that does not end returns to its starting state.
    """
    walk = [a, b]
    prev, cur = a, b
    while (nxt := step(prev, cur)) is not None:
        if (cur, nxt) == (a, b):
            return walk[:-1], True
        walk.append(nxt)
        prev, cur = cur, nxt
    back: list[Oriented] = []
    prev, cur = flip(b), flip(a)
    while (nxt := step(prev, cur)) is not None:
        back.append(flip(nxt))
        prev, cur = cur, nxt
    return [*reversed(back), *walk], False


def _molecule(graph: AssemblyGraph, walk: Sequence[Oriented], circular: bool) -> dict[str, Any]:
    length = sum(graph.segments[name].length for name, _ in walk)
    pairs = list(itertools.pairwise(walk))
    if circular:
        pairs.append((walk[-1], walk[0]))
    length -= sum(graph.overlap(left, right) for left, right in pairs)
    return {
        "circular": circular,
        "length": length,
        "steps": [node_label(node) for node in walk],
    }


# ------------------------------------------------------------------- top level


def analyze_conformations(
    gfa: str | Path,
    *,
    reads: Sequence[str | Path] = (),
    read_type: str = "hifi",
    flank: int = 1000,
    anchor: int = 500,
    min_identity: float | None = None,
    min_spanning_reads: int = 3,
    threads: int = 4,
    minimap2: str | Path | None = None,
    pairings: Mapping[str, Sequence[tuple[Sequence[tuple[str, str]], int]]] | None = None,
    plastid: str | Path | None = None,
    mtpt_min_length: int = 100,
    mtpt_min_identity: float = 80.0,
) -> dict[str, Any]:
    """Graph, repeats, conformation support and molecules; MTPTs when a plastid is given.

    Conformation support comes from ``reads`` (spanning-read counts) when
    given, otherwise from ``pairings`` (per-repeat matchings with read counts,
    e.g. ovasm ``evidence.json``), otherwise the conformations are candidates.
    """
    from .graph_origin import find_mtpts, mtpt_probes, plastid_reference, segment_origin

    if read_type not in READ_PRESETS:
        raise OrganelleParameterError(
            code="structure.read_type",
            message="read_type must be one of hifi, ont, clr.",
            details={"read_type": read_type},
        )
    if anchor <= 0 or flank < anchor:
        raise OrganelleParameterError(
            code="structure.anchor",
            message="anchor must be positive and not larger than flank.",
            details={"anchor": anchor, "flank": flank},
        )
    graph = read_assembly_graph(gfa)
    repeats = find_repeats(graph)
    base = single_copy_depth(graph)
    identity = DEFAULT_MIN_IDENTITY[read_type] if min_identity is None else min_identity
    read_paths = [Path(p) for p in reads]

    mtpt = None
    reference = None
    if plastid is not None:
        reference = plastid_reference(plastid)
        mtpt = find_mtpts(graph, reference, min_length=mtpt_min_length, min_identity=mtpt_min_identity)

    counts: dict[str, dict[str, Any]] = {}
    probes: list[Probe] = []
    evidence: dict[str, Any] | None = None
    if read_paths:
        executable = str(minimap2) if minimap2 else shutil.which("minimap2")
        if not executable:
            raise OrganelleDependencyError(
                code="structure.minimap2_missing",
                message="minimap2 is required to count repeat-spanning reads.",
                details={"tool": "minimap2"},
                retryable=False,
                suggested_action={"install": "mamba install -c bioconda minimap2"},
            )
        for repeat in repeats:
            probes.extend(build_probes(graph, repeat, flank))
        if mtpt is not None:
            probes.extend(mtpt_probes(graph, mtpt["fragments"], flank=flank))
        alignments, version = (
            run_minimap2(
                probes, read_paths, preset=READ_PRESETS[read_type], threads=threads, minimap2=executable
            )
            if probes
            else ([], "")
        )
        counts = count_spanning_reads(probes, alignments, anchor=anchor, min_identity=identity)
        evidence = {
            "source": "reads",
            "reads": [str(p) for p in read_paths],
            "read_type": read_type,
            "preset": READ_PRESETS[read_type],
            "flank": flank,
            "anchor": anchor,
            "min_identity": identity,
            "min_spanning_reads": min_spanning_reads,
            "minimap2": version,
            "alignments": len(alignments),
        }
    elif pairings:
        evidence = {"source": "pairings", "min_spanning_reads": min_spanning_reads}

    probe_index = {(p.repeat, p.pair): i for i, p in enumerate(probes)}
    rows = []
    for repeat in repeats:
        rows.append(
            _repeat_row(
                graph,
                repeat,
                counts.get(repeat.name),
                probe_index,
                anchor,
                min_spanning_reads,
                bool(read_paths),
                (pairings or {}).get(repeat.name) if not read_paths else None,
            )
        )
    chosen = {row["name"]: row["reference_pairs"] for row in rows if row["reference_pairs"]}
    for row in rows:
        for conf in row["configurations"]:
            matching = [tuple(map(parse_node_label, pair)) for pair in conf["pairs"]]
            molecules = spell_molecules(graph, {**chosen, row["name"]: matching})
            conf["molecules"] = (
                None
                if molecules is None
                else [m for m in molecules if any(s[:-1] == row["name"] for s in m["steps"])]
            )
        del row["reference_pairs"]
    master = spell_molecules(graph, chosen)

    segments = []
    for s in graph.segments.values():
        row = {
            "name": s.name,
            "length": s.length,
            "depth": round(s.depth, 3),
            "copy_ratio": round(s.depth / base, 3) if base else 0.0,
        }
        if mtpt is not None:
            fraction = mtpt["plastid_fraction"][s.name]
            row["plastid_fraction"] = fraction
            row["origin"] = segment_origin(fraction)
        segments.append(row)

    plastid_out = None
    if mtpt is not None and reference is not None:
        for fragment in mtpt["fragments"]:
            _mtpt_support(
                fragment,
                graph.segments[fragment["segment"]].length,
                counts.get(fragment["id"]),
                anchor,
                min_spanning_reads,
                bool(read_paths),
            )
        plastid_out = {
            "reference": {k: v for k, v in reference.items() if k != "sequence"},
            **{k: v for k, v in mtpt.items() if k != "plastid_fraction"},
        }

    return {
        "graph": {
            "segments": len(graph.segments),
            "links": len(graph.links),
            "paths": len(graph.paths),
            "total_length": sum(s.length for s in graph.segments.values()),
            "single_copy_depth": round(base, 3),
        },
        "segments": segments,
        "links": [
            {
                "source": node_label((lk.source, lk.source_orient)),
                "target": node_label((lk.target, lk.target_orient)),
                "overlap": lk.overlap,
                "evidence": lk.evidence,
            }
            for lk in graph.links
        ],
        "repeats": rows,
        "master_molecules": master,
        "read_evidence": evidence,
        "plastid": plastid_out,
    }


def _mtpt_support(fragment, segment_length, counts, anchor, min_reads, has_reads) -> None:
    """Reads covering an MTPT plus ``anchor`` mitochondrial bases on both sides.

    An MTPT closer than ``anchor`` to a segment end has no mitochondrial anchor
    inside its segment on that side, so plastid reads could pass as spanning;
    it is reported as not assessable instead of counted.
    """
    if fragment["start"] - 1 < anchor or segment_length - fragment["end"] < anchor:
        fragment["spanning_reads"] = None
        fragment["integration"] = "segment_end"
        return
    if not has_reads:
        fragment["spanning_reads"] = None
        fragment["integration"] = "not_assessed"
        return
    reads = sum(counts["pairs"].values()) if counts else 0
    fragment["spanning_reads"] = reads
    fragment["integration"] = "supported" if reads >= min_reads else "unsupported"


def _repeat_row(graph, repeat, counts, probe_index, anchor, min_reads, has_reads, pairing=None):
    segment = graph.segments[repeat.name]
    required = segment.length + 2 * anchor
    pair_reads = {}
    for pair in repeat.pairs():
        index = probe_index.get((repeat.name, pair))
        pair_reads[pair] = counts["pairs"].get(index, 0) if counts and index is not None else 0
    spanning = sum(pair_reads.values())
    by_matching = (
        {frozenset((parse_node_label(a), parse_node_label(b)) for a, b in pairs): n for pairs, n in pairing}
        if pairing
        else {}
    )
    source = "reads" if has_reads else "pairings" if by_matching else None
    configurations = []
    assembled = assembled_pairs(graph, repeat)
    for index, matching in enumerate(repeat.configurations(), start=1):
        if has_reads:
            support = sum(pair_reads[pair] for pair in matching)
        elif by_matching:
            support = by_matching.get(frozenset(matching))
        else:
            support = None
        configurations.append(
            {
                "id": index,
                "pairs": [[node_label(p), node_label(s)] for p, s in matching],
                "assembled": bool(assembled) and set(matching) == assembled,
                "supporting_reads": support,
            }
        )
    if source == "pairings" and any(c["supporting_reads"] is None for c in configurations):
        raise OrganelleInputError(
            code="structure.pairing_mismatch",
            message="Repeat pairings do not match the repeat's graph neighbours.",
            details={"repeat": repeat.name},
        )
    conf_total = sum(c["supporting_reads"] or 0 for c in configurations)
    longest = counts["longest"] if counts else 0
    if source is None:
        status = "candidate_only"
    elif not configurations:
        status = "complex"
    elif source == "reads" and spanning == 0 and longest < required:
        status = "not_spannable"
    elif conf_total >= min_reads:
        status = "supported"
    else:
        status = "insufficient_reads"
    for conf in configurations:
        conf["proportion"] = (
            round(conf["supporting_reads"] / conf_total, 4)
            if status == "supported" and conf_total
            else None
        )
        conf["proportion_denominator"] = conf_total if source else None
    configurations.sort(key=lambda c: (not c["assembled"], c["id"]))
    reference = None
    if configurations:
        assembled_conf = [c for c in configurations if c["assembled"]]
        best = assembled_conf[0] if assembled_conf else max(
            configurations, key=lambda c: (c["supporting_reads"] or 0, -c["id"])
        )
        reference = [tuple(map(parse_node_label, pair)) for pair in best["pairs"]]
    return {
        "name": repeat.name,
        "kind": repeat.kind,
        "length": segment.length,
        "depth": round(segment.depth, 3),
        "copy_ratio": round(repeat.copy_ratio, 3),
        "predecessors": [node_label(n) for n in repeat.predecessors],
        "successors": [node_label(n) for n in repeat.successors],
        "support_source": source,
        "pair_reads": [
            {"left": node_label(p), "right": node_label(s), "reads": n}
            for (p, s), n in pair_reads.items()
        ]
        if has_reads
        else [],
        "spanning_reads": spanning if has_reads else (conf_total if source else None),
        "ambiguous_reads": counts["ambiguous"] if counts else (0 if has_reads else None),
        "touching_reads": counts["touching"] if counts else (0 if has_reads else None),
        "longest_touching_read": longest if has_reads else None,
        "required_read_length": required,
        "status": status,
        "proportion_definition": PROPORTION_DEFINITIONS.get(source) if source else None,
        "configurations": configurations,
        "reference_pairs": reference,
    }


__all__ = [
    "AssemblyGraph",
    "analyze_conformations",
    "build_probes",
    "count_spanning_reads",
    "find_repeats",
    "parse_paf",
    "read_assembly_graph",
    "single_copy_depth",
    "spell_molecules",
]
