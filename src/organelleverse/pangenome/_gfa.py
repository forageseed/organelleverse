"""Strict GFA1 sequence-graph analytics (S/L/P/W, not jump/containment graphs).

Topology and binary PAV do not require stored sequence. Spelling requires stored
IUPAC DNA and exact M/= overlaps: indels, mismatches and unknown overlaps have no
unique consensus and are rejected, not silently linearized. W coordinates remain
native molecule coordinates; different W intervals are never concatenated.
Specification: https://gfa-spec.github.io/GFA-spec/GFA1.html
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

_STEP = tuple[str, str]
_CIGAR = re.compile(r"(?:[0-9]+[MIDNSHPX=])+")
_OPS = re.compile(r"([0-9]+)([MIDNSHPX=])")
_DNA = re.compile(r"[ACGTRYSWKMBDHVNacgtryswkmbdhvn]+")
_COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVNacgtryswkmbdhvn", "TGCAYRSWMKVHDBNtgcayrswmkvhdbn")


@dataclass(frozen=True)
class Segment:
    name: str
    sequence: str | None
    length: int | None
    raw: str


@dataclass(frozen=True)
class Link:
    source: _STEP
    target: _STEP
    overlap: str
    raw: str


@dataclass(frozen=True)
class GraphPath:
    name: str
    steps: tuple[_STEP, ...]
    overlaps: tuple[str, ...] | None
    sample: str
    molecule: str
    kind: str
    raw: str
    haplotype: str | None = None
    start: int | None = None
    end: int | None = None


@dataclass
class GFAGraph:
    segments: dict[str, Segment] = field(default_factory=dict)
    links: list[Link] = field(default_factory=list)
    paths: dict[str, GraphPath] = field(default_factory=dict)
    headers: list[str] = field(default_factory=list)
    transitions: dict[tuple[_STEP, _STEP], set[str]] = field(default_factory=dict)


def _flip(step: _STEP) -> _STEP:
    return step[0], "-" if step[1] == "+" else "+"


def _reverse_cigar(cigar: str) -> str:
    if cigar == "*":
        return cigar
    swap = {"I": "D", "D": "I"}
    return "".join(n + swap.get(op, op) for n, op in reversed(_OPS.findall(cigar)))


def _name(value: str) -> str:
    if not value or value[0] in "*=" or any(ord(c) < 33 or ord(c) > 126 for c in value):
        raise ValueError(f"invalid GFA name: {value!r}")
    if "+," in value or "-," in value:
        raise ValueError(f"invalid GFA name: {value!r}")
    return value


def _step(node: str, orient: str) -> _STEP:
    if orient not in ("+", "-"):
        raise ValueError(f"invalid orientation: {orient!r}")
    return _name(node), orient


def _cigar(value: str) -> str:
    if value != "*" and not _CIGAR.fullmatch(value):
        raise ValueError(f"invalid overlap CIGAR: {value!r}")
    return value


def _tags(fields: list[str]) -> dict[str, tuple[str, str]]:
    result = {}
    for text in fields:
        parts = text.split(":", 2)
        if len(parts) != 3 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]", parts[0]):
            raise ValueError(f"invalid optional field: {text!r}")
        tag, kind, value = parts
        if tag in result:
            raise ValueError(f"duplicate optional tag: {tag}")
        if kind not in "AifZJHB" or not value:
            raise ValueError(f"invalid optional field: {text!r}")
        result[tag] = (kind, value)
    return result


def _integer(text: str) -> int:
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"expected nonnegative integer: {text!r}")
    return int(text)


def _exact_overlap(cigar: str) -> int:
    if cigar == "*":
        raise ValueError("unknown overlap cannot define path coordinates or sequence")
    operations = _OPS.findall(cigar)
    if any(op not in ("M", "=") for _, op in operations):
        raise ValueError(f"non-exact overlap {cigar!r} requires an explicit consensus policy")
    return sum(int(n) for n, _ in operations)


def _cigar_operations(cigar: str) -> tuple[tuple[int, str], ...]:
    """Canonical overlap geometry; = and M share geometry, X stays explicit."""
    result: list[tuple[int, str]] = []
    for number, operation in _OPS.findall(cigar):
        operation = "M" if operation == "=" else operation
        count = int(number)
        if not count:
            continue
        if result and result[-1][1] == operation:
            count += result.pop()[0]
        result.append((count, operation))
    return tuple(result)


def _validate_overlap_length(graph: GFAGraph, source: _STEP, target: _STEP, cigar: str) -> None:
    if cigar == "*":
        return
    operations = _OPS.findall(cigar)
    ref_length = sum(int(n) for n, op in operations if op in "MDN=X")
    query_length = sum(int(n) for n, op in operations if op in "MIS=X")
    left, right = graph.segments[source[0]].length, graph.segments[target[0]].length
    if (left is not None and ref_length > left) or (right is not None and query_length > right):
        raise ValueError(f"overlap exceeds segment length: {source} -> {target}")


def _read_lines(lines: list[str]) -> GFAGraph:
    graph = GFAGraph()
    for line_number, raw in enumerate(lines, 1):
        raw = raw.rstrip("\r\n")
        if not raw or raw.startswith("#"):
            continue
        if not raw.isascii():
            raise ValueError(f"GFA line {line_number}: non-ASCII content")
        fields = raw.split("\t")
        kind = fields[0]
        try:
            if kind == "H":
                tags = _tags(fields[1:])
                if "VN" in tags and not tags["VN"][1].startswith("1."):
                    raise ValueError("only GFA1 is supported")
                graph.headers.append(raw)
            elif kind == "S":
                if len(fields) < 3:
                    raise ValueError("segment needs name and sequence")
                name = _name(fields[1])
                if name in graph.segments:
                    raise ValueError(f"duplicate segment: {name}")
                seq = None if fields[2] == "*" else fields[2]
                if seq is not None and not re.fullmatch(r"[A-Za-z=.]+", seq):
                    raise ValueError(f"invalid sequence for segment {name}")
                tags = _tags(fields[3:])
                length = len(seq) if seq is not None else None
                if "LN" in tags:
                    if tags["LN"][0] != "i":
                        raise ValueError("LN must have type i")
                    declared = _integer(tags["LN"][1])
                    if length is not None and length != declared:
                        raise ValueError(f"LN length disagrees with sequence: {name}")
                    length = declared
                graph.segments[name] = Segment(name, seq, length, raw)
            elif kind == "L":
                if len(fields) < 6:
                    raise ValueError("link needs two oriented endpoints and overlap")
                _tags(fields[6:])
                graph.links.append(
                    Link(
                        _step(fields[1], fields[2]),
                        _step(fields[3], fields[4]),
                        _cigar(fields[5]),
                        raw,
                    )
                )
            elif kind == "P":
                if len(fields) < 4:
                    raise ValueError("path needs name, segments and overlaps")
                name = _name(fields[1])
                if ";" in fields[2]:
                    raise ValueError("GFA1.2 jump paths are not supported")
                steps = tuple(_step(x[:-1], x[-1:]) for x in fields[2].split(","))
                overlaps = (
                    None if fields[3] == "*" else tuple(_cigar(x) for x in fields[3].split(","))
                )
                if overlaps is not None and len(overlaps) != len(steps) - 1:
                    raise ValueError("path overlap count must equal step count minus one")
                _tags(fields[4:])
                pan = name.split("#")
                sample, hap, molecule = (
                    (pan[0], pan[1], pan[2])
                    if len(pan) == 3 and pan[1].isdigit()
                    else (name, None, name)
                )
                record = GraphPath(name, steps, overlaps, sample, molecule, kind, raw, hap)
                if name in graph.paths:
                    raise ValueError(f"duplicate path: {name}")
                graph.paths[name] = record
            elif kind == "W":
                if len(fields) < 7:
                    raise ValueError("walk needs sample, haplotype, sequence, bounds and steps")
                sample, molecule = _name(fields[1]), _name(fields[3])
                _integer(fields[2])
                start = None if fields[4] == "*" else _integer(fields[4])
                end = None if fields[5] == "*" else _integer(fields[5])
                if start is not None and end is not None and end < start:
                    raise ValueError("walk end precedes start")
                tokens = re.findall(r"([><])([^><]+)", fields[6])
                if "".join(a + b for a, b in tokens) != fields[6] or not tokens:
                    raise ValueError("invalid walk traversal")
                steps = tuple(_step(b, "+" if a == ">" else "-") for a, b in tokens)
                name = f"{sample}#{fields[2]}#{molecule}:{fields[4]}-{fields[5]}"
                _tags(fields[7:])
                if name in graph.paths:
                    raise ValueError(f"duplicate walk/path: {name}")
                graph.paths[name] = GraphPath(
                    name,
                    steps,
                    tuple("0M" for _ in steps[1:]),
                    sample,
                    molecule,
                    kind,
                    raw,
                    fields[2],
                    start,
                    end,
                )
            else:
                raise ValueError(
                    f"unsupported GFA record {kind!r}; supported records are H/S/L/P/W"
                )
        except ValueError as exc:
            raise ValueError(f"GFA line {line_number}: {exc}") from exc
    if not graph.segments:
        raise ValueError("GFA has no segments")
    if set(graph.paths) & set(graph.segments):
        raise ValueError("path and segment names share a namespace and must be unique")
    transitions: dict[tuple[_STEP, _STEP], set[str]] = defaultdict(set)
    for link in graph.links:
        for step in (link.source, link.target):
            if step[0] not in graph.segments:
                raise ValueError(f"link references unknown segment: {step[0]}")
        _validate_overlap_length(graph, link.source, link.target, link.overlap)
        transitions[(link.source, link.target)].add(link.overlap)
        transitions[(_flip(link.target), _flip(link.source))].add(_reverse_cigar(link.overlap))
    graph.transitions = dict(transitions)
    walks: dict[tuple[str, str | None, str], list[tuple[int, int]]] = defaultdict(list)
    for path in graph.paths.values():
        for node, _ in path.steps:
            if node not in graph.segments:
                raise ValueError(f"path {path.name} references unknown segment: {node}")
        for index, pair in enumerate(pairwise(path.steps)):
            options = graph.transitions.get(pair)
            if not options:
                raise ValueError(f"path {path.name} has no link for transition {pair}")
            if path.kind == "W" and not any(c != "*" and not _cigar_operations(c) for c in options):
                raise ValueError(f"walk {path.name} requires a zero-overlap link")
            if path.overlaps is not None:
                cigar = path.overlaps[index]
                _validate_overlap_length(graph, *pair, cigar)
                if (
                    path.kind != "W"
                    and cigar != "*"
                    and "*" not in options
                    and not any(_cigar_operations(cigar) == _cigar_operations(c) for c in options)
                ):
                    raise ValueError(f"path {path.name} overlap disagrees with link at {pair}")
        if path.kind == "W" and path.start is not None and path.end is not None:
            lengths = [graph.segments[node].length for node, _ in path.steps]
            if all(n is not None for n in lengths) and sum(lengths) != path.end - path.start:
                raise ValueError(f"walk {path.name} length differs from its coordinate interval")
            key = (path.sample, path.haplotype, path.molecule)
            walks[key].append((path.start, path.end))
    for intervals in walks.values():
        ordered = sorted(intervals)
        if any(a[1] > b[0] for a, b in pairwise(ordered)):
            raise ValueError("walk intervals for the same sample/haplotype/molecule overlap")
    return graph


def load_gfa(path: str | Path) -> GFAGraph:
    """Read validated S/L/P/W graph; reject unsupported records explicitly."""
    return _read_lines(Path(path).read_text(encoding="utf-8-sig").splitlines())
