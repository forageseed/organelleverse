"""Deterministic FASTA/GFA/BED normalization and graph-consistency validation.

Normalization preserves scientific content (record order, IDs, descriptions,
sequence; GFA records/fields/tags/order) while producing deterministic UTF-8 LF
output with one terminal newline. ``validate_fasta_against_gfa`` proves that every
FASTA record is exactly spellable from segment boundaries through directed links,
trimming only the declared Oatk-compatible ``nM`` overlap.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import gfapy  # pyright: ignore[reportMissingTypeStubs]
from Bio import SeqIO  # pyright: ignore[reportUnknownVariableType]
from Bio.Seq import Seq  # pyright: ignore[reportUnknownVariableType]
from pydantic import ConfigDict, Field

from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.operations.spec import StrictSpecModel

__all__ = [
    "FastaStats",
    "GfaStats",
    "normalize_fasta",
    "normalize_gfa",
    "parse_gene_markers",
    "validate_fasta_against_gfa",
]

_VALID_IUPAC = frozenset("ACGTRYSWKMBDHVN")
_WRAP_COLUMNS = 80
_COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVNacgtryswkmbdhvn", "TGCAYRSWMKVHDBNtgcayrswmkvhdbn")


class FastaStats(StrictSpecModel):
    """Frozen parsed FASTA identity used for graph-consistency validation."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", revalidate_instances="always", arbitrary_types_allowed=True
    )

    record_ids: tuple[str, ...]
    descriptions: tuple[str, ...]
    sequences: tuple[str, ...]
    record_count: int = Field(ge=0)
    total_bases: int = Field(ge=0)


class GfaStats(StrictSpecModel):
    """Frozen parsed GFA identity used for graph-consistency validation."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", revalidate_instances="always", arbitrary_types_allowed=True
    )

    segment_names: tuple[str, ...]
    segment_sequences: tuple[str, ...]
    links: tuple[tuple[str, str, str, str, int], ...]
    record_count: int = Field(ge=0)
    segment_count: int = Field(ge=0)


def _parse_failed(message: str) -> OrganelleExecutionError:
    return OrganelleExecutionError(code="assembly.parse_failed", message=message)


def _validation_failed(message: str, **details: object) -> OrganelleExecutionError:
    return OrganelleExecutionError(
        code="assembly.validation_failed", message=message, details=details
    )


# ---------------------------------------------------------------------------
# FASTA
# ---------------------------------------------------------------------------


def normalize_fasta(source: Path, destination: Path) -> FastaStats:
    """Parse, validate, and write a deterministic normalized FASTA file."""
    try:
        records: Any = list(SeqIO.parse(str(source), "fasta"))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as error:
        raise _parse_failed(f"FASTA could not be parsed: {error}") from error

    ids: list[str] = []
    descriptions: list[str] = []
    sequences: list[str] = []
    total = 0
    for record in records:
        seq = str(record.seq).upper()
        if not seq:
            raise _parse_failed("FASTA record has an empty sequence")
        if any(base not in _VALID_IUPAC for base in seq):
            raise _parse_failed("FASTA sequence contains non-IUPAC DNA characters")
        record_id = str(record.id)
        if not record_id:
            raise _parse_failed("FASTA record has an empty id")
        if record_id in ids:
            raise _parse_failed(f"duplicate FASTA record id {record_id!r}")
        name = str(record.name)
        description_text = str(record.description)
        description = description_text[len(name) :].strip() if description_text else ""
        ids.append(record_id)
        descriptions.append(description)
        sequences.append(seq)
        total += len(seq)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record_id, description, sequence in zip(ids, descriptions, sequences, strict=True):
            header = record_id if not description else f"{record_id} {description}"
            handle.write(f">{header}\n")
            for offset in range(0, len(sequence), _WRAP_COLUMNS):
                handle.write(sequence[offset : offset + _WRAP_COLUMNS])
                handle.write("\n")

    return FastaStats(
        record_ids=tuple(ids),
        descriptions=tuple(descriptions),
        sequences=tuple(sequences),
        record_count=len(ids),
        total_bases=total,
    )


# ---------------------------------------------------------------------------
# GFA
# ---------------------------------------------------------------------------


def _parse_gfa(source: Path) -> tuple[Any, str]:
    try:
        raw = source.read_bytes().decode("utf-8")
    except Exception as error:
        raise _parse_failed(f"GFA could not be read: {error}") from error
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    try:
        gfa: Any = gfapy.Gfa.from_file(str(source))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    except gfapy.NotUniqueError as error:  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        raise _validation_failed(f"GFA has a duplicate element: {error}") from error
    except (
        gfapy.NotFoundError,  # pyright: ignore[reportUnknownMemberType]
        gfapy.InconsistencyError,  # pyright: ignore[reportUnknownMemberType]
        ValueError,
    ) as error:  # pyright: ignore[reportUnknownVariableType]
        # Reference/contract violations (unknown segments, bad orientations) surface here.
        raise _validation_failed(f"GFA references are inconsistent: {error}") from error
    except Exception as error:
        raise _parse_failed(f"GFA could not be parsed: {error}") from error
    return gfa, normalized  # pyright: ignore[reportUnknownVariableType]


def normalize_gfa(source: Path, destination: Path) -> GfaStats:
    """Parse, validate, and write a deterministic normalized GFA file."""
    gfa, normalized = _parse_gfa(source)

    segments: Any = list(gfa.segments)
    if not segments:
        raise _validation_failed("GFA contains no segments")
    segment_names: list[str] = []
    segment_sequences: list[str] = []
    name_set: set[str] = set()
    for segment in segments:
        name = str(segment.name)
        sequence = segment.sequence
        if sequence is None or sequence == "*" or not str(sequence):
            raise _validation_failed("GFA segment has no sequence", segment=name)
        if name in name_set:
            raise _validation_failed("GFA segment name is not unique", segment=name)
        name_set.add(name)
        segment_names.append(name)
        segment_sequences.append(str(sequence).upper())

    # Validate links reference known segments and capture them for the validator.
    link_lines: list[tuple[str, str, str, str, int]] = []
    for link in gfa.dovetails:
        from_name = str(link.from_segment.name)
        to_name = str(link.to_segment.name)
        if from_name not in name_set:
            raise _validation_failed("GFA link references an unknown segment", segment=from_name)
        if to_name not in name_set:
            raise _validation_failed("GFA link references an unknown segment", segment=to_name)
        overlap_len = _overlap_length(link.overlap)
        link_lines.append(
            (from_name, str(link.from_orient), to_name, str(link.to_orient), overlap_len)
        )

    # Validate paths reference known segments.
    for path in gfa.paths:
        for oriented in path.captured_segments:
            seg_name = str(oriented.line.name)
            if seg_name not in name_set:
                raise _validation_failed("GFA path references an unknown segment", segment=seg_name)

    destination.parent.mkdir(parents=True, exist_ok=True)
    out_lines = [line for line in normalized.split("\n") if line and not line.startswith("#")]
    with destination.open("w", encoding="utf-8") as handle:
        for line in out_lines:
            handle.write(line)
            handle.write("\n")

    return GfaStats(
        segment_names=tuple(segment_names),
        segment_sequences=tuple(segment_sequences),
        links=tuple(link_lines),
        record_count=len(out_lines),
        segment_count=len(segment_names),
    )


# ---------------------------------------------------------------------------
# Graph-consistency validation: spell FASTA records through the GFA graph.
# ---------------------------------------------------------------------------


def _reverse_complement(sequence: str) -> str:
    return str(Seq(sequence).reverse_complement())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def _overlap_length(overlap: object) -> int:
    """Extract the integer overlap length from a gfapy CIGAR or ``nM`` string."""
    text = str(overlap)
    match = re.fullmatch(r"(\d+)M", text)
    if match is None:
        return 0
    return int(match.group(1))


def validate_fasta_against_gfa(fasta_stats: FastaStats, gfa_stats: GfaStats) -> None:
    """Prove every FASTA record is exactly spellable through the GFA graph.

    The validator reconstructs the directed, oriented segment graph from the
    normalized segment sequences and the link edges recorded on ``gfa_stats``. For
    each FASTA record it searches for a walk (in either orientation) that spells the
    record sequence exactly, trimming only the declared ``nM`` overlap after
    verifying the overlapping bases match. The search is memoized and bounded by the
    record length so cycles terminate.
    """
    if not fasta_stats.record_count:
        raise _validation_failed("FASTA has no records to validate")

    if not gfa_stats.segment_names:
        raise _validation_failed("GFA has no sequence-bearing segments")

    oriented = _oriented_sequences(gfa_stats)
    edges = _oriented_edges(gfa_stats)

    for record_id, sequence in zip(fasta_stats.record_ids, fasta_stats.sequences, strict=True):
        if not _spellable(sequence, oriented, edges):
            raise _validation_failed(
                "FASTA record is not represented by the target graph",
                record=record_id,
            )


def _oriented_sequences(gfa_stats: GfaStats) -> dict[tuple[str, str], str]:
    """Map ``(segment_name, orientation)`` to its forward/reverse-complement sequence."""
    oriented: dict[tuple[str, str], str] = {}
    for name, seq in zip(gfa_stats.segment_names, gfa_stats.segment_sequences, strict=True):
        oriented[(name, "+")] = seq
        oriented[(name, "-")] = _reverse_complement(seq)
    return oriented


def _oriented_edges(
    gfa_stats: GfaStats,
) -> dict[tuple[str, str], list[tuple[str, str, int]]]:
    """Directed edges keyed by ``(from_segment, from_orient)``.

    Each declared link ``s1 + s2 + nM`` also implies the reverse-complement walk
    ``s2- s1- nM`` so a record that is the reverse complement of a forward contig is
    spellable too.
    """
    edges: dict[tuple[str, str], list[tuple[str, str, int]]] = {}
    for from_name, from_orient, to_name, to_orient, overlap in gfa_stats.links:
        edges.setdefault((from_name, from_orient), []).append((to_name, to_orient, overlap))
        # Reverse-complement mirror.
        edges.setdefault((to_name, _flip(to_orient)), []).append(
            (from_name, _flip(from_orient), overlap)
        )
    return edges


def _flip(orient: str) -> str:
    return "-" if orient == "+" else "+"


def _spellable(
    sequence: str,
    oriented: dict[tuple[str, str], str],
    edges: dict[tuple[str, str], list[tuple[str, str, int]]],
) -> bool:
    """Search whether ``sequence`` (or its reverse complement) is spellable."""
    target = sequence.upper()
    return _search_walk(target, oriented, edges)


def _search_walk(
    target: str,
    oriented: dict[tuple[str, str], str],
    edges: dict[tuple[str, str], list[tuple[str, str, int]]],
) -> bool:
    """Depth-first search bounded by the target length, memoized per (oriented node, pos).

    A circular contig's reported FASTA may be a rotation of the full loop, cut
    open partway into whichever path segment closes the loop: a segment used
    more than once in the path (a genuine repeat) cannot always be linearized
    starting at its own offset 0 while also closing cleanly at the end, so
    every offset implied by a declared incoming link to that segment is also
    tried as a starting point.
    """
    incoming_overlaps = _incoming_overlaps_by_node(edges)
    for node, seq in oriented.items():
        if not seq:
            continue
        for rotation in {0, *incoming_overlaps.get(node, ())}:
            if rotation < 0 or rotation >= len(seq):
                continue
            tail = seq[rotation:]
            if not target.startswith(tail[: min(len(tail), len(target))]):
                continue
            if _dfs(target, len(tail), node, node, oriented, edges, set()):
                return True
    return False


def _incoming_overlaps_by_node(
    edges: dict[tuple[str, str], list[tuple[str, str, int]]],
) -> dict[tuple[str, str], set[int]]:
    """Every overlap value declared on an edge arriving at each oriented node."""
    incoming: dict[tuple[str, str], set[int]] = {}
    for targets in edges.values():
        for to_name, to_orient, overlap in targets:
            incoming.setdefault((to_name, to_orient), set()).add(overlap)
    return incoming


def _dfs(
    target: str,
    consumed: int,
    current: tuple[str, str],
    start: tuple[str, str],
    oriented: dict[tuple[str, str], str],
    edges: dict[tuple[str, str], list[tuple[str, str, int]]],
    visited: set[tuple[str, str, int]],
) -> bool:
    if consumed == len(target):
        return True
    if consumed > len(target):
        return False
    state = (current[0], current[1], consumed)
    if state in visited:
        return False
    visited.add(state)

    for to_name, to_orient, overlap in edges.get(current, ()):
        to_seq = oriented.get((to_name, to_orient), "")
        if overlap < 0 or overlap >= len(to_seq):
            continue
        if consumed < overlap:
            continue
        expected_overlap = target[consumed - overlap : consumed]
        actual_overlap = to_seq[:overlap]
        if expected_overlap != actual_overlap:
            continue
        extension = to_seq[overlap:]
        new_consumed = consumed + len(extension)
        to_node = (to_name, to_orient)

        if new_consumed <= len(target):
            if target[consumed:new_consumed] != extension:
                continue
            if _dfs(target, new_consumed, to_node, start, oriented, edges, visited):
                return True
            continue

        # A circular contig's reported sequence is one full lap, cut open at
        # the walk's own starting segment: the seam where this segment's
        # tail overlaps the start segment's head is trimmed from the record
        # rather than repeated. Accept only if a declared link closes the
        # loop with an overlap that exactly accounts for the overshoot and
        # the trimmed bases genuinely match the start segment's head.
        overshoot = new_consumed - len(target)
        if _closes_circularly(
            target, consumed, extension, overshoot, to_node, start, edges, oriented
        ):
            return True
    visited.discard(state)
    return False


def _closes_circularly(
    target: str,
    consumed: int,
    extension: str,
    overshoot: int,
    to_node: tuple[str, str],
    start: tuple[str, str],
    edges: dict[tuple[str, str], list[tuple[str, str, int]]],
    oriented: dict[tuple[str, str], str],
) -> bool:
    trimmed = extension[: len(extension) - overshoot]
    if target[consumed : consumed + len(trimmed)] != trimmed:
        return False
    start_seq = oriented.get(start, "")
    closing_tail = extension[len(extension) - overshoot :]
    for closing_name, closing_orient, closing_overlap in edges.get(to_node, ()):
        if (closing_name, closing_orient) != start or closing_overlap != overshoot:
            continue
        if closing_tail == start_seq[:overshoot]:
            return True
    return False


# ---------------------------------------------------------------------------
# BED parsing / gene markers
# ---------------------------------------------------------------------------


def parse_gene_markers(source: Path) -> tuple[str, ...]:
    """Parse a BED annotation file and return sorted, lower-case gene tokens."""
    tokens: set[str] = set()
    try:
        text = source.read_text(encoding="utf-8")
    except Exception as error:
        raise _parse_failed(f"BED could not be read: {error}") from error
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) < 4:
            raise _parse_failed(f"BED line {line_number} has fewer than four columns")
        start_raw, end_raw, _name, gene = columns[1], columns[2], columns[2], columns[3]
        try:
            start = int(start_raw)
            end = int(end_raw)
        except ValueError as error:
            raise _parse_failed(f"BED line {line_number} has non-integer coordinates") from error
        if start < 0 or end <= start:
            raise _parse_failed(f"BED line {line_number} has invalid coordinate range")
        token = gene.strip().lower()
        if re.fullmatch(r"[a-z0-9_]+", token):
            tokens.add(token)
    return tuple(sorted(tokens))
