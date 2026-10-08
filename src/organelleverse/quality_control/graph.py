"""Deterministic graph, junction, and explicit-repeat QC evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib import util as importlib_util
from itertools import pairwise
from pathlib import Path
from typing import Any

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleDependencyError, OrganelleInputError
from organelleverse.quality_control.fasta import read_fasta

from .contracts import (
    ErrorCandidate,
    GraphEvidence,
    JunctionSupport,
    MappingEvidence,
    QcCheck,
    RepeatSupport,
    ResolvedAssemblyEvidence,
)
from .policy import QcPolicy
from .static import validate_gfa


def _require_pysam() -> None:
    """Fail closed if the optional pysam BAM backend is unavailable."""
    if importlib_util.find_spec("pysam") is None:
        raise OrganelleDependencyError(
            code="qc.pysam_missing",
            message="pysam is required for BAM-based graph evidence; install with 'pip install organelleverse[qc]'.",
            details={"package": "pysam", "extra": "qc"},
            retryable=False,
            suggested_action={"action": "install_extra", "extra": "qc"},
        )


@dataclass(frozen=True)
class _Step:
    name: str
    orientation: str


@dataclass(frozen=True)
class _PathRecord:
    name: str
    steps: tuple[_Step, ...]


@dataclass(frozen=True)
class _ParsedGfa:
    segments: dict[str, str]
    links: frozenset[tuple[str, str, str, str]]
    paths: tuple[_PathRecord, ...]


@dataclass(frozen=True)
class _Junction:
    sequence_id: str
    left: _Step
    right: _Step
    center: int
    sequence_length: int
    left_length: int
    right_length: int
    wraparound: bool


def collect_graph_evidence(
    evidence: ResolvedAssemblyEvidence,
    mapping: MappingEvidence,
    policy: QcPolicy,
) -> GraphEvidence:
    """Interpret only declared GFA paths and captured alignment artifacts."""

    _require_pysam()

    graph_path = evidence.assembly_graph_path
    if graph_path is None:
        return GraphEvidence(
            checks=(
                QcCheck(
                    check_id="qc.graph_topology",
                    category="structure",
                    status="not_assessed",
                    value=None,
                    message="The assembly result did not declare a GFA artifact.",
                ),
            )
        )

    graph_path = graph_path.expanduser().resolve()
    summary = validate_gfa(graph_path)
    parsed = _parse_gfa(graph_path)
    fasta_records = dict(read_fasta(evidence.primary_fasta_path))
    junctions = _path_junctions(parsed, fasta_records)
    libraries = {item.role: item for item in evidence.input_libraries}
    alignment_by_role = {item.library_role: item for item in mapping.library_alignments}

    junction_records: list[JunctionSupport] = []
    structural_candidates: list[ErrorCandidate] = []
    for junction_index, junction in enumerate(junctions, start=1):
        for library_role, library in libraries.items():
            alignment = alignment_by_role.get(library_role)
            anchor = min(
                policy.maximum_anchor_bases,
                policy.minimum_anchor_bases,
                junction.left_length,
                junction.right_length,
            )
            if (
                alignment is None
                or anchor < policy.minimum_anchor_bases
                or not _artifact_is_current(alignment.bam_artifact)
                or not _artifact_is_current(alignment.index_artifact)
            ):
                support = 0
                contradiction = 0
                status = "not_assessed"
            else:
                support, contradiction, suitable = _junction_read_counts(
                    Path(alignment.bam_artifact.uri),
                    junction,
                    anchor,
                    library.technology,
                    library.insert_size,
                    policy,
                )
                if contradiction >= policy.minimum_contradiction_support:
                    status = "contradicted"
                elif support >= policy.minimum_junction_support:
                    status = "supported"
                elif suitable:
                    status = "unsupported"
                else:
                    status = "not_assessed"

            record = JunctionSupport(
                sequence_id=junction.sequence_id,
                left_segment=f"{junction.left.name}{junction.left.orientation}",
                right_segment=f"{junction.right.name}{junction.right.orientation}",
                library_role=library_role,
                supporting_reads=support,
                contradicting_reads=contradiction,
                status=status,
                anchor_length=anchor,
            )
            junction_records.append(record)
            if status in {"unsupported", "contradicted"}:
                start = (
                    max(0, junction.sequence_length - 1)
                    if junction.wraparound
                    else max(0, junction.center - 1)
                )
                structural_candidates.append(
                    ErrorCandidate(
                        candidate_id=f"junction:{junction_index}:{library_role}",
                        sequence_id=junction.sequence_id,
                        start=start,
                        end=start + 1,
                        error_type=f"{status}_junction",
                        supporting_observations=(
                            contradiction if status == "contradicted" else support
                        ),
                        status="candidate",
                    )
                )

    repeats = _repeat_support(parsed, junction_records, mapping)
    checks = _graph_checks(summary, junction_records, repeats)
    return GraphEvidence(
        gfa_summary=summary,
        junction_support=tuple(junction_records),
        repeat_support=repeats,
        structural_error_candidates=tuple(structural_candidates),
        checks=checks,
    )


def _parse_gfa(path: Path) -> _ParsedGfa:
    segments: dict[str, str] = {}
    links: set[tuple[str, str, str, str]] = set()
    paths: list[_PathRecord] = []
    for raw_line in path.read_text().splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        fields = raw_line.split("\t")
        if fields[0] == "S" and len(fields) >= 3:
            segments[fields[1]] = fields[2].upper()
        elif fields[0] == "L" and len(fields) >= 5:
            links.add((fields[1], fields[2], fields[3], fields[4]))
        elif fields[0] == "P" and len(fields) >= 3:
            steps = tuple(_parse_step(item) for item in fields[2].split(",") if item)
            paths.append(_PathRecord(name=fields[1], steps=steps))
    return _ParsedGfa(segments=segments, links=frozenset(links), paths=tuple(paths))


def _parse_step(value: str) -> _Step:
    if len(value) < 2 or value[-1] not in "+-":
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="GFA path step is missing an orientation",
            details={"step": value},
        )
    return _Step(name=value[:-1], orientation=value[-1])


def _path_junctions(
    graph: _ParsedGfa,
    fasta_records: dict[str, str],
) -> tuple[_Junction, ...]:
    junctions: list[_Junction] = []
    for path in graph.paths:
        if len(path.steps) < 2:
            continue
        sequence_id = path.name
        if sequence_id not in fasta_records:
            if len(fasta_records) == 1 and len(graph.paths) == 1:
                sequence_id = next(iter(fasta_records))
            else:
                continue
        lengths = [_segment_length(graph, step) for step in path.steps]
        spelled = "".join(_oriented_segment(graph, step) for step in path.steps)
        reported = fasta_records[sequence_id].upper()
        if spelled and "*" not in spelled and spelled != reported:
            raise OrganelleInputError(
                code="qc.input_contract_violation",
                message="reported FASTA sequence is not spellable from its declared GFA path",
                details={"path_name": path.name, "sequence_id": sequence_id},
            )
        center = 0
        for index, (left, right) in enumerate(pairwise(path.steps)):
            center += lengths[index]
            if not _has_link(graph, left, right):
                raise OrganelleInputError(
                    code="qc.input_contract_violation",
                    message="GFA path declares an adjacency without a matching link",
                    details={"path_name": path.name, "left": left.name, "right": right.name},
                )
            junctions.append(
                _Junction(
                    sequence_id=sequence_id,
                    left=left,
                    right=right,
                    center=center,
                    sequence_length=len(reported),
                    left_length=lengths[index],
                    right_length=lengths[index + 1],
                    wraparound=False,
                )
            )
        last = path.steps[-1]
        first = path.steps[0]
        if _has_link(graph, last, first):
            junctions.append(
                _Junction(
                    sequence_id=sequence_id,
                    left=last,
                    right=first,
                    center=len(reported),
                    sequence_length=len(reported),
                    left_length=lengths[-1],
                    right_length=lengths[0],
                    wraparound=True,
                )
            )
    return tuple(junctions)


def _segment_length(graph: _ParsedGfa, step: _Step) -> int:
    sequence = graph.segments.get(step.name, "")
    return 0 if sequence == "*" else len(sequence)


def _oriented_segment(graph: _ParsedGfa, step: _Step) -> str:
    sequence = graph.segments.get(step.name, "")
    if step.orientation == "+" or sequence == "*":
        return sequence
    return _reverse_complement(sequence)


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def _has_link(graph: _ParsedGfa, left: _Step, right: _Step) -> bool:
    direct = (left.name, left.orientation, right.name, right.orientation)
    reverse = (
        right.name,
        _opposite_orientation(right.orientation),
        left.name,
        _opposite_orientation(left.orientation),
    )
    return direct in graph.links or reverse in graph.links


def _opposite_orientation(orientation: str) -> str:
    return "-" if orientation == "+" else "+"


def _artifact_is_current(artifact: ArtifactRef) -> bool:
    path = Path(artifact.uri)
    if not path.is_file():
        raise OrganelleInputError(
            code="qc.artifact_missing",
            message=f"mapping artifact is missing: {path}",
            details={"artifact_kind": artifact.kind, "uri": artifact.uri},
        )
    actual = ArtifactRef.from_path(
        path,
        kind=artifact.kind,
        format=artifact.format,
        media_type=artifact.media_type,
    )
    if actual.sha256 != artifact.sha256 or actual.size_bytes != artifact.size_bytes:
        raise OrganelleInputError(
            code="qc.artifact_digest_mismatch",
            message=f"mapping artifact digest mismatch: {path}",
            details={"artifact_kind": artifact.kind, "uri": artifact.uri},
        )
    return True


def _junction_read_counts(
    bam_path: Path,
    junction: _Junction,
    anchor: int,
    technology: str,
    insert_size: int | None,
    policy: QcPolicy,
) -> tuple[int, int, bool]:
    import pysam

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        if junction.sequence_id not in bam.references:
            return (0, 0, False)
        alignments = [
            alignment
            for alignment in bam.fetch(junction.sequence_id)
            if not alignment.is_unmapped
            and not alignment.is_secondary
            and alignment.mapping_quality >= policy.minimum_mapping_quality
        ]

    if technology == "illumina":
        return _short_read_support(alignments, junction, anchor, insert_size)
    return _long_read_support(alignments, junction, anchor)


def _long_read_support(
    alignments: list[Any],
    junction: _Junction,
    anchor: int,
) -> tuple[int, int, bool]:
    if junction.wraparound:
        starts: set[str] = set()
        ends: set[str] = set()
        suitable = False
        for alignment in alignments:
            query_length = int(alignment.infer_query_length(always=True) or 0)
            suitable = suitable or query_length >= 2 * anchor
            if int(alignment.reference_start) <= 0 and int(alignment.reference_end) >= anchor:
                starts.add(str(alignment.query_name))
            if (
                int(alignment.reference_start) <= junction.sequence_length - anchor
                and int(alignment.reference_end) >= junction.sequence_length
            ):
                ends.add(str(alignment.query_name))
        return (len(starts & ends), 0, suitable)

    supporting: set[str] = set()
    contradicting: set[str] = set()
    suitable = False
    for alignment in alignments:
        if alignment.is_supplementary:
            continue
        query_length = int(alignment.infer_query_length(always=True) or 0)
        suitable = suitable or query_length >= 2 * anchor
        start = int(alignment.reference_start)
        end = int(alignment.reference_end)
        if start <= junction.center - anchor and end >= junction.center + anchor:
            supporting.add(str(alignment.query_name))
            continue
        left_soft, right_soft = _soft_clips(alignment.cigartuples or ())
        if (
            end == junction.center and start <= junction.center - anchor and right_soft >= anchor
        ) or (start == junction.center and end >= junction.center + anchor and left_soft >= anchor):
            contradicting.add(str(alignment.query_name))
    return (len(supporting), len(contradicting), suitable)


def _short_read_support(
    alignments: list[Any],
    junction: _Junction,
    anchor: int,
    insert_size: int | None,
) -> tuple[int, int, bool]:
    if junction.wraparound:
        return (0, 0, False)
    by_name: dict[str, list[Any]] = defaultdict(list)
    for alignment in alignments:
        if not alignment.is_supplementary:
            by_name[str(alignment.query_name)].append(alignment)
    supporting: set[str] = set()
    suitable = bool(insert_size is not None and insert_size >= 2 * anchor)
    for query_name, records in by_name.items():
        start = min(int(item.reference_start) for item in records)
        end = max(int(item.reference_end) for item in records)
        compatible_insert = insert_size is None or end - start <= insert_size
        if (
            compatible_insert
            and start <= junction.center - anchor
            and end >= junction.center + anchor
        ):
            supporting.add(query_name)
    return (len(supporting), 0, suitable)


def _soft_clips(cigar: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    left = cigar[0][1] if cigar and cigar[0][0] == 4 else 0
    right = cigar[-1][1] if cigar and cigar[-1][0] == 4 else 0
    return (left, right)


def _repeat_support(
    graph: _ParsedGfa,
    junctions: list[JunctionSupport],
    mapping: MappingEvidence,
) -> tuple[RepeatSupport, ...]:
    del mapping  # repeat-local depth is not computed; spanning reads come from junctions
    repeats: list[RepeatSupport] = []
    for path in graph.paths:
        counts = Counter(step.name for step in path.steps)
        for repeat_id, copies in sorted(counts.items()):
            if copies < 2:
                continue
            related = [
                item
                for item in junctions
                if item.sequence_id == path.name
                and (item.left_segment[:-1] == repeat_id or item.right_segment[:-1] == repeat_id)
            ]
            if related and all(item.status == "supported" for item in related):
                status = "resolved"
            elif any(item.status == "contradicted" for item in related):
                status = "contradicted"
            elif any(item.status in {"supported", "unsupported"} for item in related):
                status = "ambiguous"
            else:
                status = "not_assessed"
            repeats.append(
                RepeatSupport(
                    repeat_id=repeat_id,
                    sequence_id=path.name,
                    length=len(graph.segments.get(repeat_id, "")),
                    orientation=",".join(
                        step.orientation for step in path.steps if step.name == repeat_id
                    ),
                    path_occurrences=copies,
                    spanning_reads=sum(item.supporting_reads for item in related),
                    status=status,
                )
            )
    return tuple(repeats)


def _graph_checks(
    summary: Any,
    junctions: list[JunctionSupport],
    repeats: tuple[RepeatSupport, ...],
) -> tuple[QcCheck, ...]:
    if any(item.status == "contradicted" for item in junctions):
        junction_status = "fail"
        junction_message = "At least one declared GFA adjacency has contradictory read evidence."
    elif any(item.status == "unsupported" for item in junctions):
        junction_status = "warn"
        junction_message = "At least one assessable GFA adjacency lacks required read support."
    elif junctions and all(item.status == "supported" for item in junctions):
        junction_status = "pass"
        junction_message = "All assessed GFA path adjacencies have read support."
    else:
        junction_status = "not_assessed"
        junction_message = "No GFA path adjacency could be assessed."
    repeat_status = (
        "warn"
        if any(item.status in {"ambiguous", "contradicted"} for item in repeats)
        else "pass"
        if repeats
        else "not_assessed"
    )
    return (
        QcCheck(
            check_id="qc.graph_topology",
            category="structure",
            status="pass",
            value=int(summary.component_count),
            unit="components",
            message="GFA topology was parsed without referential-integrity errors.",
        ),
        QcCheck(
            check_id="qc.junction_support",
            category="structure",
            status=junction_status,
            value=sum(item.status == "supported" for item in junctions),
            unit="supported_adjacencies",
            message=junction_message,
            finding_code=(
                "qc.contradicted_junction"
                if junction_status == "fail"
                else "qc.unsupported_junction"
                if junction_status == "warn"
                else ""
            ),
        ),
        QcCheck(
            check_id="qc.repeat_resolution",
            category="structure",
            status=repeat_status,
            value=len(repeats),
            unit="explicit_repeats",
            message=(
                "Explicit repeated path segments were assessed."
                if repeats
                else "No repeated segment occurs in a declared GFA path."
            ),
        ),
    )


__all__ = ["collect_graph_evidence"]
