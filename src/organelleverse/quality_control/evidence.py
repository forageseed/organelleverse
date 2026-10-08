"""Shell-free read-mapping and base-level assembly-QC evidence."""

from __future__ import annotations

import math
import shutil
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, cast

from organelleverse.assembly.execution import CommandOutcome, CommandRunner
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleDependencyError, OrganelleExecutionError
from organelleverse.quality_control.fasta import read_fasta

from .contracts import (
    CoverageWindow,
    ErrorCandidate,
    InputLibraryEvidence,
    LibraryAlignmentEvidence,
    LibraryMappingSummary,
    MappingEvidence,
    QcCheck,
    ResolvedAssemblyEvidence,
    SequenceDepthProfile,
)
from .environment import QcEnvironment, mapper_preset
from .policy import QC_POLICY_V5


def _require_pysam() -> None:
    """Fail closed if the optional pysam BAM backend is unavailable."""
    if importlib_util.find_spec("pysam") is None:
        raise OrganelleDependencyError(
            code="qc.pysam_missing",
            message="pysam is required for BAM-based mapping evidence; install with 'pip install organelleverse[qc]'.",
            details={"package": "pysam", "extra": "qc"},
            retryable=False,
            suggested_action={"action": "install_extra", "extra": "qc"},
        )


@dataclass(frozen=True)
class _ParsedLibrary:
    summary: LibraryMappingSummary
    any_depths: dict[str, list[int]]
    confident_depths: dict[str, list[int]]
    errors: dict[tuple[str, str], dict[int, int]]
    bam_artifact_id: str


def collect_mapping_evidence(
    evidence: ResolvedAssemblyEvidence,
    environment: QcEnvironment,
    *,
    workspace: str | Path,
    threads: int,
    timeout_seconds: float,
    runner: CommandRunner | None = None,
) -> MappingEvidence:
    """Map every declared library independently and parse deterministic evidence."""

    _require_pysam()

    if threads < 1:
        raise ValueError("threads must be at least 1")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0")

    target = Path(workspace)
    target.mkdir(parents=True, exist_ok=True)
    primary_fasta_path = evidence.primary_fasta_path.expanduser().resolve()
    command_runner = runner if runner is not None else CommandRunner()
    parsed_libraries: list[_ParsedLibrary] = []
    library_alignments: list[LibraryAlignmentEvidence] = []
    artifacts: list[ArtifactRef] = []
    bam_paths: list[Path] = []

    for index, library in enumerate(evidence.input_libraries, start=1):
        prefix = f"library_{index:03d}"
        library_dir = target / prefix
        library_dir.mkdir(parents=True, exist_ok=False)
        sam_path = library_dir / "alignment.sam"
        bam_path = library_dir / "alignment.sorted.bam"
        bai_path = library_dir / "alignment.sorted.bam.bai"

        reads = _library_read_paths(library)
        preset = mapper_preset(library.technology, library.quality_state)
        minimap_argv = (
            str(environment.minimap2.path),
            "-a",
            "-x",
            preset,
            "-t",
            str(threads),
            str(primary_fasta_path),
            *(str(path) for path in reads),
        )
        minimap_stderr = library_dir / "minimap2.stderr.log"
        minimap_outcome = command_runner.run(
            minimap_argv,
            stage=f"{prefix}.minimap2",
            cwd=library_dir,
            timeout_seconds=timeout_seconds,
            stdout_path=sam_path,
            stderr_path=minimap_stderr,
        )
        _require_success(minimap_outcome)

        sort_stdout = library_dir / "samtools-sort.stdout.log"
        sort_stderr = library_dir / "samtools-sort.stderr.log"
        sort_argv = (
            str(environment.samtools.path),
            "sort",
            "-@",
            str(threads),
            "-o",
            str(bam_path),
            "-",
        )
        sort_outcome = command_runner.run(
            sort_argv,
            stage=f"{prefix}.samtools_sort",
            cwd=library_dir,
            timeout_seconds=timeout_seconds,
            stdout_path=sort_stdout,
            stderr_path=sort_stderr,
            stdin_path=sam_path,
        )
        _require_success(sort_outcome)

        index_stdout = library_dir / "samtools-index.stdout.log"
        index_stderr = library_dir / "samtools-index.stderr.log"
        index_argv = (
            str(environment.samtools.path),
            "index",
            "-@",
            str(threads),
            str(bam_path),
            str(bai_path),
        )
        index_outcome = command_runner.run(
            index_argv,
            stage=f"{prefix}.samtools_index",
            cwd=library_dir,
            timeout_seconds=timeout_seconds,
            stdout_path=index_stdout,
            stderr_path=index_stderr,
        )
        _require_success(index_outcome)

        library_artifacts = (
            _artifact(sam_path, "alignment_sam", "sam", "application/x-sam"),
            _artifact(bam_path, "alignment_bam", "bam", "application/x-bam"),
            _artifact(bai_path, "alignment_index", "bai", "application/octet-stream"),
            _artifact(minimap_stderr, "execution_log", "text", "text/plain"),
            _artifact(sort_stdout, "execution_log", "text", "text/plain"),
            _artifact(sort_stderr, "execution_log", "text", "text/plain"),
            _artifact(index_stdout, "execution_log", "text", "text/plain"),
            _artifact(index_stderr, "execution_log", "text", "text/plain"),
        )
        artifacts.extend(library_artifacts)
        bam_paths.append(bam_path)
        library_alignments.append(
            LibraryAlignmentEvidence(
                library_role=library.role,
                bam_artifact=library_artifacts[1],
                index_artifact=library_artifacts[2],
            )
        )
        parsed_libraries.append(
            _parse_library_bam(
                bam_path,
                library,
                primary_fasta_path,
                bam_artifact_id=library_artifacts[1].object_id,
            )
        )

    combined_any, combined_confident = _combine_depths(parsed_libraries)
    windows = _coverage_windows(combined_any, combined_confident)
    profiles = _sequence_depth_profiles(combined_any)
    any_intervals_path = target / "coverage.intervals.bed"
    confident_intervals_path = target / "coverage.intervals.confident.bed"
    _write_coverage_intervals(any_intervals_path, combined_any)
    _write_coverage_intervals(confident_intervals_path, combined_confident)
    artifacts.extend(
        (
            _artifact(
                any_intervals_path,
                "coverage_intervals",
                "bed",
                "text/tab-separated-values",
            ),
            _artifact(
                confident_intervals_path,
                "coverage_intervals_confident",
                "bed",
                "text/tab-separated-values",
            ),
        )
    )

    candidates = _error_candidates(parsed_libraries)
    if candidates:
        candidate_evidence_path = target / "base-error-evidence.tsv"
        _write_base_error_evidence(candidate_evidence_path, candidates)
        candidate_evidence = _artifact(
            candidate_evidence_path,
            "base_error_evidence",
            "tsv",
            "text/tab-separated-values",
        )
        artifacts.append(candidate_evidence)
        candidates = tuple(
            candidate.model_copy(update={"evidence_artifact_ids": (candidate_evidence.object_id,)})
            for candidate in candidates
        )
    mapping_status = "pass" if parsed_libraries else "not_assessed"
    mapping_message = (
        f"Mapped {len(parsed_libraries)} declared read libraries independently."
        if parsed_libraries
        else "No declared read library was available for mapping."
    )
    kmer_check, kmer_artifacts = _collect_kmer_evidence(
        primary_fasta_path,
        bam_paths,
        environment,
        workspace=target,
        threads=threads,
        timeout_seconds=timeout_seconds,
        runner=command_runner,
    )
    artifacts.extend(kmer_artifacts)
    checks = (
        QcCheck(
            check_id="qc.read_mapping",
            category="read_support",
            status=mapping_status,
            value=len(parsed_libraries),
            unit="libraries",
            message=mapping_message,
        ),
        QcCheck(
            check_id="qc.base_error_candidates",
            category="base_accuracy",
            status=("warn" if candidates else "pass" if parsed_libraries else "not_assessed"),
            value=len(candidates) if parsed_libraries else None,
            unit="candidates" if parsed_libraries else "",
            message=(
                f"{len(candidates)} read-supported base-error candidate(s) were detected."
                if candidates
                else "No read-supported base-error candidate met the evidence threshold."
                if parsed_libraries
                else "No read library was available to assess base-error candidates."
            ),
            finding_code="qc.base_error_candidate" if candidates else "",
            evidence_artifact_ids=tuple(
                dict.fromkeys(
                    artifact_id
                    for candidate in candidates
                    for artifact_id in candidate.evidence_artifact_ids
                )
            ),
        ),
        kmer_check,
    )
    return MappingEvidence(
        library_mapping_summaries=tuple(item.summary for item in parsed_libraries),
        coverage_windows=windows,
        sequence_depth_profiles=profiles,
        base_error_candidates=candidates,
        library_alignments=tuple(library_alignments),
        coordinate_artifacts=tuple(artifacts),
        tool_identities=environment.tool_identities,
        checks=checks,
    )


def _collect_kmer_evidence(
    assembly_path: Path,
    bam_paths: Sequence[Path],
    environment: QcEnvironment,
    *,
    workspace: Path,
    threads: int,
    timeout_seconds: float,
    runner: CommandRunner,
) -> tuple[QcCheck, tuple[ArtifactRef, ...]]:
    if environment.meryl is None:
        return _unassessed_kmer("meryl is unavailable in the resolved QC environment.")

    target_reads = workspace / "target_reads.fasta"
    selected_reads = _write_target_reads(target_reads, bam_paths)
    if selected_reads == 0:
        target_reads.unlink(missing_ok=True)
        return _unassessed_kmer("No primary target reads passed the mapping-quality filter.")

    meryl = str(environment.meryl.path)
    kmer_size = QC_POLICY_V5.kmer_size
    target_database = workspace / "target_reads.meryl"
    assembly_database = workspace / "assembly.meryl"
    shared_database = workspace / "shared.meryl"
    assembly_histogram = workspace / "assembly-kmers.histogram.tsv"
    shared_histogram = workspace / "shared-kmers.histogram.tsv"
    resources = (f"threads={threads}",)
    commands = (
        (
            "count_target_reads",
            (
                "count",
                f"k={kmer_size}",
                *resources,
                str(target_reads),
                "output",
                str(target_database),
            ),
            None,
        ),
        (
            "count_assembly",
            (
                "count",
                f"k={kmer_size}",
                *resources,
                str(assembly_path),
                "output",
                str(assembly_database),
            ),
            None,
        ),
        (
            "intersect",
            (
                "intersect",
                *resources,
                str(assembly_database),
                str(target_database),
                "output",
                str(shared_database),
            ),
            None,
        ),
        (
            "histogram_assembly",
            ("histogram", *resources, str(assembly_database)),
            assembly_histogram,
        ),
        (
            "histogram_shared",
            ("histogram", *resources, str(shared_database)),
            shared_histogram,
        ),
    )
    command_artifacts: list[ArtifactRef] = []
    try:
        for name, arguments, captured_output in commands:
            stage = f"kmer.{name}"
            stem = name.replace("_", "-")
            stdout_path = captured_output or workspace / f"meryl-{stem}.stdout.log"
            stderr_path = workspace / f"meryl-{stem}.stderr.log"
            outcome = runner.run(
                (meryl, *arguments),
                stage=stage,
                cwd=workspace,
                timeout_seconds=timeout_seconds,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            command_artifacts.extend(
                (
                    _artifact(stdout_path, "execution_log", "text", "text/plain"),
                    _artifact(stderr_path, "execution_log", "text", "text/plain"),
                )
            )
            if outcome.termination != "exit" or outcome.returncode != 0:
                return _unassessed_kmer(
                    f"Optional meryl stage {stage} failed "
                    f"({outcome.termination}, return code {outcome.returncode}).",
                    finding_code="qc.kmer_tool_failed",
                    artifacts=tuple(command_artifacts),
                )

        assembly_kmers = _histogram_distinct_kmers(assembly_histogram)
        supported_kmers = _histogram_distinct_kmers(shared_histogram)
        if assembly_kmers == 0:
            return _unassessed_kmer(
                f"The assembly contains no countable {kmer_size}-mers.",
                artifacts=tuple(command_artifacts),
            )
        if supported_kmers > assembly_kmers:
            raise OrganelleExecutionError(
                code="qc.kmer_failed",
                message="meryl intersection contains more k-mers than the assembly",
                details={
                    "assembly_distinct_kmers": assembly_kmers,
                    "supported_distinct_kmers": supported_kmers,
                },
            )

        support_fraction = supported_kmers / assembly_kmers
        metrics_path = workspace / "kmer-metrics.tsv"
        metrics_path.write_text(
            "metric\tvalue\n"
            f"kmer_size\t{kmer_size}\n"
            f"assembly_distinct_kmers\t{assembly_kmers}\n"
            f"target_read_supported_assembly_kmers\t{supported_kmers}\n"
            f"assembly_kmer_support_fraction\t{support_fraction:g}\n"
        )
        metrics_artifact = _artifact(
            metrics_path,
            "kmer_metrics",
            "tsv",
            "text/tab-separated-values",
        )
        return (
            QcCheck(
                check_id="qc.kmer_evidence",
                category="base_accuracy",
                status="pass",
                value=support_fraction,
                unit="fraction",
                message=(
                    f"{supported_kmers} of {assembly_kmers} distinct assembly "
                    f"{kmer_size}-mers were present in quality-filtered target reads."
                ),
                evidence_artifact_ids=(metrics_artifact.object_id,),
            ),
            (*command_artifacts, metrics_artifact),
        )
    finally:
        target_reads.unlink(missing_ok=True)
        for database in (target_database, assembly_database, shared_database):
            if database.is_dir():
                shutil.rmtree(database)
            else:
                database.unlink(missing_ok=True)


def _unassessed_kmer(
    message: str,
    *,
    finding_code: str = "",
    artifacts: tuple[ArtifactRef, ...] = (),
) -> tuple[QcCheck, tuple[ArtifactRef, ...]]:
    return (
        QcCheck(
            check_id="qc.kmer_evidence",
            category="base_accuracy",
            status="not_assessed",
            value=None,
            message=message,
            finding_code=finding_code,
            evidence_artifact_ids=tuple(artifact.object_id for artifact in artifacts),
        ),
        artifacts,
    )


def _write_target_reads(path: Path, bam_paths: Sequence[Path]) -> int:
    import pysam

    selected = 0
    with path.open("w") as handle:
        for library_index, bam_path in enumerate(bam_paths, start=1):
            with pysam.AlignmentFile(str(bam_path), "rb") as bam:
                for alignment in bam.fetch(until_eof=True):
                    if not _accepted_alignment(alignment):
                        continue
                    sequence = (alignment.query_sequence or "").upper()
                    if len(sequence) < QC_POLICY_V5.kmer_size:
                        continue
                    selected += 1
                    handle.write(f">library_{library_index:03d}_read_{selected:09d}\n{sequence}\n")
    return selected


def _histogram_distinct_kmers(path: Path) -> int:
    distinct = 0
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if len(fields) != 2:
            raise OrganelleExecutionError(
                code="qc.kmer_failed",
                message="meryl emitted a malformed histogram",
                details={"path": str(path), "line": line_number},
            )
        try:
            multiplicity, count = (int(field) for field in fields)
        except ValueError as error:
            raise OrganelleExecutionError(
                code="qc.kmer_failed",
                message="meryl emitted a non-integer histogram",
                details={"path": str(path), "line": line_number},
            ) from error
        if multiplicity < 1 or count < 0:
            raise OrganelleExecutionError(
                code="qc.kmer_failed",
                message="meryl emitted an invalid histogram count",
                details={"path": str(path), "line": line_number},
            )
        distinct += count
    return distinct


def _library_read_paths(library: InputLibraryEvidence) -> tuple[Path, ...]:
    if library.technology == "illumina":
        assert library.read1 is not None
        paths = [Path(library.read1.uri).expanduser().resolve()]
        if library.read2 is not None:
            paths.append(Path(library.read2.uri).expanduser().resolve())
        return tuple(paths)
    assert library.reads is not None
    return (Path(library.reads.uri).expanduser().resolve(),)


def _require_success(outcome: CommandOutcome) -> None:
    if outcome.termination == "launch_error":
        raise OrganelleDependencyError(
            code="qc.dependency_unavailable",
            message=f"unable to launch QC command stage {outcome.stage}",
            details={
                "stage": outcome.stage,
                "argv0": outcome.argv[0],
                "reason": outcome.launch_error,
            },
        )
    if outcome.termination != "exit" or outcome.returncode != 0:
        raise OrganelleExecutionError(
            code="qc.mapping_failed",
            message=f"read-mapping stage failed: {outcome.stage}",
            details={
                "stage": outcome.stage,
                "termination": outcome.termination,
                "returncode": outcome.returncode,
                "stdout_path": outcome.stdout_path,
                "stderr_path": outcome.stderr_path,
            },
        )


def _parse_library_bam(
    bam_path: Path,
    library: InputLibraryEvidence,
    fasta_path: Path,
    *,
    bam_artifact_id: str,
) -> _ParsedLibrary:
    import pysam

    references = dict(read_fasta(fasta_path))
    mapped_reads = 0
    total_reads = 0
    aligned_bases = 0
    mismatches = 0
    insertions = 0
    deletions = 0
    error_positions: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    any_depths = {sequence_id: [0] * len(sequence) for sequence_id, sequence in references.items()}
    confident_depths = {
        sequence_id: [0] * len(sequence) for sequence_id, sequence in references.items()
    }

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for alignment in bam.fetch(until_eof=True):
            if alignment.is_secondary or alignment.is_supplementary:
                # Repeat-aware placements feed the any-alignment track only.
                if not alignment.is_unmapped:
                    _record_alignment_evidence(
                        alignment,
                        references,
                        error_positions,
                        any_depths,
                        confident_depths,
                        confident=False,
                    )
                continue
            total_reads += 1
            if alignment.is_unmapped:
                continue
            if alignment.mapping_quality < QC_POLICY_V5.minimum_mapping_quality:
                # Low-MAPQ primaries are repeat-like: any-alignment track only.
                _record_alignment_evidence(
                    alignment,
                    references,
                    error_positions,
                    any_depths,
                    confident_depths,
                    confident=False,
                )
                continue
            mapped_reads += 1
            cigar = alignment.cigartuples or ()
            match_bases = sum(length for operation, length in cigar if operation in {0, 7, 8})
            insertion_bases = sum(length for operation, length in cigar if operation == 1)
            deletion_bases = sum(length for operation, length in cigar if operation == 2)
            nm_value = (
                cast(str | int | float, alignment.get_tag("NM"))  # pyright: ignore[reportUnknownMemberType]
                if alignment.has_tag("NM")
                else 0
            )
            nm = int(nm_value)
            mismatch_bases = max(0, nm - insertion_bases - deletion_bases)
            aligned_bases += match_bases + insertion_bases + deletion_bases
            mismatches += mismatch_bases
            insertions += insertion_bases
            deletions += deletion_bases
            # Confident primary: feeds both tracks and records base errors.
            _record_alignment_evidence(
                alignment,
                references,
                error_positions,
                any_depths,
                confident_depths,
                confident=True,
            )

    all_depths = [depth for values in any_depths.values() for depth in values]
    denominator = max(1, aligned_bases)
    observed_errors = mismatches + insertions + deletions
    read_assembly_agreement_qv = (
        -10.0 * math.log10((observed_errors + 1) / (aligned_bases + 1)) if aligned_bases else None
    )
    summary = LibraryMappingSummary(
        library_role=library.role,
        technology=library.technology,
        quality_state=library.quality_state,
        mapped_reads=mapped_reads,
        target_fraction=mapped_reads / total_reads if total_reads else None,
        median_depth=_percentile(all_depths, 0.5),
        p5_depth=_percentile(all_depths, 0.05),
        p10_depth=_percentile(all_depths, 0.10),
        mismatch_rate=mismatches / denominator if aligned_bases else None,
        insertion_rate=insertions / denominator if aligned_bases else None,
        deletion_rate=deletions / denominator if aligned_bases else None,
        read_assembly_agreement_qv=read_assembly_agreement_qv,
    )
    return _ParsedLibrary(
        summary=summary,
        any_depths=any_depths,
        confident_depths=confident_depths,
        errors={key: dict(counts) for key, counts in error_positions.items() if counts},
        bam_artifact_id=bam_artifact_id,
    )


def _accepted_alignment(alignment: Any) -> bool:
    return bool(
        not alignment.is_unmapped
        and not alignment.is_secondary
        and not alignment.is_supplementary
        and alignment.mapping_quality >= QC_POLICY_V5.minimum_mapping_quality
    )


def _record_alignment_evidence(
    alignment: Any,
    references: dict[str, str],
    positions: dict[tuple[str, str], Counter[int]],
    any_depths: dict[str, list[int]],
    confident_depths: dict[str, list[int]],
    *,
    confident: bool,
) -> None:
    """Accumulate one alignment's per-base depth (and, for confident primaries,
    its base errors) into the two non-interchangeable depth tracks.

    Every mapped alignment passing the base-quality floor feeds the
    ``any_depths`` (repeat-aware) track. Only a confident primary (not
    secondary/supplementary, MAPQ at or above the policy floor) additionally
    feeds ``confident_depths`` and records base-error observations.
    """
    sequence_id = alignment.reference_name
    if sequence_id is None or sequence_id not in references:
        return
    reference = references[sequence_id].upper()
    query = (alignment.query_sequence or "").upper()
    qualities = alignment.query_qualities
    previous_reference = max(0, int(alignment.reference_start))
    observed: dict[tuple[str, str], set[int]] = defaultdict(set)
    for query_position, reference_position in alignment.get_aligned_pairs(matches_only=False):
        if reference_position is None:
            if confident and query_position is not None:
                observed[(sequence_id, "small_collapse")].add(previous_reference)
            continue
        previous_reference = int(reference_position)
        if query_position is None:
            if confident:
                observed[(sequence_id, "small_expansion")].add(int(reference_position))
            continue
        passes_base_quality = qualities is None or (
            int(query_position) < len(qualities)
            and int(qualities[int(query_position)]) >= QC_POLICY_V5.minimum_base_quality
        )
        if passes_base_quality and int(reference_position) < len(any_depths[sequence_id]):
            any_depths[sequence_id][int(reference_position)] += 1
            if confident:
                confident_depths[sequence_id][int(reference_position)] += 1
        if (
            confident
            and passes_base_quality
            and int(reference_position) < len(reference)
            and int(query_position) < len(query)
            and query[int(query_position)] != reference[int(reference_position)]
        ):
            observed[(sequence_id, "substitution")].add(int(reference_position))
    for key, observed_positions in observed.items():
        positions[key].update(observed_positions)


def _combine_depths(
    parsed: list[_ParsedLibrary],
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Sum per-library any-alignment and confident depths into two combined tracks."""

    any_combined: dict[str, list[int]] = {}
    confident_combined: dict[str, list[int]] = {}
    for item in parsed:
        for source, combined in (
            (item.any_depths, any_combined),
            (item.confident_depths, confident_combined),
        ):
            for sequence_id, values in source.items():
                current = combined.setdefault(sequence_id, [0] * len(values))
                if len(current) != len(values):
                    raise OrganelleExecutionError(
                        code="qc.mapping_failed",
                        message="mapped libraries disagree on reference sequence length",
                        details={"sequence_id": sequence_id},
                    )
                for position, depth in enumerate(values):
                    current[position] += depth
    return any_combined, confident_combined


def _coverage_windows(
    any_depths: dict[str, list[int]],
    confident_depths: dict[str, list[int]],
) -> tuple[CoverageWindow, ...]:
    windows: list[CoverageWindow] = []
    width = QC_POLICY_V5.coverage_window_bases
    for sequence_id, any_values in any_depths.items():
        confident_values = confident_depths.get(sequence_id, [])
        for start in range(0, len(any_values), width):
            any_window = any_values[start : start + width]
            confident_window = confident_values[start : start + width]
            end = start + len(any_window)
            windows.append(
                CoverageWindow(
                    sequence_id=sequence_id,
                    start=start,
                    end=end,
                    any_alignment_mean_depth=(statistics.fmean(any_window) if any_window else None),
                    any_alignment_minimum_depth=(float(min(any_window)) if any_window else None),
                    confident_mean_depth=(
                        statistics.fmean(confident_window) if confident_window else None
                    ),
                    confident_minimum_depth=(
                        float(min(confident_window)) if confident_window else None
                    ),
                    assessment_status="assessed" if any_window else "not_assessed",
                )
            )
    return tuple(windows)


def _sequence_depth_profiles(
    any_depths: dict[str, list[int]],
) -> tuple[SequenceDepthProfile, ...]:
    """Combined any-alignment per-base depth statistics, one profile per sequence."""

    return tuple(
        SequenceDepthProfile(
            sequence_id=sequence_id,
            median_depth=_percentile(values, 0.5),
            p5_depth=_percentile(values, 0.05),
            p10_depth=_percentile(values, 0.10),
        )
        for sequence_id, values in any_depths.items()
    )


def _write_coverage_intervals(path: Path, depths: dict[str, list[int]]) -> None:
    low_threshold = next(
        (level for level in QC_POLICY_V5.coverage_depth_levels if level > 1),
        1,
    )
    lines: list[str] = []
    for sequence_id, values in depths.items():
        labels = [
            "zero_coverage" if depth == 0 else "low_coverage" if depth < low_threshold else ""
            for depth in values
        ]
        start = 0
        while start < len(labels):
            label = labels[start]
            end = start + 1
            while end < len(labels) and labels[end] == label:
                end += 1
            if label:
                lines.append(f"{sequence_id}\t{start}\t{end}\t{label}")
            start = end
    path.write_text("".join(f"{line}\n" for line in lines))


def _error_candidates(parsed: list[_ParsedLibrary]) -> tuple[ErrorCandidate, ...]:
    counts: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    artifacts: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    confident_depths: dict[str, list[int]] = {}
    for item in parsed:
        for sequence_id, values in item.confident_depths.items():
            combined = confident_depths.setdefault(sequence_id, [0] * len(values))
            if len(combined) != len(values):
                raise OrganelleExecutionError(
                    code="qc.mapping_failed",
                    message="mapped libraries disagree on reference sequence length",
                )
            for position, depth in enumerate(values):
                combined[position] += depth
        for key, item_counts in item.errors.items():
            counts[key].update(item_counts)
            for position, support in item_counts.items():
                if support:
                    artifacts[(*key, position)].add(item.bam_artifact_id)

    candidates: list[ErrorCandidate] = []
    for sequence_id, error_type in sorted(counts):
        depths = confident_depths.get(sequence_id, [])
        position_evidence: dict[int, tuple[int, int, float]] = {}
        for position, support in counts[(sequence_id, error_type)].items():
            reference_support = depths[position] if position < len(depths) else 0
            # A read carrying a deletion relative to the assembly has no query
            # base at the deleted reference position, so it is absent from the
            # ordinary depth array. Other error types already contribute to
            # depth and must not be added twice.
            depth = (
                support + reference_support
                if error_type == "small_expansion"
                else max(support, reference_support)
            )
            fraction = support / depth if depth else 0.0
            position_evidence[position] = (support, depth, fraction)

        supported_positions = sorted(
            position
            for position in counts[(sequence_id, error_type)]
            if (
                position_evidence[position][0] >= QC_POLICY_V5.minimum_base_error_support
                and position_evidence[position][1] >= QC_POLICY_V5.minimum_base_error_depth
                and position_evidence[position][2] >= QC_POLICY_V5.minimum_base_error_fraction
            )
        )
        for cluster_index, (start, end) in enumerate(
            _cluster_positions(
                supported_positions,
                QC_POLICY_V5.error_cluster_window_bases,
            ),
            start=1,
        ):
            cluster_positions = [
                position for position in supported_positions if start <= position < end
            ]
            representative = max(
                cluster_positions,
                key=lambda position: (
                    position_evidence[position][2],
                    position_evidence[position][0],
                    -position,
                ),
            )
            support, assessed_depth, support_fraction = position_evidence[representative]
            source_artifacts = {
                artifact_id
                for position in cluster_positions
                for artifact_id in artifacts[(sequence_id, error_type, position)]
            }
            candidates.append(
                ErrorCandidate(
                    candidate_id=f"{sequence_id}:{error_type}:{cluster_index}",
                    sequence_id=sequence_id,
                    start=start,
                    end=end,
                    error_type=error_type,
                    confidence=None,
                    supporting_observations=support,
                    assessed_depth=assessed_depth,
                    support_fraction=support_fraction,
                    supporting_library_count=len(source_artifacts),
                    status="candidate",
                    evidence_artifact_ids=tuple(sorted(source_artifacts)),
                )
            )
    return tuple(candidates)


def _write_base_error_evidence(
    path: Path,
    candidates: tuple[ErrorCandidate, ...],
) -> None:
    rows = [
        "candidate_id\tsequence_id\tstart\tend\terror_type\t"
        "supporting_observations\tassessed_depth\tsupport_fraction\t"
        "supporting_library_count\tsource_bam_artifact_ids"
    ]
    rows.extend(
        f"{candidate.candidate_id}\t{candidate.sequence_id}\t{candidate.start}\t"
        f"{candidate.end}\t{candidate.error_type}\t{candidate.supporting_observations}\t"
        f"{candidate.assessed_depth}\t{candidate.support_fraction}\t"
        f"{candidate.supporting_library_count}\t"
        f"{','.join(candidate.evidence_artifact_ids)}"
        for candidate in candidates
    )
    path.write_text("".join(f"{row}\n" for row in rows))


def _cluster_positions(positions: list[int], distance: int) -> tuple[tuple[int, int], ...]:
    if not positions:
        return ()
    clusters: list[tuple[int, int]] = []
    start = positions[0]
    last = start
    for position in positions[1:]:
        if position - last > distance:
            clusters.append((start, last + 1))
            start = position
        last = position
    clusters.append((start, last + 1))
    return tuple(clusters)


def _percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[index])


def _artifact(
    path: Path,
    kind: str,
    format_name: str,
    media_type: str,
) -> ArtifactRef:
    return ArtifactRef.from_path(
        path,
        kind=kind,
        format=format_name,
        media_type=media_type,
    )


__all__ = ["collect_mapping_evidence"]
