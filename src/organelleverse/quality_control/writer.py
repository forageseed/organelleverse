"""Deterministic, atomic materialization of canonical assembly-QC Results."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import os
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from pydantic import BaseModel

from organelleverse.assembly.contracts import canonical_json_bytes
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import OrganelleResult

from .contracts import AssemblyQcReport, AssemblyQcRunManifest, ErrorCandidate
from .policy import QC_POLICY_V5

_OPERATION_VERSION = "1.2"
_SUPPORTED_SUFFIXES = frozenset({".md", ".json", ".csv", ".xlsx"})
_TABLE_NAMES = (
    "summary",
    "checks",
    "sequences",
    "junction_support",
    "repeat_support",
    "marker_hits",
    "error_candidates",
    "coverage_windows",
    "assembly_statistics",
    "library_mapping_summaries",
    "gfa_summary",
    "alternative_configurations",
    "marker_profile_assessment",
)
_DIRECTORY_FILES = {
    "summary": "summary.csv",
    "checks": "checks.csv",
    "sequences": "sequences.csv",
    "junction_support": "junction_support.csv",
    "repeat_support": "repeat_support.csv",
    "marker_hits": "marker_hits.csv",
    "error_candidates": "error_candidates.csv",
    "coverage_windows": "coverage_windows.csv",
    "assembly_statistics": "assembly_statistics.csv",
    "library_mapping_summaries": "library_mapping_summaries.csv",
    "gfa_summary": "gfa_summary.csv",
    "alternative_configurations": "alternative_configurations.csv",
    "marker_profile_assessment": "marker_profile_assessment.csv",
}
_BASE_ERROR_TYPES = frozenset({"small_expansion", "small_collapse", "substitution"})


def materialize_result(
    result: OrganelleResult,
    output: str | Path,
) -> OrganelleResult:
    """Write one successful canonical QC Result without overwriting conflicts."""

    if result.operation_id == "qc.annotation":
        from .annotation_writer import materialize_annotation_qc_result

        return materialize_annotation_qc_result(result, output)
    report, manifest = _load_source(result)
    destination = Path(output).expanduser().resolve()
    suffix = destination.suffix.lower()
    if suffix and suffix not in _SUPPORTED_SUFFIXES:
        raise OrganelleInputError(
            code="qc.unsupported_output_suffix",
            message="QC output must be a directory or .md, .json, .csv, or .xlsx file",
            details={"path": str(destination), "suffix": suffix},
        )
    if not suffix and destination.exists() and not destination.is_dir():
        raise OrganelleInputError(
            code="qc.destination_conflict",
            message="QC directory destination is occupied by a file",
            details={"path": str(destination)},
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
    try:
        if suffix:
            _write_single_file(report, temporary, suffix)
        else:
            _write_directory(report, manifest, temporary)
        _publish(temporary, destination)
    except BaseException:
        _remove(temporary)
        raise
    return _written_result(result, destination, suffix)


def _load_source(
    result: OrganelleResult,
) -> tuple[AssemblyQcReport, AssemblyQcRunManifest]:
    if result.operation_id != "qc.assembly" or result.status == "failed":
        raise OrganelleInputError(
            code="qc.write_input_contract",
            message="qc.write accepts only a non-failed qc.assembly Result",
            details={"operation_id": result.operation_id, "status": result.status},
        )
    report_artifact = _unique_artifact(result, "assembly_qc_report")
    manifest_artifact = _unique_artifact(result, "assembly_qc_run_manifest")
    report_path = _verify_artifact(report_artifact)
    manifest_path = _verify_artifact(manifest_artifact)
    try:
        report = AssemblyQcReport.model_validate_json(report_path.read_bytes())
        manifest = AssemblyQcRunManifest.model_validate_json(manifest_path.read_bytes())
    except Exception as error:
        raise OrganelleInputError(
            code="qc.artifact_contract_invalid",
            message="canonical QC report or run manifest is invalid",
            details={"reason": str(error)},
        ) from error

    provenance = result.provenance
    if (
        provenance is None
        or provenance.run_manifest_id != manifest.run_manifest_id
        or report.source_assembly_result_id != manifest.source_assembly_result_id
        or report.source_run_manifest_id != manifest.source_run_manifest_id
        or report.source_assembly_result_id not in provenance.input_object_ids
        or result.metrics.get("qc_decision") != report.decision
    ):
        raise OrganelleInputError(
            code="qc.artifact_contract_invalid",
            message="QC Result, report, and run manifest identities disagree",
        )
    for artifact in (*manifest.outputs, *report.evidence_artifacts):
        _verify_artifact(artifact)
    return report, manifest


def _unique_artifact(result: OrganelleResult, kind: str) -> ArtifactRef:
    matches = tuple(item for item in result.artifacts if item.kind == kind)
    if len(matches) != 1:
        raise OrganelleInputError(
            code="qc.write_input_contract",
            message=f"qc.assembly Result must contain exactly one {kind} artifact",
            details={"kind": kind, "count": len(matches)},
        )
    return matches[0]


def _verify_artifact(artifact: ArtifactRef) -> Path:
    path = artifact.resolve()
    try:
        current = ArtifactRef.from_path(
            path,
            kind=artifact.kind,
            format=artifact.format,
            media_type=artifact.media_type,
        )
    except OrganelleInputError as error:
        raise OrganelleInputError(
            code="qc.artifact_digest_mismatch",
            message="a canonical QC artifact is missing or unreadable",
            details={"path": str(path), "artifact_id": artifact.object_id},
        ) from error
    if current.sha256 != artifact.sha256 or current.size_bytes != artifact.size_bytes:
        raise OrganelleInputError(
            code="qc.artifact_digest_mismatch",
            message="a canonical QC artifact no longer matches its recorded digest",
            details={"path": str(path), "artifact_id": artifact.object_id},
        )
    return path


def _write_directory(
    report: AssemblyQcReport,
    manifest: AssemblyQcRunManifest,
    directory: Path,
) -> None:
    directory.mkdir()
    tables = _tables(report)
    (directory / "assembly_qc_summary.md").write_text(_markdown(report))
    _copy_canonical_artifact(
        manifest.outputs,
        "assembly_qc_report",
        directory / "assembly_qc_report.json",
    )
    # The manifest itself is not one of its own outputs; render the already
    # validated model deterministically to the same canonical JSON contract.
    (directory / "assembly_qc_run_manifest.json").write_bytes(_model_bytes(manifest))
    for name, filename in _DIRECTORY_FILES.items():
        _write_csv(tables[name], directory / filename)
    _write_error_bed(
        tuple(item for item in report.error_candidates if item.error_type in _BASE_ERROR_TYPES),
        directory / "base_error_candidates.bed",
    )
    _write_error_bed(
        tuple(item for item in report.error_candidates if item.error_type not in _BASE_ERROR_TYPES),
        directory / "structural_error_candidates.bed",
    )
    (directory / "logs").mkdir()
    _copy_logs(report, directory / "logs")
    for kind, filename in (
        ("coverage_intervals", "coverage_any_alignment.bed.gz"),
        ("coverage_intervals_confident", "coverage_confident_primary.bed.gz"),
    ):
        matches = tuple(item for item in report.evidence_artifacts if item.kind == kind)
        if len(matches) != 1:
            raise OrganelleInputError(
                code="qc.artifact_contract_invalid",
                message=f"QC report must contain exactly one {kind} artifact",
                details={"kind": kind, "count": len(matches)},
            )
        source = _verify_artifact(matches[0])
        (directory / filename).write_bytes(_gzip_bytes(source.read_bytes()))
    kmer_metrics = tuple(item for item in report.evidence_artifacts if item.kind == "kmer_metrics")
    if len(kmer_metrics) > 1:
        raise OrganelleInputError(
            code="qc.artifact_contract_invalid",
            message="QC report contains multiple k-mer metrics artifacts",
        )
    if kmer_metrics:
        source = _verify_artifact(kmer_metrics[0])
        (directory / "kmer_metrics.tsv").write_bytes(source.read_bytes())
    base_error_evidence = tuple(
        item for item in report.evidence_artifacts if item.kind == "base_error_evidence"
    )
    if len(base_error_evidence) > 1:
        raise OrganelleInputError(
            code="qc.artifact_contract_invalid",
            message="QC report contains multiple base-error evidence artifacts",
        )
    if base_error_evidence:
        source = _verify_artifact(base_error_evidence[0])
        (directory / "base_error_evidence.tsv").write_bytes(source.read_bytes())


def _copy_canonical_artifact(
    artifacts: tuple[ArtifactRef, ...],
    kind: str,
    destination: Path,
) -> None:
    matches = tuple(item for item in artifacts if item.kind == kind)
    if len(matches) != 1:
        raise OrganelleInputError(
            code="qc.artifact_contract_invalid",
            message=f"QC manifest must contain exactly one {kind} output",
            details={"kind": kind, "count": len(matches)},
        )
    destination.write_bytes(_verify_artifact(matches[0]).read_bytes())


def _copy_logs(report: AssemblyQcReport, directory: Path) -> None:
    logs = sorted(
        (item for item in report.evidence_artifacts if item.kind == "execution_log"),
        key=lambda item: (Path(item.uri).name, item.sha256),
    )
    used: set[str] = set()
    for index, artifact in enumerate(logs, 1):
        source = _verify_artifact(artifact)
        basename = source.name
        if basename in used:
            basename = f"{index:03d}-{basename}"
        used.add(basename)
        data = source.read_bytes()[: QC_POLICY_V5.log_capture_limit_bytes]
        (directory / basename).write_bytes(data)


def _write_single_file(report: AssemblyQcReport, path: Path, suffix: str) -> None:
    if suffix == ".md":
        path.write_text(_markdown(report))
    elif suffix == ".json":
        path.write_bytes(_model_bytes(report))
    elif suffix == ".csv":
        _write_csv(_tables(report)["checks"], path)
    else:
        from .xlsx import write_xlsx_tables

        write_xlsx_tables(_tables(report), path)


def _tables(report: AssemblyQcReport) -> dict[str, list[dict[str, Any]]]:
    summary = report.summary
    tables: dict[str, list[dict[str, Any]]] = {
        "summary": [
            {
                "decision": report.decision,
                "pass_count": summary.pass_count,
                "warning_count": summary.warning_count,
                "failure_count": summary.failure_count,
                "not_assessed_count": summary.not_assessed_count,
                "any_alignment_coverage_breadth": summary.any_alignment_coverage_breadth,
                "any_alignment_zero_coverage_bases": summary.any_alignment_zero_coverage_bases,
                "confident_coverage_breadth": summary.confident_coverage_breadth,
                "confident_zero_coverage_bases": summary.confident_zero_coverage_bases,
                "unsupported_junction_count": summary.unsupported_junction_count,
                "contradicted_junction_count": summary.contradicted_junction_count,
                "structural_error_candidate_count": summary.structural_error_candidate_count,
                "assessment_status": _summary_assessment_status(report),
            }
        ],
        "checks": [_row(item, assessment_status=item.status) for item in report.checks],
        "sequences": [
            _row(item, assessment_status=item.identity_evidence_status)
            for item in report.sequence_summaries
        ],
        "junction_support": [
            _row(item, assessment_status=item.status) for item in report.junction_support
        ],
        "repeat_support": [
            _row(item, assessment_status=item.status) for item in report.repeat_support
        ],
        "marker_hits": [_row(item, assessment_status="assessed") for item in report.marker_hits],
        "error_candidates": [
            _row(item, assessment_status=item.status) for item in report.error_candidates
        ],
        "coverage_windows": [
            _row(item, assessment_status=item.assessment_status) for item in report.coverage_windows
        ],
        "assembly_statistics": (
            [_row(report.assembly_statistics, assessment_status="assessed")]
            if report.assembly_statistics is not None
            else []
        ),
        "library_mapping_summaries": [
            _row(item, assessment_status="assessed") for item in report.library_mapping_summaries
        ],
        "gfa_summary": (
            [_row(report.gfa_summary, assessment_status="assessed")]
            if report.gfa_summary is not None
            else []
        ),
        "alternative_configurations": [
            _row(item, assessment_status=item.status) for item in report.alternative_configurations
        ],
        "marker_profile_assessment": (
            [
                _row(
                    report.marker_profile_assessment,
                    assessment_status=(
                        "assessed"
                        if report.marker_profile_assessment.profiles_assessed
                        else "not_assessed"
                    ),
                )
            ]
            if report.marker_profile_assessment is not None
            else []
        ),
    }
    headers = {
        "checks": (
            "check_id",
            "category",
            "status",
            "value",
            "unit",
            "message",
            "finding_code",
            "evidence_artifact_ids",
            "assessment_status",
        ),
        "sequences": (
            "sequence_id",
            "role",
            "length",
            "gc_fraction",
            "ambiguous_bases",
            "coverage_breadth",
            "median_depth",
            "p5_depth",
            "p10_depth",
            "marker_ids",
            "graph_context",
            "identity_evidence_status",
            "assessment_status",
        ),
        "junction_support": (
            "sequence_id",
            "left_segment",
            "right_segment",
            "library_role",
            "supporting_reads",
            "contradicting_reads",
            "status",
            "anchor_length",
            "assessment_status",
        ),
        "repeat_support": (
            "repeat_id",
            "sequence_id",
            "length",
            "orientation",
            "path_occurrences",
            "spanning_reads",
            "status",
            "assessment_status",
        ),
        "marker_hits": (
            "profile_id",
            "sequence_id",
            "start",
            "end",
            "strand",
            "score",
            "complete",
            "target",
            "evidence_artifact_ids",
            "assessment_status",
        ),
        "error_candidates": (
            "candidate_id",
            "sequence_id",
            "start",
            "end",
            "error_type",
            "confidence",
            "supporting_observations",
            "assessed_depth",
            "support_fraction",
            "supporting_library_count",
            "status",
            "evidence_artifact_ids",
            "assessment_status",
        ),
        "coverage_windows": (
            "sequence_id",
            "start",
            "end",
            "any_alignment_mean_depth",
            "any_alignment_minimum_depth",
            "confident_mean_depth",
            "confident_minimum_depth",
            "assessment_status",
        ),
        "assembly_statistics": (
            "sequence_count",
            "total_length",
            "largest_sequence_length",
            "n50",
            "l50",
            "overall_gc_fraction",
            "ambiguous_bases",
            "assessment_status",
        ),
        "library_mapping_summaries": (
            "library_role",
            "technology",
            "quality_state",
            "mapped_reads",
            "target_fraction",
            "median_depth",
            "p5_depth",
            "p10_depth",
            "mismatch_rate",
            "insertion_rate",
            "deletion_rate",
            "read_assembly_agreement_qv",
            "assessment_status",
        ),
        "gfa_summary": (
            "segment_count",
            "edge_count",
            "path_count",
            "component_count",
            "branch_count",
            "parallel_edge_count",
            "tip_count",
            "path_names",
            "assessment_status",
        ),
        "alternative_configurations": (
            "configuration_id",
            "sequence_id",
            "relative_support",
            "status",
            "assessment_status",
        ),
        "marker_profile_assessment": (
            "profiles_assessed",
            "profile_set_id",
            "profile_version",
            "profile_sha256",
            "genetic_code",
            "expected_target_profile_count",
            "complete_target_profile_count",
            "target_profile_recovery_fraction",
            "duplicate_complete_target_profile_count",
            "assessment_status",
        ),
    }
    for name, columns in headers.items():
        if not tables[name]:
            tables[name] = [
                {
                    column: "not_assessed" if column == "assessment_status" else ""
                    for column in columns
                }
            ]
    return {name: tables[name] for name in _TABLE_NAMES}


def _row(model: BaseModel, *, assessment_status: str) -> dict[str, Any]:
    values: dict[str, object] = model.model_dump(
        mode="json",
        exclude={"kind", "object_id"},
    )
    row: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, list):
            row[key] = ";".join(str(item) for item in cast(list[object], value))
        else:
            row[key] = value
    row["assessment_status"] = assessment_status
    return row


def _summary_assessment_status(report: AssemblyQcReport) -> str:
    return "not_assessed" if report.decision == "insufficient_evidence" else "assessed"


def _markdown(report: AssemblyQcReport) -> str:
    summary = report.summary
    lines = [
        "# Assembly QC summary",
        "",
        f"- Decision: `{report.decision}`",
        f"- Organelle: `{report.organelle}`",
        f"- Policy: `{report.policy_version}`",
        f"- Checks: {summary.pass_count} pass, {summary.warning_count} warning, "
        f"{summary.failure_count} fail, {summary.not_assessed_count} not assessed",
        "",
        "## Key evidence",
        "",
        f"- Any-alignment coverage breadth: {_display(summary.any_alignment_coverage_breadth)}",
        "- Any-alignment zero-coverage bases: "
        f"{_display(summary.any_alignment_zero_coverage_bases)}",
        f"- Confident-primary coverage breadth: {_display(summary.confident_coverage_breadth)}",
        "- Confident-primary zero-coverage bases: "
        f"{_display(summary.confident_zero_coverage_bases)}",
        f"- Unsupported junctions: {_display(summary.unsupported_junction_count)}",
        f"- Contradicted junctions: {_display(summary.contradicted_junction_count)}",
        f"- Structural error candidates: {_display(summary.structural_error_candidate_count)}",
        f"- Complete target marker-profile recovery: {_display(_marker_recovery(report))}",
        "",
        "## Checks requiring attention",
        "",
    ]
    attention = tuple(item for item in report.checks if item.status != "pass")
    if attention:
        lines.extend(f"- `{item.status}` `{item.check_id}`: {item.message}" for item in attention)
    else:
        lines.append("- None.")
    lines.extend(
        (
            "",
            "The decision is evidence-based. Metrics marked not assessed were not inferred.",
            "",
        )
    )
    return "\n".join(lines)


def _display(value: object) -> str:
    return "not assessed" if value is None else str(value)


def _marker_recovery(report: AssemblyQcReport) -> float | None:
    assessment = report.marker_profile_assessment
    return None if assessment is None else assessment.target_profile_recovery_fraction


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if value is None
                    else str(value).lower()
                    if isinstance(value, bool)
                    else value
                    for key, value in row.items()
                }
            )


def _write_error_bed(candidates: tuple[ErrorCandidate, ...], path: Path) -> None:
    lines: list[str] = []
    for item in candidates:
        name = "_".join(f"{item.candidate_id}|{item.error_type}|{item.status}".split())
        score = min(1_000, item.supporting_observations)
        lines.append(f"{item.sequence_id}\t{item.start}\t{item.end}\t{name}\t{score}\t.")
    path.write_text("".join(f"{line}\n" for line in lines))


def _model_bytes(model: Any) -> bytes:
    return canonical_json_bytes(model.model_dump(mode="json"))


def _gzip_bytes(data: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as handle:
        handle.write(data)
    return output.getvalue()


def _publish(temporary: Path, destination: Path) -> None:
    if destination.exists():
        if _same_output(temporary, destination):
            _remove(temporary)
            return
        raise OrganelleInputError(
            code="qc.destination_conflict",
            message="QC output destination already contains different content",
            details={"path": str(destination)},
        )
    try:
        os.replace(temporary, destination)
    except OSError as error:
        raise OrganelleInputError(
            code="qc.destination_conflict",
            message="unable to atomically publish QC output",
            details={"path": str(destination), "reason": str(error)},
        ) from error


def _same_output(left: Path, right: Path) -> bool:
    if left.is_file() != right.is_file() or left.is_dir() != right.is_dir():
        return False
    if left.is_file():
        return left.read_bytes() == right.read_bytes()
    left_files = {
        item.relative_to(left): item.read_bytes() for item in left.rglob("*") if item.is_file()
    }
    right_files = {
        item.relative_to(right): item.read_bytes() for item in right.rglob("*") if item.is_file()
    }
    left_dirs = {item.relative_to(left) for item in left.rglob("*") if item.is_dir()}
    right_dirs = {item.relative_to(right) for item in right.rglob("*") if item.is_dir()}
    return left_files == right_files and left_dirs == right_dirs


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _written_result(
    source: OrganelleResult,
    destination: Path,
    suffix: str,
) -> OrganelleResult:
    artifacts = (
        (_output_artifact(destination),)
        if destination.is_file()
        else tuple(
            _output_artifact(path) for path in sorted(destination.rglob("*")) if path.is_file()
        )
    )
    source_provenance = source.provenance
    upstream = (
        (source_provenance.run_manifest_id,)
        if source_provenance is not None and source_provenance.run_manifest_id is not None
        else ()
    )
    parameters_hash = hashlib.sha256(
        canonical_json_bytes({"output_mode": suffix.lstrip(".") or "directory"})
    ).hexdigest()
    provenance = ResultProvenance(
        operation_id="qc.write",
        operation_version=_OPERATION_VERSION,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        input_object_ids=(source.object_id,),
        input_artifact_hashes=tuple(item.sha256 for item in source.artifacts),
        parameters_hash=parameters_hash,
        upstream_run_manifest_ids=upstream,
    )
    return OrganelleResult(
        operation_id="qc.write",
        operation_version=_OPERATION_VERSION,
        scope=source.scope,
        status=source.status,
        summary_text="Materialized assembly QC output.",
        metrics=source.metrics,
        findings=source.findings,
        flags=tuple(dict.fromkeys((*source.flags, "qc_materialized"))),
        artifacts=artifacts,
        provenance=provenance,
    )


def _output_artifact(path: Path) -> ArtifactRef:
    suffix = path.suffix.lower()
    media_type = {
        ".md": "text/markdown",
        ".json": "application/json",
        ".csv": "text/csv",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".gz": "application/gzip",
        ".log": "text/plain",
        ".bed": "text/tab-separated-values",
    }.get(suffix, "application/octet-stream")
    format_name = path.name.removesuffix(".gz").rsplit(".", 1)[-1]
    return ArtifactRef.from_path(
        path,
        kind="qc_materialized_output",
        format=format_name,
        media_type=media_type,
    )


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


__all__ = ["materialize_result"]
