"""Canonical assembly-QC orchestration for Python and Agent callers."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, NoReturn
from uuid import uuid4

from pydantic import BaseModel

from organelleverse.assembly.contracts import canonical_json_bytes
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import (
    OrganelleDependencyError,
    OrganelleExecutionError,
    OrganelleInputError,
)
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import (
    ErrorDetail,
    Finding,
    OrganelleResult,
    ResultStatus,
)
from organelleverse.runtime import managed_run_path, publish_run

from .contracts import (
    AlternativeConfiguration,
    AssemblyQcReport,
    AssemblyQcRunManifest,
    GraphEvidence,
    MappingEvidence,
    MarkerEvidence,
    MarkerProfileAssessment,
    QcCheck,
    QcDecision,
    QcStageOutcome,
    QcSummary,
    ResolvedAssemblyEvidence,
    SequenceDepthProfile,
    SequenceIdentity,
    SequenceSummary,
)
from .environment import QcEnvironment, resolve_qc_environment
from .evidence import collect_mapping_evidence
from .graph import collect_graph_evidence
from .marker_profiles import resolve_marker_profiles
from .markers import collect_marker_evidence
from .organelle import interpret_organelle
from .policy import QC_POLICY_V5
from .static import compute_assembly_statistics, resolve_assembly_evidence, summarize_fasta

_OPERATION_VERSION = "1.5"
_DEFAULT_TIMEOUT_SECONDS = 3_600


def aggregate_decision(
    checks: Iterable[QcCheck],
    *,
    required_evidence_missing: bool,
) -> QcDecision:
    """Apply the fixed decision precedence without a numeric score."""

    statuses = tuple(item.status for item in checks)
    if "fail" in statuses:
        return "not_ready"
    if "warn" in statuses:
        return "needs_review"
    if required_evidence_missing:
        return "insufficient_evidence"
    return "ready"


def result_status_for(decision: QcDecision) -> ResultStatus:
    """Map the scientific decision to the compact core Result status."""

    return "ok" if decision == "ready" else "warning"


def run_assembly_qc(
    result: OrganelleResult,
    *,
    threads: int = 4,
    timeout_seconds: int | None = None,
) -> OrganelleResult:
    """Resolve evidence, run QC, and persist one content-addressed Result."""

    if threads < 1 or threads > 256:
        raise ValueError("threads must be between 1 and 256")
    if timeout_seconds is not None and timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    effective_timeout = timeout_seconds or _DEFAULT_TIMEOUT_SECONDS
    started_at = datetime.now(UTC)
    resolved = resolve_assembly_evidence(result)
    environment = resolve_qc_environment()
    marker_error = ""
    try:
        marker_profiles = resolve_marker_profiles()
    except OrganelleDependencyError as error:
        marker_profiles = ()
        marker_error = f"{error.code}: {error.message}"
    parameters: dict[str, object] = {
        "threads": threads,
        "timeout_seconds": effective_timeout,
        "policy_version": QC_POLICY_V5.policy_version,
        "marker_profile_ids": [item.object_id for item in marker_profiles],
    }
    parameters_hash = _sha256_json(parameters)
    run_digest = _sha256_json(
        {
            "operation_id": "qc.assembly",
            "operation_version": _OPERATION_VERSION,
            "input_result_id": result.object_id,
            "source_run_manifest_id": resolved.run_manifest.run_manifest_id,
            "policy_id": QC_POLICY_V5.object_id,
            "parameters_hash": parameters_hash,
            "environment_id": environment.environment_id,
            "marker_profile_ids": [item.object_id for item in marker_profiles],
        }
    )
    run_dir = managed_run_path("qc.assembly", f"sha256-{run_digest}")
    if run_dir.exists():
        return _load_reusable_result(run_dir, result.object_id)

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = run_dir.parent / f".{run_dir.name}.tmp-{uuid4().hex}"
    temporary.mkdir()
    try:
        try:
            mapping = collect_mapping_evidence(
                resolved,
                environment,
                workspace=temporary / "mapping",
                threads=threads,
                timeout_seconds=effective_timeout,
            )
        except OrganelleExecutionError as error:
            failed = _persist_failed_run(
                error,
                result,
                resolved,
                environment,
                parameters,
                parameters_hash,
                temporary,
                run_dir,
                started_at,
            )
            publish_run(temporary, run_dir)
            return failed

        graph = collect_graph_evidence(resolved, mapping, QC_POLICY_V5)
        mapping = _retain_mapping_evidence(mapping)
        markers = (
            collect_marker_evidence(
                resolved,
                marker_profiles,
                QC_POLICY_V5,
            )
            if marker_profiles
            else _unavailable_marker_evidence(marker_error)
        )
        organelle = interpret_organelle(
            resolved,
            mapping,
            graph,
            markers,
            QC_POLICY_V5,
        )
        computed = _build_success(
            result,
            resolved,
            environment,
            mapping,
            graph,
            markers,
            tuple(item.hmm_artifact for item in marker_profiles),
            organelle.checks,
            organelle.sequence_identities,
            organelle.alternative_configurations,
            parameters,
            parameters_hash,
            temporary,
            run_dir,
            started_at,
        )
        publish_run(temporary, run_dir)
        return computed
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _build_success(
    source_result: OrganelleResult,
    resolved: ResolvedAssemblyEvidence,
    environment: QcEnvironment,
    mapping: MappingEvidence,
    graph: GraphEvidence,
    markers: MarkerEvidence,
    marker_artifacts: tuple[ArtifactRef, ...],
    organelle_checks: tuple[QcCheck, ...],
    sequence_identities: tuple[SequenceIdentity, ...],
    alternative_configurations: tuple[AlternativeConfiguration, ...],
    parameters: dict[str, object],
    parameters_hash: str,
    temporary: Path,
    run_dir: Path,
    started_at: datetime,
) -> OrganelleResult:
    static_summaries = summarize_fasta(resolved.primary_fasta_path)
    assembly_statistics = compute_assembly_statistics(resolved.primary_fasta_path)
    any_zero_by_sequence, confident_zero_by_sequence = _zero_coverage_by_sequence(mapping)
    _validate_coverage_coordinates(
        mapping, static_summaries, any_zero_by_sequence, confident_zero_by_sequence
    )
    total_bases = assembly_statistics.total_length
    any_zero_bases = sum(any_zero_by_sequence.values())
    confident_zero_bases = sum(confident_zero_by_sequence.values())
    coverage_assessed = bool(mapping.coverage_windows)
    any_breadth = (
        (total_bases - any_zero_bases) / total_bases if coverage_assessed and total_bases else None
    )
    confident_breadth = (
        (total_bases - confident_zero_bases) / total_bases
        if coverage_assessed and total_bases
        else None
    )
    coverage_checks = _coverage_checks(coverage_assessed, any_zero_bases, confident_zero_bases)
    integrity_check = QcCheck(
        check_id="qc.evidence_integrity",
        category="integrity",
        status="pass",
        value=len(resolved.input_libraries),
        unit="libraries",
        message="Assembly Result, manifest, Genome, and consumed artifact identities agree.",
    )
    checks = (
        integrity_check,
        *mapping.checks,
        *graph.checks,
        *markers.checks,
        *organelle_checks,
        *coverage_checks,
    )
    required_missing = (
        not resolved.input_libraries
        or not coverage_assessed
        or graph.gfa_summary is None
        or not markers.profiles_assessed
    )
    decision = aggregate_decision(checks, required_evidence_missing=required_missing)
    counts = _check_counts(checks)
    unsupported = sum(item.status == "unsupported" for item in graph.junction_support)
    contradicted = sum(item.status == "contradicted" for item in graph.junction_support)
    error_candidates = (
        *mapping.base_error_candidates,
        *graph.structural_error_candidates,
    )
    summary = QcSummary(
        pass_count=counts["pass"],
        warning_count=counts["warn"],
        failure_count=counts["fail"],
        not_assessed_count=counts["not_assessed"],
        any_alignment_coverage_breadth=any_breadth,
        any_alignment_zero_coverage_bases=any_zero_bases if coverage_assessed else None,
        confident_coverage_breadth=confident_breadth,
        confident_zero_coverage_bases=confident_zero_bases if coverage_assessed else None,
        unsupported_junction_count=unsupported,
        contradicted_junction_count=contradicted,
        structural_error_candidate_count=len(graph.structural_error_candidates),
    )
    identity_by_sequence = {
        item.sequence_id: item.identity_evidence_status for item in sequence_identities
    }
    marker_ids_by_sequence: dict[str, list[str]] = {}
    for hit in markers.marker_hits:
        marker_ids_by_sequence.setdefault(hit.sequence_id, []).append(hit.profile_id)
    depth_by_sequence = {
        profile.sequence_id: profile for profile in mapping.sequence_depth_profiles
    }
    sequence_summaries = tuple(
        item.model_copy(
            update={
                "coverage_breadth": (
                    (item.length - any_zero_by_sequence.get(item.sequence_id, 0)) / item.length
                    if coverage_assessed
                    else None
                ),
                "median_depth": _depth_field(depth_by_sequence, item.sequence_id, "median_depth")
                if coverage_assessed
                else None,
                "p5_depth": _depth_field(depth_by_sequence, item.sequence_id, "p5_depth")
                if coverage_assessed
                else None,
                "p10_depth": _depth_field(depth_by_sequence, item.sequence_id, "p10_depth")
                if coverage_assessed
                else None,
                "marker_ids": tuple(marker_ids_by_sequence.get(item.sequence_id, ())),
                "identity_evidence_status": identity_by_sequence.get(
                    item.sequence_id, "unresolved"
                ),
            }
        )
        for item in static_summaries
    )
    relocated_mapping = _relocate_artifacts(
        mapping.coordinate_artifacts,
        temporary,
        run_dir,
    )
    relocated_evidence = (*relocated_mapping, *marker_artifacts)
    report = AssemblyQcReport(
        policy_version=QC_POLICY_V5.policy_version,
        source_assembly_result_id=source_result.object_id,
        source_run_manifest_id=resolved.run_manifest.run_manifest_id,
        organelle=resolved.primary_genome.organelle,
        decision=decision,
        summary=summary,
        checks=checks,
        sequence_summaries=sequence_summaries,
        junction_support=graph.junction_support,
        repeat_support=graph.repeat_support,
        marker_hits=markers.marker_hits,
        marker_profile_assessment=MarkerProfileAssessment(
            profiles_assessed=markers.profiles_assessed,
            profile_set_id=markers.profile_set_id,
            profile_version=markers.profile_version,
            profile_sha256=markers.profile_sha256,
            genetic_code=markers.genetic_code,
            expected_target_profile_count=markers.expected_target_profile_count,
            complete_target_profile_count=markers.complete_target_profile_count,
            target_profile_recovery_fraction=markers.target_profile_recovery_fraction,
            duplicate_complete_target_profile_count=(
                markers.duplicate_complete_target_profile_count
            ),
        ),
        error_candidates=error_candidates,
        coverage_windows=mapping.coverage_windows,
        library_mapping_summaries=mapping.library_mapping_summaries,
        gfa_summary=graph.gfa_summary,
        alternative_configurations=alternative_configurations,
        assembly_statistics=assembly_statistics,
        evidence_artifacts=relocated_evidence,
        tool_identities=environment.tool_identities,
    )
    report_path = temporary / "assembly_qc_report.json"
    report_path.write_bytes(_model_bytes(report))
    report_artifact = _relocated_artifact(
        report_path,
        run_dir / report_path.name,
        kind="assembly_qc_report",
    )
    manifest = AssemblyQcRunManifest(
        policy_version=QC_POLICY_V5.policy_version,
        source_assembly_result_id=source_result.object_id,
        source_run_manifest_id=resolved.run_manifest.run_manifest_id,
        parameters=FrozenMap.from_json(parameters),
        environment_id=environment.environment_id,
        tool_identities=environment.tool_identities,
        stages=(
            QcStageOutcome(stage="resolve_input", status="ok"),
            QcStageOutcome(stage="resolve_environment", status="ok"),
            QcStageOutcome(
                stage="map_reads",
                status="ok",
                output_artifact_ids=tuple(item.object_id for item in relocated_mapping),
            ),
            QcStageOutcome(stage="interpret_graph", status="ok"),
            QcStageOutcome(
                stage="scan_markers",
                status="ok" if marker_artifacts else "skipped",
                output_artifact_ids=tuple(item.object_id for item in marker_artifacts),
            ),
            QcStageOutcome(stage="aggregate_decision", status="ok"),
        ),
        outputs=(report_artifact, *relocated_evidence),
    )
    manifest_path = temporary / "assembly_qc_run_manifest.json"
    manifest_path.write_bytes(_model_bytes(manifest))
    manifest_artifact = _relocated_artifact(
        manifest_path,
        run_dir / manifest_path.name,
        kind="assembly_qc_run_manifest",
    )
    finished_at = datetime.now(UTC)
    result = OrganelleResult(
        operation_id="qc.assembly",
        operation_version=_OPERATION_VERSION,
        scope=resolved.primary_genome.organelle,
        status=result_status_for(decision),
        summary_text=_summary_text(decision, summary),
        metrics=FrozenMap.from_json(_metrics(decision, summary)),
        findings=_findings(checks),
        flags=(f"qc_{decision}",),
        artifacts=(report_artifact, manifest_artifact),
        provenance=_provenance(
            source_result,
            resolved,
            environment,
            parameters_hash,
            manifest.run_manifest_id,
            started_at,
            finished_at,
        ),
    )
    (temporary / "result.json").write_bytes(_model_bytes(result))
    _validate_success_bundle(result, report, manifest)
    return result


def _persist_failed_run(
    error: OrganelleExecutionError,
    source_result: OrganelleResult,
    resolved: ResolvedAssemblyEvidence,
    environment: QcEnvironment,
    parameters: dict[str, object],
    parameters_hash: str,
    temporary: Path,
    run_dir: Path,
    started_at: datetime,
) -> OrganelleResult:
    logs = tuple(
        _relocated_artifact(path, run_dir / path.relative_to(temporary), kind="execution_log")
        for path in sorted(temporary.rglob("*.log"))
        if path.is_file()
    )
    manifest = AssemblyQcRunManifest(
        policy_version=QC_POLICY_V5.policy_version,
        source_assembly_result_id=source_result.object_id,
        source_run_manifest_id=resolved.run_manifest.run_manifest_id,
        parameters=FrozenMap.from_json(parameters),
        environment_id=environment.environment_id,
        tool_identities=environment.tool_identities,
        stages=(
            QcStageOutcome(stage="resolve_input", status="ok"),
            QcStageOutcome(stage="resolve_environment", status="ok"),
            QcStageOutcome(
                stage="map_reads",
                status="failed",
                output_artifact_ids=tuple(item.object_id for item in logs),
            ),
        ),
        outputs=logs,
    )
    manifest_path = temporary / "assembly_qc_run_manifest.json"
    manifest_path.write_bytes(_model_bytes(manifest))
    manifest_artifact = _relocated_artifact(
        manifest_path,
        run_dir / manifest_path.name,
        kind="assembly_qc_run_manifest",
    )
    payload = error.as_dict()
    finished_at = datetime.now(UTC)
    result = OrganelleResult(
        operation_id="qc.assembly",
        operation_version=_OPERATION_VERSION,
        scope=resolved.primary_genome.organelle,
        status="failed",
        summary_text=error.message,
        artifacts=(manifest_artifact, *logs),
        provenance=_provenance(
            source_result,
            resolved,
            environment,
            parameters_hash,
            manifest.run_manifest_id,
            started_at,
            finished_at,
        ),
        errors=(
            ErrorDetail(
                code=error.code,
                message=error.message,
                details=FrozenMap.from_json(payload["details"]),
                retryable=error.retryable,
                suggested_action=FrozenMap.from_json(payload["suggested_action"]),
            ),
        ),
    )
    (temporary / "result.json").write_bytes(_model_bytes(result))
    AssemblyQcRunManifest.model_validate_json(_model_bytes(manifest))
    OrganelleResult.model_validate_json(_model_bytes(result))
    return result


def _unavailable_marker_evidence(reason: str = "") -> MarkerEvidence:
    return MarkerEvidence(
        checks=(
            QcCheck(
                check_id="qc.marker_profiles",
                category="identity",
                status="not_assessed",
                value=None,
                message=(
                    f"Release-pinned plant-organelle marker profiles are unavailable: {reason}"
                    if reason
                    else "Release-pinned plant-organelle marker profiles are unavailable."
                ),
            ),
        )
    )


def _retain_mapping_evidence(mapping: MappingEvidence) -> MappingEvidence:
    """Discard large alignment intermediates after graph evidence is complete."""

    transient_kinds = {"alignment_sam", "alignment_bam", "alignment_index"}
    for artifact in mapping.coordinate_artifacts:
        if artifact.kind in transient_kinds:
            Path(artifact.uri).unlink(missing_ok=True)
    retained = tuple(
        artifact
        for artifact in mapping.coordinate_artifacts
        if artifact.kind not in transient_kinds
    )
    return mapping.model_copy(
        update={
            "library_alignments": (),
            "coordinate_artifacts": retained,
        }
    )


def _coverage_checks(
    assessed: bool,
    any_zero_bases: int,
    confident_zero_bases: int,
) -> tuple[QcCheck, ...]:
    return (
        QcCheck(
            check_id="qc.coverage_assessed",
            category="read_support",
            status="pass" if assessed else "not_assessed",
            value=assessed,
            message=(
                "Read coverage was assessed." if assessed else "Read coverage was not assessed."
            ),
        ),
        # The any-alignment track counts every mapped alignment, so a base with
        # zero depth on it is assembly-missing evidence and blocks release.
        QcCheck(
            check_id="qc.zero_coverage",
            category="read_support",
            status=(
                "fail" if assessed and any_zero_bases else "pass" if assessed else "not_assessed"
            ),
            value=any_zero_bases if assessed else None,
            unit="bases",
            message=(
                "At least one reported base has zero read coverage."
                if assessed and any_zero_bases
                else "No zero-coverage reported base was detected."
                if assessed
                else "Zero coverage could not be assessed."
            ),
            finding_code=("qc.zero_coverage_region" if assessed and any_zero_bases else ""),
        ),
        # A confident-track gap that the any-alignment track still covers is
        # repeat-like ambiguity, not an assembly deletion: warn for review only.
        QcCheck(
            check_id="qc.confident_zero_coverage",
            category="read_support",
            status=(
                "warn"
                if assessed and confident_zero_bases
                else "pass"
                if assessed
                else "not_assessed"
            ),
            value=confident_zero_bases if assessed else None,
            unit="bases",
            message=(
                "At least one reported base has no confident primary coverage."
                if assessed and confident_zero_bases
                else "Every reported base has confident primary coverage."
                if assessed
                else "Confident coverage could not be assessed."
            ),
            finding_code=(
                "qc.confident_zero_coverage_region" if assessed and confident_zero_bases else ""
            ),
        ),
    )


def _depth_field(
    profiles: Mapping[str, SequenceDepthProfile],
    sequence_id: str,
    attribute: Literal["median_depth", "p5_depth", "p10_depth"],
) -> float | None:
    profile = profiles.get(sequence_id)
    return None if profile is None else getattr(profile, attribute)


def _zero_coverage_by_sequence(
    mapping: MappingEvidence,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return precise per-sequence zero-base counts for both coverage tracks.

    Counts come from the per-track coverage-interval BED artifacts. When an
    artifact is absent (e.g. a directly-injected MappingEvidence with no run
    store), the assessed windows are used as an exact fallback: a window whose
    track mean is exactly 0 has every base at depth 0 (the mean of non-negative
    values is 0 iff all are 0), so the whole window is zero on that track. A
    window with mean > 0 contributes 0 precise zero bases — its precise count is
    never guessed from the mean.
    """
    any_artifact = next(
        (item for item in mapping.coordinate_artifacts if item.kind == "coverage_intervals"),
        None,
    )
    confident_artifact = next(
        (
            item
            for item in mapping.coordinate_artifacts
            if item.kind == "coverage_intervals_confident"
        ),
        None,
    )
    any_zero = _read_zero_intervals(any_artifact) if any_artifact is not None else {}
    confident_zero = (
        _read_zero_intervals(confident_artifact) if confident_artifact is not None else {}
    )
    if any_artifact is None:
        for window in mapping.coverage_windows:
            if window.assessment_status == "assessed" and window.any_alignment_mean_depth == 0:
                any_zero[window.sequence_id] = any_zero.get(window.sequence_id, 0) + (
                    window.end - window.start
                )
    if confident_artifact is None:
        for window in mapping.coverage_windows:
            if window.assessment_status == "assessed" and window.confident_mean_depth == 0:
                confident_zero[window.sequence_id] = confident_zero.get(window.sequence_id, 0) + (
                    window.end - window.start
                )
    return any_zero, confident_zero


def _read_zero_intervals(artifact: ArtifactRef) -> dict[str, int]:
    zero: dict[str, int] = {}
    for line in Path(artifact.uri).read_text().splitlines():
        fields = line.split("\t")
        if len(fields) >= 4 and fields[3] == "zero_coverage":
            zero[fields[0]] = zero.get(fields[0], 0) + int(fields[2]) - int(fields[1])
    return zero


def _validate_coverage_coordinates(
    mapping: MappingEvidence,
    summaries: tuple[SequenceSummary, ...],
    any_zero_by_sequence: dict[str, int],
    confident_zero_by_sequence: dict[str, int],
) -> None:
    lengths = {item.sequence_id: item.length for item in summaries}
    for window in mapping.coverage_windows:
        length = lengths.get(window.sequence_id)
        if length is None or window.start < 0 or window.end <= window.start or window.end > length:
            raise RuntimeError("mapping evidence contains an invalid coverage window")
    for zero_by_sequence in (any_zero_by_sequence, confident_zero_by_sequence):
        if any(
            sequence_id not in lengths or count < 0 or count > lengths[sequence_id]
            for sequence_id, count in zero_by_sequence.items()
        ):
            raise RuntimeError("mapping evidence contains invalid zero-coverage intervals")


def _check_counts(checks: tuple[QcCheck, ...]) -> dict[str, int]:
    return {
        status: sum(item.status == status for item in checks)
        for status in ("pass", "warn", "fail", "not_assessed")
    }


def _metrics(decision: QcDecision, summary: QcSummary) -> dict[str, object]:
    return {
        "qc_decision": decision,
        "qc_pass_count": summary.pass_count,
        "qc_warning_count": summary.warning_count,
        "qc_failure_count": summary.failure_count,
        "qc_not_assessed_count": summary.not_assessed_count,
        "any_alignment_coverage_breadth": summary.any_alignment_coverage_breadth,
        "any_alignment_zero_coverage_bases": summary.any_alignment_zero_coverage_bases,
        "confident_coverage_breadth": summary.confident_coverage_breadth,
        "confident_zero_coverage_bases": summary.confident_zero_coverage_bases,
        "unsupported_junction_count": summary.unsupported_junction_count,
        "contradicted_junction_count": summary.contradicted_junction_count,
        "structural_error_candidate_count": summary.structural_error_candidate_count,
    }


def _findings(checks: tuple[QcCheck, ...]) -> tuple[Finding, ...]:
    return tuple(
        Finding(
            code=check.finding_code,
            metric=check.check_id,
            value=check.value,
            unit=check.unit,
            evidence_artifact_ids=check.evidence_artifact_ids,
        )
        for check in checks
        if check.finding_code
    )


def _summary_text(decision: QcDecision, summary: QcSummary) -> str:
    return (
        f"Assembly QC decision: {decision}. "
        f"{summary.failure_count} failed, {summary.warning_count} warning, "
        f"{summary.not_assessed_count} not assessed."
    )


def _provenance(
    source_result: OrganelleResult,
    resolved: ResolvedAssemblyEvidence,
    environment: QcEnvironment,
    parameters_hash: str,
    run_manifest_id: str,
    started_at: datetime,
    finished_at: datetime,
) -> ResultProvenance:
    return ResultProvenance(
        operation_id="qc.assembly",
        operation_version=_OPERATION_VERSION,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        input_object_ids=(source_result.object_id,),
        input_artifact_hashes=tuple(item.sha256 for item in source_result.artifacts),
        parameters_hash=parameters_hash,
        software_versions=FrozenMap.from_json(
            {item.name: item.version for item in environment.tool_identities}
        ),
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=max(0.0, (finished_at - started_at).total_seconds()),
        run_manifest_id=run_manifest_id,
        upstream_run_manifest_ids=(resolved.run_manifest.run_manifest_id,),
    )


def _relocate_artifacts(
    artifacts: tuple[ArtifactRef, ...],
    temporary: Path,
    run_dir: Path,
) -> tuple[ArtifactRef, ...]:
    relocated: list[ArtifactRef] = []
    for artifact in artifacts:
        path = Path(artifact.uri).resolve()
        try:
            relative = path.relative_to(temporary.resolve())
        except ValueError:
            relocated.append(artifact)
        else:
            relocated.append(artifact.model_copy(update={"uri": str(run_dir / relative)}))
    return tuple(relocated)


def _relocated_artifact(
    current_path: Path,
    final_path: Path,
    *,
    kind: str,
) -> ArtifactRef:
    artifact = ArtifactRef.from_path(
        current_path,
        kind=kind,
        format="json" if current_path.suffix == ".json" else "text",
        media_type=("application/json" if current_path.suffix == ".json" else "text/plain"),
    )
    return artifact.model_copy(update={"uri": str(final_path)})


def _load_reusable_result(run_dir: Path, input_result_id: str) -> OrganelleResult:
    result_path = run_dir / "result.json"
    if not result_path.is_file():
        _reuse_conflict(run_dir, "result.json is missing")
    result: OrganelleResult
    try:
        result = OrganelleResult.model_validate_json(result_path.read_bytes())
    except Exception as error:
        _reuse_conflict(run_dir, f"result.json is invalid: {error}")
    assert isinstance(result, OrganelleResult)
    provenance = result.provenance
    if provenance is None or provenance.input_object_ids != (input_result_id,):
        _reuse_conflict(run_dir, "result input identity does not match")
    for artifact in result.artifacts:
        _verify_reusable_artifact(artifact, run_dir)
    manifest_artifact = next(
        (item for item in result.artifacts if item.kind == "assembly_qc_run_manifest"),
        None,
    )
    if manifest_artifact is None:
        _reuse_conflict(run_dir, "QC run manifest artifact is missing")
    manifest_path = _verify_reusable_artifact(manifest_artifact, run_dir)
    manifest = AssemblyQcRunManifest.model_validate_json(manifest_path.read_bytes())
    if (
        provenance.run_manifest_id != manifest.run_manifest_id
        or manifest.source_assembly_result_id != input_result_id
    ):
        _reuse_conflict(run_dir, "QC run manifest identity does not match")
    for artifact in manifest.outputs:
        _verify_reusable_artifact(artifact, run_dir)
    if result.status != "failed":
        report_artifact = next(
            (item for item in result.artifacts if item.kind == "assembly_qc_report"),
            None,
        )
        if report_artifact is None:
            _reuse_conflict(run_dir, "QC report artifact is missing")
        report_path = _verify_reusable_artifact(report_artifact, run_dir)
        report = AssemblyQcReport.model_validate_json(report_path.read_bytes())
        if report.source_assembly_result_id != input_result_id:
            _reuse_conflict(run_dir, "QC report input identity does not match")
        for artifact in report.evidence_artifacts:
            _verify_reusable_artifact(artifact, run_dir)
    return result


def _verify_reusable_artifact(artifact: ArtifactRef, run_dir: Path) -> Path:
    candidate = Path(artifact.uri)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    if not candidate.is_file():
        _reuse_conflict(run_dir, f"artifact is missing: {candidate.name}")
    current = ArtifactRef.from_path(
        candidate,
        kind=artifact.kind,
        format=artifact.format,
        media_type=artifact.media_type,
    )
    if current.sha256 != artifact.sha256 or current.size_bytes != artifact.size_bytes:
        _reuse_conflict(run_dir, f"artifact digest mismatch: {candidate.name}")
    return candidate


def _reuse_conflict(run_dir: Path, reason: str) -> NoReturn:
    raise OrganelleInputError(
        code="qc.destination_conflict",
        message="existing content-addressed QC run failed verification",
        details={"path": str(run_dir), "reason": reason},
    )


def _validate_success_bundle(
    result: OrganelleResult,
    report: AssemblyQcReport,
    manifest: AssemblyQcRunManifest,
) -> None:
    if result.provenance is None or result.provenance.run_manifest_id != manifest.run_manifest_id:
        raise RuntimeError("QC Result and run manifest identities disagree")
    if report.source_assembly_result_id != manifest.source_assembly_result_id:
        raise RuntimeError("QC report and run manifest source identities disagree")
    if AssemblyQcReport.model_validate_json(_model_bytes(report)) != report:
        raise RuntimeError("QC report failed canonical round-trip validation")
    if AssemblyQcRunManifest.model_validate_json(_model_bytes(manifest)) != manifest:
        raise RuntimeError("QC run manifest failed canonical round-trip validation")
    if OrganelleResult.model_validate_json(_model_bytes(result)) != result:
        raise RuntimeError("QC Result failed canonical round-trip validation")


def _model_bytes(model: BaseModel) -> bytes:
    payload: object = model.model_dump(mode="json")
    return canonical_json_bytes(payload)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


__all__ = ["aggregate_decision", "result_status_for", "run_assembly_qc"]
