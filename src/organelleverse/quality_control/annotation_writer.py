"""Deterministic, atomic materialization of canonical annotation-QC Results.

This is the annotation-QC half of the unified ``qc.write`` operation. It
independently re-verifies the two canonical artifacts and the
Result/report/manifest/provenance identities before rendering the human and
machine bundle, then publishes atomically with the same byte-identical reuse and
destination-conflict semantics as the assembly writer. It does not score the
annotation, claim completeness, or copy annotation predictions: the canonical
report and run manifest are the only sources.
"""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Any, Never
from uuid import uuid4

from organelleverse.assembly.contracts import canonical_json_bytes
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import Finding, OrganelleResult

from .annotation_contracts import (
    AnnotationFeatureSummary,
    AnnotationQcReport,
    AnnotationQcRunManifest,
    GeneProfileAssessment,
    QcCheck,
    RejectedCdsCandidate,
)
from .contracts import QcDecision
from .operations import QC_WRITE_SPEC
from .writer import (
    _output_artifact,  # pyright: ignore[reportPrivateUsage]
    _package_version,  # pyright: ignore[reportPrivateUsage]
    _publish,  # pyright: ignore[reportPrivateUsage]
    _remove,  # pyright: ignore[reportPrivateUsage]
    _verify_artifact,  # pyright: ignore[reportPrivateUsage]
)

_SUPPORTED_SUFFIXES = frozenset({".md", ".json", ".csv", ".xlsx"})
_OPERATION_VERSION = QC_WRITE_SPEC.contract_version

_REPORT_KIND = "annotation_qc_report"
_MANIFEST_KIND = "annotation_qc_run_manifest"
_SOURCE_ARTIFACT_KINDS = frozenset({"annotation", "annotation_manifest"})
_ANNOTATION_QC_OPERATION_VERSION = "1.0"
_ANNOTATION_QC_STAGES = (
    "resolve_source",
    "validate_document",
    "assess_profile",
    "validate_exports",
    "aggregate_decision",
)

_TABLE_NAMES = (
    "summary",
    "checks",
    "features",
    "gene_profile",
    "rejected_cds_candidates",
)

_TABLE_HEADERS: dict[str, tuple[str, ...]] = {
    "summary": (
        "decision",
        "pass_count",
        "warning_count",
        "failure_count",
        "not_assessed_count",
        "record_count",
        "feature_count",
        "unique_pcg_count",
        "unique_trna_count",
        "unique_rrna_count",
        "rejected_cds_candidate_count",
        "partial_feature_count",
        "pseudo_feature_count",
        "translation_exempt_feature_count",
        "duplicate_gene_count",
        "expected_profile_gene_count",
        "recovered_profile_gene_count",
        "duplicated_profile_gene_count",
        "missing_profile_gene_count",
        "profile_recovery_fraction",
        "assessment_status",
    ),
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
    "features": (
        "record_id",
        "feature_id",
        "feature_type",
        "gene_name",
        "strand",
        "location_parts",
        "spliced_length",
        "is_partial",
        "is_pseudo",
        "is_translation_exempt",
        "is_trans_spliced",
        "parents",
        "assessment_status",
    ),
    "gene_profile": (
        "profile_id",
        "gene_name",
        "class",
        "status",
        "copy_count",
    ),
    "rejected_cds_candidates": (
        "gene_name",
        "start",
        "end",
        "strand",
        "parts",
        "issue_codes",
        "issue_messages",
        "assessment_status",
    ),
}


def materialize_annotation_qc_result(
    result: OrganelleResult,
    output: str | Path,
) -> OrganelleResult:
    """Write one non-failed ``qc.annotation`` Result as a human/machine bundle."""

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


# ---------------------------------------------------------------------------
# Source loading and independent identity verification
# ---------------------------------------------------------------------------


def _load_source(
    result: OrganelleResult,
) -> tuple[AnnotationQcReport, AnnotationQcRunManifest]:
    if result.operation_id != "qc.annotation" or result.status == "failed":
        raise OrganelleInputError(
            code="qc.write_input_contract",
            message="qc.write accepts only a non-failed qc.annotation Result",
            details={"operation_id": result.operation_id, "status": result.status},
        )
    if len(result.artifacts) != 2:
        raise OrganelleInputError(
            code="qc.write_input_contract",
            message="qc.annotation Result must contain exactly two canonical artifacts",
            details={"count": len(result.artifacts)},
        )
    report_artifact = _unique_artifact(result, _REPORT_KIND)
    manifest_artifact = _unique_artifact(result, _MANIFEST_KIND)
    report_path = _verify_artifact(report_artifact)
    manifest_path = _verify_artifact(manifest_artifact)
    try:
        report = AnnotationQcReport.model_validate_json(report_path.read_bytes())
        manifest = AnnotationQcRunManifest.model_validate_json(manifest_path.read_bytes())
    except Exception as error:
        raise OrganelleInputError(
            code="qc.artifact_contract_invalid",
            message="canonical annotation QC report or run manifest is invalid",
            details={"reason": str(error)},
        ) from error

    _verify_manifest_contract(report, manifest, report_artifact)
    _verify_result_contract(result, report, manifest)
    return report, manifest


def _unique_artifact(result: OrganelleResult, kind: str) -> ArtifactRef:
    matches = tuple(item for item in result.artifacts if item.kind == kind)
    if len(matches) != 1:
        raise OrganelleInputError(
            code="qc.write_input_contract",
            message=f"qc.annotation Result must contain exactly one {kind} artifact",
            details={"kind": kind, "count": len(matches)},
        )
    return matches[0]


def _verify_manifest_contract(
    report: AnnotationQcReport,
    manifest: AnnotationQcRunManifest,
    report_artifact: ArtifactRef,
) -> None:
    if (
        report.source_annotation_result_id != manifest.source_annotation_result_id
        or report.source_run_manifest_id != manifest.source_run_manifest_id
        or report.source_annotation_object_id != manifest.source_annotation_object_id
        or report.policy_version != manifest.policy_version
        or manifest.parameters_hash != hashlib.sha256(canonical_json_bytes({})).hexdigest()
        or manifest.outputs != (report_artifact,)
    ):
        _raise_contract_mismatch("report and run manifest identities disagree")

    export_checks = tuple(
        item for item in report.checks if item.check_id == "annotation.export_compatibility"
    )
    if len(export_checks) != 1:
        _raise_contract_mismatch(
            "report must contain exactly one annotation export-compatibility check"
        )
    expected_stage_statuses = tuple(
        "failed" if stage == "validate_exports" and export_checks[0].status == "fail" else "ok"
        for stage in _ANNOTATION_QC_STAGES
    )
    expected_stage_outputs = tuple(
        (report_artifact.object_id,) if stage == "validate_exports" else ()
        for stage in _ANNOTATION_QC_STAGES
    )
    if (
        tuple(item.stage for item in manifest.stages) != _ANNOTATION_QC_STAGES
        or tuple(item.status for item in manifest.stages) != expected_stage_statuses
        or tuple(item.output_artifact_ids for item in manifest.stages) != expected_stage_outputs
    ):
        _raise_contract_mismatch("run manifest stages disagree with the canonical report")


def _verify_result_contract(
    result: OrganelleResult,
    report: AnnotationQcReport,
    manifest: AnnotationQcRunManifest,
) -> None:
    provenance = result.provenance
    if provenance is None:
        _raise_contract_mismatch("Result provenance is missing")
    started_at = provenance.started_at
    finished_at = provenance.finished_at
    if started_at is None or finished_at is None:
        _raise_contract_mismatch("Result provenance timestamps are missing")
    if (
        provenance.run_manifest_id != manifest.run_manifest_id
        or provenance.input_object_ids != (report.source_annotation_result_id,)
        or provenance.parameters_hash != manifest.parameters_hash
        or provenance.package_version != manifest.package_version
        or provenance.upstream_run_manifest_ids != (report.source_run_manifest_id,)
    ):
        _raise_contract_mismatch("Result provenance disagrees with the report or run manifest")

    source_artifacts = report.evidence_artifacts
    if (
        len(source_artifacts) != 2
        or frozenset(item.kind for item in source_artifacts) != _SOURCE_ARTIFACT_KINDS
        or not {item.sha256 for item in source_artifacts}.issubset(provenance.input_artifact_hashes)
    ):
        _raise_contract_mismatch(
            "report source artifacts disagree with Result provenance input hashes"
        )

    expected_provenance = ResultProvenance(
        operation_id="qc.annotation",
        operation_version=_ANNOTATION_QC_OPERATION_VERSION,
        package_version=manifest.package_version,
        git_commit=provenance.git_commit,
        input_object_ids=(report.source_annotation_result_id,),
        input_artifact_hashes=provenance.input_artifact_hashes,
        parameters_hash=manifest.parameters_hash,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=max(0.0, (finished_at - started_at).total_seconds()),
        run_manifest_id=manifest.run_manifest_id,
        upstream_run_manifest_ids=(report.source_run_manifest_id,),
    )
    expected = OrganelleResult(
        operation_id="qc.annotation",
        operation_version=_ANNOTATION_QC_OPERATION_VERSION,
        scope="mitochondrion",
        status="ok" if report.decision == "ready" else "warning",
        summary_text=(
            f"Annotation QC decision: {report.decision}. "
            f"{report.summary.failure_count} failed, "
            f"{report.summary.warning_count} warning, "
            f"{report.summary.not_assessed_count} not assessed."
        ),
        metrics=FrozenMap.from_json(_expected_metrics(report)),
        findings=_expected_findings(report),
        flags=(f"qc_{report.decision}",),
        artifacts=result.artifacts,
        provenance=expected_provenance,
    )
    if result != expected:
        _raise_contract_mismatch(
            "Result status, metrics, findings, flags, or provenance were modified"
        )


def _expected_metrics(report: AnnotationQcReport) -> dict[str, object]:
    summary = report.summary
    return {
        "qc_decision": report.decision,
        "policy_version": report.policy_version,
        "profile_id": report.gene_profile_assessment.profile_id,
        "profile_recovery_fraction": summary.profile_recovery_fraction,
        "missing_profile_genes": report.gene_profile_assessment.missing_expected_names,
        "rejected_cds_candidate_count": len(report.rejected_candidates),
        "pass_count": summary.pass_count,
        "warning_count": summary.warning_count,
        "failure_count": summary.failure_count,
        "not_assessed_count": summary.not_assessed_count,
    }


def _expected_findings(report: AnnotationQcReport) -> tuple[Finding, ...]:
    return tuple(
        Finding(
            code=check.finding_code or check.check_id,
            metric=check.check_id,
            value=check.value,
            unit=check.unit,
            evidence_artifact_ids=check.evidence_artifact_ids,
        )
        for check in report.checks
        if check.status != "pass"
    )


def _raise_contract_mismatch(reason: str) -> Never:
    raise OrganelleInputError(
        code="qc.artifact_contract_invalid",
        message="annotation QC Result, report, manifest, and provenance identities disagree",
        details={"reason": reason},
    )


# ---------------------------------------------------------------------------
# Directory and single-file rendering
# ---------------------------------------------------------------------------


def _write_directory(
    report: AnnotationQcReport,
    manifest: AnnotationQcRunManifest,
    directory: Path,
) -> None:
    directory.mkdir()
    tables = _tables(report)
    (directory / "annotation_qc_summary.md").write_text(_markdown(report))
    _copy_canonical(report, directory / "annotation_qc_report.json")
    (directory / "annotation_qc_run_manifest.json").write_bytes(_model_bytes(manifest))
    for name in _TABLE_NAMES:
        _write_csv(tables[name], directory / f"{name}.csv", _TABLE_HEADERS[name])


def _write_single_file(report: AnnotationQcReport, path: Path, suffix: str) -> None:
    if suffix == ".md":
        path.write_text(_markdown(report))
    elif suffix == ".json":
        _copy_canonical(report, path)
    elif suffix == ".csv":
        _write_csv(_tables(report)["summary"], path, _TABLE_HEADERS["summary"])
    else:
        from .xlsx import write_xlsx_tables

        write_xlsx_tables(_tables(report), path)


def _copy_canonical(report: AnnotationQcReport, destination: Path) -> None:
    destination.write_bytes(_model_bytes(report))


def _model_bytes(model: Any) -> bytes:
    return canonical_json_bytes(model.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------


def _tables(report: AnnotationQcReport) -> dict[str, list[dict[str, Any]]]:
    summary = report.summary
    profile = report.gene_profile_assessment
    recovery = summary.profile_recovery_fraction
    return {
        "summary": [
            {
                "decision": report.decision,
                "pass_count": summary.pass_count,
                "warning_count": summary.warning_count,
                "failure_count": summary.failure_count,
                "not_assessed_count": summary.not_assessed_count,
                "record_count": summary.record_count,
                "feature_count": summary.feature_count,
                "unique_pcg_count": summary.unique_pcg_count,
                "unique_trna_count": summary.unique_trna_count,
                "unique_rrna_count": summary.unique_rrna_count,
                "rejected_cds_candidate_count": summary.rejected_cds_candidate_count,
                "partial_feature_count": summary.partial_feature_count,
                "pseudo_feature_count": summary.pseudo_feature_count,
                "translation_exempt_feature_count": summary.translation_exempt_feature_count,
                "duplicate_gene_count": summary.duplicate_gene_count,
                "expected_profile_gene_count": summary.expected_profile_gene_count,
                "recovered_profile_gene_count": summary.recovered_profile_gene_count,
                "duplicated_profile_gene_count": summary.duplicated_profile_gene_count,
                "missing_profile_gene_count": summary.missing_profile_gene_count,
                "profile_recovery_fraction": _display(recovery),
                "assessment_status": _summary_assessment_status(report.decision),
            }
        ],
        "checks": [_check_row(item) for item in report.checks],
        "features": [_feature_row(item) for item in report.feature_summaries],
        "gene_profile": _gene_profile_rows(profile, report.feature_summaries),
        "rejected_cds_candidates": [_rejected_row(item) for item in report.rejected_candidates],
    }


def _check_row(check: QcCheck) -> dict[str, Any]:
    return {
        "check_id": check.check_id,
        "category": check.category,
        "status": check.status,
        "value": check.value,
        "unit": check.unit,
        "message": check.message,
        "finding_code": check.finding_code,
        "evidence_artifact_ids": ";".join(check.evidence_artifact_ids),
        "assessment_status": _check_assessment_status(check.status),
    }


def _feature_row(feature: AnnotationFeatureSummary) -> dict[str, Any]:
    return {
        "record_id": feature.record_id,
        "feature_id": feature.feature_id,
        "feature_type": feature.feature_type,
        "gene_name": feature.gene_name,
        "strand": feature.strand,
        "location_parts": _format_parts(feature.location_parts),
        "spliced_length": feature.spliced_length,
        "is_partial": feature.is_partial,
        "is_pseudo": feature.is_pseudo,
        "is_translation_exempt": feature.is_translation_exempt,
        "is_trans_spliced": feature.is_trans_spliced,
        "parents": ";".join(feature.parents),
        "assessment_status": "assessed",
    }


def _rejected_row(candidate: RejectedCdsCandidate) -> dict[str, Any]:
    return {
        "gene_name": candidate.gene_name,
        "start": _display(candidate.start),
        "end": _display(candidate.end),
        "strand": candidate.strand,
        "parts": _format_parts(candidate.parts),
        "issue_codes": ";".join(candidate.issue_codes),
        "issue_messages": ";".join(candidate.issue_messages),
        "assessment_status": "assessed",
    }


def _gene_profile_rows(
    profile: GeneProfileAssessment,
    features: tuple[AnnotationFeatureSummary, ...],
) -> list[dict[str, Any]]:
    copies = _gene_copy_counts(features)
    rows: list[dict[str, Any]] = []
    for name in profile.expected_pcg_names:
        status = _expected_gene_status(name, profile)
        rows.append(
            {
                "profile_id": profile.profile_id,
                "gene_name": name,
                "class": "expected",
                "status": status,
                "copy_count": copies.get(name.casefold(), 0),
            }
        )
    for name in profile.variable_pcg_names:
        observed = name in profile.observed_variable_names
        rows.append(
            {
                "profile_id": profile.profile_id,
                "gene_name": name,
                "class": "variable",
                "status": "observed" if observed else "absent",
                "copy_count": copies.get(name.casefold(), 0),
            }
        )
    return rows


def _expected_gene_status(name: str, profile: GeneProfileAssessment) -> str:
    if name in profile.duplicated_expected_names:
        return "duplicated"
    if name in profile.recovered_expected_names:
        return "recovered"
    return "missing"


def _gene_copy_counts(
    features: tuple[AnnotationFeatureSummary, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for feature in features:
        if not feature.gene_name:
            continue
        key = feature.gene_name.casefold()
        counts[key] = counts.get(key, 0) + 1
    return counts


def _format_parts(parts: tuple[tuple[Any, ...], ...]) -> str:
    return ";".join("|".join(str(item) for item in part) for part in parts)


def _summary_assessment_status(decision: QcDecision) -> str:
    return "not_assessed" if decision == "insufficient_evidence" else "assessed"


def _check_assessment_status(status: str) -> str:
    return "not_assessed" if status == "not_assessed" else "assessed"


def _display(value: object) -> str:
    return "not assessed" if value is None else str(value)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def _markdown(report: AnnotationQcReport) -> str:
    summary = report.summary
    profile = report.gene_profile_assessment
    recovery = summary.profile_recovery_fraction
    lines = [
        "# Annotation QC summary",
        "",
        f"- Decision: `{report.decision}`",
        f"- Organelle: `{report.organelle}`",
        f"- Backend: `{report.backend}`",
        f"- Policy: `{report.policy_version}`",
        f"- Profile: `{profile.profile_id}`",
        f"- Reference-profile recovery: {summary.recovered_profile_gene_count}/"
        f"{summary.expected_profile_gene_count} expected core genes"
        + (f" ({recovery:.4f})" if recovery is not None else " (not assessed)"),
        f"- Records: {summary.record_count}; features: {summary.feature_count}",
        f"- Unique PCG names: {summary.unique_pcg_count}; "
        f"tRNA: {summary.unique_trna_count}; rRNA: {summary.unique_rrna_count}",
        f"- Rejected CDS candidates: {summary.rejected_cds_candidate_count}",
        f"- Partial features: {summary.partial_feature_count}; "
        f"pseudo features: {summary.pseudo_feature_count}; "
        f"translation-exempt features: {summary.translation_exempt_feature_count}",
        f"- Duplicated gene models: {summary.duplicate_gene_count}",
        f"- Checks: {summary.pass_count} pass, {summary.warning_count} warning, "
        f"{summary.failure_count} fail, {summary.not_assessed_count} not assessed",
        "",
        "## Missing expected core genes",
        "",
    ]
    missing = profile.missing_expected_names
    if missing:
        lines.extend(f"- `{name}`" for name in missing)
    else:
        lines.append("- None.")
    lines.extend(("", "## Checks requiring attention", ""))
    attention = tuple(item for item in report.checks if item.status != "pass")
    if attention:
        lines.extend(f"- `{item.status}` `{item.check_id}`: {item.message}" for item in attention)
    else:
        lines.append("- None.")
    lines.extend(
        (
            "",
            "Reference-profile recovery reports how many expected core "
            "protein-coding genes were recovered. It is not a universal accuracy "
            "metric and does not assert that absent genes are truly lost or that "
            "every prediction is biologically correct.",
            "",
        )
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output Result
# ---------------------------------------------------------------------------


def _write_csv(
    rows: list[dict[str, Any]],
    path: Path,
    fieldnames: tuple[str, ...],
) -> None:
    """Write a deterministic CSV, emitting a header row even when ``rows`` is empty."""

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ""
                        if value is None
                        else str(value).lower()
                        if isinstance(value, bool)
                        else value
                    )
                    for key, value in row.items()
                }
            )


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
        summary_text="Materialized annotation QC output.",
        metrics=source.metrics,
        findings=source.findings,
        flags=tuple(dict.fromkeys((*source.flags, "qc_materialized"))),
        artifacts=artifacts,
        provenance=provenance,
    )


__all__ = ["materialize_annotation_qc_result"]
