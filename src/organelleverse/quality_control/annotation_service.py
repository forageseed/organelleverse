"""Canonical annotation-QC orchestration for Python and Agent callers.

The service resolves and integrity-checks the source annotation evidence,
evaluates it under the fixed policy, verifies export compatibility, and
publishes one content-addressed annotation-QC Result under the managed run
store. Direct Python, Registry, and Agent JSON callers receive the same Result.
Reuse re-parses and re-verifies every cached artifact and identity; a mutated
cache fails closed as :class:`~organelleverse.core.errors.OrganelleInputError`
before the cached Result is trusted.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NoReturn
from uuid import uuid4

from pydantic import BaseModel

from organelleverse.annotation.models import AnnotationDocument
from organelleverse.annotation.writer import materialize_annotation
from organelleverse.assembly.contracts import canonical_json_bytes
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import Finding, OrganelleResult
from organelleverse.runtime import managed_run_path, publish_run

from .annotation_contracts import (
    AnnotationQcReport,
    AnnotationQcRunManifest,
    AnnotationQcSummary,
    ResolvedAnnotationEvidence,
)
from .annotation_evidence import evaluate_annotation, resolve_annotation_evidence
from .annotation_policy import ANNOTATION_QC_POLICY_V1
from .contracts import QcCheck, QcStageOutcome
from .service import aggregate_decision, result_status_for

_OPERATION_ID = "qc.annotation"
_OPERATION_VERSION = "1.0"
_POLICY = ANNOTATION_QC_POLICY_V1
_REPORT_KIND = "annotation_qc_report"
_MANIFEST_KIND = "annotation_qc_run_manifest"

_STAGES = (
    "resolve_source",
    "validate_document",
    "assess_profile",
    "validate_exports",
    "aggregate_decision",
)


def run_annotation_qc(result: OrganelleResult) -> OrganelleResult:
    """Resolve evidence, evaluate it, and persist one content-addressed Result."""

    started_at = datetime.now(UTC)
    evidence = resolve_annotation_evidence(result)
    parameters_hash = _sha256_json({})
    run_digest = _sha256_json(
        {
            "operation_id": _OPERATION_ID,
            "operation_version": _OPERATION_VERSION,
            "policy_id": _POLICY.object_id,
            "profile_id": _POLICY.profile_id,
            "source_result_id": result.object_id,
            "source_run_manifest_id": evidence.source_manifest.run_manifest_id,
            "source_annotation_object_id": evidence.document.object_id,
            "source_artifact_hashes": sorted(artifact.sha256 for artifact in result.artifacts),
            "package_version": _package_version(),
        }
    )
    run_dir = managed_run_path(_OPERATION_ID, f"sha256-{run_digest}")
    if run_dir.exists():
        return _load_reusable_result(
            run_dir,
            source_result=result,
            evidence=evidence,
            parameters_hash=parameters_hash,
            package_version=_package_version(),
            expected_run_digest=run_digest,
        )

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = run_dir.parent / f".{run_dir.name}.tmp-{uuid4().hex}"
    temporary.mkdir()
    try:
        evaluation = evaluate_annotation(evidence, _POLICY)
        export_check = _export_compatibility_check(evidence.document, temporary)
        checks = (*evaluation.checks, export_check)
        counts = _check_counts(checks)
        summary = evaluation.summary.model_copy(
            update={
                "pass_count": counts["pass"],
                "warning_count": counts["warn"],
                "failure_count": counts["fail"],
                "not_assessed_count": counts["not_assessed"],
            }
        )
        decision = aggregate_decision(
            checks,
            required_evidence_missing=any(item.status == "not_assessed" for item in checks),
        )

        report = AnnotationQcReport(
            policy_version=_POLICY.policy_version,
            source_annotation_result_id=result.object_id,
            source_run_manifest_id=evidence.source_manifest.run_manifest_id,
            source_annotation_object_id=evidence.document.object_id,
            organelle="mitochondrion",
            backend="mitochondrion",
            decision=decision,
            summary=summary,
            checks=checks,
            feature_summaries=evaluation.feature_summaries,
            gene_profile_assessment=evaluation.profile_assessment,
            rejected_candidates=evaluation.rejected_candidates,
            evidence_artifacts=(evidence.annotation_artifact, evidence.manifest_artifact),
        )
        report_path = temporary / "annotation_qc_report.json"
        report_path.write_bytes(_model_bytes(report))
        report_artifact = _relocated_artifact(
            report_path,
            run_dir / report_path.name,
            kind=_REPORT_KIND,
        )

        package_version = _package_version()
        manifest = AnnotationQcRunManifest(
            policy_version=_POLICY.policy_version,
            source_annotation_result_id=result.object_id,
            source_run_manifest_id=evidence.source_manifest.run_manifest_id,
            source_annotation_object_id=evidence.document.object_id,
            parameters_hash=parameters_hash,
            package_version=package_version,
            stages=_manifest_stages(export_check, report_artifact),
            outputs=(report_artifact,),
        )
        manifest_path = temporary / "annotation_qc_run_manifest.json"
        manifest_path.write_bytes(_model_bytes(manifest))
        manifest_artifact = _relocated_artifact(
            manifest_path,
            run_dir / manifest_path.name,
            kind=_MANIFEST_KIND,
        )

        finished_at = datetime.now(UTC)
        qc_result = OrganelleResult(
            operation_id=_OPERATION_ID,
            operation_version=_OPERATION_VERSION,
            scope="mitochondrion",
            status=result_status_for(decision),
            summary_text=_summary_text(decision, summary),
            metrics=FrozenMap.from_json(_metrics_from_report(report)),
            findings=_findings(checks),
            flags=(f"qc_{decision}",),
            artifacts=(report_artifact, manifest_artifact),
            provenance=_provenance(
                source_result=result,
                source_run_manifest_id=evidence.source_manifest.run_manifest_id,
                parameters_hash=parameters_hash,
                run_manifest_id=manifest.run_manifest_id,
                package_version=package_version,
                started_at=started_at,
                finished_at=finished_at,
            ),
        )
        (temporary / "result.json").write_bytes(_model_bytes(qc_result))
        _validate_bundle(qc_result, report, manifest)
        publish_run(temporary, run_dir)
        return qc_result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Export compatibility
# ---------------------------------------------------------------------------


def _export_compatibility_check(document: AnnotationDocument, temporary: Path) -> QcCheck:
    """Round-trip the annotation through JSON, GenBank, and GFF3 in a scratch dir.

    Uses the existing public annotation serializer so the check exercises the
    same writer an ``annotation.write`` call would. The scratch directory is
    removed regardless of outcome so it never reaches the managed run store.
    """

    export_dir = temporary / "export-check"
    try:
        materialize_annotation(document, export_dir)
    except Exception as error:
        return QcCheck(
            check_id="annotation.export_compatibility",
            category="export_contract",
            status="fail",
            value=False,
            message=f"Annotation export round-trip failed: {error}",
        )
    finally:
        shutil.rmtree(export_dir, ignore_errors=True)
    return QcCheck(
        check_id="annotation.export_compatibility",
        category="export_contract",
        status="pass",
        value=True,
        message="Canonical JSON, GenBank, and GFF3 outputs round-trip without feature loss.",
    )


# ---------------------------------------------------------------------------
# Result fragment builders
# ---------------------------------------------------------------------------


def _metrics_from_report(report: AnnotationQcReport) -> dict[str, object]:
    summary = report.summary
    return {
        "qc_decision": report.decision,
        "policy_version": _POLICY.policy_version,
        "profile_id": _POLICY.profile_id,
        "profile_recovery_fraction": summary.profile_recovery_fraction,
        "missing_profile_genes": report.gene_profile_assessment.missing_expected_names,
        "rejected_cds_candidate_count": len(report.rejected_candidates),
        "pass_count": summary.pass_count,
        "warning_count": summary.warning_count,
        "failure_count": summary.failure_count,
        "not_assessed_count": summary.not_assessed_count,
    }


def _findings(checks: tuple[QcCheck, ...]) -> tuple[Finding, ...]:
    return tuple(
        Finding(
            code=check.finding_code or check.check_id,
            metric=check.check_id,
            value=check.value,
            unit=check.unit,
            evidence_artifact_ids=check.evidence_artifact_ids,
        )
        for check in checks
        if check.status != "pass"
    )


def _summary_text(decision: str, summary: AnnotationQcSummary) -> str:
    return (
        f"Annotation QC decision: {decision}. "
        f"{summary.failure_count} failed, {summary.warning_count} warning, "
        f"{summary.not_assessed_count} not assessed."
    )


def _provenance(
    *,
    source_result: OrganelleResult,
    source_run_manifest_id: str,
    parameters_hash: str,
    run_manifest_id: str,
    package_version: str,
    started_at: datetime,
    finished_at: datetime,
) -> ResultProvenance:
    return ResultProvenance(
        operation_id=_OPERATION_ID,
        operation_version=_OPERATION_VERSION,
        package_version=package_version,
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        input_object_ids=(source_result.object_id,),
        input_artifact_hashes=tuple(artifact.sha256 for artifact in source_result.artifacts),
        parameters_hash=parameters_hash,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=max(0.0, (finished_at - started_at).total_seconds()),
        run_manifest_id=run_manifest_id,
        upstream_run_manifest_ids=(source_run_manifest_id,),
    )


# ---------------------------------------------------------------------------
# Reuse verification
# ---------------------------------------------------------------------------


def _load_reusable_result(
    run_dir: Path,
    *,
    source_result: OrganelleResult,
    evidence: ResolvedAnnotationEvidence,
    parameters_hash: str,
    package_version: str,
    expected_run_digest: str,
) -> OrganelleResult:
    """Re-parse and re-verify a cached run before trusting it.

    Every artifact digest, the Result↔manifest↔report identity chain, and the
    decision/summary metrics are recomputed. Any drift raises
    ``qc.annotation_cache_conflict`` with the managed run path and the failed
    field so the caller knows exactly which identity broke.
    """

    if run_dir.name != f"sha256-{expected_run_digest}":
        _reuse_conflict(run_dir, "managed run identity does not match inputs", "run_digest")

    result_path = run_dir / "result.json"
    if not result_path.is_file():
        _reuse_conflict(run_dir, "result.json is missing", "result_json")
    try:
        result = OrganelleResult.model_validate_json(result_path.read_bytes())
    except Exception as error:
        _reuse_conflict(run_dir, f"result.json is invalid: {error}", "result_json")

    report_artifact, manifest_artifact = _require_exact_artifacts(result, run_dir)
    report_path = _verify_reusable_artifact(
        report_artifact,
        run_dir,
        expected_name="annotation_qc_report.json",
    )
    manifest_path = _verify_reusable_artifact(
        manifest_artifact,
        run_dir,
        expected_name="annotation_qc_run_manifest.json",
    )

    try:
        report = AnnotationQcReport.model_validate_json(report_path.read_bytes())
        manifest = AnnotationQcRunManifest.model_validate_json(manifest_path.read_bytes())
    except Exception as error:
        _reuse_conflict(
            run_dir, f"cached report or manifest is invalid: {error}", "report_manifest_json"
        )

    evaluation = evaluate_annotation(evidence, _POLICY)
    scratch = run_dir.parent / f".{run_dir.name}.reuse-check-{uuid4().hex}"
    scratch.mkdir()
    try:
        export_check = _export_compatibility_check(evidence.document, scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    checks = (*evaluation.checks, export_check)
    counts = _check_counts(checks)
    summary = evaluation.summary.model_copy(
        update={
            "pass_count": counts["pass"],
            "warning_count": counts["warn"],
            "failure_count": counts["fail"],
            "not_assessed_count": counts["not_assessed"],
        }
    )
    decision = aggregate_decision(
        checks,
        required_evidence_missing=any(item.status == "not_assessed" for item in checks),
    )
    expected_report = AnnotationQcReport(
        policy_version=_POLICY.policy_version,
        source_annotation_result_id=source_result.object_id,
        source_run_manifest_id=evidence.source_manifest.run_manifest_id,
        source_annotation_object_id=evidence.document.object_id,
        organelle="mitochondrion",
        backend="mitochondrion",
        decision=decision,
        summary=summary,
        checks=checks,
        feature_summaries=evaluation.feature_summaries,
        gene_profile_assessment=evaluation.profile_assessment,
        rejected_candidates=evaluation.rejected_candidates,
        evidence_artifacts=(evidence.annotation_artifact, evidence.manifest_artifact),
    )
    if report != expected_report:
        _reuse_conflict(
            run_dir,
            "cached report does not match recomputed annotation evidence",
            "report",
        )

    expected_manifest = AnnotationQcRunManifest(
        policy_version=_POLICY.policy_version,
        source_annotation_result_id=source_result.object_id,
        source_run_manifest_id=evidence.source_manifest.run_manifest_id,
        source_annotation_object_id=evidence.document.object_id,
        parameters_hash=parameters_hash,
        package_version=package_version,
        stages=_manifest_stages(export_check, report_artifact),
        outputs=(report_artifact,),
    )
    if manifest != expected_manifest:
        _reuse_conflict(
            run_dir,
            "cached run manifest does not match recomputed identities",
            "run_manifest",
        )

    provenance = result.provenance
    if provenance is None:
        _reuse_conflict(run_dir, "result provenance is missing", "provenance")
    if provenance.started_at is None or provenance.finished_at is None:
        _reuse_conflict(run_dir, "result provenance timestamps are missing", "provenance_time")
    expected_provenance = ResultProvenance(
        operation_id=_OPERATION_ID,
        operation_version=_OPERATION_VERSION,
        package_version=package_version,
        git_commit=provenance.git_commit,
        input_object_ids=(source_result.object_id,),
        input_artifact_hashes=tuple(item.sha256 for item in source_result.artifacts),
        parameters_hash=parameters_hash,
        started_at=provenance.started_at,
        finished_at=provenance.finished_at,
        duration_seconds=max(
            0.0,
            (provenance.finished_at - provenance.started_at).total_seconds(),
        ),
        run_manifest_id=manifest.run_manifest_id,
        upstream_run_manifest_ids=(evidence.source_manifest.run_manifest_id,),
    )
    if provenance != expected_provenance:
        _reuse_conflict(
            run_dir,
            "cached provenance does not match recomputed identities",
            "provenance",
        )

    expected_result = OrganelleResult(
        operation_id=_OPERATION_ID,
        operation_version=_OPERATION_VERSION,
        scope="mitochondrion",
        status=result_status_for(decision),
        summary_text=_summary_text(decision, summary),
        metrics=FrozenMap.from_json(_metrics_from_report(report)),
        findings=_findings(checks),
        flags=(f"qc_{decision}",),
        artifacts=(report_artifact, manifest_artifact),
        provenance=provenance,
    )
    if result != expected_result:
        _reuse_conflict(
            run_dir,
            "cached Result does not match the verified report and manifest",
            "result",
        )
    return result


def _require_exact_artifacts(
    result: OrganelleResult,
    run_dir: Path,
) -> tuple[ArtifactRef, ArtifactRef]:
    if len(result.artifacts) != 2:
        _reuse_conflict(
            run_dir,
            "cached Result must contain exactly two artifacts",
            "artifacts",
        )
    by_kind = {item.kind: item for item in result.artifacts}
    if set(by_kind) != {_REPORT_KIND, _MANIFEST_KIND}:
        _reuse_conflict(
            run_dir,
            "cached Result artifact kinds do not match the release contract",
            "artifacts",
        )
    return by_kind[_REPORT_KIND], by_kind[_MANIFEST_KIND]


def _verify_reusable_artifact(
    artifact: ArtifactRef,
    run_dir: Path,
    *,
    expected_name: str,
) -> Path:
    candidate = Path(artifact.uri)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    expected = run_dir / expected_name
    if candidate.resolve(strict=False) != expected.resolve(strict=False):
        _reuse_conflict(
            run_dir,
            f"artifact URI must resolve to {expected_name!r} inside the managed run",
            f"{artifact.kind}.uri",
        )
    if not candidate.is_file():
        _reuse_conflict(run_dir, f"artifact is missing: {candidate.name}", artifact.kind)
    current = ArtifactRef.from_path(
        candidate,
        kind=artifact.kind,
        format=artifact.format,
        media_type=artifact.media_type,
    )
    if current.sha256 != artifact.sha256 or current.size_bytes != artifact.size_bytes:
        _reuse_conflict(run_dir, f"artifact digest mismatch: {candidate.name}", artifact.kind)
    return candidate


def _reuse_conflict(run_dir: Path, reason: str, field: str) -> NoReturn:
    raise OrganelleInputError(
        code="qc.annotation_cache_conflict",
        message="existing content-addressed annotation QC run failed verification",
        details={"path": str(run_dir), "reason": reason, "field": field},
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _check_counts(checks: tuple[QcCheck, ...]) -> dict[str, int]:
    return {
        status: sum(item.status == status for item in checks)
        for status in ("pass", "warn", "fail", "not_assessed")
    }


def _manifest_stages(
    export_check: QcCheck,
    report_artifact: ArtifactRef,
) -> tuple[QcStageOutcome, ...]:
    return tuple(
        QcStageOutcome(
            stage=stage,
            status="failed"
            if stage == "validate_exports" and export_check.status == "fail"
            else "ok",
            output_artifact_ids=(report_artifact.object_id,) if stage == "validate_exports" else (),
        )
        for stage in _STAGES
    )


def _relocated_artifact(current_path: Path, final_path: Path, *, kind: str) -> ArtifactRef:
    artifact = ArtifactRef.from_path(
        current_path,
        kind=kind,
        format="json",
        media_type="application/json",
    )
    return artifact.model_copy(update={"uri": str(final_path)})


def _validate_bundle(
    result: OrganelleResult,
    report: AnnotationQcReport,
    manifest: AnnotationQcRunManifest,
) -> None:
    if result.provenance is None or result.provenance.run_manifest_id != manifest.run_manifest_id:
        raise RuntimeError("annotation QC Result and run manifest identities disagree")
    if report.source_annotation_result_id != manifest.source_annotation_result_id:
        raise RuntimeError("annotation QC report and run manifest source identities disagree")
    if AnnotationQcReport.model_validate_json(_model_bytes(report)) != report:
        raise RuntimeError("annotation QC report failed canonical round-trip validation")
    if AnnotationQcRunManifest.model_validate_json(_model_bytes(manifest)) != manifest:
        raise RuntimeError("annotation QC run manifest failed canonical round-trip validation")
    if OrganelleResult.model_validate_json(_model_bytes(result)) != result:
        raise RuntimeError("annotation QC Result failed canonical round-trip validation")


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


__all__ = ["run_annotation_qc"]
