"""Strict-contract tests for the annotation-QC models and fixed policy.

These tests prove the annotation-QC models are closed, strict, deterministic, and
do not carry a universal score or completeness field. They are written before the
contracts exist so their RED failure proves the contracts are missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from organelleverse.quality_control.annotation_contracts import (
    AnnotationFeatureSummary,
    AnnotationQcReport,
    AnnotationQcRunManifest,
    AnnotationQcSummary,
    AnnotationSourceRunManifest,
    GeneProfileAssessment,
    RejectedCdsCandidate,
)
from organelleverse.quality_control.annotation_policy import (
    ANNOTATION_QC_POLICY_V1,
    AnnotationQcPolicy,
)
from organelleverse.quality_control.contracts import QcCheck, QcStageOutcome
from tests.quality_control.annotation_helpers import annotation_result_fixture


def test_annotation_qc_summary_is_closed_strict_and_has_no_score() -> None:
    summary = AnnotationQcSummary(
        pass_count=1,
        warning_count=0,
        failure_count=0,
        not_assessed_count=0,
        record_count=1,
        feature_count=3,
        unique_pcg_count=1,
        unique_trna_count=1,
        unique_rrna_count=1,
        rejected_cds_candidate_count=0,
        partial_feature_count=0,
        pseudo_feature_count=0,
        translation_exempt_feature_count=0,
        duplicate_gene_count=0,
        expected_profile_gene_count=24,
        recovered_profile_gene_count=24,
        duplicated_profile_gene_count=0,
        missing_profile_gene_count=0,
        profile_recovery_fraction=1.0,
    )
    assert not hasattr(summary, "score")
    with pytest.raises(ValidationError):
        AnnotationQcSummary.model_validate(
            {**summary.model_dump(mode="json"), "score": 100}
        )
    with pytest.raises(ValidationError):
        AnnotationQcSummary.model_validate(
            {**summary.model_dump(mode="json"), "feature_count": "3"}
        )


def test_annotation_qc_summary_rejects_non_finite_and_out_of_range_fraction() -> None:
    base = AnnotationQcSummary(
        pass_count=0,
        warning_count=0,
        failure_count=0,
        not_assessed_count=0,
        record_count=0,
        feature_count=0,
        unique_pcg_count=0,
        unique_trna_count=0,
        unique_rrna_count=0,
        rejected_cds_candidate_count=0,
        partial_feature_count=0,
        pseudo_feature_count=0,
        translation_exempt_feature_count=0,
        duplicate_gene_count=0,
        expected_profile_gene_count=0,
        recovered_profile_gene_count=0,
        duplicated_profile_gene_count=0,
        missing_profile_gene_count=0,
        profile_recovery_fraction=None,
    )
    for bad in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValidationError):
            AnnotationQcSummary.model_validate(
                {**base.model_dump(mode="json"), "profile_recovery_fraction": bad}
            )


def test_policy_pins_exact_core_and_variable_sets() -> None:
    policy = ANNOTATION_QC_POLICY_V1
    assert policy.policy_version == "organelleverse.annotation-qc-policy.v1"
    assert policy.profile_id == "organelleverse.plant-mito-pcg.v1"
    assert len(policy.expected_pcg_names) == 24
    assert len(policy.variable_pcg_names) == 18
    assert "nad5" in policy.expected_pcg_names
    assert "rps10" in policy.variable_pcg_names
    assert not set(name.casefold() for name in policy.expected_pcg_names) & set(
        name.casefold() for name in policy.variable_pcg_names
    )


def test_policy_profile_digest_is_pinned_and_stable() -> None:
    policy = ANNOTATION_QC_POLICY_V1
    assert policy.profile_sha256 == _expected_profile_sha256(policy)
    assert policy.object_id.startswith("annotation_qc_policy:sha256:")
    again = ANNOTATION_QC_POLICY_V1
    assert again is policy


def _expected_profile_sha256(policy: AnnotationQcPolicy) -> str:
    import hashlib

    payload = {
        "profile_id": policy.profile_id,
        "profile_version": policy.profile_version,
        "scope": policy.scope,
        "expected_pcg_names": list(policy.expected_pcg_names),
        "variable_pcg_names": list(policy.variable_pcg_names),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_source_run_manifest_identity_matches_helper_manifest(tmp_path: Path) -> None:
    result = annotation_result_fixture(tmp_path)
    manifest_artifact = next(
        artifact for artifact in result.artifacts if artifact.kind == "annotation_manifest"
    )
    manifest = AnnotationSourceRunManifest.model_validate_json(
        Path(manifest_artifact.uri).read_bytes()
    )
    raw = json.loads(Path(manifest_artifact.uri).read_text())
    assert manifest.run_manifest_id == raw["run_manifest_id"]
    assert manifest.computed_run_manifest_id == manifest.run_manifest_id
    assert manifest.schema_version == "organelleverse.annotation-run.v1"
    assert manifest.operation_id == "annotation.annotate"
    assert manifest.backend == "mitochondrion"
    annotation_artifact = next(
        artifact for artifact in result.artifacts if artifact.kind == "annotation"
    )
    from organelleverse.annotation.writer import load_document_json

    document = load_document_json(annotation_artifact.uri)
    assert manifest.annotation_object_id == document.object_id


def test_annotation_qc_report_round_trips_through_json() -> None:
    policy = ANNOTATION_QC_POLICY_V1
    summary = AnnotationQcSummary(
        pass_count=1,
        warning_count=0,
        failure_count=0,
        not_assessed_count=0,
        record_count=1,
        feature_count=1,
        unique_pcg_count=1,
        unique_trna_count=0,
        unique_rrna_count=0,
        rejected_cds_candidate_count=0,
        partial_feature_count=0,
        pseudo_feature_count=0,
        translation_exempt_feature_count=0,
        duplicate_gene_count=0,
        expected_profile_gene_count=len(policy.expected_pcg_names),
        recovered_profile_gene_count=1,
        duplicated_profile_gene_count=0,
        missing_profile_gene_count=len(policy.expected_pcg_names) - 1,
        profile_recovery_fraction=1 / len(policy.expected_pcg_names),
    )
    assessment = GeneProfileAssessment(
        profile_id=policy.profile_id,
        profile_version=policy.profile_version,
        profile_sha256=policy.profile_sha256,
        scope=policy.scope,
        expected_pcg_names=policy.expected_pcg_names,
        variable_pcg_names=policy.variable_pcg_names,
        recovered_expected_names=("atp1",),
        missing_expected_names=tuple(
            name for name in policy.expected_pcg_names if name != "atp1"
        ),
        duplicated_expected_names=(),
        observed_variable_names=(),
        absent_variable_names=policy.variable_pcg_names,
        expected_profile_gene_count=len(policy.expected_pcg_names),
        recovered_profile_gene_count=1,
        duplicated_profile_gene_count=0,
        missing_profile_gene_count=len(policy.expected_pcg_names) - 1,
        recovery_fraction=1 / len(policy.expected_pcg_names),
    )
    feature = AnnotationFeatureSummary(
        record_id="MT",
        feature_id="atp1-cds",
        feature_type="cds",
        gene_name="atp1",
        strand=1,
        location_parts=((1, 100, 1),),
        spliced_length=99,
        is_partial=False,
        is_pseudo=False,
        is_translation_exempt=False,
        is_trans_spliced=False,
        parents=(),
    )
    check = QcCheck(
        check_id="annotation.profile_recovery",
        category="biological_review",
        status="warn",
        value=len(assessment.missing_expected_names),
        unit="genes",
        message="Some expected profile genes were not recovered.",
    )
    report = AnnotationQcReport(
        schema_version="organelleverse.annotation-qc.v1",
        policy_version=policy.policy_version,
        source_annotation_result_id="result:sha256:" + "0" * 64,
        source_run_manifest_id="annotation-run:sha256:" + "0" * 64,
        source_annotation_object_id="annotation_document:sha256:" + "0" * 64,
        organelle="mitochondrion",
        backend="mitochondrion",
        decision="needs_review",
        summary=summary,
        checks=(check,),
        feature_summaries=(feature,),
        gene_profile_assessment=assessment,
        rejected_candidates=(),
        evidence_artifacts=(),
    )
    restored = AnnotationQcReport.model_validate_json(report.model_dump_json())
    assert restored == report
    assert restored.object_id == report.object_id
    assert restored.schema_version == "organelleverse.annotation-qc.v1"


def test_annotation_qc_report_rejects_free_form_and_unknown_fields() -> None:
    summary = AnnotationQcSummary(
        pass_count=0,
        warning_count=0,
        failure_count=0,
        not_assessed_count=0,
        record_count=0,
        feature_count=0,
        unique_pcg_count=0,
        unique_trna_count=0,
        unique_rrna_count=0,
        rejected_cds_candidate_count=0,
        partial_feature_count=0,
        pseudo_feature_count=0,
        translation_exempt_feature_count=0,
        duplicate_gene_count=0,
        expected_profile_gene_count=0,
        recovered_profile_gene_count=0,
        duplicated_profile_gene_count=0,
        missing_profile_gene_count=0,
        profile_recovery_fraction=None,
    )
    with pytest.raises(ValidationError):
        AnnotationQcReport(
            schema_version="organelleverse.annotation-qc.v1",
            policy_version=ANNOTATION_QC_POLICY_V1.policy_version,
            source_annotation_result_id="result:sha256:" + "0" * 64,
            source_run_manifest_id="annotation-run:sha256:" + "0" * 64,
            source_annotation_object_id="annotation_document:sha256:" + "0" * 64,
            organelle="mitochondrion",
            backend="mitochondrion",
            decision="ready",
            summary=summary,
            checks=(),
            feature_summaries=(),
            gene_profile_assessment=None,
            rejected_candidates=(),
            evidence_artifacts=(),
            custom_score=0.9,  # type: ignore[call-arg]  # intentional: extra fields are forbidden
        )


def test_annotation_qc_run_manifest_round_trips_with_canonical_identity() -> None:
    manifest = AnnotationQcRunManifest(
        schema_version="organelleverse.annotation-qc-run.v1",
        operation_id="qc.annotation",
        policy_version=ANNOTATION_QC_POLICY_V1.policy_version,
        source_annotation_result_id="result:sha256:" + "0" * 64,
        source_run_manifest_id="annotation-run:sha256:" + "0" * 64,
        source_annotation_object_id="annotation_document:sha256:" + "0" * 64,
        parameters_hash="0" * 64,
        package_version="0.0.1",
        stages=(
            QcStageOutcome(stage="resolve_source", status="ok"),
            QcStageOutcome(stage="aggregate_decision", status="ok"),
        ),
        outputs=(),
    )
    restored = AnnotationQcRunManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest
    assert restored.run_manifest_id == manifest.run_manifest_id
    assert restored.run_manifest_id.startswith("annotation-qc-run:sha256:")


def test_rejected_cds_candidate_preserves_ordered_parts_and_issues() -> None:
    candidate = RejectedCdsCandidate(
        gene_name="cox1",
        start=10,
        end=90,
        strand=1,
        parts=((10, 50, 1), (60, 90, 1)),
        issue_codes=("premature_stop", "missing_start"),
        issue_messages=("premature stop codon", "unsupported start codon"),
    )
    restored = RejectedCdsCandidate.model_validate_json(candidate.model_dump_json())
    assert restored == candidate
    assert restored.parts == ((10, 50, 1), (60, 90, 1))
    with pytest.raises(ValidationError):
        RejectedCdsCandidate(
            gene_name="cox1",
            start=10,
            end=90,
            strand=0,  # type: ignore[arg-type]  # intentional: invalid strand must be rejected
            parts=((10, 50, 1),),
            issue_codes=("premature_stop",),
            issue_messages=("x",),
        )
