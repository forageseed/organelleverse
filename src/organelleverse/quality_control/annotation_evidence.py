"""Annotation-QC source trust boundary and deterministic evaluation.

The resolver validates the full Result↔manifest↔document identity chain before
any managed output is created; identity or digest disagreement raises a typed
:class:`~organelleverse.core.errors.OrganelleInputError` and is never converted
into a scientific ``needs_review`` result. The evaluator summarizes every
feature deterministically, assesses the fixed plant mitochondrial gene profile,
and emits stable checks without ever computing a universal score or completeness
percentage.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ValidationError

from organelleverse.annotation.models import AnnotationFeature
from organelleverse.annotation.validation import validate_document
from organelleverse.annotation.writer import load_document_json
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.frozen import FrozenJson, FrozenMap, thaw_json
from organelleverse.core.result import OrganelleResult
from organelleverse.quality_control.contracts import QcCheck, QcCheckStatus, QcCheckValue

from .annotation_contracts import (
    AnnotationEvaluation,
    AnnotationFeatureSummary,
    AnnotationQcSummary,
    AnnotationSourceRunManifest,
    GeneProfileAssessment,
    LocationPartRecord,
    RejectedCdsCandidate,
    RejectedPartRecord,
    ResolvedAnnotationEvidence,
)
from .annotation_policy import AnnotationQcPolicy

_ANNOTATION_KIND = "annotation"
_MANIFEST_KIND = "annotation_manifest"
_EXCEPTION_VALUES = {"rna editing", "ribosomal slippage", "trans-splicing"}


# ---------------------------------------------------------------------------
# Input error helper
# ---------------------------------------------------------------------------


def _input_error(code: str, message: str, details: Mapping[str, Any] | None = None) -> OrganelleInputError:
    return OrganelleInputError(code=code, message=message, details=details or {})


# ---------------------------------------------------------------------------
# Typed JSON narrowing helpers for the source trust boundary
# ---------------------------------------------------------------------------


def _require_field(container: Mapping[str, FrozenJson], field: str) -> FrozenJson:
    """Return a required trust-boundary field, failing closed if absent."""

    if field not in container:
        raise _input_error(
            "qc.annotation_metadata_mismatch",
            f"annotation trust-boundary field {field!r} is missing",
            {"field": field},
        )
    return container[field]


def _require_json_array(value: FrozenJson, *, field: str) -> tuple[FrozenJson, ...]:
    """Narrow a frozen JSON value to a sequence, failing closed otherwise."""

    if not isinstance(value, tuple):
        raise _input_error(
            "qc.annotation_metadata_mismatch",
            f"annotation trust-boundary field {field!r} must be a JSON array",
            {"field": field, "value_type": type(value).__name__},
        )
    return value


def _require_str_value(value: FrozenJson, *, field: str) -> str:
    if not isinstance(value, str):
        raise _input_error(
            "qc.annotation_metadata_mismatch",
            f"annotation trust-boundary field {field!r} must be a string",
            {"field": field, "value_type": type(value).__name__},
        )
    return value


def _require_int_value(value: FrozenJson, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _input_error(
            "qc.annotation_metadata_mismatch",
            f"annotation trust-boundary field {field!r} must be an integer",
            {"field": field, "value_type": type(value).__name__},
        )
    return value


def _require_optional_int(value: FrozenJson | None, *, field: str) -> int | None:
    if value is None:
        return None
    return _require_int_value(value, field=field)


def _require_strand_value(value: FrozenJson, *, field: str) -> Literal[-1, 1]:
    if value != -1 and value != 1:
        raise _input_error(
            "qc.annotation_metadata_mismatch",
            f"annotation trust-boundary field {field!r} strand must be -1 or 1",
            {"field": field, "value_type": type(value).__name__},
        )
    return cast(Literal[-1, 1], value)


def _require_str_sequence_field(
    container: Mapping[str, FrozenJson], field: str
) -> tuple[str, ...]:
    """Narrow a required metrics field to an ordered tuple of strings."""

    array = _require_json_array(_require_field(container, field), field=field)
    return tuple(_require_str_value(item, field=field) for item in array)


def _require_str_array(value: FrozenJson, *, field: str) -> tuple[str, ...]:
    return tuple(
        _require_str_value(item, field=field) for item in _require_json_array(value, field=field)
    )


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def _require_unique_artifact(result: OrganelleResult, kind: str) -> ArtifactRef:
    matches = tuple(artifact for artifact in result.artifacts if artifact.kind == kind)
    if len(matches) == 0:
        raise _input_error(
            "qc.annotation_artifact_missing",
            f"annotation result is missing a {kind} artifact",
            {"artifact_kind": kind},
        )
    if len(matches) > 1:
        raise _input_error(
            "qc.annotation_input_contract_violation",
            f"annotation result has multiple {kind} artifacts",
            {"artifact_kind": kind, "count": len(matches)},
        )
    return matches[0]


def _reverify_artifact(artifact: ArtifactRef) -> Path:
    """Recompute the artifact digest and return its resolved path."""

    path = Path(artifact.uri)
    if not path.is_file():
        raise _input_error(
            "qc.annotation_artifact_missing",
            f"artifact file is missing: {path}",
            {"uri": artifact.uri, "artifact_kind": artifact.kind},
        )
    actual = ArtifactRef.from_path(
        path,
        kind=artifact.kind,
        format=artifact.format,
        media_type=artifact.media_type,
    )
    if actual.sha256 != artifact.sha256 or actual.size_bytes != artifact.size_bytes:
        raise _input_error(
            "qc.annotation_artifact_digest_mismatch",
            f"artifact digest mismatch: {path}",
            {
                "uri": artifact.uri,
                "artifact_kind": artifact.kind,
                "expected_sha256": artifact.sha256,
                "actual_sha256": actual.sha256,
            },
        )
    return path


def _normalize_rejected_candidate(raw: FrozenJson) -> RejectedCdsCandidate:
    """Parse one source rejected-candidate record into a strict contract.

    Every nested field is narrowed with a typed helper so malformed candidates
    fail closed as ``OrganelleInputError`` instead of leaking ``KeyError`` or
    ``TypeError`` across the trust boundary.
    """

    if not isinstance(raw, FrozenMap):
        raise _input_error(
            "qc.annotation_metadata_mismatch",
            "rejected CDS candidate is not a JSON object",
        )

    parts_value = _require_json_array(
        _require_field(raw, "parts"),
        field="rejected_cds_candidate.parts",
    )
    parts: list[RejectedPartRecord] = []
    for index, part in enumerate(parts_value):
        if not isinstance(part, FrozenMap):
            raise _input_error(
                "qc.annotation_metadata_mismatch",
                f"rejected CDS candidate part {index} is not a JSON object",
            )
        parts.append(
            (
                _require_int_value(
                    _require_field(part, "start"),
                    field=f"rejected_cds_candidate.parts[{index}].start",
                ),
                _require_int_value(
                    _require_field(part, "end"),
                    field=f"rejected_cds_candidate.parts[{index}].end",
                ),
                _require_strand_value(
                    _require_field(part, "strand"),
                    field=f"rejected_cds_candidate.parts[{index}].strand",
                ),
            )
        )

    try:
        return RejectedCdsCandidate(
            gene_name=_require_str_value(
                _require_field(raw, "gene_name"),
                field="rejected_cds_candidate.gene_name",
            ),
            start=_require_optional_int(
                raw.get("start"), field="rejected_cds_candidate.start"
            ),
            end=_require_optional_int(
                raw.get("end"), field="rejected_cds_candidate.end"
            ),
            strand=_require_strand_value(
                _require_field(raw, "strand"),
                field="rejected_cds_candidate.strand",
            ),
            parts=tuple(parts),
            issue_codes=_require_str_array(
                _require_field(raw, "issue_codes"),
                field="rejected_cds_candidate.issue_codes",
            ),
            issue_messages=_require_str_array(
                _require_field(raw, "issue_messages"),
                field="rejected_cds_candidate.issue_messages",
            ),
        )
    except (ValidationError, ValueError) as error:
        raise _input_error(
            "qc.annotation_metadata_mismatch",
            "rejected CDS candidate could not be normalized into a strict record",
            {"reason": str(error)},
        ) from error


def _require_equal_metrics_and_metadata(
    metrics_value: FrozenJson,
    metadata_value: FrozenJson,
    *,
    field: str,
) -> None:
    if thaw_json(metrics_value) != thaw_json(metadata_value):
        raise _input_error(
            "qc.annotation_metadata_mismatch",
            f"result metrics and annotation document metadata disagree for {field}",
            {"field": field},
        )


def resolve_annotation_evidence(result: OrganelleResult) -> ResolvedAnnotationEvidence:
    """Resolve and integrity-check the annotation evidence for one QC run.

    Enforces ``operation_id == "annotation.annotate"``, a non-failed status, the
    mitochondrial scope, exactly one ``annotation`` and one ``annotation_manifest``
    artifact, recomputed artifact digests, a strict source manifest whose semantic
    identity matches its recorded run identity, and a complete
    Result↔manifest↔document identity chain including requested stages, feature
    counts, rejected CDS candidates, and missing-core metadata. Any violation
    raises a typed :class:`OrganelleInputError` before managed output is created.
    """

    if result.operation_id != "annotation.annotate":
        raise _input_error(
            "qc.annotation_input_contract_violation",
            (
                "annotation QC consumes only annotation.annotate results; "
                f"got {result.operation_id!r}"
            ),
            {"operation_id": result.operation_id},
        )
    if result.status == "failed":
        raise _input_error(
            "qc.annotation_input_contract_violation",
            "annotation QC cannot consume a failed annotation result",
            {"status": result.status},
        )

    annotation_artifact = _require_unique_artifact(result, _ANNOTATION_KIND)
    manifest_artifact = _require_unique_artifact(result, _MANIFEST_KIND)

    annotation_path = _reverify_artifact(annotation_artifact)
    manifest_path = _reverify_artifact(manifest_artifact)

    try:
        document = load_document_json(annotation_path)
    except (ValidationError, ValueError) as error:
        raise _input_error(
            "qc.annotation_artifact_digest_mismatch",
            "canonical annotation artifact could not be parsed",
            {"reason": str(error)},
        ) from error

    try:
        manifest = AnnotationSourceRunManifest.model_validate_json(manifest_path.read_bytes())
    except (ValidationError, ValueError) as error:
        raise _input_error(
            "qc.annotation_manifest_invalid",
            "annotation run manifest is malformed",
            {"reason": str(error)},
        ) from error

    if manifest.computed_run_manifest_id != manifest.run_manifest_id:
        raise _input_error(
            "qc.annotation_manifest_identity_mismatch",
            "annotation run manifest identity does not match its semantic payload",
            {
                "recorded_run_manifest_id": manifest.run_manifest_id,
                "computed_run_manifest_id": manifest.computed_run_manifest_id,
            },
        )

    provenance = result.provenance
    if provenance is None:
        raise _input_error(
            "qc.annotation_input_contract_violation",
            "annotation result has no provenance",
        )

    def _identity_mismatch(field: str, expected: object, actual: object) -> OrganelleInputError:
        return _input_error(
            "qc.annotation_input_contract_violation",
            f"annotation identity check failed: {field}",
            {"field": field, "expected": expected, "actual": actual},
        )

    if provenance.run_manifest_id != manifest.run_manifest_id:
        raise _identity_mismatch(
            "run_manifest_id", manifest.run_manifest_id, provenance.run_manifest_id
        )
    if provenance.parameters_hash != manifest.parameters_hash:
        raise _identity_mismatch(
            "parameters_hash", manifest.parameters_hash, provenance.parameters_hash
        )
    if manifest.input_object_id not in provenance.input_object_ids:
        raise _identity_mismatch(
            "input_object_id",
            manifest.input_object_id,
            list(provenance.input_object_ids),
        )
    if manifest.input_artifact_hash not in provenance.input_artifact_hashes:
        raise _identity_mismatch(
            "input_artifact_hash",
            manifest.input_artifact_hash,
            list(provenance.input_artifact_hashes),
        )
    if manifest.annotation_object_id != document.object_id:
        raise _identity_mismatch(
            "annotation_object_id", document.object_id, manifest.annotation_object_id
        )
    if (
        provenance.actual_backend != manifest.backend
        or manifest.backend != document.backend
    ):
        raise _identity_mismatch(
            "backend",
            manifest.backend,
            (provenance.actual_backend, document.backend),
        )
    if result.scope != "mitochondrion":
        raise _input_error(
            "qc.annotation_input_contract_violation",
            "annotation QC v1 supports only mitochondrial annotation results",
            {"scope": result.scope},
        )

    metrics = result.metrics
    requested_stages_metric = _require_str_sequence_field(metrics, "requested_stages")
    if requested_stages_metric != document.requested_stages:
        raise _identity_mismatch(
            "requested_stages",
            document.requested_stages,
            requested_stages_metric,
        )
    completed_stages_metric = _require_str_sequence_field(metrics, "completed_stages")
    if completed_stages_metric != document.completed_stages:
        raise _identity_mismatch(
            "completed_stages",
            document.completed_stages,
            completed_stages_metric,
        )

    recomputed_counts = validate_document(document, document.requested_stages).feature_counts
    _require_equal_metrics_and_metadata(
        _require_field(metrics, "feature_counts"),
        recomputed_counts,
        field="feature_counts",
    )

    metadata = document.source_metadata
    metrics_rejected = _require_field(metrics, "rejected_cds_candidates")
    metadata_rejected = _require_field(metadata, "rejected_cds_candidates")
    _require_equal_metrics_and_metadata(
        metrics_rejected, metadata_rejected, field="rejected_cds_candidates"
    )
    # Normalize into strict records so malformed source candidates fail closed.
    for raw in _require_json_array(metadata_rejected, field="rejected_cds_candidates"):
        _normalize_rejected_candidate(raw)

    _require_equal_metrics_and_metadata(
        _require_field(metrics, "missing_core_genes"),
        _require_field(metadata, "missing_core_genes"),
        field="missing_core_genes",
    )

    return ResolvedAnnotationEvidence(
        source_result=result,
        source_manifest=manifest,
        document=document,
        annotation_artifact=annotation_artifact,
        manifest_artifact=manifest_artifact,
        annotation_path=annotation_path,
        manifest_path=manifest_path,
    )


# ---------------------------------------------------------------------------
# Evaluator helpers
# ---------------------------------------------------------------------------


def _gene_name(feature: AnnotationFeature) -> str:
    values = feature.qualifier_values("gene")
    return values[0].strip() if values else ""


def _is_partial(feature: AnnotationFeature) -> bool:
    return any(qualifier.name.casefold() == "partial" for qualifier in feature.qualifiers)


def _is_pseudo(feature: AnnotationFeature) -> bool:
    return any(
        qualifier.name.casefold() in {"pseudo", "pseudogene"} for qualifier in feature.qualifiers
    )


def _has_translation_exception(feature: AnnotationFeature) -> bool:
    for qualifier in feature.qualifiers:
        name = qualifier.name.casefold()
        if name == "transl_except" and qualifier.values:
            return True
        if name == "exception" and any(
            value.strip().casefold() in _EXCEPTION_VALUES for value in qualifier.values
        ):
            return True
    return False


def _is_trans_spliced(feature: AnnotationFeature) -> bool:
    return any(
        qualifier.name.casefold() == "exception"
        and any(value.strip().casefold() == "trans-splicing" for value in qualifier.values)
        for qualifier in feature.qualifiers
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def _check(
    check_id: str,
    category: str,
    status: QcCheckStatus,
    *,
    value: QcCheckValue,
    unit: str,
    message: str,
) -> QcCheck:
    return QcCheck(
        check_id=check_id,
        category=category,
        status=status,
        value=value,
        unit=unit,
        message=message,
    )


def evaluate_annotation(
    evidence: ResolvedAnnotationEvidence,
    policy: AnnotationQcPolicy,
) -> AnnotationEvaluation:
    """Evaluate canonical structure, CDS exceptions, and biological review signals.

    Calls :func:`validate_document`, summarizes every feature deterministically,
    normalizes gene names case-insensitively to the policy spelling, assesses the
    fixed profile, and emits stable checks. Validation errors are ``fail``; every
    explicit translation exception, missing expected name, duplicate, partial or
    pseudo feature, or rejected CDS candidate is ``warn``; descriptive RNA
    observations are ``pass``. Variable genes never produce a review signal when
    absent.
    """

    document = evidence.document
    validation = validate_document(document, document.requested_stages)

    expected_by_casefold = {name.casefold(): name for name in policy.expected_pcg_names}
    variable_by_casefold = {name.casefold(): name for name in policy.variable_pcg_names}

    feature_summaries: list[AnnotationFeatureSummary] = []
    cds_counts: Counter[str] = Counter()
    observed_expected: set[str] = set()
    observed_variable_keys: set[str] = set()
    unique_pcg: set[str] = set()
    unique_trna: set[str] = set()
    unique_rrna: set[str] = set()
    partial_count = 0
    pseudo_count = 0
    translation_exempt_count = 0
    has_translation_exception = False
    record_count = len(document.records)
    feature_count = 0

    for record in document.records:
        for feature in record.features:
            feature_count += 1
            ftype = feature.type.casefold()
            gene = _gene_name(feature)
            location_parts: list[LocationPartRecord] = []
            for part in feature.parts:
                location_parts.append((part.start, part.end, part.strand))
            parts = tuple(location_parts)
            spliced_length = sum(part.end - part.start for part in feature.parts)
            is_partial = _is_partial(feature)
            is_pseudo = _is_pseudo(feature)
            is_trans_spliced = _is_trans_spliced(feature)
            has_exception = _has_translation_exception(feature)
            is_translation_exempt = is_pseudo or has_exception

            if is_partial:
                partial_count += 1
            if is_pseudo:
                pseudo_count += 1
            if ftype == "cds":
                if has_exception:
                    has_translation_exception = True
                if is_translation_exempt:
                    translation_exempt_count += 1
                key = gene.casefold()
                if key:
                    if key in expected_by_casefold:
                        observed_expected.add(key)
                        unique_pcg.add(key)
                        cds_counts[key] += 1
                    elif key in variable_by_casefold:
                        observed_variable_keys.add(key)
                        unique_pcg.add(key)
                        cds_counts[key] += 1
            elif ftype == "trna" and gene:
                unique_trna.add(gene)
            elif ftype == "rrna" and gene:
                unique_rrna.add(gene)

            feature_summaries.append(
                AnnotationFeatureSummary(
                    record_id=record.seqid,
                    feature_id=feature.feature_id,
                    feature_type=feature.type,
                    gene_name=gene,
                    strand=feature.parts[0].strand,
                    location_parts=parts,
                    spliced_length=spliced_length,
                    is_partial=is_partial,
                    is_pseudo=is_pseudo,
                    is_translation_exempt=is_translation_exempt,
                    is_trans_spliced=is_trans_spliced,
                    parents=feature.parents,
                )
            )

    recovered_expected = tuple(
        expected_by_casefold[name.casefold()]
        for name in policy.expected_pcg_names
        if name.casefold() in observed_expected
    )
    missing_expected = tuple(
        expected_by_casefold[name.casefold()]
        for name in policy.expected_pcg_names
        if name.casefold() not in observed_expected
    )
    duplicated_expected = tuple(
        expected_by_casefold[name.casefold()]
        for name in policy.expected_pcg_names
        if cds_counts[name.casefold()] > 1
    )
    observed_variable_names = tuple(
        variable_by_casefold[name.casefold()]
        for name in policy.variable_pcg_names
        if name.casefold() in observed_variable_keys
    )
    absent_variable_names = tuple(
        variable_by_casefold[name.casefold()]
        for name in policy.variable_pcg_names
        if name.casefold() not in observed_variable_keys
    )

    expected_count = len(policy.expected_pcg_names)
    recovered_count = len(recovered_expected)
    duplicated_count = len(duplicated_expected)
    missing_count = len(missing_expected)
    recovery_fraction = recovered_count / expected_count if expected_count else None

    assessment = GeneProfileAssessment(
        profile_id=policy.profile_id,
        profile_version=policy.profile_version,
        profile_sha256=policy.profile_sha256,
        scope=policy.scope,
        expected_pcg_names=policy.expected_pcg_names,
        variable_pcg_names=policy.variable_pcg_names,
        recovered_expected_names=recovered_expected,
        missing_expected_names=missing_expected,
        duplicated_expected_names=duplicated_expected,
        observed_variable_names=observed_variable_names,
        absent_variable_names=absent_variable_names,
        expected_profile_gene_count=expected_count,
        recovered_profile_gene_count=recovered_count,
        duplicated_profile_gene_count=duplicated_count,
        missing_profile_gene_count=missing_count,
        recovery_fraction=recovery_fraction,
    )

    rejected_raw = document.source_metadata.get("rejected_cds_candidates")
    if rejected_raw is None:
        rejected_candidates: tuple[RejectedCdsCandidate, ...] = ()
    else:
        rejected_candidates = tuple(
            _normalize_rejected_candidate(raw)
            for raw in _require_json_array(
                rejected_raw, field="rejected_cds_candidates"
            )
        )
    duplicate_gene_count = sum(1 for count in cds_counts.values() if count > 1)

    canonical_status: QcCheckStatus = "fail" if not validation.valid else "pass"
    translation_status: QcCheckStatus = "warn" if has_translation_exception else "pass"
    profile_status: QcCheckStatus = "warn" if (missing_count or duplicated_count) else "pass"
    duplicate_status: QcCheckStatus = "warn" if duplicate_gene_count else "pass"
    partial_status: QcCheckStatus = "warn" if partial_count else "pass"
    pseudo_status: QcCheckStatus = "warn" if pseudo_count else "pass"
    rejected_status: QcCheckStatus = "warn" if rejected_candidates else "pass"

    error_count = len(validation.errors)
    checks = (
        _check(
            "annotation.canonical_structure",
            "structural_contract",
            canonical_status,
            value=error_count,
            unit="errors",
            message=(
                "Canonical structure, coordinates, hierarchy, and requested stages are consistent."
                if canonical_status == "pass"
                else f"{error_count} canonical structure or requested-stage validation error(s)."
            ),
        ),
        _check(
            "annotation.translation_exceptions",
            "cds_process",
            translation_status,
            value=translation_exempt_count,
            unit="features",
            message=(
                "No explicit CDS translation exceptions."
                if translation_status == "pass"
                else f"{translation_exempt_count} CDS feature(s) carry explicit translation exceptions."
            ),
        ),
        _check(
            "annotation.profile_recovery",
            "biological_review",
            profile_status,
            value=missing_count,
            unit="genes",
            message=(
                f"Recovered {recovered_count}/{expected_count} expected profile genes."
                if profile_status == "pass"
                else (
                    f"Missing {missing_count} and duplicated {duplicated_count} expected "
                    f"profile gene(s); recovered {recovered_count}/{expected_count}."
                )
            ),
        ),
        _check(
            "annotation.duplicate_gene_models",
            "biological_review",
            duplicate_status,
            value=duplicate_gene_count,
            unit="genes",
            message=(
                "No duplicate PCG gene models."
                if duplicate_status == "pass"
                else f"{duplicate_gene_count} PCG gene name(s) modeled more than once."
            ),
        ),
        _check(
            "annotation.partial_features",
            "biological_review",
            partial_status,
            value=partial_count,
            unit="features",
            message=(
                "No partial features." if partial_status == "pass"
                else f"{partial_count} partial feature(s) flagged for review."
            ),
        ),
        _check(
            "annotation.pseudo_features",
            "biological_review",
            pseudo_status,
            value=pseudo_count,
            unit="features",
            message=(
                "No pseudo/pseudogene features." if pseudo_status == "pass"
                else f"{pseudo_count} pseudo/pseudogene feature(s) flagged for review."
            ),
        ),
        _check(
            "annotation.rejected_cds_candidates",
            "biological_review",
            rejected_status,
            value=len(rejected_candidates),
            unit="candidates",
            message=(
                "No rejected CDS candidates." if rejected_status == "pass"
                else f"{len(rejected_candidates)} rejected CDS candidate(s) preserved for review."
            ),
        ),
        _check(
            "annotation.rna_observations",
            "biological_review",
            "pass",
            value=f"unique_trna={len(unique_trna)},unique_rrna={len(unique_rrna)}",
            unit="genes",
            message=(
                f"Observed {len(unique_trna)} unique tRNA and {len(unique_rrna)} unique rRNA gene(s)."
            ),
        ),
    )

    statuses = tuple(check.status for check in checks)
    summary = AnnotationQcSummary(
        pass_count=statuses.count("pass"),
        warning_count=statuses.count("warn"),
        failure_count=statuses.count("fail"),
        not_assessed_count=statuses.count("not_assessed"),
        record_count=record_count,
        feature_count=feature_count,
        unique_pcg_count=len(unique_pcg),
        unique_trna_count=len(unique_trna),
        unique_rrna_count=len(unique_rrna),
        rejected_cds_candidate_count=len(rejected_candidates),
        partial_feature_count=partial_count,
        pseudo_feature_count=pseudo_count,
        translation_exempt_feature_count=translation_exempt_count,
        duplicate_gene_count=duplicate_gene_count,
        expected_profile_gene_count=expected_count,
        recovered_profile_gene_count=recovered_count,
        duplicated_profile_gene_count=duplicated_count,
        missing_profile_gene_count=missing_count,
        profile_recovery_fraction=recovery_fraction,
    )

    return AnnotationEvaluation(
        checks=checks,
        feature_summaries=tuple(feature_summaries),
        profile_assessment=assessment,
        rejected_candidates=rejected_candidates,
        summary=summary,
    )


__all__ = [
    "evaluate_annotation",
    "resolve_annotation_evidence",
]
