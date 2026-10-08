"""Trust-boundary resolver and evaluation tests for annotation QC.

The resolver must reject every untrustworthy source before any managed output
is created. The evaluator must report profile recovery and review signals
without ever emitting a universal score or completeness percentage. These tests
are written before the resolver/evaluator exist so their RED failure proves the
functions are missing.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.result import ErrorDetail
from organelleverse.quality_control.annotation_contracts import ResolvedAnnotationEvidence
from organelleverse.quality_control.annotation_evidence import (
    _normalize_rejected_candidate,  # pyright: ignore[reportPrivateUsage]
    evaluate_annotation,
    resolve_annotation_evidence,
)
from organelleverse.quality_control.annotation_policy import (
    ANNOTATION_QC_POLICY_V1,
    EXPECTED_PCG_NAMES,
)
from tests.quality_control.annotation_helpers import annotation_result_fixture

_ANNOTATION_KIND = "annotation"
_MANIFEST_KIND = "annotation_manifest"


# ---------------------------------------------------------------------------
# Test helpers for mutating the frozen Result / on-disk manifest
# ---------------------------------------------------------------------------


def _drop_artifact(result: Any, kind: str) -> Any:
    artifacts = tuple(artifact for artifact in result.artifacts if artifact.kind != kind)
    return result.model_copy(update={"artifacts": artifacts})


def _duplicate_artifact(result: Any, kind: str) -> Any:
    target = next(artifact for artifact in result.artifacts if artifact.kind == kind)
    artifacts = (*result.artifacts, target)
    return result.model_copy(update={"artifacts": artifacts})


def _redocument(result: Any, *, tweak: str = "tampered") -> Any:
    """Rewrite the canonical annotation with a changed (non-counted) field.

    The document identity changes but feature counts do not, so the
    ``annotation_object_id`` cross-check is the first identity check to fail.
    """

    from organelleverse.annotation.writer import load_document_json, write_document_json

    annotation_artifact = next(
        artifact for artifact in result.artifacts if artifact.kind == _ANNOTATION_KIND
    )
    path = Path(annotation_artifact.uri)
    document = load_document_json(path)
    records = tuple(
        record.model_copy(update={"description": f"{record.description} ({tweak})"})
        for record in document.records
    )
    rewritten = document.model_copy(update={"records": records})
    write_document_json(rewritten, path)
    rebuilt = ArtifactRef.from_path(
        path,
        kind=_ANNOTATION_KIND,
        format="json",
        media_type="application/json",
    )
    artifacts = tuple(
        rebuilt if artifact.kind == _ANNOTATION_KIND else artifact
        for artifact in result.artifacts
    )
    return result.model_copy(update={"artifacts": artifacts})


def _remanifest(
    result: Any,
    mutate: Callable[[dict[str, Any]], None],
) -> Any:
    """Rewrite the source manifest on disk and rebuild its artifact reference."""

    manifest_artifact = next(
        artifact for artifact in result.artifacts if artifact.kind == _MANIFEST_KIND
    )
    path = Path(manifest_artifact.uri)
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(
        json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    rebuilt = ArtifactRef.from_path(
        path,
        kind=_MANIFEST_KIND,
        format="json",
        media_type="application/json",
    )
    artifacts = tuple(
        rebuilt if artifact.kind == _MANIFEST_KIND else artifact for artifact in result.artifacts
    )
    return result.model_copy(update={"artifacts": artifacts})


def _metrics_dict(result: Any) -> dict[str, object]:
    """Copy a Result's frozen metrics into a mutable ``dict[str, object]``.

    The value type stays ``object`` so a test can overwrite a field with any
    JSON-compatible shape (e.g. a nested ``dict[str, int]``) before re-freezing.
    """

    return {key: value for key, value in result.metrics.items()}


@pytest.mark.parametrize(
    ("missing_field", "nested"),
    [
        ("parts", False),
        ("gene_name", False),
        ("start", True),
        ("end", True),
        ("strand", True),
    ],
)
def test_rejected_candidate_missing_nested_field_fails_closed(
    missing_field: str,
    nested: bool,
) -> None:
    candidate: dict[str, object] = {
        "gene_name": "cox1",
        "start": 0,
        "end": 36,
        "strand": 1,
        "parts": ({"start": 0, "end": 36, "strand": 1},),
        "issue_codes": ("premature_stop",),
        "issue_messages": ("internal stop",),
    }
    if nested:
        parts = cast(tuple[dict[str, object], ...], candidate["parts"])
        part = dict(parts[0])
        del part[missing_field]
        candidate["parts"] = (part,)
    else:
        del candidate[missing_field]

    with pytest.raises(OrganelleInputError) as caught:
        _normalize_rejected_candidate(FrozenMap(candidate))

    assert caught.value.code == "qc.annotation_metadata_mismatch"


# ---------------------------------------------------------------------------
# Resolver: trust-boundary rejection cases
# ---------------------------------------------------------------------------


def test_resolver_rejects_changed_annotation_before_managed_output(tmp_path: Path) -> None:
    result = annotation_result_fixture(tmp_path)
    annotation = next(item for item in result.artifacts if item.kind == _ANNOTATION_KIND)
    Path(annotation.uri).write_text("{}")
    with pytest.raises(OrganelleInputError) as caught:
        resolve_annotation_evidence(result)
    assert caught.value.code == "qc.annotation_artifact_digest_mismatch"


def test_resolver_accepts_valid_fixture(tmp_path: Path) -> None:
    evidence = resolve_annotation_evidence(annotation_result_fixture(tmp_path))
    assert isinstance(evidence, ResolvedAnnotationEvidence)
    assert evidence.source_manifest.backend == "mitochondrion"
    assert (
        evidence.source_manifest.computed_run_manifest_id
        == evidence.source_manifest.run_manifest_id
    )
    assert evidence.document.backend == "mitochondrion"
    assert evidence.annotation_artifact.kind == _ANNOTATION_KIND
    assert evidence.manifest_artifact.kind == _MANIFEST_KIND
    assert evidence.annotation_path.is_file()
    assert evidence.manifest_path.is_file()


def test_resolver_rejects_wrong_operation(tmp_path: Path) -> None:
    result = annotation_result_fixture(tmp_path).model_copy(update={"operation_id": "assembly.assemble"})
    with pytest.raises(OrganelleInputError):
        resolve_annotation_evidence(result)


def test_resolver_rejects_failed_status(tmp_path: Path) -> None:
    result = annotation_result_fixture(tmp_path).model_copy(
        update={
            "status": "failed",
            "errors": (ErrorDetail(code="annotation_failed", message="boom"),),
        }
    )
    with pytest.raises(OrganelleInputError):
        resolve_annotation_evidence(result)


def test_resolver_rejects_wrong_scope(tmp_path: Path) -> None:
    result = annotation_result_fixture(tmp_path).model_copy(update={"scope": "plastid"})
    with pytest.raises(OrganelleInputError):
        resolve_annotation_evidence(result)


def test_resolver_rejects_missing_annotation_artifact(tmp_path: Path) -> None:
    result = _drop_artifact(annotation_result_fixture(tmp_path), _ANNOTATION_KIND)
    with pytest.raises(OrganelleInputError) as caught:
        resolve_annotation_evidence(result)
    assert caught.value.code == "qc.annotation_artifact_missing"


def test_resolver_rejects_missing_manifest_artifact(tmp_path: Path) -> None:
    result = _drop_artifact(annotation_result_fixture(tmp_path), _MANIFEST_KIND)
    with pytest.raises(OrganelleInputError) as caught:
        resolve_annotation_evidence(result)
    assert caught.value.code == "qc.annotation_artifact_missing"


def test_resolver_rejects_duplicate_annotation_artifact(tmp_path: Path) -> None:
    result = _duplicate_artifact(annotation_result_fixture(tmp_path), _ANNOTATION_KIND)
    with pytest.raises(OrganelleInputError):
        resolve_annotation_evidence(result)


def test_resolver_rejects_malformed_manifest(tmp_path: Path) -> None:
    result = _remanifest(
        annotation_result_fixture(tmp_path), lambda data: data.clear()
    )
    with pytest.raises(OrganelleInputError) as caught:
        resolve_annotation_evidence(result)
    assert caught.value.code == "qc.annotation_manifest_invalid"


def test_resolver_rejects_mismatched_computed_run_id(tmp_path: Path) -> None:
    result = _remanifest(
        annotation_result_fixture(tmp_path),
        lambda data: data.update({"run_manifest_id": "annotation-run:sha256:" + "0" * 64}),
    )
    with pytest.raises(OrganelleInputError) as caught:
        resolve_annotation_evidence(result)
    assert caught.value.code == "qc.annotation_manifest_identity_mismatch"


def test_resolver_rejects_provenance_run_manifest_mismatch(tmp_path: Path) -> None:
    base = annotation_result_fixture(tmp_path)
    provenance = base.provenance
    assert provenance is not None
    result = base.model_copy(
        update={"provenance": provenance.model_copy(update={"run_manifest_id": "bogus"})}
    )
    with pytest.raises(OrganelleInputError):
        resolve_annotation_evidence(result)


def test_resolver_rejects_parameters_hash_mismatch(tmp_path: Path) -> None:
    base = annotation_result_fixture(tmp_path)
    other = "1" * 64
    provenance = base.provenance
    assert provenance is not None
    result = base.model_copy(
        update={"provenance": provenance.model_copy(update={"parameters_hash": other})}
    )
    with pytest.raises(OrganelleInputError):
        resolve_annotation_evidence(result)


def test_resolver_rejects_input_identity_mismatch(tmp_path: Path) -> None:
    base = annotation_result_fixture(tmp_path)
    provenance = base.provenance
    assert provenance is not None
    result = base.model_copy(
        update={
            "provenance": provenance.model_copy(
                update={"input_object_ids": ("genome:sha256:" + "9" * 64,)}
            )
        }
    )
    with pytest.raises(OrganelleInputError):
        resolve_annotation_evidence(result)


def test_resolver_rejects_annotation_object_id_mismatch(tmp_path: Path) -> None:
    result = _redocument(annotation_result_fixture(tmp_path))
    with pytest.raises(OrganelleInputError):
        resolve_annotation_evidence(result)


def test_resolver_rejects_backend_mismatch(tmp_path: Path) -> None:
    base = annotation_result_fixture(tmp_path)
    provenance = base.provenance
    assert provenance is not None
    result = base.model_copy(
        update={
            "status": "warning",
            "provenance": provenance.model_copy(
                update={
                    "requested_backend": "plastid",
                    "actual_backend": "plastid",
                    "attempted_backends": ("mitochondrion", "plastid"),
                }
            ),
        }
    )
    with pytest.raises(OrganelleInputError):
        resolve_annotation_evidence(result)


def test_resolver_rejects_requested_stages_mismatch(tmp_path: Path) -> None:
    base = annotation_result_fixture(tmp_path)
    metrics = _metrics_dict(base)
    metrics["requested_stages"] = ("pcg",)
    result = base.model_copy(update={"metrics": FrozenMap(metrics)})
    with pytest.raises(OrganelleInputError):
        resolve_annotation_evidence(result)


def test_resolver_rejects_feature_counts_mismatch(tmp_path: Path) -> None:
    base = annotation_result_fixture(tmp_path)
    metrics = _metrics_dict(base)
    metrics["feature_counts"] = {"cds": 99, "trna": 1, "rrna": 1}
    result = base.model_copy(update={"metrics": FrozenMap(metrics)})
    with pytest.raises(OrganelleInputError) as caught:
        resolve_annotation_evidence(result)
    assert caught.value.code == "qc.annotation_metadata_mismatch"


def test_resolver_rejects_rejected_candidates_mismatch(tmp_path: Path) -> None:
    base = annotation_result_fixture(tmp_path, rejected_gene="cox1")
    metrics = _metrics_dict(base)
    metrics["rejected_cds_count"] = 0
    metrics["rejected_cds_candidates"] = ()
    result = base.model_copy(update={"metrics": FrozenMap(metrics)})
    with pytest.raises(OrganelleInputError) as caught:
        resolve_annotation_evidence(result)
    assert caught.value.code == "qc.annotation_metadata_mismatch"


def test_resolver_rejects_missing_core_metadata_mismatch(tmp_path: Path) -> None:
    base = annotation_result_fixture(tmp_path, missing_core_genes=("cox1",))
    metrics = _metrics_dict(base)
    metrics["missing_core_genes"] = ()
    result = base.model_copy(update={"metrics": FrozenMap(metrics)})
    with pytest.raises(OrganelleInputError) as caught:
        resolve_annotation_evidence(result)
    assert caught.value.code == "qc.annotation_metadata_mismatch"


@pytest.mark.parametrize("field", ["requested_stages", "completed_stages"])
def test_resolver_rejects_missing_stage_metric(tmp_path: Path, field: str) -> None:
    """A missing stage metric must fail closed, never raise ``KeyError``."""

    base = annotation_result_fixture(tmp_path)
    metrics = _metrics_dict(base)
    del metrics[field]
    result = base.model_copy(update={"metrics": FrozenMap(metrics)})
    with pytest.raises(OrganelleInputError):
        resolve_annotation_evidence(result)


@pytest.mark.parametrize("field", ["requested_stages", "completed_stages"])
def test_resolver_rejects_non_sequence_stage_metric(tmp_path: Path, field: str) -> None:
    """A non-sequence stage metric must fail closed, never raise ``TypeError``."""

    base = annotation_result_fixture(tmp_path)
    metrics = _metrics_dict(base)
    metrics[field] = 5
    result = base.model_copy(update={"metrics": FrozenMap(metrics)})
    with pytest.raises(OrganelleInputError):
        resolve_annotation_evidence(result)


@pytest.mark.parametrize(
    "field", ["feature_counts", "rejected_cds_candidates", "missing_core_genes"]
)
def test_resolver_rejects_missing_trust_boundary_metric(
    tmp_path: Path, field: str
) -> None:
    """A missing trust-boundary metric must fail closed rather than default silently."""

    base = annotation_result_fixture(tmp_path)
    metrics = _metrics_dict(base)
    del metrics[field]
    result = base.model_copy(update={"metrics": FrozenMap(metrics)})
    with pytest.raises(OrganelleInputError):
        resolve_annotation_evidence(result)


@pytest.mark.parametrize(
    "field", ["feature_counts", "rejected_cds_candidates", "missing_core_genes"]
)
def test_resolver_rejects_non_container_trust_boundary_metric(
    tmp_path: Path, field: str
) -> None:
    """A non-container trust-boundary metric must fail closed, never raise ``TypeError``."""

    base = annotation_result_fixture(tmp_path)
    metrics = _metrics_dict(base)
    metrics[field] = 5
    result = base.model_copy(update={"metrics": FrozenMap(metrics)})
    with pytest.raises(OrganelleInputError):
        resolve_annotation_evidence(result)


# ---------------------------------------------------------------------------
# Evaluator: profile recovery and review signals
# ---------------------------------------------------------------------------


def test_evaluation_reports_profile_recovery_without_universal_completeness(
    tmp_path: Path,
) -> None:
    evidence = resolve_annotation_evidence(
        annotation_result_fixture(
            tmp_path,
            genes=("nad1", "nad5", "rps10"),
            rejected_gene="cox1",
        )
    )
    evaluated = evaluate_annotation(evidence, ANNOTATION_QC_POLICY_V1)
    assert evaluated.profile_assessment.profile_id == "organelleverse.plant-mito-pcg.v1"
    assert evaluated.profile_assessment.recovered_expected_names == ("nad1", "nad5")
    assert "cox1" in evaluated.profile_assessment.missing_expected_names
    assert evaluated.profile_assessment.observed_variable_names == ("rps10",)
    assert any(item.status == "warn" for item in evaluated.checks)
    assert not hasattr(evaluated.summary, "completeness")
    assert not hasattr(evaluated.summary, "score")


def test_evaluation_passes_when_full_expected_profile_recovered(tmp_path: Path) -> None:
    evidence = resolve_annotation_evidence(
        annotation_result_fixture(tmp_path, genes=EXPECTED_PCG_NAMES)
    )
    evaluated = evaluate_annotation(evidence, ANNOTATION_QC_POLICY_V1)
    assessment = evaluated.profile_assessment
    assert assessment.recovered_expected_names == EXPECTED_PCG_NAMES
    assert assessment.missing_expected_names == ()
    assert assessment.duplicated_expected_names == ()
    assert assessment.recovery_fraction == 1.0
    by_id = {check.check_id: check for check in evaluated.checks}
    assert by_id["annotation.canonical_structure"].status == "pass"
    assert by_id["annotation.profile_recovery"].status == "pass"
    assert all(check.status == "pass" for check in evaluated.checks)
    assert evaluated.summary.failure_count == 0
    assert evaluated.summary.warning_count == 0


def test_evaluation_normalizes_gene_name_case_to_policy_spelling(tmp_path: Path) -> None:
    evidence = resolve_annotation_evidence(
        annotation_result_fixture(tmp_path, genes=("NAD1", "Cox1"))
    )
    evaluated = evaluate_annotation(evidence, ANNOTATION_QC_POLICY_V1)
    # Names are normalized to the policy spelling and returned in policy order.
    assert evaluated.profile_assessment.recovered_expected_names == ("cox1", "nad1")


def test_evaluation_never_warns_for_absent_variable_genes(tmp_path: Path) -> None:
    evidence = resolve_annotation_evidence(
        annotation_result_fixture(tmp_path, genes=EXPECTED_PCG_NAMES)
    )
    evaluated = evaluate_annotation(evidence, ANNOTATION_QC_POLICY_V1)
    by_id = {check.check_id: check for check in evaluated.checks}
    # No variable gene was published, yet profile recovery stays a pass.
    assert evaluated.profile_assessment.observed_variable_names == ()
    assert by_id["annotation.profile_recovery"].status == "pass"


def test_evaluation_marks_duplicate_gene_models_as_warn(tmp_path: Path) -> None:
    evidence = resolve_annotation_evidence(
        annotation_result_fixture(tmp_path, genes=EXPECTED_PCG_NAMES, duplicate_gene="nad1")
    )
    evaluated = evaluate_annotation(evidence, ANNOTATION_QC_POLICY_V1)
    by_id = {check.check_id: check for check in evaluated.checks}
    assert by_id["annotation.duplicate_gene_models"].status == "warn"
    assert "nad1" in evaluated.profile_assessment.duplicated_expected_names
    assert evaluated.summary.duplicate_gene_count >= 1


def test_evaluation_marks_partial_and_pseudo_features_as_warn(tmp_path: Path) -> None:
    evidence = resolve_annotation_evidence(
        annotation_result_fixture(
            tmp_path,
            genes=EXPECTED_PCG_NAMES,
            partial_gene="cob",
            pseudo_gene="atp6",
        )
    )
    evaluated = evaluate_annotation(evidence, ANNOTATION_QC_POLICY_V1)
    by_id = {check.check_id: check for check in evaluated.checks}
    assert by_id["annotation.partial_features"].status == "warn"
    assert by_id["annotation.pseudo_features"].status == "warn"
    assert evaluated.summary.partial_feature_count >= 1
    assert evaluated.summary.pseudo_feature_count >= 1


def test_evaluation_marks_translation_exceptions_as_warn(tmp_path: Path) -> None:
    evidence = resolve_annotation_evidence(
        annotation_result_fixture(
            tmp_path,
            genes=EXPECTED_PCG_NAMES,
            transl_except_gene="cox2",
        )
    )
    evaluated = evaluate_annotation(evidence, ANNOTATION_QC_POLICY_V1)
    by_id = {check.check_id: check for check in evaluated.checks}
    assert by_id["annotation.translation_exceptions"].status == "warn"


def test_evaluation_fails_canonical_structure_on_invalid_coordinates(tmp_path: Path) -> None:
    evidence = resolve_annotation_evidence(
        annotation_result_fixture(tmp_path, genes=EXPECTED_PCG_NAMES, invalid_extra="rps3")
    )
    evaluated = evaluate_annotation(evidence, ANNOTATION_QC_POLICY_V1)
    by_id = {check.check_id: check for check in evaluated.checks}
    assert by_id["annotation.canonical_structure"].status == "fail"
    assert evaluated.summary.failure_count == 1


def test_evaluation_marks_rejected_cds_candidates_as_warn(tmp_path: Path) -> None:
    evidence = resolve_annotation_evidence(
        annotation_result_fixture(tmp_path, genes=EXPECTED_PCG_NAMES, rejected_gene="cox1")
    )
    evaluated = evaluate_annotation(evidence, ANNOTATION_QC_POLICY_V1)
    by_id = {check.check_id: check for check in evaluated.checks}
    assert by_id["annotation.rejected_cds_candidates"].status == "warn"
    assert evaluated.summary.rejected_cds_candidate_count == 1
    assert evaluated.rejected_candidates[0].gene_name == "cox1"
    assert evaluated.rejected_candidates[0].issue_codes == ("premature_stop",)


def test_evaluation_reports_rna_counts_and_passes(tmp_path: Path) -> None:
    evidence = resolve_annotation_evidence(
        annotation_result_fixture(
            tmp_path,
            genes=EXPECTED_PCG_NAMES,
            trna_genes=("trnC", "trnD"),
            rrna_genes=("rrn18",),
        )
    )
    evaluated = evaluate_annotation(evidence, ANNOTATION_QC_POLICY_V1)
    by_id = {check.check_id: check for check in evaluated.checks}
    assert by_id["annotation.rna_observations"].status == "pass"
    assert evaluated.summary.unique_trna_count == 2
    assert evaluated.summary.unique_rrna_count == 1


def test_evaluation_emits_checks_in_deterministic_order(tmp_path: Path) -> None:
    evidence = resolve_annotation_evidence(
        annotation_result_fixture(
            tmp_path,
            genes=("nad1",),
            rejected_gene="cox1",
            duplicate_gene="nad1",
            partial_gene="cob",
            pseudo_gene="atp6",
            transl_except_gene="cox2",
        )
    )
    evaluated = evaluate_annotation(evidence, ANNOTATION_QC_POLICY_V1)
    expected_order = (
        "annotation.canonical_structure",
        "annotation.translation_exceptions",
        "annotation.profile_recovery",
        "annotation.duplicate_gene_models",
        "annotation.partial_features",
        "annotation.pseudo_features",
        "annotation.rejected_cds_candidates",
        "annotation.rna_observations",
    )
    assert tuple(check.check_id for check in evaluated.checks) == expected_order


def test_evaluation_summary_counts_match_check_statuses(tmp_path: Path) -> None:
    evidence = resolve_annotation_evidence(
        annotation_result_fixture(
            tmp_path,
            genes=("nad1",),
            rejected_gene="cox1",
        )
    )
    evaluated = evaluate_annotation(evidence, ANNOTATION_QC_POLICY_V1)
    statuses = [check.status for check in evaluated.checks]
    assert evaluated.summary.pass_count == statuses.count("pass")
    assert evaluated.summary.warning_count == statuses.count("warn")
    assert evaluated.summary.failure_count == statuses.count("fail")
    assert evaluated.summary.not_assessed_count == statuses.count("not_assessed")
