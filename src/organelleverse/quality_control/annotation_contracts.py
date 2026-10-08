"""Strict, immutable annotation-QC contract vocabulary.

Every model is a frozen Pydantic v2 model with ``extra="forbid"``. There is no
public ``dict[str, Any]`` surface and no universal 0-100 score, grade, or
completeness percentage. Profile recovery is reported under that exact name and
is never presented as genome completeness.

These contracts reuse the shared :class:`~organelleverse.quality_control.contracts.QcCheck`,
:class:`~organelleverse.quality_control.contracts.QcDecision`, and
:class:`~organelleverse.quality_control.contracts.QcStageOutcome` vocabulary.
Assembly-only contracts remain assembly-only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, field_serializer, field_validator

from organelleverse.annotation.models import AnnotationDocument
from organelleverse.assembly.contracts import canonical_json_bytes
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.base import StrictFrozenModel
from organelleverse.core.errors import OrganelleInternalError
from organelleverse.core.frozen import FrozenJson, FrozenMap, freeze_json, thaw_json
from organelleverse.core.result import OrganelleResult

from .contracts import (
    FractionFloat,
    QcCheck,
    QcDecision,
    QcStageOutcome,
    StrictBool,
    StrictNonNegInt,
    StrictPosInt,
)

# ---------------------------------------------------------------------------
# Shared strict aliases
# ---------------------------------------------------------------------------

#: One ordered ``(start, end, strand)`` location part (half-open, 1-based strand).
LocationPartRecord = tuple[StrictNonNegInt, StrictPosInt, Literal[-1, 1]]

#: One ordered ``(start, end, strand)`` part of a rejected CDS candidate.
RejectedPartRecord = tuple[StrictNonNegInt, StrictPosInt, Literal[-1, 1]]


def _freeze_mapping(value: object, field_name: str) -> FrozenMap[FrozenJson]:
    try:
        frozen = freeze_json({} if value is None else value)
    except OrganelleInternalError as error:
        raise ValueError(str(error)) from error
    if not isinstance(frozen, FrozenMap):
        raise ValueError(f"{field_name} must be a JSON object")
    return frozen


# ---------------------------------------------------------------------------
# Source trust-boundary manifest
# ---------------------------------------------------------------------------


class AnnotationSourceRunManifest(
    StrictFrozenModel[Literal["annotation_source_run_manifest"]]
):
    """Strict view of the source ``annotation.annotate`` run manifest.

    Only the fixed fields emitted by the annotation backend are admitted across
    the QC trust boundary. Free-form manifest dictionaries never become part of
    the public QC contract.
    """

    kind: Literal["annotation_source_run_manifest"] = "annotation_source_run_manifest"
    schema_version: Literal["organelleverse.annotation-run.v1"]
    operation_id: Literal["annotation.annotate"]
    input_object_id: str
    input_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameters_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend: Literal["mitochondrion"]
    annotation_object_id: str
    software_versions: FrozenMap[FrozenJson]
    database_hashes: FrozenMap[FrozenJson]
    argv: tuple[tuple[str, ...], ...]
    run_manifest_id: str
    commands_file: str
    logs: tuple[str, ...] = ()

    @field_validator("software_versions", "database_hashes", mode="before")
    @classmethod
    def _freeze_objects(cls, value: object) -> FrozenMap[FrozenJson]:
        return _freeze_mapping(value, "annotation run mapping field")

    @field_serializer("software_versions", "database_hashes")
    def _serialize_objects(self, value: FrozenMap[FrozenJson]) -> object:
        return thaw_json(value)

    def semantic_payload(self) -> dict[str, object]:
        """Canonical, ordering-independent identity payload."""

        return {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "input_object_id": self.input_object_id,
            "input_artifact_hash": self.input_artifact_hash,
            "parameters_hash": self.parameters_hash,
            "backend": self.backend,
            "annotation_object_id": self.annotation_object_id,
            "software_versions": dict(
                sorted(cast("dict[str, object]", thaw_json(self.software_versions)).items())
            ),
            "database_hashes": dict(
                sorted(cast("dict[str, object]", thaw_json(self.database_hashes)).items())
            ),
            "argv": [list(item) for item in self.argv],
        }

    @property
    def computed_run_manifest_id(self) -> str:
        """Recompute the manifest identity from its canonical semantic payload."""

        payload = json.dumps(
            self.semantic_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        return f"annotation-run:sha256:{hashlib.sha256(payload).hexdigest()}"


# ---------------------------------------------------------------------------
# Descriptive feature and rejected-candidate records
# ---------------------------------------------------------------------------


class RejectedCdsCandidate(StrictFrozenModel[Literal["rejected_cds_candidate"]]):
    """One CDS candidate the annotation backend rejected before publication.

    Provenance remains the source annotation run; QC does not rediscover
    rejected candidates from the already filtered published feature set.
    """

    kind: Literal["rejected_cds_candidate"] = "rejected_cds_candidate"
    gene_name: str = Field(min_length=1)
    start: StrictNonNegInt | None = None
    end: StrictPosInt | None = None
    strand: Literal[-1, 1]
    parts: tuple[RejectedPartRecord, ...]
    issue_codes: tuple[str, ...]
    issue_messages: tuple[str, ...]


class AnnotationFeatureSummary(StrictFrozenModel[Literal["annotation_feature_summary"]]):
    """One descriptive record per published annotation feature.

    Descriptive only: overlapping or compound features are not inherently errors
    in plant mitochondrial genomes.
    """

    kind: Literal["annotation_feature_summary"] = "annotation_feature_summary"
    record_id: str = Field(min_length=1)
    feature_id: str = Field(min_length=1)
    feature_type: str = Field(min_length=1)
    gene_name: str
    strand: Literal[-1, 1]
    location_parts: tuple[LocationPartRecord, ...]
    spliced_length: StrictPosInt
    is_partial: StrictBool = False
    is_pseudo: StrictBool = False
    is_translation_exempt: StrictBool = False
    is_trans_spliced: StrictBool = False
    parents: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Gene profile assessment and aggregate summary
# ---------------------------------------------------------------------------


class GeneProfileAssessment(StrictFrozenModel[Literal["gene_profile_assessment"]]):
    """Recovery of the fixed plant mitochondrial core/variable PCG profile.

    Variable PCGs never enter the expected-profile denominator and never produce
    a review signal when absent. Missing expected genes are a review signal, not
    an assertion that the genome is biologically incomplete.
    """

    kind: Literal["gene_profile_assessment"] = "gene_profile_assessment"
    profile_id: str = Field(min_length=1)
    profile_version: str = Field(min_length=1)
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope: str = Field(min_length=1)
    expected_pcg_names: tuple[str, ...]
    variable_pcg_names: tuple[str, ...]
    recovered_expected_names: tuple[str, ...]
    missing_expected_names: tuple[str, ...]
    duplicated_expected_names: tuple[str, ...]
    observed_variable_names: tuple[str, ...]
    absent_variable_names: tuple[str, ...]
    expected_profile_gene_count: StrictNonNegInt
    recovered_profile_gene_count: StrictNonNegInt
    duplicated_profile_gene_count: StrictNonNegInt
    missing_profile_gene_count: StrictNonNegInt
    recovery_fraction: FractionFloat | None


class AnnotationQcSummary(StrictFrozenModel[Literal["annotation_qc_summary"]]):
    """Aggregate check counts plus descriptive annotation evidence metrics.

    No field here is a universal pass/fail threshold. There is deliberately no
    ``score``, ``grade``, or ``completeness`` field.
    """

    kind: Literal["annotation_qc_summary"] = "annotation_qc_summary"
    pass_count: StrictNonNegInt
    warning_count: StrictNonNegInt
    failure_count: StrictNonNegInt
    not_assessed_count: StrictNonNegInt
    record_count: StrictNonNegInt
    feature_count: StrictNonNegInt
    unique_pcg_count: StrictNonNegInt
    unique_trna_count: StrictNonNegInt
    unique_rrna_count: StrictNonNegInt
    rejected_cds_candidate_count: StrictNonNegInt
    partial_feature_count: StrictNonNegInt
    pseudo_feature_count: StrictNonNegInt
    translation_exempt_feature_count: StrictNonNegInt
    duplicate_gene_count: StrictNonNegInt
    expected_profile_gene_count: StrictNonNegInt
    recovered_profile_gene_count: StrictNonNegInt
    duplicated_profile_gene_count: StrictNonNegInt
    missing_profile_gene_count: StrictNonNegInt
    profile_recovery_fraction: FractionFloat | None


# ---------------------------------------------------------------------------
# Canonical report and run manifest
# ---------------------------------------------------------------------------


class AnnotationQcReport(StrictFrozenModel[Literal["annotation_qc_report"]]):
    """The immutable canonical annotation-QC report.

    Holds tuples of strict records only; there is no free-form public mapping, no
    0-100 score, and no genome-completeness percentage.
    """

    kind: Literal["annotation_qc_report"] = "annotation_qc_report"
    schema_version: Literal["organelleverse.annotation-qc.v1"] = (
        "organelleverse.annotation-qc.v1"
    )
    policy_version: str
    source_annotation_result_id: str
    source_run_manifest_id: str
    source_annotation_object_id: str
    organelle: Literal["mitochondrion"]
    backend: Literal["mitochondrion"]
    decision: QcDecision
    summary: AnnotationQcSummary
    checks: tuple[QcCheck, ...] = ()
    feature_summaries: tuple[AnnotationFeatureSummary, ...] = ()
    gene_profile_assessment: GeneProfileAssessment
    rejected_candidates: tuple[RejectedCdsCandidate, ...] = ()
    evidence_artifacts: tuple[ArtifactRef, ...] = ()


class AnnotationQcRunManifest(StrictFrozenModel[Literal["annotation_qc_run_manifest"]]):
    """Stable, content-addressed manifest for one annotation-QC run."""

    kind: Literal["annotation_qc_run_manifest"] = "annotation_qc_run_manifest"
    schema_version: Literal["organelleverse.annotation-qc-run.v1"] = (
        "organelleverse.annotation-qc-run.v1"
    )
    operation_id: Literal["qc.annotation"] = "qc.annotation"
    policy_version: str
    source_annotation_result_id: str
    source_run_manifest_id: str
    source_annotation_object_id: str
    parameters_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    package_version: str
    stages: tuple[QcStageOutcome, ...] = ()
    outputs: tuple[ArtifactRef, ...] = ()

    def semantic_payload(self) -> dict[str, object]:
        """Canonical, ordering-independent JSON identity payload."""

        return {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "policy_version": self.policy_version,
            "source_annotation_result_id": self.source_annotation_result_id,
            "source_run_manifest_id": self.source_run_manifest_id,
            "source_annotation_object_id": self.source_annotation_object_id,
            "parameters_hash": self.parameters_hash,
            "package_version": self.package_version,
            "stages": [item.model_dump(mode="json", exclude={"object_id"}) for item in self.stages],
            "outputs": [item.object_id for item in self.outputs],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())

    @property
    def run_manifest_id(self) -> str:
        digest = hashlib.sha256(self.canonical_bytes()).hexdigest()
        return f"annotation-qc-run:sha256:{digest}"

    def _identity_payload(self) -> dict[str, Any]:
        payload = self.semantic_payload()
        payload["kind"] = self.kind
        return payload


# ---------------------------------------------------------------------------
# Resolved evidence and evaluation containers (internal organization)
# ---------------------------------------------------------------------------


class ResolvedAnnotationEvidence(StrictFrozenModel[Literal["resolved_annotation_evidence"]]):
    """The fully resolved, integrity-checked annotation evidence for one QC run.

    Path-bearing internal record: the canonical annotation and source manifest are
    resolved to host paths; the source Result, manifest, and document remain typed
    contracts.
    """

    kind: Literal["resolved_annotation_evidence"] = "resolved_annotation_evidence"
    source_result: OrganelleResult
    source_manifest: AnnotationSourceRunManifest
    document: AnnotationDocument
    annotation_artifact: ArtifactRef
    manifest_artifact: ArtifactRef
    annotation_path: Path
    manifest_path: Path

    @field_serializer("annotation_path", "manifest_path")
    def _serialize_path(self, value: Path) -> str:
        return str(value)


class AnnotationEvaluation(StrictFrozenModel[Literal["annotation_evaluation"]]):
    """The deterministic evaluation products consumed by the QC service."""

    kind: Literal["annotation_evaluation"] = "annotation_evaluation"
    checks: tuple[QcCheck, ...]
    feature_summaries: tuple[AnnotationFeatureSummary, ...]
    profile_assessment: GeneProfileAssessment
    rejected_candidates: tuple[RejectedCdsCandidate, ...]
    summary: AnnotationQcSummary


__all__ = [
    "AnnotationEvaluation",
    "AnnotationFeatureSummary",
    "AnnotationQcReport",
    "AnnotationQcRunManifest",
    "AnnotationQcSummary",
    "AnnotationSourceRunManifest",
    "GeneProfileAssessment",
    "LocationPartRecord",
    "RejectedCdsCandidate",
    "RejectedPartRecord",
    "ResolvedAnnotationEvidence",
]
