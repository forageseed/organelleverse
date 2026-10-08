"""Strict, immutable assembly-QC contract vocabulary.

Every model is a frozen Pydantic v2 model with ``extra="forbid"``. There is no
public ``dict[str, Any]`` surface: every observation is a typed record, every
artifact reference is an :class:`~organelleverse.core.artifacts.ArtifactRef`, and
every path-bearing internal record uses :class:`pathlib.Path`.

The canonical report (:class:`AssemblyQcReport`) holds tuples of these records.
It never assigns a 0-100 score or an A-F grade.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_serializer,
    field_validator,
    model_validator,
)

from organelleverse.assembly.contracts import canonical_json_bytes
from organelleverse.assembly.manifests import AssemblyRunManifest
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.base import StrictFrozenModel
from organelleverse.core.errors import OrganelleInternalError
from organelleverse.core.frozen import FrozenJson, FrozenMap, freeze_json, thaw_json
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OrganelleResult

# ---------------------------------------------------------------------------
# Closed enumerations
# ---------------------------------------------------------------------------

QcCheckStatus = Literal["pass", "warn", "fail", "not_assessed"]
QcDecision = Literal["ready", "needs_review", "not_ready", "insufficient_evidence"]
IdentityEvidenceStatus = Literal["target_supported", "context_supported", "unresolved"]
JunctionStatus = Literal["supported", "unsupported", "contradicted", "not_assessed"]
RepeatStatus = Literal["resolved", "ambiguous", "contradicted", "not_assessed"]
ErrorCandidateStatus = Literal["confirmed", "candidate", "rejected", "not_assessed"]
CoverageAssessmentStatus = Literal["assessed", "not_assessed"]
Strand = Literal["+", "-"]
OrganelleKind = Literal["mitochondrion", "plastid"]

# ---------------------------------------------------------------------------
# Strict scalar aliases
#
# Every public numeric/boolean scalar rejects string/bool/int coercion and
# (for floats) NaN/Inf. These are field-level constraints so JSON tuple
# round-trip is unaffected (global strict mode would reject JSON arrays as
# tuples).
# ---------------------------------------------------------------------------

StrictNonNegInt = Annotated[StrictInt, Field(ge=0)]
StrictPosInt = Annotated[StrictInt, Field(gt=0)]
FiniteFloat = Annotated[StrictFloat, Field(allow_inf_nan=False)]
NonNegFloat = Annotated[StrictFloat, Field(ge=0, allow_inf_nan=False)]
FractionFloat = Annotated[StrictFloat, Field(ge=0, le=1, allow_inf_nan=False)]
PosFloat = Annotated[StrictFloat, Field(gt=0, allow_inf_nan=False)]

# Strict QcCheck value: a legitimate int, finite float, string, bool, or null.
# String and bool are themselves strict so bytes/dict/list cannot coerce in;
# floats reject NaN/Inf.
QcCheckValue = StrictInt | FiniteFloat | StrictStr | StrictBool | None

# Closed input-library enumerations, mirroring the released assembly contracts.
InputTechnology = Literal["illumina", "pacbio_hifi", "pacbio_clr", "ont"]
InputLayout = Literal["", "paired_end", "single_end"]
InputQualityState = Literal["", "raw", "corrected", "duplex", "hq", "ccs"]

# Valid long-read technology/quality_state pairs (illumina carries no state).
_LONG_QUALITY_PAIRS: dict[str, frozenset[str]] = {
    "pacbio_hifi": frozenset({"ccs"}),
    "pacbio_clr": frozenset({"raw", "corrected"}),
    "ont": frozenset({"raw", "corrected", "duplex", "hq"}),
}


def _freeze_mapping(value: object, field_name: str) -> FrozenMap[FrozenJson]:
    try:
        frozen = freeze_json({} if value is None else value)
    except OrganelleInternalError as error:
        raise ValueError(str(error)) from error
    if not isinstance(frozen, FrozenMap):
        raise ValueError(f"{field_name} must be a JSON object")
    return frozen


# ---------------------------------------------------------------------------
# Check, summary, and per-sequence records
# ---------------------------------------------------------------------------


class QcCheck(StrictFrozenModel[Literal["qc_check"]]):
    """One machine-readable quality check."""

    kind: Literal["qc_check"] = "qc_check"
    check_id: str
    category: str
    status: QcCheckStatus
    value: QcCheckValue = None
    unit: str = ""
    message: str
    finding_code: str = ""
    evidence_artifact_ids: tuple[str, ...] = ()


class QcSummary(StrictFrozenModel[Literal["qc_summary"]]):
    """Aggregate check counts plus the headline evidence metrics.

    The numeric metrics are ``None`` when their evidence class was not assessed;
    they are never invented zeroes. None of these values is a universal
    pass/fail threshold. There is deliberately no cross-library mapping/consensus
    QV: such a value cannot be honestly reduced from heterogeneous libraries
    (see :class:`LibraryMappingSummary.read_assembly_agreement_qv`).
    """

    kind: Literal["qc_summary"] = "qc_summary"
    pass_count: StrictNonNegInt
    warning_count: StrictNonNegInt
    failure_count: StrictNonNegInt
    not_assessed_count: StrictNonNegInt
    any_alignment_coverage_breadth: FractionFloat | None = None
    any_alignment_zero_coverage_bases: StrictNonNegInt | None = None
    confident_coverage_breadth: FractionFloat | None = None
    confident_zero_coverage_bases: StrictNonNegInt | None = None
    unsupported_junction_count: StrictNonNegInt | None = None
    contradicted_junction_count: StrictNonNegInt | None = None
    structural_error_candidate_count: StrictNonNegInt | None = None


class SequenceSummary(StrictFrozenModel[Literal["sequence_summary"]]):
    """Descriptive and identity-evidence summary for one reported sequence."""

    kind: Literal["sequence_summary"] = "sequence_summary"
    sequence_id: str
    role: str
    length: StrictPosInt
    gc_fraction: FractionFloat
    ambiguous_bases: StrictNonNegInt
    coverage_breadth: FractionFloat | None = None
    median_depth: NonNegFloat | None = None
    p5_depth: NonNegFloat | None = None
    p10_depth: NonNegFloat | None = None
    marker_ids: tuple[str, ...] = ()
    graph_context: str = ""
    identity_evidence_status: IdentityEvidenceStatus = "unresolved"


# ---------------------------------------------------------------------------
# Evidence records
# ---------------------------------------------------------------------------


class InputLibraryEvidence(StrictFrozenModel[Literal["input_library_evidence"]]):
    """One resolved input read library and its declared sequencing metadata.

    A closed, coherent contract: illumina requires a short-read layout and mates
    and carries no quality state; long-read technologies require a valid
    technology/quality_state pair and ``reads`` only.
    """

    kind: Literal["input_library_evidence"] = "input_library_evidence"
    role: str
    technology: InputTechnology
    layout: InputLayout = ""
    quality_state: InputQualityState = ""
    read1: ArtifactRef | None = None
    read2: ArtifactRef | None = None
    reads: ArtifactRef | None = None
    read_length: StrictPosInt | None = None
    insert_size: StrictPosInt | None = None

    @model_validator(mode="after")
    def _validate_coherent_library(self) -> InputLibraryEvidence:
        technology = self.technology
        if technology == "illumina":
            if self.layout == "":
                raise ValueError("illumina library requires a paired_end or single_end layout")
            if self.read1 is None:
                raise ValueError("illumina library requires read1")
            if self.read_length is None:
                raise ValueError("illumina library requires a positive read_length")
            if self.layout == "paired_end" and self.read2 is None:
                raise ValueError("paired_end illumina library requires read2")
            if self.layout == "single_end" and self.read2 is not None:
                raise ValueError("single_end illumina library forbids read2")
            if self.reads is not None:
                raise ValueError("illumina library must not declare long reads")
            if self.quality_state != "":
                raise ValueError("illumina library must not declare a quality state")
        else:
            allowed = _LONG_QUALITY_PAIRS[technology]
            if self.quality_state not in allowed:
                raise ValueError(
                    f"{technology} does not support quality_state={self.quality_state!r}"
                )
            if self.reads is None:
                raise ValueError(f"{technology} library requires reads")
            if self.layout != "" or self.read1 is not None or self.read2 is not None:
                raise ValueError("long-read library must not declare short-read layout or mates")
            if self.read_length is not None or self.insert_size is not None:
                raise ValueError("long-read library must not declare read_length or insert_size")
        return self


class GfaSummary(StrictFrozenModel[Literal["gfa_summary"]]):
    """Topology counts derived from a GFA without circularity heuristics.

    ``parallel_edge_count`` is exactly what its name says: the number of
    segment pairs joined by more than one link. It is not a graph
    bubble/superbubble (a precisely defined assembly-graph structure), which
    would require a dedicated detector; this count is reported under its
    honest name to avoid that implication.
    """

    kind: Literal["gfa_summary"] = "gfa_summary"
    segment_count: StrictNonNegInt
    edge_count: StrictNonNegInt
    path_count: StrictNonNegInt
    component_count: StrictNonNegInt
    branch_count: StrictNonNegInt
    parallel_edge_count: StrictNonNegInt
    tip_count: StrictNonNegInt
    path_names: tuple[str, ...] = ()


class JunctionSupport(StrictFrozenModel[Literal["junction_support"]]):
    """Read support for one reported GFA adjacency."""

    kind: Literal["junction_support"] = "junction_support"
    sequence_id: str
    left_segment: str
    right_segment: str
    library_role: str
    supporting_reads: StrictNonNegInt
    contradicting_reads: StrictNonNegInt
    status: JunctionStatus = "not_assessed"
    anchor_length: StrictNonNegInt


class RepeatSupport(StrictFrozenModel[Literal["repeat_support"]]):
    """Path-occurrence and spanning-read evidence for one repeat context.

    ``path_occurrences`` is the number of times a segment appears in a declared
    GFA path; it is not a read-depth copy-number inference. Repeat-local depth
    is not computed (the previous ``median_depth`` was the largest genome-wide
    library median, which is not a repeat-local quantity), so no depth field is
    exposed here.
    """

    kind: Literal["repeat_support"] = "repeat_support"
    repeat_id: str
    sequence_id: str
    length: StrictPosInt
    orientation: str = ""
    path_occurrences: StrictNonNegInt
    spanning_reads: StrictNonNegInt
    status: RepeatStatus = "not_assessed"


class MarkerHit(StrictFrozenModel[Literal["marker_hit"]]):
    """One consolidated organelle-marker profile hit."""

    kind: Literal["marker_hit"] = "marker_hit"
    profile_id: str
    sequence_id: str
    start: StrictNonNegInt
    end: StrictPosInt
    strand: Strand = "+"
    score: NonNegFloat | None = None
    complete: StrictBool = False
    target: str = ""
    evidence_artifact_ids: tuple[str, ...] = ()


class MarkerProfileSet(StrictFrozenModel[Literal["marker_profile_set"]]):
    """Versioned marker HMM input with an explicit genetic code."""

    kind: Literal["marker_profile_set"] = "marker_profile_set"
    profile_set_id: str
    version: str
    hmm_artifact: ArtifactRef
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    genetic_code: Annotated[StrictInt, Field(ge=1, le=33)] | None = None
    target: OrganelleKind | None = None

    @model_validator(mode="after")
    def _validate_profile_digest(self) -> MarkerProfileSet:
        if self.sha256 != self.hmm_artifact.sha256:
            raise ValueError("sha256 must match the HMM artifact digest")
        return self


class ErrorCandidate(StrictFrozenModel[Literal["error_candidate"]]):
    """A base-level or structural assembly-error candidate.

    Base-level candidates expose the assessed confident-read depth and support
    fraction used for promotion. ``confidence`` remains reserved for a
    calibrated probability and is not populated from the support fraction.
    Structural candidates may leave the added read-level fields unset.
    """

    kind: Literal["error_candidate"] = "error_candidate"
    candidate_id: str
    sequence_id: str
    start: StrictNonNegInt
    end: StrictPosInt
    error_type: str
    confidence: FractionFloat | None = None
    supporting_observations: StrictNonNegInt = 0
    assessed_depth: StrictNonNegInt | None = None
    support_fraction: FractionFloat | None = None
    supporting_library_count: StrictNonNegInt | None = None
    status: ErrorCandidateStatus = "candidate"
    evidence_artifact_ids: tuple[str, ...] = ()


class CoverageWindow(StrictFrozenModel[Literal["coverage_window"]]):
    """Depth statistics for one deterministic coverage window.

    Two non-interchangeable tracks are reported per window, each with its own
    mean and minimum. The ``any_alignment`` track is repeat-aware: it counts
    every mapped alignment (primary, secondary, supplementary, any MAPQ) passing
    the base-quality floor, so a true assembly gap can fail on it. The
    ``confident`` track is restricted to primary alignments at or above the
    mapping-quality floor; a gap here only warns, because ambiguous repeat
    mapping is not assembly-missing evidence.

    The per-track minimum is a window-level descriptive statistic. The precise
    zero-base count is never inferred from it: it is counted exactly from the
    per-track coverage-interval BED artifacts (see
    :class:`MappingEvidence.coordinate_artifacts`).
    """

    kind: Literal["coverage_window"] = "coverage_window"
    sequence_id: str
    start: StrictNonNegInt
    end: StrictPosInt
    any_alignment_mean_depth: NonNegFloat | None = None
    any_alignment_minimum_depth: NonNegFloat | None = None
    confident_mean_depth: NonNegFloat | None = None
    confident_minimum_depth: NonNegFloat | None = None
    assessment_status: CoverageAssessmentStatus = "not_assessed"


class SequenceDepthProfile(StrictFrozenModel[Literal["sequence_depth_profile"]]):
    """Combined per-base depth statistics for one sequence (any-alignment track).

    Aggregated across every declared library over the repeat-aware
    ``any_alignment`` per-base depth array. These are the real per-base depths,
    not window means; they populate :class:`SequenceSummary` descriptive depth
    fields. A confident-track equivalent is intentionally not exposed here to
    keep the per-sequence surface honest and small.
    """

    kind: Literal["sequence_depth_profile"] = "sequence_depth_profile"
    sequence_id: str
    median_depth: NonNegFloat | None = None
    p5_depth: NonNegFloat | None = None
    p10_depth: NonNegFloat | None = None


class AssemblyStatistics(StrictFrozenModel[Literal["assembly_statistics"]]):
    """Descriptive whole-assembly contiguity and composition statistics.

    Pure description: these values never participate in a pass/fail decision.
    ``n50``/``l50`` use the standard contig definition (lengths sorted
    descending; ``n50`` is the length of the shortest contig in the smallest set
    whose cumulative length reaches half the total, ``l50`` is the count of
    contigs in that set).
    """

    kind: Literal["assembly_statistics"] = "assembly_statistics"
    sequence_count: StrictPosInt
    total_length: StrictPosInt
    largest_sequence_length: StrictPosInt
    n50: StrictNonNegInt
    l50: StrictNonNegInt
    overall_gc_fraction: FractionFloat
    ambiguous_bases: StrictNonNegInt


class ToolIdentity(StrictFrozenModel[Literal["tool_identity"]]):
    """Stable identity of one tool used during the QC run."""

    kind: Literal["tool_identity"] = "tool_identity"
    name: str
    version: str = ""
    executable_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    environment_id: str = ""


class LibraryMappingSummary(StrictFrozenModel[Literal["library_mapping_summary"]]):
    """Per-library read-to-assembly mapping summary.

    ``read_assembly_agreement_qv`` is ``-10 log10`` of a one-pseudocount
    smoothed per-library reads-vs-assembly mismatch/indel rate over primary
    aligned bases. It is
    **not** a consensus assembly QV (Merqury QV): it conflates sequencing
    error, mapping error, intra-sample heterogeneity, and assembly error. The
    declared ``technology`` and ``quality_state`` are reported alongside it so
    heterogeneous libraries are never reduced to one comparable headline QV.
    """

    kind: Literal["library_mapping_summary"] = "library_mapping_summary"
    library_role: str
    technology: InputTechnology
    quality_state: InputQualityState = ""
    mapped_reads: StrictNonNegInt
    target_fraction: FractionFloat | None = None
    median_depth: NonNegFloat | None = None
    p5_depth: NonNegFloat | None = None
    p10_depth: NonNegFloat | None = None
    mismatch_rate: NonNegFloat | None = None
    insertion_rate: NonNegFloat | None = None
    deletion_rate: NonNegFloat | None = None
    read_assembly_agreement_qv: NonNegFloat | None = None


class LibraryAlignmentEvidence(StrictFrozenModel[Literal["library_alignment_evidence"]]):
    """Coordinate-sorted alignment artifacts for one declared read library."""

    kind: Literal["library_alignment_evidence"] = "library_alignment_evidence"
    library_role: str
    bam_artifact: ArtifactRef
    index_artifact: ArtifactRef


class SequenceIdentity(StrictFrozenModel[Literal["sequence_identity"]]):
    """Per-sequence target-identity evidence status."""

    kind: Literal["sequence_identity"] = "sequence_identity"
    sequence_id: str
    identity_evidence_status: IdentityEvidenceStatus = "unresolved"


class AlternativeConfiguration(StrictFrozenModel[Literal["alternative_configuration"]]):
    """One supported alternative organelle configuration."""

    kind: Literal["alternative_configuration"] = "alternative_configuration"
    configuration_id: str
    sequence_id: str
    relative_support: NonNegFloat | None = None
    status: Literal["supported", "unresolved", "contradicted"] = "unresolved"


# ---------------------------------------------------------------------------
# Evidence containers (internal organization; the report flattens these)
# ---------------------------------------------------------------------------


class MappingEvidence(StrictFrozenModel[Literal["mapping_evidence"]]):
    """Read-mapping evidence: summaries, coverage, base errors, artifacts."""

    kind: Literal["mapping_evidence"] = "mapping_evidence"
    library_mapping_summaries: tuple[LibraryMappingSummary, ...] = ()
    coverage_windows: tuple[CoverageWindow, ...] = ()
    sequence_depth_profiles: tuple[SequenceDepthProfile, ...] = ()
    base_error_candidates: tuple[ErrorCandidate, ...] = ()
    library_alignments: tuple[LibraryAlignmentEvidence, ...] = ()
    coordinate_artifacts: tuple[ArtifactRef, ...] = ()
    tool_identities: tuple[ToolIdentity, ...] = ()
    checks: tuple[QcCheck, ...] = ()


class GraphEvidence(StrictFrozenModel[Literal["graph_evidence"]]):
    """GFA topology, junction, repeat, and structural-error evidence."""

    kind: Literal["graph_evidence"] = "graph_evidence"
    gfa_summary: GfaSummary | None = None
    junction_support: tuple[JunctionSupport, ...] = ()
    repeat_support: tuple[RepeatSupport, ...] = ()
    structural_error_candidates: tuple[ErrorCandidate, ...] = ()
    checks: tuple[QcCheck, ...] = ()


class MarkerEvidence(StrictFrozenModel[Literal["marker_evidence"]]):
    """Marker-hit, profile-recovery, and per-sequence identity evidence.

    Target-profile recovery is recovery of the exact HMM profiles searched in
    this run. It is not a genome-completeness estimate.
    """

    kind: Literal["marker_evidence"] = "marker_evidence"
    profiles_assessed: StrictBool = False
    profile_set_id: str = ""
    profile_version: str = ""
    profile_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    genetic_code: Annotated[StrictInt, Field(ge=1, le=33)] | None = None
    expected_target_profile_count: StrictNonNegInt = 0
    complete_target_profile_count: StrictNonNegInt = 0
    target_profile_recovery_fraction: FractionFloat | None = None
    duplicate_complete_target_profile_count: StrictNonNegInt = 0
    marker_hits: tuple[MarkerHit, ...] = ()
    sequence_identities: tuple[SequenceIdentity, ...] = ()
    checks: tuple[QcCheck, ...] = ()


class MarkerProfileAssessment(StrictFrozenModel[Literal["marker_profile_assessment"]]):
    """Report-level summary of the exact marker profile scan."""

    kind: Literal["marker_profile_assessment"] = "marker_profile_assessment"
    profiles_assessed: StrictBool
    profile_set_id: str = ""
    profile_version: str = ""
    profile_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    genetic_code: Annotated[StrictInt, Field(ge=1, le=33)] | None = None
    expected_target_profile_count: StrictNonNegInt = 0
    complete_target_profile_count: StrictNonNegInt = 0
    target_profile_recovery_fraction: FractionFloat | None = None
    duplicate_complete_target_profile_count: StrictNonNegInt = 0


class OrganelleEvidence(StrictFrozenModel[Literal["organelle_evidence"]]):
    """Organelle-specific structural checks and alternative configurations."""

    kind: Literal["organelle_evidence"] = "organelle_evidence"
    checks: tuple[QcCheck, ...] = ()
    sequence_identities: tuple[SequenceIdentity, ...] = ()
    alternative_configurations: tuple[AlternativeConfiguration, ...] = ()


class ResolvedAssemblyEvidence(StrictFrozenModel[Literal["resolved_assembly_evidence"]]):
    """The fully resolved, integrity-checked input evidence for one QC run.

    Path-bearing internal record: primary FASTA and optional GFA are resolved to
    host paths; the source Result, manifest, and genome remain typed contracts.
    """

    kind: Literal["resolved_assembly_evidence"] = "resolved_assembly_evidence"
    source_result: OrganelleResult
    run_manifest: AssemblyRunManifest
    primary_genome: OrganelleGenome
    primary_fasta_path: Path
    assembly_graph_path: Path | None = None
    input_libraries: tuple[InputLibraryEvidence, ...] = ()

    @field_serializer("primary_fasta_path", "assembly_graph_path")
    def _serialize_path(self, value: Path | None) -> str | None:
        return None if value is None else str(value)


# ---------------------------------------------------------------------------
# Canonical report and run manifest
# ---------------------------------------------------------------------------


class AssemblyQcReport(StrictFrozenModel[Literal["assembly_qc_report"]]):
    """The immutable canonical assembly-QC report.

    Holds tuples of strict records only; there is no free-form public mapping,
    no 0-100 score, and no A-F grade.
    """

    kind: Literal["assembly_qc_report"] = "assembly_qc_report"
    schema_version: Literal["organelleverse.assembly-qc.v1"] = "organelleverse.assembly-qc.v1"
    policy_version: str
    source_assembly_result_id: str
    source_run_manifest_id: str
    organelle: OrganelleKind
    decision: QcDecision
    summary: QcSummary
    checks: tuple[QcCheck, ...] = ()
    sequence_summaries: tuple[SequenceSummary, ...] = ()
    junction_support: tuple[JunctionSupport, ...] = ()
    repeat_support: tuple[RepeatSupport, ...] = ()
    marker_hits: tuple[MarkerHit, ...] = ()
    marker_profile_assessment: MarkerProfileAssessment | None = None
    error_candidates: tuple[ErrorCandidate, ...] = ()
    coverage_windows: tuple[CoverageWindow, ...] = ()
    library_mapping_summaries: tuple[LibraryMappingSummary, ...] = ()
    gfa_summary: GfaSummary | None = None
    alternative_configurations: tuple[AlternativeConfiguration, ...] = ()
    assembly_statistics: AssemblyStatistics | None = None
    evidence_artifacts: tuple[ArtifactRef, ...] = ()
    tool_identities: tuple[ToolIdentity, ...] = ()


class QcStageOutcome(StrictFrozenModel[Literal["qc_stage_outcome"]]):
    """One ordered QC pipeline stage outcome."""

    kind: Literal["qc_stage_outcome"] = "qc_stage_outcome"
    stage: str
    status: Literal["ok", "failed", "skipped"]
    output_artifact_ids: tuple[str, ...] = ()


class AssemblyQcRunManifest(StrictFrozenModel[Literal["assembly_qc_run_manifest"]]):
    """Stable, content-addressed manifest for one QC run."""

    kind: Literal["assembly_qc_run_manifest"] = "assembly_qc_run_manifest"
    schema_version: Literal["organelleverse.assembly-qc-run.v1"] = (
        "organelleverse.assembly-qc-run.v1"
    )
    operation_id: Literal["qc.assembly"] = "qc.assembly"
    policy_version: str
    source_assembly_result_id: str
    source_run_manifest_id: str
    parameters: FrozenMap[FrozenJson] = Field(default_factory=FrozenMap)
    environment_id: str = ""
    tool_identities: tuple[ToolIdentity, ...] = ()
    stages: tuple[QcStageOutcome, ...] = ()
    outputs: tuple[ArtifactRef, ...] = ()

    @field_validator("parameters", mode="before")
    @classmethod
    def _freeze_parameters(cls, value: object) -> FrozenMap[FrozenJson]:
        return _freeze_mapping(value, "parameters")

    @field_serializer("parameters")
    def _serialize_parameters(self, value: FrozenMap[FrozenJson]) -> object:
        return thaw_json(value)

    def semantic_payload(self) -> dict[str, object]:
        """Canonical, ordering-independent JSON identity payload."""
        return {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "policy_version": self.policy_version,
            "source_assembly_result_id": self.source_assembly_result_id,
            "source_run_manifest_id": self.source_run_manifest_id,
            "parameters": thaw_json(self.parameters),
            "environment_id": self.environment_id,
            "tool_identities": [
                item.model_dump(mode="json", exclude={"object_id"}) for item in self.tool_identities
            ],
            "stages": [item.model_dump(mode="json", exclude={"object_id"}) for item in self.stages],
            "outputs": [item.object_id for item in self.outputs],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())

    @property
    def run_manifest_id(self) -> str:
        digest = hashlib.sha256(self.canonical_bytes()).hexdigest()
        return f"assembly-qc-run:sha256:{digest}"

    def _identity_payload(self) -> dict[str, Any]:
        # The run manifest's canonical identity is its semantic payload; the
        # inherited object_id is derived from the same content.
        payload = self.semantic_payload()
        payload["kind"] = self.kind
        return payload


__all__ = [
    "AlternativeConfiguration",
    "AssemblyQcReport",
    "AssemblyQcRunManifest",
    "AssemblyStatistics",
    "CoverageAssessmentStatus",
    "CoverageWindow",
    "ErrorCandidate",
    "ErrorCandidateStatus",
    "GfaSummary",
    "GraphEvidence",
    "IdentityEvidenceStatus",
    "InputLibraryEvidence",
    "JunctionStatus",
    "JunctionSupport",
    "LibraryAlignmentEvidence",
    "LibraryMappingSummary",
    "MappingEvidence",
    "MarkerEvidence",
    "MarkerHit",
    "MarkerProfileAssessment",
    "MarkerProfileSet",
    "OrganelleEvidence",
    "OrganelleKind",
    "QcCheck",
    "QcCheckStatus",
    "QcDecision",
    "QcStageOutcome",
    "QcSummary",
    "RepeatStatus",
    "RepeatSupport",
    "ResolvedAssemblyEvidence",
    "SequenceDepthProfile",
    "SequenceIdentity",
    "SequenceSummary",
    "Strand",
    "ToolIdentity",
]
