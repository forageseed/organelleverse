from __future__ import annotations

import pytest
from pydantic import ValidationError

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.quality_control.contracts import (
    AssemblyQcReport,
    AssemblyQcRunManifest,
    AssemblyStatistics,
    CoverageWindow,
    GfaSummary,
    InputLibraryEvidence,
    LibraryMappingSummary,
    QcCheck,
    QcSummary,
    RepeatSupport,
    SequenceSummary,
)
from organelleverse.quality_control.policy import QC_POLICY_V5, QcPolicy


def test_qc_models_are_closed_and_strict() -> None:
    with pytest.raises(ValidationError):
        QcCheck(
            check_id="sequence.valid",
            category="integrity",
            status="pass",
            value=1,
            unit="records",
            message="valid",
            evidence_artifact_ids=(),
            surprise=True,
        )
    with pytest.raises(ValidationError):
        QcSummary(
            pass_count="1",
            warning_count=0,
            failure_count=0,
            not_assessed_count=0,
        )


def test_sequence_identity_status_is_closed() -> None:
    with pytest.raises(ValidationError):
        SequenceSummary(
            sequence_id="ctg1",
            role="primary",
            length=100,
            gc_fraction=0.5,
            ambiguous_bases=0,
            identity_evidence_status="contaminant",
        )


def test_qc_check_status_is_closed() -> None:
    with pytest.raises(ValidationError):
        QcCheck(check_id="x", category="c", status="excellent", message="m")


def test_qc_summary_bounds_reject_negatives() -> None:
    with pytest.raises(ValidationError):
        QcSummary(pass_count=-1, warning_count=0, failure_count=0, not_assessed_count=0)


def test_qc_summary_fraction_bounds_reject_out_of_range() -> None:
    with pytest.raises(ValidationError):
        QcSummary(
            pass_count=0,
            warning_count=0,
            failure_count=0,
            not_assessed_count=0,
            any_alignment_coverage_breadth=1.5,
        )


def test_valid_qc_check_and_summary_round_trip() -> None:
    check = QcCheck(
        check_id="structure.junction_support",
        category="structural_accuracy",
        status="pass",
        value=12,
        unit="junctions",
        message="All reported junctions have read support.",
        evidence_artifact_ids=("artifact:sha256:abc",),
    )
    summary = QcSummary(
        pass_count=20,
        warning_count=0,
        failure_count=0,
        not_assessed_count=1,
        any_alignment_coverage_breadth=1.0,
    )
    assert check.status == "pass"
    assert summary.any_alignment_coverage_breadth == 1.0
    # frozen + strict: no public dict surface, identity is content-derived
    assert check.object_id.startswith("qc_check:sha256:")
    assert summary.object_id.startswith("qc_summary:sha256:")


def test_sequence_summary_requires_positive_length() -> None:
    with pytest.raises(ValidationError):
        SequenceSummary(sequence_id="ctg1", role="primary", length=0, gc_fraction=0.5)


def test_policy_v5_has_the_pinned_thresholds() -> None:
    assert QC_POLICY_V5.policy_version == "organelleverse.assembly-qc-policy.v5"
    assert QC_POLICY_V5.minimum_mapping_quality == 20
    assert QC_POLICY_V5.coverage_depth_levels == (1, 5, 10)
    assert QC_POLICY_V5.minimum_junction_support == 3
    assert QC_POLICY_V5.minimum_base_error_support == 3
    assert QC_POLICY_V5.minimum_base_error_depth == 5
    assert QC_POLICY_V5.minimum_base_error_fraction == 0.8
    assert QC_POLICY_V5.marker_evalue == 1e-5
    assert isinstance(QC_POLICY_V5, QcPolicy)


def test_report_rejects_free_form_mapping_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AssemblyQcReport(
            schema_version="organelleverse.assembly-qc.v1",
            policy_version="organelleverse.assembly-qc-policy.v5",
            source_assembly_result_id="result:sha256:" + "0" * 64,
            source_run_manifest_id="assembly-run:sha256:" + "0" * 64,
            organelle="mitochondrion",
            decision="ready",
            summary=QcSummary(pass_count=1, warning_count=0, failure_count=0, not_assessed_count=0),
            checks=(),
            sequence_summaries=(),
            evidence_artifacts=(),
            tool_identities=(),
            custom_metric={"qc_score": 87},
        )


def test_run_manifest_round_trips_with_canonical_identity() -> None:
    manifest = AssemblyQcRunManifest(
        schema_version="organelleverse.assembly-qc-run.v1",
        operation_id="qc.assembly",
        policy_version="organelleverse.assembly-qc-policy.v5",
        source_assembly_result_id="result:sha256:" + "0" * 64,
        source_run_manifest_id="assembly-run:sha256:" + "0" * 64,
        parameters={"threads": 4},
        environment_id="environment:sha256:" + "0" * 64,
        stages=(),
        outputs=(),
    )
    restored = AssemblyQcRunManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest
    assert restored.run_manifest_id == manifest.run_manifest_id
    assert restored.run_manifest_id.startswith("assembly-qc-run:sha256:")


# -- strict scalar fields: reject string coercion and NaN/Inf ----------------


def test_summary_rejects_string_coercion_for_floats_and_ints() -> None:
    with pytest.raises(ValidationError):
        QcSummary(
            pass_count=1,
            warning_count=0,
            failure_count=0,
            not_assessed_count=0,
            any_alignment_coverage_breadth="0.5",
        )
    with pytest.raises(ValidationError):
        QcSummary(
            pass_count=1,
            warning_count=0,
            failure_count=0,
            not_assessed_count=0,
            any_alignment_zero_coverage_bases="3",
        )


# -- Headline honesty: no disguised assembly QV ------------------------------
#
# The reads-vs-assembly mismatch/indel rate is NOT a consensus assembly QV
# (Merqury QV): it blends sequencing error, mapping error, heterogeneity, and
# assembly error. The per-library value is therefore reported under its honest
# name ``read_assembly_agreement_qv`` alongside the declared technology and
# quality_state. The report summary carries no cross-library ``mapping_qv`` at
# all, because reducing heterogeneous libraries to one comparable QV is not
# honest.


def test_qc_summary_has_no_mapping_qv_field() -> None:
    summary = QcSummary(pass_count=1, warning_count=0, failure_count=0, not_assessed_count=0)
    assert not hasattr(summary, "mapping_qv")
    with pytest.raises(ValidationError):
        QcSummary.model_validate(
            {
                "pass_count": 1,
                "warning_count": 0,
                "failure_count": 0,
                "not_assessed_count": 0,
                "mapping_qv": 42.0,
            }
        )


def test_library_mapping_summary_reports_agreement_qv_with_technology() -> None:
    summary = LibraryMappingSummary(
        library_role="hifi",
        technology="pacbio_hifi",
        quality_state="ccs",
        mapped_reads=100,
        read_assembly_agreement_qv=30.0,
    )
    assert summary.read_assembly_agreement_qv == 30.0
    assert summary.technology == "pacbio_hifi"
    assert summary.quality_state == "ccs"
    assert not hasattr(summary, "consensus_qv")


# -- Coverage evidence: two explicit, non-interchangeable tracks ----------------
#
# Secondary/supplementary/low-MAPQ alignments contribute to a repeat-aware
# ``any_alignment`` coverage breadth (a real assembly gap can fail on it) but
# not to a ``confident`` primary, MAPQ-filtered breadth. A region covered only
# by ambiguous alignments is repeat-like evidence, not a confident gap, so a
# confident-only gap warns rather than masquerading as an assembly deletion.


def test_coverage_window_carries_two_explicit_depth_tracks() -> None:
    window = CoverageWindow(
        sequence_id="ctg1",
        start=0,
        end=1_000,
        any_alignment_mean_depth=12.0,
        confident_mean_depth=4.0,
        assessment_status="assessed",
    )
    assert window.any_alignment_mean_depth == 12.0
    assert window.confident_mean_depth == 4.0
    assert not hasattr(window, "mean_depth")
    assert not hasattr(window, "minimum_depth")


def test_coverage_window_reports_per_track_minimum_depths() -> None:
    # Each track reports its own minimum so a window mean is never used to guess
    # precise zero bases; the precise zero count comes from the BED artifacts.
    window = CoverageWindow(
        sequence_id="ctg1",
        start=0,
        end=1_000,
        any_alignment_mean_depth=12.0,
        any_alignment_minimum_depth=2.0,
        confident_mean_depth=4.0,
        confident_minimum_depth=0.0,
        assessment_status="assessed",
    )
    assert window.any_alignment_minimum_depth == 2.0
    assert window.confident_minimum_depth == 0.0


def test_assembly_qc_report_exposes_full_evidence_surface() -> None:
    report = AssemblyQcReport(
        policy_version="organelleverse.assembly-qc-policy.v5",
        source_assembly_result_id="result:sha256:" + "0" * 64,
        source_run_manifest_id="assembly-run:sha256:" + "0" * 64,
        organelle="mitochondrion",
        decision="ready",
        summary=QcSummary(pass_count=1, warning_count=0, failure_count=0, not_assessed_count=0),
    )
    # The service must not discard per-library mappings, graph topology, or
    # alternative configurations; they are first-class report fields.
    assert report.library_mapping_summaries == ()
    assert report.gfa_summary is None
    assert report.alternative_configurations == ()
    assert report.assembly_statistics is None


def test_assembly_statistics_is_a_descriptive_model() -> None:
    stats = AssemblyStatistics(
        sequence_count=3,
        total_length=600,
        largest_sequence_length=300,
        n50=300,
        l50=1,
        overall_gc_fraction=0.4,
        ambiguous_bases=2,
    )
    assert stats.n50 == 300
    assert stats.l50 == 1
    # Descriptive only: it carries no assessment_status and no pass/fail signal.
    assert not hasattr(stats, "status")


def test_qc_summary_reports_two_non_interchangeable_coverage_tracks() -> None:
    summary = QcSummary(
        pass_count=1,
        warning_count=0,
        failure_count=0,
        not_assessed_count=0,
        any_alignment_coverage_breadth=1.0,
        any_alignment_zero_coverage_bases=0,
        confident_coverage_breadth=0.9,
        confident_zero_coverage_bases=1_000,
    )
    assert summary.any_alignment_coverage_breadth == 1.0
    assert summary.confident_coverage_breadth == 0.9
    assert not hasattr(summary, "coverage_breadth")
    assert not hasattr(summary, "zero_coverage_bases")


def test_summary_rejects_nan_and_inf_floats() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError):
            QcSummary(
                pass_count=1,
                warning_count=0,
                failure_count=0,
                not_assessed_count=0,
                any_alignment_coverage_breadth=bad,
            )


def test_sequence_summary_rejects_numeric_string_and_non_finite() -> None:
    with pytest.raises(ValidationError):
        SequenceSummary(
            sequence_id="c", role="primary", length="100", gc_fraction=0.5, ambiguous_bases=0
        )
    with pytest.raises(ValidationError):
        SequenceSummary(
            sequence_id="c", role="primary", length=100, gc_fraction="0.5", ambiguous_bases=0
        )
    with pytest.raises(ValidationError):
        SequenceSummary(
            sequence_id="c",
            role="primary",
            length=100,
            gc_fraction=0.5,
            ambiguous_bases=0,
            median_depth=float("inf"),
        )


def test_policy_rejects_string_coercion_and_non_finite_evalue() -> None:
    with pytest.raises(ValidationError):
        QcPolicy(
            policy_version="x",
            minimum_mapping_quality="20",
            minimum_base_quality=20,
            coverage_depth_levels=(1, 5, 10),
            coverage_window_bases=1_000,
            minimum_junction_support=3,
            minimum_contradiction_support=3,
            minimum_anchor_bases=250,
            maximum_anchor_bases=5_000,
            error_cluster_window_bases=50,
            marker_evalue=1e-5,
            kmer_size=21,
            log_capture_limit_bytes=1_048_576,
        )
    with pytest.raises(ValidationError):
        QcPolicy(
            policy_version="x",
            minimum_mapping_quality=20,
            minimum_base_quality=20,
            coverage_depth_levels=(1, 5, 10),
            coverage_window_bases=1_000,
            minimum_junction_support=3,
            minimum_contradiction_support=3,
            minimum_anchor_bases=250,
            maximum_anchor_bases=5_000,
            error_cluster_window_bases=50,
            marker_evalue=float("nan"),
            kmer_size=21,
            log_capture_limit_bytes=1_048_576,
        )
    with pytest.raises(ValidationError):
        QcPolicy(
            policy_version="x",
            minimum_mapping_quality=20,
            minimum_base_quality=20,
            coverage_depth_levels=("1", "5"),
            coverage_window_bases=1_000,
            minimum_junction_support=3,
            minimum_contradiction_support=3,
            minimum_anchor_bases=250,
            maximum_anchor_bases=5_000,
            error_cluster_window_bases=50,
            marker_evalue=1e-5,
            kmer_size=21,
            log_capture_limit_bytes=1_048_576,
        )


def test_strict_scalars_preserve_json_round_trip() -> None:
    summary = QcSummary(
        pass_count=3,
        warning_count=1,
        failure_count=0,
        not_assessed_count=2,
        any_alignment_coverage_breadth=0.875,
        any_alignment_zero_coverage_bases=12,
    )
    restored = QcSummary.model_validate_json(summary.model_dump_json())
    assert restored == summary
    assert restored.any_alignment_coverage_breadth == 0.875


# -- InputLibraryEvidence: closed coherent library contract ------------------


def _ref(role: str) -> ArtifactRef:
    return ArtifactRef(
        kind="read",
        uri=f"{role}.fastq",
        format="fastq",
        media_type="application/x-fastq",
        sha256="a" * 64,
        size_bytes=8,
    )


@pytest.mark.parametrize(
    "library",
    [
        InputLibraryEvidence(
            role="pe",
            technology="illumina",
            layout="paired_end",
            read1=_ref("r1"),
            read2=_ref("r2"),
            read_length=150,
            insert_size=300,
        ),
        InputLibraryEvidence(
            role="se",
            technology="illumina",
            layout="single_end",
            read1=_ref("se"),
            read_length=150,
        ),
        InputLibraryEvidence(
            role="hifi",
            technology="pacbio_hifi",
            quality_state="ccs",
            reads=_ref("hifi"),
        ),
        InputLibraryEvidence(
            role="clr",
            technology="pacbio_clr",
            quality_state="corrected",
            reads=_ref("clr"),
        ),
        InputLibraryEvidence(
            role="ont",
            technology="ont",
            quality_state="hq",
            reads=_ref("ont"),
        ),
    ],
)
def test_input_library_valid_combinations_round_trip(library: InputLibraryEvidence) -> None:
    restored = InputLibraryEvidence.model_validate_json(library.model_dump_json())
    assert restored == library


@pytest.mark.parametrize(
    "payload",
    [
        # illumina must declare a short-read layout
        {"role": "x", "technology": "illumina", "read1": _ref("r1")},
        # paired_end requires read2
        {"role": "x", "technology": "illumina", "layout": "paired_end", "read1": _ref("r1")},
        # single_end forbids read2
        {
            "role": "x",
            "technology": "illumina",
            "layout": "single_end",
            "read1": _ref("r1"),
            "read2": _ref("r2"),
        },
        # illumina carries no long reads and no quality state
        {
            "role": "x",
            "technology": "illumina",
            "layout": "single_end",
            "read1": _ref("r1"),
            "reads": _ref("r1"),
        },
        {
            "role": "x",
            "technology": "illumina",
            "layout": "single_end",
            "read1": _ref("r1"),
            "quality_state": "ccs",
        },
        # long-read technologies forbid short-read layout/read1/read2
        {
            "role": "x",
            "technology": "pacbio_hifi",
            "quality_state": "ccs",
            "reads": _ref("r"),
            "layout": "paired_end",
        },
        {
            "role": "x",
            "technology": "ont",
            "quality_state": "hq",
            "reads": _ref("r"),
            "read1": _ref("r1"),
        },
        # invalid technology-quality_state pair
        {"role": "x", "technology": "pacbio_hifi", "quality_state": "raw", "reads": _ref("r")},
    ],
)
def test_input_library_rejects_incoherent_combinations(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        InputLibraryEvidence.model_validate(payload)


def test_input_library_rejects_unknown_technology() -> None:
    with pytest.raises(ValidationError):
        InputLibraryEvidence(
            role="x",
            technology="nanopore",  # not a released technology
            quality_state="hq",
            reads=_ref("r"),
        )


# -- QcCheck.value: strict scalar variants, NaN/Inf rejected -----------------


@pytest.mark.parametrize("value", [3, 3.5, "pass", True, None])
def test_qc_check_value_accepts_legitimate_scalars(value: object) -> None:
    check = QcCheck(check_id="c", category="cat", status="pass", value=value, message="m")
    assert check.value == value


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_qc_check_value_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValidationError):
        QcCheck(check_id="c", category="cat", status="pass", value=value, message="m")


def test_qc_check_value_round_trips_through_json() -> None:
    check = QcCheck(check_id="c", category="cat", status="pass", value=12, message="m")
    restored = QcCheck.model_validate_json(check.model_dump_json())
    assert restored == check and restored.value == 12


def test_qc_check_value_rejects_bytes() -> None:
    with pytest.raises(ValidationError):
        QcCheck(check_id="c", category="cat", status="pass", value=b"yes", message="m")


# -- InputLibraryEvidence: illumina read_length / insert_size contract --------


def test_input_library_illumina_requires_positive_read_length() -> None:
    # missing read_length
    with pytest.raises(ValidationError):
        InputLibraryEvidence.model_validate(
            {"role": "x", "technology": "illumina", "layout": "single_end", "read1": _ref("r1")}
        )
    # zero read_length
    with pytest.raises(ValidationError):
        InputLibraryEvidence.model_validate(
            {
                "role": "x",
                "technology": "illumina",
                "layout": "single_end",
                "read1": _ref("r1"),
                "read_length": 0,
            }
        )


def test_input_library_illumina_insert_size_when_present_is_positive() -> None:
    with pytest.raises(ValidationError):
        InputLibraryEvidence.model_validate(
            {
                "role": "x",
                "technology": "illumina",
                "layout": "paired_end",
                "read1": _ref("r1"),
                "read2": _ref("r2"),
                "read_length": 150,
                "insert_size": 0,
            }
        )


def test_input_library_illumina_single_end_allows_absent_insert_size() -> None:
    library = InputLibraryEvidence.model_validate(
        {
            "role": "x",
            "technology": "illumina",
            "layout": "single_end",
            "read1": _ref("r1"),
            "read_length": 150,
        }
    )
    assert library.insert_size is None
    assert library.read_length == 150


# -- Graph/repeat metric honesty: parallel edges and path occurrences --------
#
# ``bubble_count`` previously counted only parallel edges (a pair of segments
# joined by more than one link) but implied an assembly-graph bubble/superbubble
# (a defined structure). It is renamed to the honest ``parallel_edge_count``.
# ``RepeatSupport.inferred_copies`` was the GFA path occurrence count, not a
# copy-number inference; it is renamed ``path_occurrences``. The repeat
# ``median_depth`` was the largest genome-wide library median, not a
# repeat-local depth, so it is removed entirely.


def test_gfa_summary_counts_parallel_edges_under_an_honest_name() -> None:
    summary = GfaSummary(
        segment_count=2,
        edge_count=3,
        path_count=1,
        component_count=1,
        branch_count=0,
        parallel_edge_count=1,
        tip_count=0,
        path_names=("ctg1",),
    )
    assert summary.parallel_edge_count == 1
    assert not hasattr(summary, "bubble_count")


def test_repeat_support_records_path_occurrences_without_local_depth() -> None:
    repeat = RepeatSupport(
        repeat_id="IR",
        sequence_id="ctg1",
        length=25_000,
        orientation="+,-",
        path_occurrences=2,
        spanning_reads=12,
        status="resolved",
    )
    assert repeat.path_occurrences == 2
    assert not hasattr(repeat, "inferred_copies")
    assert not hasattr(repeat, "median_depth")


def test_repeat_support_rejects_legacy_fields() -> None:
    with pytest.raises(ValidationError):
        RepeatSupport.model_validate(
            {
                "repeat_id": "IR",
                "sequence_id": "ctg1",
                "length": 25_000,
                "path_occurrences": 2,
                "spanning_reads": 12,
                # legacy field must be rejected by the closed contract
                "inferred_copies": 2,
            }
        )
    with pytest.raises(ValidationError):
        RepeatSupport.model_validate(
            {
                "repeat_id": "IR",
                "sequence_id": "ctg1",
                "length": 25_000,
                "path_occurrences": 2,
                "spanning_reads": 12,
                "median_depth": 30.0,
            }
        )
