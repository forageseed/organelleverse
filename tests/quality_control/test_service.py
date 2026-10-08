from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleDependencyError, OrganelleExecutionError
from organelleverse.quality_control.contracts import (
    AssemblyQcReport,
    AssemblyQcRunManifest,
    CoverageWindow,
    GfaSummary,
    GraphEvidence,
    JunctionSupport,
    MappingEvidence,
    MarkerEvidence,
    MarkerProfileSet,
    QcCheck,
)
from organelleverse.quality_control.environment import QcEnvironment, QcExecutable
from organelleverse.quality_control.service import run_assembly_qc

from .test_static import _publish_assembly_evidence


def _environment(tmp_path: Path) -> QcEnvironment:
    return QcEnvironment(
        environment_id="qc-environment:test",
        source="installed",
        minimap2=QcExecutable(
            name="minimap2",
            path=tmp_path / "minimap2",
            version="test",
            sha256="a" * 64,
            source="installed",
        ),
        samtools=QcExecutable(
            name="samtools",
            path=tmp_path / "samtools",
            version="test",
            sha256="b" * 64,
            source="installed",
        ),
    )


def _mapping(kmer_metrics: ArtifactRef | None = None) -> MappingEvidence:
    return MappingEvidence(
        coverage_windows=(
            CoverageWindow(
                sequence_id="ctg1",
                start=0,
                end=34,
                any_alignment_mean_depth=20.0,
                confident_mean_depth=10.0,
                assessment_status="assessed",
            ),
        ),
        checks=(
            QcCheck(
                check_id="qc.read_mapping",
                category="read_support",
                status="pass",
                value=2,
                unit="libraries",
                message="mapped",
            ),
            QcCheck(
                check_id="qc.kmer_evidence",
                category="base_accuracy",
                status="pass" if kmer_metrics is not None else "not_assessed",
                value=0.75 if kmer_metrics is not None else None,
                unit="fraction" if kmer_metrics is not None else "",
                message="assessed" if kmer_metrics is not None else "optional",
                evidence_artifact_ids=(
                    (kmer_metrics.object_id,) if kmer_metrics is not None else ()
                ),
            ),
        ),
        coordinate_artifacts=((kmer_metrics,) if kmer_metrics is not None else ()),
    )


def _graph() -> GraphEvidence:
    return GraphEvidence(
        gfa_summary=GfaSummary(
            segment_count=1,
            edge_count=1,
            path_count=1,
            component_count=1,
            branch_count=0,
            parallel_edge_count=0,
            tip_count=0,
            path_names=("ctg1",),
        ),
        junction_support=(
            JunctionSupport(
                sequence_id="ctg1",
                left_segment="ctg1+",
                right_segment="ctg1-",
                library_role="hifi",
                supporting_reads=5,
                contradicting_reads=0,
                status="supported",
                anchor_length=250,
            ),
        ),
        checks=(
            QcCheck(
                check_id="qc.junction_support",
                category="structure",
                status="pass",
                value=1,
                unit="junctions",
                message="supported",
            ),
        ),
    )


def _install_deterministic_collectors(monkeypatch, tmp_path: Path) -> None:
    import organelleverse.quality_control.service as service

    metrics_path = tmp_path / "kmer-metrics.tsv"
    metrics_path.write_text("metric\tvalue\nassembly_kmer_support_fraction\t0.75\n")
    metrics = ArtifactRef.from_path(
        metrics_path,
        kind="kmer_metrics",
        format="tsv",
        media_type="text/tab-separated-values",
    )
    any_coverage_path = tmp_path / "coverage-any.bed"
    any_coverage_path.write_text("")
    confident_coverage_path = tmp_path / "coverage-confident.bed"
    confident_coverage_path.write_text("")
    any_coverage = ArtifactRef.from_path(
        any_coverage_path,
        kind="coverage_intervals",
        format="bed",
        media_type="text/tab-separated-values",
    )
    confident_coverage = ArtifactRef.from_path(
        confident_coverage_path,
        kind="coverage_intervals_confident",
        format="bed",
        media_type="text/tab-separated-values",
    )
    mapping = _mapping(metrics).model_copy(
        update={"coordinate_artifacts": (metrics, any_coverage, confident_coverage)}
    )
    monkeypatch.setattr(service, "resolve_qc_environment", lambda: _environment(tmp_path))
    monkeypatch.setattr(service, "resolve_marker_profiles", lambda: (), raising=False)
    monkeypatch.setattr(
        service,
        "collect_mapping_evidence",
        lambda *args, **kwargs: mapping,
    )
    monkeypatch.setattr(service, "collect_graph_evidence", lambda *args, **kwargs: _graph())


def test_service_scans_resolved_marker_profiles_and_records_their_identity(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)
    _install_deterministic_collectors(monkeypatch, tmp_path)
    import organelleverse.quality_control.service as service

    profiles: list[MarkerProfileSet] = []
    for target in ("mitochondrion", "plastid"):
        path = tmp_path / f"{target}.fam"
        path.write_text(f"{target} profile\n")
        artifact = ArtifactRef.from_path(path, kind="hmm_profile", format="fam")
        profiles.append(
            MarkerProfileSet(
                profile_set_id="oatkdb-embryophyta",
                version="v20230921",
                hmm_artifact=artifact,
                sha256=artifact.sha256,
                target=target,
            )
        )
    captured: list[tuple[MarkerProfileSet, ...]] = []

    def collect_markers(evidence, profile_sets, policy):
        captured.append(profile_sets)
        return MarkerEvidence(
            profiles_assessed=True,
            profile_set_id="oatkdb-embryophyta",
            profile_version="v20230921",
            profile_sha256="d" * 64,
            expected_target_profile_count=1,
            checks=(
                QcCheck(
                    check_id="qc.target_marker_support",
                    category="identity",
                    status="pass",
                    value=1,
                    unit="marker_hits",
                    message="target marker detected",
                    evidence_artifact_ids=tuple(
                        item.hmm_artifact.object_id for item in profile_sets
                    ),
                ),
            ),
        )

    monkeypatch.setattr(service, "resolve_marker_profiles", lambda: tuple(profiles))
    monkeypatch.setattr(service, "collect_marker_evidence", collect_markers, raising=False)

    result = run_assembly_qc(published.result)

    assert captured == [tuple(profiles)]
    report_artifact = next(item for item in result.artifacts if item.kind == "assembly_qc_report")
    report = AssemblyQcReport.model_validate_json(Path(report_artifact.uri).read_bytes())
    assert report.marker_hits == ()
    assert report.marker_profile_assessment is not None
    assert report.marker_profile_assessment.profiles_assessed is True
    assert report.marker_profile_assessment.expected_target_profile_count == 1
    assert {item.sha256 for item in report.evidence_artifacts if item.kind == "hmm_profile"} == {
        item.sha256 for item in (profile.hmm_artifact for profile in profiles)
    }
    manifest_artifact = next(
        item for item in result.artifacts if item.kind == "assembly_qc_run_manifest"
    )
    manifest = AssemblyQcRunManifest.model_validate_json(Path(manifest_artifact.uri).read_bytes())
    marker_stage = next(item for item in manifest.stages if item.stage == "scan_markers")
    assert marker_stage.status == "ok"
    assert set(marker_stage.output_artifact_ids) == {
        profile.hmm_artifact.object_id for profile in profiles
    }


def test_unavailable_marker_profiles_remain_explicit_optional_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)
    _install_deterministic_collectors(monkeypatch, tmp_path)
    import organelleverse.quality_control.service as service

    def unavailable() -> tuple[MarkerProfileSet, ...]:
        raise OrganelleDependencyError(
            code="qc.marker_profiles_unavailable",
            message="network unreachable",
        )

    monkeypatch.setattr(service, "resolve_marker_profiles", unavailable)

    result = run_assembly_qc(published.result)

    report_artifact = next(item for item in result.artifacts if item.kind == "assembly_qc_report")
    report = AssemblyQcReport.model_validate_json(Path(report_artifact.uri).read_bytes())
    marker_check = next(item for item in report.checks if item.check_id == "qc.marker_profiles")
    assert marker_check.status == "not_assessed"
    assert report.marker_profile_assessment is not None
    assert report.marker_profile_assessment.profiles_assessed is False
    assert "qc.marker_profiles_unavailable: network unreachable" in marker_check.message
    manifest_artifact = next(
        item for item in result.artifacts if item.kind == "assembly_qc_run_manifest"
    )
    manifest = AssemblyQcRunManifest.model_validate_json(Path(manifest_artifact.uri).read_bytes())
    marker_stage = next(item for item in manifest.stages if item.stage == "scan_markers")
    assert marker_stage.status == "skipped"
    assert marker_stage.output_artifact_ids == ()


def test_service_builds_compact_canonical_result(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)
    _install_deterministic_collectors(monkeypatch, tmp_path)

    result = run_assembly_qc(
        published.result,
        threads=2,
        timeout_seconds=60,
    )

    assert result.operation_id == "qc.assembly"
    assert result.operation_version == "1.5"
    assert result.status == "warning"
    assert result.metrics["qc_decision"] == "insufficient_evidence"
    assert set(result.metrics) == {
        "qc_decision",
        "qc_pass_count",
        "qc_warning_count",
        "qc_failure_count",
        "qc_not_assessed_count",
        "any_alignment_coverage_breadth",
        "any_alignment_zero_coverage_bases",
        "confident_coverage_breadth",
        "confident_zero_coverage_bases",
        "unsupported_junction_count",
        "contradicted_junction_count",
        "structural_error_candidate_count",
    }
    assert {item.kind for item in result.artifacts} == {
        "assembly_qc_report",
        "assembly_qc_run_manifest",
    }
    assert result.provenance is not None
    assert result.provenance.input_object_ids == (published.result.object_id,)
    assert result.suggested_operations == ()


def test_zero_coverage_prevents_ready(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)
    _install_deterministic_collectors(monkeypatch, tmp_path)
    import organelleverse.quality_control.service as service

    mapping = _mapping().model_copy(
        update={
            "coverage_windows": (
                CoverageWindow(
                    sequence_id="ctg1",
                    start=0,
                    end=34,
                    any_alignment_mean_depth=0.0,
                    confident_mean_depth=0.0,
                    assessment_status="assessed",
                ),
            )
        }
    )
    monkeypatch.setattr(service, "collect_mapping_evidence", lambda *args, **kwargs: mapping)

    result = run_assembly_qc(published.result)

    assert result.status == "warning"
    assert result.metrics["qc_decision"] == "not_ready"
    assert any(item.code == "qc.zero_coverage_region" for item in result.findings)
    assert result.suggested_operations == ()


def test_confident_only_coverage_gap_warns_without_faking_an_assembly_gap(
    monkeypatch, tmp_path: Path
) -> None:
    # A repeat-like region: reads map (any-alignment breadth is full) but only
    # ambiguously, so the confident primary track has a zero gap. This must warn
    # for review, not fail as an assembly deletion.
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)
    _install_deterministic_collectors(monkeypatch, tmp_path)
    import organelleverse.quality_control.service as service

    mapping = _mapping().model_copy(
        update={
            "coverage_windows": (
                CoverageWindow(
                    sequence_id="ctg1",
                    start=0,
                    end=34,
                    any_alignment_mean_depth=20.0,
                    confident_mean_depth=0.0,
                    assessment_status="assessed",
                ),
            )
        }
    )
    monkeypatch.setattr(service, "collect_mapping_evidence", lambda *args, **kwargs: mapping)

    result = run_assembly_qc(published.result)

    assert result.metrics["qc_decision"] == "needs_review"
    assert not any(item.code == "qc.zero_coverage_region" for item in result.findings)
    assert any(item.code == "qc.confident_zero_coverage_region" for item in result.findings)
    assert result.metrics["any_alignment_zero_coverage_bases"] == 0
    assert result.metrics["confident_zero_coverage_bases"] == 34


def test_started_mapping_failure_returns_failed_result_with_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)
    _install_deterministic_collectors(monkeypatch, tmp_path)
    import organelleverse.quality_control.service as service

    def fail_mapping(*args, **kwargs):
        mapping_workspace = Path(kwargs["workspace"])
        mapping_workspace.mkdir()
        (mapping_workspace / "minimap2.stderr.log").write_text("backend failed\n")
        raise OrganelleExecutionError(
            code="qc.mapping_failed",
            message="mapping failed",
            details={"stage": "library_001.minimap2"},
        )

    monkeypatch.setattr(service, "collect_mapping_evidence", fail_mapping)

    result = run_assembly_qc(published.result)

    assert result.status == "failed"
    assert result.errors[0].code == "qc.mapping_failed"
    assert not any(item.kind == "assembly_qc_report" for item in result.artifacts)
    assert any(item.kind == "assembly_qc_run_manifest" for item in result.artifacts)
    log = next(item for item in result.artifacts if item.kind == "execution_log")
    assert Path(log.uri).read_text() == "backend failed\n"


@pytest.mark.skipif(
    shutil.which("minimap2") is None or shutil.which("samtools") is None,
    reason="tiny real service integration requires minimap2 and samtools",
)
def test_tiny_real_service_executes_shell_free_mapping_pipeline(
    monkeypatch, tmp_path: Path
) -> None:
    import organelleverse.quality_control.service as service

    monkeypatch.setattr(service, "resolve_marker_profiles", lambda: ())
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)

    result = run_assembly_qc(
        published.result,
        threads=1,
        timeout_seconds=60,
    )

    assert result.status == "warning"
    assert result.metrics["qc_decision"] == "not_ready"
    assert all(Path(item.uri).is_file() for item in result.artifacts)
    run_dir = Path(result.artifacts[0].uri).parent
    assert not tuple(run_dir.rglob("*.sam"))
    assert not tuple(run_dir.rglob("*.bam"))
    assert not tuple(run_dir.rglob("*.bai"))
