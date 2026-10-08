"""Decision/finding ladder for assembly QC error injection (design §13.2).

Junction-level evidence shape (supported / unsupported / contradicted / repeat
copy + spanning reads) is unit-locked in ``test_graph_evidence.py`` and
circular-origin rotation invariance in ``test_circular_origin.py``. This module
asserts the missing layer: how that evidence flows through the full service into
a scientific decision, finding code, and metric.

Marker-dependent identity scenarios (conflicting marker, MTPT) are isolated at
the ``interpret_organelle`` contract seam so each structural-context branch can
be injected deterministically without scanning the release-pinned OatkDB
profiles. End-to-end profile resolution and scanning are covered separately by
the service and marker-profile tests.
"""

from __future__ import annotations

from pathlib import Path

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.genome import OrganelleGenome
from organelleverse.quality_control.contracts import (
    CoverageWindow,
    ErrorCandidate,
    GfaSummary,
    GraphEvidence,
    JunctionSupport,
    MappingEvidence,
    MarkerEvidence,
    MarkerHit,
    QcCheck,
    ResolvedAssemblyEvidence,
)
from organelleverse.quality_control.organelle import interpret_organelle
from organelleverse.quality_control.policy import QC_POLICY_V5
from organelleverse.quality_control.service import aggregate_decision, run_assembly_qc

from .test_service import _install_deterministic_collectors
from .test_static import _publish_assembly_evidence

# ---------------------------------------------------------------------------
# Full-service injection: a contradicted junction must block release end-to-end.
# ---------------------------------------------------------------------------


def _contradicted_graph() -> GraphEvidence:
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
                supporting_reads=0,
                contradicting_reads=3,
                status="contradicted",
                anchor_length=250,
            ),
        ),
        structural_error_candidates=(
            ErrorCandidate(
                candidate_id="ctg1.junction",
                sequence_id="ctg1",
                start=0,
                end=34,
                error_type="contradicted_junction",
            ),
        ),
        checks=(
            QcCheck(
                check_id="qc.graph_topology",
                category="structure",
                status="pass",
                value=1,
                unit="components",
                message="GFA topology parsed.",
            ),
            QcCheck(
                check_id="qc.junction_support",
                category="structure",
                status="fail",
                value=0,
                unit="supported_adjacencies",
                message="A declared adjacency has contradictory read evidence.",
                finding_code="qc.contradicted_junction",
            ),
        ),
    )


def test_contradicted_junction_blocks_release_end_to_end(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)
    _install_deterministic_collectors(monkeypatch, tmp_path)
    import organelleverse.quality_control.service as service

    monkeypatch.setattr(service, "collect_graph_evidence", lambda *a, **kw: _contradicted_graph())

    result = run_assembly_qc(published.result)

    assert result.status == "warning"
    assert result.metrics["qc_decision"] == "not_ready"
    assert any(item.code == "qc.contradicted_junction" for item in result.findings)
    assert result.metrics["contradicted_junction_count"] == 1
    assert result.metrics["structural_error_candidate_count"] == 1
    assert result.suggested_operations == ()


# ---------------------------------------------------------------------------
# Marker identity decision ladder at the interpret_organelle contract seam.
# ---------------------------------------------------------------------------


def _resolved(tmp_path: Path, fasta: str, organelle: str) -> ResolvedAssemblyEvidence:
    fasta_path = tmp_path / "assembly.fa"
    fasta_path.write_text(fasta)
    genome = OrganelleGenome(
        organelle=organelle,
        sequence=ArtifactRef.from_path(fasta_path, kind="sequence", format="fasta"),
    )
    return ResolvedAssemblyEvidence.model_construct(
        kind="resolved_assembly_evidence",
        source_result=None,
        run_manifest=None,
        primary_genome=genome,
        primary_fasta_path=fasta_path,
        assembly_graph_path=None,
        input_libraries=(),
    )


def _coverage(*sequence_ids: str) -> MappingEvidence:
    return MappingEvidence(
        coverage_windows=tuple(
            CoverageWindow(
                sequence_id=sid,
                start=0,
                end=10,
                any_alignment_mean_depth=20.0,
                any_alignment_minimum_depth=10.0,
                confident_mean_depth=20.0,
                confident_minimum_depth=10.0,
                assessment_status="assessed",
            )
            for sid in sequence_ids
        )
    )


def _supported_junction(sequence_id: str, left: str, right: str) -> JunctionSupport:
    return JunctionSupport(
        sequence_id=sequence_id,
        left_segment=left,
        right_segment=right,
        library_role="hifi",
        supporting_reads=5,
        contradicting_reads=0,
        status="supported",
        anchor_length=250,
    )


def _graph(junctions: tuple[JunctionSupport, ...], *, components: int = 1) -> GraphEvidence:
    names = sorted({j.sequence_id for j in junctions})
    return GraphEvidence(
        gfa_summary=GfaSummary(
            segment_count=len(names) or 1,
            edge_count=len(junctions),
            path_count=len(names) or 1,
            component_count=components,
            branch_count=0,
            parallel_edge_count=0,
            tip_count=0,
            path_names=tuple(names),
        ),
        junction_support=junctions,
    )


def _decision(
    evidence: ResolvedAssemblyEvidence,
    mapping: MappingEvidence,
    graph: GraphEvidence,
    markers: MarkerEvidence,
) -> tuple[str, set[str], list[str], list[str]]:
    org = interpret_organelle(evidence, mapping, graph, markers, QC_POLICY_V5)
    decision = aggregate_decision(
        org.checks,
        required_evidence_missing=not markers.profiles_assessed and not markers.marker_hits,
    )
    codes = {check.finding_code for check in org.checks if check.finding_code}
    statuses = [str(check.status) for check in org.checks]
    identities = [si.identity_evidence_status for si in org.sequence_identities]
    return decision, codes, statuses, identities


def test_supported_context_without_marker_is_insufficient_identity_evidence(
    tmp_path: Path,
) -> None:
    evidence = _resolved(tmp_path, ">ctg1\nACGTACGTACGT\n", "mitochondrion")
    graph = _graph((_supported_junction("ctg1", "ctg1+", "ctg1-"),))

    decision, codes, _, identities = _decision(evidence, _coverage("ctg1"), graph, MarkerEvidence())

    assert identities == ["unresolved"]
    assert decision == "insufficient_evidence"
    assert codes == set()


def test_unresolved_per_sequence_identity_warns(tmp_path: Path) -> None:
    evidence = _resolved(tmp_path, ">ctg1\nACGTACGT\n>ctg2\nACGTACGT\n", "mitochondrion")
    # ctg1 has supported boundary context; ctg2 is read-supported but has none.
    graph = _graph((_supported_junction("ctg1", "ctg1+", "ctg1-"),))

    decision, codes, _, identities = _decision(
        evidence, _coverage("ctg1", "ctg2"), graph, MarkerEvidence()
    )

    assert identities == ["unresolved", "unresolved"]
    assert "qc.target_identity_unresolved" not in codes
    assert decision == "insufficient_evidence"


def test_unintegrated_conflicting_marker_fails_identity(tmp_path: Path) -> None:
    evidence = _resolved(tmp_path, ">ctg1\nACGTACGT\n", "mitochondrion")
    graph = _graph((_supported_junction("ctg1", "ctg1+", "ctg1-"),))
    # A plastid marker hit on a mitochondrial contig with only one supported
    # boundary is not a boundary-integrated MTPT, so it remains a conflict.
    markers = MarkerEvidence(
        marker_hits=(
            MarkerHit(
                profile_id="plastid:rbcL", sequence_id="ctg1", start=0, end=6, target="plastid"
            ),
        )
    )

    decision, codes, _, _ = _decision(evidence, _coverage("ctg1"), graph, markers)

    assert "qc.conflicting_organelle_marker" in codes
    assert decision == "not_ready"


def test_boundary_supported_mtpt_does_not_block_ready(tmp_path: Path) -> None:
    evidence = _resolved(tmp_path, ">ctg1\nACGTACGT\n", "mitochondrion")
    # Two distinct supported boundaries on ctg1 make the plastid-like tract a
    # boundary-integrated MTPT rather than conflicting contamination.
    graph = _graph(
        (
            _supported_junction("ctg1", "ctg1+", "ctg1-"),
            _supported_junction("ctg1", "ctg1+", "ctg2+"),
        )
    )
    markers = MarkerEvidence(
        marker_hits=(
            MarkerHit(
                profile_id="plastid:rbcL", sequence_id="ctg1", start=0, end=6, target="plastid"
            ),
        )
    )

    decision, _, _, identities = _decision(evidence, _coverage("ctg1"), graph, markers)

    assert identities == ["context_supported"]
    assert decision == "ready"
