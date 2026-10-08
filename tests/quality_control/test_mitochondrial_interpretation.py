from __future__ import annotations

from organelleverse.quality_control.contracts import (
    GfaSummary,
    GraphEvidence,
    JunctionSupport,
    MappingEvidence,
    MarkerEvidence,
    MarkerHit,
)
from organelleverse.quality_control.organelle import interpret_organelle
from organelleverse.quality_control.policy import QC_POLICY_V5

from .test_plastid_interpretation import _mapping, _resolved


def _mitochondrial_graph(*, supported_boundaries: int) -> GraphEvidence:
    return GraphEvidence(
        gfa_summary=GfaSummary(
            segment_count=5,
            edge_count=4,
            path_count=2,
            component_count=2,
            branch_count=1,
            parallel_edge_count=0,
            tip_count=2,
            path_names=("ctg", "minor"),
        ),
        junction_support=tuple(
            JunctionSupport(
                sequence_id="ctg",
                left_segment=f"S{index}+",
                right_segment=f"S{index + 1}+",
                library_role="hifi",
                supporting_reads=5,
                contradicting_reads=0,
                status="supported",
                anchor_length=250,
            )
            for index in range(supported_boundaries)
        ),
    )


def test_assembly_wide_identity_without_support_is_not_ready_input(tmp_path) -> None:
    interpreted = interpret_organelle(
        _resolved(tmp_path, organelle="mitochondrion"),
        MappingEvidence(),
        GraphEvidence(),
        MarkerEvidence(),
        QC_POLICY_V5,
    )

    assert interpreted.sequence_identities[0].identity_evidence_status == "unresolved"
    identity_check = next(
        item for item in interpreted.checks if item.check_id == "qc.target_identity"
    )
    assert identity_check.status == "not_assessed"
    assert identity_check.finding_code == ""


def test_boundary_supported_plastid_marker_is_mtpt_context_not_contamination(
    tmp_path,
) -> None:
    markers = MarkerEvidence(
        marker_hits=(
            MarkerHit(
                profile_id="plastid:rbcL",
                sequence_id="ctg",
                start=200,
                end=500,
                complete=True,
                target="plastid",
            ),
        )
    )

    interpreted = interpret_organelle(
        _resolved(tmp_path, organelle="mitochondrion"),
        _mapping(),
        _mitochondrial_graph(supported_boundaries=2),
        markers,
        QC_POLICY_V5,
    )

    assert interpreted.sequence_identities[0].identity_evidence_status == "context_supported"
    assert not any(item.status == "fail" for item in interpreted.checks)
    mtpt = next(item for item in interpreted.checks if item.check_id == "qc.mtpt_context")
    assert mtpt.status == "pass"


def test_duplicate_library_support_for_one_boundary_does_not_claim_mtpt_context(
    tmp_path,
) -> None:
    markers = MarkerEvidence(
        marker_hits=(
            MarkerHit(
                profile_id="plastid:rbcL",
                sequence_id="ctg",
                start=200,
                end=500,
                complete=True,
                target="plastid",
            ),
        )
    )
    graph = _mitochondrial_graph(supported_boundaries=1)
    duplicate = graph.junction_support[0].model_copy(update={"library_role": "ont"})
    graph = graph.model_copy(update={"junction_support": (*graph.junction_support, duplicate)})

    interpreted = interpret_organelle(
        _resolved(tmp_path, organelle="mitochondrion"),
        _mapping(),
        graph,
        markers,
        QC_POLICY_V5,
    )

    assert interpreted.sequence_identities[0].identity_evidence_status == "unresolved"
    assert any(item.status == "fail" for item in interpreted.checks)


def test_read_supported_target_sequence_with_plastid_marker_is_probable_mtpt(
    tmp_path,
) -> None:
    markers = MarkerEvidence(
        marker_hits=(
            MarkerHit(
                profile_id="mitochondrion:atp1",
                sequence_id="ctg",
                start=50,
                end=150,
                complete=True,
                target="mitochondrion",
            ),
            MarkerHit(
                profile_id="plastid:rbcL",
                sequence_id="ctg",
                start=200,
                end=500,
                complete=True,
                target="plastid",
            ),
        )
    )

    interpreted = interpret_organelle(
        _resolved(tmp_path, organelle="mitochondrion"),
        _mapping(),
        GraphEvidence(),
        markers,
        QC_POLICY_V5,
    )

    conflict = next(
        item for item in interpreted.checks if item.check_id == "qc.conflicting_organelle_marker"
    )
    mtpt = next(item for item in interpreted.checks if item.check_id == "qc.mtpt_context")
    assert interpreted.sequence_identities[0].identity_evidence_status == "target_supported"
    assert conflict.status == "pass"
    assert mtpt.status == "warn"
    assert mtpt.value == 1


def test_multipartite_branched_mitochondrial_graph_is_not_penalized(tmp_path) -> None:
    interpreted = interpret_organelle(
        _resolved(tmp_path, organelle="mitochondrion"),
        _mapping(),
        _mitochondrial_graph(supported_boundaries=1),
        MarkerEvidence(),
        QC_POLICY_V5,
    )

    structure = next(
        item for item in interpreted.checks if item.check_id == "qc.mitochondrial_structure"
    )
    assert structure.status == "pass"
    assert "circle" not in structure.message.lower()
