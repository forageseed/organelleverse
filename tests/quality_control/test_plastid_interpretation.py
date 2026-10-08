from __future__ import annotations

from pathlib import Path

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.genome import OrganelleGenome
from organelleverse.quality_control.contracts import (
    CoverageWindow,
    GfaSummary,
    GraphEvidence,
    JunctionSupport,
    MappingEvidence,
    MarkerEvidence,
    MarkerHit,
    RepeatSupport,
    ResolvedAssemblyEvidence,
)
from organelleverse.quality_control.organelle import interpret_organelle
from organelleverse.quality_control.policy import QC_POLICY_V5


def _resolved(tmp_path: Path, *, organelle: str = "plastid") -> ResolvedAssemblyEvidence:
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(">ctg\n" + "ACGT" * 250 + "\n")
    artifact = ArtifactRef.from_path(fasta, kind="sequence", format="fasta")
    genome = OrganelleGenome(organelle=organelle, sequence=artifact)  # type: ignore[arg-type]
    return ResolvedAssemblyEvidence.model_construct(
        kind="resolved_assembly_evidence",
        source_result=None,
        run_manifest=None,
        primary_genome=genome,
        primary_fasta_path=fasta,
        assembly_graph_path=None,
        input_libraries=(),
    )


def _mapping() -> MappingEvidence:
    return MappingEvidence(
        coverage_windows=(
            CoverageWindow(
                sequence_id="ctg",
                start=0,
                end=1_000,
                any_alignment_mean_depth=20.0,
                any_alignment_minimum_depth=10.0,
                confident_mean_depth=20.0,
                confident_minimum_depth=10.0,
                assessment_status="assessed",
            ),
        )
    )


def _supported_graph(*, repeat: bool = False) -> GraphEvidence:
    repeats = (
        (
            RepeatSupport(
                repeat_id="IR",
                sequence_id="ctg",
                length=25_000,
                orientation="+,-",
                path_occurrences=2,
                spanning_reads=12,
                status="resolved",
            ),
        )
        if repeat
        else ()
    )
    return GraphEvidence(
        gfa_summary=GfaSummary(
            segment_count=3,
            edge_count=3,
            path_count=1,
            component_count=1,
            branch_count=0,
            parallel_edge_count=0,
            tip_count=0,
            path_names=("ctg",),
        ),
        junction_support=(
            JunctionSupport(
                sequence_id="ctg",
                left_segment="A+",
                right_segment="B+",
                library_role="hifi",
                supporting_reads=5,
                contradicting_reads=0,
                status="supported",
                anchor_length=250,
            ),
        ),
        repeat_support=repeats,
    )


def test_direct_target_marker_is_target_supported(tmp_path: Path) -> None:
    markers = MarkerEvidence(
        marker_hits=(
            MarkerHit(
                profile_id="plastid:rbcL",
                sequence_id="ctg",
                start=100,
                end=400,
                complete=True,
                target="plastid",
            ),
        )
    )

    interpreted = interpret_organelle(
        _resolved(tmp_path),
        _mapping(),
        _supported_graph(),
        markers,
        QC_POLICY_V5,
    )

    assert interpreted.sequence_identities[0].identity_evidence_status == "target_supported"
    identity_check = next(
        item for item in interpreted.checks if item.check_id == "qc.target_identity"
    )
    assert identity_check.status == "pass"


def test_unassessed_markers_do_not_turn_read_and_graph_context_into_identity(
    tmp_path: Path,
) -> None:
    interpreted = interpret_organelle(
        _resolved(tmp_path),
        _mapping(),
        _supported_graph(),
        MarkerEvidence(),
        QC_POLICY_V5,
    )

    assert interpreted.sequence_identities[0].identity_evidence_status == "unresolved"
    identity_check = next(
        item for item in interpreted.checks if item.check_id == "qc.target_identity"
    )
    assert identity_check.status == "not_assessed"


def test_missing_inverted_repeat_is_not_a_failure(tmp_path: Path) -> None:
    interpreted = interpret_organelle(
        _resolved(tmp_path),
        _mapping(),
        _supported_graph(repeat=False),
        MarkerEvidence(),
        QC_POLICY_V5,
    )

    ir_check = next(
        item for item in interpreted.checks if item.check_id == "qc.plastid_inverted_repeat"
    )
    assert ir_check.status == "not_assessed"
    assert not any(item.status == "fail" for item in interpreted.checks)


def test_supported_inverted_repeat_exposes_flip_flop_alternative(tmp_path: Path) -> None:
    interpreted = interpret_organelle(
        _resolved(tmp_path),
        _mapping(),
        _supported_graph(repeat=True),
        MarkerEvidence(),
        QC_POLICY_V5,
    )

    assert interpreted.alternative_configurations[0].status == "supported"


def test_read_supported_plastid_with_both_marker_sets_is_not_called_contamination(
    tmp_path: Path,
) -> None:
    markers = MarkerEvidence(
        marker_hits=(
            MarkerHit(
                profile_id="plastid:rbcL",
                sequence_id="ctg",
                start=100,
                end=400,
                complete=True,
                target="plastid",
            ),
            MarkerHit(
                profile_id="mitochondrion:atp1",
                sequence_id="ctg",
                start=500,
                end=800,
                complete=True,
                target="mitochondrion",
            ),
        )
    )

    interpreted = interpret_organelle(
        _resolved(tmp_path),
        _mapping(),
        _supported_graph(),
        markers,
        QC_POLICY_V5,
    )

    conflict = next(
        item for item in interpreted.checks if item.check_id == "qc.conflicting_organelle_marker"
    )
    assert conflict.status == "pass"
    assert conflict.value == 0
