"""Conservative plant-organelle interpretation of captured QC evidence."""

from __future__ import annotations

from .contracts import (
    AlternativeConfiguration,
    GraphEvidence,
    MappingEvidence,
    MarkerEvidence,
    OrganelleEvidence,
    QcCheck,
    ResolvedAssemblyEvidence,
    SequenceIdentity,
)
from .policy import QcPolicy
from .static import summarize_fasta


def interpret_organelle(
    evidence: ResolvedAssemblyEvidence,
    mapping: MappingEvidence,
    graph: GraphEvidence,
    markers: MarkerEvidence,
    policy: QcPolicy,
) -> OrganelleEvidence:
    """Apply the three-state identity ladder and organelle-specific rules."""

    del policy  # policy thresholds were already applied by evidence collectors
    target = evidence.primary_genome.organelle
    profiles_assessed = markers.profiles_assessed or bool(markers.marker_hits)
    sequence_ids = [summary.sequence_id for summary in summarize_fasta(evidence.primary_fasta_path)]
    supported_boundaries: dict[str, set[tuple[str, str]]] = {}
    for item in graph.junction_support:
        if item.status == "supported":
            supported_boundaries.setdefault(item.sequence_id, set()).add(
                (item.left_segment, item.right_segment)
            )
    read_supported = {
        item.sequence_id
        for item in mapping.coverage_windows
        if item.assessment_status == "assessed"
        and item.any_alignment_mean_depth is not None
        and item.any_alignment_mean_depth > 0
    }
    hits_by_sequence = {
        sequence_id: tuple(hit for hit in markers.marker_hits if hit.sequence_id == sequence_id)
        for sequence_id in sequence_ids
    }

    boundary_supported_mtpt: set[str] = set()
    target_linked_mtpt: set[str] = set()
    if target == "mitochondrion":
        boundary_supported_mtpt = {
            sequence_id
            for sequence_id, hits in hits_by_sequence.items()
            if sequence_id in read_supported
            and len(supported_boundaries.get(sequence_id, set())) >= 2
            and any(hit.target == "plastid" for hit in hits)
        }
        target_linked_mtpt = {
            sequence_id
            for sequence_id, hits in hits_by_sequence.items()
            if sequence_id in read_supported
            and any(hit.target == "mitochondrion" for hit in hits)
            and any(hit.target == "plastid" for hit in hits)
        }
    mtpt_sequences = boundary_supported_mtpt | target_linked_mtpt

    identities: list[SequenceIdentity] = []
    unintegrated_conflicts = 0
    for sequence_id in sequence_ids:
        hits = hits_by_sequence[sequence_id]
        direct = any(hit.target == target for hit in hits)
        conflicts = [hit for hit in hits if hit.target and hit.target != target]
        if direct:
            status = "target_supported"
        elif sequence_id in mtpt_sequences:
            status = "context_supported"
        else:
            status = "unresolved"
        mixed_target_context = direct and sequence_id in read_supported
        if conflicts and sequence_id not in mtpt_sequences and not mixed_target_context:
            unintegrated_conflicts += len(conflicts)
        identities.append(
            SequenceIdentity(
                sequence_id=sequence_id,
                identity_evidence_status=status,
            )
        )

    checks: list[QcCheck] = [
        _identity_check(identities, profiles_assessed=profiles_assessed),
        QcCheck(
            check_id="qc.conflicting_organelle_marker",
            category="identity",
            status="fail" if unintegrated_conflicts else "pass",
            value=unintegrated_conflicts,
            unit="marker_hits",
            message=(
                "Conflicting marker evidence lacks supported integration context."
                if unintegrated_conflicts
                else "No unresolved conflicting-organelle marker evidence remains."
            ),
            finding_code=("qc.conflicting_organelle_marker" if unintegrated_conflicts else ""),
        ),
    ]
    alternatives = _alternative_configurations(graph, target)
    if target == "plastid":
        checks.extend(_plastid_checks(graph))
    else:
        checks.extend(
            _mitochondrial_checks(
                graph,
                boundary_supported_mtpt,
                target_linked_mtpt,
            )
        )
    return OrganelleEvidence(
        checks=tuple(checks),
        sequence_identities=tuple(identities),
        alternative_configurations=alternatives,
    )


def _identity_check(
    identities: list[SequenceIdentity],
    *,
    profiles_assessed: bool,
) -> QcCheck:
    if not profiles_assessed:
        return QcCheck(
            check_id="qc.target_identity",
            category="identity",
            status="not_assessed",
            value=None,
            unit="",
            message="Target identity was not assessed because marker profiles were unavailable.",
        )
    supported = sum(
        item.identity_evidence_status in {"target_supported", "context_supported"}
        for item in identities
    )
    unresolved = len(identities) - supported
    if supported == 0:
        status = "fail"
        finding_code = "qc.target_identity_unsupported"
        message = "No reported sequence has target or supported-context identity evidence."
    elif unresolved:
        status = "warn"
        finding_code = "qc.target_identity_unresolved"
        message = "At least one reported sequence has unresolved target identity."
    else:
        status = "pass"
        finding_code = ""
        message = "Every reported sequence has target or supported-context identity evidence."
    return QcCheck(
        check_id="qc.target_identity",
        category="identity",
        status=status,
        value=unresolved,
        unit="unresolved_sequences",
        message=message,
        finding_code=finding_code,
    )


def _plastid_checks(graph: GraphEvidence) -> tuple[QcCheck, ...]:
    repeats = tuple(
        item
        for item in graph.repeat_support
        if item.path_occurrences == 2 and "+" in item.orientation and "-" in item.orientation
    )
    if any(item.status == "resolved" for item in repeats):
        ir_status = "pass"
        message = "A two-copy repeat has resolved read-supported path context."
    elif repeats:
        ir_status = "warn"
        message = "A two-copy repeat is present but its path context is unresolved."
    else:
        ir_status = "not_assessed"
        message = "No two-copy repeat was declared; repeat absence is not a failure."
    contradicted = sum(item.status == "contradicted" for item in graph.junction_support)
    return (
        QcCheck(
            check_id="qc.plastid_inverted_repeat",
            category="plastid_structure",
            status=ir_status,
            value=len(repeats),
            unit="two_copy_repeats",
            message=message,
        ),
        QcCheck(
            check_id="qc.plastid_structure",
            category="plastid_structure",
            status="fail" if contradicted else "pass" if graph.gfa_summary else "not_assessed",
            value=contradicted,
            unit="contradicted_junctions",
            message=(
                "Plastid graph structure has contradictory junction evidence."
                if contradicted
                else "No contradictory plastid junction evidence was detected."
                if graph.gfa_summary
                else "No plastid graph structure was available."
            ),
            finding_code="qc.contradicted_junction" if contradicted else "",
        ),
    )


def _mitochondrial_checks(
    graph: GraphEvidence,
    boundary_supported_mtpt: set[str],
    target_linked_mtpt: set[str],
) -> tuple[QcCheck, ...]:
    contradicted = sum(item.status == "contradicted" for item in graph.junction_support)
    probable_only = target_linked_mtpt - boundary_supported_mtpt
    mtpt_sequences = boundary_supported_mtpt | target_linked_mtpt
    return (
        QcCheck(
            check_id="qc.mitochondrial_structure",
            category="mitochondrial_structure",
            status="fail" if contradicted else "pass" if graph.gfa_summary else "not_assessed",
            value=(graph.gfa_summary.component_count if graph.gfa_summary is not None else None),
            unit="components",
            message=(
                "Mitochondrial graph has contradictory junction evidence."
                if contradicted
                else "Observed linear, multipartite, or branched graph forms are retained without shape scoring."
                if graph.gfa_summary
                else "No mitochondrial graph structure was available."
            ),
            finding_code="qc.contradicted_junction" if contradicted else "",
        ),
        QcCheck(
            check_id="qc.mtpt_context",
            category="mitochondrial_structure",
            status=(
                "warn" if probable_only else "pass" if boundary_supported_mtpt else "not_assessed"
            ),
            value=len(mtpt_sequences),
            unit="sequences",
            message=(
                "Plastid-like marker evidence has read-supported mitochondrial boundary context."
                if boundary_supported_mtpt and not probable_only
                else "Plastid-like marker evidence is target-linked and read-supported, but graph boundaries were not resolved."
                if probable_only
                else "No boundary-supported plastid-like tract was identified."
            ),
            finding_code="qc.mtpt_context_unresolved" if probable_only else "",
        ),
    )


def _alternative_configurations(
    graph: GraphEvidence,
    target: str,
) -> tuple[AlternativeConfiguration, ...]:
    alternatives: list[AlternativeConfiguration] = []
    for repeat in graph.repeat_support:
        if repeat.status == "resolved":
            status = "supported"
        elif repeat.status == "contradicted":
            status = "contradicted"
        else:
            status = "unresolved"
        inverted = "+" in repeat.orientation and "-" in repeat.orientation
        label = "flip_flop" if target == "plastid" and inverted else "repeat_mediated"
        alternatives.append(
            AlternativeConfiguration(
                configuration_id=f"{label}:{repeat.repeat_id}",
                sequence_id=repeat.sequence_id,
                relative_support=None,
                status=status,
            )
        )
    return tuple(alternatives)


__all__ = ["interpret_organelle"]
