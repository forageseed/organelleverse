"""Auditable filters for clean-room tRNA calls."""

from __future__ import annotations

from collections import defaultdict

from .calibration import load_trna_model
from .models import TRNACandidate, TRNAFeatureSet, TRNAFilterDecision, TRNAScore


def filter_candidate(
    candidate: TRNACandidate,
    score: TRNAScore,
    features: TRNAFeatureSet | None,
) -> TRNAFilterDecision:
    params = load_trna_model(candidate.organelle)
    if candidate.length < 55 or candidate.length > 130:
        return TRNAFilterDecision(False, "partial_length", "low")
    if "n" in candidate.anticodon:
        return TRNAFilterDecision(False, "ambiguous_anticodon", "low")
    if features is None:
        return TRNAFilterDecision(False, "missing_structure", "low")
    if features.accepted_stem_count < 3:
        return TRNAFilterDecision(False, "weak_stem_set", "low")
    if score.structure_score < params.min_structure_score:
        return TRNAFilterDecision(False, "low_structure_score", "low")
    if score.total < params.min_total_score:
        return TRNAFilterDecision(False, "low_total_score", "low")
    confidence = "high" if score.total >= params.min_total_score + 5 else "medium"
    return TRNAFilterDecision(True, "pass", confidence)


def apply_overcall_density_filter(
    calls: list[tuple[TRNACandidate, TRNAScore, TRNAFeatureSet | None]],
    *,
    max_calls_per_gene: int = 8,
) -> list[tuple[TRNACandidate, TRNAScore, TRNAFilterDecision]]:
    grouped: dict[str, list[tuple[TRNACandidate, TRNAScore, TRNAFeatureSet | None]]] = defaultdict(
        list
    )
    for candidate, score, features in calls:
        grouped[candidate.gene_name].append((candidate, score, features))

    output: list[tuple[TRNACandidate, TRNAScore, TRNAFilterDecision]] = []
    for gene_calls in grouped.values():
        ranked = sorted(gene_calls, key=lambda item: item[1].total, reverse=True)
        passed_count = 0
        for candidate, score, features in ranked:
            decision = filter_candidate(candidate, score, features)
            if decision.passed:
                passed_count += 1
                if passed_count > max_calls_per_gene:
                    decision = TRNAFilterDecision(False, "overcall_density", "low")
            output.append((candidate, score, decision))
    return sorted(output, key=lambda item: (item[0].start, item[0].end, item[0].gene_name))
