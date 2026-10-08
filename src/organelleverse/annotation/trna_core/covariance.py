"""Covariance-like scoring for clean-room tRNA candidates."""

from __future__ import annotations

import math

from .anticodon import isotypes_for_anticodon
from .calibration import TRNAModelParameters, load_trna_model
from .features import best_structural_model
from .models import TRNACandidate, TRNAFeatureSet, TRNAScore


def score_feature_set(
    features: TRNAFeatureSet, params: TRNAModelParameters | None = None
) -> TRNAScore:
    model = params or load_trna_model("mitochondrion")
    structure_score = features.structure_score
    covariance_score = _weighted_stem_score(features, model)
    total = structure_score + covariance_score
    return TRNAScore(
        total=total,
        sequence_score=0.0,
        structure_score=structure_score,
        covariance_score=covariance_score,
        anticodon_score=0.0,
        intron_score=0.0,
        pseudogene_penalty=0.0,
        overcall_penalty=0.0,
    )


def score_candidate(
    candidate: TRNACandidate, params: TRNAModelParameters | None = None
) -> TRNAScore:
    model = params or load_trna_model(candidate.organelle)
    features = _best_candidate_features(candidate, model)
    sequence_score = _sequence_complexity_score(candidate.sequence)
    low_complexity_penalty = model.low_complexity_penalty if sequence_score < 2.0 else 0.0
    anticodon_score = (
        model.anticodon_match_score
        if candidate.amino_acid in isotypes_for_anticodon(candidate.anticodon, candidate.organelle)
        else -model.anticodon_mismatch_penalty
    )
    if features is None:
        pseudogene_penalty = model.no_structure_penalty + low_complexity_penalty
        return TRNAScore(
            total=sequence_score + anticodon_score - pseudogene_penalty,
            sequence_score=sequence_score,
            structure_score=0.0,
            covariance_score=0.0,
            anticodon_score=anticodon_score,
            intron_score=0.0,
            pseudogene_penalty=pseudogene_penalty,
            overcall_penalty=0.0,
        )

    structure_score = features.structure_score
    covariance_score = _weighted_stem_score(features, model) + _anticodon_position_score(
        features, len(candidate.sequence)
    )
    pseudogene_penalty = low_complexity_penalty
    total = (
        sequence_score + structure_score + covariance_score + anticodon_score - pseudogene_penalty
    )
    return TRNAScore(
        total=total,
        sequence_score=sequence_score,
        structure_score=structure_score,
        covariance_score=covariance_score,
        anticodon_score=anticodon_score,
        intron_score=0.0,
        pseudogene_penalty=pseudogene_penalty,
        overcall_penalty=0.0,
    )


def best_candidate_features(
    candidate: TRNACandidate, params: TRNAModelParameters | None = None
) -> TRNAFeatureSet | None:
    return _best_candidate_features(candidate, params or load_trna_model(candidate.organelle))


def _best_candidate_features(
    candidate: TRNACandidate, params: TRNAModelParameters
) -> TRNAFeatureSet | None:
    best: TRNAFeatureSet | None = None
    for offset in _anticodon_offsets(candidate.sequence, candidate.anticodon):
        features = best_structural_model(candidate.sequence, offset, organelle=candidate.organelle)
        if features is None:
            continue
        if best is None or _weighted_stem_score(features, params) > _weighted_stem_score(
            best, params
        ):
            best = features
    return best


def _anticodon_offsets(sequence: str, anticodon: str) -> list[int]:
    seq = sequence.upper().replace("U", "T")
    motif = anticodon.upper().replace("U", "T")
    offsets = []
    start = 0
    while True:
        index = seq.find(motif, start)
        if index < 0:
            return offsets
        offsets.append(index)
        start = index + 1


def _weighted_stem_score(features: TRNAFeatureSet, params: TRNAModelParameters) -> float:
    score = 0.0
    observed = set()
    for stem in features.stems:
        observed.add(stem.name)
        score += stem.score * params.stem_weights.get(stem.name, 1.0)
    if set(params.critical_stems).issubset(observed):
        score += params.critical_stem_bonus
    return score


def _sequence_complexity_score(sequence: str) -> float:
    bases = [base for base in sequence.upper().replace("U", "T") if base in {"A", "C", "G", "T"}]
    if not bases:
        return 0.0
    counts = [bases.count(base) for base in {"A", "C", "G", "T"} if bases.count(base)]
    entropy = -sum((count / len(bases)) * math.log2(count / len(bases)) for count in counts)
    ambiguous_penalty = (len(sequence) - len(bases)) / max(len(sequence), 1) * 4.0
    return max(0.0, entropy * 2.0 - ambiguous_penalty)


def _anticodon_position_score(features: TRNAFeatureSet, length: int) -> float:
    if features.anticodon_offset is None:
        return -8.0
    relative = features.anticodon_offset / max(length, 1)
    if 0.35 <= relative <= 0.55:
        return 4.0
    if 0.25 <= relative <= 0.65:
        return 1.0
    return -5.0
