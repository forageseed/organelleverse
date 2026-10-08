from __future__ import annotations

import csv

from organelleverse.annotation.trna_core.covariance import score_candidate
from organelleverse.annotation.trna_core.features import best_structural_model
from organelleverse.annotation.trna_core.filters import (
    apply_overcall_density_filter,
    filter_candidate,
)
from organelleverse.annotation.trna_core.models import (
    StemFeature,
    TRNACandidate,
    TRNAFeatureSet,
    TRNAScore,
)
from tests._paths import PROJECT_ROOT

ROOT = PROJECT_ROOT


def _rows(name: str) -> list[dict[str, str]]:
    with (ROOT / "tests" / "fixtures" / "trna_core" / name).open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _aa(gene: str) -> str:
    return gene.removeprefix("trn").split("(", 1)[0]


def _candidate(row: dict[str, str]) -> TRNACandidate:
    return TRNACandidate(
        row["sequence"],
        int(row["start"]),
        int(row["end"]),
        1 if row["strand"] == "+" else -1,
        row["anticodon"],
        _aa(row["gene"]),
        source=row["source"],
    )


def test_filter_candidate_rejects_partial_short_call():
    candidate = TRNACandidate("A" * 40, 1, 40, 1, "gaa", "F")
    score = TRNAScore(120, 4, 30, 40, 6, 0, 0, 0)

    decision = filter_candidate(candidate, score, None)

    assert decision.passed is False
    assert decision.reason == "partial_length"


def test_filter_candidate_rejects_missing_structure_even_with_anticodon():
    candidate = TRNACandidate("A" * 30 + "GAA" + "A" * 40, 1, 73, 1, "gaa", "F")
    score = score_candidate(candidate)

    decision = filter_candidate(candidate, score, None)

    assert decision.passed is False
    assert decision.reason == "missing_structure"


def test_filter_candidate_accepts_real_high_scoring_positive():
    row = next(row for row in _rows("positives.tsv") if len(row["sequence"]) <= 90)
    candidate = _candidate(row)
    features = best_structural_model(
        candidate.sequence,
        candidate.sequence.find(candidate.anticodon.upper()),
        organelle="mitochondrion",
    )
    score = score_candidate(candidate)

    decision = filter_candidate(candidate, score, features)

    assert decision.passed is True
    assert decision.confidence == "high"


def test_overcall_density_filter_marks_excess_same_gene_calls():
    calls = []
    features = TRNAFeatureSet(
        anticodon_offset=30,
        stems=(
            StemFeature("acceptor", (0, 6), (66, 72), score=14),
            StemFeature("anticodon", (25, 29), (36, 40), score=10),
            StemFeature("t_arm", (50, 54), (60, 64), score=10),
        ),
        structure_score=34,
    )
    for idx in range(8):
        candidate = TRNACandidate("A" * 73, idx * 100 + 1, idx * 100 + 73, 1, "gaa", "F")
        score = TRNAScore(120 - idx, 4, 30, 40, 6, 0, 0, 0)
        calls.append((candidate, score, features))

    decisions = apply_overcall_density_filter(calls, max_calls_per_gene=4)

    assert sum(1 for _, _, decision in decisions if decision.passed) == 4
    assert any(decision.reason == "overcall_density" for _, _, decision in decisions)
