from __future__ import annotations

import csv
from statistics import median

from organelleverse.annotation.trna_core.covariance import score_candidate
from organelleverse.annotation.trna_core.models import TRNACandidate
from tests._paths import PROJECT_ROOT

ROOT = PROJECT_ROOT


def _rows(name: str) -> list[dict[str, str]]:
    with (ROOT / "tests" / "fixtures" / "trna_core" / name).open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _amino_acid(gene: str) -> str:
    return gene.removeprefix("trn").split("(", 1)[0]


def _candidate(row: dict[str, str]) -> TRNACandidate:
    return TRNACandidate(
        sequence=row["sequence"],
        start=int(row["start"]),
        end=int(row["end"]),
        strand=1 if row["strand"] == "+" else -1,
        anticodon=row["anticodon"],
        amino_acid=_amino_acid(row["gene"]),
        source=row["source"],
    )


def test_real_positive_fixture_scores_above_hard_negative_distribution():
    positives = [
        _candidate(row)
        for row in _rows("positives.tsv")
        if 55 <= len(row["sequence"]) <= 130 and row["sequence"].find(row["anticodon"].upper()) >= 0
    ][:40]
    negatives = [
        _candidate(row)
        for row in _rows("hard_negatives.tsv")
        if 55 <= len(row["sequence"]) <= 130 and row["sequence"].find(row["anticodon"].upper()) >= 0
    ][:40]

    positive_scores = [score_candidate(candidate).total for candidate in positives]
    negative_scores = [score_candidate(candidate).total for candidate in negatives]

    assert len(positive_scores) >= 20
    assert len(negative_scores) >= 20
    assert median(positive_scores) > median(negative_scores) + 12


def test_shuffled_positive_sequence_scores_lower_than_real_locus():
    row = next(row for row in _rows("positives.tsv") if 55 <= len(row["sequence"]) <= 90)
    real = _candidate(row)
    shuffled = TRNACandidate(
        sequence="".join(sorted(row["sequence"])),
        start=real.start,
        end=real.end,
        strand=real.strand,
        anticodon=real.anticodon,
        amino_acid=real.amino_acid,
        source="synthetic_negative",
    )

    assert score_candidate(real).total > score_candidate(shuffled).total + 20


def test_missing_structure_is_penalized_as_pseudogene_like():
    candidate = TRNACandidate("A" * 30 + "GAA" + "A" * 40, 1, 73, 1, "gaa", "F")
    score = score_candidate(candidate)

    assert score.pseudogene_penalty > 0
    assert score.total < 0
