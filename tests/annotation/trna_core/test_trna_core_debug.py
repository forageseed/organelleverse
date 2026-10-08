from __future__ import annotations

import csv

from organelleverse.annotation.trna_core.debug import write_debug_candidates
from organelleverse.annotation.trna_core.models import (
    TRNACall,
    TRNACandidate,
    TRNAFeatureSet,
    TRNAFilterDecision,
    TRNAScore,
)


def test_write_debug_candidates_records_scores_and_reasons(tmp_path):
    candidate = TRNACandidate("A" * 73, 1, 73, 1, "gaa", "F")
    call = TRNACall(
        candidate=candidate,
        features=TRNAFeatureSet(anticodon_offset=30, structure_score=30),
        score=TRNAScore(100, 4, 30, 50, 6, 0, 0, 0),
        decision=TRNAFilterDecision(True, "pass", "high"),
    )
    output = tmp_path / "debug.tsv"

    write_debug_candidates(output, [call])

    rows = list(csv.DictReader(output.open(), delimiter="\t"))
    assert rows[0]["gene"] == "trnF(gaa)"
    assert rows[0]["passed"] == "true"
    assert rows[0]["reason"] == "pass"
    assert rows[0]["total"] == "100"
