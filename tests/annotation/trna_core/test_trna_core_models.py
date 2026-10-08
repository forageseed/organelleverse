from __future__ import annotations

import pytest

from organelleverse.annotation.trna_core.models import (
    StemFeature,
    StemPair,
    TRNACall,
    TRNACandidate,
    TRNAFeatureSet,
    TRNAFilterDecision,
    TRNAScore,
)


def test_candidate_uses_one_based_inclusive_coordinates_and_gene_name():
    candidate = TRNACandidate(
        sequence="G" * 73,
        start=10,
        end=82,
        strand=1,
        anticodon="GAA",
        amino_acid="F",
    )

    assert candidate.length == 73
    assert candidate.anticodon == "gaa"
    assert candidate.gene_name == "trnF(gaa)"
    assert candidate.source == "native_cleanroom"


def test_candidate_formats_formyl_methionine_gene_name():
    candidate = TRNACandidate(
        sequence="G" * 73,
        start=10,
        end=82,
        strand=-1,
        anticodon="cat",
        amino_acid="fM",
    )

    assert candidate.gene_name == "trnfM(cat)"


@pytest.mark.parametrize(
    ("start", "end", "strand"),
    [(0, 10, 1), (10, 9, 1), (1, 10, 0), (1, 10, 2)],
)
def test_candidate_rejects_invalid_coordinates(start: int, end: int, strand: int):
    with pytest.raises(ValueError):
        TRNACandidate(
            sequence="A" * 10, start=start, end=end, strand=strand, anticodon="gaa", amino_acid="F"
        )


def test_call_keeps_source_confidence_and_score_components():
    candidate = TRNACandidate("A" * 73, 1, 73, 1, "gaa", "F")
    features = TRNAFeatureSet(
        anticodon_offset=34,
        stems=(
            StemFeature(
                name="acceptor",
                left=(0, 6),
                right=(66, 72),
                pairs=(StemPair(0, 72, "G", "C", "wc"),),
                mismatches=0,
            ),
        ),
    )
    score = TRNAScore(
        total=50.0,
        sequence_score=4.0,
        structure_score=16.0,
        covariance_score=20.0,
        anticodon_score=6.0,
        intron_score=0.0,
        pseudogene_penalty=0.0,
        overcall_penalty=0.0,
    )
    decision = TRNAFilterDecision(passed=True, reason="pass", confidence="high")
    call = TRNACall(candidate=candidate, features=features, score=score, decision=decision)

    assert call.source == "native_cleanroom"
    assert call.confidence == "high"
    assert call.gene_name == "trnF(gaa)"
    assert call.score_components["covariance_score"] == 20.0
