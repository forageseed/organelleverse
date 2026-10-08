from __future__ import annotations

import csv

from organelleverse.annotation.trna_core.features import (
    best_structural_model,
    classify_pair,
    find_structural_models,
    offset_to_genomic_span,
)
from organelleverse.annotation.trna_core.models import TRNACandidate
from tests._paths import PROJECT_ROOT

ROOT = PROJECT_ROOT


def _fixture_rows(name: str) -> list[dict[str, str]]:
    with (ROOT / "tests" / "fixtures" / "trna_core" / name).open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_classify_pair_distinguishes_pair_classes():
    assert classify_pair("G", "C") == "wc"
    assert classify_pair("A", "U") == "wc"
    assert classify_pair("G", "T") == "wobble"
    assert classify_pair("A", "C") == "mismatch"
    assert classify_pair("N", "C") == "gap"


def test_real_pmga_positive_yields_multiple_structural_stems():
    row = next(row for row in _fixture_rows("positives.tsv") if len(row["sequence"]) <= 90)
    sequence = row["sequence"]
    anticodon_offset = sequence.find(row["anticodon"].upper())

    models = find_structural_models(sequence, anticodon_offset, organelle="mitochondrion")
    best = best_structural_model(sequence, anticodon_offset, organelle="mitochondrion")

    assert anticodon_offset >= 0
    assert models
    assert best is not None
    assert best.accepted_stem_count >= 3
    assert best.structure_score > 10
    assert best.stem("acceptor") is not None
    assert best.stem("anticodon") is not None


def test_low_complexity_anticodon_window_does_not_make_valid_model():
    sequence = "A" * 30 + "GAA" + "A" * 40

    assert best_structural_model(sequence, 30, organelle="mitochondrion") is None


def test_hard_negative_near_edge_does_not_pass_core_stem_set():
    row = next(
        row
        for row in _fixture_rows("hard_negatives.tsv")
        if 0 <= row["sequence"].find(row["anticodon"].upper()) < 10
    )
    sequence = row["sequence"]
    anticodon_offset = sequence.find(row["anticodon"].upper())

    best = best_structural_model(sequence, anticodon_offset, organelle="mitochondrion")

    assert best is None or best.accepted_stem_count < 3


def test_offset_to_genomic_span_handles_reverse_strand():
    candidate = TRNACandidate(
        "A" * 73, start=100, end=172, strand=-1, anticodon="gaa", amino_acid="F"
    )

    assert offset_to_genomic_span(candidate, (0, 2)) == (170, 172)
    assert offset_to_genomic_span(candidate, (70, 72)) == (100, 102)
