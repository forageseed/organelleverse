from __future__ import annotations

import csv

from organelleverse.annotation.trna_core.candidates import (
    deduplicate_candidates,
    scan_candidate_windows,
)
from organelleverse.annotation.trna_core.models import TRNACandidate
from tests._paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _positive_row() -> dict[str, str]:
    with (ROOT / "tests" / "fixtures" / "trna_core" / "positives.tsv").open() as handle:
        return next(
            row for row in csv.DictReader(handle, delimiter="\t") if len(row["sequence"]) <= 90
        )


def _reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def test_scan_candidate_windows_finds_real_positive_on_plus_strand():
    row = _positive_row()
    genome = "N" * 40 + row["sequence"] + "N" * 40

    candidates = scan_candidate_windows(genome, organelle="mitochondrion")

    assert any(
        candidate.strand == 1 and candidate.anticodon == row["anticodon"]
        for candidate in candidates
    )


def test_scan_candidate_windows_finds_real_positive_on_minus_strand():
    row = _positive_row()
    genome = "N" * 40 + _reverse_complement(row["sequence"]) + "N" * 40

    candidates = scan_candidate_windows(genome, organelle="mitochondrion")

    assert any(
        candidate.strand == -1 and candidate.anticodon == row["anticodon"]
        for candidate in candidates
    )


def test_scan_candidate_windows_can_reconstruct_circular_boundary_candidate():
    row = _positive_row()
    sequence = row["sequence"]
    rotated = sequence[40:] + sequence[:40]

    candidates = scan_candidate_windows(rotated, organelle="mitochondrion", circular=True)

    assert any(
        candidate.sequence == sequence and len(candidate.segments) == 2 for candidate in candidates
    )


def test_deduplicate_candidates_keeps_one_same_locus_call():
    a = TRNACandidate("A" * 73, 10, 82, 1, "gaa", "F")
    b = TRNACandidate("A" * 73, 12, 84, 1, "gaa", "F")
    kept = deduplicate_candidates([a, b])

    assert len(kept) == 1
    assert kept[0].start == 10
