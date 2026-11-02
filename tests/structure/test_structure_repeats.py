"""SSR unit-class breakdown, per-unit threshold overrides and repeat size classes.

Complements ``test_misa_ssr`` (MISA thresholds, record boundaries) and
``test_merged_repeats`` (diagonal merging) with the report-level fields:
``ssr_by_unit``, ``unit_thresholds``, the ``sequence_id`` TSV column and
``repeat_size_classes``.
"""

from __future__ import annotations

import random
from pathlib import Path

from organelleverse.structure import compute_multiconf, compute_repeats, multiconf, repeats, write_repeats
from organelleverse.structure.structure import _find_ssrs


def _block(seed: int, length: int) -> str:
    """Deterministic pseudo-random ACGT block with no long internal repeats."""
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


def test_unit_thresholds_override_misa_defaults() -> None:
    assert _find_ssrs("A" * 9, 1, 6, 3) == []  # below the MISA mono threshold of 10
    ssrs = _find_ssrs("A" * 9, 1, 6, 3, unit_thresholds={1: 5})
    assert [(s["unit"], s["copies"]) for s in ssrs] == [("A", 9)]


def test_min_copy_still_floors_overridden_thresholds() -> None:
    # An override cannot drop a unit below the legacy ``min_copy`` floor.
    assert _find_ssrs("A" * 9, 1, 1, 12, unit_thresholds={1: 5}) == []


def test_every_ssr_carries_its_record_and_unit_breakdown(tmp_path: Path) -> None:
    fasta = tmp_path / "multi.fasta"
    fasta.write_text(">r1\nACGTTGCA" + "AT" * 6 + "\n>r2\nTGCATGAC" + "A" * 12 + "\n")
    result = repeats(fasta)
    ssrs = result.metrics["ssrs"]
    assert [(s["sequence_id"], s["unit"]) for s in ssrs] == [
        ("r1", "AT"),
        ("r1", "ATAT"),
        ("r2", "A"),
        ("r2", "AA"),
        ("r2", "AAA"),
        ("r2", "AAAA"),
    ]
    assert result.metrics["ssr_by_unit"] == {"1": 1, "2": 2, "3": 1, "4": 2}
    assert "1-mer: 1" in result.summary_text

    data = compute_repeats(fasta)
    assert data["ssr_count"] == 6
    assert data["ssr_by_unit"] == result.metrics["ssr_by_unit"]
    assert data["ssrs"] == [dict(s) for s in ssrs]

    out = write_repeats(result, tmp_path / "ssr.tsv")
    lines = out.read_text().splitlines()
    assert lines[0] == "sequence_id\tstart\tend\tunit\tlength\tcopies"
    assert lines[1] == "r1\t9\t20\tAT\t12\t6"


def test_unit_thresholds_reach_the_result_and_provenance(tmp_path: Path) -> None:
    fasta = tmp_path / "a9.fasta"
    fasta.write_text(">r\nCGTACGATCG" + "A" * 9 + "GCTAGCTAGC\n")
    assert repeats(fasta).metrics["ssr_count"] == 0
    tuned = repeats(fasta, unit_thresholds={1: 5})
    assert tuned.metrics["ssr_by_unit"] == {"1": 1}
    assert compute_repeats(fasta, unit_thresholds={1: 5})["ssr_by_unit"] == {"1": 1}


def test_repeat_size_classes_split_at_100_bp_and_1_kb(tmp_path: Path) -> None:
    fasta = tmp_path / "sizes.fasta"
    arm = _block(11, 120)
    fasta.write_text(">r\n" + arm + _block(12, 200) + arm + _block(13, 200) + "\n")
    classes = multiconf(fasta).metrics["repeat_size_classes"]
    assert set(classes) == {"large_ge_1000", "medium_100_999", "small_lt_100"}
    assert classes["medium_100_999"] == 1
    assert classes["large_ge_1000"] == 0
    assert compute_multiconf(fasta)["repeat_size_classes"] == classes

    big = _block(21, 1100)
    fasta.write_text(">r\n" + big + _block(22, 300) + big + _block(23, 300) + "\n")
    classes = multiconf(fasta).metrics["repeat_size_classes"]
    assert classes["large_ge_1000"] == 1
