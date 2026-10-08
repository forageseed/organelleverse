"""validate_transfers_depth: lower-bound criterion by default, two-sided formula when the organelle depth is given."""

from __future__ import annotations

from pathlib import Path

import pytest

pysam = pytest.importorskip("pysam")

from organelleverse.transfer.transfer import validate_transfers_depth  # noqa: E402

READ = 50


def _make_bam(path: Path, tiles: list[tuple[int, int, int]]) -> Path:
    """Reads tiled so the depth is exactly ``d`` on [start, end): ``d`` reads at every 50 bp."""
    header = {"HD": {"VN": "1.6", "SO": "unsorted"}, "SQ": [{"SN": "chr1", "LN": 3000}]}
    unsorted = path.with_suffix(".unsorted.bam")
    with pysam.AlignmentFile(str(unsorted), "wb", header=header) as out:
        n = 0
        for start, end, depth in tiles:
            for copy in range(depth):
                for pos in range(start, end - READ + 1, READ):
                    a = pysam.AlignedSegment()
                    a.query_name = f"r{n}"
                    n += 1
                    a.query_sequence = "A" * READ
                    a.flag = 0
                    a.reference_id = 0
                    a.reference_start = pos
                    a.mapping_quality = 60
                    a.cigar = [(0, READ)]
                    a.query_qualities = pysam.qualitystring_to_array("I" * READ)
                    out.write(a)
    pysam.sort("-o", str(path), str(unsorted))
    pysam.index(str(path))
    return path


def _candidate(start: int, end: int) -> dict:
    return {
        "nuclear_seqid": "chr1", "nuclear_start": start, "nuclear_end": end,
        "organelle_seqid": "mt", "organelle_start": 1, "organelle_end": end - start + 1,
        "identity": 99.0, "length": end - start + 1, "evalue": 1e-50, "bitscore": 500.0,
    }


@pytest.fixture
def bam(tmp_path: Path) -> Path:
    # chr1 depth: 10x on 1-1000 (a nuclear-depth insert), 2x on 1001-1500 (too low), 30x on 1501-2000
    return _make_bam(tmp_path / "d.bam", [(0, 1000, 10), (1000, 1500, 2), (1500, 2000, 30)])


CANDIDATES = [_candidate(101, 900), _candidate(1101, 1400), _candidate(1601, 1900)]  # 10x, 2x, 30x


def _supported(result) -> list[bool]:
    return [c["depth_supported"] for c in result.metrics["candidates"]]


def test_default_is_a_lower_bound_so_organelle_multi_mapping_is_not_held_against_an_insert(bam: Path) -> None:
    result = validate_transfers_depth(CANDIDATES, bam, nuclear_mean_depth=10.0)
    assert _supported(result) == [True, False, True]  # 30x is fine; only 2x (< 5x) is rejected
    assert result.metrics["criterion"] == "lower_bound"
    assert result.metrics["minimum_depth"] == 5.0


def test_tolerance_sets_the_lower_bound(bam: Path) -> None:
    result = validate_transfers_depth(CANDIDATES, bam, nuclear_mean_depth=10.0, tolerance=0.9)
    assert _supported(result) == [True, True, True]  # minimum depth 1x
    assert validate_transfers_depth(CANDIDATES, bam, nuclear_mean_depth=10.0, tolerance=0.1).metrics[
        "minimum_depth"
    ] == 9.0


def test_given_organelle_depth_selects_the_two_sided_formula(bam: Path) -> None:
    result = validate_transfers_depth(CANDIDATES, bam, nuclear_mean_depth=10.0, organelle_mean_depth=20.0)
    assert result.metrics["criterion"] == "two_sided"
    assert result.metrics["expected_depth"] == 30.0
    assert _supported(result) == [False, False, True]  # expected 30x +/- 50%: only the 30x candidate


def test_an_explicit_zero_organelle_depth_is_the_two_sided_formula_too(bam: Path) -> None:
    result = validate_transfers_depth(CANDIDATES, bam, nuclear_mean_depth=10.0, organelle_mean_depth=0.0)
    assert result.metrics["criterion"] == "two_sided"
    assert _supported(result) == [True, False, False]  # 10x +/- 50%: the 30x candidate is now too high
