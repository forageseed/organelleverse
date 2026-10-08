from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.heteroplasmy import quantify_variant_fractions


def _bam(path: Path, bases: list[str]) -> Path:
    pysam = pytest.importorskip(chr(112)+chr(121)+chr(115)+chr(97)+chr(109))
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "MT", "LN": 40}]}
    unsorted = path.with_suffix(".unsorted.bam")
    with pysam.AlignmentFile(str(unsorted), "wb", header=header) as out:
        for i, base in enumerate(bases):
            read = pysam.AlignedSegment()
            read.query_name = f"r{i}"
            read.query_sequence = "A" * 9 + base + "A" * 10
            read.flag = 0
            read.reference_id = 0
            read.reference_start = 0
            read.mapping_quality = 60
            read.cigar = [(0, 20)]
            read.query_qualities = pysam.qualitystring_to_array("I" * 20)
            out.write(read)
    sorted_bam = path
    pysam.sort("-o", str(sorted_bam), str(unsorted))
    pysam.index(str(sorted_bam))
    return sorted_bam


def _sites(path: Path) -> Path:
    path.write_text("site_id\tseqid\tposition\tref\talt\nsite1\tMT\t10\tA\tG\n", encoding="utf-8")
    return path


def test_known_variant_fraction_and_wilson_interval(tmp_path: Path) -> None:
    bam = _bam(tmp_path / "reads.bam", ["A"] * 7 + ["G"] * 3)
    result = quantify_variant_fractions(bam, _sites(tmp_path / "sites.tsv"), min_depth=5)
    row = result.metrics["sites"][0]
    assert (row["ref_count"], row["alt_count"], row["depth"]) == (7, 3, 10)
    assert row["alt_fraction"] == 0.3
    assert row["ci95_lower"] < 0.3 < row["ci95_upper"]
    assert row["status"] == "ok"
    assert "candidate_sites_only" in result.flags


def test_low_depth_is_not_reported_as_zero_fraction(tmp_path: Path) -> None:
    bam = _bam(tmp_path / "reads.bam", ["A", "G"])
    row = quantify_variant_fractions(bam, _sites(tmp_path / "sites.tsv"), min_depth=5).metrics[
        "sites"
    ][0]
    assert row["depth"] == 2
    assert row["alt_fraction"] is None
    assert row["ci95_lower"] is None
    assert row["status"] == "low_depth"


def test_malformed_candidate_alleles_fail_clearly(tmp_path: Path) -> None:
    path = tmp_path / "sites.tsv"
    path.write_text("site_id\tseqid\tposition\tref\talt\ns\tMT\t1\tA\tA\n", encoding="utf-8")
    result = quantify_variant_fractions(tmp_path / "missing.bam", path)
    assert result.errors
    assert result.errors[0].code == "heteroplasmy.invalid_sites"
