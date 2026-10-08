from __future__ import annotations

import csv
from pathlib import Path

import pytest

from organelleverse.rna_editing.efficiency import quantify_known_site_efficiency

pysam = pytest.importorskip("pysam")


def _write_bam(path: Path) -> None:
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chrM", "LN": 100}]}
    with pysam.AlignmentFile(path, "wb", header=header) as bam:
        cases = [("C", 60, 40, 0)] * 6 + [("T", 60, 40, 0)] * 3 + [("A", 60, 40, 0)] * 2
        cases += [
            ("T", 60, 5, 0),
            ("T", 10, 40, 0),
            ("T", 60, 40, 256),
            ("T", 60, 40, 2048),
            ("T", 60, 40, 512),
        ]
        for index, (base, mapq, baseq, flag) in enumerate(cases):
            read = pysam.AlignedSegment()
            read.query_name = f"read{index}"
            sequence = list("A" * 20)
            sequence[10] = base
            read.query_sequence = "".join(sequence)
            read.flag = flag
            read.reference_id = 0
            read.reference_start = 0
            read.mapping_quality = mapq
            read.cigar = ((0, 20),)
            read.query_qualities = pysam.qualitystring_to_array(chr(baseq + 33) * 20)
            bam.write(read)
        # Both overlapping mates and a mapped orphan are independent passing
        # observations under the documented filters.
        for name, flag in (("overlap", 99), ("overlap", 147), ("orphan", 73)):
            read = pysam.AlignedSegment()
            read.query_name = name
            sequence = list("A" * 20)
            sequence[10] = "T"
            read.query_sequence = "".join(sequence)
            read.flag = flag
            read.reference_id = 0
            read.reference_start = 0
            read.next_reference_id = 0 if name == "overlap" else -1
            read.next_reference_start = 0 if name == "overlap" else -1
            read.template_length = 20 if flag == 99 else (-20 if flag == 147 else 0)
            read.mapping_quality = 60
            read.cigar = ((0, 20),)
            read.query_qualities = pysam.qualitystring_to_array("I" * 20)
            bam.write(read)
    pysam.index(str(path))


def test_known_site_pileup_counts_and_quality_filters(tmp_path: Path) -> None:
    bam = tmp_path / "reads.bam"
    _write_bam(bam)
    sites = tmp_path / "sites.tsv"
    with sites.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["site_id", "seqid", "position", "ref", "edited", "strand"])
        writer.writerow(["site1", "chrM", 11, "C", "T", "+"])

    result = quantify_known_site_efficiency(bam, sites)

    assert result.status == "ok"
    row = result.metrics["sites"][0]
    assert (row["ref_count"], row["edited_count"], row["other_count"], row["depth"]) == (
        6,
        6,
        2,
        14,
    )
    assert row["editing_fraction"] == pytest.approx(6 / 14)
    assert 0 <= row["ci95_lower"] < 6 / 14 < row["ci95_upper"] <= 1
    assert "known_sites_only" in result.flags


def test_invalid_site_table_fails_with_actionable_result(tmp_path: Path) -> None:
    site_file = tmp_path / "bad.tsv"
    site_file.write_text("site_id\tseqid\tposition\tref\tedited\tstrand\nx\tchrM\t0\tC\tT\t+\n")

    result = quantify_known_site_efficiency(tmp_path / "missing.bam", site_file)

    assert result.status == "failed"
    assert result.errors[0].code == "rna_editing.invalid_sites"


def test_efficiency_capability_is_discoverable_from_core() -> None:
    from organelleverse.capabilities.discovery import discover_capabilities

    index = discover_capabilities()
    capability = index.describe("rna_editing.quantify_efficiency")

    assert capability.capability_id == "rna_editing.quantify_efficiency"
    assert capability.origins[0].channel == "core"
