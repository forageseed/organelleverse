"""CSV / FASTA / JSONL export, and the baseline-resume cache. Offline."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.fetch.cache import (
    append_records,
    merge_with_baseline,
    read_baseline,
    read_jsonl,
    write_jsonl,
)
from organelleverse.fetch.export import genbank_to_fasta, merge_metadata_csv, write_metadata_csv

GBFF = """LOCUS       NC_000932  20 bp    DNA     circular PLN
VERSION     NC_000932.1
ORIGIN
        1 acgtacgtac gtacgtacgt
//
LOCUS       OP474144  10 bp    DNA     circular PLN
VERSION     OP474144.1
ORIGIN
        1 ttttaaaacc
//
"""


# ─────────────────────────────────────────────────────────────
# FASTA
# ─────────────────────────────────────────────────────────────


def test_genbank_streams_out_as_fasta(tmp_path: Path) -> None:
    gb = tmp_path / "in.gbff"
    gb.write_text(GBFF)
    fasta = tmp_path / "out.fasta"

    assert genbank_to_fasta(gb, fasta) == 2
    lines = fasta.read_text().splitlines()
    assert lines[0] == ">NC_000932.1"
    assert lines[1] == "ACGTACGTACGTACGTACGT"
    assert lines[2] == ">OP474144.1"
    assert lines[3] == "TTTTAAAACC"


def test_fasta_export_can_be_restricted_to_a_subset(tmp_path: Path) -> None:
    """Subsetting must not require re-downloading anything."""
    gb = tmp_path / "in.gbff"
    gb.write_text(GBFF)
    fasta = tmp_path / "subset.fasta"

    assert genbank_to_fasta(gb, fasta, accessions=["OP474144.1"]) == 1
    assert ">NC_000932.1" not in fasta.read_text()
    assert ">OP474144.1" in fasta.read_text()


def test_fasta_wraps_long_sequences(tmp_path: Path) -> None:
    gb = tmp_path / "in.gbff"
    gb.write_text(GBFF)
    fasta = tmp_path / "wrapped.fasta"
    genbank_to_fasta(gb, fasta, line_width=5)
    assert "ACGTA\nCGTAC\n" in fasta.read_text()


# ─────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────


def test_metadata_csv_leads_with_the_columns_a_human_reads_first(tmp_path: Path) -> None:
    out = write_metadata_csv(
        [{"accession": "B.1", "length": 2, "organism": "Zea mays", "weird": "x"}],
        tmp_path / "m.csv",
    )
    header = out.read_text().splitlines()[0].split(",")
    assert header[:3] == ["accession", "organism", "length"]
    assert header[-1] == "weird"  # unexpected columns survive, just at the end


def test_metadata_csv_is_sorted_by_accession(tmp_path: Path) -> None:
    out = write_metadata_csv([{"accession": "Z.1"}, {"accession": "A.1"}], tmp_path / "m.csv")
    with out.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r["accession"] for r in rows] == ["A.1", "Z.1"]


def test_merging_two_metadata_tables_lets_the_newer_one_win(tmp_path: Path) -> None:
    first = write_metadata_csv([{"accession": "A.1", "length": "1"}], tmp_path / "a.csv")
    second = write_metadata_csv([{"accession": "A.1", "length": "999"}], tmp_path / "b.csv")
    merged = merge_metadata_csv(first, second, tmp_path / "m.csv")
    with merged.open() as handle:
        (row,) = list(csv.DictReader(handle))
    assert row["length"] == "999"


# ─────────────────────────────────────────────────────────────
# JSONL cache / resume — a 58k pull WILL be interrupted
# ─────────────────────────────────────────────────────────────


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = write_jsonl([{"accession": "A.1"}, {"accession": "B.1"}], tmp_path / "m.jsonl")
    assert [r["accession"] for r in read_jsonl(path)] == ["A.1", "B.1"]


def test_a_truncated_final_line_is_dropped_not_fatal(tmp_path: Path) -> None:
    """A killed process leaves a half-written line. That is expected, not corruption."""
    path = tmp_path / "m.jsonl"
    path.write_text('{"accession": "A.1"}\n{"accession": "B.1"\n')
    assert [r["accession"] for r in read_jsonl(path)] == ["A.1"]


def test_append_keeps_what_an_interrupted_run_already_had(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    append_records([{"accession": "A.1"}], path)
    append_records([{"accession": "B.1"}], path)
    assert len(read_jsonl(path)) == 2


def test_baseline_reads_a_plain_accession_list(tmp_path: Path) -> None:
    path = tmp_path / "have.txt"
    path.write_text("# mine\naccession\nNC_000932.1\nOP474144.1\n")
    assert read_baseline(path) == {"NC_000932.1", "OP474144.1"}


def test_baseline_reads_jsonl_metadata(tmp_path: Path) -> None:
    path = write_jsonl([{"accession": "NC_000932.1"}], tmp_path / "m.jsonl")
    assert read_baseline(path) == {"NC_000932.1"}


def test_a_missing_baseline_is_an_error_not_an_empty_set(tmp_path: Path) -> None:
    """Silently treating a typo'd path as 'nothing cached' re-downloads everything."""
    with pytest.raises(OrganelleInputError):
        read_baseline(tmp_path / "nope.jsonl")


def test_merge_with_baseline_prefers_the_freshly_fetched_record(tmp_path: Path) -> None:
    """NCBI updates records in place, so the new copy is the truthful one."""
    baseline = write_jsonl([{"accession": "A.1", "length": 1}], tmp_path / "base.jsonl")
    merged = merge_with_baseline(
        baseline, [{"accession": "A.1", "length": 999}], tmp_path / "out.jsonl"
    )
    assert merged == [{"accession": "A.1", "length": 999}]
