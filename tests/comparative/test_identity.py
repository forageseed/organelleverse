"""Unit tests for the mVISTA-style genome identity core."""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from organelleverse.capabilities.adapters import comparative as comparative_adapter
from organelleverse.comparative.identity import (
    _accumulate_piece_events,
    _anchor_blocks,
    _piece_plan,
    _window_rows,
    compute_genome_identity,
    load_reference_features,
)
from organelleverse.core.errors import OrganelleDependencyError, OrganelleInputError


def _events(reference_length: int = 16):
    matches = np.zeros(reference_length, dtype=np.int64)
    mismatches = np.zeros(reference_length, dtype=np.int64)
    deletions = np.zeros(reference_length, dtype=np.int64)
    return matches, mismatches, deletions


def test_piece_events_counts_matches_mismatches_indels() -> None:
    matches, mismatches, deletions = _events()
    edits = _accumulate_piece_events("ACGTACGT", "ACGTTCGT", 0, matches, mismatches, deletions)
    assert edits == 1
    assert matches.sum() == 7
    assert mismatches.sum() == 1
    assert mismatches[4] == 1
    assert deletions.sum() == 0


def test_piece_events_counts_deletion_and_insertion() -> None:
    matches, mismatches, deletions = _events()
    # reference has "TTT" where the query has "GG"; the insertion relative to
    # the reference consumes no reference positions.
    edits = _accumulate_piece_events("ACGTTTACGT", "ACGGGACGT", 0, matches, mismatches, deletions)
    assert edits == 3
    # 7 identical columns ("ACG" + "ACGT") are placement-invariant.
    assert matches.sum() == 7
    # edits that are not insertions land on reference positions
    assert mismatches.sum() + deletions.sum() == 3
    assert matches.sum() + mismatches.sum() == 9


def test_piece_events_empty_query_marks_all_deletions() -> None:
    matches, mismatches, deletions = _events()
    _accumulate_piece_events("ACGTACGT", "", 4, matches, mismatches, deletions)
    assert deletions.sum() == 8
    assert deletions[4] == 1 and deletions[11] == 1


def test_window_rows_identity_definition() -> None:
    matches, mismatches, deletions = _events(10)
    matches[0:4] = 1
    mismatches[4:6] = 1
    deletions[6:10] = 1
    rows = _window_rows("s", matches, mismatches, deletions, 10, window_size=5, step=5)
    assert [row["identity"] for row in rows] == [80.0, 0.0]
    assert rows[0]["covered"] == 5
    assert rows[1]["covered"] == 5
    assert rows[1]["matches"] == 0 and rows[1]["deletions"] == 4


def test_anchor_blocks_single_block_and_strand() -> None:
    base = comparative_adapter._IDENTITY_BASE
    blocks = _anchor_blocks(base, comparative_adapter._identity_query_a(base), preset="asm10")
    assert len(blocks) == 1
    assert blocks[0]["reference_start"] == 0
    assert blocks[0]["reference_end"] == len(base)
    assert blocks[0]["strand"] == 1

    blocks_reverse = _anchor_blocks(
        base, comparative_adapter._identity_query_b(base), preset="asm10"
    )
    assert len(blocks_reverse) == 1
    assert blocks_reverse[0]["strand"] == -1


def test_piece_plan_aligns_blocks_and_interblock_gaps() -> None:
    base = comparative_adapter._IDENTITY_BASE
    query = comparative_adapter._identity_query_a(base)
    blocks = _anchor_blocks(base, query, preset="asm10")
    pieces = _piece_plan(base, query, blocks)
    # one block → one piece (the synthetic pair anchors as a single block)
    assert len(pieces) == 1
    reference_piece, _query_piece, offset = pieces[0]
    assert offset == 0
    assert len(reference_piece) == len(base)


def _write_fasta(path: Path, sequence: str, name: str = "rec") -> Path:
    path.write_text(f">{name}\n{sequence}\n")
    return path


def test_compute_genome_identity_end_to_end(tmp_path: Path) -> None:
    base = comparative_adapter._IDENTITY_BASE
    reference = _write_fasta(tmp_path / "ref.fa", base, "ref")
    query_a = _write_fasta(tmp_path / "a.fa", comparative_adapter._identity_query_a(base), "a")
    query_b = _write_fasta(tmp_path / "b.fa", comparative_adapter._identity_query_b(base), "b")
    table = tmp_path / "out" / "table.tsv"

    result = compute_genome_identity(
        reference, [query_a, query_b], window_size=100, table_output=table
    )

    assert result["reference_length"] == 2400
    assert result["samples"] == ["a", "b"]
    assert len(result["windows"]) == 48
    assert len(result["segments"]) == 2
    assert {segment["strand"] for segment in result["segments"]} == {1, -1}
    assert result["features"] == []
    assert Path(result["table"]).is_file()
    lines = table.read_text().strip().splitlines()
    assert lines[0].startswith("sample\treference_start")
    assert len(lines) == 49
    # last window spans the reference tail
    assert result["windows"][-1]["end"] == 2400


def test_compute_genome_identity_default_step_equals_window(tmp_path: Path) -> None:
    base = comparative_adapter._IDENTITY_BASE
    reference = _write_fasta(tmp_path / "ref.fa", base)
    query = _write_fasta(tmp_path / "a.fa", comparative_adapter._identity_query_a(base))
    result = compute_genome_identity(reference, [query])
    assert result["window_size"] == 100
    assert result["step"] == 100


def test_compute_genome_identity_rejects_bad_inputs(tmp_path: Path) -> None:
    base = comparative_adapter._IDENTITY_BASE
    reference = _write_fasta(tmp_path / "ref.fa", base)
    query = _write_fasta(tmp_path / "a.fa", comparative_adapter._identity_query_a(base))

    with pytest.raises(OrganelleInputError, match="at least one query"):
        compute_genome_identity(reference, [])
    with pytest.raises(OrganelleInputError, match="window_size"):
        compute_genome_identity(reference, [query], window_size=0)
    with pytest.raises(OrganelleInputError, match="step"):
        compute_genome_identity(reference, [query], step=0)

    multi = tmp_path / "multi.fa"
    multi.write_text(f">r1\n{base}\n>r2\n{base}\n")
    with pytest.raises(OrganelleInputError, match="exactly one record"):
        compute_genome_identity(multi, [query])


def test_compute_genome_identity_missing_dependency_is_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = comparative_adapter._IDENTITY_BASE
    reference = _write_fasta(tmp_path / "ref.fa", base)
    query = _write_fasta(tmp_path / "a.fa", comparative_adapter._identity_query_a(base))
    monkeypatch.setitem(sys.modules, "mappy", None)
    with pytest.raises(OrganelleDependencyError, match="pip install mappy"):
        compute_genome_identity(reference, [query])


def _write_minimal_genbank(path: Path, base: str, records: int = 1) -> Path:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqFeature import SeqFeature, SimpleLocation
    from Bio.SeqRecord import SeqRecord

    bio_records = []
    for index in range(records):
        record = SeqRecord(Seq(base), id=f"REF{index + 1}", name=f"REF{index + 1}")
        record.annotations["molecule_type"] = "DNA"
        record.annotations["topology"] = "circular"
        record.features.append(SeqFeature(SimpleLocation(0, len(base)), type="source"))
        if index == 0:
            record.features.append(
                SeqFeature(
                    SimpleLocation(100, 450) + SimpleLocation(550, 980),
                    type="CDS",
                    qualifiers={"gene": ["rbcL"]},
                )
            )
            record.features.append(
                SeqFeature(
                    SimpleLocation(1500, 1571, strand=-1),
                    type="tRNA",
                    qualifiers={"gene": ["trnH-GUG"]},
                )
            )
            record.features.append(SeqFeature(SimpleLocation(450, 550), type="intron"))
            record.features.append(
                SeqFeature(
                    SimpleLocation(2000, 2100),
                    type="misc_feature",
                    qualifiers={"note": ["skipped"]},
                )
            )
        bio_records.append(record)
    SeqIO.write(bio_records, path, "genbank")
    return path


def test_load_reference_features_categorizes_and_flattens(tmp_path: Path) -> None:
    base = comparative_adapter._IDENTITY_BASE
    genbank = _write_minimal_genbank(tmp_path / "ref.gbk", base)

    rows = load_reference_features(genbank)

    by_name = {}
    for row in rows:
        by_name.setdefault(row["name"], []).append(row)
    rbcL = by_name["rbcL"]
    assert [row["category"] for row in rbcL] == ["cds", "cds"]
    assert [(row["start"], row["end"]) for row in rbcL] == [(100, 450), (550, 980)]
    assert all(row["strand"] == 1 for row in rbcL)
    trnh = by_name["trnH-GUG"]
    assert trnh[0]["category"] == "nc_gene"
    assert trnh[0]["strand"] == -1
    categories = {row["category"] for row in rows}
    assert categories == {"cds", "nc_gene", "intron"}
    # the misc_feature and source rows are skipped entirely
    assert len(rows) == 4


def test_load_reference_features_requires_single_record(tmp_path: Path) -> None:
    base = comparative_adapter._IDENTITY_BASE
    genbank = _write_minimal_genbank(tmp_path / "two.gbk", base, records=2)
    with pytest.raises(OrganelleInputError, match="exactly one record"):
        load_reference_features(genbank)


def _block(rs: int, re: int, qs: int, qe: int, strand: int = 1):
    return dict(
        reference_start=rs, reference_end=re, query_start=qs, query_end=qe, strand=strand, mapq=60
    )


@pytest.mark.parametrize("strand", [1, -1])
def test_collinear_gaps_and_tails_are_oriented_and_globally_aligned(strand: int) -> None:
    from organelleverse._sequtil import reverse_complement

    reference = "AACGTGCTAGCCATCGTACG"
    query = reference if strand == 1 else reverse_complement(reference)
    n = len(reference)
    blocks = [_block(3, 7, 3, 7), _block(12, 17, 12, 17)]
    if strand == -1:
        blocks = [
            _block(
                b["reference_start"],
                b["reference_end"],
                n - b["query_end"],
                n - b["query_start"],
                -1,
            )
            for b in blocks
        ]
    pieces = _piece_plan(reference, query, blocks)
    assert sum(len(r) for r, _, _ in pieces) == n
    assert all(r == q for r, q, _ in pieces)


def test_inversion_gap_does_not_reuse_query_or_fabricate_deletions() -> None:
    reference = "AACCGGTT" * 10
    blocks = [_block(0, 30, 0, 30), _block(40, 60, 40, 60, -1), _block(60, 80, 60, 80)]
    matches, mismatches, deletions = _events(len(reference))
    for ref, query, offset in _piece_plan(reference, reference, blocks):
        _accumulate_piece_events(ref, query, offset, matches, mismatches, deletions)
    assert not (matches + mismatches + deletions)[30:40].any()
    assert (matches + mismatches + deletions).max() == 1


@pytest.mark.parametrize("scenario", ["inversion", "inverted_repeat", "rotation", "reverse"])
def test_real_anchor_rearrangements_preserve_one_to_one_coordinates(tmp_path: Path, scenario: str):
    import random

    from organelleverse._sequtil import reverse_complement as rc

    rng = random.Random(45)
    reference = "".join(rng.choices("ACGT", k=30000))
    if scenario == "inverted_repeat":
        repeat = "".join(rng.choices("ACGT", k=5000))
        reference = reference[:10000] + repeat + reference[10000:15000] + rc(repeat)
        query = reference[:15000] + rc(reference[15000:20000]) + reference[20000:]
    elif scenario == "inversion":
        query = reference[:10000] + rc(reference[10000:20000]) + reference[20000:]
    elif scenario == "rotation":
        query = reference[10000:] + reference[:10000]
    else:
        query = rc(reference)
    result = compute_genome_identity(
        _write_fasta(tmp_path / "ref.fa", reference), [_write_fasta(tmp_path / "query.fa", query)]
    )
    for axis in ("reference", "query"):
        intervals = sorted((b[f"{axis}_start"], b[f"{axis}_end"]) for b in result["segments"])
        assert all(a[1] <= b[0] for a, b in pairwise(intervals))
    rows = result["windows"]
    # A short unanchored breakpoint is explicitly uncovered, never a false deletion.
    assert sum(w["matches"] for w in rows) >= 0.99 * len(reference)
    assert sum(w["mismatches"] + w["deletions"] for w in rows) == 0
    if scenario != "inversion":
        assert all(w["identity"] == 100 for w in rows)


def test_genbank_coordinates_must_belong_to_reference(tmp_path: Path):
    base = comparative_adapter._IDENTITY_BASE
    gb = _write_minimal_genbank(tmp_path / "ref.gb", base)
    with pytest.raises(OrganelleInputError, match="must match"):
        compute_genome_identity(
            _write_fasta(tmp_path / "r.fa", base[::-1]),
            [_write_fasta(tmp_path / "q.fa", base)],
            reference_genbank=gb,
        )


def test_duplicate_sample_names_are_rejected(tmp_path: Path):
    base = comparative_adapter._IDENTITY_BASE
    ref = _write_fasta(tmp_path / "ref.fa", base)
    with pytest.raises(OrganelleInputError, match="must be unique"):
        compute_genome_identity(ref, [ref, ref])


def test_repeat_gene_copies_have_distinct_arrow_keys(tmp_path: Path):
    from Bio import SeqIO
    from Bio.SeqFeature import SeqFeature, SimpleLocation

    gb = _write_minimal_genbank(tmp_path / "ref.gb", comparative_adapter._IDENTITY_BASE)
    record = SeqIO.read(gb, "genbank")
    record.features.append(
        SeqFeature(
            SimpleLocation(1800, 1871, strand=-1), type="tRNA", qualifiers={"gene": ["trnH-GUG"]}
        )
    )
    SeqIO.write(record, gb, "genbank")
    copies = [row for row in load_reference_features(gb) if row["name"] == "trnH-GUG"]
    assert len({row["key"] for row in copies}) == 2
