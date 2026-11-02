"""Compare the Rust reader with independent pysam-written BAM fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from organelleverse.operations.analysis_runners import _bam_coverage, _run_validate_rna
from organelleverse.quality_control.contracts import InputLibraryEvidence
from organelleverse.quality_control.evidence import _parse_library_bam
from organelleverse.quality_control.graph import _read_length
from organelleverse.transfer.transfer import _region_depth

pysam = pytest.importorskip("pysam")  # Reference fixture writer only.
rubam = pytest.importorskip("rubam")


@pytest.fixture()
def alignments(tmp_path: Path) -> tuple[Path, Path]:
    fasta = tmp_path / "reference.fa"
    fasta.write_text(">ctg\n" + "A" * 200 + "\n")
    sam = tmp_path / "reads.sam"
    sam.write_text(
        "@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:ctg\tLN:200\n"
        "match\t0\tctg\t11\t60\t20M\t*\t0\t0\t" + "A" * 20 + "\t" + "I" * 20 + "\tNM:i:0\n"
        "indel\t16\tctg\t31\t60\t3S5M2I4M3D6M2H\t*\t0\t0\t"
        + "A" * 20
        + "\t"
        + "I" * 20
        + "\tNM:i:5\n"
        "skip\t0\tctg\t51\t60\t5M8N5M\t*\t0\t0\t" + "A" * 10 + "\t" + "I" * 10 + "\tNM:i:0\n"
        "secondary\t256\tctg\t71\t0\t10M\t*\t0\t0\t" + "A" * 10 + "\t" + "I" * 10 + "\tNM:i:0\n"
        "supplementary\t2048\tctg\t81\t60\t10M\t*\t0\t0\t"
        + "A" * 10
        + "\t"
        + "I" * 10
        + "\tNM:i:0\n"
        "unmapped\t4\t*\t0\t0\t*\t*\t0\t0\tAAAAA\tIIIII\n"
    )
    bam = tmp_path / "reads.bam"
    pysam.sort("-o", str(bam), str(sam))
    pysam.index(str(bam))
    return bam, fasta


def test_rust_mapping_metrics_match_reference_reader(alignments, monkeypatch) -> None:
    bam, fasta = alignments
    library = InputLibraryEvidence.model_construct(
        role="fixture", technology="pacbio_hifi", quality_state="ccs"
    )
    actual = _parse_library_bam(bam, library, fasta, bam_artifact_id="fixture")
    monkeypatch.setattr(rubam, "AlignmentFile", pysam.AlignmentFile)
    expected = _parse_library_bam(bam, library, fasta, bam_artifact_id="fixture")
    assert actual == expected


def test_indexed_full_reference_matches_reference_reader(alignments) -> None:
    bam, _ = alignments
    with (
        rubam.AlignmentFile(str(bam), "rb") as rust,
        pysam.AlignmentFile(str(bam), "rb") as reference,
    ):
        actual = [
            (r.query_name, r.cigartuples, _read_length(r))
            for r in rust.fetch("ctg", 0, rust.get_reference_length("ctg"))
        ]
        expected = [
            (r.query_name, r.cigartuples, r.infer_query_length(always=True))
            for r in reference.fetch("ctg")
        ]
    assert actual == expected


def test_desktop_coverage_runs_without_pysam(alignments, monkeypatch) -> None:
    bam, _ = alignments
    monkeypatch.setitem(sys.modules, "pysam", None)
    rows = _bam_coverage(str(bam), [("ctg", "A" * 200)])
    assert rows[0]["covered_bases"] > 0
    assert rows[0]["covered_bases"] < 200  # Rust emits zero-depth columns too.
    assert rows[0]["mean_depth"] >= 1


def test_real_mapping_evidence_runs_without_pysam(alignments, monkeypatch) -> None:
    bam, fasta = alignments
    monkeypatch.setitem(sys.modules, "pysam", None)
    library = InputLibraryEvidence.model_construct(
        role="fixture", technology="pacbio_hifi", quality_state="ccs"
    )
    result = _parse_library_bam(bam, library, fasta, bam_artifact_id="fixture")
    assert result.summary.mapped_reads == 3
    assert result.summary.target_fraction == 0.75


def test_transfer_depth_counts_segments_not_n_bases(alignments) -> None:
    bam, _ = alignments
    with rubam.AlignmentFile(str(bam), "rb") as reader:
        assert _region_depth(reader, "ctg", 11, 30) == 1.0


def test_rna_junctions_and_editing_use_rust_base_counts(alignments, tmp_path, monkeypatch) -> None:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqFeature import CompoundLocation, SeqFeature, SimpleLocation
    from Bio.SeqRecord import SeqRecord

    bam, _ = alignments
    sam = tmp_path / "reads.sam"
    with sam.open("a") as handle:
        for index in range(5):
            handle.write(f"edit-{index}\t0\tctg\t111\t60\t5M\t*\t0\t0\tGGGGG\tIIIII\tNM:i:5\n")
    pysam.sort("-o", str(bam), str(sam))
    pysam.index(str(bam))
    record = SeqRecord(Seq("A" * 200), id="ctg", annotations={"molecule_type": "DNA"})
    record.features = [
        SeqFeature(CompoundLocation([SimpleLocation(0, 55), SimpleLocation(63, 200)]), type="gene")
    ]
    genbank = tmp_path / "reference.gb"
    SeqIO.write(record, genbank, "genbank")
    monkeypatch.setitem(sys.modules, "pysam", None)
    result = _run_validate_rna(
        {"genbank": str(genbank), "bam": str(bam)}, output_dir=tmp_path, log=lambda _: None
    )
    assert "junctions=1 (annotated=1)" in result["summary_text"]
    assert "editing candidates=5" in result["summary_text"]
