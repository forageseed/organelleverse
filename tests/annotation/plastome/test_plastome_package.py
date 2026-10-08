"""Plastome backend package tests: references, tool resolution, hit parsing.

The full-pipeline smoke test (real chloroplast genome through LOSAT/NCBI) lives
in the benchmark scripts under ``scripts/annotation_benchmarks/``; these tests
pin the importable surface and the unit-level behavior of the restored package.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.annotation.plastome import (
    PlastomeAnnotationPipeline,
    default_plastome_reference_dir,
    find_blast_tools,
)
from organelleverse.annotation.plastome.blast import parse_blast_hits
from organelleverse.annotation.plastome.models import BlastTools, ReferenceFeature, ReferenceQuery


def test_reference_directory_is_packaged_and_complete() -> None:
    reference_dir = Path(default_plastome_reference_dir())
    assert reference_dir.is_dir()
    genbank_files = sorted(reference_dir.glob("*.gb"))
    # 31 reference plastomes restored from the pre-deletion tree.
    assert len(genbank_files) >= 30
    assert (reference_dir / "product.txt").is_file()


def test_rrna_reference_fastas_are_packaged() -> None:
    rrna_dir = Path(default_plastome_reference_dir()).parent / "rrna_refs"
    for name in ("rrn16.fasta", "rrn23.fasta", "rrn45.fasta", "rrn5.fasta"):
        assert (rrna_dir / name).is_file(), name


def test_find_blast_tools_reports_losat_field() -> None:
    tools = find_blast_tools(None, None, None)
    assert isinstance(tools, BlastTools)
    assert hasattr(tools, "losat")
    # With no explicit LOSAT the field may be None; NCBI resolution stays unchanged.
    assert tools.blastn is None or Path(tools.blastn).is_file()


def test_find_blast_tools_accepts_explicit_losat(tmp_path: Path) -> None:
    fake = tmp_path / "LOSAT"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    tools = find_blast_tools(None, None, None, losat_path=fake)
    assert tools.losat == str(fake)


def _query(query_id: str, group: str, seq: str, gene: str = "rbcL") -> ReferenceQuery:
    from Bio.SeqFeature import SeqFeature

    feature = SeqFeature(type="CDS", qualifiers={"gene": [gene]})
    reference_feature = ReferenceFeature(
        feature_id=f"{gene}_1",
        feature=feature,
        sequence=seq,
        gene=gene,
        feature_type="CDS",
        reference_name="Reference",
    )
    return ReferenceQuery(query_id=query_id, group=group, sequence=seq, reference_feature=reference_feature)


def test_parse_blast_hits_accepts_ncbi_14_column_output(tmp_path: Path) -> None:
    path = tmp_path / "hits.tsv"
    path.write_text(
        "rbcL|Ath\ttarget\t95.500\t500\t22\t0\t1\t500\t100\t599\t0.0\t900\t500\t154478\n"
        "rbcL|Ath\ttarget\t50.000\t100\t50\t0\t1\t100\t7000\t7100\t1e-5\t80\t500\t154478\n"
    )
    queries = [_query("rbcL|Ath", "reference4", "M" * 500)]
    hits = parse_blast_hits(path, min_identity=0.9, qcoverage_range=(0.5, 2.0), queries=queries)
    assert set(hits) == {"rbcL|Ath"}
    hit = hits["rbcL|Ath"]
    assert hit.start == 99 and hit.end == 599 and hit.strand == 1
    assert hit.qcov == 1.0 and hit.pident == 95.5
    assert hit.qlen == 500


def test_parse_blast_hits_accepts_losat_12_column_output(tmp_path: Path) -> None:
    """LOSAT emits the fixed NCBI default 12 columns (no qlen/slen)."""
    path = tmp_path / "hits.tsv"
    path.write_text(
        "rbcL|Ath\ttarget\t95.500\t500\t22\t0\t1\t500\t100\t599\t0.0\t900\n"
        "rbcL|Ath\ttarget\t40.000\t80\t48\t0\t1\t80\t7000\t7079\t1e-3\t40\n"
    )
    queries = [_query("rbcL|Ath", "reference4", "M" * 500)]
    hits = parse_blast_hits(path, min_identity=0.9, qcoverage_range=(0.5, 2.0), queries=queries)
    assert set(hits) == {"rbcL|Ath"}
    hit = hits["rbcL|Ath"]
    # qlen recovered from the query sequence, not a column.
    assert hit.qlen == 500 and hit.qcov == 1.0
    assert hit.start == 99 and hit.end == 599


def test_pipeline_dependency_check_passes_with_losat_only(tmp_path: Path) -> None:
    """LOSAT alone satisfies the search dependency (no NCBI BLAST+ needed)."""
    fake_losat = tmp_path / "LOSAT"
    fake_losat.write_text("#!/bin/sh\n")
    fake_losat.chmod(0o755)
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    pipeline = PlastomeAnnotationPipeline(
        input_dir=str(input_dir),
        output_dir=str(tmp_path / "out"),
        reference_dir=str(default_plastome_reference_dir()),
        losat_path=str(fake_losat),
    )
    assert pipeline.tools.losat == str(fake_losat)
    assert pipeline.verify_dependencies()
    assert pipeline.dependency_issues == []
