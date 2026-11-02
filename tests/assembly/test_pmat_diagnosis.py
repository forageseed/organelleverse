from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleInputError


@pytest.fixture
def diagnosis(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "scripts/assembly_benchmarks")
    )
    return importlib.import_module("diagnose_pmat_outputs")


def test_reconstruction_reports_unavailable_contigs_and_links(tmp_path, diagnosis):
    fasta = tmp_path / "contigs.fa"
    fasta.write_text(">contig00001\nAACG\n>contig00002\nTTGA\n")
    graph = tmp_path / "graph.txt"
    graph.write_text(
        "1\tcontig00001\t4\t10\n2\tcontig00002\t4\t10\n3\tcontig00003\t2\t1\n"
        "C\t1\t3'\t2\t5'\t10\nC\t2\t3'\t3\t5'\t1\n"
    )
    out = tmp_path / "available.gfa"
    report = diagnosis.reconstruct_available_graph(fasta, graph, out)
    assert report["scope"] == "available_sequence_induced_subgraph"
    assert report["missing_sequence_ids"] == ["3"]
    assert report["input_connections"] == 2
    assert report["omitted_connections"] == [["C", "2", "3'", "3", "5'", "1"]]
    # Reproduce the upstream export, not a reference-dependent corrected graph.
    assert "L\t1\t-\t2\t-\t0M\n" in out.read_text()


@pytest.mark.parametrize(
    "left,right,source,target",
    [
        ("3'", "5'", "-", "-"),
        ("5'", "3'", "+", "+"),
        ("3'", "3'", "-", "+"),
        ("5'", "5'", "+", "-"),
    ],
)
def test_orientation_audit_preserves_sequences_and_metadata(
    tmp_path, diagnosis, left, right, source, target
):
    fasta = tmp_path / "contigs.fa"
    fasta.write_text(">contig00001\nAACG\n>contig00002\nTTGA\n")
    graph = tmp_path / "graph.txt"
    graph.write_text(f"C\t1\t{left}\t2\t{right}\t10\n")
    exported, audited = tmp_path / "exported.gfa", tmp_path / "audited.gfa"
    report = diagnosis.reconstruct_available_graph(fasta, graph, exported)
    assert report["scope"] == "complete"
    before = exported.read_text()
    assert f"L\t1\t{source}\t2\t{target}\t0M\n" in before
    diagnosis.audit_link_orientation(exported, audited)
    flip = {"+": "-", "-": "+"}
    assert audited.read_text() == before.replace(
        f"L\t1\t{source}\t2\t{target}\t0M",
        f"L\t1\t{flip[source]}\t2\t{flip[target]}\t0M",
    )
    assert exported.read_text() == before


def test_reconstruction_rejects_unknown_endpoint_convention(tmp_path, diagnosis):
    fasta, graph = tmp_path / "contigs.fa", tmp_path / "graph.txt"
    fasta.write_text(">contig00001\nAACG\n")
    graph.write_text("C\t1\tunknown\t1\t5'\t1\n")
    with pytest.raises(OrganelleInputError, match="Invalid PMAT connection"):
        diagnosis.reconstruct_available_graph(fasta, graph, tmp_path / "out.gfa")
