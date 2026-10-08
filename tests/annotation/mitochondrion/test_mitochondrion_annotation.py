"""Mitochondrial annotation functions reused inside organelleverse.annotation.

These tests are adapted from the original mitochondrial annotation tests and
use organelleverse.annotation modules directly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from organelleverse.annotation.mitochondrion.boundary import (
    _correct_start_codon_conservative,
    _correct_stop_codon_conservative,
    _refine_boundary_by_tblastn,
    _restore_phase_continuity,
)
from organelleverse.annotation.mitochondrion.db import DBManager
from organelleverse.annotation.mitochondrion.fasta import load_fasta, validate_fasta
from organelleverse.annotation.mitochondrion.models.gene import (
    ExonRecord,
    GeneAnnotation,
    Strand,
)
from organelleverse.annotation.mitochondrion.models.genome import GenomeSequence
from organelleverse.annotation.mitochondrion.trans_splicing import (
    TRANS_SPLICED_CONFIG,
    merge_exons_to_gene,
    parse_exon_id,
)
from organelleverse.annotation.mitochondrion.trna import (
    RawTRNA,
    _parse_aragorn_output,
    _parse_trna_name,
    _run_aragorn,
    _standardize_trna_name,
)


@pytest.fixture
def mitochondrion_tiny_fasta(tmp_path: Path) -> Path:
    seq = "ATG" + "GCN" * 100 + "ATG" + "GCN" * 200 + "TAA" + "N" * 200
    fasta_path = tmp_path / "test_mito.fasta"
    record = SeqRecord(Seq(seq), id="test_contig1", description="test mitochondrion")
    SeqIO.write(record, str(fasta_path), "fasta")
    return fasta_path


@pytest.fixture
def mitochondrion_multi_contig_fasta(tmp_path: Path) -> Path:
    fasta_path = tmp_path / "test_multicontig.fasta"
    records = [
        SeqRecord(Seq("ATGCGN" * 50), id="contig1", description=""),
        SeqRecord(Seq("ATGCGN" * 30), id="contig2", description=""),
        SeqRecord(Seq("ATGCGN" * 20), id="contig3", description=""),
    ]
    SeqIO.write(records, str(fasta_path), "fasta")
    return fasta_path


def _create_mock_genome(length: int = 1_000_000) -> GenomeSequence:
    return GenomeSequence(
        seqid="mock_genome",
        sequence="A" * length,
        is_circular=True,
    )


def test_packaged_mitochondrion_db_is_available():
    db = DBManager()

    assert db.verify() == []
    assert db.combined_hmm.exists()
    assert len(list(db.hmm_dir.glob("*.hmm"))) == 49
    assert (db.hmm_dir / "rnaseh.hmm").exists()


def test_load_single_contig(mitochondrion_tiny_fasta: Path):
    genome = load_fasta(mitochondrion_tiny_fasta)

    assert genome.seqid == "test_contig1"
    assert len(genome.sequence) > 0
    assert genome.gc_content > 0
    assert genome.contig_map is None


def test_load_multi_contig(mitochondrion_multi_contig_fasta: Path):
    genome = load_fasta(mitochondrion_multi_contig_fasta)

    assert genome.contig_map is not None
    assert len(genome.contig_map) == 3
    assert genome.contig_map[0].original_id == "contig1"
    assert "N" * 200 in genome.sequence


def test_validate_fasta_reports_short_sequence(mitochondrion_tiny_fasta: Path):
    genome = load_fasta(mitochondrion_tiny_fasta)
    warnings = validate_fasta(genome)

    assert len(genome.sequence) < 10_000
    assert any("short" in warning.lower() for warning in warnings)


def test_load_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        load_fasta("/nonexistent/path.fasta")


def test_reverse_complement(mitochondrion_tiny_fasta: Path):
    genome = load_fasta(mitochondrion_tiny_fasta)
    rc = genome.reverse_complement
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}

    assert len(rc) == len(genome.sequence)
    assert rc[-1] == comp[genome.sequence[0]]


def test_trna_name_standardization_uses_ncbi_format():
    test_cases = [
        ("I", "AAU", "trnI(aat)"),
        ("F", "GAA", "trnF(gaa)"),
        ("M", "CAU", "trnM(cat)"),
        ("L", "UAA", "trnL(taa)"),
        ("S", "GCU", "trnS(gct)"),
        ("fMet", "CAU", "trnM(cat)"),
    ]

    for aa, anticodon, expected in test_cases:
        assert _standardize_trna_name(aa, anticodon) == expected


def test_trna_name_roundtrip_and_legacy_conversion():
    aa, anticodon = _parse_trna_name("trnI(aat)")
    assert (aa, anticodon) == ("I", "AAT")
    assert _standardize_trna_name(aa, anticodon) == "trnI(aat)"

    aa, anticodon = _parse_trna_name("trnI-AAU")
    assert (aa, anticodon) == ("I", "AAU")
    assert _standardize_trna_name(aa, anticodon) == "trnI(aat)"

    aa, anticodon = _parse_trna_name("RefVi3086_tRNA_trnfM-CAU_0001_0515")
    assert (aa, anticodon) == ("fM", "CAU")
    assert _standardize_trna_name(aa, anticodon) == "trnfM(cat)"


def test_trna_raw_name_matches_ncbi_format():
    gene_name = _standardize_trna_name("I", "AAU")
    mito_trna = RawTRNA(
        gene_name=gene_name,
        start=1000,
        end=1070,
        strand=1,
        anticodon="AAT",
        amino_acid="I",
        score=50,
        source="ARAGORN",
    )

    assert mito_trna.gene_name.lower() == "trnI(aat)".lower()


def test_aragorn_parser_reads_batch_output(tmp_path: Path):
    output = tmp_path / "aragorn.txt"
    output.write_text(
        "\n".join(
            [
                ">mini",
                "1   tRNA-Phe                        [1,74]\t35  \t(gaa)",
                "5   tRNA-Asp                    c[45367,45440]\t35  \t(gtc)",
            ]
        )
    )

    hits = _parse_aragorn_output(output)

    assert [(hit.gene_name, hit.start, hit.end, hit.strand, hit.score) for hit in hits] == [
        ("trnF(gaa)", 1, 74, 1, 35.0),
        ("trnD(gtc)", 45367, 45440, -1, 35.0),
    ]


def test_aragorn_runner_searches_trna_not_tmrna(tmp_path: Path, monkeypatch):
    from organelleverse.annotation.mitochondrion import trna

    fasta = tmp_path / "mini.fasta"
    fasta.write_text(">mini\nACGT\n")

    def fake_run(cmd, capture_output=True, text=True, timeout=300):
        assert "-t" in cmd
        assert "-m" not in cmd
        assert "-gcstd" in cmd
        output = Path(cmd[cmd.index("-o") + 1])
        output.write_text(">mini\n1   tRNA-Phe                        [1,74]\t35  \t(gaa)\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(
        trna.shutil, "which", lambda name: "/usr/bin/aragorn" if name == "aragorn" else None
    )
    monkeypatch.setattr(trna.subprocess, "run", fake_run)

    hits = _run_aragorn(fasta, tmp_path)

    assert [hit.gene_name for hit in hits] == ["trnF(gaa)"]


def test_parse_trans_spliced_exon_ids():
    assert parse_exon_id("ArthCpNC-037304_cds181_nad5_1_230") == (
        "nad5",
        1,
        230,
    )
    assert parse_exon_id("refCp_cds46_rps12_2_232") == ("rps12", 2, 232)
    assert parse_exon_id("invalid") is None
    assert parse_exon_id("prefix_gene_abc_xyz") is None


def test_merge_trans_spliced_exons_complete_gene():
    genome = _create_mock_genome()
    exon_hits = {
        1: [(1000, 1230, Strand.PLUS, 100.0, 230, 230)],
        2: [(5000, 6216, Strand.PLUS, 98.5, 1216, 1216)],
        3: [(10000, 10022, Strand.PLUS, 100.0, 22, 22)],
        4: [(20000, 20395, Strand.PLUS, 95.0, 395, 395)],
        5: [(30000, 30147, Strand.PLUS, 99.0, 147, 147)],
    }

    result = merge_exons_to_gene(
        "nad5",
        exon_hits,
        TRANS_SPLICED_CONFIG["nad5"],
        genome,
    )

    assert result is not None
    assert result.gene_name == "nad5"
    assert result.genomic_start == 1000
    assert result.genomic_end == 30147
    assert len(result.exons) == 5


def test_merge_trans_spliced_exons_rejects_incomplete_or_overwide_hits():
    genome = _create_mock_genome(2_000_000)
    incomplete_hits = {
        1: [(1000, 1230, Strand.PLUS, 100.0, 230, 230)],
        3: [(10000, 10022, Strand.PLUS, 100.0, 22, 22)],
        5: [(30000, 30147, Strand.PLUS, 99.0, 147, 147)],
    }
    overwide_hits = {
        1: [(1000, 1800, Strand.PLUS, 100.0, 800, 800)],
        2: [(600000, 615000, Strand.PLUS, 98.5, 1500, 1500)],
    }

    assert (
        merge_exons_to_gene(
            "nad5",
            incomplete_hits,
            TRANS_SPLICED_CONFIG["nad5"],
            genome,
        )
        is None
    )
    assert (
        merge_exons_to_gene(
            "cox2",
            overwide_hits,
            TRANS_SPLICED_CONFIG["cox2"],
            genome,
        )
        is None
    )


def test_merge_trans_spliced_exons_selects_best_hit_and_renumbers():
    genome = _create_mock_genome()
    exon_hits = {
        5: [(30000, 30147, Strand.PLUS, 99.0, 147, 147)],
        1: [
            (1000, 1230, Strand.PLUS, 90.0, 230, 230),
            (2000, 2100, Strand.PLUS, 100.0, 100, 230),
        ],
        3: [(10000, 10022, Strand.PLUS, 100.0, 22, 22)],
        2: [(5000, 6216, Strand.PLUS, 98.5, 1216, 1216)],
        4: [(20000, 20395, Strand.PLUS, 95.0, 395, 395)],
    }

    result = merge_exons_to_gene(
        "nad5",
        exon_hits,
        TRANS_SPLICED_CONFIG["nad5"],
        genome,
    )

    assert result is not None
    assert [(exon.number, exon.start) for exon in result.exons] == [
        (1, 1000),
        (2, 5000),
        (3, 10000),
        (4, 20000),
        (5, 30000),
    ]


def test_trans_spliced_config_core_values():
    for gene in ("nad1", "nad2", "nad5", "nad4", "nad7", "cox2", "rps3", "cox1"):
        assert gene in TRANS_SPLICED_CONFIG

    assert TRANS_SPLICED_CONFIG["nad5"]["exons"] == 5
    assert TRANS_SPLICED_CONFIG["nad5"]["max_span"] == 2_000_000
    assert TRANS_SPLICED_CONFIG["nad5"]["min_exon_bp"] == 20
    assert TRANS_SPLICED_CONFIG["nad1"]["exons"] == 5
    assert TRANS_SPLICED_CONFIG["nad1"]["max_span"] == 1_500_000
    assert TRANS_SPLICED_CONFIG["rps3"]["max_exon_gap"] == 50_000


def test_boundary_start_codon_search_crosses_origin_plus_strand():
    seq = "GCTGCTTAA" + "N" * 18 + "ATG"
    genome = GenomeSequence(seqid="test", sequence=seq, is_circular=True)
    ann = GeneAnnotation(
        gene_name="test_gene_xyz",
        gene_type="CDS",
        exons=[ExonRecord(start=28, end=9, strand=Strand.PLUS, number=1)],
        strand=Strand.PLUS,
    )

    corrected = _correct_start_codon_conservative(
        ann,
        genome,
        DBManager(),
        search_range=10,
    )

    assert corrected.exons[0].start == 28
    assert corrected.exons[0].end == 9


def test_boundary_stop_codon_search_crosses_origin_plus_strand():
    seq = "TAA" + "GCT" * 7 + "ATG" + "GCT"
    genome = GenomeSequence(seqid="test", sequence=seq, is_circular=True)
    ann = GeneAnnotation(
        gene_name="test_gene_xyz",
        gene_type="CDS",
        exons=[ExonRecord(start=25, end=30, strand=Strand.PLUS, number=1)],
        strand=Strand.PLUS,
    )

    corrected = _correct_stop_codon_conservative(
        ann,
        genome,
        DBManager(),
        search_range=10,
    )

    assert corrected.exons[0].start == 25
    assert corrected.exons[0].end == 3


def test_phase_continuity_restoration_for_multi_exon_genes():
    # exon 1 one base short puts TAA (32..34) in frame inside the CDS; the restored frame reads ATA AAA
    genome = GenomeSequence(seqid="test", sequence="A" * 31 + "T" + "A" * 68, is_circular=False)
    ann = GeneAnnotation(
        gene_name="nad5",
        gene_type="CDS",
        exons=[
            ExonRecord(start=10, end=19, strand=Strand.PLUS, number=1, phase=0),
            ExonRecord(start=30, end=40, strand=Strand.PLUS, number=2, phase=2),
        ],
        strand=Strand.PLUS,
    )

    restored = _restore_phase_continuity(ann, genome)

    assert restored.exons[0].end == 20
    assert "phase continuity restored" in restored.notes[-1]


def test_tblastn_refinement_skips_trans_spliced_and_multi_exon_genes():
    genome = GenomeSequence(seqid="test", sequence="A" * 100, is_circular=False)
    trans_spliced = GeneAnnotation(
        gene_name="nad5",
        gene_type="CDS",
        exons=[ExonRecord(start=10, end=30, strand=Strand.PLUS, number=1)],
        strand=Strand.PLUS,
    )
    multi_exon = GeneAnnotation(
        gene_name="atp1",
        gene_type="CDS",
        exons=[
            ExonRecord(start=10, end=20, strand=Strand.PLUS, number=1),
            ExonRecord(start=30, end=40, strand=Strand.PLUS, number=2),
        ],
        strand=Strand.PLUS,
    )

    db = DBManager()
    assert _refine_boundary_by_tblastn(trans_spliced, genome, db).source_method != "tblastn"
    assert _refine_boundary_by_tblastn(multi_exon, genome, db).source_method != "tblastn"
