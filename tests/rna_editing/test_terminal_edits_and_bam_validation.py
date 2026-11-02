"""Start/stop-gain edits only at terminal codons; a BAM is actually read when validating."""

from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.io_genome import read_genbank_genome
from organelleverse.rna_editing.rna_edit import predict_edits, predict_edits_prep, validate_edits


def _genbank(path: Path, sequence: str, features: list[tuple[str, int, int]]) -> Path:
    lines = [
        f"LOCUS       fake          {len(sequence)} bp    DNA     circular     01-JAN-2026",
        "DEFINITION  synthetic.",
        "ACCESSION   FAKE",
        "VERSION     FAKE.1",
        "FEATURES             Location/Qualifiers",
    ]
    for gene, start, end in features:
        lines += [f"     CDS             {start}..{end}", f'                     /gene="{gene}"']
    lines.append("ORIGIN")
    lines += [f"{i + 1:>9} {sequence[i:i + 60]}" for i in range(0, len(sequence), 60)]
    lines.append("//")
    path.write_text("\n".join(lines) + "\n")
    return path


def _genome(tmp_path: Path, name: str, cds: str):
    sequence = "TTTTT" + cds + "TTTTT"
    path = _genbank(tmp_path / f"{name}.gb", sequence, [(name, 6, 5 + len(cds))])
    return read_genbank_genome(path, organelle="mitochondrion", species="Synthetic")


def test_internal_acg_is_not_a_start_gain(tmp_path: Path) -> None:
    # cox1-like: ATG start, an internal ACG at codon 3, TAA stop
    genome = _genome(tmp_path, "cox1", "ATGGCTACGGCTGCTTAA")
    sites = predict_edits_prep(genome).metrics["sites"]
    assert not [s for s in sites if dict(s).get("type") == "start_gain"]


def test_first_codon_acg_is_a_start_gain_at_its_c(tmp_path: Path) -> None:
    genome = _genome(tmp_path, "cox1", "ACGGCTACGGCTGCTTAA")
    starts = [dict(s) for s in predict_edits_prep(genome).metrics["sites"] if dict(s).get("type") == "start_gain"]
    assert len(starts) == 1
    assert starts[0]["position"] == 6 + 1  # the C of ACG, second base of the CDS
    assert starts[0]["strand"] == "+" and starts[0]["seqid"]


def test_heuristic_reads_the_codons_not_only_the_gene_name(tmp_path: Path) -> None:
    plain = _genome(tmp_path, "atp9", "ATGGCTGCTGCTTAA")
    assert predict_edits(plain, backend="heuristic").metrics["candidate_sites"] == 0

    stop_gain = _genome(tmp_path, "atp9b", "ATGGCTGCTGCTCGA")
    sites = [dict(s) for s in predict_edits(stop_gain, backend="heuristic").metrics["sites"]]
    assert [(s["kind"], s["position"], s["confidence"]) for s in sites] == [("stop_gain", 6 + 12, "medium")]


def test_a_file_that_is_not_a_bam_is_not_validated(tmp_path: Path) -> None:
    not_bam = tmp_path / "x.bam"
    not_bam.write_text("[project]\nname = 'x'\n")
    result = validate_edits([{"seqid": "chrM", "position": 11, "strand": "+"}], bam_path=not_bam)
    assert result.status == "failed"


def test_bam_reads_decide_support(tmp_path: Path) -> None:
    rubam = pytest.importorskip("rubam")
    from tests.bam_fixtures import bam_header

    bam = tmp_path / "reads.bam"
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chrM", "LN": 100}]}
    with rubam.AlignmentFile(bam, "wb", header=bam_header(header, tmp_path)) as out:
        for index, base in enumerate(["C"] * 6 + ["T"] * 4):
            read = rubam.AlignedSegment()
            read.query_name = f"r{index}"
            sequence = list("A" * 20)
            sequence[10] = base
            read.query_sequence = "".join(sequence)
            read.flag = 0
            read.reference_id = 0
            read.reference_start = 0
            read.mapping_quality = 60
            read.cigartuples = ((0, 20),)
            read.query_qualities = rubam.qualitystring_to_array("I" * 20)
            out.write(read)
    rubam.index(str(bam))

    sites = [
        {"seqid": "chrM", "position": 11, "strand": "+"},  # 4/10 reads edited
        {"seqid": "chrM", "genome_position": 5, "strand": "+"},  # no T there
    ]
    result = validate_edits(sites, bam_path=bam)
    assert result.status == "ok"
    assert result.metrics["validated"] == 1
