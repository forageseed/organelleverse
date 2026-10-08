"""Regression tests for audited scientific counting errors."""

import pytest

from organelleverse.diversity import (
    compute_neutral_tests,
    compute_nucleotide_diversity,
    neutral_tests,
    nucleotide_diversity,
)
from organelleverse.heteroplasmy import quantify_variant_fractions
from organelleverse.rna_seq.quantification import quantify_splicing_efficiency


@pytest.mark.parametrize("operation", [nucleotide_diversity, neutral_tests])
def test_public_diversity_rejects_unequal_alignments(tmp_path, operation):
    fasta = tmp_path / "unequal.fa"
    fasta.write_text(">one\nACGT\n>two\nACGTA\n")
    result = operation(fasta)
    assert result.status == "failed"
    assert result.errors[0].code.endswith(".unequal_alignment_lengths")


@pytest.mark.parametrize("operation", [compute_nucleotide_diversity, compute_neutral_tests])
def test_typed_diversity_rejects_unequal_alignments(tmp_path, operation):
    fasta = tmp_path / "unequal.fa"
    fasta.write_text(">one\nACGT\n>two\nACGTA\n")
    with pytest.raises(ValueError, match="equal lengths"):
        operation(fasta)


@pytest.mark.parametrize("operation", [neutral_tests, compute_neutral_tests])
def test_neutral_estimators_ignore_inaccessible_columns_without_moving_sites(tmp_path, operation):
    pytest.importorskip("allel")
    fasta = tmp_path / "gapped.fa"
    compact = tmp_path / "compact.fa"
    sequences = ["A--AAA", "G--GGA", "A--AGA", "G--AGT"]
    fasta.write_text("".join(f">s{i}\n{s}\n" for i, s in enumerate(sequences)))
    compact.write_text("".join(f">s{i}\n{s.replace('-', '')}\n" for i, s in enumerate(sequences)))
    result, expected = operation(fasta), operation(compact)
    metrics = result.metrics if hasattr(result, "metrics") else result
    reference = expected.metrics if hasattr(expected, "metrics") else expected
    assert metrics["comparable_sites"] == 4
    assert metrics["segregating_sites"] == 4
    assert metrics["theta_w"] == pytest.approx(6 / 11, abs=1e-6)
    for field in ("pi", "theta_w", "tajima_d"):
        assert metrics[field] == pytest.approx(reference[field], abs=1e-6)


@pytest.mark.parametrize("operation", [neutral_tests, compute_neutral_tests])
def test_neutral_late_variants_are_not_truncated_by_comparable_count(tmp_path, operation):
    pytest.importorskip("allel")
    fasta = tmp_path / "late.fa"
    fasta.write_text(">one\n----AC\n>two\n----GT\n")
    result = operation(fasta)
    metrics = result.metrics if hasattr(result, "metrics") else result
    assert metrics["theta_w"] == 1.0
    assert metrics["pi"] == 1.0
    assert metrics["segregating_sites"] == metrics["comparable_sites"] == 2


def _write_bam(tmp_path, cigar_rows):
    pysam = pytest.importorskip("pysam")
    bam_path = tmp_path / "reads.bam"
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "MT", "LN": 500}]}
    with pysam.AlignmentFile(str(bam_path), "wb", header=header) as bam:
        for i, (cigar, flag) in enumerate(cigar_rows):
            read = pysam.AlignedSegment()
            length = sum(n for op, n in cigar if op in (0, 1, 4, 7, 8))
            read.query_name = f"r{i}"
            read.query_sequence = "A" * length
            read.query_qualities = pysam.qualitystring_to_array("I" * length)
            read.flag = flag
            read.reference_id = 0
            read.reference_start = 0
            read.mapping_quality = 60
            read.cigar = cigar
            bam.write(read)
    pysam.index(str(bam_path))
    return bam_path


def test_splicing_counts_are_invariant_to_match_mismatch_cigar_encoding(tmp_path):
    bam = _write_bam(
        tmp_path,
        [
            ([(0, 160)], 0),
            ([(7, 80), (8, 1), (7, 79)], 0),
            ([(7, 80), (1, 2), (8, 1), (7, 79)], 0),
            ([(7, 80), (2, 1), (7, 79)], 0),
            ([(0, 80), (3, 10), (0, 80)], 0),
        ],
    )
    introns = tmp_path / "introns.tsv"
    introns.write_text("intron_id\tgene_id\tseqid\tstart\tend\tstrand\ni1\tnad1\tMT\t81\t90\t+\n")
    result = quantify_splicing_efficiency(bam, introns, min_anchor=8)
    assert result.status == "ok"
    row = result.metrics["introns"][0]
    assert row["spliced_reads"] == 1
    assert row["unspliced_reads"] == 3
    assert row["excision_efficiency"] == 0.25


def test_heteroplasmy_duplicate_option_controls_pileup_filter(tmp_path):
    bam = _write_bam(
        tmp_path,
        [
            ([(0, 20)], 0),
            ([(0, 20)], 1024),
            ([(0, 20)], 256),
            ([(0, 20)], 512),
            ([(0, 20)], 2048),
        ],
    )
    sites = tmp_path / "sites.tsv"
    sites.write_text("site_id\tseqid\tposition\tref\talt\ns1\tMT\t10\tA\tG\n")
    excluded = quantify_variant_fractions(bam, sites, min_depth=1, exclude_duplicates=True)
    included = quantify_variant_fractions(bam, sites, min_depth=1, exclude_duplicates=False)
    assert excluded.status == included.status == "ok"
    assert excluded.metrics["sites"][0]["ref_count"] == 1
    assert included.metrics["sites"][0]["ref_count"] == 2
    assert included.metrics["sites"][0]["alt_count"] == 0
