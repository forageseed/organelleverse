from __future__ import annotations

import csv
from pathlib import Path

import pysam

from organelleverse.capabilities.parser import parse_capability_bundle
from organelleverse.rna_seq.quantification import (
    quantify_exon_expression,
    quantify_splicing_efficiency,
)


def _bam(tmp_path):
    bam_path = tmp_path / "reads.bam"
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chrM", "LN": 200}, {"SN": "chrN", "LN": 200}]}
    with pysam.AlignmentFile(str(bam_path), "wb", header=header) as out:
        # Exact splice across chrM: positions 41-50 (1-based), 10 bp intron.
        for name, start, cigar, flag in [
            ("spliced", 29, [(0, 12), (3, 10), (0, 12)], 0),
            ("unspliced", 29, [(0, 34)], 0),
            ("other-junction", 29, [(0, 12), (3, 9), (0, 13)], 0),
            ("duplicate", 29, [(0, 12), (3, 10), (0, 12)], 1024),
        ]:
            read = pysam.AlignedSegment()
            read.query_name, read.query_sequence = name, "A" * sum(n for op, n in cigar if op in (0, 1, 4, 7, 8))
            read.flag, read.reference_id, read.reference_start = flag, 0, start
            read.mapping_quality, read.cigartuples = 60, cigar
            read.query_qualities = pysam.qualitystring_to_array("I" * len(read.query_sequence))
            out.write(read)
        # One read in an unrelated nuclear contig contributes to CPM denominator.
        read = pysam.AlignedSegment()
        read.query_name, read.query_sequence = "nuclear", "A" * 20
        read.flag, read.reference_id, read.reference_start = 0, 1, 5
        read.mapping_quality, read.cigartuples = 60, [(0, 20)]
        read.query_qualities = pysam.qualitystring_to_array("I" * 20)
        out.write(read)
    pysam.index(str(bam_path))
    return bam_path


def _write_table(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_splicing_exact_junction_and_unspliced_intron(tmp_path):
    bam = _bam(tmp_path)
    sites = _write_table(tmp_path / "introns.tsv",
        ["intron_id", "gene_id", "seqid", "start", "end", "strand"],
        [{"intron_id": "i1", "gene_id": "nad1", "seqid": "chrM", "start": 42, "end": 51, "strand": "-"}])
    result = quantify_splicing_efficiency(bam, sites, scope="mitochondrion", min_anchor=8)
    row = result.metrics["introns"][0]
    assert row["spliced_reads"] == 1  # default excludes PCR duplicate
    assert row["unspliced_reads"] == 1
    assert row["excision_efficiency"] == 0.5
    assert row["strand"] == "-"


def test_exon_gene_read_counts_and_library_cpm(tmp_path):
    bam = _bam(tmp_path)
    exons = _write_table(tmp_path / "exons.tsv",
        ["feature_id", "gene_id", "seqid", "start", "end", "strand"],
        [{"feature_id": "e1", "gene_id": "nad1", "seqid": "chrM", "start": 30, "end": 70, "strand": "-"},
         {"feature_id": "e2", "gene_id": "nad1", "seqid": "chrM", "start": 71, "end": 100, "strand": "-"}])
    result = quantify_exon_expression(bam, exons, scope="mitochondrion")
    row = result.metrics["genes"][0]
    assert row["gene_id"] == "nad1"
    assert row["read_count"] == 3
    assert row["cpm"] == 750000.0
    assert result.metrics["library_mapped_reads"] == 4


def test_exon_expression_streams_alignments_without_query_name_collisions(tmp_path):
    bam = tmp_path / "reused-names.bam"
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chrM", "LN": 200}]}
    with pysam.AlignmentFile(str(bam), "wb", header=header) as out:
        for start in (10, 50):
            read = pysam.AlignedSegment()
            read.query_name = "same-name"
            read.query_sequence = "A" * 20
            read.flag, read.reference_id, read.reference_start = 0, 0, start
            read.mapping_quality, read.cigartuples = 60, [(0, 20)]
            read.query_qualities = pysam.qualitystring_to_array("I" * 20)
            out.write(read)
    pysam.index(str(bam))
    exons = _write_table(
        tmp_path / "exons.tsv",
        ["feature_id", "gene_id", "seqid", "start", "end", "strand"],
        [
            {"feature_id": "e1", "gene_id": "nad1", "seqid": "chrM", "start": 1, "end": 40, "strand": "+"},
            {"feature_id": "e2", "gene_id": "nad1", "seqid": "chrM", "start": 41, "end": 100, "strand": "+"},
        ],
    )

    result = quantify_exon_expression(bam, exons)

    assert result.metrics["genes"][0]["read_count"] == 2
    assert result.metrics["library_mapped_reads"] == 2


def test_invalid_strand_fails_clearly(tmp_path):
    bam = _bam(tmp_path)
    introns = _write_table(tmp_path / "bad.tsv",
        ["intron_id", "gene_id", "seqid", "start", "end", "strand"],
        [{"intron_id": "i1", "gene_id": "nad1", "seqid": "chrM", "start": 42, "end": 51, "strand": "."}])
    assert quantify_splicing_efficiency(bam, introns).status == "failed"


def test_capability_bundles_resolve_to_quantifiers():
    bundle_root = Path(__file__).parents[2] / "src" / "organelleverse" / "capabilities"
    bundles = sorted(bundle_root.glob("rna-seq-*/capability.toml"))
    assert {parse_capability_bundle(path).capability.id for path in bundles} == {
        "rna_seq.quantify_splicing_efficiency",
        "rna_seq.quantify_exon_expression",
    }
