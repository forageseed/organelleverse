"""Exon queries follow the complete CDS reading frame on either strand."""
from pathlib import Path

import pytest
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from organelleverse.annotation.plastome.models import BlastHit, ReferenceFeature, ReferenceRecord
from organelleverse.annotation.plastome.pipeline import _protein_extended_parts
from organelleverse.annotation.plastome.references import build_plastome_queries


@pytest.mark.parametrize("strand", [1, -1])
def test_split_codon_queries_and_target_exons_preserve_cds_sequence(strand):
    sequence = "ATGG" + "C" * 20 + "CTAAATAA"
    locations = [(0, 4), (24, 32)]
    if strand == -1:
        sequence = str(Seq(sequence).reverse_complement())
        locations = [(32 - end, 32 - start) for start, end in locations]
    parts = [FeatureLocation(start, end, strand=strand) for start, end in locations]
    feature = SeqFeature(CompoundLocation(parts), type="CDS", qualifiers={"gene": ["test"]})
    record = SeqRecord(Seq(sequence))
    exons = tuple(ReferenceFeature(f"test:exon{i}", feature, str(part.extract(record.seq)), "test", "CDS_exon", "test", i, 2) for i, part in enumerate(parts, 1))
    reference = ReferenceRecord(Path("test.gb"), record, exons)
    queries = build_plastome_queries(reference)
    assert [q.sequence for q in queries] == ["M", "K"]
    peptide_parts = [(0, 3), (26, 29)]
    if strand == -1:
        peptide_parts = [(32 - end, 32 - start) for start, end in peptide_parts]
    hits = [BlastHit(q.query_id, 100, 1, start, end, strand, 20, 1, 1, 1, 1, 1) for q, (start, end) in zip(queries, peptide_parts, strict=True)]
    restored = _protein_extended_parts(list(zip(queries, hits, strict=True)))
    assert restored == parts
    assert str(CompoundLocation(restored).extract(record.seq)) == "ATGGCTAAATAA"


@pytest.mark.parametrize("strand", [1, -1])
def test_mixed_strand_exons_keep_biological_order(strand):
    locations = [FeatureLocation(0, 4, strand=strand), FeatureLocation(24, 32, strand=-strand)]
    feature = SeqFeature(CompoundLocation(locations), type="CDS")
    exons = tuple(ReferenceFeature(f"test:exon{i}", feature, dna, "test", "CDS_exon", "test", i, 2) for i, dna in enumerate(["ATGG", "CTAAATAA"], 1))
    queries = build_plastome_queries(ReferenceRecord(Path("test.gb"), SeqRecord(Seq("C" * 32)), exons))
    assert [q.sequence for q in queries] == ["M", "K"]
