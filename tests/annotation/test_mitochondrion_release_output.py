from __future__ import annotations

from pathlib import Path

from organelleverse.annotation.genbank import parse_genbank
from organelleverse.annotation.mitochondrion.boundary import repair_complete_cds_terminals
from organelleverse.annotation.mitochondrion.cds import partition_valid_cds, validate_cds
from organelleverse.annotation.mitochondrion.db import DBManager
from organelleverse.annotation.mitochondrion.gff import write_genbank
from organelleverse.annotation.mitochondrion.models.gene import (
    ExonRecord,
    GeneAnnotation,
    Strand,
)
from organelleverse.annotation.mitochondrion.models.genome import GenomeSequence
from organelleverse.annotation.validation import validate_document


def test_non_triplet_cds_is_a_validation_error() -> None:
    genome = GenomeSequence(seqid="mito", sequence="ATGA", is_circular=False)
    annotation = GeneAnnotation(
        gene_name="atp1",
        exons=[ExonRecord(start=1, end=4, strand=Strand.PLUS)],
        strand=Strand.PLUS,
    )

    result = validate_cds([annotation], genome)

    issue = next(item for item in result.issues if item.issue_type == "frame_shift")
    assert issue.severity == "error"
    assert result.is_ok is False


def test_non_triplet_cds_is_not_automatically_exempted(tmp_path: Path) -> None:
    genome = GenomeSequence(seqid="mito", sequence="ATGA", is_circular=False)
    annotation = GeneAnnotation(
        gene_name="atp1",
        exons=[ExonRecord(start=1, end=4, strand=Strand.PLUS)],
        strand=Strand.PLUS,
    )

    output = tmp_path / "partial.gb"
    write_genbank([annotation], [], [], genome, output)
    document = parse_genbank(output)

    cds = next(
        feature
        for record in document.records
        for feature in record.features
        if feature.type == "CDS"
    )
    assert cds.qualifier_values("partial") == ()
    report = validate_document(document, ("pcg",))
    assert report.valid is False
    assert {issue.code for issue in report.errors} == {"cds_translation_inconsistent"}


def test_invalid_cds_candidate_is_rejected_without_removing_valid_copy() -> None:
    genome = GenomeSequence(
        seqid="mito",
        sequence="ATGAAATAAATGAAAA",
        is_circular=False,
    )
    valid = GeneAnnotation(
        gene_name="atp1",
        exons=[ExonRecord(start=1, end=9, strand=Strand.PLUS)],
        strand=Strand.PLUS,
    )
    invalid = GeneAnnotation(
        gene_name="atp1",
        exons=[ExonRecord(start=10, end=17, strand=Strand.PLUS)],
        strand=Strand.PLUS,
    )

    accepted, rejected = partition_valid_cds([valid, invalid], genome)

    assert accepted == [valid]
    assert len(rejected) == 1
    assert rejected[0].gene_name == "atp1"
    assert rejected[0].start == 10
    assert rejected[0].end == 17
    assert rejected[0].issue_codes == ("frame_shift", "missing_stop")


def test_internal_stop_candidate_is_rejected() -> None:
    genome = GenomeSequence(seqid="mito", sequence="ATGTAGTAA", is_circular=False)
    annotation = GeneAnnotation(
        gene_name="atp1",
        exons=[ExonRecord(start=1, end=9, strand=Strand.PLUS)],
        strand=Strand.PLUS,
    )

    accepted, rejected = partition_valid_cds([annotation], genome)

    assert accepted == []
    assert rejected[0].issue_codes == ("premature_stop",)


def test_complete_cds_without_start_or_stop_is_rejected() -> None:
    genome = GenomeSequence(seqid="mito", sequence="AAAAAATAAATGAAA", is_circular=False)
    missing_start = GeneAnnotation(
        gene_name="atp1",
        exons=[ExonRecord(start=1, end=9, strand=Strand.PLUS)],
        strand=Strand.PLUS,
    )
    missing_stop = GeneAnnotation(
        gene_name="atp1",
        exons=[ExonRecord(start=10, end=15, strand=Strand.PLUS)],
        strand=Strand.PLUS,
    )

    accepted, rejected = partition_valid_cds([missing_start, missing_stop], genome)

    assert accepted == []
    assert rejected[0].issue_codes == ("missing_start",)
    assert rejected[1].issue_codes == ("missing_stop",)


def test_rejected_start_codon_message_has_deterministic_expected_order() -> None:
    genome = GenomeSequence(seqid="mito", sequence="ATCAAATAA", is_circular=False)
    annotation = GeneAnnotation(
        gene_name="rps10",
        exons=[ExonRecord(start=1, end=9, strand=Strand.PLUS)],
        strand=Strand.PLUS,
    )

    _accepted, rejected = partition_valid_cds([annotation], genome, DBManager())

    assert rejected[0].issue_messages == ("Start codon: ATC (expected ACG/ATG)",)


def test_partial_cds_exempts_only_its_missing_terminal_codon() -> None:
    genome = GenomeSequence(seqid="mito", sequence="AAAAAATAAATGAAA", is_circular=False)
    partial_5prime = GeneAnnotation(
        gene_name="atp1",
        exons=[ExonRecord(start=1, end=9, strand=Strand.PLUS)],
        strand=Strand.PLUS,
        is_partial_5prime=True,
    )
    partial_3prime = GeneAnnotation(
        gene_name="atp1",
        exons=[ExonRecord(start=10, end=15, strand=Strand.PLUS)],
        strand=Strand.PLUS,
        is_partial_3prime=True,
    )

    accepted, rejected = partition_valid_cds([partial_5prime, partial_3prime], genome)

    assert accepted == [partial_5prime, partial_3prime]
    assert rejected == ()


def test_rna_editing_exception_does_not_hide_frame_shift_or_internal_stop() -> None:
    genome = GenomeSequence(seqid="mito", sequence="ATGTAGTAAATGA", is_circular=False)
    internal_stop = GeneAnnotation(
        gene_name="atp6",
        exons=[ExonRecord(start=1, end=9, strand=Strand.PLUS)],
        strand=Strand.PLUS,
        exceptions=["RNA editing"],
    )
    frame_shift = GeneAnnotation(
        gene_name="atp6",
        exons=[ExonRecord(start=10, end=13, strand=Strand.PLUS)],
        strand=Strand.PLUS,
        exceptions=["RNA editing"],
    )

    accepted, rejected = partition_valid_cds([internal_stop, frame_shift], genome)

    assert accepted == []
    assert rejected[0].issue_codes == ("premature_stop",)
    assert rejected[1].issue_codes == ("frame_shift",)


def test_empty_exon_candidate_is_reported_without_coordinate_crash() -> None:
    genome = GenomeSequence(seqid="mito", sequence="ATGAAATAA", is_circular=False)
    annotation = GeneAnnotation(
        gene_name="atp1",
        exons=[],
        strand=Strand.PLUS,
    )

    accepted, rejected = partition_valid_cds([annotation], genome)

    assert accepted == []
    assert rejected[0].start is None
    assert rejected[0].end is None
    assert rejected[0].parts == ()
    assert rejected[0].issue_codes == ("missing_seq",)


def test_uniform_exon_strand_must_match_gene_strand() -> None:
    genome = GenomeSequence(seqid="mito", sequence="ATGAAATAA", is_circular=False)
    annotation = GeneAnnotation(
        gene_name="rps10",
        exons=[ExonRecord(start=1, end=9, strand=Strand.MINUS)],
        strand=Strand.PLUS,
    )

    accepted, rejected = partition_valid_cds([annotation], genome)

    assert accepted == []
    assert rejected[0].issue_codes == ("strand_mismatch",)


def test_minus_strand_exons_are_translated_in_genbank_order() -> None:
    genome = GenomeSequence(
        seqid="mito",
        sequence="TTAAAAAAATTTCAT",
        is_circular=False,
    )
    annotation = GeneAnnotation(
        gene_name="rps10",
        exons=[
            ExonRecord(start=10, end=15, strand=Strand.MINUS),
            ExonRecord(start=1, end=3, strand=Strand.MINUS),
        ],
        strand=Strand.MINUS,
    )

    accepted, rejected = partition_valid_cds([annotation], genome)

    assert accepted == [annotation]
    assert rejected == ()


def test_plus_trans_spliced_exons_preserve_biological_order() -> None:
    genome = GenomeSequence(
        seqid="mito",
        sequence="TAACCCCCCATGAAA",
        is_circular=False,
    )
    annotation = GeneAnnotation(
        gene_name="nad2",
        exons=[
            ExonRecord(start=10, end=15, strand=Strand.PLUS, number=1),
            ExonRecord(start=1, end=3, strand=Strand.PLUS, number=2),
        ],
        strand=Strand.PLUS,
    )

    accepted, rejected = partition_valid_cds([annotation], genome)

    assert accepted == [annotation]
    assert rejected == ()


def test_terminal_repair_selects_nearby_complete_orf() -> None:
    genome = GenomeSequence(
        seqid="mito",
        sequence="CCCATGAAATAACCC",
        is_circular=False,
    )
    annotation = GeneAnnotation(
        gene_name="ccmC",
        exons=[ExonRecord(start=1, end=11, strand=Strand.PLUS)],
        strand=Strand.PLUS,
    )

    repaired = repair_complete_cds_terminals(annotation, genome, DBManager(), 20)
    accepted, rejected = partition_valid_cds([repaired], genome, DBManager())

    assert [(exon.start, exon.end) for exon in repaired.exons] == [(4, 12)]
    assert accepted == [repaired]
    assert rejected == ()


def test_terminal_repair_uses_each_mixed_strand_terminal_exon() -> None:
    genome = GenomeSequence(
        seqid="mito",
        sequence="TTACCCCCCATGAAA",
        is_circular=False,
    )
    annotation = GeneAnnotation(
        gene_name="nad2",
        exons=[
            ExonRecord(start=10, end=15, strand=Strand.PLUS, number=1),
            ExonRecord(start=2, end=3, strand=Strand.MINUS, number=2),
        ],
        strand=Strand.PLUS,
    )

    repaired = repair_complete_cds_terminals(annotation, genome, DBManager(), 20)
    accepted, rejected = partition_valid_cds([repaired], genome, DBManager())

    assert [(exon.start, exon.end) for exon in repaired.exons] == [(10, 15), (1, 3)]
    assert accepted == [repaired]
    assert rejected == ()


def test_terminal_repair_accepts_mttb_ttg_start_contract() -> None:
    genome = GenomeSequence(seqid="mito", sequence="TTGAAATAA", is_circular=False)
    annotation = GeneAnnotation(
        gene_name="mttB",
        exons=[ExonRecord(start=1, end=6, strand=Strand.PLUS)],
        strand=Strand.PLUS,
    )

    repaired = repair_complete_cds_terminals(annotation, genome, DBManager(), 20)
    accepted, rejected = partition_valid_cds([repaired], genome, DBManager())

    assert [(exon.start, exon.end) for exon in repaired.exons] == [(1, 9)]
    assert accepted == [repaired]
    assert rejected == ()


def test_tgg_is_not_accepted_as_c_to_u_stop_gain() -> None:
    genome = GenomeSequence(seqid="mito", sequence="ATGAAATGG", is_circular=False)
    annotation = GeneAnnotation(
        gene_name="atp6",
        exons=[ExonRecord(start=1, end=9, strand=Strand.PLUS)],
        strand=Strand.PLUS,
    )

    repaired = repair_complete_cds_terminals(annotation, genome, DBManager(), 20)
    accepted, rejected = partition_valid_cds([repaired], genome, DBManager())

    assert repaired.exceptions == []
    assert accepted == []
    assert rejected[0].issue_codes == ("missing_stop",)


def test_minus_trans_spliced_genbank_preserves_annotated_order(tmp_path: Path) -> None:
    genome = GenomeSequence(
        seqid="mito",
        sequence="TTTCATCCCTTA",
        is_circular=False,
    )
    annotation = GeneAnnotation(
        gene_name="nad2",
        exons=[
            ExonRecord(start=1, end=6, strand=Strand.MINUS, number=1),
            ExonRecord(start=10, end=12, strand=Strand.MINUS, number=2),
        ],
        strand=Strand.MINUS,
    )
    output = tmp_path / "minus-trans-spliced.gb"

    write_genbank([annotation], [], [], genome, output)
    document = parse_genbank(output)
    feature = next(
        item for record in document.records for item in record.features if item.type == "CDS"
    )

    assert [(part.start, part.end) for part in feature.parts] == [(0, 6), (9, 12)]
    assert feature.extract(document.records[0].sequence) == "ATGAAATAA"
