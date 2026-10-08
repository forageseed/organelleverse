"""Per-contig records for a multi-contig mitochondrial annotation."""

from __future__ import annotations

from organelleverse.annotation.contigs import (
    contig_layout,
    merge_document_by_contigs,
    merged_stat_lines,
    split_document_by_contigs,
)
from organelleverse.annotation.models import (
    AnnotationDocument,
    AnnotationFeature,
    AnnotationRecord,
    FeatureQualifier,
    LocationPart,
)

# ctgA 0-999, spacer 1000-1199, ctgB 1200-2199, spacer, ctgC 2400-3399
_LAYOUT = [("ctgA", 0, 1000), ("ctgB", 1200, 1000), ("ctgC", 2400, 1000)]


def _feature(index, type_, parts, **qualifiers):
    return AnnotationFeature(
        feature_id=f"merged:{index}:{type_}",
        seqid="merged",
        type=type_,
        operator="join" if len(parts) > 1 else "single",
        parts=tuple(LocationPart(start=a, end=b, strand=c) for a, b, c in parts),
        qualifiers=tuple(FeatureQualifier(name=k, values=tuple(v)) for k, v in qualifiers.items()),
        parents=(),
    )


def _document(features):
    record = AnnotationRecord(
        seqid="merged",
        name="merged",
        description="d",
        sequence="A" * 3400,
        features=tuple(features),
    )
    return AnnotationDocument(
        backend="mitochondrion",
        requested_stages=("pcg",),
        completed_stages=("pcg",),
        records=(record,),
    )


def test_layout_reads_contig_names_and_spacers(tmp_path):
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">ctgA\n" + "A" * 1000 + "\n>ctgB\n" + "C" * 1000 + "\n>ctgC\n" + "G" * 1000 + "\n")
    assert contig_layout(fasta) == _LAYOUT


def test_a_single_contig_has_no_layout(tmp_path):
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">only\n" + "A" * 100 + "\n")
    assert contig_layout(fasta) == []


def test_features_get_contig_names_and_local_coordinates():
    doc = split_document_by_contigs(
        _document([_feature(0, "CDS", [(1300, 1500, 1)], gene=["cox1"])]), _LAYOUT
    )
    assert [r.seqid for r in doc.records] == ["ctgA", "ctgB", "ctgC"]
    (feature,) = doc.records[1].features
    assert (feature.parts[0].start, feature.parts[0].end) == (100, 300)
    assert feature.seqid == "ctgB"
    assert not doc.records[0].features and not doc.records[2].features
    assert len(doc.records[1].sequence) == 1000


def test_a_cis_join_inside_one_contig_stays_one_join():
    doc = split_document_by_contigs(
        _document([_feature(0, "CDS", [(1210, 1300, 1), (1500, 1600, 1)], gene=["cox2"])]),
        _LAYOUT,
    )
    (feature,) = doc.records[1].features
    assert feature.operator == "join"
    assert [(p.start, p.end) for p in feature.parts] == [(10, 100), (300, 400)]
    assert "trans_splicing" not in {q.name for q in feature.qualifiers}


def test_a_gene_across_contigs_is_split_and_flagged():
    nad5 = _feature(
        0,
        "CDS",
        [(100, 200, 1), (300, 400, 1), (1300, 1400, 1), (2500, 2600, 1)],
        gene=["nad5"],
        translation=["MKK"],
        codon_start=["1"],
    )
    doc = split_document_by_contigs(_document([nad5]), _LAYOUT)
    on_a, on_b, on_c = (r.features for r in doc.records)
    assert [(p.start, p.end) for p in on_a[0].parts] == [(100, 200), (300, 400)]
    assert [(p.start, p.end) for p in on_b[0].parts] == [(100, 200)]
    assert on_b[0].operator == "single"
    assert [(p.start, p.end) for p in on_c[0].parts] == [(100, 200)]
    for piece in (on_a[0], on_b[0], on_c[0]):
        names = {q.name for q in piece.qualifiers}
        assert {"trans_splicing", "exception", "note", "codon_start", "partial"} <= names
        assert "translation" not in names
    # frame of each piece inside the whole gene: 200 bases precede ctgB's part
    assert on_a[0].qualifier_values("codon_start") == ("1",)
    assert on_b[0].qualifier_values("codon_start") == ("2",)
    # the gene runs on past the last part on ctgA, and starts before the first on ctgB
    assert on_a[0].parts[0].start_status == "exact"
    assert on_a[0].parts[-1].end_status == "after"
    assert on_b[0].parts[0].start_status == "before"
    assert on_c[0].parts[-1].end_status == "exact"
    assert "ctgB" in on_a[0].qualifier_values("note")[0]
    crossing = doc.source_metadata["cross_contig_features"]
    assert crossing[0]["gene"] == "nad5"


def test_an_existing_note_is_extended_not_replaced():
    nad = _feature(0, "CDS", [(100, 200, 1), (1300, 1400, 1)], gene=["nad1"], note=["kept"])
    doc = split_document_by_contigs(_document([nad]), _LAYOUT)
    notes = doc.records[0].features[0].qualifier_values("note")
    assert notes[0] == "kept" and "across contigs" in notes[1]


def test_a_split_document_passes_canonical_validation():
    from organelleverse.annotation.validation import validate_document

    sequence = "ATG" + "GCT" * 60 + "TAA"  # 186 bases, one gene in two pieces
    merged = "A" * 100 + sequence[:93] + "N" * 1000 + sequence[93:] + "A" * 1000
    layout = [("ctgA", 0, 193), ("ctgB", 1193, len(merged) - 1193)]
    gene = _feature(0, "CDS", [(100, 193, 1), (1193 + 0, 1193 + 93, 1)], gene=["cox1"])
    record = AnnotationRecord(
        seqid="merged", name="merged", description="d", sequence=merged, features=(gene,)
    )
    doc = AnnotationDocument(
        backend="mitochondrion",
        requested_stages=("pcg",),
        completed_stages=("pcg",),
        records=(record,),
    )
    split = split_document_by_contigs(doc, layout)
    assert not validate_document(split).errors


def test_the_contig_map_is_recorded_in_merged_coordinates():
    doc = split_document_by_contigs(_document([_feature(0, "CDS", [(1300, 1500, 1)])]), _LAYOUT)
    contig_map = doc.source_metadata["contig_map"]
    assert [(c["id"], c["merged_start"], c["merged_end"]) for c in contig_map] == [
        ("ctgA", 1, 1000),
        ("ctgB", 1201, 2200),
        ("ctgC", 2401, 3400),
    ]


def test_without_a_layout_the_document_is_untouched():
    doc = _document([_feature(0, "CDS", [(10, 50, 1)])])
    assert split_document_by_contigs(doc, []) is doc


def _split_nad5_and_trna():
    merged = _document(
        [
            _feature(
                0,
                "CDS",
                [(100, 200, 1), (1300, 1400, 1)],
                gene=["nad5"],
                codon_start=["1"],
                translation=["MKK"],
            ),
            _feature(1, "tRNA", [(2500, 2600, -1)], gene=["trnA"]),
        ]
    )
    return split_document_by_contigs(merged, _LAYOUT)


def test_merge_shifts_features_back_onto_the_merged_sequence():
    merged = merge_document_by_contigs(_split_nad5_and_trna())
    (record,) = merged.records
    assert record.seqid == "merged_3_contigs"
    assert len(record.sequence) == 3400
    # the nad5 pieces stay one entry per contig, in merged coordinates
    assert [(p.start, p.end) for p in record.features[0].parts] == [(100, 200)]
    assert [(p.start, p.end) for p in record.features[1].parts] == [(1300, 1400)]
    assert [(p.start, p.end) for p in record.features[2].parts] == [(2500, 2600)]
    assert record.features[2].parts[0].strand == -1
    assert len({feature.feature_id for feature in record.features}) == 3


def test_merge_restores_spacers_and_the_contig_map():
    merged = merge_document_by_contigs(_split_nad5_and_trna())
    sequence = merged.records[0].sequence
    assert sequence[1000:1200] == "N" * 200
    assert sequence[2200:2400] == "N" * 200
    assert [(c["id"], c["merged_start"], c["merged_end"]) for c in merged.source_metadata["contig_map"]] == [
        ("ctgA", 1, 1000),
        ("ctgB", 1201, 2200),
        ("ctgC", 2401, 3400),
    ]


def test_merged_stat_lines_list_every_contig_span():
    lines = merged_stat_lines(merge_document_by_contigs(_split_nad5_and_trna()))
    assert lines == [
        "contig_id\tlength\tmerged_start\tmerged_end",
        "ctgA\t1000\t1\t1000",
        "ctgB\t1000\t1201\t2200",
        "ctgC\t1000\t2401\t3400",
    ]


def test_a_single_record_document_merges_to_itself():
    doc = _document([_feature(0, "CDS", [(10, 50, 1)])])
    assert merge_document_by_contigs(doc) is doc
    assert merged_stat_lines(doc) == []
