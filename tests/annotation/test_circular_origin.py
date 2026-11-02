"""Genes across the origin of a circular molecule."""

from __future__ import annotations

from organelleverse.annotation.contigs import (
    circular_flank,
    extend_circular_molecules,
    fold_circular_flank,
)
from organelleverse.annotation.models import (
    AnnotationDocument,
    AnnotationFeature,
    AnnotationRecord,
    FeatureQualifier,
    LocationPart,
)

_L = 1000  # true molecule length; the record carries 200 extra bases of its own start


def _feature(index, parts, gene="atp1", type_="CDS", **extra):
    qualifiers = {"gene": [gene], **extra}
    return AnnotationFeature(
        feature_id=f"r:{index}:{type_}",
        seqid="r",
        type=type_,
        operator="join" if len(parts) > 1 else "single",
        parts=tuple(LocationPart(start=a, end=b, strand=c) for a, b, c in parts),
        qualifiers=tuple(FeatureQualifier(name=k, values=tuple(v)) for k, v in qualifiers.items()),
        parents=(),
    )


def _document(features, extension=200, name="r"):
    record = AnnotationRecord(
        seqid=name,
        name=name,
        description="d",
        sequence="A" * (_L + extension),
        features=tuple(features),
    )
    return AnnotationDocument(
        backend="mitochondrion", requested_stages=(), completed_stages=(), records=(record,)
    )


def _parts(document, index=0):
    return [(p.start, p.end, p.strand) for p in document.records[0].features[index].parts]


def test_a_circular_molecule_gets_a_copy_of_its_start(tmp_path):
    source, target = tmp_path / "in.fasta", tmp_path / "out.fasta"
    source.write_text(">m.1 circular=true\n" + "ACGT" * 500 + "\n>m.2 circular=false\nTTGGCCAA\n")
    flanks = extend_circular_molecules(
        source, target, [("m.1", "circular=true", True), ("m.2", "circular=false", False)]
    )
    assert flanks == [circular_flank(2000), 0]
    lines = target.read_text().splitlines()
    assert len(lines[1]) == 2000 + flanks[0] and lines[3] == "TTGGCCAA"


def test_the_flank_never_exceeds_half_the_molecule():
    assert circular_flank(10_000) == 5_000
    assert circular_flank(400_000) == 15_000


def test_a_plus_gene_across_the_end_is_a_join_over_the_origin():
    document = fold_circular_flank(_document([_feature(0, [(950, 1060, 1)])]), [_L])
    assert _parts(document) == [(950, 1000, 1), (0, 60, 1)]
    assert document.records[0].features[0].operator == "join"
    assert len(document.records[0].sequence) == _L


def test_a_minus_gene_across_the_end_lists_the_wrapped_part_first():
    document = fold_circular_flank(_document([_feature(0, [(950, 1060, -1)])]), [_L])
    assert _parts(document) == [(0, 60, -1), (950, 1000, -1)]


def test_a_gene_wholly_inside_the_copy_is_dropped_when_the_original_is_annotated():
    inside = _feature(0, [(1020, 1100, 1)], gene="nad9")
    kept = _feature(1, [(10, 90, 1)], gene="nad9")
    document = fold_circular_flank(_document([inside, kept]), [_L])
    assert [f.feature_id for f in document.records[0].features] == ["r:1:CDS"]


def test_a_gene_found_only_in_the_copy_moves_back_to_the_molecule_start():
    # nad3-like: both copies are identical, the search kept the one in the appended repeat
    only_copy = _feature(0, [(1020, 1100, -1)], gene="nad3")
    document = fold_circular_flank(_document([only_copy]), [_L])
    assert _parts(document) == [(20, 100, -1)]


def test_a_gene_cut_by_the_end_of_the_copy_is_not_moved_back():
    cut = _feature(0, [(1100, 1200, 1)], gene="nad3")
    cut = cut.evolve(parts=(cut.parts[0].evolve(end_status="after"),))
    document = fold_circular_flank(_document([cut]), [_L])
    assert not document.records[0].features


def test_a_gene_wholly_before_the_end_is_untouched():
    gene = _feature(0, [(100, 400, 1)])
    document = fold_circular_flank(_document([gene]), [_L])
    assert document.records[0].features[0] == gene


def test_a_spliced_cds_with_its_last_exon_across_the_origin_wraps_only_that_exon():
    gene = _feature(0, [(800, 900, 1), (980, 1050, 1)])
    document = fold_circular_flank(_document([gene]), [_L])
    assert _parts(document) == [(800, 900, 1), (980, 1000, 1), (0, 50, 1)]


def test_the_cut_remainder_of_a_spanning_gene_at_the_start_is_dropped():
    spanning = _feature(0, [(950, 1060, 1)])
    remainder = _feature(1, [(0, 58, 1)], note=["partial CDS: the gene continues"])
    other_gene = _feature(2, [(0, 58, 1)], gene="nad9")
    document = fold_circular_flank(_document([spanning, remainder, other_gene]), [_L])
    names = [(f.qualifier_values("gene")[0], f.operator) for f in document.records[0].features]
    assert names == [("atp1", "join"), ("nad9", "single")]


def test_a_non_circular_record_is_left_alone():
    gene = _feature(0, [(950, 1060, 1)])
    document = _document([gene])
    assert fold_circular_flank(document, [None]) is document


def test_the_contig_map_follows_the_trimmed_records():
    records = tuple(
        AnnotationRecord(
            seqid=name,
            name=name,
            description="d",
            sequence="A" * length,
            features=(),
        )
        for name, length in (("a", 1200), ("b", 500))
    )
    document = AnnotationDocument(
        backend="mitochondrion",
        requested_stages=(),
        completed_stages=(),
        records=records,
        source_metadata={"contig_map": [{"id": "a", "length": 1200}]},
    )
    folded = fold_circular_flank(document, [1000, None])
    assert [len(r.sequence) for r in folded.records] == [1000, 500]
    contig_map = folded.source_metadata["contig_map"]
    assert [(c["id"], c["merged_start"], c["merged_end"]) for c in contig_map] == [
        ("a", 1, 1000),
        ("b", 1201, 1700),
    ]
