from dataclasses import replace

from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from organelleverse.annotation.plastome.exon_models import (
    assemble_exon_models,
    select_mature_model_copies,
)
from organelleverse.annotation.plastome.models import BlastHit, ReferenceFeature, ReferenceQuery


def fixture():
    template = SeqFeature(
        CompoundLocation(
            [
                FeatureLocation(0, 3, strand=-1),
                FeatureLocation(6, 9, strand=1),
                FeatureLocation(12, 15, strand=1),
            ]
        ),
        type="CDS",
        qualifiers={"gene": ["example"], "trans_splicing": [""]},
    )
    queries = tuple(
        ReferenceQuery(
            f"q{i}",
            "reference4",
            "M",
            ReferenceFeature(f"ref:1:exon{i}", template, "AAA", "example", "CDS_exon", "ref", i, 3),
        )
        for i in range(1, 4)
    )

    def hit(q, start, end, strand):
        return BlastHit(q, 100, 1, start, end, strand, 10, 0, 1)

    hits = {
        "q1": [hit("q1", 0, 3, 1)],
        "q2": [hit("q2", 6, 9, 1), hit("q2", 24, 27, -1)],
        "q3": [hit("q3", 12, 15, 1), hit("q3", 18, 21, -1)],
    }
    return "ATGCCCAAACCCTAACCCTTACCCTTT", queries, hits, {"ref": "CATCCCAAACCCTAA"}


def test_complete_trans_spliced_copies_preserve_shared_first_exon():
    models = assemble_exon_models(*fixture())
    selected = select_mature_model_copies(models)
    assert len(selected) == 2
    assert all(m.mature_dna == "ATGAAATAA" and m.protein == "MK*" for m in selected)
    assert selected[0].location.parts[0] == selected[1].location.parts[0]
    assert selected[0].location.parts[1].strand == -selected[1].location.parts[1].strand


def test_missing_exon_does_not_make_partial_complete_product():
    seq, queries, hits, refs = fixture()
    assert assemble_exon_models(seq, queries[:-1], hits, refs) == []
    assert assemble_exon_models(seq, queries, {**hits, "q3": []}, refs) == []


def test_reference_products_are_not_stitched_together():
    seq, queries, hits, refs = fixture()
    queries = (
        *queries[:2],
        replace(
            queries[2],
            reference_feature=replace(queries[2].reference_feature, feature_id="ref:other:exon3"),
        ),
    )
    assert assemble_exon_models(seq, queries, hits, refs) == []


def test_broken_cis_block_is_rejected():
    seq, queries, hits, refs = fixture()
    hits = {**hits, "q2": hits["q2"][:1], "q3": hits["q3"][1:]}
    assert assemble_exon_models(seq, queries, hits, refs) == []


def test_equal_protein_different_dna_is_ambiguous():
    model = assemble_exon_models(*fixture())[0]
    assert select_mature_model_copies([model, replace(model, mature_dna="ATGAAGTAA")]) == []
    assert select_mature_model_copies([model, model]) == [model]


def test_out_of_bounds_or_overlapping_exons_are_not_candidates():
    seq, queries, hits, refs = fixture()
    assert (
        assemble_exon_models(
            seq, queries, {**hits, "q3": [replace(hits["q3"][0], start=26, end=29)]}, refs
        )
        == []
    )
    assert (
        assemble_exon_models(
            seq, queries, {**hits, "q3": [replace(hits["q3"][0], start=6, end=9)]}, refs
        )
        == []
    )


def test_all_source_exons_do_not_make_a_missing_stop_complete():
    seq, queries, hits, refs = fixture()
    seq = seq[:12] + "CAA" + seq[15:18] + "TTG" + seq[21:]
    assert assemble_exon_models(seq, queries, hits, refs) == []


def test_all_source_exons_do_not_make_a_missing_start_complete():
    seq, queries, hits, refs = fixture()
    assert assemble_exon_models("ACG" + seq[3:], queries, hits, refs) == []


def test_complete_model_uses_the_genetic_code_alternative_starts():
    seq, queries, hits, refs = fixture()
    models = assemble_exon_models("GTG" + seq[3:], queries, hits, refs)
    assert len(select_mature_model_copies(models)) == 2


def test_complete_model_rejects_internal_stops():
    seq, queries, hits, refs = fixture()
    seq = seq[:6] + "TAA" + seq[9:24] + "TTA"
    assert assemble_exon_models(seq, queries, hits, refs) == []


def test_complete_alternative_start_protein_matches_complete_cds_translation():
    seq, queries, hits, refs = fixture()
    baseline = assemble_exon_models(seq, queries, hits, refs)
    for start in ("GTG", "ATA"):
        models = assemble_exon_models(start + seq[3:], queries, hits, refs)
        assert len(models) == len(baseline) == 2
        assert all(m.protein == "MK*" for m in models)
        assert [m.protein_score for m in models] == [m.protein_score for m in baseline]


def test_alternative_start_normalization_does_not_change_internal_codon():
    seq, queries, hits, refs = fixture()
    seq = "GTG" + seq[3:6] + "GTG" + seq[9:24] + "CAC"
    models = assemble_exon_models(seq, queries, hits, refs)
    assert len(models) == 2
    assert all(m.protein == "MV*" for m in models)
