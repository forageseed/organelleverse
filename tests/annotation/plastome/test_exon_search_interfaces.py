from pathlib import Path

from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from organelleverse.annotation.plastome import blast
from organelleverse.annotation.plastome.models import (
    ReferenceFeature,
    ReferenceQuery,
    ReferenceRecord,
)
from organelleverse.annotation.plastome.references import build_exon_query_panel

candidate = blast


def query():
    feature = ReferenceFeature(
        "ref:1:exon1",
        SeqFeature(FeatureLocation(0, 6, strand=1)),
        "ATGAAA",
        "example",
        "CDS_exon",
        "ref",
        1,
        2,
    )
    return ReferenceQuery("q", "reference4", "MK", feature)


def test_all_eligible_hits_keep_opposite_strand_copy_and_legacy_best(tmp_path):
    path = tmp_path / "hits.tsv"
    path.write_text(
        "q\ttarget\t100\t2\t0\t0\t1\t2\t10\t15\t0\t50\nq\ttarget\t100\t2\t0\t0\t1\t2\t35\t30\t0\t49\n"
    )
    queries = [query()]
    eligible = candidate.parse_blast_hit_lists(path, 0.4, (0.5, 1.1), queries)
    assert [(h.start, h.end, h.strand) for h in eligible["q"]] == [(9, 15, 1), (29, 35, -1)]
    assert candidate.parse_blast_hits(path, 0.4, (0.5, 1.1), queries)["q"] == eligible["q"][0]


def test_both_formats_and_filtering_share_legacy_contract(tmp_path):
    path = tmp_path / "hits.tsv"
    path.write_text(
        "q\ttarget\t10\t2\t0\t0\t1\t2\t10\t15\t0\t500\nq\ttarget\t100\t4\t0\t0\t1\t4\t10\t21\t0\t500\nq\ttarget\t90\t2\t0\t0\t1\t2\t35\t30\t0\t49\t2\t40\n"
    )
    queries = [query()]
    assert len(candidate.parse_blast_hit_lists(path, 0.4, (0.5, 1.1), queries)["q"]) == 1
    assert candidate.parse_blast_hits(path, 0.4, (0.5, 1.1), queries)["q"].bitscore == 49


def test_equal_rank_retains_original_first_hit_as_legacy_parser(tmp_path):
    path = tmp_path / "hits.tsv"
    path.write_text(
        "q\ttarget\t100\t2\t0\t0\t1\t2\t10\t15\t0\t50\nq\ttarget\t100\t2\t0\t0\t1\t2\t35\t30\t0\t50\n"
    )
    queries = [query()]
    assert candidate.parse_blast_hits(path, 0.4, (0.5, 1.1), queries)["q"].start == 9


def reference(name, count):
    parts = [FeatureLocation(i * 12, i * 12 + 6, strand=1) for i in range(count)]
    template = SeqFeature(CompoundLocation(parts), type="CDS", qualifiers={"gene": ["example"]})
    features = tuple(
        ReferenceFeature(
            f"{name}:1:exon{i + 1}", template, "ATGAAA", "example", "CDS_exon", name, i + 1, count
        )
        for i in range(count)
    )
    return ReferenceRecord(Path(name + ".gb"), SeqRecord(Seq("ATGAAACCCCCC" * count)), features)


def test_all_reference_architectures_and_unique_query_ids():
    panel = build_exon_query_panel((reference("a", 2), reference("b", 3)))
    assert len(panel) == 5 and len({q.query_id for q in panel}) == 5
    assert {q.reference_feature.exon_count for q in panel} == {2, 3}


def test_incomplete_reference_product_is_not_queried():
    from dataclasses import replace

    ref = reference("a", 3)
    assert build_exon_query_panel((replace(ref, features=ref.features[:-1]),)) == ()
