"""Sequence-verified IR copies of explicitly trans-spliced CDS products."""

import pytest
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from organelleverse.annotation.plastome.ir_mirror import (
    _mirror_shared_trans_location,
    mirror_ir_copies,
)


@pytest.fixture
def target():
    block = "ACGTCAGTACCGTTAGGCTAACGTACGAGT"
    seq = "G" * 20 + block + "C" * 50 + str(Seq(block).reverse_complement()) + "A" * 20
    return seq, ((20, 50, 1), (100, 130, -1))


def signature(location):
    return [(int(p.start), int(p.end), p.strand) for p in location.parts]


@pytest.mark.parametrize("shared_strand", [1, -1])
def test_shared_exon_preserved_and_ir_exons_reversed(target, shared_strand):
    seq, regions = target
    source = CompoundLocation(
        [
            FeatureLocation(4, 10, strand=shared_strand),
            FeatureLocation(22, 28, strand=1),
            FeatureLocation(38, 44, strand=1),
        ]
    )
    mapped = _mirror_shared_trans_location(source, seq, regions, trans_splicing=True)
    assert signature(mapped) == [(4, 10, shared_strand), (122, 128, -1), (106, 112, -1)]
    assert mapped.extract(Seq(seq)) == source.extract(Seq(seq))
    assert signature(
        _mirror_shared_trans_location(mapped, seq, regions, trans_splicing=True)
    ) == signature(source)


def test_no_explicit_trans_splicing_does_not_duplicate_boundary_model(target):
    seq, regions = target
    source = CompoundLocation([FeatureLocation(4, 10, strand=1), FeatureLocation(22, 28, strand=1)])
    assert _mirror_shared_trans_location(source, seq, regions, trans_splicing=False) is None


def test_exon_crossing_ir_boundary_is_not_treated_as_shared(target):
    seq, regions = target
    source = CompoundLocation(
        [FeatureLocation(16, 24, strand=1), FeatureLocation(38, 44, strand=1)]
    )
    assert _mirror_shared_trans_location(source, seq, regions, trans_splicing=True) is None


def test_exons_split_across_two_ir_copies_are_not_a_coherent_block(target):
    seq, regions = target
    source = CompoundLocation(
        [
            FeatureLocation(4, 10, strand=1),
            FeatureLocation(22, 28, strand=1),
            FeatureLocation(106, 112, strand=-1),
        ]
    )
    assert _mirror_shared_trans_location(source, seq, regions, trans_splicing=True) is None


def test_shared_exon_sequence_must_not_substitute_for_missing_ir_mate(target):
    seq, regions = target
    source = CompoundLocation(
        [
            FeatureLocation(4, 10, strand=1),
            FeatureLocation(22, 28, strand=1),
            FeatureLocation(38, 44, strand=1),
        ]
    )
    changed = seq[:100] + "N" * 30 + seq[130:]
    assert _mirror_shared_trans_location(source, changed, regions, trans_splicing=True) is None


def test_origin_wrapped_ir_uses_existing_modular_mapping():
    block = "ACGTCAGTACCGTTAGGCTAACGTACGAGT"
    seq = list("N" * 160)
    for offset, base in enumerate(block):
        seq[(140 + offset) % 160] = base
    seq[50:80] = str(Seq(block).reverse_complement())
    seq = "".join(seq)
    source = CompoundLocation(
        [
            FeatureLocation(25, 31, strand=-1),
            FeatureLocation(142, 148, strand=1),
            FeatureLocation(0, 6, strand=1),
        ]
    )
    mapped = _mirror_shared_trans_location(
        source, seq, ((140, 170, 1), (50, 80, -1)), trans_splicing=True
    )
    assert signature(mapped) == [(25, 31, -1), (72, 78, -1), (54, 60, -1)]
    assert mapped.extract(Seq(seq)) == source.extract(Seq(seq))


def test_shared_exon_is_not_a_duplicate_of_distinct_complete_product(target):
    seq, regions = target
    source = SeqFeature(
        CompoundLocation(
            [
                FeatureLocation(4, 10, strand=-1),
                FeatureLocation(22, 28, strand=1),
                FeatureLocation(38, 44, strand=1),
            ]
        ),
        type="CDS",
        qualifiers={
            "gene": ["synthetic_shared_product"],
            "trans_splicing": [""],
            "codon_start": ["1"],
            "transl_table": ["11"],
            "product": ["synthetic protein"],
            "translation": ["old translation"],
        },
    )
    features = [source]
    assert mirror_ir_copies(features, seq, regions) == 1
    assert len(features) == 2
    assert signature(features[1].location) == [(4, 10, -1), (122, 128, -1), (106, 112, -1)]
    assert features[1].extract(Seq(seq)) == features[0].extract(Seq(seq))
    assert features[1].qualifiers == {
        k: v for k, v in source.qualifiers.items() if k != "translation"
    }
    assert mirror_ir_copies(features, seq, regions) == 0
    assert len(features) == 2
    features[1].qualifiers["product"][0] = "modified copy"
    assert source.qualifiers["product"] == ["synthetic protein"]


def test_existing_other_copy_is_not_added_again(target):
    seq, regions = target
    source = SeqFeature(
        CompoundLocation(
            [
                FeatureLocation(4, 10, strand=-1),
                FeatureLocation(22, 28, strand=1),
                FeatureLocation(38, 44, strand=1),
            ]
        ),
        type="CDS",
        qualifiers={"gene": ["synthetic_shared_product"], "trans_splicing": [""]},
    )
    other = SeqFeature(
        _mirror_shared_trans_location(source.location, seq, regions, trans_splicing=True),
        type="CDS",
        qualifiers={"gene": ["synthetic_shared_product"], "trans_splicing": [""]},
    )
    features = [other, source]
    assert mirror_ir_copies(features, seq, regions) == 0
    assert len(features) == 2


def test_gene_span_without_cds_product_is_not_mirrored(target):
    seq, regions = target
    gene = SeqFeature(
        CompoundLocation([FeatureLocation(4, 10, strand=-1), FeatureLocation(22, 28, strand=1)]),
        type="gene",
        qualifiers={"gene": ["synthetic_shared_product"], "trans_splicing": [""]},
    )
    features = [gene]
    assert mirror_ir_copies(features, seq, regions) == 0
    assert features == [gene]


def test_unmarked_shared_cds_does_not_create_a_copy(target):
    seq, regions = target
    feature = SeqFeature(
        CompoundLocation(
            [
                FeatureLocation(4, 10, strand=-1),
                FeatureLocation(22, 28, strand=1),
            ]
        ),
        type="CDS",
        qualifiers={"gene": ["shared"]},
    )
    assert mirror_ir_copies([feature], seq, regions) == 0


def test_edited_ir_codon_uses_its_own_mirrored_strand():
    block = "AC" + "TAGAAA" + "GCTAACGTACGAGTACGTACGT"
    seq = "GGGGATG" + "G" * 13 + block + "C" * 50 + str(Seq(block).reverse_complement()) + "A" * 20
    feature = SeqFeature(
        CompoundLocation(
            [
                FeatureLocation(4, 7, strand=1),
                FeatureLocation(22, 28, strand=1),
            ]
        ),
        type="CDS",
        qualifiers={
            "gene": ["shared"],
            "trans_splicing": [""],
            "transl_except": ["(pos:23..25,aa:Gln)"],
        },
    )
    features = [feature]
    assert mirror_ir_copies(features, seq, ((20, 50, 1), (100, 130, -1))) == 1
    assert features[1].qualifiers["transl_except"] == ["(pos:complement(126..128),aa:Gln)"]
    assert feature.qualifiers["transl_except"] == ["(pos:23..25,aa:Gln)"]


def test_edited_codon_spanning_opposite_strand_exons_keeps_only_its_three_bases():
    from organelleverse.annotation.plastome.cds_refine import reannotate_edited_stops

    seq = list("N" * 40)
    seq[4:8] = "ACAT"
    seq[22:27] = "AAAAA"
    feature = SeqFeature(
        CompoundLocation(
            [
                FeatureLocation(4, 8, strand=-1),
                FeatureLocation(22, 27, strand=1),
            ]
        ),
        type="CDS",
        qualifiers={"transl_except": ["old"]},
    )
    assert feature.extract(Seq("".join(seq))) == "ATGTAAAAA"
    reannotate_edited_stops(feature, "".join(seq))
    assert feature.qualifiers["transl_except"] == ["(pos:join(complement(5),23..24),aa:Gln)"]
