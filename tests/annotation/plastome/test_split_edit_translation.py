"""Edited codons keep their biological order across mixed-strand exon joins."""

import pytest
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from organelleverse.annotation.plastome.cds_refine import reannotate_edited_stops
from organelleverse.annotation.plastome.pipeline import (
    _apply_transl_except,
    _retranslate_cds_features,
)


@pytest.mark.parametrize(
    "shared,shared_end,repeat_end,exception",
    [
        ("ACAT", 8, 27, "join(complement(5),23..24)"),
        ("TACAT", 9, 26, "join(complement(5..6),23)"),
    ],
)
def test_split_edit_is_applied_to_exported_translation(shared, shared_end, repeat_end, exception):
    seq = list("N" * 40)
    seq[4:shared_end] = shared
    seq[22:repeat_end] = "A" * (repeat_end - 22)
    feature = SeqFeature(
        CompoundLocation(
            [
                FeatureLocation(4, shared_end, strand=-1),
                FeatureLocation(22, repeat_end, strand=1),
            ]
        ),
        type="CDS",
        qualifiers={"transl_except": ["old"]},
    )
    genome = "".join(seq)
    assert feature.extract(Seq(genome)) == "ATGTAAAAA"
    reannotate_edited_stops(feature, genome)
    assert feature.qualifiers["transl_except"] == [f"(pos:{exception},aa:Gln)"]
    _retranslate_cds_features([feature], genome)
    assert feature.qualifiers["translation"] == ["MQK"]


def test_exception_must_cover_exactly_the_codon_in_coding_order():
    feature = SeqFeature(
        FeatureLocation(0, 9, strand=1),
        type="CDS",
        qualifiers={
            "transl_except": ["(pos:4..5,aa:Gln)", "(pos:5..7,aa:Gln)"],
        },
    )
    assert _apply_transl_except(feature, "M*K") == "M*K"
