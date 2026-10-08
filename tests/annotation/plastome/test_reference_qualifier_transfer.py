"""Reference-transferred plastome features must not carry the reference record's identity.

Before the fix every CDS copied from the nearest reference kept its
protein_id, db_xref and locus_tag (e.g. Arabidopsis annotated with Theobroma
``ThcaC_*`` tags), plus coordinate-bearing qualifiers that point into the
reference sequence.
"""

from __future__ import annotations

from Bio.SeqFeature import FeatureLocation, SeqFeature

from organelleverse.annotation.plastome.pipeline import _copy_feature


def _reference_cds() -> SeqFeature:
    return SeqFeature(
        FeatureLocation(72370, 73897, strand=1),
        type="CDS",
        qualifiers={
            "gene": ["psbB"],
            "product": ["photosystem II CP47 chlorophyll apoprotein"],
            "locus_tag": ["ThcaC_p047"],
            "old_locus_tag": ["ThcaCp047"],
            "protein_id": ["YP_004021342.1"],
            "db_xref": ["GeneID:9978145"],
            "inference": ["similar to DNA sequence:INSD:AB000000.1"],
            "translation": ["MGLPWYRVHTVVLNDPGRLL"],
            "transl_except": ["(pos:72371..72373,aa:Met)"],
            "exception": ["RNA editing"],
            "note": ["photosystem II protein"],
            "codon_start": ["1"],
            "transl_table": ["11"],
        },
    )


def test_copy_feature_drops_reference_record_identifiers() -> None:
    copied = _copy_feature(_reference_cds(), FeatureLocation(100, 1627, strand=1), {})

    for key in (
        "locus_tag",
        "old_locus_tag",
        "protein_id",
        "db_xref",
        "inference",
        "translation",
        "transl_except",
    ):
        assert key not in copied.qualifiers, key


def test_copy_feature_keeps_gene_semantics() -> None:
    copied = _copy_feature(_reference_cds(), FeatureLocation(100, 1627, strand=1), {})

    assert copied.qualifiers["gene"] == ["psbB"]
    assert copied.qualifiers["product"] == ["photosystem II CP47 chlorophyll apoprotein"]
    assert copied.qualifiers["exception"] == ["RNA editing"]
    assert copied.qualifiers["note"] == ["photosystem II protein"]
    assert copied.qualifiers["transl_table"] == ["11"]
    assert int(copied.location.start) == 100


def test_trna_anticodon_with_reference_coordinates_is_dropped() -> None:
    trna = SeqFeature(
        FeatureLocation(7784, 7872, strand=-1),
        type="tRNA",
        qualifiers={
            "gene": ["trnS-GCU"],
            "product": ["tRNA-Ser"],
            "anticodon": ["(pos:complement(7835..7837),aa:Ser,seq:gct)"],
            "codon_recognized": ["AGC"],
        },
    )

    copied = _copy_feature(trna, FeatureLocation(10, 98, strand=-1), {})

    assert "anticodon" not in copied.qualifiers
    assert copied.qualifiers["codon_recognized"] == ["AGC"]
