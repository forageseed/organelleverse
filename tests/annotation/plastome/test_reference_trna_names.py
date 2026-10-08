"""Reference tRNAs without /gene are named at parse time.

Several NCBI records (Eucalyptus, Marchantia, Oryza, Populus, Cryptomeria)
give tRNAs only a product; unnamed transferred gene spans could not guide
intron-tRNA splicing (Juniperus trnV-UAC was misread as trnR with Cryptomeria
as its reference).
"""

from __future__ import annotations

from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from organelleverse.annotation.plastome.references import _fill_trna_gene_names, _trna_name_from


def test_names_come_from_codon_note_or_product() -> None:
    assert _trna_name_from({"product": ["tRNA-Val"], "codon_recognized": ["GUA"]}) == "trnV-UAC"
    assert _trna_name_from({"product": ["tRNA-Ile"], "codon_recognized": ["AUA"]}) == "trnI-CAU"
    assert _trna_name_from({"product": ["tRNA-Gln"], "note": ["trnQ; tRNA-Gln(UUG)"]}) == "trnQ-UUG"
    assert _trna_name_from({"product": ["tRNA-fMet"]}) == "trnfM"
    assert _trna_name_from({"product": ["tRNA-Leu"]}) == "trnL"
    assert _trna_name_from({"note": ["pseudo trnV-AAC"]}) is None


def test_gene_features_follow_their_trna() -> None:
    spliced = CompoundLocation(
        [FeatureLocation(1000, 1038, strand=1), FeatureLocation(1560, 1597, strand=1)]
    )
    trna = SeqFeature(
        spliced,
        type="tRNA",
        qualifiers={"locus_tag": ["X_t4"], "product": ["tRNA-Val"], "codon_recognized": ["GUA"]},
    )
    by_tag = SeqFeature(
        FeatureLocation(1000, 1597, strand=1), type="gene", qualifiers={"locus_tag": ["X_t4"]}
    )
    leu = SeqFeature(
        FeatureLocation(3000, 3085, strand=-1), type="tRNA", qualifiers={"product": ["tRNA-Leu"]}
    )
    by_span = SeqFeature(FeatureLocation(3000, 3085, strand=-1), type="gene")
    named = SeqFeature(
        FeatureLocation(5000, 5072, strand=1), type="tRNA", qualifiers={"gene": ["trnH-GUG"]}
    )
    rec = SeqRecord(Seq("A" * 6000), features=[trna, by_tag, leu, by_span, named])

    _fill_trna_gene_names(rec)

    assert trna.qualifiers["gene"] == ["trnV-UAC"]
    assert by_tag.qualifiers["gene"] == ["trnV-UAC"]
    assert by_span.qualifiers["gene"] == ["trnL"]
    assert named.qualifiers["gene"] == ["trnH-GUG"]
