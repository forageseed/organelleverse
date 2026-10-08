"""CDS carrying only /product get their gene symbol; identical features are not emitted twice."""

from __future__ import annotations

from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from organelleverse.annotation.plastome.pipeline import _drop_identical_features
from organelleverse.annotation.plastome.references import _fill_cds_gene_names


def _cds(start, end, strand=1, **quals):
    return SeqFeature(FeatureLocation(start, end, strand=strand), type="CDS", qualifiers={k: [v] for k, v in quals.items()})


def test_product_only_cds_get_their_gene_symbol():
    rec = SeqRecord(Seq("A" * 1000))
    rec.features = [
        _cds(0, 90, product="photosystem I protein M"),
        _cds(100, 190, product="maturase"),
        _cds(200, 290, product="hypothetical protein"),
        _cds(300, 390, product="ribosomal protein S4", gene="rps4"),
        _cds(400, 490, product="anything", locus_tag="X_1"),
        SeqFeature(FeatureLocation(400, 490, strand=1), type="gene", qualifiers={"gene": ["ycf12"], "locus_tag": ["X_1"]}),
    ]
    _fill_cds_gene_names(rec)
    genes = [f.qualifiers.get("gene", [None])[0] for f in rec.features if f.type == "CDS"]
    assert genes == ["psaM", "matK", None, "rps4", "ycf12"]


def test_identical_features_collapse_but_different_copies_stay():
    a = _cds(10, 100, -1, gene="ndhB")
    twin = _cds(10, 100, -1, gene="ndhB")
    other_copy = _cds(500, 590, 1, gene="ndhB")
    split = SeqFeature(CompoundLocation([FeatureLocation(10, 40, strand=-1), FeatureLocation(60, 100, strand=-1)]),
                       type="CDS", qualifiers={"gene": ["ndhB"]})
    unnamed = _cds(10, 100, -1)
    out = _drop_identical_features([a, twin, other_copy, split, unnamed, _cds(10, 100, -1)])
    assert out == [a, other_copy, split, unnamed, out[-1]] and len(out) == 5
