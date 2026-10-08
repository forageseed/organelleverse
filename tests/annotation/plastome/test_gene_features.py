"""Gene features are derived from the final CDS/tRNA/rRNA features.

Transferred gene features disagreed with refined CDS and native tRNA/rRNA
(405 gene-only loci across 8 benchmark plastomes, each locus drawn twice on
structure maps); rebuilding them leaves only true gene-only loci.
"""

from __future__ import annotations

from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from organelleverse.annotation.plastome.gene_features import rebuild_gene_features

LENGTH = 150_000


def _feature(kind: str, parts, **qualifiers) -> SeqFeature:
    locs = [FeatureLocation(s, e, strand=st) for s, e, st in parts]
    location = locs[0] if len(locs) == 1 else CompoundLocation(locs)
    return SeqFeature(location, type=kind, qualifiers={k: [v] for k, v in qualifiers.items()})


def _genes(features):
    return [f for f in features if f.type == "gene"]


def _span(feature):
    return int(feature.location.start), int(feature.location.end)


def test_native_trna_replaces_a_differently_named_transferred_gene() -> None:
    transferred = _feature("gene", [(1371, 3931, -1)], gene="trnK")  # Triticum naming
    trna = _feature("tRNA", [(3893, 3931, -1), (1371, 1407, -1)], gene="trnK-UUU")

    genes = _genes(rebuild_gene_features([transferred, trna], LENGTH))

    assert len(genes) == 1
    assert genes[0].qualifiers["gene"] == ["trnK-UUU"]
    assert _span(genes[0]) == (1371, 3931)
    assert genes[0].location.strand == -1


def test_gene_follows_a_refined_cds() -> None:
    transferred = _feature("gene", [(63538, 63660, -1)], gene="psbJ")
    cds = _feature("CDS", [(63537, 63660, -1)], gene="psbJ")  # stop codon added by refinement

    genes = _genes(rebuild_gene_features([transferred, cds], LENGTH))

    assert [(_span(g), g.qualifiers["gene"]) for g in genes] == [((63537, 63660), ["psbJ"])]


def test_trans_spliced_gene_keeps_its_parts() -> None:
    cds = _feature(
        "CDS",
        [(69000, 69114, -1), (98000, 98232, 1), (98770, 98796, 1)],
        gene="rps12",
        trans_splicing="",
    )
    part_gene = _feature("gene", [(97990, 98240, 1)], gene="3' rps12")

    genes = _genes(rebuild_gene_features([cds, part_gene], LENGTH))

    assert len(genes) == 1
    assert list(genes[0].location.parts) == list(cds.location.parts)
    assert "trans_splicing" in genes[0].qualifiers


def test_gene_only_pseudogene_is_kept_and_stray_copies_are_dropped() -> None:
    cds = _feature("CDS", [(82639, 82921, -1)], gene="rpl23")
    pseudo = _feature("gene", [(54776, 55820, 1)], gene="rpl23", pseudo="")
    stray = _feature("gene", [(30000, 30300, 1)], gene="rpl23")
    rna_gene = _feature("gene", [(5000, 5072, 1)], gene="tRNA-Gly")

    genes = _genes(rebuild_gene_features([cds, pseudo, stray, rna_gene], LENGTH))

    assert sorted(_span(g) for g in genes) == [(54776, 55820), (82639, 82921)]


def test_cds_across_the_origin_does_not_span_the_genome() -> None:
    cds = _feature("CDS", [(LENGTH - 30, LENGTH, 1), (0, 300, 1)], gene="psbA")

    genes = _genes(rebuild_gene_features([cds], LENGTH))

    assert list(genes[0].location.parts) == list(cds.location.parts)
