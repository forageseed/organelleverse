"""Curation checks for the packaged plastome GenBank references.

The plastome backend transfers gene names from the nearest reference, so a
mislabelled reference feature reappears in every genome annotated from it.
These checks pin the corrections made to the NCBI records (Theobroma psbB was
named psi_psbT; Theobroma/Prunus rpoC2 carried the rpoC1 product; Ginkgo
4.5S rRNA was named rrn4/rrn45).
"""

from __future__ import annotations

import statistics
from functools import cache

import pytest
from Bio import SeqIO

from organelleverse.annotation.data import plastome_data_dir, plastome_reference_dir

RRNA_GENES = {"rrn16", "rrn23", "rrn4.5", "rrn5"}


@cache
def _features():
    rows = []
    for path in sorted(plastome_reference_dir().glob("*.gb")):
        record = SeqIO.read(path, "genbank")
        for feature in record.features:
            if feature.type in {"CDS", "rRNA"}:
                gene = feature.qualifiers.get("gene", [""])[0]
                product = feature.qualifiers.get("product", [""])[0]
                rows.append((path.stem, feature.type, gene, product))
    return tuple(rows)


@pytest.mark.parametrize(
    ("marker", "gene"),
    [("CP47", "psbB"), ("CP43", "psbC")],
)
def test_photosystem_ii_antenna_genes_match_their_products(marker: str, gene: str) -> None:
    wrong = [
        (ref, g)
        for ref, kind, g, product in _features()
        if kind == "CDS" and marker in product and g != gene
    ]
    assert wrong == []


def test_rpoc2_is_not_labelled_with_the_rpoc1_product() -> None:
    wrong = [
        ref
        for ref, kind, gene, product in _features()
        if kind == "CDS" and gene == "rpoC2" and product.rstrip().endswith("beta' subunit")
    ]
    assert wrong == []


def test_rrna_gene_names_are_standard() -> None:
    odd = sorted(
        {
            (ref, gene)
            for ref, kind, gene, _ in _features()
            if kind == "rRNA" and gene.startswith("rrn") and gene not in RRNA_GENES
        }
    )
    # rrn16S / rrn5S style suffixes are normalised elsewhere; truncated names are not.
    assert [item for item in odd if item[1].rstrip("S") not in RRNA_GENES] == []


@pytest.mark.parametrize("gene", ["rrn16", "rrn23", "rrn45", "rrn5"])
def test_rrna_query_references_have_no_length_outliers(gene: str) -> None:
    # A 263 bp "4.5S" and a 135 bp "5S" query made nhmmer extend every
    # 4.5S/5S call in every plastome over the same flanking spacer.
    path = plastome_data_dir() / "rrna_refs" / f"{gene}.fasta"
    lengths = [len(record) for record in SeqIO.parse(path, "fasta")]
    median = statistics.median(lengths)
    assert lengths
    assert all(abs(length - median) <= 0.10 * median for length in lengths), lengths


def test_every_reference_trna_is_named_after_parsing() -> None:
    from organelleverse.annotation.plastome.references import _parse_reference

    unnamed = [
        (path.stem, feature.feature_id)
        for path in sorted(plastome_reference_dir().glob("*.gb"))
        for feature in _parse_reference(path).features
        if feature.feature_type == "tRNA"
        and not feature.gene
        and "pseudo" not in " ".join(feature.feature.qualifiers.get("note", []))
    ]
    assert unnamed == []


def test_no_placeholder_products_or_obsolete_psbz_name() -> None:
    bad = [
        (ref, gene, product)
        for ref, kind, gene, product in _features()
        if product == "no product string in file" or gene == "lhbA"
    ]
    assert bad == []
