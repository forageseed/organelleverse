"""Metadata filters. Pure functions, no network."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from organelleverse.core.errors import OrganelleParameterError
from organelleverse.fetch.filters import (
    FetchFilters,
    apply_filters,
    count_ambiguous,
    is_refseq_accession,
    parse_partial_date,
)


def record(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "accession": "OP474144.1",
        "length": 154_478,
        "status": "live",
        "create_date": "2023/02/12",
        "update_date": "2024/06/11",
        "organism": "Arabidopsis thaliana",
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────
# esummary tier
# ─────────────────────────────────────────────────────────────


def test_length_window() -> None:
    plastome = record(length=154_478)
    mito = record(accession="NC_007982.1", length=569_630)
    kept = apply_filters([plastome, mito], FetchFilters(min_length=200_000))
    assert [r["accession"] for r in kept] == ["NC_007982.1"]

    kept = apply_filters([plastome, mito], FetchFilters(max_length=200_000))
    assert [r["accession"] for r in kept] == ["OP474144.1"]


def test_empty_length_interval_is_rejected_up_front() -> None:
    """min > max returns zero records — say so now, not after 500 requests."""
    with pytest.raises(OrganelleParameterError):
        FetchFilters(min_length=200_000, max_length=100_000)


def test_refseq_only_splits_the_curated_from_the_primary_submission() -> None:
    """NC_000932.1 and its GenBank original are the SAME genome — the core
    redundancy problem for organelles."""
    refseq = record(accession="NC_000932.1")
    genbank = record(accession="OP474144.1")
    kept = apply_filters([refseq, genbank], FetchFilters(refseq_only=True))
    assert [r["accession"] for r in kept] == ["NC_000932.1"]


def test_source_database_genbank_excludes_refseq() -> None:
    refseq = record(accession="NC_000932.1")
    genbank = record(accession="OP474144.1")
    kept = apply_filters([refseq, genbank], FetchFilters(source_database="genbank"))
    assert [r["accession"] for r in kept] == ["OP474144.1"]


def test_unknown_source_database_is_rejected() -> None:
    with pytest.raises(OrganelleParameterError):
        FetchFilters(source_database="ena")


@pytest.mark.parametrize(
    "accession,expected",
    [
        ("NC_000932.1", True),
        ("NW_123456.1", True),
        ("OP474144.1", False),
        ("MT012345.1", False),
    ],
)
def test_refseq_prefix_detection(accession: str, expected: bool) -> None:
    assert is_refseq_accession(accession) is expected


def test_suppressed_records_are_dropped_by_default() -> None:
    live = record(status="live")
    dead = record(accession="XX000000.1", status="suppressed")
    kept = apply_filters([live, dead], FetchFilters())
    assert [r["accession"] for r in kept] == ["OP474144.1"]


def test_release_date_window() -> None:
    old = record(accession="A1.1", create_date="2019/01/01")
    new = record(accession="B1.1", create_date="2024/05/01")
    kept = apply_filters([old, new], FetchFilters(min_release_date="2023"))
    assert [r["accession"] for r in kept] == ["B1.1"]


# ─────────────────────────────────────────────────────────────
# date parsing — NCBI mixes year, year/month, and full dates
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2024", date(2024, 1, 1)),
        ("2024/06", date(2024, 6, 1)),
        ("2024/06/11", date(2024, 6, 11)),
        ("2024-06-11", date(2024, 6, 11)),
    ],
)
def test_partial_dates(value: str, expected: date) -> None:
    assert parse_partial_date(value) == expected


def test_bad_date_is_rejected() -> None:
    with pytest.raises(OrganelleParameterError):
        parse_partial_date("last Tuesday")


def test_inverted_date_interval_is_rejected() -> None:
    with pytest.raises(OrganelleParameterError):
        FetchFilters(min_release_date="2024", max_release_date="2020")


# ─────────────────────────────────────────────────────────────
# GenBank tier — must never silently no-op
# ─────────────────────────────────────────────────────────────


def test_genbank_tier_filter_without_genbank_metadata_raises() -> None:
    """A filter that silently matches everything is how a wrong dataset ends up
    in a paper. Fail loudly instead."""
    with pytest.raises(OrganelleParameterError) as excinfo:
        apply_filters([record()], FetchFilters(min_gene_count=100))
    assert excinfo.value.code == "input.genbank_metadata_required"


def test_gene_count_window_with_genbank_metadata() -> None:
    """Plastomes carry ~130 genes; a record with 12 is a fragment, not a genome."""
    good = record(gene_count=130)
    fragment = record(accession="X1.1", gene_count=12)
    kept = apply_filters(
        [good, fragment], FetchFilters(min_gene_count=100), have_genbank_metadata=True
    )
    assert [r["accession"] for r in kept] == ["OP474144.1"]


def test_max_ambiguous_drops_gappy_assemblies() -> None:
    clean = record(ambiguous_count=0)
    gappy = record(accession="X1.1", ambiguous_count=5000)
    kept = apply_filters(
        [clean, gappy], FetchFilters(max_ambiguous=100), have_genbank_metadata=True
    )
    assert [r["accession"] for r in kept] == ["OP474144.1"]


def test_has_proteins_requires_a_positive_count() -> None:
    annotated = record(protein_count=80)
    bare = record(accession="X1.1", protein_count=0)
    kept = apply_filters(
        [annotated, bare], FetchFilters(has_proteins=True), have_genbank_metadata=True
    )
    assert [r["accession"] for r in kept] == ["OP474144.1"]


def test_collection_date_and_geography() -> None:
    cn = record(collection_date="2021/05", geo_location="China: Yunnan")
    us = record(accession="X1.1", collection_date="2018/03", geo_location="USA")
    kept = apply_filters(
        [cn, us],
        FetchFilters(min_collection_date="2020", geo_location="China"),
        have_genbank_metadata=True,
    )
    assert [r["accession"] for r in kept] == ["OP474144.1"]


def test_text_filters_are_case_insensitive_substrings() -> None:
    r = record(submitter_institution="Chinese Academy of Sciences")
    kept = apply_filters(
        [r], FetchFilters(submitter_institution="chinese academy"), have_genbank_metadata=True
    )
    assert len(kept) == 1


def test_active_reports_only_the_filters_in_force() -> None:
    """The manifest records the scope of the claim — so it must be exact."""
    active = FetchFilters(min_length=100_000, refseq_only=True).active()
    assert active == {"min_length": 100_000, "refseq_only": True}


def test_count_ambiguous_counts_non_acgt() -> None:
    assert count_ambiguous("ACGTNNNRYacgt") == 5


# ─────────────────────────────────────────────────────────────
# Gene records are not genomes
#
# Verified against Viridiplantae (2026-07-13): of 237,010 chloroplast records
# titled "complete sequence", 236,172 (99.6%) are under 10 kb — genes, introns
# and intergenic spacers. Counting one as a genome is the same class of error as
# counting one chromosome as a genome.
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "title",
    [
        "Camassia howellii isolate CAHO_H4_7 trnD-trnY intergenic spacer, partial sequence",
        "Cucumis sativus cultivar Calypso apocytochrome b (cob) gene, complete cds",
        "Arabidopsis thaliana voucher X ribulose-1,5-bisphosphate carboxylase (rbcL) gene",
        "Zea mays maturase K (matK) gene, partial cds; chloroplast",
        "Oryza sativa 18S ribosomal RNA gene, complete sequence",
    ],
)
def test_gene_and_spacer_records_are_rejected(title: str) -> None:
    from organelleverse.fetch.filters import looks_like_gene_record

    assert looks_like_gene_record(title) is True


@pytest.mark.parametrize(
    "title",
    [
        "Arabidopsis thaliana chloroplast, complete genome",
        "Cucumis sativus mitochondrion chromosome 2, complete sequence",
        "Begonia fimbristipula mitochondrion, complete genome",
        "Arabidopsis thaliana strain Can-0 genome assembly, organelle: plastid:chloroplast",
        "UNVERIFIED: Homalomena perplexa chromosome 9 chloroplast sequence",
    ],
)
def test_genome_and_chromosome_records_survive(title: str) -> None:
    """A chromosome is a genome part however small — Begonia's chromosome 9 is
    2,354 bp and must never be filtered out as a fragment."""
    from organelleverse.fetch.filters import looks_like_gene_record

    assert looks_like_gene_record(title) is False


def test_gene_records_are_dropped_by_default() -> None:
    genome = record(title="Arabidopsis thaliana chloroplast, complete genome")
    fragment = record(accession="X1.1", length=971, title="trnD-trnY intergenic spacer")
    kept = apply_filters([genome, fragment], FetchFilters())
    assert [r["accession"] for r in kept] == ["OP474144.1"]


def test_gene_record_exclusion_can_be_turned_off() -> None:
    fragment = record(accession="X1.1", length=971, title="trnD-trnY intergenic spacer")
    kept = apply_filters([fragment], FetchFilters(exclude_gene_records=False))
    assert len(kept) == 1


# ─────────────────────────────────────────────────────────────
# Both organelles are mostly gene fragments (Viridiplantae, 2026-07-13):
#
#   mitochondrion  89,567 records — 68,877 (77%) titled "gene"
#                                    6,557 titled "chromosome" (real molecules)
#   chloroplast  1,779,010 records — 236,172 of the 237,010 "complete sequence"
#                                    ones are under 10 kb: genes and spacers
#
# The molecules are NOT separable by length. Taxillus chinensis has a 1,260 bp
# mitochondrial chromosome; Pinus contorta has a 339 bp NADH gene record. Only
# the title tells them apart, and "chromosome" must win.
# ─────────────────────────────────────────────────────────────

REAL_FRAGMENTS = [
    "Rafflesia sp. voucher itci01 MatR (matR) gene, partial cds; mitochondrial",
    "Rafflesia sp. voucher itci01 ATP synthase F0 subunit 6 (ATP6) gene, partial cds",
    "Pinus contorta subsp. contorta isolate 95-5 NADH dehydrogenase subunit 5 gene",
    "Camassia howellii isolate CAHO_H4_7 trnD-trnY intergenic spacer, partial sequence",
]
REAL_MOLECULES = [
    "Taxillus chinensis chromosome 18 mitochondrion, complete sequence",  # 1,260 bp
    "Alnus rubra chromosome 2 mitochondrion, complete sequence",  # 1,592 bp
    "Begonia fimbristipula mitochondrion, complete genome",  # 2,354 bp chr 9
    "UNVERIFIED: Homalomena perplexa chromosome 9 chloroplast sequence",  # 1,959 bp
]


@pytest.mark.parametrize("title", REAL_FRAGMENTS)
def test_real_gene_fragments_are_dropped_in_both_organelles(title: str) -> None:
    from organelleverse.fetch.filters import looks_like_gene_record

    assert looks_like_gene_record(title) is True


@pytest.mark.parametrize("title", REAL_MOLECULES)
def test_a_tiny_real_chromosome_is_never_mistaken_for_a_fragment(title: str) -> None:
    """Taxillus chinensis chromosome 18 is 1,260 bp — smaller than a 339 bp gene
    record is far from, and four times smaller than some. A length cutoff would
    delete it. The word "chromosome" is what saves it.
    """
    from organelleverse.fetch.filters import looks_like_gene_record

    assert looks_like_gene_record(title) is False
