"""Pure-function core of the fetch suite: batching, hashing, parsing, retry.

All offline. No network, no monkeypatching of urllib.
"""

from __future__ import annotations

import pytest

from organelleverse.core.errors import OrganelleExecutionError, OrganelleParameterError
from organelleverse.fetch._http import (
    MAX_URL_LENGTH,
    batch_accessions_for_url,
    retry_with_backoff,
)
from organelleverse.fetch.manifest import accession_set_sha256, build_manifest
from organelleverse.fetch.ncbi import (
    build_organelle_term,
    parse_esearch_xml,
    parse_release_number,
)

# ─────────────────────────────────────────────────────────────
# accession set hashing — the determinism primitive
# ─────────────────────────────────────────────────────────────


def test_accession_hash_is_order_independent() -> None:
    """The hash identifies a *set*, so input order must not change it."""
    a = accession_set_sha256(["NC_000932.1", "NC_037304.1", "MT_000001.1"])
    b = accession_set_sha256(["MT_000001.1", "NC_000932.1", "NC_037304.1"])
    assert a == b


def test_accession_hash_deduplicates() -> None:
    """A duplicated accession is the same set."""
    once = accession_set_sha256(["NC_000932.1"])
    twice = accession_set_sha256(["NC_000932.1", "NC_000932.1"])
    assert once == twice


def test_accession_hash_is_sensitive_to_membership() -> None:
    """Adding a record must change the hash, or the benchmark is worthless."""
    a = accession_set_sha256(["NC_000932.1"])
    b = accession_set_sha256(["NC_000932.1", "NC_037304.1"])
    assert a != b


def test_accession_hash_is_a_sha256_hexdigest() -> None:
    digest = accession_set_sha256(["NC_000932.1"])
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_empty_accession_set_is_rejected() -> None:
    """An empty set has no scientific meaning; fail loudly rather than hash ''."""
    with pytest.raises(OrganelleParameterError):
        accession_set_sha256([])


# ─────────────────────────────────────────────────────────────
# URL batching — NCBI returns HTTP 414 above ~2000 chars
# ─────────────────────────────────────────────────────────────


def test_batching_keeps_every_url_under_the_limit() -> None:
    accessions = [f"NC_{i:06d}.1" for i in range(500)]
    base = 120
    batches = batch_accessions_for_url(accessions, base_url_length=base)
    for batch in batches:
        url_len = base + len(",".join(batch))
        assert url_len <= MAX_URL_LENGTH


def test_batching_loses_no_accession_and_preserves_no_duplicates() -> None:
    accessions = [f"NC_{i:06d}.1" for i in range(500)]
    batches = batch_accessions_for_url(accessions, base_url_length=120)
    flat = [a for batch in batches for a in batch]
    assert flat == accessions  # order preserved, nothing dropped


def test_small_input_is_a_single_batch() -> None:
    batches = batch_accessions_for_url(["NC_000932.1", "NC_037304.1"], base_url_length=100)
    assert len(batches) == 1


def test_batching_rejects_a_base_url_that_leaves_no_room() -> None:
    with pytest.raises(OrganelleParameterError):
        batch_accessions_for_url(["NC_000932.1"], base_url_length=MAX_URL_LENGTH)


# ─────────────────────────────────────────────────────────────
# retry / backoff — NCBI rate-limits aggressively
# ─────────────────────────────────────────────────────────────


def test_retry_returns_the_value_on_first_success() -> None:
    calls: list[int] = []

    def op() -> str:
        calls.append(1)
        return "ok"

    assert retry_with_backoff(op, max_attempts=3, initial_delay=0.0) == "ok"
    assert len(calls) == 1


def test_retry_recovers_from_transient_failures() -> None:
    calls: list[int] = []

    def op() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise OSError("transient")
        return "ok"

    assert retry_with_backoff(op, max_attempts=5, initial_delay=0.0) == "ok"
    assert len(calls) == 3


def test_retry_raises_a_structured_retryable_error_when_exhausted() -> None:
    def op() -> str:
        raise OSError("always down")

    with pytest.raises(OrganelleExecutionError) as excinfo:
        retry_with_backoff(op, max_attempts=2, initial_delay=0.0)

    error = excinfo.value
    assert error.retryable is True
    assert error.code.startswith("network.")
    assert error.as_dict()["details"]["attempts"] == 2


def test_retry_does_not_swallow_a_parameter_error() -> None:
    """A bad request will never succeed on retry — fail fast, don't hammer NCBI."""

    def op() -> str:
        raise OrganelleParameterError(code="input.bad", message="nope")

    with pytest.raises(OrganelleParameterError):
        retry_with_backoff(op, max_attempts=5, initial_delay=0.0)


# ─────────────────────────────────────────────────────────────
# NCBI response parsing
# ─────────────────────────────────────────────────────────────


def test_parse_release_number() -> None:
    assert parse_release_number("236\n") == 236


def test_parse_release_number_rejects_garbage() -> None:
    with pytest.raises(OrganelleExecutionError):
        parse_release_number("<html>404</html>")


ESEARCH_XML = """<?xml version="1.0" encoding="UTF-8" ?>
<eSearchResult>
  <Count>58082</Count>
  <RetMax>3</RetMax>
  <IdList>
    <Id>2760155037</Id>
    <Id>2760155036</Id>
    <Id>2760155035</Id>
  </IdList>
</eSearchResult>
"""


def test_parse_esearch_xml() -> None:
    count, uids = parse_esearch_xml(ESEARCH_XML)
    assert count == 58082
    assert uids == ["2760155037", "2760155036", "2760155035"]


def test_parse_esearch_xml_handles_zero_hits() -> None:
    xml = "<eSearchResult><Count>0</Count><IdList></IdList></eSearchResult>"
    count, uids = parse_esearch_xml(xml)
    assert count == 0
    assert uids == []


def test_parse_esearch_xml_rejects_an_error_page() -> None:
    with pytest.raises(OrganelleExecutionError):
        parse_esearch_xml("<html>Service unavailable</html>")


# ─────────────────────────────────────────────────────────────
# Entrez query construction — the scoping of a scientific claim
# ─────────────────────────────────────────────────────────────


def test_organelle_term_is_organelle_specific() -> None:
    plastid = build_organelle_term(organelle="plastid")
    mito = build_organelle_term(organelle="mitochondrion")
    assert "chloroplast[filter]" in plastid
    assert "mitochondrion[filter]" in mito
    assert plastid != mito


def test_organelle_term_defaults_to_complete_genomes() -> None:
    term = build_organelle_term(organelle="plastid")
    assert '"complete genome"[Title]' in term


def test_organelle_term_scopes_to_taxon() -> None:
    term = build_organelle_term(organelle="plastid", taxon="Viridiplantae")
    assert "Viridiplantae[Organism]" in term


def test_organelle_term_can_drop_the_completeness_filter() -> None:
    """Plant mitochondria are hard to assemble; recall sometimes beats precision."""
    term = build_organelle_term(organelle="mitochondrion", complete_only=False)
    assert "complete genome" not in term


def test_organelle_term_rejects_an_unknown_organelle() -> None:
    with pytest.raises(OrganelleParameterError):
        build_organelle_term(organelle="nucleus")


# ─────────────────────────────────────────────────────────────
# esummary parsing — fixture is a real nuccore response
#
# Regression guard: the first version of this parser read `Slen` and `Organism`,
# neither of which nuccore returns. Every offline test passed and every real
# record came back with length=0 — which would have silently emptied any
# min_length filter. Fixtures come from the wire, not from memory.
# ─────────────────────────────────────────────────────────────

REAL_ESUMMARY = """<?xml version="1.0" encoding="UTF-8" ?>
<eSummaryResult>
<DocSum>
	<Id>2441074688</Id>
	<Item Name="Caption" Type="String">OP474144</Item>
	<Item Name="Title" Type="String">Arabidopsis thaliana chloroplast, complete genome</Item>
	<Item Name="Gi" Type="Integer">2441074688</Item>
	<Item Name="UpdateDate" Type="String">2024/06/11</Item>
	<Item Name="TaxId" Type="Integer">3702</Item>
	<Item Name="Length" Type="Integer">154478</Item>
	<Item Name="Status" Type="String">live</Item>
	<Item Name="AccessionVersion" Type="String">OP474144.1</Item>
</DocSum>
</eSummaryResult>"""


def test_esummary_reads_the_length_item_that_nuccore_actually_sends() -> None:
    from organelleverse.fetch.ncbi import parse_esummary_xml

    (record,) = parse_esummary_xml(REAL_ESUMMARY)
    assert record["length"] == 154478  # `Length`, not `Slen`
    assert record["accession"] == "OP474144.1"
    assert record["taxid"] == "3702"


def test_esummary_derives_organism_from_title_since_nuccore_sends_none() -> None:
    from organelleverse.fetch.ncbi import parse_esummary_xml

    (record,) = parse_esummary_xml(REAL_ESUMMARY)
    assert record["organism"] == "Arabidopsis thaliana"


def test_esummary_tolerates_the_legacy_slen_item() -> None:
    from organelleverse.fetch.ncbi import parse_esummary_xml

    xml = """<eSummaryResult><DocSum>
      <Item Name="AccessionVersion" Type="String">X1.1</Item>
      <Item Name="Title" Type="String">Zea mays mitochondrion, complete genome</Item>
      <Item Name="Slen" Type="Integer">569630</Item>
    </DocSum></eSummaryResult>"""
    (record,) = parse_esummary_xml(xml)
    assert record["length"] == 569630
    assert record["organism"] == "Zea mays"


# ─────────────────────────────────────────────────────────────
# The completeness clause is organelle-specific
#
# `"complete sequence"[Title]` means opposite things per organelle:
#   mitochondrion — how a multipartite chromosome is titled (needed)
#   chloroplast   — 99.6% gene/spacer fragments (poison)
# ─────────────────────────────────────────────────────────────


def test_mitochondrion_query_admits_multipartite_chromosome_records() -> None:
    term = build_organelle_term(organelle="mitochondrion")
    assert '"complete sequence"[Title]' in term
    assert "chromosome[Title]" in term


def test_plastid_query_excludes_complete_sequence_which_is_236k_fragments() -> None:
    """Adding it turns 58,082 plastomes into 295,091 mostly-fragments."""
    term = build_organelle_term(organelle="plastid")
    assert '"complete sequence"[Title]' not in term


def test_plastid_query_admits_the_titles_that_never_say_complete() -> None:
    """`genome assembly, organelle: plastid:chloroplast` is a real 154 kb
    plastome. The old clause missed 3,136 genomes titled this way."""
    term = build_organelle_term(organelle="plastid")
    assert '"genome assembly"[Title]' in term
    assert '"whole genome shotgun"[Title]' in term
    assert "chromosome[Title]" in term  # Homalomena perplexa: 12 plastid chromosomes


def test_sources_tried_is_recorded_when_given() -> None:
    manifest = build_manifest(
        source="gir",
        organelle="nuclear",
        accessions=["gir:Rosaceae/v1.2_Malus_domestica"],
        scope={"taxon": "Malus domestica"},
        sources_tried=[
            {"source": "tair", "status": "miss", "detail": "not arabidopsis"},
            {"source": "ncbi", "status": "miss", "detail": "no assembly"},
            {"source": "gir", "status": "hit", "detail": "matched"},
        ],
    )
    assert manifest["sources_tried"] == [
        {"source": "tair", "status": "miss", "detail": "not arabidopsis"},
        {"source": "ncbi", "status": "miss", "detail": "no assembly"},
        {"source": "gir", "status": "hit", "detail": "matched"},
    ]


def test_sources_tried_is_absent_when_not_given() -> None:
    manifest = build_manifest(
        source="gir",
        organelle="nuclear",
        accessions=["gir:x"],
        scope={},
    )
    assert "sources_tried" not in manifest


def test_the_source_literal_accepts_every_new_source() -> None:
    for source in ("ngdc_gwh", "gir", "imp", "pgd", "tair"):
        manifest = build_manifest(source=source, organelle="nuclear", accessions=["x:1"], scope={})
        assert manifest["source"] == source
