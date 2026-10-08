"""Staged retrieval against a fake transport. Fully offline.

The transport is injected, so these tests exercise the real staging logic
(esearch history -> esummary paging -> filter -> epost -> efetch) without
touching the network.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import pytest

from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleExecutionError, OrganelleParameterError
from organelleverse.fetch.filters import FetchFilters
from organelleverse.fetch.ncbi import entrez_query, fetch_accessions, refseq_snapshot

# esearch with usehistory=y returns a server-side result set, not the ids.
ESEARCH_XML = """<eSearchResult>
  <Count>2</Count>
  <QueryKey>1</QueryKey>
  <WebEnv>MCID_FAKE</WebEnv>
</eSearchResult>"""

EPOST_XML = """<ePostResult>
  <QueryKey>1</QueryKey>
  <WebEnv>MCID_POSTED</WebEnv>
</ePostResult>"""

# Verbatim shape of a real nuccore esummary response (fetched 2026-07-13).
# An invented fixture hid a real bug here: the length item is `Length`, not
# `Slen`, and there is no `Organism` item at all. Fixtures come from the wire.
ESUMMARY_XML = """<eSummaryResult>
  <DocSum>
    <Id>111</Id>
    <Item Name="Caption" Type="String">NC_000932</Item>
    <Item Name="Title" Type="String">Arabidopsis thaliana chloroplast, complete genome</Item>
    <Item Name="CreateDate" Type="String">2023/02/12</Item>
    <Item Name="UpdateDate" Type="String">2024/06/11</Item>
    <Item Name="TaxId" Type="Integer">3702</Item>
    <Item Name="Length" Type="Integer">154478</Item>
    <Item Name="Status" Type="String">live</Item>
    <Item Name="AccessionVersion" Type="String">NC_000932.1</Item>
  </DocSum>
  <DocSum>
    <Id>222</Id>
    <Item Name="Caption" Type="String">OP474144</Item>
    <Item Name="Title" Type="String">Oryza sativa chloroplast, complete genome</Item>
    <Item Name="CreateDate" Type="String">2019/03/01</Item>
    <Item Name="UpdateDate" Type="String">2023/01/05</Item>
    <Item Name="TaxId" Type="Integer">4530</Item>
    <Item Name="Length" Type="Integer">134525</Item>
    <Item Name="Status" Type="String">live</Item>
    <Item Name="AccessionVersion" Type="String">OP474144.1</Item>
  </DocSum>
</eSummaryResult>"""

GBSEQ_XML = """<GBSet>
  <GBSeq>
    <GBSeq_accession-version>NC_000932.1</GBSeq_accession-version>
    <GBSeq_length>154478</GBSeq_length>
    <GBSeq_organism>Arabidopsis thaliana</GBSeq_organism>
    <GBSeq_feature-table>
      <GBFeature>
        <GBFeature_key>source</GBFeature_key>
        <GBFeature_quals>
          <GBQualifier><GBQualifier_name>collection_date</GBQualifier_name>
            <GBQualifier_value>2021/05</GBQualifier_value></GBQualifier>
          <GBQualifier><GBQualifier_name>geo_loc_name</GBQualifier_name>
            <GBQualifier_value>China: Yunnan</GBQualifier_value></GBQualifier>
        </GBFeature_quals>
      </GBFeature>
      <GBFeature><GBFeature_key>gene</GBFeature_key></GBFeature>
      <GBFeature><GBFeature_key>gene</GBFeature_key></GBFeature>
      <GBFeature><GBFeature_key>CDS</GBFeature_key></GBFeature>
    </GBSeq_feature-table>
    <GBSeq_sequence>acgtacgtnn</GBSeq_sequence>
  </GBSeq>
  <GBSeq>
    <GBSeq_accession-version>OP474144.1</GBSeq_accession-version>
    <GBSeq_length>134525</GBSeq_length>
    <GBSeq_organism>Oryza sativa</GBSeq_organism>
    <GBSeq_feature-table>
      <GBFeature><GBFeature_key>gene</GBFeature_key></GBFeature>
    </GBSeq_feature-table>
    <GBSeq_sequence>acgtacgtac</GBSeq_sequence>
  </GBSeq>
</GBSet>"""

GBFF = "LOCUS       NC_000932  154478 bp    DNA     circular PLN\nVERSION     NC_000932.1\n//\n"

# What ftp.ncbi.nlm.nih.gov/refseq/release/plastid/ actually serves.
REFSEQ_INDEX = """<html><body>
<a href="plastid.1.genomic.gbff.gz">plastid.1.genomic.gbff.gz</a>
<a href="plastid.2.genomic.gbff.gz">plastid.2.genomic.gbff.gz</a>
<a href="plastid.1.protein.faa.gz">plastid.1.protein.faa.gz</a>
</body></html>"""


class FakeTransport:
    """Replays canned responses and records what was asked for.

    ``posted`` matters: with history paging, the accessions to download travel
    in the epost *body*, not the efetch URL.
    """

    def __init__(self, *, release: bytes = b"236\n", hits: int = 2) -> None:
        self.calls: list[str] = []
        self.posted: list[str] = []
        self._release = release
        self._hits = hits

    def get(self, url: str, *, timeout: float = 0.0) -> bytes:
        self.calls.append(url)
        if "RELEASE_NUMBER" in url:
            return self._release
        if "esearch" in url:
            return ESEARCH_XML.replace("<Count>2</Count>", f"<Count>{self._hits}</Count>").encode()
        if "esummary" in url:
            return ESUMMARY_XML.encode()
        if "efetch" in url and "retmode=xml" in url:
            return GBSEQ_XML.encode()
        if "efetch" in url:
            return GBFF.encode()
        if url.endswith(".gbff.gz"):
            return gzip.compress(GBFF.encode())
        if "refseq/release" in url and url.endswith("/"):
            return REFSEQ_INDEX.encode()
        raise AssertionError(f"unexpected GET: {url}")

    def post(self, url: str, data: bytes, *, timeout: float = 0.0) -> bytes:
        self.calls.append(url)
        if "epost" in url:
            from urllib.parse import parse_qs

            ids = parse_qs(data.decode()).get("id", [""])[0]
            self.posted.extend(i for i in ids.split(",") if i)
            return EPOST_XML.encode()
        raise AssertionError(f"unexpected POST: {url}")

    def urls_matching(self, needle: str) -> list[str]:
        return [u for u in self.calls if needle in u]


def manifest_of(data: OrganelleData) -> Any:
    return data.payload["manifest"]


# ─────────────────────────────────────────────────────────────
# staged retrieval
# ─────────────────────────────────────────────────────────────


def test_entrez_query_returns_organelle_data(tmp_path: Path) -> None:
    data = entrez_query(
        organelle="plastid", taxon="Viridiplantae", dest=tmp_path, transport=FakeTransport()
    )
    assert isinstance(data, OrganelleData)
    assert data.modality == "organelle_records"


def test_entrez_query_fetches_metadata_before_sequences(tmp_path: Path) -> None:
    """The gget-virus staging rule: never download a sequence you will discard."""
    transport = FakeTransport()
    entrez_query(organelle="plastid", dest=tmp_path, transport=transport)

    order = [
        "esearch" if "esearch" in u else "esummary" if "esummary" in u else "efetch"
        for u in transport.calls
        if any(k in u for k in ("esearch", "esummary", "efetch"))
    ]
    assert order.index("esearch") < order.index("esummary") < order.index("efetch")


def test_entrez_query_only_downloads_records_that_survived_the_filter(
    tmp_path: Path,
) -> None:
    """The filter must cut the download, not just the final table.

    With history paging the accessions travel in the epost body, so that is
    where the evidence lives.
    """
    transport = FakeTransport()
    data = entrez_query(
        organelle="plastid",
        dest=tmp_path,
        transport=transport,
        min_length=140_000,  # keeps Arabidopsis (154478), drops Oryza (134525)
    )
    assert transport.posted == ["NC_000932.1"]
    assert manifest_of(data)["n_records"] == 1
    assert manifest_of(data)["hits_before_filter"] == 2
    assert manifest_of(data)["n_filtered_out"] == 1


def test_entrez_query_records_a_manifest_with_a_stable_accession_hash(
    tmp_path: Path,
) -> None:
    data = entrez_query(organelle="plastid", dest=tmp_path, transport=FakeTransport())
    m = manifest_of(data)

    assert m["schema_version"] == "organelleverse.fetch.manifest.v1"
    assert m["source"] == "entrez_query"
    assert m["organelle"] == "plastid"
    assert m["n_records"] == 2
    assert len(m["accession_set_sha256"]) == 64
    assert list(m["accessions"]) == ["NC_000932.1", "OP474144.1"]


def test_entrez_query_is_reproducible_across_runs(tmp_path: Path) -> None:
    """Same inputs, same accession set hash — the property a benchmark rests on."""
    first = entrez_query(organelle="plastid", dest=tmp_path / "a", transport=FakeTransport())
    second = entrez_query(organelle="plastid", dest=tmp_path / "b", transport=FakeTransport())
    assert manifest_of(first)["accession_set_sha256"] == manifest_of(second)["accession_set_sha256"]


def test_entrez_query_writes_a_content_addressed_artifact(tmp_path: Path) -> None:
    data = entrez_query(organelle="plastid", dest=tmp_path, transport=FakeTransport())
    records = data.artifacts["records"]
    assert records.format == "genbank"
    assert len(records.sha256) == 64
    assert Path(records.uri).is_file()


def test_entrez_query_declares_a_scoped_claim(tmp_path: Path) -> None:
    """A claim must name organelle + taxon + every active filter — never 'fetch works'."""
    data = entrez_query(
        organelle="plastid",
        taxon="Viridiplantae",
        dest=tmp_path,
        min_length=140_000,
        transport=FakeTransport(),
    )
    scope = manifest_of(data)["scope"]
    assert scope["organelle"] == "plastid"
    assert scope["taxon"] == "Viridiplantae"
    assert scope["complete_only"] is True
    assert scope["min_length"] == 140_000


def test_entrez_query_reports_zero_hits_without_fetching(tmp_path: Path) -> None:
    transport = FakeTransport(hits=0)
    data = entrez_query(
        organelle="mitochondrion", dest=tmp_path, transport=transport, try_name_variants=False
    )
    assert manifest_of(data)["n_records"] == 0
    assert transport.urls_matching("efetch") == []
    assert transport.posted == []


# ─────────────────────────────────────────────────────────────
# whole-corpus paging — the reason history exists
# ─────────────────────────────────────────────────────────────


def test_entrez_query_pages_the_whole_result_set_by_default(tmp_path: Path) -> None:
    """max_records=None means all of them: 5 hits at page_size=2 -> 3 esummary pages."""
    transport = FakeTransport(hits=5)
    entrez_query(
        organelle="plastid", dest=tmp_path, transport=transport, page_size=2, max_records=None
    )
    pages = transport.urls_matching("esummary")
    assert len(pages) == 3
    assert "retstart=0&retmax=2" in pages[0]
    assert "retstart=2&retmax=2" in pages[1]
    assert "retstart=4&retmax=1" in pages[2]


def test_entrez_query_respects_an_explicit_max_records(tmp_path: Path) -> None:
    transport = FakeTransport(hits=100)
    entrez_query(
        organelle="plastid", dest=tmp_path, transport=transport, page_size=2, max_records=3
    )
    pages = transport.urls_matching("esummary")
    assert len(pages) == 2  # 2 + 1


def test_esummary_pages_use_the_history_server(tmp_path: Path) -> None:
    transport = FakeTransport()
    entrez_query(organelle="plastid", dest=tmp_path, transport=transport)
    (page,) = transport.urls_matching("esummary")
    assert "WebEnv=MCID_FAKE" in page
    assert "query_key=1" in page


# ─────────────────────────────────────────────────────────────
# name variants — plant nomenclature is messier than viral
# ─────────────────────────────────────────────────────────────


def test_a_subspecies_that_misses_falls_back_to_the_species(tmp_path: Path) -> None:
    class MissThenHit(FakeTransport):
        def __init__(self) -> None:
            super().__init__()
            self.terms: list[str] = []

        def get(self, url: str, *, timeout: float = 0.0) -> bytes:
            if "esearch" in url:
                self.calls.append(url)
                self.terms.append(url)
                # The exact subspecies match returns nothing; the species does.
                hits = 0 if "subsp." in url or "subsp" in url else 2
                return ESEARCH_XML.replace("<Count>2</Count>", f"<Count>{hits}</Count>").encode()
            return super().get(url, timeout=timeout)

    transport = MissThenHit()
    data = entrez_query(
        organelle="plastid",
        taxon="Zea mays subsp. mays",
        dest=tmp_path,
        transport=transport,
    )
    assert manifest_of(data)["n_records"] == 2
    assert manifest_of(data)["scope"]["taxon"] == "Zea mays"
    assert len(manifest_of(data)["names_tried"]) >= 2


def test_name_variants_are_not_tried_when_disabled(tmp_path: Path) -> None:
    transport = FakeTransport(hits=0)
    entrez_query(
        organelle="plastid",
        taxon="Zea mays subsp. mays",
        dest=tmp_path,
        transport=transport,
        try_name_variants=False,
    )
    assert len(transport.urls_matching("esearch")) == 1


# ─────────────────────────────────────────────────────────────
# GenBank metadata tier
# ─────────────────────────────────────────────────────────────


def test_a_genbank_tier_filter_pulls_genbank_metadata_automatically(
    tmp_path: Path,
) -> None:
    """Asking for min_gene_count implies the richer fetch — no silent no-op."""
    transport = FakeTransport()
    data = entrez_query(
        organelle="plastid",
        dest=tmp_path,
        transport=transport,
        filters=FetchFilters(min_gene_count=2),
        page_size=10,
    )
    assert transport.urls_matching("retmode=xml")  # GBSeq was fetched
    # Arabidopsis has 2 genes, Oryza has 1 -> only Arabidopsis survives.
    assert list(manifest_of(data)["accessions"]) == ["NC_000932.1"]
    assert manifest_of(data)["scope"]["genbank_metadata"] is True


def test_genbank_metadata_reaches_the_records(tmp_path: Path) -> None:
    data = entrez_query(
        organelle="plastid",
        dest=tmp_path,
        transport=FakeTransport(),
        genbank_metadata=True,
        page_size=10,
    )
    arabidopsis = next(r for r in data.payload["records"] if r["accession"] == "NC_000932.1")
    assert arabidopsis["gene_count"] == 2
    assert arabidopsis["protein_count"] == 1
    assert arabidopsis["ambiguous_count"] == 2  # the two n's
    assert arabidopsis["collection_date"] == "2021/05"
    assert arabidopsis["geo_location"] == "China: Yunnan"


# ─────────────────────────────────────────────────────────────
# baseline / resume
# ─────────────────────────────────────────────────────────────


def test_baseline_skips_records_a_previous_run_already_has(tmp_path: Path) -> None:
    baseline = tmp_path / "have.txt"
    baseline.write_text("NC_000932.1\n")

    transport = FakeTransport()
    data = entrez_query(organelle="plastid", dest=tmp_path, transport=transport, baseline=baseline)
    assert transport.posted == ["OP474144.1"]  # the one we lacked
    assert manifest_of(data)["baseline_skipped"] == 1


# ─────────────────────────────────────────────────────────────
# explicit accession list
# ─────────────────────────────────────────────────────────────


def test_fetch_accessions_posts_the_ids_and_returns_a_set_hash(tmp_path: Path) -> None:
    transport = FakeTransport()
    data = fetch_accessions(
        accessions=["OP474144.1", "NC_000932.1"], dest=tmp_path, transport=transport
    )
    m = manifest_of(data)
    assert m["source"] == "accession_list"
    assert list(m["accessions"]) == ["NC_000932.1", "OP474144.1"]
    assert len(m["accession_set_sha256"]) == 64
    assert "NC_000932.1" in transport.posted


# ─────────────────────────────────────────────────────────────
# refseq_snapshot — the deterministic channel
# ─────────────────────────────────────────────────────────────


def test_refseq_snapshot_pins_the_release_number(tmp_path: Path) -> None:
    data = refseq_snapshot(organelle="plastid", dest=tmp_path, transport=FakeTransport())
    m = manifest_of(data)
    assert m["source"] == "refseq_release"
    assert m["refseq_release"] == 236


def test_refseq_snapshot_rejects_an_explicit_noncurrent_release(tmp_path: Path) -> None:
    """The live release directory must never be relabelled as an old release."""
    transport = FakeTransport()
    with pytest.raises(OrganelleParameterError) as raised:
        refseq_snapshot(organelle="plastid", dest=tmp_path, release=235, transport=transport)
    assert raised.value.code == "input.refseq_release_unavailable"
    assert raised.value.details == {"requested": 235, "current": 236}
    assert transport.urls_matching("RELEASE_NUMBER")
    assert transport.urls_matching("/plastid/") == []


def test_refseq_snapshot_rejects_an_unreadable_release_number(tmp_path: Path) -> None:
    transport = FakeTransport(release=b"<html>oops</html>")
    with pytest.raises(OrganelleExecutionError):
        refseq_snapshot(organelle="plastid", dest=tmp_path, transport=transport)


# ─────────────────────────────────────────────────────────────
# Honest accounting — a silent cap reads as "this is everything"
# ─────────────────────────────────────────────────────────────


def test_a_capped_query_reports_truncation_not_a_giant_filter_rejection(
    tmp_path: Path,
) -> None:
    """Regression: the manifest once claimed a filter rejected 143 records when
    only 6 had ever been examined. Three numbers, never conflated:
    hits_before_filter (matched) / examined (looked at) / n_records (kept).
    """
    transport = FakeTransport(hits=100)
    data = entrez_query(organelle="plastid", dest=tmp_path, transport=transport, max_records=2)
    m = manifest_of(data)

    assert m["hits_before_filter"] == 100  # what the database matched
    assert m["examined"] == 2  # what we actually pulled metadata for
    assert m["n_records"] == 2  # what survived
    assert m["n_filtered_out"] == 0  # the filter rejected nothing
    assert m["truncated"] is True  # and say so out loud


def test_an_uncapped_query_is_not_truncated(tmp_path: Path) -> None:
    data = entrez_query(
        organelle="plastid", dest=tmp_path, transport=FakeTransport(hits=2), max_records=None
    )
    m = manifest_of(data)
    assert m["truncated"] is False
    assert m["examined"] == m["hits_before_filter"] == 2
