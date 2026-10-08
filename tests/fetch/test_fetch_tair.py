"""TAIR: Arabidopsis-only, current release discovered by directory listing.

Uses the real ``api/download-files/list``+``download`` endpoints found by
reading TAIR's own SPA bundle — not the design's original (wrong)
``index-auto.jsp`` guess. See this plan's Task 7 intro for the full story.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleExecutionError, OrganelleParameterError
from organelleverse.fetch._http import HttpResponse
from organelleverse.fetch.tair import fetch_tair

_GFF_LISTING = {
    "Araport11_GFF3_genes_transposons.20250813.gff.gz": {
        "type": "file",
        "path": "Genes/Araport11_genome_release/Araport11_GFF3_genes_transposons.20250813.gff.gz",
        "lastModified": "2025-08-13T00:00:00.000Z",
        "size": 1,
    },
    # A second, non-archived, top-level candidate with an OLDER lastModified
    # than the entry above. This is the "real choice" case: both entries are
    # top-level files matching the GFF3 pattern, so _discover_current_file's
    # sort()/candidates[-1] logic must actually compare lastModified values
    # rather than there being only one qualifying candidate to trivially pick.
    "Araport11_GFF3_genes_transposons.20250101.gff.gz": {
        "type": "file",
        "path": "Genes/Araport11_genome_release/Araport11_GFF3_genes_transposons.20250101.gff.gz",
        "lastModified": "2025-01-01T00:00:00.000Z",
        "size": 1,
    },
    "archived": {
        "Araport11_GFF3_genes_transposons.20241001.gff.gz": {
            "type": "file",
            "path": "Genes/Araport11_genome_release/archived/Araport11_GFF3_genes_transposons.20241001.gff.gz",
            "lastModified": "2024-10-01T00:00:00.000Z",
        }
    },
}
_CDS_LISTING = {
    "Araport11_cds_20220914.gz": {
        "type": "file",
        "path": "Genes/Araport11_genome_release/Araport11_blastsets/Araport11_cds_20220914.gz",
        "lastModified": "2024-03-08T21:43:57.000Z",
        "size": 1,
    },
}
_PEP_LISTING = {
    "Araport11_pep_20250411.gz": {
        "type": "file",
        "path": "Sequences/Araport11_blastsets/Araport11_pep_20250411.gz",
        "lastModified": "2025-04-11T00:00:00.000Z",
        "size": 1,
    },
    "Archived": {
        "Araport11_pep_20220914.gz": {
            "type": "file",
            "path": "Sequences/Araport11_blastsets/Archived/Araport11_pep_20220914.gz",
            "lastModified": "2022-09-14T00:00:00.000Z",
        }
    },
}


class FakeTransport:
    def __init__(self, *, empty_listings: bool = False) -> None:
        self.list_calls: list[str] = []
        self.download_calls: list[str] = []
        self._empty_listings = empty_listings

    def get_response(
        self, url: str, *, method: str = "GET", data=None, timeout: float = 120.0, headers=None
    ) -> HttpResponse:
        self.list_calls.append(url)
        if self._empty_listings:
            return HttpResponse(status=200, body=b"{}")
        if "dir=Genes/Araport11_genome_release/Araport11_blastsets" in url:
            return HttpResponse(status=200, body=json.dumps(_CDS_LISTING).encode())
        if "dir=Genes/Araport11_genome_release" in url:
            return HttpResponse(status=200, body=json.dumps(_GFF_LISTING).encode())
        if "dir=Sequences/Araport11_blastsets" in url:
            return HttpResponse(status=200, body=json.dumps(_PEP_LISTING).encode())
        return HttpResponse(status=404, body=b"{}")

    def download_to_path(
        self,
        url: str,
        dest: Path,
        *,
        method: str = "GET",
        data=None,
        timeout: float = 120.0,
        headers=None,
    ) -> int:
        self.download_calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "Araport11_GFF3_genes_transposons.20250813.gff.gz" in url:
            dest.write_bytes(
                gzip.compress(b"##gff-version 3\nChr1\tsrc\tgene\t1\t10\t.\t+\t.\tID=g1\n")
            )
            return 200
        if "Araport11_pep_20250411.gz" in url:
            dest.write_bytes(gzip.compress(b">AT1G01010.1\nMEEQVGFGF\n"))
            return 200
        if "Araport11_cds_20220914.gz" in url:
            dest.write_bytes(gzip.compress(b">AT1G01010.1\nATGGAGGAG\n"))
            return 200
        if "TAIR10_chr_all.fas.gz" in url:
            dest.write_bytes(gzip.compress(b">Chr1\nACGTACGT\n"))
            return 200
        dest.write_bytes(b"")
        return 404


def test_fetch_picks_the_newest_gff_ignoring_the_archived_subdirectory(tmp_path: Path) -> None:
    transport = FakeTransport()
    data = fetch_tair(
        taxon="Arabidopsis thaliana", dest=tmp_path, include=("gff3",), transport=transport
    )
    (record,) = data.payload["records"]
    assert record["accession"] == "tair:Araport11"
    content = gzip.decompress(Path(record["files"]["gff3"]).read_bytes())
    assert content.startswith(b"##gff-version 3")
    assert any("20250813" in url for url in transport.download_calls)
    # The archived-subdirectory entry is excluded structurally (nested one
    # level down, never even considered as a candidate).
    assert not any("20241001" in url for url in transport.download_calls)
    # The 20250101 entry is a genuine top-level, non-archived rival: it
    # matches the same filename pattern and is a real candidate, but loses
    # the lastModified comparison to 20250813. Asserting it's not downloaded
    # exercises the sort()/candidates[-1] "pick newest" logic itself, rather
    # than "the only candidate available."
    assert not any("20250101" in url for url in transport.download_calls)


def test_protein_is_discovered_under_its_own_directory_not_the_gff_one(tmp_path: Path) -> None:
    transport = FakeTransport()
    data = fetch_tair(
        taxon="Arabidopsis thaliana", dest=tmp_path, include=("protein",), transport=transport
    )
    (record,) = data.payload["records"]
    assert "protein" in record["files"]
    assert any("Araport11_pep_20250411.gz" in url for url in transport.download_calls)


def test_genome_uses_the_fixed_stable_path_with_no_listing_call(tmp_path: Path) -> None:
    transport = FakeTransport()
    data = fetch_tair(
        taxon="Arabidopsis thaliana", dest=tmp_path, include=("genome",), transport=transport
    )
    (record,) = data.payload["records"]
    assert "genome" in record["files"]
    assert any("TAIR10_chr_all.fas.gz" in url for url in transport.download_calls)
    assert transport.list_calls == []  # no directory discovery needed for genome


def test_a_non_arabidopsis_taxon_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(OrganelleParameterError) as raised:
        fetch_tair(taxon="Malus domestica", dest=tmp_path, transport=FakeTransport())
    assert raised.value.code == "input.non_arabidopsis_taxon"


def test_tpm_is_never_offered_and_is_a_per_record_miss(tmp_path: Path) -> None:
    transport = FakeTransport()
    data = fetch_tair(
        taxon="Arabidopsis thaliana", dest=tmp_path, include=("protein", "tpm"), transport=transport
    )
    (record,) = data.payload["records"]
    assert {"kind": "tpm", "reason": "not_offered_by_this_source"} in record["missing"]


def test_an_empty_listing_is_a_per_kind_miss_not_an_error(tmp_path: Path) -> None:
    transport = FakeTransport(empty_listings=True)
    data = fetch_tair(
        taxon="Arabidopsis thaliana", dest=tmp_path, include=("gff3",), transport=transport
    )
    (record,) = data.payload["records"]
    assert {"kind": "gff3", "reason": "not_found_in_current_release"} in record["missing"]


def test_a_waf_page_saved_as_a_gz_is_rejected_and_leaves_no_file_behind(tmp_path: Path) -> None:
    class Waf(FakeTransport):
        def download_to_path(self, url: str, dest: Path, **kwargs) -> int:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(gzip.compress(b"<html>app shell</html>"))
            return 200

    with pytest.raises(OrganelleExecutionError) as raised:
        fetch_tair(taxon="Arabidopsis thaliana", dest=tmp_path, include=("gff3",), transport=Waf())
    assert raised.value.code == "network.tair.content_invalid"
    assert not (tmp_path / "gff3.gz").exists()
