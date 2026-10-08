"""IMP: cleaned-whitelist hit vs. probed hit, and their differing confidence.

See docs/superpowers/specs/2026-07-19-nuclear-genome-alternate-sources-design.md
"IMP cache cleaning" — a whitelist hit is "heuristic_consistent", a fresh
probe hit is a strictly lower "unverified_probe".
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleExecutionError, OrganelleParameterError
from organelleverse.fetch._http import HttpResponse
from organelleverse.fetch.imp import fetch_imp
from organelleverse.fetch.imp_cache import CLEANED_CACHE_PATH


class FakeTransport:
    """Serves a whitelisted species (Lus1) and a probe-only species (Zzz9)."""

    def __init__(self) -> None:
        self.head_calls: list[str] = []

    def head(self, url: str, *, timeout: float = 30.0, headers=None) -> int:
        self.head_calls.append(url)
        if "/Zaa1/" in url:
            return 200
        return 404

    def get_response(self, url: str, **kwargs) -> HttpResponse:
        raise AssertionError("imp.py must stream via download_to_path, not get_response")

    def download_to_path(
        self, url: str, dest: Path, *, method="GET", data=None, timeout=120.0, headers=None
    ) -> int:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith(".prot.fasta"):
            dest.write_bytes(b">Lus1_g1\nMEEQVGFGF\n")
            return 200
        if url.endswith(".gff3.gz"):
            dest.write_bytes(
                gzip.compress(b"##gff-version 3\nChr1\tsrc\tgene\t1\t10\t.\t+\t.\tID=g1\n")
            )
            return 200
        if url.endswith(".fa.gz"):
            dest.write_bytes(gzip.compress(b">Lus1_chr1\nACGTACGT\n"))
            return 200
        return 404


def test_a_whitelisted_species_is_a_heuristic_consistent_hit(tmp_path: Path) -> None:
    transport = FakeTransport()
    data = fetch_imp(taxon="Linum usitatissimum", dest=tmp_path, transport=transport)
    (record,) = data.payload["records"]
    assert record["accession"] == "imp:Lus1"
    assert record["confidence"] == "heuristic_consistent"
    assert transport.head_calls == []  # cache hit: no probing needed


def test_an_unlisted_species_falls_back_to_probing_and_is_unverified(tmp_path: Path) -> None:
    transport = FakeTransport()
    data = fetch_imp(taxon="Zea aabbccnotarealspecies", dest=tmp_path, transport=transport)
    (record,) = data.payload["records"]
    assert record["confidence"] == "unverified_probe"
    assert any("/Zaa1/" in url for url in transport.head_calls)


def test_a_species_absent_from_both_cache_and_probe_is_a_plain_miss(tmp_path: Path) -> None:
    class NeverFound(FakeTransport):
        def head(self, url: str, *, timeout: float = 30.0, headers=None) -> int:
            return 404

    data = fetch_imp(taxon="Absent species", dest=tmp_path, transport=NeverFound())
    assert list(data.payload["records"]) == []


def test_tpm_is_never_fetched_unless_explicitly_requested(tmp_path: Path) -> None:
    transport = FakeTransport()
    data = fetch_imp(taxon="Linum usitatissimum", dest=tmp_path, transport=transport)
    (record,) = data.payload["records"]
    assert "tpm" not in record["files"]
    assert not any("TPM" in url for url in getattr(transport, "download_calls", []))


def test_a_missing_gff_for_a_present_species_is_a_per_kind_miss(tmp_path: Path) -> None:
    class NoGff(FakeTransport):
        def download_to_path(self, url: str, dest: Path, **kwargs) -> int:
            if url.endswith(".gff3.gz"):
                return 404
            return super().download_to_path(url, dest, **kwargs)

    data = fetch_imp(
        taxon="Linum usitatissimum", dest=tmp_path, include=("protein", "gff3"), transport=NoGff()
    )
    (record,) = data.payload["records"]
    assert "gff3" not in record["files"]
    assert {"kind": "gff3", "reason": "not_found_for_this_id"} in record["missing"]


def test_an_unknown_include_kind_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(OrganelleParameterError) as raised:
        fetch_imp(
            taxon="Linum usitatissimum",
            dest=tmp_path,
            include=("bogus",),
            transport=FakeTransport(),
        )
    assert raised.value.code == "input.unknown_include"


def test_a_kind_imp_does_not_offer_is_a_per_record_miss_not_an_error(tmp_path: Path) -> None:
    data = fetch_imp(
        taxon="Linum usitatissimum",
        dest=tmp_path,
        include=("protein", "rna"),
        transport=FakeTransport(),
    )
    (record,) = data.payload["records"]
    assert "rna" not in record["files"]
    assert {"kind": "rna", "reason": "not_offered_by_this_source"} in record["missing"]


def test_a_content_invalid_download_is_rejected_and_leaves_no_file_behind(tmp_path: Path) -> None:
    class Waf(FakeTransport):
        def download_to_path(self, url: str, dest: Path, **kwargs) -> int:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(gzip.compress(b"<html>blocked</html>"))
            return 200

    with pytest.raises(OrganelleExecutionError) as raised:
        fetch_imp(taxon="Linum usitatissimum", dest=tmp_path, include=("genome",), transport=Waf())
    assert raised.value.code == "network.imp.content_invalid"
    # The bad content must never reach the real output path — only a
    # same-filesystem temp copy is validated, and only a passing validation
    # gets renamed into place.
    assert not (tmp_path / "genome.gz").exists()


def test_tpm_content_is_not_fasta_shaped_and_is_not_rejected(tmp_path: Path) -> None:
    """TPM is a TSV table, not FASTA/GFF — the content-integrity check must
    skip it rather than reject every genuine TPM file for lacking a ">"."""

    class WithTpm(FakeTransport):
        def download_to_path(self, url: str, dest: Path, **kwargs) -> int:
            if url.endswith(".all.rnaseq.TPM.txt"):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"gene_id\tsample1\tsample2\nLus1_g1\t12.3\t9.8\n")
                return 200
            return super().download_to_path(url, dest, **kwargs)

    data = fetch_imp(
        taxon="Linum usitatissimum", dest=tmp_path, include=("tpm",), transport=WithTpm()
    )
    (record,) = data.payload["records"]
    assert Path(record["files"]["tpm"]).read_bytes().startswith(b"gene_id\t")


def test_the_cleaned_cache_fixture_actually_backs_this_test_species() -> None:
    import json

    with CLEANED_CACHE_PATH.open() as handle:
        cache = json.load(handle)
    assert cache["Linum_usitatissimum"]["imp_id"] == "Lus1"
