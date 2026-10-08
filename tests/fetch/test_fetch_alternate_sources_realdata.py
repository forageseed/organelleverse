"""Real-network checks against the four live alternate nuclear sources.

Skipped by default — set ORGANELLEVERSE_REALDATA_NETWORK=1 to run. Uses
Arabidopsis thaliana throughout: guaranteed present on TAIR (it serves
nothing else); GIR/IMP/PGD's real coverage of it was not verified while
writing the plan, so a miss on those three is tolerated and reported, not
asserted against — see this file's module docstring reasoning in the plan.
"""

from __future__ import annotations

import gzip
import os

import pytest

from organelleverse.fetch.gir import fetch_gir
from organelleverse.fetch.imp import fetch_imp
from organelleverse.fetch.pgd import fetch_pgd
from organelleverse.fetch.tair import fetch_tair

_NETWORK_OPT_IN = os.environ.get("ORGANELLEVERSE_REALDATA_NETWORK") == "1"
realdata_network = pytest.mark.skipif(
    not _NETWORK_OPT_IN,
    reason="set ORGANELLEVERSE_REALDATA_NETWORK=1 to run real-network fetch tests",
)

_TAXON = "Arabidopsis thaliana"


@pytest.mark.realdata
@realdata_network
def test_tair_real_fetch_finds_arabidopsis_protein(tmp_path) -> None:
    data = fetch_tair(taxon=_TAXON, dest=tmp_path, include=("protein",))
    (record,) = data.payload["records"]
    assert record["accession"] == "tair:Araport11"
    protein_path = record["files"]["protein"]
    with open(protein_path, "rb") as handle:
        content = gzip.decompress(handle.read())
    assert content.lstrip().startswith(b">")


@pytest.mark.realdata
@realdata_network
def test_gir_real_fetch_does_not_raise(tmp_path) -> None:
    data = fetch_gir(taxon=_TAXON, dest=tmp_path, include=("protein",))
    records = data.payload["records"]
    if records:
        assert records[0]["files"]
    print(f"\n  GIR real fetch for {_TAXON}: {'hit' if records else 'miss'}")


@pytest.mark.realdata
@realdata_network
def test_imp_real_fetch_does_not_raise(tmp_path) -> None:
    data = fetch_imp(taxon=_TAXON, dest=tmp_path, include=("protein",))
    records = data.payload["records"]
    if records:
        assert records[0]["confidence"] in ("heuristic_consistent", "unverified_probe")
    print(f"\n  IMP real fetch for {_TAXON}: {'hit' if records else 'miss'}")


@pytest.mark.realdata
@realdata_network
def test_pgd_real_fetch_does_not_raise(tmp_path) -> None:
    data = fetch_pgd(taxon=_TAXON, dest=tmp_path, include=("protein",))
    records = data.payload["records"]
    if records:
        assert records[0]["accession"] == "pgd:Arabidopsis_thaliana"
    print(f"\n  PGD real fetch for {_TAXON}: {'hit' if records else 'miss'}")
