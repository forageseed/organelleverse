"""Gene records and nuclear assemblies. Offline.

Two channels that are NOT the organelle-genome channel:

* ``fetch_genes`` — the fragments ``entrez_query`` throws away are the target
  here (barcoding, phylogeny). Same machinery, filter inverted, no grouping:
  a matK record is not a molecule of a genome.
* ``fetch_nuclear_genome`` — nuclear genomes are assemblies (GCF_/GCA_) behind
  the Datasets API. No Entrez query returns them, and no Datasets query returns
  an organelle. ERC needs both sides, so both channels have to exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleExecutionError, OrganelleParameterError
from organelleverse.fetch.genes import build_gene_term
from organelleverse.fetch.nuclear import fetch_nuclear_genome, parse_assembly_reports

# ─────────────────────────────────────────────────────────────
# gene term construction
# ─────────────────────────────────────────────────────────────


def test_a_plastid_gene_search_uses_the_gene_field() -> None:
    term = build_gene_term(gene="matK", taxon="Oryza", scope="plastid")
    assert term == "Oryza[Organism] AND matK[Gene] AND chloroplast[filter]"


def test_a_mitochondrial_gene_search() -> None:
    term = build_gene_term(gene="cox1", taxon="Oryza", scope="mitochondrion")
    assert "mitochondrion[filter]" in term
    assert "cox1[Gene]" in term


def test_a_nuclear_gene_search_excludes_both_organelles() -> None:
    """There is no `nuclear[filter]`; excluding the organelles is what makes it
    nuclear."""
    term = build_gene_term(gene="PPR", taxon="Oryza", scope="nuclear")
    assert "NOT mitochondrion[filter]" in term
    assert "NOT chloroplast[filter]" in term


def test_an_empty_gene_symbol_is_rejected() -> None:
    with pytest.raises(OrganelleParameterError):
        build_gene_term(gene="  ", taxon="Oryza")


def test_an_unknown_scope_is_rejected() -> None:
    with pytest.raises(OrganelleParameterError):
        build_gene_term(gene="matK", scope="plastome")  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────
# Datasets assembly reports (real response shape, 2026-07-13)
# ─────────────────────────────────────────────────────────────

REPORT = json.dumps(
    {
        "total_count": 265,
        "reports": [
            {
                "accession": "GCF_000001735.4",
                "organism": {"organism_name": "Arabidopsis thaliana", "tax_id": 3702},
                "assembly_info": {
                    "assembly_name": "TAIR10.1",
                    "assembly_level": "Chromosome",
                    "refseq_category": "reference genome",
                    "submission_date": "2018-03-15",
                },
                "assembly_stats": {
                    "total_sequence_length": "119146348",
                    "contig_n50": 11194537,
                },
                "annotation_info": {"stats": {"gene_counts": {"total": 38312}}},
            }
        ],
    }
)


class FakeTransport:
    def __init__(self, body: bytes = REPORT.encode()) -> None:
        self.calls: list[str] = []
        self._body = body

    def get(self, url: str, *, timeout: float = 0.0) -> bytes:
        self.calls.append(url)
        if "download" in url:
            return b"PK\x03\x04fake-zip"
        return self._body

    def post(
        self, url: str, data: bytes, *, timeout: float = 0.0, headers=None
    ) -> bytes:
        self.calls.append(url)
        if "download" in url:
            return b"PK\x03\x04fake-zip"
        raise AssertionError("Datasets POST is download-only")


def test_assembly_report_is_flattened() -> None:
    (report,) = parse_assembly_reports(REPORT)
    assert report["accession"] == "GCF_000001735.4"
    assert report["length"] == 119_146_348
    assert report["gene_count"] == 38_312
    assert report["refseq_category"] == "reference genome"


def test_a_non_json_response_is_a_retryable_error() -> None:
    with pytest.raises(OrganelleExecutionError):
        parse_assembly_reports(b"<html>502 Bad Gateway</html>")


def test_nuclear_fetch_defaults_to_proteins_not_the_genome() -> None:
    """TAIR10 is 119 Mb of genome and ~35 MB of protein. OrthoFinder eats the
    proteins; downloading the genome to get them wastes two orders of magnitude.
    """
    transport = FakeTransport()
    data = fetch_nuclear_genome(
        taxon="Arabidopsis thaliana", dest="/tmp/x", transport=transport, download=False
    )
    # OrganelleData freezes payloads, so the list arrives as a tuple.
    assert list(data.payload["manifest"]["scope"]["include"]) == ["protein"]
    assert data.modality == "nuclear_assemblies"


def test_survey_mode_transfers_nothing(tmp_path: Path) -> None:
    transport = FakeTransport()
    data = fetch_nuclear_genome(
        taxon="Arabidopsis thaliana", dest=tmp_path, transport=transport, download=False
    )
    assert not any("download" in url for url in transport.calls)
    assert data.artifacts == {}
    assert data.payload["records"][0]["accession"] == "GCF_000001735.4"


def test_download_writes_a_content_addressed_archive(tmp_path: Path) -> None:
    transport = FakeTransport()
    data = fetch_nuclear_genome(
        taxon="Arabidopsis thaliana", dest=tmp_path, transport=transport, download=True
    )
    assert any("download" in url for url in transport.calls)
    archive = data.artifacts["assemblies"]
    assert archive.format == "zip"
    assert len(archive.sha256) == 64


def test_an_unknown_include_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(OrganelleParameterError):
        fetch_nuclear_genome(
            taxon="Arabidopsis thaliana",
            dest=tmp_path,
            include=("proteome",),
            transport=FakeTransport(),
        )
