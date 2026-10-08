"""NGDC/GWH — a supplementary organelle sample source. Offline.

The channel adds ~1,270 plastid and 5 mitochondrial *assemblies* under GWH
accessions absent from NCBI. It does NOT add new species: a 40-genome random
sample found 0 species NCBI lacked. ``ngdc_only`` is an accession-level claim,
never a species-level one. Everything here guards the two ways the channel can
go wrong: a folder named like an organelle that is not one (Ilex vomitoria), and
a nuclear genome slipping in as an organelle one.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleExecutionError, OrganelleParameterError
from organelleverse.fetch.ngdc import (
    classify_folder,
    fetch_ngdc,
    gwh_assembly_metadata,
    parse_plants_index,
)

# The GWH plants directory listing, in the shape it actually serves — every one
# of these folder names is real (2026-07-14).
PLANTS_INDEX = """<html><body>
<a href="../">../</a>
<a href="Panax_notoginseng_PN_mt_GWHBHOF01000000/">Panax_notoginseng_PN_mt_GWHBHOF01000000/</a>
<a href="Camellia_sinensis_var._assamica_mitochondria_genome_GWHAAIC00000000/">x/</a>
<a href="Helicteres_angustifolia_plant_cpDNA_seq_302_GWHABWV01000000/">x/</a>
<a href="Murraya_alata_chloroplast_GWHBHGJ01000000/">x/</a>
<a href="Ilex_vomitoria_24_GWHAOTG01000000/">x/</a>
<a href="Triticum_aestivum_Iso_Wt_W2093_GWHDGRH00000000/">x/</a>
</body></html>"""

# Which organism each real GWH accession belongs to (2026-07-14).
_ORGANISM_BY_ACC = {
    "GWHBHOF01000000": "Panax notoginseng",
    "GWHAAIC00000000": "Camellia sinensis",
    "GWHABWV01000000": "Helicteres angustifolia",
    "GWHBHGJ01000000": "Murraya alata",
}

GWH_META = """{
  "organism": "Helicteres angustifolia",
  "taxId": "190244",
  "assemblyName": "plant_cpDNA_seq_302",
  "assemblyLevel": "Chromosome",
  "bioprojectAccession": "PRJCA002188",
  "biosampleAccession": "SAMC134033",
  "submitterOrganization": "National Resource Center for Chinese Materia Medica",
  "releaseTime": "2024-02-03 05:28:58.0",
  "ftpPathDna": "https://download.cncb.ac.cn/gwh/Plants/x/GWHABWV01000000.genome.fasta.gz"
}"""


class FakeTransport:
    """Serves the index, the API, and small gzipped FASTA blobs."""

    def __init__(self, *, fasta: bytes | None = None) -> None:
        self.calls: list[str] = []
        self._fasta = fasta if fasta is not None else gzip.compress(b">seq\nACGT\n")

    def get(self, url: str, *, timeout: float = 0.0) -> bytes:
        self.calls.append(url)
        if url.endswith("/gwh/Plants/"):
            return PLANTS_INDEX.encode()
        if "/api/public/assembly/" in url:
            accession = url.rsplit("/", 1)[-1]
            organism = _ORGANISM_BY_ACC.get(accession, "Unknown species")
            meta = GWH_META.replace("Helicteres angustifolia", organism)
            return meta.encode()
        if url.endswith(".genome.fasta.gz"):
            return self._fasta
        if url.endswith(".gff.gz"):
            raise OrganelleExecutionError(
                code="network.transient", message="no gff", retryable=True
            )
        raise AssertionError(f"unexpected URL: {url}")

    def post(self, url: str, data: bytes, *, timeout: float = 0.0) -> bytes:
        raise AssertionError("NGDC is GET-only")


# ─────────────────────────────────────────────────────────────
# classification — token, not substring
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "folder,expected",
    [
        ("Panax_notoginseng_PN_mt_GWHBHOF01000000", "mitochondrion"),
        ("Camellia_sinensis_mitochondria_genome_GWHAAIC00000000", "mitochondrion"),
        ("Helicteres_angustifolia_plant_cpDNA_seq_GWHABWV01000000", "plastid"),
        ("Murraya_alata_chloroplast_GWHBHGJ01000000", "plastid"),
    ],
)
def test_organelle_folders_are_classified(folder: str, expected: str) -> None:
    assert classify_folder(folder) == expected


def test_ilex_vomitoria_is_not_a_mitochondrion() -> None:
    """vo-MITO-ria contains 'mito'. Substring matching would misfile a holly."""
    assert classify_folder("Ilex_vomitoria_24_GWHAOTG01000000") is None


def test_a_nuclear_genome_is_not_an_organelle() -> None:
    assert classify_folder("Triticum_aestivum_Iso_Wt_W2093_GWHDGRH00000000") is None


# ─────────────────────────────────────────────────────────────
# index parsing
# ─────────────────────────────────────────────────────────────


def test_the_index_yields_only_organelle_genomes() -> None:
    records = parse_plants_index(PLANTS_INDEX)
    organelles = {r["organelle"] for r in records}
    assert organelles == {"mitochondrion", "plastid"}
    accessions = {r["accession"] for r in records}
    assert "GWHAOTG01000000" not in accessions  # the holly
    assert "GWHDGRH00000000" not in accessions  # wheat


def test_every_indexed_record_has_a_gwh_accession() -> None:
    for record in parse_plants_index(PLANTS_INDEX):
        assert record["accession"].startswith("GWH")
        assert record["source"] == "ngdc_gwh"


# ─────────────────────────────────────────────────────────────
# metadata
# ─────────────────────────────────────────────────────────────


def test_metadata_carries_the_assembly_level_ncbi_lacks() -> None:
    meta = gwh_assembly_metadata(FakeTransport(), "GWHABWV01000000")
    assert meta["organism"] == "Helicteres angustifolia"
    assert meta["assembly_level"] == "Chromosome"
    assert "Chinese Materia Medica" in meta["submitter_organization"]


def test_a_non_json_metadata_response_is_a_retryable_error() -> None:
    class Bad(FakeTransport):
        def get(self, url: str, *, timeout: float = 0.0) -> bytes:
            return b"<html>502</html>"

    with pytest.raises(OrganelleExecutionError):
        gwh_assembly_metadata(Bad(), "GWHABWV01000000")


# ─────────────────────────────────────────────────────────────
# fetch — the whole channel
# ─────────────────────────────────────────────────────────────


def test_fetch_returns_only_ngdc_only_plastid_genomes(tmp_path: Path) -> None:
    transport = FakeTransport()
    data = fetch_ngdc(organelle="plastid", dest=tmp_path, transport=transport)

    manifest = data.payload["manifest"]
    assert manifest["source"] == "ngdc_gwh"
    # ngdc_only is an ACCESSION-level claim: the GWH accession is absent from
    # NCBI. It is NOT a species-level claim — the species usually is in NCBI.
    assert manifest["scope"]["ngdc_only"] is True
    assert all(a.startswith("GWH") for a in manifest["accessions"])
    assert all(r["organelle"] == "plastid" for r in data.payload["records"])


def test_fetch_writes_the_fasta_and_tolerates_a_missing_gff(tmp_path: Path) -> None:
    """A genome without a GFF is normal; the FASTA is what must arrive."""
    transport = FakeTransport()
    data = fetch_ngdc(
        organelle="plastid",
        dest=tmp_path,
        taxon="Helicteres",
        formats=("fasta", "gff"),
        transport=transport,
    )
    (record,) = data.payload["records"]
    assert Path(record["files"]["fasta"]).is_file()
    assert "gff" not in record["files"]  # the fake refuses the gff
    assert data.artifacts  # the fasta is an artifact


def test_taxon_filters_on_the_authoritative_organism(tmp_path: Path) -> None:
    data = fetch_ngdc(
        organelle="plastid", dest=tmp_path, taxon="Helicteres", transport=FakeTransport()
    )
    assert len(data.payload["records"]) == 1
    assert data.payload["records"][0]["organism"] == "Helicteres angustifolia"


def test_an_oversize_genome_is_rejected_not_silently_kept(tmp_path: Path) -> None:
    """A name-based classifier will eventually be wrong. A nuclear-sized blob
    named like an organelle must be rejected out loud, never kept."""
    import os

    huge = gzip.compress(os.urandom(1000))  # incompressible: gz stays ~1 KB
    transport = FakeTransport(fasta=huge)
    data = fetch_ngdc(
        organelle="plastid",
        dest=tmp_path,
        taxon="Helicteres",  # narrow to one, so the assertion is about size only
        transport=transport,
        max_gz_bytes=100,  # anything over 100 bytes is "too big" for this test
    )
    assert list(data.payload["records"]) == []
    assert data.payload["manifest"]["rejected_by_size"]
    reason = data.payload["manifest"]["rejected_by_size"][0]
    assert reason["reason"] == "too large for an organelle genome"


def test_the_mitochondrial_side_is_small_but_present(tmp_path: Path) -> None:
    """NGDC has ~5 plant mitogenomes. Panax notoginseng (sanqi) is one."""
    data = fetch_ngdc(organelle="mitochondrion", dest=tmp_path, transport=FakeTransport())
    organisms = {r["organism"] for r in data.payload["records"]}
    assert any("Panax" in o for o in organisms)


def test_unknown_format_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(OrganelleParameterError):
        fetch_ngdc(organelle="plastid", dest=tmp_path, formats=("bam",), transport=FakeTransport())


def test_unknown_organelle_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(OrganelleParameterError):
        fetch_ngdc(organelle="nucleus", dest=tmp_path, transport=FakeTransport())  # type: ignore[arg-type]
