"""GIR: FTP family/version walk, no browsable index — see the design doc's
"Verified upstream facts" for why every lookup scans every family.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleExecutionError, OrganelleParameterError
from organelleverse.fetch.gir import fetch_gir


def _tar_gz_bytes(member_name: str, content: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


_PROTEIN_TAR = _tar_gz_bytes("Malus_domestica.pro.fa", b">MD001\nMEEQVGFGF\n")
_CDS_TAR = _tar_gz_bytes("Malus_domestica.cds.fa", b">MD001\nATGGAGGAG\n")
_GFF_TAR = _tar_gz_bytes(
    "Malus_domestica.gff3", b"##gff-version 3\nChr1\tsrc\tgene\t1\t10\t.\t+\t.\tID=g1\n"
)


class FakeFtpClient:
    """Two version-tagged directories for the same species, one family only."""

    def __init__(self) -> None:
        self.download_calls: list[str] = []
        self._files = {
            "(v1.0)Malus_domestica": {
                "Malus_domestica.pro.fa.tar.gz": b"stale-version-should-not-be-picked",
            },
            "(v1.2)Malus_domestica": {
                "Malus_domestica.pro.fa.tar.gz": _PROTEIN_TAR,
                "Malus_domestica.cds.fa.tar.gz": _CDS_TAR,
                "Malus_domestica.gff.tar.gz": _GFF_TAR,
                # no .genome.fa.tar.gz — genome is absent for this species.
            },
        }

    def list_families(self) -> list[str]:
        return ["Rosaceae", "Actinidiaceae"]

    def list_species_dirs(self, family: str) -> list[str]:
        if family == "Rosaceae":
            return list(self._files)
        return ["(v1.0)Actinidia_chinensis"]

    def list_files(self, family: str, species_dir: str) -> list[str]:
        return list(self._files.get(species_dir, {}))

    def download(self, remote_path: str, dest: Path) -> None:
        self.download_calls.append(remote_path)
        _family_dir, species_dir, filename = remote_path.strip("/").split("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self._files[species_dir][filename])


def test_fetch_picks_the_highest_version_directory(tmp_path: Path) -> None:
    ftp = FakeFtpClient()
    data = fetch_gir(taxon="Malus domestica", dest=tmp_path, transport=None, ftp_client=ftp)
    (record,) = data.payload["records"]
    assert record["accession"] == "gir:Rosaceae/(v1.2)Malus_domestica"
    protein_path = Path(record["files"]["protein"])
    assert protein_path.read_bytes() == b">MD001\nMEEQVGFGF\n"


def test_fetch_requests_multiple_kinds_and_records_a_missing_one(tmp_path: Path) -> None:
    ftp = FakeFtpClient()
    data = fetch_gir(
        taxon="Malus domestica",
        dest=tmp_path,
        include=("protein", "cds", "gff3", "genome"),
        transport=None,
        ftp_client=ftp,
    )
    (record,) = data.payload["records"]
    assert set(record["files"]) == {"protein", "cds", "gff3"}
    assert list(record["missing"]) == [{"kind": "genome", "reason": "not_present_for_this_species"}]
    assert set(data.artifacts) == {"protein", "cds", "gff3"}


def test_a_species_not_found_under_any_family_is_a_plain_miss(tmp_path: Path) -> None:
    ftp = FakeFtpClient()
    data = fetch_gir(taxon="Nonexistent species", dest=tmp_path, transport=None, ftp_client=ftp)
    assert list(data.payload["records"]) == []
    assert list(data.payload["manifest"]["accessions"]) == []


def test_an_unresolvable_taxon_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(OrganelleParameterError) as raised:
        fetch_gir(taxon="not a binomial at all!!", dest=tmp_path, ftp_client=FakeFtpClient())
    assert raised.value.code == "input.unresolvable_taxon"


def test_an_unknown_include_kind_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(OrganelleParameterError) as raised:
        fetch_gir(
            taxon="Malus domestica", dest=tmp_path, include=("bogus",), ftp_client=FakeFtpClient()
        )
    assert raised.value.code == "input.unknown_include"


def test_a_kind_gir_does_not_offer_is_a_per_record_miss_not_an_error(tmp_path: Path) -> None:
    """``tpm``/``rna``/``seq-report`` are recognized cross-source values GIR
    just doesn't have — a miss, never ``input.unknown_include``."""
    data = fetch_gir(
        taxon="Malus domestica",
        dest=tmp_path,
        include=("protein", "rna"),
        ftp_client=FakeFtpClient(),
    )
    (record,) = data.payload["records"]
    assert "rna" not in record["files"]
    assert {"kind": "rna", "reason": "not_offered_by_this_source"} in record["missing"]


def test_an_ftp_login_failure_is_a_named_error_code(tmp_path: Path) -> None:
    class LoginFails:
        def list_families(self) -> list[str]:
            raise OrganelleExecutionError(
                code="network.gir.ftp_login_failed", message="bad password", retryable=False
            )

    with pytest.raises(OrganelleExecutionError) as raised:
        fetch_gir(taxon="Malus domestica", dest=tmp_path, ftp_client=LoginFails())
    assert raised.value.code == "network.gir.ftp_login_failed"


def test_a_corrupted_tar_is_rejected_by_content_integrity(tmp_path: Path) -> None:
    class Corrupting(FakeFtpClient):
        def download(self, remote_path: str, dest: Path) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"not actually a tar.gz file")

    with pytest.raises(OrganelleExecutionError) as raised:
        fetch_gir(taxon="Malus domestica", dest=tmp_path, ftp_client=Corrupting())
    assert raised.value.code == "network.gir.content_invalid"


def test_a_valid_tar_with_bad_member_content_leaves_no_file_at_final_path(tmp_path: Path) -> None:
    """A syntactically-valid .tar.gz whose member has the right filename but
    non-FASTA content (e.g. a mispackaged HTML error page) must be rejected
    by ``validate_downloaded_content`` *before* anything lands at the real
    final path — never write-then-validate-in-place."""
    bad_protein_tar = _tar_gz_bytes("Malus_domestica.pro.fa", b"not fasta content")

    class BadContent(FakeFtpClient):
        def __init__(self) -> None:
            super().__init__()
            self._files["(v1.2)Malus_domestica"]["Malus_domestica.pro.fa.tar.gz"] = bad_protein_tar

    with pytest.raises(OrganelleExecutionError) as raised:
        fetch_gir(taxon="Malus domestica", dest=tmp_path, ftp_client=BadContent())
    assert raised.value.code == "network.gir.content_invalid"

    final_path = tmp_path / "protein.fa"
    assert not final_path.exists()
    assert list(tmp_path.iterdir()) == []
