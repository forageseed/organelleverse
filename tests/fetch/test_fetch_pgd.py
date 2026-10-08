"""PGD: POST+ZIP against pgdatabaseAPI, with md5sum-sidecar verification."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleExecutionError, OrganelleParameterError
from organelleverse.fetch.pgd import fetch_pgd


def _pgd_zip(inner_filename: str, gz_content: bytes, *, correct_md5: bool = True) -> bytes:
    digest = hashlib.md5(gz_content).hexdigest()
    if not correct_md5:
        digest = "0" * 32
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(inner_filename, gz_content)
        archive.writestr("PGdownload.md5sum", f"{digest}  {inner_filename}\n")
    return buffer.getvalue()


def _synthetic_sequence(seed: str, n_lines: int, line_len: int = 60) -> str:
    """Deterministic, non-repeating base sequence.

    A short, highly repetitive fixture (a few dozen bytes) gzip-compresses
    down to almost nothing, which made the resulting fake PGD zip small
    enough that a previous fix shrank the module's real ``_MIN_ZIP_BYTES``
    safety threshold to fit — instead of fixing the fixture. Chaining
    ``sha256`` gives each line new bytes so the content behaves like real
    sequence data under gzip, keeping the fake zip realistically sized
    (well over 1000 bytes, like the genuine PGD API's responses) without
    reducing readability of what's being faked (still starts with ``>``).
    """
    bases = "ACGT"
    lines: list[str] = []
    state = seed.encode()
    for _ in range(n_lines):
        state = hashlib.sha256(state).digest()
        chars: list[str] = []
        chunk = state
        while len(chars) < line_len:
            chars.extend(bases[b % 4] for b in chunk)
            chunk = hashlib.sha256(chunk).digest()
        lines.append("".join(chars[:line_len]))
    return "\n".join(lines)


_PEP_FASTA = f">AT1G01010.1\n{_synthetic_sequence('pep', 90)}\n".encode()
_GENOME_FASTA = f">Chr1\n{_synthetic_sequence('genome', 90)}\n".encode()
_GFF_BODY = "\n".join(
    f"Chr1\tsrc\tgene\t{i * 100 + 1}\t{i * 100 + 91}\t.\t+\t.\tID=g{i};Name=gene{i}"
    for i in range(200)
)
_GFF_TEXT = f"##gff-version 3\n{_GFF_BODY}\n".encode()

_PEP_GZ = gzip.compress(_PEP_FASTA)
_GENOME_GZ = gzip.compress(_GENOME_FASTA)
_GFF_GZ = gzip.compress(_GFF_TEXT)


class FakeTransport:
    def __init__(self, *, known_species: str = "malus_domestica", bad_md5: bool = False) -> None:
        self.posted_files: list[str] = []
        self._known_species = known_species
        self._bad_md5 = bad_md5

    def download_to_path(
        self, url: str, dest: Path, *, method="GET", data=None, timeout=120.0, headers=None
    ) -> int:
        assert method == "POST"
        body = json.loads(data)
        filename = body["files"]
        self.posted_files.append(filename)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not filename.startswith(self._known_species + "."):
            dest.write_bytes(b"")
            return 404

        if filename.endswith(".pep.fa.gz"):
            zip_bytes = _pgd_zip(filename, _PEP_GZ, correct_md5=not self._bad_md5)
        elif filename.endswith(".genomic.fa.gz"):
            zip_bytes = _pgd_zip(filename, _GENOME_GZ, correct_md5=not self._bad_md5)
        elif filename.endswith(".genomic.gff.gz"):
            dest.write_bytes(b"")
            return 404  # this species has no genomic.gff.gz — fallback path
        elif filename.endswith(".longest.gff.gz"):
            zip_bytes = _pgd_zip(filename, _GFF_GZ, correct_md5=not self._bad_md5)
        else:
            dest.write_bytes(b"")
            return 404

        dest.write_bytes(zip_bytes)
        return 200


def test_fetch_downloads_protein_and_verifies_md5sum(tmp_path: Path) -> None:
    transport = FakeTransport()
    data = fetch_pgd(taxon="Malus domestica", dest=tmp_path, transport=transport)
    (record,) = data.payload["records"]
    assert record["accession"] == "pgd:Malus_domestica"
    assert gzip.decompress(Path(record["files"]["protein"]).read_bytes()) == _PEP_FASTA
    assert "malus_domestica.pep.fa.gz" in transport.posted_files


def test_gff_falls_back_to_longest_gff_when_genomic_gff_is_absent(tmp_path: Path) -> None:
    transport = FakeTransport()
    data = fetch_pgd(
        taxon="Malus domestica", dest=tmp_path, include=("protein", "gff3"), transport=transport
    )
    (record,) = data.payload["records"]
    assert "gff3" in record["files"]
    assert "malus_domestica.genomic.gff.gz" in transport.posted_files
    assert "malus_domestica.longest.gff.gz" in transport.posted_files


def test_a_species_pgd_has_never_heard_of_is_a_plain_miss(tmp_path: Path) -> None:
    transport = FakeTransport(known_species="malus_domestica")
    data = fetch_pgd(taxon="Nonexistent species", dest=tmp_path, transport=transport)
    assert list(data.payload["records"]) == []


def test_cds_is_never_offered_and_is_a_per_record_miss(tmp_path: Path) -> None:
    transport = FakeTransport()
    data = fetch_pgd(
        taxon="Malus domestica", dest=tmp_path, include=("protein", "cds"), transport=transport
    )
    (record,) = data.payload["records"]
    assert "cds" not in record["files"]
    assert {"kind": "cds", "reason": "not_offered_by_this_source"} in record["missing"]


def test_existence_is_probed_via_protein_even_when_only_cds_is_requested(tmp_path: Path) -> None:
    """A request for only an unsupported kind must not silently skip the
    existence probe and report a false species-level miss."""
    transport = FakeTransport()
    data = fetch_pgd(taxon="Malus domestica", dest=tmp_path, include=("cds",), transport=transport)
    (record,) = data.payload["records"]
    assert record["accession"] == "pgd:Malus_domestica"
    assert record["files"] == {}


def test_the_protein_existence_probe_leaves_no_file_behind_when_unrequested(tmp_path: Path) -> None:
    transport = FakeTransport()
    fetch_pgd(taxon="Malus domestica", dest=tmp_path, include=("genome",), transport=transport)
    assert not (tmp_path / "protein.gz").exists()
    assert (tmp_path / "genome.gz").exists()


def test_a_wrong_md5sum_is_rejected_as_content_invalid_and_leaves_no_file_behind(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(bad_md5=True)
    with pytest.raises(OrganelleExecutionError) as raised:
        fetch_pgd(taxon="Malus domestica", dest=tmp_path, transport=transport)
    assert raised.value.code == "network.pgd.content_invalid"
    assert not (tmp_path / "protein.gz").exists()


def test_an_unresolvable_taxon_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(OrganelleParameterError) as raised:
        fetch_pgd(taxon="not a binomial!!", dest=tmp_path, transport=FakeTransport())
    assert raised.value.code == "input.unresolvable_taxon"


def test_a_truly_unknown_include_value_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(OrganelleParameterError) as raised:
        fetch_pgd(
            taxon="Malus domestica", dest=tmp_path, include=("bogus",), transport=FakeTransport()
        )
    assert raised.value.code == "input.unknown_include"


def test_a_corrupted_zip_is_content_invalid(tmp_path: Path) -> None:
    class Corrupting(FakeTransport):
        def download_to_path(self, url: str, dest: Path, **kwargs) -> int:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"not a zip file, but long enough to pass the size check" * 30)
            return 200

    with pytest.raises(OrganelleExecutionError) as raised:
        fetch_pgd(taxon="Malus domestica", dest=tmp_path, transport=Corrupting())
    assert raised.value.code == "network.pgd.content_invalid"
