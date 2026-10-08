"""Transparent gzip verification in the unified artifact reader."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.io_verify import read_verified_artifact


def _write_gz(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb") as compressed:
            compressed.write(payload)


def test_gzip_fastq_verifies_and_hashes_compressed_bytes(tmp_path: Path) -> None:
    payload = b"@r1\nACGT\n+\nIIII\n@r2\nTTTT\n+\nJJJJ\n"
    path = tmp_path / "reads.fastq.gz"
    _write_gz(path, payload)
    artifact = read_verified_artifact(path, kind="long_read", format="fastq")
    assert artifact.validated is True
    assert artifact.format == "fastq"
    # Content addressing describes the file on disk (the compressed bytes),
    # so the digest is stable whether an input is compressed or not.
    assert artifact.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert artifact.size_bytes == path.stat().st_size


def test_gzip_fasta_verifies(tmp_path: Path) -> None:
    path = tmp_path / "genome.fasta.gz"
    _write_gz(path, b">chr1\nACGTACGT\n")
    artifact = read_verified_artifact(path, kind="sequence", format="fasta")
    assert artifact.validated is True


def test_gzip_fastq_with_invalid_content_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "reads.fastq.gz"
    _write_gz(path, b"not a fastq at all\nsecond line\n")
    with pytest.raises(OrganelleInputError) as raised:
        read_verified_artifact(path, kind="long_read", format="fastq")
    assert raised.value.code == "input.invalid_fastq"


def test_truncated_gzip_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "reads.fastq.gz"
    _write_gz(path, b"@r1\nACGT\n+\nIIII\n")
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(OrganelleInputError):
        read_verified_artifact(path, kind="long_read", format="fastq")


def test_gzip_bomb_is_capped(tmp_path: Path) -> None:
    path = tmp_path / "reads.fastq.gz"
    _write_gz(path, b"@r\nA\n+\nI\n" * 100_000)
    with pytest.raises(OrganelleInputError) as raised:
        read_verified_artifact(path, kind="long_read", format="fastq", max_bytes=1000)
    assert raised.value.code == "input.artifact_too_large"


def test_read_long_reads_accepts_gzip(tmp_path: Path) -> None:
    from organelleverse.io_reads import read_long_reads

    path = tmp_path / "lib.fastq.gz"
    _write_gz(path, b"@r1\nACGTACGTACGTACGT\n+\n" + b"I" * 16 + b"\n")
    data = read_long_reads(path, technology="pacbio_hifi", quality_state="ccs")
    assert data.modality == "sequencing_reads"


def test_multimember_gzip_verifies(tmp_path: Path) -> None:
    """Concatenated gzip members (what `gzip` CLI produces for appended logs)."""
    payload = b"@r1\nACGT\n+\nIIII\n"
    path = tmp_path / "reads.fastq.gz"
    _write_gz(path, payload)
    with path.open("ab") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb") as second:
            second.write(b"@r2\nTTTT\n+\nJJJJ\n")
    artifact = read_verified_artifact(path, kind="long_read", format="fastq")
    assert artifact.validated is True
    assert artifact.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_trailing_garbage_after_gzip_is_rejected(tmp_path: Path) -> None:
    payload = b"@r1\nACGT\n+\nIIII\n"
    path = tmp_path / "reads.fastq.gz"
    _write_gz(path, payload)
    with path.open("ab") as raw:
        raw.write(b"not a gzip member")
    with pytest.raises(OrganelleInputError) as raised:
        read_verified_artifact(path, kind="long_read", format="fastq")
    assert raised.value.code == "input.invalid_gzip"


def test_corrupt_gzip_body_is_rejected(tmp_path: Path) -> None:
    payload = b"@r1\n" + b"A" * 4096 + b"\n+\n" + b"I" * 4096 + b"\n"
    path = tmp_path / "reads.fastq.gz"
    _write_gz(path, payload)
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF  # flip bits mid-stream, past the header
    path.write_bytes(bytes(raw))
    with pytest.raises(OrganelleInputError) as raised:
        read_verified_artifact(path, kind="long_read", format="fastq")
    assert raised.value.code == "input.invalid_gzip"


def test_non_utf8_input_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "reads.fastq"
    path.write_bytes(b"@r1\nACGT\xff\xfe\n+\nIIII\n")
    with pytest.raises(OrganelleInputError) as raised:
        read_verified_artifact(path, kind="long_read", format="fastq")
    assert raised.value.code == "input.invalid_fastq"


def test_disk_size_cap_fires_before_decompression(tmp_path: Path) -> None:
    path = tmp_path / "reads.fastq.gz"
    _write_gz(path, b"@r1\nACGT\n+\nIIII\n")
    with pytest.raises(OrganelleInputError) as raised:
        read_verified_artifact(path, kind="long_read", format="fastq", max_bytes=4)
    assert raised.value.code == "input.artifact_too_large"
