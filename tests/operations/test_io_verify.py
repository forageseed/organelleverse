"""read_verified_artifact: validated=True must be earned, not asserted.

Before this module, io.read_long_reads accepted any readable file and declared it
FASTQ with validated=True, because _format_for guessed the format from the
filename suffix and no parser ever ran.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import IO, Any, cast

import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.io_verify import VerifiedFormat, read_verified_artifact

_FASTA = ">seq1\nACGTACGT\n"
_FASTQ = "@r1\nACGT\n+\n!!!!\n"
_GENBANK = "LOCUS       X   8 bp    DNA     circular\nFEATURES\nORIGIN\n        1 acgtacgt\n//\n"
_JUNK = "hello world\nnot a sequence\n"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("text", "format"),
    [(_FASTA, "fasta"), (_FASTQ, "fastq"), (_GENBANK, "genbank")],
)
def test_a_conforming_file_is_accepted(tmp_path: Path, text: str, format: str) -> None:
    artifact = read_verified_artifact(
        _write(tmp_path, "input", text),
        kind="sequence",
        format=cast("VerifiedFormat", format),
    )
    assert artifact.validated is True
    assert artifact.format == format


def test_the_digest_covers_exactly_the_parsed_snapshot(tmp_path: Path) -> None:
    """Validation and hashing must describe one snapshot. Hashed independently
    here rather than by re-reading through the helper."""
    path = _write(tmp_path, "input.fasta", _FASTA)
    artifact = read_verified_artifact(path, kind="sequence", format="fasta")
    assert artifact.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert artifact.size_bytes == len(path.read_bytes())


@pytest.mark.parametrize(
    ("text", "format", "code"),
    [
        (_JUNK, "fasta", "input.invalid_fasta"),
        (_JUNK, "fastq", "input.invalid_fastq"),
        (_JUNK, "genbank", "input.invalid_genbank"),
        (_FASTA, "fastq", "input.invalid_fastq"),
        (_FASTA, "genbank", "input.invalid_genbank"),
        (_GENBANK, "fasta", "input.invalid_fasta"),
        (_FASTQ, "fasta", "input.invalid_fasta"),
    ],
)
def test_a_nonconforming_file_is_rejected(
    tmp_path: Path, text: str, format: str, code: str
) -> None:
    with pytest.raises(OrganelleInputError) as excinfo:
        read_verified_artifact(
            _write(tmp_path, "input", text),
            kind="sequence",
            format=cast("VerifiedFormat", format),
        )
    assert excinfo.value.code == code


def test_errors_never_echo_content_or_a_full_path(tmp_path: Path) -> None:
    secret = "SUPERSECRETTOKEN"
    path = _write(tmp_path, "secret_input", f"{secret}\n")
    with pytest.raises(OrganelleInputError) as excinfo:
        read_verified_artifact(path, kind="sequence", format="fasta")
    rendered = str(excinfo.value) + str(excinfo.value.as_dict())
    assert secret not in rendered
    assert str(tmp_path) not in rendered
    assert "secret_input" in rendered


def test_an_oversized_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(OrganelleInputError) as excinfo:
        read_verified_artifact(
            _write(tmp_path, "input.fasta", _FASTA),
            kind="sequence",
            format="fasta",
            max_bytes=3,
        )
    assert excinfo.value.code == "input.artifact_too_large"


def test_gzip_streams_are_transparently_decoded(tmp_path: Path) -> None:
    """Contract since transparent gzip support: a real gzip container is
    decoded and its content verified; magic bytes followed by a malformed
    container are rejected as an unreadable gzip stream, not as bad FASTA."""
    import gzip

    import hashlib

    good = tmp_path / "good.fasta"
    with good.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb") as compressed:
            compressed.write(_FASTA.encode("utf-8"))
    artifact = read_verified_artifact(good, kind="sequence", format="fasta")
    assert artifact.validated is True
    assert artifact.sha256 == hashlib.sha256(good.read_bytes()).hexdigest()

    malformed = tmp_path / "malformed.fasta"
    malformed.write_bytes(b"\x1f\x8b" + _FASTA.encode("utf-8"))
    with pytest.raises(OrganelleInputError) as excinfo:
        read_verified_artifact(malformed, kind="sequence", format="fasta")
    assert excinfo.value.code == "input.invalid_gzip"


def test_a_non_regular_file_is_refused(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(OrganelleInputError) as excinfo:
        read_verified_artifact(fifo, kind="sequence", format="fasta")
    assert excinfo.value.code == "input.missing_artifact"


def test_genbank_verification_writes_nothing(tmp_path: Path) -> None:
    """The readers declare read_files only, so a scratch file would make that
    declaration false. GenBank is parsed from memory for this reason."""
    before = set(os.listdir(tempfile.gettempdir()))
    read_verified_artifact(
        _write(tmp_path, "input.gb", _GENBANK), kind="annotation", format="genbank"
    )
    assert set(os.listdir(tempfile.gettempdir())) == before


def test_the_file_is_opened_exactly_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the single-snapshot guarantee structurally.

    Asserting that the digest is merely *correct* does not pin this: an
    implementation that reopens the path to validate produces the same digest
    whenever the file has not changed in between, so
    test_the_digest_covers_exactly_the_parsed_snapshot passes either way.
    Counting opens is what distinguishes one read from two.
    """
    path = _write(tmp_path, "input.fasta", _FASTA)
    opens: list[str] = []
    real_open = Path.open
    real_read_bytes = Path.read_bytes

    def counting_open(self: Path, mode: str = "r") -> IO[Any]:
        if self == path:
            opens.append(str(self))
        return real_open(self, mode)

    def counting_read_bytes(self: Path) -> bytes:
        if self == path:
            opens.append(str(self))
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "open", counting_open)
    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    read_verified_artifact(path, kind="sequence", format="fasta")
    assert len(opens) == 1, f"the artifact file was read {len(opens)} times, expected exactly one"
