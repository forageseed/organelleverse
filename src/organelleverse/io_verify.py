"""Read a file once and return an artifact whose ``validated`` flag is earned.

``ArtifactRef.from_path`` hashes a path and sets ``validated=True`` after
existence and hashing alone, and no released reader verified its declared format
before this module existed. Reopening the path to validate would let
``validated=True`` describe content differing from the recorded digest, so
verification streams the file exactly once: the same single pass hashes the
bytes on disk and parses the (decompressed) content, and memory stays bounded
by the I/O buffer regardless of file size. Buffering whole-file snapshots here
used to cap usable inputs at a few GB of RAM and blocked full-depth WGS
FASTQ.gz libraries.
"""

from __future__ import annotations

import hashlib
import io
import re
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError

VerifiedFormat = Literal["fasta", "fastq", "genbank"]

#: Cap on the bytes hashed from disk. Sized so one full-depth PacBio HiFi
#: WGS library (tens of GB gzipped) verifies through the released reader.
DEFAULT_MAX_BYTES = 64 * 1024 * 1024 * 1024

#: The decompressed stream may exceed ``max_bytes`` (legitimate FASTQ.gz
#: expands ~3x), but never ``max_bytes + 4x the disk bytes``: that surplus
#: ratio only appears in compression bombs.
_DECOMP_RATIO = 4

_GZIP_MAGIC = b"\x1f\x8b"
_RAW_CHUNK = 1024 * 1024

_MEDIA_TYPES: dict[str, str] = {
    "fasta": "text/x-fasta",
    "fastq": "text/x-fastq",
    "genbank": "text/x-genbank",
}


class _SinglePassStream(io.RawIOBase):
    """One bounded-memory pass: hash disk bytes, yield decompressed bytes.

    gzip containers are inflated incrementally (multi-member aware); the
    recorded digest and size always describe the file on disk, so content
    addressing stays stable whether an input is compressed or not.
    """

    def __init__(self, path: Path, basename: str, max_bytes: int) -> None:
        super().__init__()
        self._handle = path.open("rb")
        self._basename = basename
        self._max_bytes = max_bytes
        self._hasher = hashlib.sha256()
        self._disk_bytes = 0
        self._decompressed_bytes = 0
        self._pending = b""
        self._pending_offset = 0
        self._decompressor: zlib.Decompress | None = None
        self._checked_magic = False
        self._input_eof = False

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:  # type: ignore[override]
        view = memoryview(buffer).cast("B")
        while self._pending_offset >= len(self._pending) and not self._input_eof:
            self._fill()
        if self._pending_offset >= len(self._pending):
            return 0
        data = self._pending[self._pending_offset :]
        count = min(len(view), len(data))
        view[:count] = data[:count]
        self._pending_offset += count
        return count

    def close(self) -> None:
        """Finish hashing before the file handle goes away.

        Whoever closes the stream (a verifier's ``with`` block, the garbage
        collector, or this module), any bytes a parser stopped short of are
        still hashed so the recorded digest always covers the whole file.
        """
        if not self.closed:
            try:
                while not self._input_eof:
                    self._fill()
            finally:
                self._handle.close()
                super().close()

    @property
    def sha256(self) -> str:
        return self._hasher.hexdigest()

    @property
    def disk_bytes(self) -> int:
        return self._disk_bytes

    def _fill(self) -> None:
        chunk = self._handle.read(_RAW_CHUNK)
        if not chunk:
            self._input_eof = True
            if self._decompressor is not None and not self._decompressor.eof:
                raise _bad_gzip(self._basename, "truncated gzip stream")
            return
        self._hasher.update(chunk)
        self._disk_bytes += len(chunk)
        if self._disk_bytes > self._max_bytes:
            raise _too_large(self._basename, self._max_bytes, decompressed=False)
        if not self._checked_magic:
            self._checked_magic = True
            if chunk[:2] == _GZIP_MAGIC:
                self._decompressor = zlib.decompressobj(31)
        if self._decompressor is None:
            self._pending = chunk
            self._pending_offset = 0
            self._decompressed_bytes += len(chunk)
            return
        self._pending = self._inflate(chunk)
        self._pending_offset = 0
        self._decompressed_bytes += len(self._pending)
        limit = self._max_bytes + _DECOMP_RATIO * self._disk_bytes
        if self._decompressed_bytes > limit:
            raise _too_large(self._basename, limit, decompressed=True)

    def _inflate(self, chunk: bytes) -> bytes:
        pieces: list[bytes] = []
        while chunk:
            decompressor = self._decompressor
            assert decompressor is not None
            try:
                pieces.append(decompressor.decompress(chunk))
            except zlib.error as error:
                raise _bad_gzip(self._basename, str(error)) from error
            if not decompressor.eof:
                break
            rest = decompressor.unused_data
            if not rest or set(rest) == {0}:
                break  # end of stream, optionally NUL padding
            if rest[:2] == _GZIP_MAGIC:
                self._decompressor = zlib.decompressobj(31)
                chunk = rest
                continue
            raise _bad_gzip(self._basename, "trailing data after the gzip stream")
        return b"".join(pieces)


def read_verified_artifact(
    path: Path,
    *,
    kind: str,
    format: VerifiedFormat,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> ArtifactRef:
    """Hash and verify one file from a single streaming pass.

    gzip-compressed inputs are decompressed transparently for verification;
    the recorded digest and size always describe the file on disk (the
    compressed bytes), so content addressing stays stable whether an input is
    compressed or not.
    """
    candidate = Path(path)
    if not candidate.is_file():
        raise OrganelleInputError(
            code="input.missing_artifact",
            message=f"Artifact does not exist: {candidate.name}",
            details={"basename": candidate.name},
        )
    stream = _SinglePassStream(candidate, candidate.name, max_bytes)
    try:
        _VERIFIERS[format](stream, candidate.name)
    finally:
        stream.close()
    return ArtifactRef(
        kind=kind,
        uri=str(candidate),
        format=format,
        media_type=_MEDIA_TYPES[format],
        sha256=stream.sha256,
        size_bytes=stream.disk_bytes,
        validated=True,
    )


class _LineAssembler:
    """Feed text chunks in, get complete lines out."""

    __slots__ = ("_tail",)

    def __init__(self) -> None:
        self._tail = ""

    def push(self, text: str) -> list[str]:
        self._tail += text
        parts = self._tail.split("\n")
        self._tail = parts.pop()
        return parts

    def flush(self) -> list[str]:
        parts = [self._tail] if self._tail else []
        self._tail = ""
        return parts


def _iter_lines(stream: _SinglePassStream, format: VerifiedFormat, basename: str) -> Iterator[str]:
    """Iterate decoded text lines, translating decode failures to typed errors."""
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")()
    reader = io.BufferedReader(stream, buffer_size=_RAW_CHUNK)
    assembler = _LineAssembler()
    try:
        while True:
            chunk = reader.read(_RAW_CHUNK)
            if not chunk:
                break
            try:
                text = decoder.decode(chunk)
            except UnicodeDecodeError as error:
                raise _utf8_error(format, basename) from error
            yield from assembler.push(text)
        try:
            tail = decoder.decode(b"", final=True)
        except UnicodeDecodeError as error:
            raise _utf8_error(format, basename) from error
        yield from assembler.push(tail)
        yield from assembler.flush()
    finally:
        if not reader.closed:
            reader.detach()  # keep `stream` open; the verifier owns its lifetime


def _utf8_error(format: VerifiedFormat, basename: str) -> OrganelleInputError:
    return OrganelleInputError(
        code=f"input.invalid_{format}",
        message=f"Artifact is not valid UTF-8 text: {basename}",
        details={"basename": basename},
    )


def _too_large(basename: str, max_bytes: int, *, decompressed: bool) -> OrganelleInputError:
    scope = "Decompressed artifact" if decompressed else "Artifact"
    return OrganelleInputError(
        code="input.artifact_too_large",
        message=f"{scope} exceeds the {max_bytes} byte limit: {basename}",
        details={"basename": basename, "max_bytes": max_bytes},
    )


def _bad_gzip(basename: str, reason: str) -> OrganelleInputError:
    return OrganelleInputError(
        code="input.invalid_gzip",
        message=f"Artifact is not a readable gzip stream: {basename}",
        details={"basename": basename, "reason": reason},
    )


def _invalid(format: VerifiedFormat, basename: str, reason: str) -> OrganelleInputError:
    """Typed rejection that never echoes parsed content or a full path."""
    return OrganelleInputError(
        code=f"input.invalid_{format}",
        message=f"Artifact is not valid {format}: {reason}",
        details={"basename": basename, "reason": reason},
    )


_DNA = frozenset("ACGTURYKMSWBDHVN-*.acgturykmswbdhvn")
_NON_DNA = re.compile(r"[^ACGTURYKMSWBDHVN*.acgturykmswbdhvn-]")


def _verify_fasta(stream: _SinglePassStream, basename: str) -> None:
    ids: set[str] = set()
    started = False
    in_record = False
    record_has_sequence = False
    for raw in _iter_lines(stream, "fasta", basename):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if in_record and not record_has_sequence:
                raise _invalid("fasta", basename, "a record has an empty sequence")
            header = line[1:].split()
            if not header:
                raise _invalid("fasta", basename, "a record has an empty identifier")
            if header[0] in ids:
                raise _invalid("fasta", basename, "duplicate record identifier")
            ids.add(header[0])
            started = True
            in_record = True
            record_has_sequence = False
            continue
        if not started:
            raise _invalid("fasta", basename, "no record starts the file")
        if _NON_DNA.search(line):
            raise _invalid("fasta", basename, "a record contains non-nucleotide residues")
        record_has_sequence = True
    if not started:
        raise _invalid("fasta", basename, "no record starts the file")
    if in_record and not record_has_sequence:
        raise _invalid("fasta", basename, "a record has an empty sequence")


def _verify_fastq(stream: _SinglePassStream, basename: str) -> None:
    records = 0
    quartet: list[str] = []
    for raw in _iter_lines(stream, "fastq", basename):
        line = raw.strip()
        if not line:
            continue
        quartet.append(line)
        if len(quartet) < 4:
            continue
        head, seq, plus, qual = quartet
        if not head.startswith("@"):
            raise _invalid("fastq", basename, "a record header does not start with @")
        if not plus.startswith("+"):
            raise _invalid("fastq", basename, "a record separator does not start with +")
        if not seq:
            raise _invalid("fastq", basename, "a record has an empty sequence")
        if len(seq) != len(qual):
            raise _invalid("fastq", basename, "sequence and quality lengths differ")
        quartet = []
        records += 1
    if not records:
        raise _invalid("fastq", basename, "no records found")
    if quartet:
        raise _invalid("fastq", basename, "record count is not a multiple of four lines")


def _verify_genbank(stream: _SinglePassStream, basename: str) -> None:
    from Bio import SeqIO

    handle = _IteratorFile(_iter_lines(stream, "genbank", basename))
    try:
        records = SeqIO.parse(handle, "genbank")  # pyright: ignore[reportUnknownMemberType]
        ids: list[str] = []
        has_sequence = False
        for record in records:
            ids.append(record.id)
            if record.seq is not None and len(record.seq):
                has_sequence = True
    except OrganelleInputError:
        raise
    except Exception as error:
        raise _invalid("genbank", basename, "the file does not parse as GenBank") from error
    if not ids:
        raise _invalid("genbank", basename, "no GenBank records found")
    if len(set(ids)) != len(ids):
        raise _invalid("genbank", basename, "duplicate record identifier")
    if not has_sequence:
        raise _invalid("genbank", basename, "no record carries sequence content")


class _IteratorFile:
    """Minimal read/readline adapter so SeqIO can consume a line iterator."""

    def __init__(self, lines: Iterator[str]) -> None:
        self._lines = iter(lines)

    def readline(self, limit: int = -1) -> str:
        try:
            return next(self._lines)
        except StopIteration:
            return ""

    def read(self, size: int = -1) -> str:
        if size is not None and size >= 0:
            return "".join(next(self._lines, "") for _ in range(size))
        return "".join(self._lines)


_VERIFIERS: dict[str, Callable[[_SinglePassStream, str], None]] = {
    "fasta": _verify_fasta,
    "fastq": _verify_fastq,
    "genbank": _verify_genbank,
}
