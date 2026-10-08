"""Streaming long-read statistics with a content-addressed cache.

The parser validates record structure while streaming FASTA, FASTQ, and gzip
inputs and never loads a full read set into memory. Statistics are keyed by the
complete source artifact SHA256, the parser version, and the statistic schema
version. A cache hit reopens and rehashes the source artifact before use but
never reparses its records.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import re
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import IO, Any, Literal

from pydantic import ConfigDict, Field, model_validator

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleExecutionError, OrganelleInputError
from organelleverse.operations.spec import StrictSpecModel

SCHEMA_VERSION: Literal["organelleverse.read-statistics.v1"] = "organelleverse.read-statistics.v1"
PARSER_VERSION: Literal["1"] = "1"

_GZIP_MAGIC = b"\x1f\x8b"
_CHUNK_SIZE = 64 * 1024
_WHITESPACE = re.compile(r"\s")


def _input_error(code: str, message: str, **details: object) -> OrganelleInputError:
    return OrganelleInputError(code=code, message=message, details=details)


class ReadStatistics(StrictSpecModel):
    """Streaming statistics for one long-read artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["organelleverse.read-statistics.v1"] = SCHEMA_VERSION
    parser_version: Literal["1"] = PARSER_VERSION
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(ge=1)
    total_bases: int = Field(ge=0)
    minimum_length: int = Field(ge=0)
    maximum_length: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_length_bounds(self) -> ReadStatistics:
        if self.maximum_length < self.minimum_length:
            raise ValueError("maximum_length must be greater than or equal to minimum_length")
        if self.record_count and self.total_bases < self.minimum_length:
            raise ValueError("total_bases must be at least minimum_length")
        return self


def cache_key(artifact_sha256: str) -> str:
    """Build the content-addressed cache key for one artifact's statistics."""
    return f"{SCHEMA_VERSION}|{PARSER_VERSION}|{artifact_sha256}"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_gzip(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == _GZIP_MAGIC


def open_record_text(path: Path) -> IO[str]:
    """Open plain or gzip sequence records, detected by the gzip magic bytes."""
    if _is_gzip(path):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _parse_fasta(lines: Iterable[str]) -> tuple[int, int, int, int]:
    record_count = 0
    total_bases = 0
    minimum_length: int | None = None
    maximum_length = 0
    current_length: int | None = None
    for line in lines:
        record = line.rstrip("\n").rstrip("\r")
        if record.startswith(">"):
            if not record[1:].strip():
                raise _input_error(
                    "assembly.malformed_long_read_record",
                    "FASTA record header has no name",
                )
            if current_length is not None:
                if current_length == 0:
                    raise _input_error(
                        "assembly.malformed_long_read_record",
                        "FASTA record has an empty sequence",
                    )
                total_bases += current_length
                minimum_length = (
                    current_length
                    if minimum_length is None
                    else min(minimum_length, current_length)
                )
                maximum_length = max(maximum_length, current_length)
            record_count += 1
            current_length = 0
        else:
            stripped = record.strip()
            if not stripped:
                continue
            if current_length is None:
                raise _input_error(
                    "assembly.malformed_long_read_record",
                    "FASTA sequence appears before any header",
                )
            if _WHITESPACE.search(stripped):
                raise _input_error(
                    "assembly.malformed_long_read_record",
                    "FASTA sequence line contains internal whitespace",
                )
            current_length += len(stripped)
    if current_length is None:
        raise _input_error(
            "assembly.malformed_long_read_record",
            "FASTA input has no records",
        )
    if current_length == 0:
        raise _input_error(
            "assembly.malformed_long_read_record",
            "FASTA record has an empty sequence",
        )
    total_bases += current_length
    minimum_length = (
        current_length if minimum_length is None else min(minimum_length, current_length)
    )
    maximum_length = max(maximum_length, current_length)
    assert minimum_length is not None
    return record_count, total_bases, minimum_length, maximum_length


def iter_fastq_records(lines: Iterable[str]) -> Iterator[tuple[str, str, str]]:
    """Yield validated four-line FASTQ records; shared with read QC.

    Empty streams are allowed so genuinely empty filter outputs can be measured.
    Assembly retains its nonempty-input boundary in ``_parse_fastq``.
    """
    header = sequence = ""
    state = 0  # 0 header, 1 sequence, 2 plus, 3 quality
    seq_len = 0
    for line in lines:
        record = line.rstrip("\n").rstrip("\r")
        if state == 0:
            if not record.startswith("@"):
                raise _input_error(
                    "assembly.malformed_long_read_record",
                    "FASTQ record is missing its @header line",
                )
            if not record[1:].strip():
                raise _input_error(
                    "assembly.malformed_long_read_record",
                    "FASTQ record header has no name",
                )
            header = record[1:]
            seq_len = 0
            state = 1
        elif state == 1:
            if not record:
                raise _input_error(
                    "assembly.malformed_long_read_record",
                    "FASTQ record has an empty sequence",
                )
            if _WHITESPACE.search(record):
                raise _input_error(
                    "assembly.malformed_long_read_record",
                    "FASTQ sequence line contains internal whitespace",
                )
            sequence = record
            seq_len = len(record)
            state = 2
        elif state == 2:
            if not record.startswith("+"):
                raise _input_error(
                    "assembly.malformed_long_read_record",
                    "FASTQ record is missing its +separator line",
                )
            state = 3
        else:
            if len(record) != seq_len:
                raise _input_error(
                    "assembly.malformed_long_read_record",
                    "FASTQ quality length does not match sequence length",
                )
            yield header, sequence, record
            state = 0
    if state != 0:
        raise _input_error(
            "assembly.malformed_long_read_record",
            "FASTQ input is truncated before a complete record",
        )


def _parse_fastq(lines: Iterable[str]) -> tuple[int, int, int, int]:
    record_count = total_bases = maximum_length = 0
    minimum_length: int | None = None
    for _header, sequence, _quality in iter_fastq_records(lines):
        length = len(sequence)
        record_count += 1
        total_bases += length
        minimum_length = length if minimum_length is None else min(minimum_length, length)
        maximum_length = max(maximum_length, length)
    if minimum_length is None:
        raise _input_error("assembly.malformed_long_read_record", "FASTQ input has no records")
    return record_count, total_bases, minimum_length, maximum_length


class _HashingReader(io.RawIOBase):
    """Hash exactly the disk bytes consumed by the parser, including gzip bytes."""

    def __init__(self, path: Path, update: Callable[[memoryview], object]) -> None:
        self.source = path.open("rb")
        self.update = update

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        count = self.source.readinto(buffer)
        self.update(memoryview(buffer)[:count])
        return count

    def close(self) -> None:
        self.source.close()
        super().close()


def parse_records(
    path: Path, fmt: str, *, hash_update: Callable[[memoryview], object] | None = None
) -> tuple[int, int, int, int]:
    raw = None
    if hash_update is None:
        handle = open_record_text(path)
    else:
        raw = io.BufferedReader(_HashingReader(path, hash_update))
        compressed = raw.peek(2)[:2] == _GZIP_MAGIC
        handle = io.TextIOWrapper(
            gzip.GzipFile(fileobj=raw, mode="rb") if compressed else raw,
            encoding="utf-8",
        )
    try:
        lines = iter(handle)
        if fmt == "fasta":
            return _parse_fasta(lines)
        if fmt == "fastq":
            return _parse_fastq(lines)
        raise _input_error(
            "assembly.unsupported_long_read_format",
            f"long-read statistics cannot parse format {fmt!r}",
            format=fmt,
        )
    finally:
        handle.close()
        if raw is not None:
            raw.close()  # GzipFile does not close a caller-owned file object.


def _verify_source(artifact: ArtifactRef) -> str:
    """Reopen and rehash the source artifact, returning its observed SHA256."""
    path = Path(artifact.uri)
    if not path.is_file():
        raise OrganelleExecutionError(
            code="assembly.output_incomplete",
            message=f"declared input artifact is missing: {artifact.uri}",
            details={"uri": artifact.uri},
        )
    observed = _hash_file(path)
    _check_digest(artifact, observed)
    return observed


def _check_digest(artifact: ArtifactRef, observed: str) -> None:
    if observed != artifact.sha256:
        raise OrganelleExecutionError(
            code="assembly.output_incomplete",
            message="declared input artifact hash no longer matches its content",
            details={
                "uri": artifact.uri,
                "declared_sha256": artifact.sha256,
                "actual_sha256": observed,
            },
        )


class ReadStatisticsCache:
    """Content-addressed, read-only cache of computed read statistics."""

    def __init__(self) -> None:
        self._entries: dict[str, ReadStatistics] = {}

    def lookup(self, artifact_sha256: str) -> ReadStatistics | None:
        """Return cached statistics for a content hash without rehashing."""
        return self._entries.get(cache_key(artifact_sha256))

    def get(self, artifact: ArtifactRef) -> ReadStatistics | None:
        """Rehash the source artifact and return cached statistics on a match.

        A hit never reparses records: the source is reopened and rehashed to
        prove it still matches the cached artifact identity.
        """
        observed = _verify_source(artifact)
        return self.lookup(observed)

    def publish(self, statistics: ReadStatistics) -> None:
        key = cache_key(statistics.artifact_sha256)
        existing = self._entries.get(key)
        if existing is not None and existing != statistics:
            raise OrganelleExecutionError(
                code="assembly.read_statistics_cache_conflict",
                message="cached read statistics conflict for one content identity",
                details={"artifact_sha256": statistics.artifact_sha256},
            )
        self._entries[key] = statistics


def stream_read_statistics(
    artifact: ArtifactRef,
    *,
    cache: ReadStatisticsCache | None = None,
) -> ReadStatistics:
    """Stream one long-read artifact into immutable statistics.

    On a cache hit, rehash before returning. On a miss, hash the exact bytes
    parsed in one pass, then independently rehash the path to retain detection
    of replacement/modification during parsing. No metadata-only trust cache
    can weaken the existing content verification boundary.
    """
    if cache is not None:
        cached = cache.lookup(artifact.sha256)
        if cached is not None:
            _verify_source(artifact)
            return cached
    path = Path(artifact.uri)
    if not path.is_file():
        raise OrganelleExecutionError(
            code="assembly.output_incomplete",
            message=f"declared input artifact is missing: {artifact.uri}",
            details={"uri": artifact.uri},
        )
    digest = hashlib.sha256()
    record_count, total_bases, minimum_length, maximum_length = parse_records(
        path,
        artifact.format,
        hash_update=digest.update,
    )
    observed = digest.hexdigest()
    _check_digest(artifact, observed)
    # Intentionally retain the independent post-parse verification: the path
    # may now refer to different bytes from the descriptor the parser consumed.
    _verify_source(artifact)
    return ReadStatistics(
        artifact_sha256=observed,
        record_count=record_count,
        total_bases=total_bases,
        minimum_length=minimum_length,
        maximum_length=maximum_length,
    )
