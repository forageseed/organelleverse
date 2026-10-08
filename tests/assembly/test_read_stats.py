"""Streaming long-read statistics and content-addressed cache tests."""

from __future__ import annotations

import builtins
import gzip
import hashlib
import sys
from pathlib import Path
from typing import IO, Any

import pytest
from pydantic import ValidationError

from organelleverse.assembly import read_stats
from organelleverse.assembly.read_stats import (
    PARSER_VERSION,
    SCHEMA_VERSION,
    ReadStatistics,
    ReadStatisticsCache,
    stream_read_statistics,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleExecutionError, OrganelleInputError


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _gzip_write(path: Path, text: str) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _artifact(path: Path, *, fmt: str, media_type: str = "application/x-fastq") -> ArtifactRef:
    return ArtifactRef.from_path(path, kind="long_read", format=fmt, media_type=media_type)


def _expected_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# ReadStatistics model
# ---------------------------------------------------------------------------


def test_read_statistics_is_frozen_with_fixed_schema_and_parser_versions() -> None:
    stats = ReadStatistics(
        artifact_sha256="a" * 64,
        record_count=3,
        total_bases=12,
        minimum_length=3,
        maximum_length=5,
    )
    assert stats.schema_version == SCHEMA_VERSION
    assert stats.parser_version == PARSER_VERSION

    with pytest.raises(ValidationError):
        stats.record_count = 9  # type: ignore[misc]


def test_read_statistics_rejects_unknown_fields_and_bad_hashes() -> None:
    with pytest.raises(ValidationError):
        ReadStatistics(
            artifact_sha256="not-a-hex",
            record_count=1,
            total_bases=4,
            minimum_length=4,
            maximum_length=4,
        )
    payload = ReadStatistics(
        artifact_sha256="a" * 64,
        record_count=1,
        total_bases=4,
        minimum_length=4,
        maximum_length=4,
    ).model_dump(mode="python", round_trip=True)
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        ReadStatistics.model_validate(payload)


def test_read_statistics_requires_maximum_at_least_minimum() -> None:
    with pytest.raises(ValidationError):
        ReadStatistics(
            artifact_sha256="a" * 64,
            record_count=1,
            total_bases=4,
            minimum_length=9,
            maximum_length=4,
        )


# ---------------------------------------------------------------------------
# Streaming parser: FASTA / FASTQ / gzip
# ---------------------------------------------------------------------------


def test_multiline_fasta_totals_record_count_and_length_extremes(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "reads.fasta",
        ">r1\nACGT\nACGT\n>r2\nACGTACGT\n",
    )
    stats = stream_read_statistics(_artifact(path, fmt="fasta", media_type="text/x-fasta"))
    assert stats.record_count == 2
    assert stats.total_bases == 16  # 8 + 8
    assert stats.minimum_length == 8
    assert stats.maximum_length == 8
    assert stats.artifact_sha256 == _expected_sha256(path)


def test_fasta_header_requires_a_nonempty_name(tmp_path: Path) -> None:
    path = _write(tmp_path / "reads.fasta", ">   \nACGT\n")
    with pytest.raises(OrganelleInputError) as raised:
        stream_read_statistics(_artifact(path, fmt="fasta", media_type="text/x-fasta"))
    assert raised.value.code == "assembly.malformed_long_read_record"


def test_fastq_sequence_and_quality_lengths_are_validated(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "reads.fastq",
        "@r1\nACGTACGT\n+\nIIIIIIII\n@r2\nACGT\n+\nIIII\n",
    )
    stats = stream_read_statistics(_artifact(path, fmt="fastq"))
    assert stats.record_count == 2
    assert stats.total_bases == 12
    assert stats.minimum_length == 4
    assert stats.maximum_length == 8


def test_gzip_fastq_is_streamed_like_plain_fastq(tmp_path: Path) -> None:
    raw = "@r1\nACGT\n+\nIIII\n@r2\nACGTAC\n+\nIIIIII\n"
    plain = _write(tmp_path / "plain.fastq", raw)
    gz = _gzip_write(tmp_path / "reads.fastq.gz", raw)
    plain_stats = stream_read_statistics(_artifact(plain, fmt="fastq"))
    gz_stats = stream_read_statistics(_artifact(gz, fmt="fastq"))
    assert gz_stats.record_count == plain_stats.record_count == 2
    assert gz_stats.total_bases == plain_stats.total_bases == 10
    # The full-file SHA256 differs because one payload is gzip-compressed.
    assert gz_stats.artifact_sha256 != plain_stats.artifact_sha256
    assert gz_stats.artifact_sha256 == _expected_sha256(gz)


def test_gzip_fasta_streaming(tmp_path: Path) -> None:
    gz = _gzip_write(tmp_path / "reads.fasta.gz", ">r1\nACGTACGT\n>r2\nACGT\n")
    stats = stream_read_statistics(_artifact(gz, fmt="fasta", media_type="text/x-fasta"))
    assert stats.record_count == 2
    assert stats.total_bases == 12


# ---------------------------------------------------------------------------
# Adversarial / malformed inputs
# ---------------------------------------------------------------------------


def test_empty_input_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "empty.fastq", "")
    with pytest.raises(OrganelleInputError):
        stream_read_statistics(_artifact(path, fmt="fastq"))


def test_truncated_fastq_missing_quality_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "trunc.fastq", "@r1\nACGT\n+\n")
    with pytest.raises(OrganelleInputError):
        stream_read_statistics(_artifact(path, fmt="fastq"))


def test_malformed_fastq_missing_header_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "bad.fastq", "r1\nACGT\n+\nIIII\n")
    with pytest.raises(OrganelleInputError):
        stream_read_statistics(_artifact(path, fmt="fastq"))


def test_sequence_quality_length_mismatch_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "mismatch.fastq", "@r1\nACGTAC\n+\nIIII\n")
    with pytest.raises(OrganelleInputError):
        stream_read_statistics(_artifact(path, fmt="fastq"))


def test_fasta_without_header_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "noheader.fasta", "ACGTACGT\n")
    with pytest.raises(OrganelleInputError):
        stream_read_statistics(_artifact(path, fmt="fasta", media_type="text/x-fasta"))


def test_truncated_fasta_header_without_sequence_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "trunc.fasta", ">r1\nACGT\n>r2\n")
    with pytest.raises(OrganelleInputError):
        stream_read_statistics(_artifact(path, fmt="fasta", media_type="text/x-fasta"))


# ---------------------------------------------------------------------------
# Hash integrity and tampering
# ---------------------------------------------------------------------------


def test_source_file_tampering_is_detected_after_artifact_was_built(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path / "reads.fastq", "@r1\nACGT\n+\nIIII\n")
    artifact = _artifact(path, fmt="fastq")
    path.write_text("@r1\nACGTACGT\n+\nIIIIIIII\n", encoding="utf-8")
    with pytest.raises(OrganelleExecutionError):
        stream_read_statistics(artifact)


def test_source_change_during_parse_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write(tmp_path / "reads.fastq", "@r1\nACGT\n+\nIIII\n")
    artifact = _artifact(path, fmt="fastq")
    real_parse = read_stats.parse_records

    def parse_then_replace(path_obj: Path, fmt: str, **kwargs) -> tuple[int, int, int, int]:
        parsed = real_parse(path_obj, fmt, **kwargs)
        path_obj.write_text("@r1\nACGTACGT\n+\nIIIIIIII\n", encoding="utf-8")
        return parsed

    monkeypatch.setattr(read_stats, "parse_records", parse_then_replace)
    with pytest.raises(OrganelleExecutionError) as raised:
        stream_read_statistics(artifact)
    assert raised.value.code == "assembly.output_incomplete"


def test_cache_tamper_detection_rehashes_before_a_hit(tmp_path: Path) -> None:
    path = _write(tmp_path / "reads.fastq", "@r1\nACGT\n+\nIIII\n")
    artifact = _artifact(path, fmt="fastq")
    cache = ReadStatisticsCache()
    cache.publish(stream_read_statistics(artifact))
    path.write_text("@r1\nACGTACGT\n+\nIIIIIIII\n", encoding="utf-8")
    with pytest.raises(OrganelleExecutionError):
        cache.get(artifact)


# ---------------------------------------------------------------------------
# Content-addressed cache identity
# ---------------------------------------------------------------------------


def test_cache_key_carries_full_sha256_and_parser_and_schema_versions() -> None:
    sha = "a" * 64
    key = read_stats.cache_key(sha)
    assert sha in key
    assert SCHEMA_VERSION in key
    assert PARSER_VERSION in key


def test_cache_hit_rehashes_but_does_not_reparse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write(tmp_path / "reads.fastq", "@r1\nACGT\n+\nIIII\n@r2\nACGTAC\n+\nIIIIII\n")
    artifact = _artifact(path, fmt="fastq")
    cache = ReadStatisticsCache()

    calls = {"parse": 0}
    real_parse = read_stats.parse_records

    def counting_parse(path_obj: Path, fmt: str, **kwargs) -> tuple[int, int, int, int]:
        calls["parse"] += 1
        return real_parse(path_obj, fmt, **kwargs)

    monkeypatch.setattr(read_stats, "parse_records", counting_parse)

    first = stream_read_statistics(artifact, cache=cache)
    assert calls["parse"] == 1
    cache.publish(first)

    second = stream_read_statistics(artifact, cache=cache)
    assert second == first
    assert calls["parse"] == 1  # hit: rehashed, not reparsed


def test_cache_miss_for_changed_content_does_not_cross_hit(tmp_path: Path) -> None:
    one = _write(tmp_path / "a.fastq", "@r1\nACGT\n+\nIIII\n")
    two = _write(tmp_path / "b.fastq", "@r1\nACGTACGT\n+\nIIIIIIII\n")
    cache = ReadStatisticsCache()
    one_artifact = _artifact(one, fmt="fastq")
    cache.publish(stream_read_statistics(one_artifact))
    other = stream_read_statistics(_artifact(two, fmt="fastq"), cache=cache)
    assert other.total_bases == 8
    cached = cache.get(one_artifact)
    assert cached is not None
    assert other.artifact_sha256 != cached.artifact_sha256


def test_publishing_after_success_lets_a_later_call_hit(tmp_path: Path) -> None:
    path = _write(tmp_path / "reads.fastq", "@r1\nACGT\n+\nIIII\n")
    artifact = _artifact(path, fmt="fastq")
    cache = ReadStatisticsCache()
    stats = stream_read_statistics(artifact)
    assert cache.get(artifact) is None
    cache.publish(stats)
    assert cache.get(artifact) == stats


def test_cache_rejects_conflicting_statistics_for_one_content_identity() -> None:
    cache = ReadStatisticsCache()
    first = ReadStatistics(
        artifact_sha256="a" * 64,
        record_count=1,
        total_bases=4,
        minimum_length=4,
        maximum_length=4,
    )
    conflicting = ReadStatistics(
        artifact_sha256="a" * 64,
        record_count=1,
        total_bases=8,
        minimum_length=8,
        maximum_length=8,
    )
    cache.publish(first)
    with pytest.raises(OrganelleExecutionError) as raised:
        cache.publish(conflicting)
    assert raised.value.code == "assembly.read_statistics_cache_conflict"


# ---------------------------------------------------------------------------
# No unbounded whole-file read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compressed", [False, True])
def test_miss_fuses_hash_and_parse_but_retains_post_parse_hash(tmp_path, monkeypatch, compressed):
    path = tmp_path / "reads"
    raw = "@r1\nACGT\n+\nIIII\n"
    (_gzip_write if compressed else _write)(path, raw)
    artifact = _artifact(path, fmt="fastq")
    original_hash = read_stats._hash_file
    calls = []

    def counted_hash(source):
        calls.append(source)
        return original_hash(source)

    monkeypatch.setattr(read_stats, "_hash_file", counted_hash)
    assert stream_read_statistics(artifact).total_bases == 4
    assert calls == [path]  # Only the intentionally independent verification pass.


def test_gzip_members_are_all_hashed_and_counted(tmp_path):
    path = tmp_path / "multi.fastq.gz"
    path.write_bytes(gzip.compress(b"@a\nAC\n+\nII\n") + gzip.compress(b"@b\nACGT\n+\nIIII\n"))
    stats = stream_read_statistics(_artifact(path, fmt="fastq"))
    assert (stats.record_count, stats.total_bases) == (2, 6)


def test_whitespace_check_is_equivalent_for_all_unicode_codepoints():
    # The compiled scan replaces the former Python per-character loop without
    # changing which sequence characters are rejected, including Unicode spaces.
    for value in range(sys.maxunicode + 1):
        character = chr(value)
        assert bool(read_stats._WHITESPACE.search(character)) == character.isspace()


class _BoundedReadProbe:
    """Wraps a binary file handle and fails if read() is called without a size."""

    def __init__(self, handle: IO[bytes]) -> None:
        self._handle = handle

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)

    def read(self, size: int = -1) -> bytes:
        # Disallow bare / huge reads; only bounded chunk reads are permitted.
        assert size > 0, "unbounded whole-file read() is forbidden"
        return self._handle.read(size)


def test_streaming_never_performs_an_unbounded_whole_file_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write(
        tmp_path / "reads.fastq",
        "".join(f"@r{i}\nACGT\n+\nIIII\n" for i in range(64)),
    )
    artifact = _artifact(path, fmt="fastq")
    real_open = builtins.open

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        handle = real_open(file, mode, *args, **kwargs)
        if "b" in mode:
            return _BoundedReadProbe(handle)
        return handle

    monkeypatch.setattr(read_stats, "open", guarded_open, raising=False)
    stats = stream_read_statistics(artifact)
    assert stats.record_count == 64
