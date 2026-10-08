"""Tests for the real-data fetch fixes (2026-08-15).

Plant-mitochondria queries return thousands of accessions and exposed two
transient-transport defects in the efetch enrichment stage, fixed the same
day: URI-length overflow (HTTP 414) at large batch sizes, and chunked-body
truncation (IncompleteRead) not being retried.
"""

from __future__ import annotations

import http.client
from pathlib import Path

import pytest

from organelleverse.fetch._http import retry_with_backoff
from organelleverse.fetch.genbank_meta import fetch_genbank_metadata


def _fake_records(n: int) -> list[dict[str, object]]:
    return [{"accession": f"ABC123{i:04d}.1"} for i in range(n)]


def test_efetch_batches_capped_at_ten_for_thousands_of_accessions() -> None:
    seen_batches: list[list[str]] = []

    class _TinyTransport:
        def get(self, url: str, *, timeout: float) -> bytes:
            ids = url.split("id=")[1]
            seen_batches.append(ids.split(","))
            # empty GBSeq set is fine - only batching is under test
            return b"<?xml version='1.0'?><GBSet></GBSet>"

    fetch_genbank_metadata(_TinyTransport(), accessions=[r["accession"] for r in _fake_records(97)])
    assert seen_batches, "no efetch calls made"
    assert all(len(batch) <= 10 for batch in seen_batches), [
        len(b) for b in seen_batches
    ]
    assert sum(len(b) for b in seen_batches) == 97


def test_incomplete_read_is_retried_and_succeeds() -> None:
    attempts = {"n": 0}

    def flaky() -> bytes:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise http.client.IncompleteRead(b"partial chunk")
        return b"done"

    out = retry_with_backoff(flaky, max_attempts=5, initial_delay=0.01)
    assert out == b"done"
    assert attempts["n"] == 3


def test_incomplete_read_exhausts_retries() -> None:
    def always_truncated() -> bytes:
        raise http.client.IncompleteRead(b"")

    from organelleverse.core.errors import OrganelleExecutionError

    with pytest.raises(OrganelleExecutionError) as exc:
        retry_with_backoff(always_truncated, max_attempts=2, initial_delay=0.01)
    assert "transport failed after 2 attempts" in exc.value.message
