"""Transient transport failures retry without accepting unverified bytes."""

import hashlib
import io
import urllib.error
import urllib.request
from http.client import IncompleteRead

import pytest

from organelleverse.assembly import environments
from organelleverse.core.errors import OrganelleDependencyError


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("temporary DNS failure"),
        TimeoutError("timed out"),
        ConnectionResetError("reset"),
        IncompleteRead(b"partial", 100),
        urllib.error.HTTPError("https://example.org/db", 503, "unavailable", {}, None),
    ],
)
def test_download_retries_transient_failure_and_verifies_bytes(tmp_path, monkeypatch, error):
    calls = []
    sleeps = []
    content = b"verified database"

    def open_url(url, *, timeout):
        calls.append((url, timeout))
        if len(calls) == 1:
            raise error
        return io.BytesIO(content)

    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    monkeypatch.setattr(environments.time, "sleep", sleeps.append)
    path = tmp_path / "database"
    assert (
        environments._DefaultDownloader().download(
            "https://example.org/db", path, hashlib.sha256(content).hexdigest()
        )
        == content
    )
    assert path.read_bytes() == content
    assert calls == [("https://example.org/db", 180)] * 2
    assert sleeps == [2]


@pytest.mark.parametrize("status,attempts", [(404, 1), (429, 3), (503, 3)])
def test_download_failure_is_bounded_and_never_published(tmp_path, monkeypatch, status, attempts):
    calls = []

    def open_url(url, *, timeout):
        calls.append(url)
        raise urllib.error.HTTPError(url, status, "failed", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    monkeypatch.setattr(environments.time, "sleep", lambda _: None)
    path = tmp_path / "database"
    with pytest.raises(OrganelleDependencyError, match="retry the managed installation") as exc:
        environments._DefaultDownloader().download("https://example.org/db", path, "a" * 64)
    assert exc.value.details["attempts"] == attempts
    assert len(calls) == attempts
    assert not path.exists()


def test_checksum_mismatch_is_not_retried_or_published(tmp_path, monkeypatch):
    calls = []

    def open_url(url, *, timeout):
        calls.append(url)
        return io.BytesIO(b"wrong bytes")

    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    path = tmp_path / "database"
    with pytest.raises(OrganelleDependencyError, match="hash does not match"):
        environments._DefaultDownloader().download("https://example.org/db", path, "a" * 64)
    assert len(calls) == 1
    assert not path.exists()
