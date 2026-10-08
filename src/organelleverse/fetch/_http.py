"""HTTP plumbing for the fetch suite: transport, retry/backoff, URL batching.

Standard library only — ``requests`` is not an OrganelleVerse dependency.
``urllib`` is imported lazily inside the transport so that importing this
module (and therefore Operation Registry discovery) never touches the network
stack.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from ..core.errors import OrganelleError, OrganelleExecutionError, OrganelleParameterError

__all__ = [
    "MAX_URL_LENGTH",
    "HttpResponse",
    "Transport",
    "UrllibTransport",
    "batch_accessions_for_url",
    "default_transport",
    "download_with_retry",
    "eutils_delay",
    "get_with_retry",
    "retry_with_backoff",
]

# NCBI returns HTTP 414 for long URLs; keep every request under this.
MAX_URL_LENGTH = 2000

# E-utilities allows 3 requests/sec anonymously, 10/sec with an API key.
_DELAY_WITHOUT_KEY = 0.34
_DELAY_WITH_KEY = 0.11

# Statuses that mean "try again later", shared by get_with_retry/download_with_retry.
_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})

T = TypeVar("T")


@dataclass(frozen=True)
class HttpResponse:
    """One HTTP response: status and body together.

    Never raised for a real response, even a 404 or 403 — only a genuine
    connection failure (no response received at all) raises. This is what
    lets a caller tell "confirmed absent" apart from "the server errored"
    without exception-driven control flow for what is a normal HTTP outcome.
    """

    status: int
    body: bytes


class Transport(Protocol):
    """Minimal HTTP surface, so tests can inject a fake and stay offline.

    ``post`` exists for ``epost``: a long accession list must go in the request
    body, not the URL, or NCBI answers 414 regardless of how you batch it.
    ``head``/``get_response``/``download_to_path`` back the alternate nuclear
    sources: existence probing (IMP), telling a miss apart from an error
    (GIR/PGD/TAIR), and streaming large files without buffering them whole.
    """

    def get(
        self, url: str, *, timeout: float = ..., headers: Mapping[str, str] | None = ...
    ) -> bytes: ...

    def post(
        self,
        url: str,
        data: bytes,
        *,
        timeout: float = ...,
        headers: Mapping[str, str] | None = ...,
    ) -> bytes: ...

    def head(
        self, url: str, *, timeout: float = ..., headers: Mapping[str, str] | None = ...
    ) -> int: ...

    def get_response(
        self,
        url: str,
        *,
        method: str = ...,
        data: bytes | None = ...,
        timeout: float = ...,
        headers: Mapping[str, str] | None = ...,
    ) -> HttpResponse: ...

    def download_to_path(
        self,
        url: str,
        dest: Path,
        *,
        method: str = ...,
        data: bytes | None = ...,
        timeout: float = ...,
        headers: Mapping[str, str] | None = ...,
    ) -> int: ...


def _merged_headers(extra: Mapping[str, str] | None) -> dict[str, str]:
    merged = {"User-Agent": "organelleverse-fetch"}
    if extra:
        merged.update(extra)
    return merged


class UrllibTransport:
    """The real transport. Imports ``urllib`` lazily, per the discovery-safety rule."""

    def get(
        self, url: str, *, timeout: float = 120.0, headers: Mapping[str, str] | None = None
    ) -> bytes:
        from urllib.request import Request, urlopen

        request = Request(url, headers=_merged_headers(headers))
        with urlopen(request, timeout=timeout) as response:
            data: bytes = response.read()
        return data

    def post(
        self,
        url: str,
        data: bytes,
        *,
        timeout: float = 120.0,
        headers: Mapping[str, str] | None = None,
    ) -> bytes:
        from urllib.request import Request, urlopen

        merged = _merged_headers(headers)
        merged.setdefault("Content-Type", "application/x-www-form-urlencoded")
        request = Request(url, data=data, headers=merged)
        with urlopen(request, timeout=timeout) as response:
            body: bytes = response.read()
        return body

    def head(
        self, url: str, *, timeout: float = 30.0, headers: Mapping[str, str] | None = None
    ) -> int:
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        request = Request(url, method="HEAD", headers=_merged_headers(headers))
        try:
            with urlopen(request, timeout=timeout) as response:
                return int(response.status)
        except HTTPError as exc:
            return exc.code

    def get_response(
        self,
        url: str,
        *,
        method: str = "GET",
        data: bytes | None = None,
        timeout: float = 120.0,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        request = Request(url, data=data, method=method, headers=_merged_headers(headers))
        try:
            with urlopen(request, timeout=timeout) as response:
                return HttpResponse(status=int(response.status), body=response.read())
        except HTTPError as exc:
            return HttpResponse(status=exc.code, body=exc.read())

    def download_to_path(
        self,
        url: str,
        dest: Path,
        *,
        method: str = "GET",
        data: bytes | None = None,
        timeout: float = 1800.0,
        headers: Mapping[str, str] | None = None,
    ) -> int:
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        request = Request(url, data=data, method=method, headers=_merged_headers(headers))
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urlopen(request, timeout=timeout) as response:
                status = int(response.status)
                with dest.open("wb") as handle:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        handle.write(chunk)
                return status
        except HTTPError as exc:
            with dest.open("wb") as handle:
                handle.write(exc.read())
            return exc.code


def default_transport() -> Transport:
    return UrllibTransport()


def ncbi_api_key() -> str | None:
    """NCBI's own convention (``NCBI_API_KEY``), not an OrganelleVerse switch."""
    key = os.environ.get("NCBI_API_KEY")
    return key or None


def eutils_delay() -> float:
    """Seconds to wait between E-utilities calls, per NCBI's rate limit."""
    return _DELAY_WITH_KEY if ncbi_api_key() else _DELAY_WITHOUT_KEY


from http.client import IncompleteRead as _IncompleteRead


def retry_with_backoff(
    operation: Callable[[], T],
    *,
    max_attempts: int = 5,
    initial_delay: float = 3.0,
    multiplier: float = 2.0,
) -> T:
    """Run ``operation``, retrying transient transport failures with backoff.

    An :class:`OrganelleError` other than a transport failure (a bad query, a
    malformed response) will never succeed on retry, so it is re-raised
    immediately rather than hammering NCBI. Only a raw ``OSError``/
    ``TimeoutError`` bubbling out of ``operation`` is retried — an
    already-raised ``OrganelleError``'s ``retryable`` flag is never consulted
    here; see ``get_with_retry``/``download_with_retry`` for how a transient
    HTTP status is turned into a plain ``OSError`` so this loop catches it.
    """
    delay = initial_delay
    last: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except OrganelleError:
            raise  # a contract/parameter failure is not transient
        except _IncompleteRead as exc:  # a truncated chunked body is transient
            last = exc
        except (OSError, TimeoutError) as exc:
            last = exc
            if attempt < max_attempts:
                if delay:
                    time.sleep(delay)
                delay *= multiplier

    raise OrganelleExecutionError(
        code="network.transient",
        message=f"transport failed after {max_attempts} attempts: {last}",
        details={"attempts": max_attempts, "last_error": str(last)},
        retryable=True,
        suggested_action={
            "retry": "increase max_attempts, or set NCBI_API_KEY to raise the rate limit"
        },
    )


def get_with_retry(
    transport: Transport,
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    timeout: float = 120.0,
    headers: Mapping[str, str] | None = None,
    max_attempts: int = 5,
    initial_delay: float = 3.0,
) -> HttpResponse:
    """``transport.get_response``, but a transient status backs off and retries.

    A plain HTTP response (even 404/403) is returned as-is on the first try —
    only a status in ``429``/``5xx`` raises a plain ``OSError`` so the
    existing ``retry_with_backoff`` retries it exactly like a genuine
    connection failure would.
    """

    def attempt() -> HttpResponse:
        response = transport.get_response(
            url, method=method, data=data, timeout=timeout, headers=headers
        )
        if response.status in _TRANSIENT_STATUSES:
            raise OSError(f"HTTP {response.status} from {url}")
        return response

    return retry_with_backoff(attempt, max_attempts=max_attempts, initial_delay=initial_delay)


def download_with_retry(
    transport: Transport,
    url: str,
    dest: Path,
    *,
    method: str = "GET",
    data: bytes | None = None,
    timeout: float = 1800.0,
    headers: Mapping[str, str] | None = None,
    max_attempts: int = 5,
    initial_delay: float = 3.0,
) -> int:
    """``transport.download_to_path``, with the same transient-status retry as ``get_with_retry``."""

    def attempt() -> int:
        status = transport.download_to_path(
            url, dest, method=method, data=data, timeout=timeout, headers=headers
        )
        if status in _TRANSIENT_STATUSES:
            raise OSError(f"HTTP {status} from {url}")
        return status

    return retry_with_backoff(attempt, max_attempts=max_attempts, initial_delay=initial_delay)


def batch_accessions_for_url(
    accessions: list[str],
    *,
    base_url_length: int,
    max_url_length: int = MAX_URL_LENGTH,
) -> list[list[str]]:
    """Split accessions into batches whose comma-joined URL stays under the limit.

    Order is preserved and nothing is dropped: the concatenation of the batches
    equals the input.
    """
    if not accessions:
        return []

    budget = max_url_length - base_url_length
    longest = max(len(a) for a in accessions)
    if budget < longest:
        raise OrganelleParameterError(
            code="input.url_budget_exhausted",
            message=(
                f"base URL of {base_url_length} chars leaves {budget} chars, "
                f"too few for an accession of {longest} chars"
            ),
            details={"base_url_length": base_url_length, "budget": budget},
        )

    batches: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    for accession in accessions:
        # +1 for the separating comma, except on the first element of a batch.
        addition = len(accession) + (1 if current else 0)
        if current and current_len + addition > budget:
            batches.append(current)
            current = [accession]
            current_len = len(accession)
        else:
            current.append(accession)
            current_len += addition

    if current:
        batches.append(current)
    return batches
