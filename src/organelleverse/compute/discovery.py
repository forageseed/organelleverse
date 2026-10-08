"""Lazy, import-free discovery of compute providers (spec §6).

Discovery reads only entry-point name/value, distribution name/version, and a
canonical digest of the installed ``RECORD`` entries via
:mod:`importlib.metadata`. It **never** calls :meth:`EntryPoint.load`, never
imports a factory, never starts a process, never opens a socket, and never
probes a host. Bare ``import organelleverse`` performs no discovery at all.

This mirrors the metadata-only discipline of
:mod:`organelleverse.capabilities.discovery`: installed → discovered → trusted
→ enabled are four separate gates, and discovery is the read-only second one.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from organelleverse.core.errors import OrganelleContractError

from .contracts import ProviderCandidate

__all__ = [
    "DistributionLike",
    "EntryPointLike",
    "compute_provider_record_digest",
    "discover_compute_providers",
]

_ENTRY_POINT_GROUP = "organelleverse.compute_providers"


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


class DistributionLike(Protocol):
    """The subset of :class:`importlib.metadata.Distribution` discovery uses."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def files(self) -> Sequence[PurePosixPath] | None: ...

    def read_text(self, filename: str) -> str | None: ...


class EntryPointLike(Protocol):
    """The subset of :class:`importlib.metadata.EntryPoint` discovery uses."""

    name: str
    value: str

    @property
    def dist(self) -> DistributionLike | None: ...


def _record_entries(record_text: str) -> list[list[str]]:
    """Parse installed ``RECORD`` CSV into canonical ``[path, hash, size]`` rows."""
    entries: list[list[str]] = []
    for row in csv.reader(io.StringIO(record_text)):
        if not row:
            continue
        path = str(PurePosixPath(Path(row[0]).as_posix()))
        archive_hash = row[1] if len(row) > 1 else ""
        size = row[2] if len(row) > 2 else ""
        entries.append([path, archive_hash, size])
    return sorted(entries)


def compute_provider_record_digest(distribution: DistributionLike) -> str:
    """Canonical SHA256 of the distribution's installed ``RECORD`` entries.

    The payload binds the full entry — POSIX-normalized path, recorded
    per-file archive hash, and size — not just the path list.  Reinstalling
    the same version with different bytes, or rewriting ``RECORD``, changes
    the digest and invalidates trust at the gate.  (An in-place edit that
    leaves ``RECORD`` untouched is outside this binding, exactly as it is
    outside pip's own install-time model; detecting it would require
    re-hashing every installed file at discovery time.)

    When ``RECORD`` cannot be read, the payload falls back to the sorted
    normalized path set from :attr:`Distribution.files`; when the
    distribution reports no files at all, a deterministic empty payload is
    hashed.  Discovery never raises for a malformed distribution — it simply
    derives a weaker identity, and a provider with no verifiable record of
    what it installed is refused at the trust gate.
    """
    record_text: str | None = None
    try:
        record_text = distribution.read_text("RECORD")
    except OSError:
        record_text = None
    if record_text:
        payload: object = {"record": _record_entries(record_text)}
        return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    files = distribution.files
    paths: tuple[str, ...] = ()
    if files:
        paths = tuple(sorted(str(PurePosixPath(Path(str(p)).as_posix())) for p in files))
    return "sha256:" + hashlib.sha256(_canonical_json_bytes({"paths": paths})).hexdigest()


def _entry_point_sort_key(entry_point: EntryPointLike) -> tuple[str, str, str]:
    dist = entry_point.dist
    dist_name = dist.name if dist is not None else ""
    dist_version = dist.version if dist is not None else ""
    return (entry_point.name, dist_name, dist_version)


def _to_candidate(entry_point: EntryPointLike) -> ProviderCandidate:
    dist = entry_point.dist
    if dist is None:
        raise OrganelleContractError(
            code="compute.provider_entry_point_orphan",
            message=(
                "compute provider entry point has no distribution metadata; "
                "an installed distribution identity is required for trust"
            ),
            details={"entry_point_name": entry_point.name, "entry_point_value": entry_point.value},
        )
    return ProviderCandidate(
        provider_id=entry_point.name,
        distribution_name=dist.name,
        distribution_version=dist.version,
        entry_point_locator=entry_point.value,
        distribution_record_digest=compute_provider_record_digest(dist),
    )


def discover_compute_providers(
    *,
    entry_points: Sequence[EntryPointLike] | None = None,
) -> tuple[ProviderCandidate, ...]:
    """Return provider candidates declared by installed distributions.

    Reads entry-point metadata only; never imports or loads any provider
    factory. Candidates are sorted deterministically by
    ``(provider_id, distribution_name, distribution_version)``.

    Pass an explicit empty sequence to select none; pass ``None`` to scan the
    standard ``organelleverse.compute_providers`` entry-point group.
    """
    if entry_points is None:
        selected = tuple(
            cast(EntryPointLike, ep)
            for ep in importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
        )
    else:
        selected = tuple(entry_points)
    candidates = [_to_candidate(ep) for ep in sorted(selected, key=_entry_point_sort_key)]
    return tuple(candidates)
