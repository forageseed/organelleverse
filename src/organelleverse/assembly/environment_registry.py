"""Locked persistent registry of verified environment providers.

The registry is the single source of truth for which backend providers have been
installed and verified. It stores canonical JSON protected by an advisory
POSIX lock and atomically replaces the registry file from a sibling temporary
file so concurrent writers and crash-interrupted writes can never produce a
truncated or interleaved registry.

Read paths (``list`` / ``locate``) are strictly read-only: they create no
directories and write no files, even when the registry does not yet exist.
Only :meth:`InstallationRegistry.register_verified` materializes the tool root,
the lock, and the registry file.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Literal, TextIO, cast

from organelleverse.operations.spec import StrictSpecModel

from .environment_contracts import ResolvedProvider

__all__ = [
    "InstallationRegistry",
    "RegisteredComponent",
    "RegisteredProvider",
    "default_tool_root",
]

_SCHEMA_VERSION = "organelleverse.environment-registry.v1"
_REGISTRY_FILENAME = "environments.json"
_LOCK_FILENAME = "environments.lock"

_ComponentKind = Literal["executable", "resource", "host_provider"]


# ---------------------------------------------------------------------------
# persisted records
# ---------------------------------------------------------------------------


class RegisteredComponent(StrictSpecModel):
    """A single verified component persisted in the registry."""

    role: str
    kind: _ComponentKind
    path: Path
    sha256: str
    version: str | None = None


class RegisteredProvider(StrictSpecModel):
    """A verified provider persisted in the registry.

    ``backend_id`` and ``version`` are derived from the provider's primary
    component (the executable/host_provider that names the backend), mirroring
    the resolver's primary-item convention. ``source`` records where the
    provider was actually discovered (``ResolvedProvider.discovery_source``).
    """

    backend_id: str
    provider_digest: str
    capability_contract_digest: str
    platform: str
    source: str
    prefix: Path | None
    version: str | None
    components: tuple[RegisteredComponent, ...]


# ---------------------------------------------------------------------------
# tool root resolution
# ---------------------------------------------------------------------------


def default_tool_root() -> Path:
    """Resolve the persistent tool root without creating it.

    Precedence: ``$ORGANELLEVERSE_TOOL_ROOT`` (used verbatim), else
    ``$XDG_DATA_HOME/organelleverse/tools``, else
    ``~/.local/share/organelleverse/tools``.
    """
    explicit = os.environ.get("ORGANELLEVERSE_TOOL_ROOT")
    if explicit:
        return Path(explicit)
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "organelleverse" / "tools"
    return Path.home() / ".local" / "share" / "organelleverse" / "tools"


class _AdvisoryLock:
    """Exclusive POSIX advisory file lock held for the duration of a write.

    Mirrors ``assembly/environments.py``'s ``_advisory_lock`` so the registry
    uses the same locking idiom as the rest of the package.
    """

    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path
        self._handle: TextIO | None = None

    def __enter__(self) -> _AdvisoryLock:
        handle = self._lock_path.open("a+", encoding="utf-8")
        self._handle = handle
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


class InstallationRegistry:
    """Locked, atomically-updated registry of verified providers.

    The constructor performs no I/O and creates nothing. Reads are lazy and
    side-effect free; only :meth:`register_verified` writes.
    """

    def __init__(self, *, tool_root: Path | None = None) -> None:
        self._tool_root = tool_root if tool_root is not None else default_tool_root()

    @property
    def tool_root(self) -> Path:
        return self._tool_root

    @property
    def _registry_path(self) -> Path:
        return self._tool_root / _REGISTRY_FILENAME

    # ------------------------------------------------------------------
    # read-only API
    # ------------------------------------------------------------------

    def list(self) -> tuple[RegisteredProvider, ...]:
        """Return every recorded provider. Creates no files or directories."""
        return self._read()

    def locate(self, backend_id: str) -> tuple[RegisteredProvider, ...]:
        """Return recorded providers for *backend_id*. Read-only."""
        return tuple(p for p in self._read() if p.backend_id == backend_id)

    # ------------------------------------------------------------------
    # write API
    # ------------------------------------------------------------------

    def register_verified(self, provider: ResolvedProvider) -> RegisteredProvider:
        """Record a verified provider.

        Idempotent by provider digest: re-registering an identical provider is
        a true no-op (no file write). Distinct provider digests are retained so
        exact installed versions remain discoverable.
        """
        record = _provider_to_record(provider)
        self._tool_root.mkdir(parents=True, exist_ok=True)
        with _AdvisoryLock(self._tool_root / _LOCK_FILENAME):
            records = list(self._read())
            exact = next(
                (
                    existing
                    for existing in records
                    if existing.backend_id == record.backend_id
                    and existing.provider_digest == record.provider_digest
                ),
                None,
            )
            compacted = [
                existing
                for existing in records
                if existing.provider_digest == record.provider_digest
                or not _same_provider_components(existing, record)
            ]
            if exact is not None:
                if compacted != records:
                    self._write_atomic(compacted)
                return exact
            compacted.append(record)
            self._write_atomic(compacted)
            return record

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _read(self) -> tuple[RegisteredProvider, ...]:
        """Load the registry without creating anything; empty when absent."""
        path = self._registry_path
        if not path.is_file():
            return ()
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"environment registry at {path} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"environment registry at {path} must contain a JSON object")
        registry_data = cast(dict[str, object], raw)
        schema_version = registry_data.get("schema_version")
        if schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"environment registry at {path} has unsupported schema_version "
                f"{schema_version!r}; expected {_SCHEMA_VERSION!r}"
            )
        providers = registry_data.get("providers", [])
        if not isinstance(providers, list):
            raise ValueError(f"environment registry at {path} has non-list 'providers'")
        provider_items = cast(list[dict[str, object]], providers)
        return tuple(RegisteredProvider.model_validate(item) for item in provider_items)

    def _write_atomic(self, records: list[RegisteredProvider]) -> None:
        """Canonical JSON written via a sibling temp + atomic replace."""
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "providers": [r.model_dump(mode="json") for r in records],
        }
        text = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        final_path = self._registry_path
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=f"{_REGISTRY_FILENAME}.", dir=self._tool_root)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp_path, final_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise


# ---------------------------------------------------------------------------
# provider -> record mapping
# ---------------------------------------------------------------------------


def _primary_component(provider: ResolvedProvider) -> object:
    """Return the primary backend component of *provider*.

    Mirrors the resolver's primary-item selection: the first
    executable/host_provider component, which by convention carries the
    backend id as its role. Falls back to the first component otherwise.
    """
    for component in provider.components:
        if component.kind in ("executable", "host_provider"):
            return component
    return provider.components[0]


def _provider_to_record(provider: ResolvedProvider) -> RegisteredProvider:
    primary = _primary_component(provider)
    components = tuple(
        RegisteredComponent(
            role=component.role,
            kind=component.kind,
            path=component.path,
            sha256=component.sha256,
            version=component.version,
        )
        for component in provider.components
    )
    return RegisteredProvider(
        backend_id=primary.role,  # type: ignore[attr-defined]
        provider_digest=provider.provider_digest,
        capability_contract_digest=provider.capability_contract_digest,
        platform=provider.platform,
        source=provider.discovery_source,
        prefix=provider.prefix,
        version=primary.version,  # type: ignore[attr-defined]
        components=components,
    )


def _same_provider_components(left: RegisteredProvider, right: RegisteredProvider) -> bool:
    """Recognize an identity-algorithm migration without collapsing real versions."""
    return (
        left.backend_id == right.backend_id
        and left.platform == right.platform
        and left.prefix == right.prefix
        and left.capability_contract_digest == right.capability_contract_digest
        and left.components == right.components
    )
