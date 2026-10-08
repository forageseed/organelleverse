"""Explicit, owner-controlled trust for compute providers (spec §6).

A distribution being *installed* does not activate or trust it. Three separate
gates apply before any provider factory is loaded:

1. **Discovered** — a candidate appears via entry-point metadata only.
2. **Trusted / enabled** — the owner explicitly authorizes the static installed
   identity (distribution name+version, entry-point locator, installed RECORD
   digest). Only then is the factory loaded exactly once and its closed
   :class:`~organelleverse.compute.contracts.ComputeProviderSpec` validated and
   bound to the trust record by its declaration digest.
3. **Re-verified on use** — a changed distribution version, locator, installed
   RECORD digest, or declaration digest invalidates trust
   (``compute.provider_trust_stale``) until the owner re-enables it.

The store is a JSON document at ``$ORGANELLEVERSE_HOME/l5/providers.json``,
mirroring the owner-controlled, atomic-record pattern of
:class:`~organelleverse.capabilities.trust.TrustStore`: file permissions are
tightened to 0o600, writes go through an atomic temp-rename, and the document
is schema-versioned. It is a private authorization document, not a public
artifact; concurrent writers are last-writer-wins.

Factory loading is delegated to an injectable loader so this module stays free
of any concrete provider import. The default loader resolves the entry point
lazily; tests inject a fake.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from organelleverse.core.errors import OrganelleContractError, OrganellePermissionError

from .contracts import ComputeProviderSpec, EnabledProvider, ProviderCandidate

__all__ = [
    "ComputeProviderTrustStore",
    "ProviderFactoryLoader",
    "canonical_declaration_digest",
    "default_factory_loader",
]

_LOCK = threading.RLock()
_SCHEMA_VERSION = "organelleverse.compute.trust.v1"


class ProviderFactoryLoader(Protocol):
    """Loads a trusted provider's factory and returns its closed declaration.

    Called exactly once per enable, *after* the static installed identity has
    been authorized. Implementations must not import or execute anything before
    being called.
    """

    def __call__(self, candidate: ProviderCandidate) -> ComputeProviderSpec: ...


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def canonical_declaration_digest(declaration: ComputeProviderSpec) -> str:
    """Canonical SHA256 of the closed declaration.

    The payload is the declaration's canonical JSON, so semantically identical
    declarations hash equally regardless of field declaration order. Trust is
    bound to this digest; a later declaration whose digest differs disables the
    provider.
    """
    return "sha256:" + hashlib.sha256(
        _canonical_json_bytes(declaration.model_dump(mode="json", by_alias=True))
    ).hexdigest()


def default_factory_loader(candidate: ProviderCandidate) -> ComputeProviderSpec:
    """Resolve the entry point lazily and validate the returned declaration.

    This is the only place the compute subsystem imports a provider factory.
    It runs solely on the explicit ``enable()`` path, never on import or
    discovery.
    """
    group = "organelleverse.compute_providers"
    match = None
    for entry_point in __import__("importlib.metadata", fromlist=["entry_points"]).entry_points(
        group=group
    ):
        if entry_point.name == candidate.provider_id and entry_point.value == candidate.entry_point_locator:
            match = entry_point
            break
    if match is None:
        raise OrganelleContractError(
            code="compute.provider_entry_point_missing",
            message=(
                "the trusted provider entry point is no longer declared by its "
                "distribution; the installation changed after trust was recorded"
            ),
            details={
                "provider_id": candidate.provider_id,
                "entry_point_locator": candidate.entry_point_locator,
            },
        )
    declaration = match.load()()
    if not isinstance(declaration, ComputeProviderSpec):
        raise OrganelleContractError(
            code="compute.provider_declaration_invalid",
            message=(
                "provider factory did not return a ComputeProviderSpec; the "
                "closed declaration contract was violated"
            ),
            details={"provider_id": candidate.provider_id},
        )
    return declaration


def _default_path() -> Path:
    home = Path(os.environ.get("ORGANELLEVERSE_HOME", Path.home() / ".organelleverse"))
    return home / "l5" / "providers.json"


class _TrustedProvider:
    """In-document shape of one trust record (not a public model)."""

    __slots__ = (
        "declaration_digest",
        "distribution_name",
        "distribution_record_digest",
        "distribution_version",
        "entry_point_locator",
        "provider_id",
    )

    def __init__(self, **kwargs: str) -> None:
        for s in self.__slots__:
            setattr(self, s, kwargs[s])

    def to_dict(self) -> dict[str, str]:
        return {s: getattr(self, s) for s in self.__slots__}

    def matches_identity(self, candidate: ProviderCandidate) -> bool:
        return (
            self.distribution_name == candidate.distribution_name
            and self.distribution_version == candidate.distribution_version
            and self.entry_point_locator == candidate.entry_point_locator
            and self.distribution_record_digest == candidate.distribution_record_digest
        )


class ComputeProviderTrustStore:
    """Owner-controlled authorization store for compute providers.

    A trust record says: "I have reviewed the installed identity
    (distribution+version+locator+RECORD digest) of this provider and the closed
    declaration it produced, and I authorize loading it." Any later change to
    that static identity invalidates the record.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        factory_loader: ProviderFactoryLoader | None = None,
    ) -> None:
        self.path = path or _default_path()
        self._loader: Callable[[ProviderCandidate], ComputeProviderSpec] = (
            factory_loader or default_factory_loader
        )

    # -- document I/O ------------------------------------------------------

    def _read(self) -> dict[str, _TrustedProvider]:
        if not self.path.exists():
            return {}
        if os.name != "nt" and self.path.stat().st_mode & 0o077:
            raise OrganellePermissionError(
                code="compute.trust_store_unsafe",
                message="compute trust store is group- or world-accessible",
                details={"path": str(self.path), "mode": oct(self.path.stat().st_mode & 0o777)},
            )
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise OrganellePermissionError(
                code="compute.trust_store_invalid",
                message="compute trust store is unreadable or invalid",
                details={"path": str(self.path), "reason": str(error)},
            ) from error
        out: dict[str, _TrustedProvider] = {}
        for record in payload.get("trusted", []):
            out[record["provider_id"]] = _TrustedProvider(**record)
        return out

    def _write(self, records: dict[str, _TrustedProvider]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(
                {"schema": _SCHEMA_VERSION,
                 "trusted": [records[k].to_dict() for k in sorted(records)]},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                if os.name != "nt":
                    os.chmod(temporary, 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            if os.name != "nt":
                os.chmod(self.path, 0o600)
        finally:
            if temporary.exists():
                temporary.unlink()

    # -- the four-gate lifecycle ------------------------------------------

    def enable(self, candidate: ProviderCandidate) -> EnabledProvider:
        """Authorize and load a provider.

        First authorizes by loading the factory, validating its closed
        declaration, and storing the declaration digest bound to the candidate's
        static installed identity. A later change to any of those invalidates
        trust until the provider is re-enabled.
        """
        with _LOCK:
            declaration = self._loader(candidate)
            declaration_digest = canonical_declaration_digest(declaration)
            record = _TrustedProvider(
                provider_id=candidate.provider_id,
                distribution_name=candidate.distribution_name,
                distribution_version=candidate.distribution_version,
                entry_point_locator=candidate.entry_point_locator,
                distribution_record_digest=candidate.distribution_record_digest,
                declaration_digest=declaration_digest,
            )
            records = self._read()
            records[candidate.provider_id] = record
            self._write(records)
            return EnabledProvider(
                candidate=candidate,
                declaration=declaration,
                declaration_digest=declaration_digest,
            )

    def is_current(self, candidate: ProviderCandidate) -> bool:
        """True iff a trust record exists and still matches the static identity."""
        with _LOCK:
            record = self._read().get(candidate.provider_id)
        return record is not None and record.matches_identity(candidate)

    def declaration_digest_for(self, provider_id: str) -> str | None:
        """The trusted declaration digest, or ``None`` if untrusted."""
        with _LOCK:
            record = self._read().get(provider_id)
        return record.declaration_digest if record is not None else None

    def require_trusted(self, candidate: ProviderCandidate) -> None:
        """Raise unless the candidate is currently trusted."""
        if not self.is_current(candidate):
            raise OrganellePermissionError(
                code="compute.provider_untrusted",
                message=(
                    "compute provider is installed but not explicitly trusted for "
                    "the current installed identity"
                ),
                details={"provider_id": candidate.provider_id},
            )

    def require_declaration_current(
        self, candidate: ProviderCandidate, declaration_digest: str
    ) -> None:
        """Raise if the loaded declaration no longer matches the trusted digest."""
        trusted = self.declaration_digest_for(candidate.provider_id)
        if trusted is None or trusted != declaration_digest:
            raise OrganellePermissionError(
                code="compute.provider_trust_stale",
                message=(
                    "the provider's declaration changed after trust was recorded; "
                    "review and re-enable it"
                ),
                details={"provider_id": candidate.provider_id},
            )

    def forget(self, provider_id: str) -> bool:
        """Remove the trust record for a provider. Returns whether one existed."""
        with _LOCK:
            records = self._read()
            existed = records.pop(provider_id, None) is not None
            if existed:
                self._write(records)
            return existed

    def trusted_provider_ids(self) -> tuple[str, ...]:
        with _LOCK:
            return tuple(sorted(self._read()))
