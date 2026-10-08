"""Capability-facing adapter over the existing content-addressed L2 asset layer."""

from __future__ import annotations

import re
import threading
from typing import Protocol

from pydantic import Field

import organelleverse.assets as l2_assets
from organelleverse.core.errors import OrganelleDependencyError, OrganelleError
from organelleverse.operations.spec import StrictSpecModel

_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*$")
_HASH_PATTERN = r"^sha256:[0-9a-f]{64}$"


class AssetIdentity(StrictSpecModel):
    """The only L2 fact scientific admission consumes."""

    locator: str = Field(min_length=1)
    content_hash: str = Field(pattern=_HASH_PATTERN)


class AssetResolver(Protocol):
    """A scheme-specific content identity resolver."""

    def content_hash(self, locator: str) -> str | None: ...


class _PackagedAssetAdapter:
    def content_hash(self, locator: str) -> str | None:
        return l2_assets.content_hash(locator)


_RESOLVERS: dict[str, AssetResolver] = {"ov-asset": _PackagedAssetAdapter()}
_LOCK = threading.RLock()


def register_asset_resolver(scheme: str, resolver: AssetResolver) -> None:
    """Register or replace one explicit URI-scheme resolver."""
    if _SCHEME_RE.fullmatch(scheme) is None:
        raise ValueError(f"invalid asset resolver scheme: {scheme!r}")
    with _LOCK:
        _RESOLVERS[scheme] = resolver


def _failure(
    locator: str, reason: str, cause: OrganelleError | None = None
) -> OrganelleDependencyError:
    cause_details: object = None if cause is None else cause.as_dict()
    return OrganelleDependencyError(
        code="capability.asset_unresolvable",
        message=f"asset identity could not be established for {locator}",
        details={
            "locator": locator,
            "reason": reason,
            "cause": cause_details,
        },
    )


def resolve_asset_identity(locator: str) -> AssetIdentity:
    """Resolve and verify one asset identity, failing closed on every gap."""
    scheme, separator, _ = locator.partition(":")
    if not separator or _SCHEME_RE.fullmatch(scheme) is None:
        raise _failure(locator, "asset.unresolvable")
    with _LOCK:
        resolver = _RESOLVERS.get(scheme)
    if resolver is None:
        raise _failure(locator, "asset.unresolvable")
    try:
        content_hash = resolver.content_hash(locator)
    except OrganelleError as error:
        reason = (
            "asset.content_mismatch"
            if error.code == "asset.content_mismatch"
            else "asset.unresolvable"
        )
        raise _failure(locator, reason, error) from error
    if content_hash is None:
        raise _failure(locator, "asset.unresolvable")
    try:
        return AssetIdentity(locator=locator, content_hash=content_hash)
    except ValueError as error:
        raise _failure(locator, "asset.content_mismatch") from error


__all__ = [
    "AssetIdentity",
    "AssetResolver",
    "register_asset_resolver",
    "resolve_asset_identity",
]
