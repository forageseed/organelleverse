"""Resolve an asset locator to a concrete directory, and verify its content (L2).

Resolution is deliberately pluggable: the built-in resolver covers assets that
ship inside the installed package, and callers may register additional resolvers
for user-supplied, site-local, or downloaded assets. A locator's meaning never
depends on which resolver answered — the content hash does.

Every function here performs file reads only. Nothing in this module imports a
scientific module, installs anything, or spawns a subprocess, so it is safe to
call during discovery.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol

from organelleverse.core.errors import OrganelleDependencyError, OrganelleInputError

from .identity import AssetLocator
from .manifest import (
    ASSET_MANIFEST_FILENAME,
    ASSET_MANIFEST_SUFFIX,
    AssetManifest,
    compute_content,
)

__all__ = [
    "AssetResolver",
    "PackagedAssetResolver",
    "content_hash",
    "describe",
    "list_assets",
    "list_resolvers",
    "locate",
    "register_resolver",
    "unregister_resolver",
    "verify",
]


class AssetResolver(Protocol):
    """Anything that can turn a locator into a directory on this machine."""

    @property
    def source(self) -> str:
        """A short, stable label recorded in provenance (e.g. ``"packaged"``)."""
        ...

    def index(self) -> dict[str, Path]:
        """Return every locator this resolver can currently answer."""
        ...


class PackagedAssetResolver:
    """Assets shipped inside the installed ``organelleverse`` package.

    Discovers assets by walking the package for ``asset.toml`` files, so adding
    an asset to the tree requires no registration code. The index is built once
    and cached; call :meth:`invalidate` after writing a new manifest.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or Path(__file__).resolve().parent.parent
        self._index: dict[str, Path] | None = None
        self._lock = threading.Lock()

    @property
    def source(self) -> str:
        return "packaged"

    def invalidate(self) -> None:
        """Drop the cached index so the next query rescans the tree."""
        with self._lock:
            self._index = None

    def index(self) -> dict[str, Path]:
        with self._lock:
            if self._index is None:
                self._index = self._scan()
            return dict(self._index)

    def _scan(self) -> dict[str, Path]:
        """Index every asset in the tree: directory manifests and file sidecars."""
        manifests = {
            *self._root.rglob(ASSET_MANIFEST_FILENAME),
            *self._root.rglob(f"*{ASSET_MANIFEST_SUFFIX}"),
        }
        found: dict[str, Path] = {}
        for manifest_path in sorted(manifests):
            if manifest_path.name == ASSET_MANIFEST_FILENAME:
                root = manifest_path.parent
            else:
                root = manifest_path.with_name(manifest_path.name[: -len(ASSET_MANIFEST_SUFFIX)])
            found[str(AssetManifest.from_directory(root).locator)] = root
        return found


_RESOLVERS: list[AssetResolver] = [PackagedAssetResolver()]
_RESOLVER_LOCK = threading.Lock()


def register_resolver(resolver: AssetResolver) -> None:
    """Add a resolver for assets that do not ship inside the package.

    Arguments:
        resolver: The resolver to consult after the built-in one.
    """
    with _RESOLVER_LOCK:
        _RESOLVERS.append(resolver)


def unregister_resolver(resolver: AssetResolver) -> bool:
    """Remove a previously registered resolver.

    Arguments:
        resolver: The resolver to remove.

    Returns:
        True when the resolver was registered and has been removed.
    """
    with _RESOLVER_LOCK:
        if resolver not in _RESOLVERS:
            return False
        _RESOLVERS.remove(resolver)
        return True


def list_resolvers() -> tuple[AssetResolver, ...]:
    """Return the registered resolvers, built-in first."""
    with _RESOLVER_LOCK:
        return tuple(_RESOLVERS)


def _coerce(locator: AssetLocator | str) -> AssetLocator:
    return locator if isinstance(locator, AssetLocator) else AssetLocator.parse(locator)


def locate(locator: AssetLocator | str) -> Path:
    """Resolve one locator to the directory holding the asset.

    Arguments:
        locator: The asset locator, as a string or parsed value.

    Returns:
        The asset's root directory.

    Raises:
        OrganelleDependencyError: No registered resolver can answer the locator,
            or two resolvers answer it with different directories.
    """
    wanted = str(_coerce(locator))
    hits: list[tuple[str, Path]] = []
    for resolver in list_resolvers():
        found = resolver.index().get(wanted)
        if found is not None:
            hits.append((resolver.source, found))
    if not hits:
        raise OrganelleDependencyError(
            code="asset.unresolvable",
            message=f"No registered resolver provides {wanted}",
            details={"locator": wanted, "resolvers": [r.source for r in list_resolvers()]},
        )
    distinct = {str(path) for _, path in hits}
    if len(distinct) > 1:
        raise OrganelleDependencyError(
            code="asset.conflict",
            message=f"Resolvers disagree on where {wanted} lives",
            details={
                "locator": wanted,
                "candidates": [{"source": source, "path": str(path)} for source, path in hits],
            },
        )
    return hits[0][1]


def describe(locator: AssetLocator | str) -> AssetManifest:
    """Return one asset's declared manifest without verifying its content.

    Arguments:
        locator: The asset locator.

    Returns:
        The asset's declared manifest.
    """
    return AssetManifest.from_directory(locate(locator))


def verify(locator: AssetLocator | str) -> AssetManifest:
    """Recompute an asset's content and check it against its declaration.

    Arguments:
        locator: The asset locator.

    Returns:
        The manifest, with the declared content confirmed to match what is on
        disk right now.

    Raises:
        OrganelleInputError: The bytes on disk do not match the declaration.
    """
    root = locate(locator)
    manifest = AssetManifest.from_directory(root)
    observed = compute_content(root)
    if observed != manifest.content:
        raise OrganelleInputError(
            code="asset.content_mismatch",
            message=f"Asset content does not match its manifest: {manifest.locator}",
            details={
                "locator": str(manifest.locator),
                "root": str(root),
                "declared": manifest.content.model_dump(),
                "observed": observed.model_dump(),
            },
        )
    return manifest


def content_hash(locator: AssetLocator | str) -> str:
    """Return the verified ``sha256:`` content hash of one asset.

    This is the value a contract's verification record binds, so that pulling a
    different reference library or model invalidates the record.

    Arguments:
        locator: The asset locator.

    Returns:
        The verified content hash.
    """
    return verify(locator).content.merkle


def list_assets() -> tuple[AssetManifest, ...]:
    """Return every asset any registered resolver can currently answer."""
    seen: dict[str, AssetManifest] = {}
    for resolver in list_resolvers():
        for key, root in resolver.index().items():
            seen.setdefault(key, AssetManifest.from_directory(root))
    return tuple(seen[key] for key in sorted(seen))
