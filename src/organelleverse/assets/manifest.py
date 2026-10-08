"""Declared identity, content hash, and provenance for one knowledge asset (L2).

The manifest is a file named ``asset.toml`` at the root of the asset directory.
It *declares* what the asset is and what its content hash should be; the hash is
*recomputed* on resolution and compared, so a manifest cannot vouch for content
it does not match.

The manifest file is excluded from its own content hash — otherwise declaring
the hash would change it.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field

from organelleverse.core.errors import OrganelleInputError

from .identity import AssetLocator

__all__ = [
    "ASSET_MANIFEST_FILENAME",
    "ASSET_MANIFEST_SCHEMA_VERSION",
    "ASSET_MANIFEST_SUFFIX",
    "AssetContent",
    "AssetManifest",
    "AssetProvenance",
    "compute_content",
    "manifest_path_for",
]

ASSET_MANIFEST_FILENAME = "asset.toml"
ASSET_MANIFEST_SUFFIX = ".asset.toml"
ASSET_MANIFEST_SCHEMA_VERSION = "organelleverse.asset.v1"

_EXCLUDED_DIRS = frozenset({"__pycache__"})
# Code is never part of an asset: a data directory that is also an importable
# package still ships only its data under this layer's identity. What the code
# does with the asset is the method layer's business, and its version travels
# with the package, not with the reference set.
_EXCLUDED_SUFFIXES = frozenset({".py", ".pyc", ".pyi"})
_READ_CHUNK = 1024 * 1024


def manifest_path_for(root: Path) -> Path:
    """Return where an asset root's manifest lives.

    A directory asset carries ``asset.toml`` inside it. A single-file asset —
    a model checkpoint, one reference database — carries a sibling
    ``<name>.asset.toml``, because a file cannot contain its own manifest.
    """
    return (
        root / ASSET_MANIFEST_FILENAME
        if root.is_dir()
        else root.with_name(root.name + ASSET_MANIFEST_SUFFIX)
    )


class AssetProvenance(BaseModel):
    """Where an asset came from and how to rebuild it.

    These fields carry the information that was previously written by hand into
    ``PROVENANCE.md`` files, in a form a contract and a verification record can
    reference.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = ""
    license: str = ""
    rebuild: str = ""
    derived_from: tuple[str, ...] = ()
    notes: str = ""


class AssetContent(BaseModel):
    """The declared content identity of an asset directory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    files: int = Field(ge=0)
    bytes: int = Field(ge=0)
    merkle: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class AssetManifest(BaseModel):
    """One asset's declared identity, content hash, and provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["organelleverse.asset.v1"] = ASSET_MANIFEST_SCHEMA_VERSION
    locator: AssetLocator
    summary: str = ""
    content: AssetContent
    provenance: AssetProvenance = AssetProvenance()

    @classmethod
    def from_directory(cls, root: Path) -> Self:
        """Read and validate the ``asset.toml`` at the root of an asset.

        Arguments:
            root: The asset directory containing ``asset.toml``.

        Returns:
            The validated manifest. The content hash is *not* verified here; use
            :func:`organelleverse.assets.content_hash` for that.

        Raises:
            OrganelleInputError: The manifest is missing or malformed.
        """
        path = manifest_path_for(root)
        if not path.is_file():
            raise OrganelleInputError(
                code="asset.manifest_missing",
                message=f"No asset manifest for: {root}",
                details={"root": str(root), "expected_manifest": str(path)},
            )
        try:
            raw: Mapping[str, object] = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
            raise OrganelleInputError(
                code="asset.manifest_invalid",
                message=f"Unreadable asset manifest: {path}",
                details={"path": str(path), "error": str(error)},
            ) from error

        asset = _table(raw, "asset")
        if not asset:
            raise OrganelleInputError(
                code="asset.manifest_invalid",
                message=f"Asset manifest has no [asset] table: {path}",
                details={"path": str(path)},
            )
        declared_schema = _text(raw, "schema", ASSET_MANIFEST_SCHEMA_VERSION)
        if declared_schema != ASSET_MANIFEST_SCHEMA_VERSION:
            raise OrganelleInputError(
                code="asset.manifest_unsupported_schema",
                message=f"Unsupported asset manifest schema: {declared_schema!r}",
                details={
                    "path": str(path),
                    "declared": declared_schema,
                    "supported": ASSET_MANIFEST_SCHEMA_VERSION,
                },
            )
        try:
            return cls(
                locator=AssetLocator(
                    namespace=_text(asset, "namespace"),
                    name=_text(asset, "name"),
                    version=_text(asset, "version"),
                ),
                summary=_text(asset, "summary"),
                content=AssetContent.model_validate(_table(raw, "content")),
                provenance=AssetProvenance.model_validate(_table(raw, "provenance")),
            )
        except (TypeError, ValueError) as error:
            raise OrganelleInputError(
                code="asset.manifest_invalid",
                message=f"Invalid asset manifest: {path}",
                details={"path": str(path), "error": str(error)},
            ) from error


def _table(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return one nested TOML table, empty when absent or not a table."""
    value = mapping.get(key)
    return cast("Mapping[str, object]", value) if isinstance(value, Mapping) else {}


def _text(mapping: Mapping[str, object], key: str, default: str = "") -> str:
    """Return one string field, falling back when absent or not a string."""
    value = mapping.get(key)
    return value if isinstance(value, str) else default


def _is_nested_asset(directory: Path, root: Path) -> bool:
    """True when a directory *inside* this asset declares an asset of its own.

    An asset's content stops at a nested asset boundary, so versioning a model
    directory does not silently re-version the reference set that contains it.

    The containment check is load-bearing: without it, computing the content of
    a nested asset would see its own container's manifest overhead and exclude
    every one of its files.
    """
    if directory == root or not directory.is_relative_to(root):
        return False
    return (directory / ASSET_MANIFEST_FILENAME).is_file()


def _is_nested_file_asset(candidate: Path, root: Path) -> bool:
    """True when a file inside this asset is itself a single-file asset.

    A sidecar manifest makes its file a separately versioned asset, so the
    container must stop at it exactly as it stops at a nested directory —
    otherwise swapping a model checkpoint would also move the identity of the
    reference set that happens to sit beside it.
    """
    if candidate.parent == root and candidate.name == ASSET_MANIFEST_FILENAME:
        return False
    return (
        candidate != root and candidate.with_name(candidate.name + ASSET_MANIFEST_SUFFIX).is_file()
    )


def _asset_files(root: Path) -> list[Path]:
    """Every hashed file under an asset root.

    Excluded: the manifest itself, caches, and anything belonging to a nested
    asset.
    """
    files: list[Path] = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root)
        if candidate.name == ASSET_MANIFEST_FILENAME and candidate.parent == root:
            continue
        if candidate.name.endswith(ASSET_MANIFEST_SUFFIX):
            continue
        if candidate.suffix in _EXCLUDED_SUFFIXES:
            continue
        if _EXCLUDED_DIRS.intersection(relative.parts):
            continue
        if any(_is_nested_asset(parent, root) for parent in candidate.parents):
            continue
        if _is_nested_file_asset(candidate, root):
            continue
        files.append(candidate)
    return files


def compute_content(root: Path) -> AssetContent:
    """Compute an asset directory's content identity.

    The hash is a Merkle over the canonical JSON of ``[[relative_posix_path,
    sha256_hex], ...]`` sorted by path, so it is stable across filesystems and
    changes if any file's name or bytes change.

    Arguments:
        root: The asset directory to hash.

    Returns:
        The computed file count, total byte size, and ``sha256:`` Merkle hash.

    Raises:
        OrganelleInputError: The directory does not exist.
    """
    if root.is_file():
        paths = [root]
    elif root.is_dir():
        paths = _asset_files(root)
    else:
        raise OrganelleInputError(
            code="asset.root_missing",
            message=f"Asset root does not exist: {root}",
            details={"root": str(root)},
        )
    base = root.parent if root.is_file() else root
    entries: list[list[str]] = []
    total_bytes = 0
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
                digest.update(chunk)
        entries.append([path.relative_to(base).as_posix(), digest.hexdigest()])
        total_bytes += path.stat().st_size
    entries.sort(key=lambda entry: entry[0])
    canonical = json.dumps(entries, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return AssetContent(
        files=len(entries),
        bytes=total_bytes,
        merkle=f"sha256:{hashlib.sha256(canonical).hexdigest()}",
    )
