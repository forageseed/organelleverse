"""Deterministic content identity for capability bundles and fixtures."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from organelleverse.core.errors import OrganelleContractError

_READ_CHUNK = 1024 * 1024


def _unsafe_path(path: Path, root: Path) -> OrganelleContractError:
    return OrganelleContractError(
        code="capability.bundle_path_unsafe",
        message=f"capability bundle contains an unsafe path: {path}",
        details={"bundle_root": str(root), "path": str(path)},
    )


def _files(root: Path) -> tuple[Path, ...]:
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise OrganelleContractError(
            code="capability.bundle_read_failed",
            message=f"capability bundle root could not be read: {root}",
            details={"bundle_root": str(root), "reason": str(error)},
        ) from error
    if not resolved_root.is_dir():
        raise OrganelleContractError(
            code="capability.bundle_read_failed",
            message=f"capability bundle root is not a directory: {root}",
            details={"bundle_root": str(root)},
        )
    found: list[Path] = []
    for candidate in sorted(resolved_root.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            raise _unsafe_path(candidate, resolved_root)
        if not candidate.is_file():
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise OrganelleContractError(
                code="capability.bundle_read_failed",
                message=f"capability bundle file could not be resolved: {candidate}",
                details={"bundle_root": str(resolved_root), "path": str(candidate)},
            ) from error
        if not resolved.is_relative_to(resolved_root):
            raise _unsafe_path(candidate, resolved_root)
        found.append(resolved)
    return tuple(found)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
                digest.update(chunk)
    except OSError as error:
        raise OrganelleContractError(
            code="capability.bundle_read_failed",
            message=f"capability bundle file could not be read: {path}",
            details={"path": str(path), "reason": str(error)},
        ) from error
    return digest.hexdigest()


def hash_file(path: Path) -> str:
    """Return one regular file's content identity."""
    if path.is_symlink() or not path.is_file():
        raise OrganelleContractError(
            code="capability.bundle_path_unsafe",
            message=f"content identity requires a regular, non-symlink file: {path}",
            details={"path": str(path)},
        )
    return f"sha256:{_file_digest(path.resolve(strict=True))}"


def _hash_leaves(entries: Iterable[tuple[str, str]]) -> str:
    leaves = [[name, digest] for name, digest in entries]
    leaves.sort(key=lambda entry: entry[0])
    canonical = json.dumps(
        leaves,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def hash_entries(entries: Iterable[tuple[str, bytes]]) -> str:
    """Hash named byte entries using the canonical sorted Merkle-list encoding."""
    return _hash_leaves((name, hashlib.sha256(content).hexdigest()) for name, content in entries)


def hash_tree(root: Path) -> str:
    """Hash every file below *root* using the specification's canonical Merkle list."""
    resolved_root = root.resolve(strict=True)
    return _hash_leaves(
        (path.relative_to(resolved_root).as_posix(), _file_digest(path))
        for path in _files(resolved_root)
    )


def hash_bundle(bundle_root: Path) -> str:
    """Return the identity of all files in one capability bundle."""
    return hash_tree(bundle_root)


def hash_fixture_dataset(bundle_root: Path) -> str:
    """Return the identity of ``fixtures/**`` or the stable empty-tree hash."""
    fixture_root = bundle_root / "fixtures"
    if fixture_root.is_dir():
        return hash_tree(fixture_root)
    canonical = json.dumps(
        [], sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


__all__ = ["hash_bundle", "hash_entries", "hash_file", "hash_fixture_dataset", "hash_tree"]
