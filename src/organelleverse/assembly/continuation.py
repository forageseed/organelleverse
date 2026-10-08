"""Canonical, content-addressed directory manifests for continuation operations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.operations.spec import StrictSpecModel

__all__ = [
    "DirectoryManifest",
    "DirectoryManifestFile",
    "materialize_directory_manifest",
]


class DirectoryManifestFile(StrictSpecModel):
    relative_path: str = Field(min_length=1)
    artifact_role: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_safe_relative_path(self) -> DirectoryManifestFile:
        raw_parts = self.relative_path.split("/")
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or "\\" in self.relative_path
            or "\x00" in self.relative_path
            or any(part in {"", ".", ".."} for part in raw_parts)
        ):
            raise ValueError("directory manifest path must be a safe POSIX relative path")
        return self


class DirectoryManifest(StrictSpecModel):
    schema_version: Literal["organelleverse.directory-manifest.v1"] = (
        "organelleverse.directory-manifest.v1"
    )
    role: Literal["pmat_subsample", "pmat_assembly_result"]
    files: tuple[DirectoryManifestFile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_entries(self) -> DirectoryManifest:
        paths = [item.relative_path for item in self.files]
        roles = [item.artifact_role for item in self.files]
        if len(paths) != len(set(paths)) or len(roles) != len(set(roles)):
            raise ValueError("directory manifest paths and artifact roles must be unique")
        return self

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


def materialize_directory_manifest(
    manifest: DirectoryManifest,
    artifacts: Mapping[str, ArtifactRef],
    destination: Path,
) -> None:
    """Verify every file, then atomically restore its declared directory tree."""
    verified: list[tuple[DirectoryManifestFile, Path]] = []
    for item in manifest.files:
        artifact = artifacts.get(item.artifact_role)
        if artifact is None:
            raise _input_error(
                f"directory manifest artifact role {item.artifact_role!r} is missing"
            )
        source = Path(artifact.uri)
        if source.is_symlink() or not source.is_file():
            raise _input_error(f"directory manifest source {item.artifact_role!r} is unsafe")
        actual = _hash_file(source)
        if artifact.sha256 != item.sha256 or actual != item.sha256:
            raise _input_error(f"directory manifest hash mismatch for {item.artifact_role!r}")
        verified.append((item, source))

    if destination.exists():
        raise _input_error(f"directory manifest destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for item, source in verified:
            target = temporary / PurePosixPath(item.relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            target.chmod(0o600)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_error(message: str) -> OrganelleInputError:
    return OrganelleInputError(code="assembly.invalid_continuation", message=message)
