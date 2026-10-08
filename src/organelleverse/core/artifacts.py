"""Portable, content-addressed artifact references."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .base import StrictFrozenModel
from .errors import OrganelleInputError


class ArtifactRef(StrictFrozenModel):
    """Reference one local artifact by semantic kind and content hash."""

    schema_version: Literal["organelleverse.artifact.v1"] = "organelleverse.artifact.v1"
    kind: str = Field(min_length=2, pattern=r"^[a-z][a-z0-9_]*$")
    uri: str = Field(min_length=1)
    format: str = Field(min_length=1)
    media_type: str = "application/octet-stream"
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    validated: bool = False

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "format": self.format,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        kind: str,
        format: str,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        candidate = Path(path)
        if not candidate.is_file():
            raise OrganelleInputError(
                code="input.missing_artifact",
                message=f"Artifact does not exist: {candidate}",
                details={"path": str(candidate)},
            )
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return cls(
            kind=kind,
            uri=str(candidate),
            format=format,
            media_type=media_type,
            sha256=digest.hexdigest(),
            size_bytes=candidate.stat().st_size,
            validated=True,
        )

    def resolve(self, base_dir: Path | None = None) -> Path:
        candidate = Path(self.uri)
        if candidate.is_absolute() or base_dir is None:
            return candidate
        return base_dir / candidate
