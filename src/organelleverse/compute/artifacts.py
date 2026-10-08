"""The one artifact vocabulary every compute transport must use (spec §10).

MCP JSON is control plane, not bulk data plane. Artifacts are addressed by
SHA256 and size, never embedded (no base64 FASTQ/GFA). Target paths are
provider-generated, never model-supplied. This module defines the contract
types; concrete transports (WSL content cache, SSH SFTP via paramiko, shared
filesystem for Slurm) implement :class:`ArtifactTransport` against it in
L5-02 Task 3 / L5-03.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, JsonValue, model_validator

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.operations.spec import StrictSpecModel

__all__ = [
    "ArtifactTransport",
    "ArtifactVerificationReport",
    "RemoteArtifact",
    "RemoteArtifactManifest",
    "StagedArtifact",
]

_HEX64 = r"^[0-9a-f]{64}$"
_LOCATOR = r"^sha256/[0-9a-f]{2}/[0-9a-f]{64}\.[0-9]+$"
_RUN_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_OP_ID = r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$"
_TARGET_DIGEST = r"^sha256:[0-9a-f]{64}$"


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


class StagedArtifact(StrictSpecModel):
    """An input artifact staged to a target's content cache, addressed by content."""

    object_id: str = Field(min_length=1)
    sha256: str = Field(pattern=_HEX64)
    size_bytes: int = Field(ge=0)
    transport: str = Field(min_length=1)
    locator: str = Field(pattern=_LOCATOR)


class RemoteArtifact(StrictSpecModel):
    """An output artifact the worker produced, addressed by content (spec §10).

    The locator is provider-generated from SHA256 and size; caller/model paths
    never appear in it. The original Unicode filename is metadata only and is
    never a cache path.
    """

    object_id: str = Field(min_length=1)
    kind: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    format: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    sha256: str = Field(pattern=_HEX64)
    size_bytes: int = Field(ge=0)
    transport: str = Field(min_length=1)
    locator: str = Field(pattern=_LOCATOR)


class RemoteArtifactManifest(StrictSpecModel):
    """The worker's content-addressed output manifest (spec §9, §10).

    ``manifest_id`` is the canonical SHA256 of every preceding field (ordered
    artifact records included, ``manifest_id`` itself excluded). The worker
    writes it only after every output file has been hashed and sized.
    """

    provider_run_id: str = Field(pattern=_RUN_ID)
    operation_id: str = Field(pattern=_OP_ID)
    target_digest: str = Field(pattern=_TARGET_DIGEST)
    prepared_environment_digest: str = Field(pattern=_TARGET_DIGEST)
    result_snapshot: dict[str, JsonValue]
    artifacts: tuple[RemoteArtifact, ...]
    manifest_id: str = Field(pattern=_TARGET_DIGEST)

    @model_validator(mode="after")
    def _manifest_id_must_cover_preceding_fields(self) -> RemoteArtifactManifest:
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("manifest_id", None)
        expected = "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        if self.manifest_id != expected:
            raise ValueError(
                "manifest_id does not match the canonical digest of the manifest fields"
            )
        return self


class ArtifactVerificationReport(StrictSpecModel):
    """Result of comparing expected vs observed content identity (spec §9)."""

    expected_sha256: str = Field(pattern=_HEX64)
    expected_size_bytes: int = Field(ge=0)
    observed_sha256: str = Field(pattern=_HEX64)
    observed_size_bytes: int = Field(ge=0)
    status: Literal["verified", "mismatch"]

    @model_validator(mode="after")
    def _status_matches_observation(self) -> ArtifactVerificationReport:
        actually_verified = (
            self.expected_sha256 == self.observed_sha256
            and self.expected_size_bytes == self.observed_size_bytes
        )
        if actually_verified != (self.status == "verified"):
            raise ValueError(
                "ArtifactVerificationReport.status must equal 'verified' exactly when "
                "expected and observed sha256 and size both match"
            )
        return self


class ArtifactTransport(Protocol):
    """The bulk-data-plane contract (spec §10). Stage inputs, fetch outputs,
    verify content identity. Implemented per transport (WSL/SSH/shared-fs)."""

    def stage(self, artifact: ArtifactRef, target: object) -> StagedArtifact: ...
    def fetch(self, artifact: RemoteArtifact, destination: Path) -> ArtifactRef: ...
    def verify(self, artifact: StagedArtifact | RemoteArtifact) -> ArtifactVerificationReport: ...
