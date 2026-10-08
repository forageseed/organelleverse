from __future__ import annotations

import pytest
from pydantic import ValidationError

from organelleverse.compute.artifacts import (
    ArtifactVerificationReport,
    RemoteArtifact,
    RemoteArtifactManifest,
    StagedArtifact,
)

_HEX = "a" * 64


def _staged(object_id="in/1") -> StagedArtifact:
    return StagedArtifact(
        object_id=object_id, sha256=_HEX, size_bytes=10,
        transport="wsl_content_cache", locator=f"sha256/aa/{_HEX}.10",
    )


def _remote(object_id="out/1") -> RemoteArtifact:
    return RemoteArtifact(
        object_id=object_id, kind="genome", format="fasta", media_type="text/fasta",
        sha256=_HEX, size_bytes=10, transport="wsl_content_cache",
        locator=f"sha256/aa/{_HEX}.10",
    )


def test_staged_and_remote_are_closed_and_pattern_checked():
    with pytest.raises(ValidationError):
        StagedArtifact(object_id="x", sha256="bad", size_bytes=1, transport="t", locator="bad")
    with pytest.raises(ValidationError):
        StagedArtifact(  # extra field
            object_id="x", sha256=_HEX, size_bytes=1, transport="t",
            locator=f"sha256/aa/{_HEX}.1", extra=1,
        )


def test_verification_status_must_match_observation():
    # matching hashes but status=mismatch -> invalid
    with pytest.raises(ValidationError):
        ArtifactVerificationReport(
            expected_sha256=_HEX, expected_size_bytes=10,
            observed_sha256=_HEX, observed_size_bytes=10, status="mismatch",
        )
    # mismatched hashes but status=verified -> invalid
    with pytest.raises(ValidationError):
        ArtifactVerificationReport(
            expected_sha256=_HEX, expected_size_bytes=10,
            observed_sha256="b" * 64, observed_size_bytes=10, status="verified",
        )
    # consistent -> ok
    ArtifactVerificationReport(
        expected_sha256=_HEX, expected_size_bytes=10,
        observed_sha256=_HEX, observed_size_bytes=10, status="verified",
    )


def test_manifest_id_must_cover_preceding_fields():
    base = dict(
        provider_run_id="run1", operation_id="assembly.assemble",
        target_digest=f"sha256:{_HEX}", prepared_environment_digest=f"sha256:{_HEX}",
        result_snapshot={"ok": True}, artifacts=(_remote(),),
    )
    # correct manifest_id
    import hashlib
    import json
    payload = {
        "provider_run_id": "run1", "operation_id": "assembly.assemble",
        "target_digest": f"sha256:{_HEX}", "prepared_environment_digest": f"sha256:{_HEX}",
        "result_snapshot": {"ok": True},
        "artifacts": [_remote().model_dump(mode="json")],
    }
    digest = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    RemoteArtifactManifest(manifest_id=digest, **base)

    # wrong manifest_id -> invalid
    with pytest.raises(ValidationError):
        RemoteArtifactManifest(manifest_id=f"sha256:{'0'*64}", **base)
