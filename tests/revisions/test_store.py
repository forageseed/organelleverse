"""Immutable, append-only common artifact revision store."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleContractError
from organelleverse.revisions import (
    ArtifactRevision,
    ArtifactRevisionStore,
    EvidenceLink,
)

_SHA = "a" * 64
_SHA_OTHER = "b" * 64


def _ref(sha: str = _SHA) -> ArtifactRef:
    return ArtifactRef(
        kind="report",
        uri="/tmp/host-artifacts/report.md",
        format="markdown",
        media_type="text/markdown",
        sha256=sha,
        size_bytes=12,
    )


def _revision(
    revision_id: str,
    artifact_id: str,
    *,
    parent: str | None = None,
    sha: str = _SHA,
    status: str = "available",
    diagnostics: tuple[str, ...] = (),
    base_time: datetime | None = None,
) -> ArtifactRevision:
    base = base_time or datetime(2026, 8, 14, tzinfo=UTC)
    return ArtifactRevision(
        revision_id=revision_id,
        artifact_id=artifact_id,
        parent_revision_id=parent,
        kind="markdown",
        author="agent",
        content_ref=_ref(sha),
        status=status,  # type: ignore[arg-type]
        diagnostics=diagnostics,
        evidence_links=(
            EvidenceLink(
                evidence_id="ev-1", kind="run", target_id="run-1", label="durable run"
            ),
        ),
        created_at=base + timedelta(seconds=1),
    )


def test_append_and_retrieve_preserves_parents(tmp_path: Path) -> None:
    store = ArtifactRevisionStore(tmp_path)
    store.register(_revision("rev-1", "art-1"))
    store.register(_revision("rev-2", "art-1", parent="rev-1", sha=_SHA_OTHER))

    assert store.get("rev-1").parent_revision_id is None
    assert store.get("rev-2").parent_revision_id == "rev-1"
    assert store.get("rev-1").content_ref.sha256 == _SHA
    assert store.get("rev-2").content_ref.sha256 == _SHA_OTHER


def test_revision_models_are_frozen_and_closed() -> None:
    revision = _revision("rev-1", "art-1")
    with pytest.raises(ValidationError):
        revision.diagnostics = ("mutated",)  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ArtifactRevision.model_validate({**revision.model_dump(), "extra": "no"})


def test_store_exposes_no_mutation_api(tmp_path: Path) -> None:
    store = ArtifactRevisionStore(tmp_path)
    for forbidden in ("update", "delete", "patch", "replace", "remove", "mutate"):
        assert not hasattr(store, forbidden)


def test_duplicate_revision_id_is_rejected(tmp_path: Path) -> None:
    store = ArtifactRevisionStore(tmp_path)
    store.register(_revision("rev-1", "art-1"))
    with pytest.raises(OrganelleContractError) as info:
        store.register(_revision("rev-1", "art-1", sha=_SHA_OTHER))
    assert info.value.code == "revision.duplicate_id"


def test_parent_from_other_artifact_is_rejected(tmp_path: Path) -> None:
    store = ArtifactRevisionStore(tmp_path)
    store.register(_revision("rev-1", "art-1"))
    with pytest.raises(OrganelleContractError) as info:
        store.register(_revision("rev-2", "art-2", parent="rev-1"))
    assert info.value.code == "revision.parent_mismatch"


def test_unknown_parent_is_rejected(tmp_path: Path) -> None:
    store = ArtifactRevisionStore(tmp_path)
    with pytest.raises(OrganelleContractError) as info:
        store.register(_revision("rev-2", "art-1", parent="missing"))
    assert info.value.code == "revision.unknown_parent"


def test_unknown_revision_raises_structured_error(tmp_path: Path) -> None:
    store = ArtifactRevisionStore(tmp_path)
    with pytest.raises(OrganelleContractError) as info:
        store.get("missing")
    assert info.value.code == "revision.unknown_revision"


def test_lineage_orders_oldest_first_and_finds_common_ancestor(tmp_path: Path) -> None:
    store = ArtifactRevisionStore(tmp_path)
    base = datetime(2026, 8, 14, tzinfo=UTC)
    store.register(_revision("rev-1", "art-1", base_time=base))
    store.register(
        _revision("rev-2a", "art-1", parent="rev-1", base_time=base + timedelta(seconds=10))
    )
    store.register(
        _revision("rev-2b", "art-1", parent="rev-1", base_time=base + timedelta(seconds=20))
    )

    lineage = store.lineage("art-1")
    assert [r.revision_id for r in lineage] == ["rev-1", "rev-2a", "rev-2b"]
    assert store.common_ancestor("rev-2a", "rev-2b").revision_id == "rev-1"
    assert store.common_ancestor("rev-1", "rev-2a").revision_id == "rev-1"


def test_broken_revision_stays_addressable_with_diagnostics(tmp_path: Path) -> None:
    store = ArtifactRevisionStore(tmp_path)
    store.register(_revision("rev-1", "art-1"))
    store.register(
        _revision(
            "rev-2",
            "art-1",
            parent="rev-1",
            status="broken",
            diagnostics=("artifact content unavailable",),
        )
    )

    broken = store.get("rev-2")
    assert broken.status == "broken"
    assert broken.diagnostics == ("artifact content unavailable",)
    assert [r.revision_id for r in store.lineage("art-1")] == ["rev-1", "rev-2"]


def test_persistence_restores_after_restart(tmp_path: Path) -> None:
    store = ArtifactRevisionStore(tmp_path)
    store.register(_revision("rev-1", "art-1"))
    store.register(_revision("rev-2", "art-1", parent="rev-1", sha=_SHA_OTHER))

    reopened = ArtifactRevisionStore(tmp_path)
    assert reopened.get("rev-2").parent_revision_id == "rev-1"
    files = sorted(p.name for p in tmp_path.glob("*.json"))
    assert files == ["rev-1.json", "rev-2.json"]
    record = json.loads((tmp_path / "rev-1.json").read_text(encoding="utf-8"))
    assert {"revision_id", "artifact_id", "parent_revision_id", "content_ref"} <= set(record)


def test_corrupt_history_fails_closed(tmp_path: Path) -> None:
    store = ArtifactRevisionStore(tmp_path)
    store.register(_revision("rev-1", "art-1"))
    (tmp_path / "rev-1.json").write_text("{ not canonical json", encoding="utf-8")

    with pytest.raises(OrganelleContractError) as info:
        ArtifactRevisionStore(tmp_path)
    assert info.value.code == "revision.history_invalid"
