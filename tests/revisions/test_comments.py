"""Immutable typed comments anchored to exact artifact revisions."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleContractError
from organelleverse.revisions import (
    ArtifactRevision,
    ArtifactRevisionStore,
    CommentRecord,
    CommentSelection,
    CommentStore,
    ImageDimensions,
    NormalizedRegion,
    TableCellKey,
    TextRange,
)

_CREATED = datetime(2026, 8, 14, tzinfo=UTC)


def _ref(sha: str) -> ArtifactRef:
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
    *,
    kind: str = "markdown",
    parent: str | None = None,
    dimensions: ImageDimensions | None = None,
) -> ArtifactRevision:
    return ArtifactRevision(
        revision_id=revision_id,
        artifact_id="art-1",
        parent_revision_id=parent,
        kind=kind,  # type: ignore[arg-type]
        author="agent",
        content_ref=_ref("a" * 64 if revision_id == "rev-1" else "b" * 64),
        status="available",
        diagnostics=(),
        evidence_links=(),
        dimensions=dimensions,
        created_at=_CREATED,
    )


def _stores(tmp_path: Path) -> tuple[ArtifactRevisionStore, CommentStore]:
    revisions = ArtifactRevisionStore(tmp_path / "revisions")
    comments = CommentStore(tmp_path / "comments", revisions=revisions)
    return revisions, comments


def test_text_selection_resolves_against_text_revision(tmp_path: Path) -> None:
    revisions, comments = _stores(tmp_path)
    revisions.register(_revision("rev-1"))

    comment = comments.add(
        CommentRecord(
            comment_id=f"comment-{'c' * 32}",
            thread_id="thread-1",
            revision_id="rev-1",
            selection=CommentSelection(
                text_range=TextRange(unit="line", start=2, end=4),
                image_region=None,
                table_cell=None,
                whole_artifact=False,
                notebook_cell=None,
            ),
            body="tighten this claim",
            status="open",
            proposed_revision_id=None,
            created_at=_CREATED,
        )
    )

    assert comments.get(comment.comment_id).status == "open"
    assert [c.comment_id for c in comments.by_revision("rev-1")] == [comment.comment_id]
    assert [c.comment_id for c in comments.by_thread("thread-1")] == [comment.comment_id]


def test_whole_artifact_selection_is_kind_independent(tmp_path: Path) -> None:
    revisions, comments = _stores(tmp_path)
    revisions.register(_revision("rev-1"))

    comment = comments.add(
        CommentRecord(
            comment_id=f"comment-{'c' * 32}",
            thread_id="thread-1",
            revision_id="rev-1",
            selection=CommentSelection(
                text_range=None,
                image_region=None,
                table_cell=None,
                whole_artifact=True,
                notebook_cell=None,
            ),
            body="export this revision",
            status="open",
            proposed_revision_id=None,
            created_at=_CREATED,
        )
    )

    assert comment.selection.whole_artifact is True


def test_image_region_selection_against_image_revision(tmp_path: Path) -> None:
    revisions, comments = _stores(tmp_path)
    revisions.register(
        _revision(
            "rev-img",
            kind="image",
            dimensions=ImageDimensions(width=640, height=480),
        )
    )

    comment = comments.add(
        CommentRecord(
            comment_id=f"comment-{'c' * 32}",
            thread_id="thread-1",
            revision_id="rev-img",
            selection=CommentSelection(
                text_range=None,
                image_region=NormalizedRegion(x=0.1, y=0.1, width=0.5, height=0.4),
                table_cell=None,
                whole_artifact=False,
                notebook_cell=None,
            ),
            body="arrow tip is clipped here",
            status="open",
            proposed_revision_id=None,
            created_at=_CREATED,
        )
    )

    assert comment.selection.image_region is not None


def test_table_key_selection_against_table_revision(tmp_path: Path) -> None:
    revisions, comments = _stores(tmp_path)
    revisions.register(_revision("rev-table", kind="table"))

    comment = comments.add(
        CommentRecord(
            comment_id=f"comment-{'c' * 32}",
            thread_id="thread-1",
            revision_id="rev-table",
            selection=CommentSelection(
                text_range=None,
                image_region=None,
                table_cell=TableCellKey(row_key="row-3", column_key="gc-content"),
                whole_artifact=False,
                notebook_cell=None,
            ),
            body="this outlier needs a note",
            status="open",
            proposed_revision_id=None,
            created_at=_CREATED,
        )
    )

    assert comment.selection.table_cell is not None


def test_mismatched_selection_kind_is_rejected(tmp_path: Path) -> None:
    revisions, comments = _stores(tmp_path)
    revisions.register(_revision("rev-1"))

    with pytest.raises(OrganelleContractError) as info:
        comments.add(
            CommentRecord(
                comment_id=f"comment-{'c' * 32}",
                thread_id="thread-1",
                revision_id="rev-1",
                selection=CommentSelection(
                    text_range=None,
                    image_region=NormalizedRegion(x=0.0, y=0.0, width=0.5, height=0.5),
                    table_cell=None,
                    whole_artifact=False,
                    notebook_cell=None,
                ),
                body="image region on a text revision",
                status="open",
                proposed_revision_id=None,
                created_at=_CREATED,
            )
        )
    assert info.value.code == "comment.selection_kind_mismatch"


def test_notebook_selection_is_reserved_for_notebook_01(tmp_path: Path) -> None:
    revisions, comments = _stores(tmp_path)
    revisions.register(_revision("rev-1"))

    with pytest.raises(OrganelleContractError) as info:
        comments.add(
            CommentRecord(
                comment_id=f"comment-{'c' * 32}",
                thread_id="thread-1",
                revision_id="rev-1",
                selection=CommentSelection(
                    text_range=None,
                    image_region=None,
                    table_cell=None,
                    whole_artifact=False,
                    notebook_cell="cell-3",
                ),
                body="notebook cell comment",
                status="open",
                proposed_revision_id=None,
                created_at=_CREATED,
            )
        )
    assert info.value.code == "comment.selection_unresolvable"


def test_selection_requires_exactly_one_variant() -> None:
    with pytest.raises(ValidationError):
        CommentSelection(
            text_range=TextRange(unit="line", start=1, end=2),
            image_region=None,
            table_cell=None,
            whole_artifact=True,
            notebook_cell=None,
        )
    with pytest.raises(ValidationError):
        CommentSelection(
            text_range=None,
            image_region=None,
            table_cell=None,
            whole_artifact=False,
            notebook_cell=None,
        )


def test_normalized_region_must_stay_on_canvas() -> None:
    with pytest.raises(ValidationError):
        NormalizedRegion(x=0.9, y=0.1, width=0.5, height=0.2)


def test_comment_targets_never_drift_across_revisions(tmp_path: Path) -> None:
    revisions, comments = _stores(tmp_path)
    revisions.register(_revision("rev-1"))
    comment = comments.add(
        CommentRecord(
            comment_id=f"comment-{'c' * 32}",
            thread_id="thread-1",
            revision_id="rev-1",
            selection=CommentSelection(
                text_range=TextRange(unit="line", start=1, end=2),
                image_region=None,
                table_cell=None,
                whole_artifact=False,
                notebook_cell=None,
            ),
            body="original request",
            status="open",
            proposed_revision_id=None,
            created_at=_CREATED,
        )
    )
    revisions.register(_revision("rev-2", parent="rev-1"))

    persisted = comments.get(comment.comment_id)
    assert persisted.revision_id == "rev-1"
    assert persisted.selection.text_range == TextRange(unit="line", start=1, end=2)
    assert comments.by_revision("rev-1") == (persisted,)
    assert comments.by_revision("rev-2") == ()


def test_comment_lifecycle_is_one_way(tmp_path: Path) -> None:
    revisions, comments = _stores(tmp_path)
    revisions.register(_revision("rev-1"))
    comment = comments.add(
        CommentRecord(
            comment_id=f"comment-{'c' * 32}",
            thread_id="thread-1",
            revision_id="rev-1",
            selection=CommentSelection(
                text_range=None,
                image_region=None,
                table_cell=None,
                whole_artifact=True,
                notebook_cell=None,
            ),
            body="approve then apply",
            status="open",
            proposed_revision_id=None,
            created_at=_CREATED,
        )
    )

    rejected = comments.mark_rejected(comment.comment_id)
    assert rejected.status == "rejected"
    with pytest.raises(OrganelleContractError) as info:
        comments.mark_applied(comment.comment_id, "rev-2")
    assert info.value.code == "comment.transition_invalid"


def test_comment_for_unknown_revision_is_rejected(tmp_path: Path) -> None:
    _, comments = _stores(tmp_path)

    with pytest.raises(OrganelleContractError) as info:
        comments.add(
            CommentRecord(
                comment_id=f"comment-{'c' * 32}",
                thread_id="thread-1",
                revision_id="missing",
                selection=CommentSelection(
                    text_range=None,
                    image_region=None,
                    table_cell=None,
                    whole_artifact=True,
                    notebook_cell=None,
                ),
                body="dangling target",
                status="open",
                proposed_revision_id=None,
                created_at=_CREATED,
            )
        )
    assert info.value.code == "revision.unknown_revision"


def test_comment_store_exposes_no_free_mutation_api(tmp_path: Path) -> None:
    _, comments = _stores(tmp_path)
    for forbidden in ("update", "delete", "patch", "edit", "set_body"):
        assert not hasattr(comments, forbidden)


def test_comments_persist_and_restore(tmp_path: Path) -> None:
    revisions, comments = _stores(tmp_path)
    revisions.register(_revision("rev-1"))
    comments.add(
        CommentRecord(
            comment_id=f"comment-{'c' * 32}",
            thread_id="thread-1",
            revision_id="rev-1",
            selection=CommentSelection(
                text_range=None,
                image_region=None,
                table_cell=None,
                whole_artifact=True,
                notebook_cell=None,
            ),
            body="durable",
            status="open",
            proposed_revision_id=None,
            created_at=_CREATED,
        )
    )

    reopened = CommentStore(
        tmp_path / "comments", revisions=ArtifactRevisionStore(tmp_path / "revisions")
    )
    assert len(reopened.by_thread("thread-1")) == 1
