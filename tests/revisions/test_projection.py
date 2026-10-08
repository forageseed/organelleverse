"""Browser-safe server-side revision and diff projection."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleContractError
from organelleverse.revisions import ArtifactRevision, ImageDimensions
from organelleverse.revisions.projection import (
    INLINE_CONTENT_LIMIT_BYTES,
    build_revision_diff,
    build_revision_view,
)

_CREATED = datetime(2026, 8, 14, tzinfo=UTC)


def _text_revision(
    revision_id: str,
    tmp_path: Path,
    text: str,
    *,
    artifact_id: str = "art-1",
    parent: str | None = None,
    kind: str = "markdown",
) -> ArtifactRevision:
    target = tmp_path / f"{revision_id}.md"
    target.write_text(text, encoding="utf-8")
    ref = ArtifactRef.from_path(
        target, kind="report", format="markdown", media_type="text/markdown"
    )
    return ArtifactRevision(
        revision_id=revision_id,
        artifact_id=artifact_id,
        parent_revision_id=parent,
        kind=kind,  # type: ignore[arg-type]
        author="agent",
        content_ref=ref,
        status="available",
        diagnostics=(),
        evidence_links=(),
        created_at=_CREATED,
    )


def _image_revision(revision_id: str, *, parent: str | None = None) -> ArtifactRevision:
    return ArtifactRevision(
        revision_id=revision_id,
        artifact_id="art-img",
        parent_revision_id=parent,
        kind="image",
        author="agent",
        content_ref=ArtifactRef(
            kind="figure",
            uri="/tmp/host-artifacts/figure.png",
            format="png",
            media_type="image/png",
            sha256="c" * 64,
            size_bytes=8,
        ),
        status="available",
        diagnostics=(),
        evidence_links=(),
        dimensions=ImageDimensions(width=640, height=480),
        created_at=_CREATED,
    )


def test_view_is_browser_safe(tmp_path: Path) -> None:
    revision = _text_revision("rev-1", tmp_path, "# Title\n")

    view = build_revision_view(revision)

    dumped = view.model_dump_json()
    assert "uri" not in type(view).model_fields
    assert "host-artifacts" not in dumped
    assert "/tmp" not in dumped
    assert view.revision_id == "rev-1"
    assert view.artifact_id == "art-1"
    assert view.kind == "markdown"
    assert view.status == "available"
    assert view.sha256 == revision.content_ref.sha256
    assert view.media_type == "text/markdown"
    assert view.size_bytes == revision.content_ref.size_bytes


def test_text_view_carries_source_and_rendering_contract(tmp_path: Path) -> None:
    markdown = _text_revision("rev-1", tmp_path, "# Title\n")
    plain = _text_revision("rev-2", tmp_path, "plain\n", kind="text")

    markdown_view = build_revision_view(markdown)
    plain_view = build_revision_view(plain)

    assert markdown_view.content == "# Title\n"
    assert markdown_view.rendered_media_type == "text/markdown"
    assert plain_view.content == "plain\n"
    assert plain_view.rendered_media_type == "text/plain"


def test_large_content_stays_reference_only(tmp_path: Path) -> None:
    large = "x" * (INLINE_CONTENT_LIMIT_BYTES + 1)
    revision = _text_revision("rev-1", tmp_path, large)

    view = build_revision_view(revision)

    assert view.content is None
    assert view.content_inline is False
    assert view.status == "available"


def test_unavailable_content_reports_diagnostics(tmp_path: Path) -> None:
    missing = ArtifactRevision(
        revision_id="rev-missing",
        artifact_id="art-1",
        parent_revision_id=None,
        kind="markdown",
        author="agent",
        content_ref=ArtifactRef(
            kind="report",
            uri="/tmp/host-artifacts/gone.md",
            format="markdown",
            media_type="text/markdown",
            sha256="d" * 64,
            size_bytes=4,
        ),
        status="available",
        diagnostics=(),
        evidence_links=(),
        created_at=_CREATED,
    )

    view = build_revision_view(missing)

    assert view.content is None
    assert view.content_inline is False
    assert view.diagnostics
    assert all("gone.md" not in line and "/tmp" not in line for line in view.diagnostics)


def test_image_view_projects_dimensions_without_pixel_payload(tmp_path: Path) -> None:
    view = build_revision_view(_image_revision("rev-img"))

    assert view.content is None
    assert view.dimensions is not None
    assert (view.dimensions.width, view.dimensions.height) == (640, 480)


def test_diff_is_deterministic_structured_hunks(tmp_path: Path) -> None:
    old = _text_revision("rev-1", tmp_path, "alpha\nbeta\n")
    new = _text_revision(
        "rev-2", tmp_path, "alpha\ngamma\n", parent="rev-1"
    )

    diff = build_revision_diff(old, new)
    repeated = build_revision_diff(old, new)

    assert diff.old_revision_id == "rev-1"
    assert diff.new_revision_id == "rev-2"
    assert diff.content_changed is True
    assert diff == repeated
    lines = [line for hunk in diff.hunks for line in hunk.lines]
    kinds = {line.kind for line in lines}
    assert kinds == {"context", "removed", "added"}
    assert any(line.kind == "removed" and line.text == "beta" for line in lines)
    assert any(line.kind == "added" and line.text == "gamma" for line in lines)


def test_identical_content_diff_is_empty(tmp_path: Path) -> None:
    old = _text_revision("rev-1", tmp_path, "same\n")
    new = _text_revision("rev-2", tmp_path, "same\n", parent="rev-1")

    diff = build_revision_diff(old, new)

    assert diff.content_changed is False
    assert diff.hunks == ()


def test_diff_of_unavailable_content_is_undetermined_not_empty(tmp_path: Path) -> None:
    old = _text_revision("rev-1", tmp_path, "alpha\n")
    new = ArtifactRevision(
        revision_id="rev-2",
        artifact_id="art-1",
        parent_revision_id="rev-1",
        kind="markdown",
        author="agent",
        content_ref=ArtifactRef(
            kind="report",
            uri="/tmp/host-artifacts/gone.md",
            format="markdown",
            media_type="text/markdown",
            sha256="e" * 64,
            size_bytes=4,
        ),
        status="available",
        diagnostics=(),
        evidence_links=(),
        created_at=_CREATED,
    )

    diff = build_revision_diff(old, new)

    assert diff.content_changed is None
    assert diff.diagnostics


def test_diff_rejects_non_text_kinds(tmp_path: Path) -> None:
    with pytest.raises(OrganelleContractError) as info:
        build_revision_diff(_image_revision("rev-a"), _image_revision("rev-b"))
    assert info.value.code == "revision.diff_unsupported_kind"
