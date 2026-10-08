"""Server-side, browser-safe revision and diff projection.

Views expose stable IDs and browser-safe content only: the managed-artifact
URI never crosses this boundary, and large content stays a reference on the
server instead of being inlined.
"""

from __future__ import annotations

import difflib
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue

from organelleverse.core.errors import OrganelleContractError
from organelleverse.operations.spec import StrictSpecModel

from .models import ArtifactRevision

__all__ = [
    "INLINE_CONTENT_LIMIT_BYTES",
    "ArtifactRevisionView",
    "DiffHunk",
    "DiffLine",
    "EvidenceLinkView",
    "ImageDimensionsView",
    "RevisionDiffView",
    "build_revision_diff",
    "build_revision_view",
]

INLINE_CONTENT_LIMIT_BYTES = 256 * 1024

_TEXT_KINDS = frozenset({"text", "markdown"})
_RENDERED_MEDIA_TYPES = {"text": "text/plain", "markdown": "text/markdown"}
_CONTENT_UNAVAILABLE = "artifact content unavailable"
_CONTENT_NOT_TEXT = "artifact content is not valid UTF-8 text"


class EvidenceLinkView(StrictSpecModel):
    evidence_id: str
    kind: str
    target_id: str
    label: str = ""


class ImageDimensionsView(StrictSpecModel):
    width: int
    height: int


class ArtifactRevisionView(StrictSpecModel):
    """Browser-safe projection of one revision; no host paths or URIs."""

    revision_id: str
    artifact_id: str
    parent_revision_id: str | None
    kind: str
    status: str
    sha256: str
    media_type: str
    size_bytes: int
    created_at: datetime
    diagnostics: tuple[str, ...] = ()
    evidence_links: tuple[EvidenceLinkView, ...] = ()
    content: str | None = None
    content_inline: bool = False
    rendered_media_type: str | None = None
    dimensions: ImageDimensionsView | None = None
    operation: str = ""
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class DiffLine(StrictSpecModel):
    kind: Literal["context", "removed", "added"]
    text: str


class DiffHunk(StrictSpecModel):
    old_start: int
    new_start: int
    lines: tuple[DiffLine, ...]


class RevisionDiffView(StrictSpecModel):
    old_revision_id: str
    new_revision_id: str
    content_changed: bool | None
    hunks: tuple[DiffHunk, ...] = ()
    diagnostics: tuple[str, ...] = ()


def _read_inline_text(revision: ArtifactRevision) -> tuple[str | None, tuple[str, ...]]:
    """Read revision content inline, or report diagnostics without paths."""
    try:
        data = Path(revision.content_ref.uri).read_bytes()
    except OSError:
        return None, (_CONTENT_UNAVAILABLE,)
    if len(data) > INLINE_CONTENT_LIMIT_BYTES:
        return None, ()
    try:
        return data.decode("utf-8"), ()
    except UnicodeDecodeError:
        return None, (_CONTENT_NOT_TEXT,)


def build_revision_view(revision: ArtifactRevision) -> ArtifactRevisionView:
    """Project one revision; unreadable content keeps the view addressable."""
    if revision.kind in _TEXT_KINDS:
        content, diagnostics = _read_inline_text(revision)
        content_inline = content is not None
    else:
        content, diagnostics, content_inline = None, (), False

    return ArtifactRevisionView(
        revision_id=revision.revision_id,
        artifact_id=revision.artifact_id,
        parent_revision_id=revision.parent_revision_id,
        kind=revision.kind,
        status=revision.status,
        sha256=revision.content_ref.sha256,
        media_type=revision.content_ref.media_type,
        size_bytes=revision.content_ref.size_bytes,
        created_at=revision.created_at,
        diagnostics=tuple(revision.diagnostics) + diagnostics,
        evidence_links=tuple(
            EvidenceLinkView(
                evidence_id=link.evidence_id,
                kind=link.kind,
                target_id=link.target_id,
                label=link.label,
            )
            for link in revision.evidence_links
        ),
        content=content,
        content_inline=content_inline,
        rendered_media_type=_RENDERED_MEDIA_TYPES.get(revision.kind)
        if revision.kind in _TEXT_KINDS
        else None,
        dimensions=ImageDimensionsView(
            width=revision.dimensions.width, height=revision.dimensions.height
        )
        if revision.dimensions is not None
        else None,
        operation=revision.operation,
        parameters=dict(revision.parameters),
    )


def build_revision_diff(
    old: ArtifactRevision, new: ArtifactRevision
) -> RevisionDiffView:
    """Deterministic structured line diff for text and markdown revisions.

    Content that is unavailable or not inlinable yields an undetermined
    ``content_changed`` — never an invented empty diff.
    """
    _require_diffable(old, new)
    old_text, old_diagnostics = _read_inline_text(old)
    new_text, new_diagnostics = _read_inline_text(new)
    diagnostics = old_diagnostics + new_diagnostics
    if old_text is None or new_text is None:
        return RevisionDiffView(
            old_revision_id=old.revision_id,
            new_revision_id=new.revision_id,
            content_changed=None,
            hunks=(),
            diagnostics=diagnostics,
        )

    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    lines: list[DiffLine] = []
    changed = False
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        a=old_lines, b=new_lines, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            lines.extend(DiffLine(kind="context", text=text) for text in old_lines[i1:i2])
            continue
        changed = True
        lines.extend(DiffLine(kind="removed", text=text) for text in old_lines[i1:i2])
        lines.extend(DiffLine(kind="added", text=text) for text in new_lines[j1:j2])

    return RevisionDiffView(
        old_revision_id=old.revision_id,
        new_revision_id=new.revision_id,
        content_changed=changed,
        hunks=(
            (DiffHunk(old_start=1, new_start=1, lines=tuple(lines)),)
            if changed
            else ()
        ),
        diagnostics=(),
    )


def _require_diffable(old: ArtifactRevision, new: ArtifactRevision) -> None:
    if old.kind not in _TEXT_KINDS or new.kind not in _TEXT_KINDS:
        raise OrganelleContractError(
            code="revision.diff_unsupported_kind",
            message="structured diff is defined for text and markdown revisions only",
            details={"old_kind": old.kind, "new_kind": new.kind},
        )
    if old.artifact_id != new.artifact_id:
        raise OrganelleContractError(
            code="revision.diff_artifact_mismatch",
            message="diff compares revisions of the same artifact only",
            details={
                "old_artifact_id": old.artifact_id,
                "new_artifact_id": new.artifact_id,
            },
        )
