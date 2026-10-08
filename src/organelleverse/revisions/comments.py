"""Immutable typed comments anchored to exact artifact revisions."""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator

from organelleverse.core.errors import OrganelleContractError
from organelleverse.operations.spec import StrictSpecModel

from .models import ArtifactRevision
from .store import ArtifactRevisionStore, fsync_directory

__all__ = [
    "CommentRecord",
    "CommentSelection",
    "CommentStore",
    "NormalizedRegion",
    "TableCellKey",
    "TextRange",
]


class TextRange(StrictSpecModel):
    """A character or line range within one specific text revision."""

    unit: Literal["character", "line"]
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _end_not_before_start(self) -> TextRange:
        if self.end < self.start:
            raise ValueError("end must not precede start")
        return self


class NormalizedRegion(StrictSpecModel):
    """A normalized [0, 1] rectangle anchored to one exact image revision."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _stays_on_canvas(self) -> NormalizedRegion:
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("region extends beyond the normalized canvas")
        return self


class TableCellKey(StrictSpecModel):
    """A stable row/column key within one table revision."""

    row_key: str = Field(min_length=1)
    column_key: str = Field(min_length=1)


class CommentSelection(StrictSpecModel):
    """Exactly one typed selection target, per the desktop design section 8.

    ``notebook_cell`` is the reserved Notebook-01 variant: it validates as a
    selection here, but the comment store rejects it as unresolvable until
    that phase lands.
    """

    text_range: TextRange | None = None
    image_region: NormalizedRegion | None = None
    table_cell: TableCellKey | None = None
    notebook_cell: str | None = None
    whole_artifact: bool = False

    @model_validator(mode="after")
    def _exactly_one_variant(self) -> CommentSelection:
        variants = (
            self.text_range is not None,
            self.image_region is not None,
            self.table_cell is not None,
            self.notebook_cell is not None,
            self.whole_artifact,
        )
        if sum(variants) != 1:
            raise ValueError("a comment selection must set exactly one variant")
        return self


class CommentRecord(StrictSpecModel):
    """One immutable comment plus its one-way review lifecycle.

    The body, selection, and cited revision never change; only the review
    status may transition, once, from ``open`` to ``applied`` or
    ``rejected``.
    """

    comment_id: str = Field(pattern=r"^comment-[a-z0-9][a-z0-9._-]*$")
    thread_id: str = Field(min_length=1)
    revision_id: str = Field(min_length=1)
    selection: CommentSelection
    body: str = Field(min_length=1)
    status: Literal["open", "applied", "rejected"] = "open"
    proposed_revision_id: str | None = None
    created_at: datetime


_KIND_SELECTIONS: dict[str, frozenset[str]] = {
    "text": frozenset({"text_range", "whole_artifact"}),
    "markdown": frozenset({"text_range", "whole_artifact"}),
    "image": frozenset({"image_region", "whole_artifact"}),
    "table": frozenset({"table_cell", "whole_artifact"}),
}


def _error(code: str, message: str, **details: object) -> OrganelleContractError:
    return OrganelleContractError(code=code, message=message, details=details)


def _active_variant(selection: CommentSelection) -> str:
    if selection.notebook_cell is not None:
        return "notebook_cell"
    if selection.text_range is not None:
        return "text_range"
    if selection.image_region is not None:
        return "image_region"
    if selection.table_cell is not None:
        return "table_cell"
    return "whole_artifact"


class CommentStore:
    """Append-only comment persistence validating against exact revisions."""

    def __init__(self, root: Path, *, revisions: ArtifactRevisionStore) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._revisions = revisions
        self._lock = threading.Lock()
        self._records: dict[str, CommentRecord] = {}
        for path in sorted(self._root.glob("comment-*.json")):
            self._restore(self._read_record(path))

    @staticmethod
    def _history_invalid(path: Path, error: Exception) -> OrganelleContractError:
        return _error(
            "comment.history_invalid",
            "a persisted comment record is not valid canonical JSON",
            path=str(path),
            reason=str(error),
        )

    @classmethod
    def _read_record(cls, path: Path) -> CommentRecord:
        try:
            return CommentRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as error:
            raise cls._history_invalid(path, error) from error

    def _restore(self, record: CommentRecord) -> None:
        if record.comment_id in self._records:
            raise _error(
                "comment.history_invalid",
                "two persisted comments share one comment id",
                comment_id=record.comment_id,
            )
        self._records[record.comment_id] = record

    def _path(self, comment_id: str) -> Path:
        return self._root / f"{comment_id}.json"

    def _write(self, record: CommentRecord) -> None:
        destination = self._path(record.comment_id)
        temporary = self._root / f".{record.comment_id}.{uuid4().hex}.partial"
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(record.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            fsync_directory(self._root)
        finally:
            if temporary.exists():
                temporary.unlink()

    def add(self, comment: CommentRecord) -> CommentRecord:
        """Append one fresh open comment after validating its target."""
        if comment.status != "open" or comment.proposed_revision_id is not None:
            raise _error(
                "comment.initial_state_invalid",
                "new comments must start open with no proposed revision",
                comment_id=comment.comment_id,
            )
        revision = self._revisions.get(comment.revision_id)
        self._require_resolvable_selection(comment, revision)
        with self._lock:
            if comment.comment_id in self._records:
                raise _error(
                    "comment.duplicate_id",
                    "a comment with this id already exists",
                    comment_id=comment.comment_id,
                )
            self._write(comment)
            self._records[comment.comment_id] = comment
            return comment

    @staticmethod
    def _require_resolvable_selection(
        comment: CommentRecord, revision: ArtifactRevision
    ) -> None:
        variant = _active_variant(comment.selection)
        if variant == "notebook_cell":
            raise _error(
                "comment.selection_unresolvable",
                "notebook cell selections arrive with the Notebook-01 phase",
                comment_id=comment.comment_id,
            )
        allowed = _KIND_SELECTIONS[revision.kind]
        if variant not in allowed:
            raise _error(
                "comment.selection_kind_mismatch",
                "the selection variant does not resolve against this revision kind",
                comment_id=comment.comment_id,
                selection_variant=variant,
                revision_kind=revision.kind,
            )

    def get(self, comment_id: str) -> CommentRecord:
        try:
            return self._records[comment_id]
        except KeyError:
            raise _error(
                "comment.unknown_comment",
                "no comment exists with this id",
                comment_id=comment_id,
            ) from None

    def by_revision(self, revision_id: str) -> tuple[CommentRecord, ...]:
        with self._lock:
            records = [
                record
                for record in self._records.values()
                if record.revision_id == revision_id
            ]
        return tuple(sorted(records, key=lambda r: (r.created_at, r.comment_id)))

    def by_thread(self, thread_id: str) -> tuple[CommentRecord, ...]:
        with self._lock:
            records = [
                record for record in self._records.values()
                if record.thread_id == thread_id
            ]
        return tuple(sorted(records, key=lambda r: (r.created_at, r.comment_id)))

    def mark_applied(self, comment_id: str, proposed_revision_id: str) -> CommentRecord:
        """One-way open -> applied transition recording the successor."""
        with self._lock:
            record = self._require_open(comment_id)
            updated = record.model_copy(
                update={"status": "applied", "proposed_revision_id": proposed_revision_id}
            )
            self._write(updated)
            self._records[comment_id] = updated
            return updated

    def mark_rejected(self, comment_id: str) -> CommentRecord:
        """One-way open -> rejected transition; nothing is authored."""
        with self._lock:
            record = self._require_open(comment_id)
            updated = record.model_copy(update={"status": "rejected"})
            self._write(updated)
            self._records[comment_id] = updated
            return updated

    def _require_open(self, comment_id: str) -> CommentRecord:
        record = self.get(comment_id)
        if record.status != "open":
            raise _error(
                "comment.transition_invalid",
                "only an open comment can transition",
                comment_id=comment_id,
                status=record.status,
            )
        return record
