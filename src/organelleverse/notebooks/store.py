"""Atomic, append-only persistence for two-level notebook revisions."""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from organelleverse.core.errors import OrganelleContractError

from .models import CellRevision, NotebookRevision

__all__ = ["NotebookStore"]

_NON_TERMINAL = frozenset({"queued", "running"})
_INTERRUPTED = "cell execution interrupted before becoming terminal"


def _error(code: str, message: str, **details: object) -> OrganelleContractError:
    return OrganelleContractError(code=code, message=message, details=details)


class NotebookStore:
    """One canonical JSON file per cell or notebook revision beneath ``root``.

    Append-only by construction. On open, any cell left ``queued`` or
    ``running`` by a previous process becomes a durable ``failed`` with an
    interrupted diagnostic — never a silent success.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cells: dict[str, CellRevision] = {}
        self._notebooks: dict[str, NotebookRevision] = {}
        for path in sorted(self._root.glob("*.json")):
            self._restore(self._read_record(path))

    # -- persistence -------------------------------------------------------

    @staticmethod
    def _history_invalid(path: Path, error: Exception) -> OrganelleContractError:
        return _error(
            "notebook.history_invalid",
            "a persisted notebook record is not valid canonical JSON",
            path=str(path),
            reason=str(error),
        )

    @classmethod
    def _read_record(cls, path: Path) -> CellRevision | NotebookRevision:
        text = path.read_text(encoding="utf-8")
        try:
            raw = json.loads(text)
        except (OSError, ValueError) as error:
            raise cls._history_invalid(path, error) from error
        if not isinstance(raw, dict):
            raise cls._history_invalid(path, ValueError("record is not a JSON object"))
        try:
            if "cell_revision_id" in raw:
                return CellRevision.model_validate(raw)
            return NotebookRevision.model_validate(raw)
        except ValidationError as error:
            raise cls._history_invalid(path, error) from error

    def _restore(self, record: CellRevision | NotebookRevision) -> None:
        if isinstance(record, CellRevision):
            if record.cell_revision_id in self._cells:
                raise _error(
                    "notebook.history_invalid",
                    "two persisted cell revisions share one id",
                    cell_revision_id=record.cell_revision_id,
                )
            if record.status in _NON_TERMINAL:
                record = record.model_copy(
                    update={
                        "status": "failed",
                        "diagnostics": (*record.diagnostics, _INTERRUPTED),
                        "execution": record.execution.model_copy(
                            update={"finished_at": datetime.now(UTC)}
                        )
                        if record.execution is not None
                        else None,
                    }
                )
                self._write(record)
            self._cells[record.cell_revision_id] = record
        else:
            if record.notebook_revision_id in self._notebooks:
                raise _error(
                    "notebook.history_invalid",
                    "two persisted notebook revisions share one id",
                    notebook_revision_id=record.notebook_revision_id,
                )
            self._notebooks[record.notebook_revision_id] = record

    def _write(self, record: CellRevision | NotebookRevision) -> None:
        record_id = (
            record.cell_revision_id
            if isinstance(record, CellRevision)
            else record.notebook_revision_id
        )
        destination = self._root / f"{record_id}.json"
        temporary = self._root / f".{record_id}.{uuid4().hex}.partial"
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(record.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            self._fsync_directory(self._root)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    # -- append API --------------------------------------------------------

    def register_cell(self, cell: CellRevision) -> CellRevision:
        with self._lock:
            if cell.cell_revision_id in self._cells:
                raise _error(
                    "notebook.duplicate_cell",
                    "a cell revision with this id already exists",
                    cell_revision_id=cell.cell_revision_id,
                )
            if cell.parent_cell_revision_id is not None:
                parent = self._cells.get(cell.parent_cell_revision_id)
                if parent is None:
                    raise _error(
                        "notebook.unknown_cell_parent",
                        "the parent cell revision does not exist",
                        cell_revision_id=cell.cell_revision_id,
                    )
                if parent.notebook_id != cell.notebook_id:
                    raise _error(
                        "notebook.cell_parent_mismatch",
                        "the parent cell revision belongs to another notebook",
                        cell_revision_id=cell.cell_revision_id,
                    )
            self._write(cell)
            self._cells[cell.cell_revision_id] = cell
            return cell

    def register_notebook(self, revision: NotebookRevision) -> NotebookRevision:
        with self._lock:
            if revision.notebook_revision_id in self._notebooks:
                raise _error(
                    "notebook.duplicate_notebook",
                    "a notebook revision with this id already exists",
                    notebook_revision_id=revision.notebook_revision_id,
                )
            for parent_id in revision.parent_notebook_revision_ids:
                parent = self._notebooks.get(parent_id)
                if parent is None:
                    raise _error(
                        "notebook.unknown_parent",
                        "a parent notebook revision does not exist",
                        notebook_revision_id=revision.notebook_revision_id,
                        parent_notebook_revision_id=parent_id,
                    )
                if parent.notebook_id != revision.notebook_id:
                    raise _error(
                        "notebook.parent_mismatch",
                        "a parent notebook revision belongs to another notebook",
                        notebook_revision_id=revision.notebook_revision_id,
                    )
            for cell_id in revision.cell_revisions:
                cell = self._cells.get(cell_id)
                if cell is None or cell.notebook_id != revision.notebook_id:
                    raise _error(
                        "notebook.cell_not_in_notebook",
                        "a selected cell revision does not belong to this notebook",
                        notebook_revision_id=revision.notebook_revision_id,
                        cell_revision_id=cell_id,
                    )
            if revision.milestone_name is not None:
                self._require_free_milestone(revision.notebook_id, revision.milestone_name)
            self._write(revision)
            self._notebooks[revision.notebook_revision_id] = revision
            return revision

    def _require_free_milestone(self, notebook_id: str, name: str) -> None:
        for revision in self._notebooks.values():
            if (
                revision.notebook_id == notebook_id
                and revision.milestone_name == name
            ):
                raise _error(
                    "notebook.milestone_duplicate",
                    "this milestone name is already used within the notebook",
                    notebook_id=notebook_id,
                    milestone_name=name,
                )

    # -- read API ----------------------------------------------------------

    def get_cell(self, cell_revision_id: str) -> CellRevision:
        try:
            return self._cells[cell_revision_id]
        except KeyError:
            raise _error(
                "notebook.unknown_cell",
                "no cell revision exists with this id",
                cell_revision_id=cell_revision_id,
            ) from None

    def get_notebook(self, notebook_revision_id: str) -> NotebookRevision:
        try:
            return self._notebooks[notebook_revision_id]
        except KeyError:
            raise _error(
                "notebook.unknown_notebook",
                "no notebook revision exists with this id",
                notebook_revision_id=notebook_revision_id,
            ) from None

    def lineage(self, notebook_id: str) -> tuple[NotebookRevision, ...]:
        """All notebook revisions of one notebook, oldest first."""
        revisions = [
            revision
            for revision in self._notebooks.values()
            if revision.notebook_id == notebook_id
        ]
        return tuple(
            sorted(
                revisions,
                key=lambda revision: (
                    revision.created_at,
                    revision.notebook_revision_id,
                ),
            )
        )

    def distinct_notebooks(self) -> tuple[NotebookRevision, ...]:
        """The latest revision of each known notebook, ordered by creation."""
        latest: dict[str, NotebookRevision] = {}
        for revision in self._notebooks.values():
            current = latest.get(revision.notebook_id)
            if current is None or (revision.created_at, revision.notebook_revision_id) > (
                current.created_at,
                current.notebook_revision_id,
            ):
                latest[revision.notebook_id] = revision
        return tuple(
            sorted(latest.values(), key=lambda r: (r.created_at, r.notebook_revision_id))
        )

    # -- derived appends ---------------------------------------------------

    def restore(self, notebook_revision_id: str) -> NotebookRevision:
        """Append a new branch revision selecting an older revision's cells."""
        source = self.get_notebook(notebook_revision_id)
        return self.register_notebook(
            NotebookRevision(
                notebook_revision_id=f"notebook-{uuid4().hex}",
                notebook_id=source.notebook_id,
                parent_notebook_revision_ids=(source.notebook_revision_id,),
                cell_revisions=source.cell_revisions,
                kernel_environment_refs=source.kernel_environment_refs,
                milestone_name=None,
                created_at=datetime.now(UTC),
            )
        )

    def name_milestone(
        self, notebook_revision_id: str, name: str
    ) -> NotebookRevision:
        """Append a human-approved milestone name over an existing revision."""
        source = self.get_notebook(notebook_revision_id)
        if not name.strip():
            raise _error(
                "notebook.milestone_invalid",
                "a milestone name must be a nonblank string",
                notebook_revision_id=notebook_revision_id,
            )
        with self._lock:
            self._require_free_milestone(source.notebook_id, name)
            named = NotebookRevision(
                notebook_revision_id=f"notebook-{uuid4().hex}",
                notebook_id=source.notebook_id,
                parent_notebook_revision_ids=(source.notebook_revision_id,),
                cell_revisions=source.cell_revisions,
                kernel_environment_refs=source.kernel_environment_refs,
                milestone_name=name,
                created_at=datetime.now(UTC),
            )
            self._write(named)
            self._notebooks[named.notebook_revision_id] = named
            return named
