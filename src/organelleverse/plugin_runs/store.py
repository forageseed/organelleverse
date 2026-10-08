"""Atomic, restart-safe persistence for normal plugin runs."""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from organelleverse.core.errors import OrganelleContractError, OrganelleInputError

from .models import PluginRunEvent, PluginRunRecord, RunError

__all__ = ["PluginRunStore"]

_TERMINAL = frozenset({"succeeded", "failed"})
_DESKTOP_STATE_FILES = frozenset({"trust.json"})


def _interrupted_error() -> RunError:
    return RunError(
        error_code="plugin_run.run_interrupted",
        message="the plugin-run service stopped before this run became terminal",
        details={},
        retryable=False,
        suggested_action={},
    )


class PluginRunStore:
    """One canonical JSON file per application run receipt beneath ``root``."""

    def __init__(self, root: Path, *, legacy_root: Path | None = None) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._records: dict[str, PluginRunRecord] = {}
        self._events: dict[str, list[PluginRunEvent]] = {}
        for path in sorted(self._root.glob("*.json")):
            self._restore(self._read_record(path))
        if legacy_root is not None:
            self._migrate_legacy_root(Path(legacy_root))

    @staticmethod
    def _history_invalid(path: Path, error: Exception) -> OrganelleContractError:
        return OrganelleContractError(
            code="plugin_run.history_invalid",
            message="a persisted plugin-run record is not valid canonical JSON",
            details={"path": str(path), "reason": str(error)},
        )

    @classmethod
    def _read_record(cls, path: Path) -> PluginRunRecord:
        try:
            return PluginRunRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as error:
            raise cls._history_invalid(path, error) from error

    def _restore(self, record: PluginRunRecord) -> None:
        if record.status not in _TERMINAL:
            record = self._interrupt(record)
            self._write(record)
        self._records[record.run_id] = record

    def _migrate_legacy_root(self, legacy_root: Path) -> None:
        """Copy recognizable pre-Task-1 run records into the dedicated store."""

        if not legacy_root.is_dir() or legacy_root == self._root:
            return
        for path in sorted(legacy_root.glob("*.json")):
            if path.name in _DESKTOP_STATE_FILES:
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise self._history_invalid(path, error) from error
            if not isinstance(raw, dict) or "run_id" not in raw:
                continue
            try:
                record = PluginRunRecord.model_validate(raw)
            except ValidationError as error:
                raise self._history_invalid(path, error) from error
            if record.run_id in self._records:
                continue
            if record.status not in _TERMINAL:
                record = self._interrupt(record)
            self._write(record)
            self._records[record.run_id] = record

    @staticmethod
    def _interrupt(record: PluginRunRecord) -> PluginRunRecord:
        return record.model_copy(
            update={
                "status": "failed",
                "completed_at": datetime.now(UTC),
                "error": _interrupted_error(),
            }
        )

    def _path(self, run_id: str) -> Path:
        return self._root / f"{run_id}.json"

    def _write(self, record: PluginRunRecord) -> None:
        destination = self._path(record.run_id)
        temporary = self._root / f".{record.run_id}.{uuid4().hex}.partial"
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(record.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            _fsync_directory(self._root)
        finally:
            if temporary.exists():
                temporary.unlink()

    def create(self, record: PluginRunRecord) -> PluginRunRecord:
        with self._lock:
            if record.run_id in self._records:
                raise OrganelleContractError(
                    code="plugin_run.duplicate_id",
                    message="a plugin run with this id already exists",
                    details={"run_id": record.run_id},
                )
            self._write(record)
            self._records[record.run_id] = record
            return record

    def update(self, record: PluginRunRecord) -> PluginRunRecord:
        with self._lock:
            existing = self._records.get(record.run_id)
            if existing is None:
                raise OrganelleInputError(
                    code="input.unknown_run",
                    message=f"unknown plugin run: {record.run_id}",
                    details={"run_id": record.run_id},
                )
            if existing.status in _TERMINAL:
                return existing
            self._write(record)
            self._records[record.run_id] = record
            return record

    def get(self, run_id: str) -> PluginRunRecord:
        with self._lock:
            try:
                return self._records[run_id]
            except KeyError:
                raise OrganelleInputError(
                    code="input.unknown_run",
                    message=f"unknown plugin run: {run_id}",
                    details={"run_id": run_id},
                ) from None

    def list(self) -> tuple[PluginRunRecord, ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._records.values(),
                    key=lambda record: (record.submitted_at, record.run_id),
                )
            )

    def append_event(self, event: PluginRunEvent) -> PluginRunEvent:
        with self._lock:
            if event.run_id not in self._records:
                raise OrganelleInputError(
                    code="input.unknown_run",
                    message=f"unknown plugin run: {event.run_id}",
                    details={"run_id": event.run_id},
                )
            self._events.setdefault(event.run_id, []).append(event)
            return event

    def events(self, run_id: str) -> tuple[PluginRunEvent, ...]:
        with self._lock:
            if run_id not in self._records:
                raise OrganelleInputError(
                    code="input.unknown_run",
                    message=f"unknown plugin run: {run_id}",
                    details={"run_id": run_id},
                )
            return tuple(self._events.get(run_id, []))

    def interrupt_nonterminal(self) -> tuple[PluginRunRecord, ...]:
        with self._lock:
            interrupted: list[PluginRunRecord] = []
            for run_id, record in tuple(self._records.items()):
                if record.status not in _TERMINAL:
                    terminal = self._interrupt(record)
                    self._write(terminal)
                    self._records[run_id] = terminal
                    interrupted.append(terminal)
            return tuple(interrupted)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
