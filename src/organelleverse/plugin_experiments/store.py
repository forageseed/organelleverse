"""Atomic, restart-safe experiment JSON persistence (Plugin-04, Task 2).

One canonical JSON file per experiment. Every write is temp-file + fsync +
atomic ``os.replace``. All transitions are serialized under one lock. On
construction, every persisted ``queued``/``running`` experiment and trial is
converted to terminal failure with exact code ``experiment.run_interrupted`` —
an interrupted run is never retried. Invalid persisted JSON raises
``experiment.history_invalid`` instead of silently dropping history.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from organelleverse.core.errors import OrganelleContractError, OrganelleInputError
from organelleverse.core.result import ErrorDetail

from .adaptive_models import (
    AdaptiveStudyRecord,
    FinalExecutionClaim,
    FinalExecutionLink,
    RepeatEvidence,
)
from .adaptive_store_ops import append_repeat, recover_interrupted, validate_replacement
from .models import ExperimentRecord, ExperimentTrial

__all__ = ["ExperimentStore"]

_INTERRUPTED = "experiment.run_interrupted"
_TERMINAL = frozenset({"succeeded", "failed"})
_SAFE_RECORD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,128}$")


def _interrupted_error() -> ErrorDetail:
    return ErrorDetail(
        code=_INTERRUPTED,
        message="the experiment runner stopped before this work became terminal",
    )


class ExperimentStore:
    """One canonical JSON file per experiment beneath ``root``."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._records: dict[str, ExperimentRecord] = {}
        self._studies: dict[str, AdaptiveStudyRecord] = {}
        for path in sorted(self._root.glob("*.json")):
            try:
                raw = path.read_text(encoding="utf-8")
                decoded = cast(object, json.loads(raw))
                if not isinstance(decoded, dict):
                    raise ValueError("record JSON must be an object")
                payload = cast(dict[str, object], decoded)
                record_kind = payload.get("record_kind")
                if record_kind == "adaptive_study.v1":
                    study = AdaptiveStudyRecord.model_validate(payload)
                    self._validate_path_identity(path, study.study_id)
                    recovered = recover_interrupted(study)
                    if recovered != study:
                        self._write_study(recovered)
                    if recovered.study_id in self._studies or recovered.study_id in self._records:
                        raise ValueError("duplicate durable record identity")
                    self._studies[recovered.study_id] = recovered
                    continue
                if record_kind is not None:
                    raise ValueError(f"unknown record_kind: {record_kind!r}")
                record = ExperimentRecord.model_validate(payload)
                self._validate_path_identity(path, record.experiment_id)
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                raise OrganelleContractError(
                    code="experiment.history_invalid",
                    message="a persisted experiment record is not valid canonical JSON",
                    details={"path": str(path), "reason": str(error)},
                ) from error
            if record.status in {"queued", "running"}:
                record = self._interrupt(record)
                self._write(record)
            if record.experiment_id in self._records or record.experiment_id in self._studies:
                raise OrganelleContractError(
                    code="experiment.history_invalid",
                    message="durable experiment identities must be unique",
                    details={"path": str(path), "experiment_id": record.experiment_id},
                )
            self._records[record.experiment_id] = record

    @staticmethod
    def _interrupt(record: ExperimentRecord) -> ExperimentRecord:
        trials = tuple(
            trial
            if trial.status in {"succeeded", "failed"}
            else trial.model_copy(update={"status": "failed", "error": _interrupted_error()})
            for trial in record.trials
        )
        return record.model_copy(
            update={
                "status": "failed",
                "completed_at": datetime.now(UTC),
                "trials": trials,
                "best_trial_index": None,
            }
        )

    def _path(self, experiment_id: str) -> Path:
        self._validate_record_id(experiment_id)
        return self._root / f"{experiment_id}.json"

    @staticmethod
    def _validate_record_id(record_id: str) -> None:
        if _SAFE_RECORD_ID.fullmatch(record_id) is None:
            raise OrganelleContractError(
                code="experiment.invalid_id",
                message="durable record ID is not path-safe",
                details={"record_id": record_id},
            )

    @classmethod
    def _validate_path_identity(cls, path: Path, record_id: str) -> None:
        if _SAFE_RECORD_ID.fullmatch(record_id) is None:
            raise ValueError("durable record ID is not path-safe")
        if path.stem != record_id:
            raise ValueError("durable filename does not match payload identity")

    def _write(self, record: ExperimentRecord) -> None:
        self._write_json(record.experiment_id, record.model_dump_json())

    def _write_study(self, record: AdaptiveStudyRecord) -> None:
        self._write_json(record.study_id, record.model_dump_json())

    def _write_json(self, record_id: str, payload: str) -> None:
        destination = self._path(record_id)
        temporary = self._root / f".{record_id}.{os.getpid()}.partial"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(self._root)

    def create(self, record: ExperimentRecord) -> ExperimentRecord:
        with self._lock:
            self._validate_record_id(record.experiment_id)
            if record.experiment_id in self._records or record.experiment_id in self._studies:
                raise OrganelleContractError(
                    code="experiment.duplicate_id",
                    message="an experiment with this id already exists",
                    details={"experiment_id": record.experiment_id},
                )
            self._write(record)
            self._records[record.experiment_id] = record
            return record

    def update(self, record: ExperimentRecord) -> ExperimentRecord:
        with self._lock:
            existing = self._records.get(record.experiment_id)
            if existing is None:
                raise OrganelleInputError(
                    code="experiment.unknown_experiment",
                    message=f"unknown experiment: {record.experiment_id}",
                )
            if existing.status in _TERMINAL:
                return existing
            self._write(record)
            self._records[record.experiment_id] = record
            return record

    def update_trial(self, experiment_id: str, trial: ExperimentTrial) -> ExperimentRecord:
        """Atomically replace one trial while preserving every other durable outcome."""
        with self._lock:
            record = self._records.get(experiment_id)
            if record is None:
                raise OrganelleInputError(
                    code="experiment.unknown_experiment",
                    message=f"unknown experiment: {experiment_id}",
                )
            if record.status in _TERMINAL:
                return record
            trials = list(record.trials)
            if trial.index >= len(trials):
                raise OrganelleInputError(
                    code="experiment.unknown_trial",
                    message=f"unknown experiment trial: {trial.index}",
                    details={"experiment_id": experiment_id, "trial_index": trial.index},
                )
            trials[trial.index] = trial
            updated = record.model_copy(update={"trials": tuple(trials)})
            self._write(updated)
            self._records[experiment_id] = updated
            return updated

    def interrupt_nonterminal(self) -> tuple[ExperimentRecord, ...]:
        """Terminalize all accepted but unfinished records before service shutdown."""
        with self._lock:
            interrupted: list[ExperimentRecord] = []
            for experiment_id, record in tuple(self._records.items()):
                if record.status not in _TERMINAL:
                    recovered = self._interrupt(record)
                    self._write(recovered)
                    self._records[experiment_id] = recovered
                    interrupted.append(recovered)
            return tuple(interrupted)

    def get(self, experiment_id: str) -> ExperimentRecord:
        with self._lock:
            try:
                return self._records[experiment_id]
            except KeyError:
                raise OrganelleInputError(
                    code="experiment.unknown_experiment",
                    message=f"unknown experiment: {experiment_id}",
                ) from None

    def list(self) -> tuple[ExperimentRecord, ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._records.values(),
                    key=lambda record: (record.submitted_at, record.experiment_id),
                )
            )

    # -- adaptive studies ---------------------------------------------------

    def create_study(self, record: AdaptiveStudyRecord) -> AdaptiveStudyRecord:
        with self._lock:
            record = AdaptiveStudyRecord.model_validate(record.model_dump(mode="json"))
            self._validate_record_id(record.study_id)
            if not self._is_pristine_study(record):
                raise OrganelleContractError(
                    code="optimization.study_not_pristine",
                    message="new adaptive studies must be pristine created records",
                    details={"study_id": record.study_id},
                )
            if record.study_id in self._studies or record.study_id in self._records:
                raise OrganelleContractError(
                    code="optimization.duplicate_study_id",
                    message="a durable record with this study ID already exists",
                    details={"study_id": record.study_id},
                )
            self._write_study(record)
            self._studies[record.study_id] = record
            return record

    def get_study(self, study_id: str) -> AdaptiveStudyRecord:
        with self._lock:
            try:
                return self._studies[study_id]
            except KeyError:
                raise OrganelleInputError(
                    code="optimization.unknown_study",
                    message=f"unknown adaptive study: {study_id}",
                ) from None

    def list_studies(self) -> tuple[AdaptiveStudyRecord, ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._studies.values(),
                    key=lambda record: (record.submitted_at, record.study_id),
                )
            )

    def update_study(self, record: AdaptiveStudyRecord) -> AdaptiveStudyRecord:
        with self._lock:
            record = AdaptiveStudyRecord.model_validate(record.model_dump(mode="json"))
            existing = self._require_study(record.study_id)
            if existing == record:
                return existing
            validate_replacement(existing, record)
            self._write_study(record)
            self._studies[record.study_id] = record
            return record

    def append_repeat_evidence(
        self, study_id: str, evidence: RepeatEvidence
    ) -> AdaptiveStudyRecord:
        with self._lock:
            existing = self._require_study(study_id)
            updated = append_repeat(existing, evidence)
            if updated == existing:
                return existing
            self._write_study(updated)
            self._studies[study_id] = updated
            return updated

    def record_final_execution(
        self, study_id: str, link: FinalExecutionLink
    ) -> AdaptiveStudyRecord:
        with self._lock, self._study_file_lock(study_id):
            existing = self._read_study_from_disk(study_id)
            if existing.final_execution_claim is not None:
                raise OrganelleContractError(
                    code="optimization.final_execution_in_progress",
                    message="the final execution is already claimed",
                    details={"study_id": study_id},
                )
            return self._record_final_execution(existing, link)

    def claim_final_execution(
        self, study_id: str, claim: FinalExecutionClaim
    ) -> AdaptiveStudyRecord:
        with self._lock, self._study_file_lock(study_id):
            existing = self._read_study_from_disk(study_id)
            if existing.execution_selection is None:
                raise OrganelleContractError(
                    code="optimization.study_not_executable",
                    message="only a study with a verified selection can claim execution",
                    details={"study_id": study_id},
                )
            if existing.execution_selection.parameter_digest != claim.executed_parameter_digest:
                raise OrganelleContractError(
                    code="optimization.final_execution_mismatch",
                    message="final execution claim does not match selected parameters",
                    details={"study_id": study_id},
                )
            if existing.final_execution_link is not None:
                return existing
            if existing.final_execution_claim is not None:
                return existing
            updated = existing.model_copy(update={"final_execution_claim": claim})
            self._write_study(updated)
            self._studies[study_id] = updated
            return updated

    def complete_final_execution(
        self, study_id: str, claim_id: str, link: FinalExecutionLink
    ) -> AdaptiveStudyRecord:
        with self._lock, self._study_file_lock(study_id):
            existing = self._read_study_from_disk(study_id)
            if existing.final_execution_link is not None:
                if existing.final_execution_link == link:
                    return existing
                raise OrganelleContractError(
                    code="optimization.final_execution_conflict",
                    message="the final execution link is write-once",
                    details={"study_id": study_id},
                )
            claim = existing.final_execution_claim
            if claim is None or claim.claim_id != claim_id:
                raise OrganelleContractError(
                    code="optimization.final_execution_claim_mismatch",
                    message="only the durable claim owner may complete final execution",
                    details={"study_id": study_id},
                )
            if claim.executed_parameter_digest != link.executed_parameter_digest:
                raise OrganelleContractError(
                    code="optimization.final_execution_mismatch",
                    message="final execution does not match its durable claim",
                    details={"study_id": study_id},
                )
            updated = existing.model_copy(
                update={"final_execution_claim": None, "final_execution_link": link}
            )
            self._write_study(updated)
            self._studies[study_id] = updated
            return updated

    def _record_final_execution(
        self, existing: AdaptiveStudyRecord, link: FinalExecutionLink
    ) -> AdaptiveStudyRecord:
        study_id = existing.study_id
        if existing.execution_selection is None:
            raise OrganelleContractError(
                code="optimization.study_not_executable",
                message="only a study with a verified execution selection can link a run",
                details={"study_id": study_id},
            )
        if existing.execution_selection.parameter_digest != link.executed_parameter_digest:
            raise OrganelleContractError(
                code="optimization.final_execution_mismatch",
                message="final execution parameters do not match the selected parameters",
                details={"study_id": study_id},
            )
        if existing.final_execution_link is not None:
            if existing.final_execution_link == link:
                return existing
            raise OrganelleContractError(
                code="optimization.final_execution_conflict",
                message="the final execution link is write-once",
                details={"study_id": study_id},
            )
        updated = existing.model_copy(update={"final_execution_link": link})
        self._write_study(updated)
        self._studies[study_id] = updated
        return updated

    def _read_study_from_disk(self, study_id: str) -> AdaptiveStudyRecord:
        path = self._path(study_id)
        if not path.exists():
            self._require_study(study_id)
        try:
            decoded = cast(object, json.loads(path.read_text(encoding="utf-8")))
            study = AdaptiveStudyRecord.model_validate(decoded)
            self._validate_path_identity(path, study.study_id)
        except (json.JSONDecodeError, ValidationError, ValueError) as error:
            raise OrganelleContractError(
                code="experiment.history_invalid",
                message="a persisted adaptive study is not valid canonical JSON",
                details={"path": str(path), "reason": str(error)},
            ) from error
        self._studies[study_id] = study
        return study

    @contextmanager
    def _study_file_lock(self, study_id: str) -> Generator[None]:
        self._validate_record_id(study_id)
        lock_path = self._root / f".{study_id}.final.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            _lock_file_descriptor(descriptor)
            yield
        finally:
            _unlock_file_descriptor(descriptor)
            os.close(descriptor)

    def _require_study(self, study_id: str) -> AdaptiveStudyRecord:
        try:
            return self._studies[study_id]
        except KeyError:
            raise OrganelleInputError(
                code="optimization.unknown_study",
                message=f"unknown adaptive study: {study_id}",
            ) from None

    @staticmethod
    def _is_pristine_study(record: AdaptiveStudyRecord) -> bool:
        return (
            record.status == "created"
            and not record.attempts
            and not record.aggregates
            and not record.pareto_candidate_digests
            and record.decision is None
            and record.adopted_parameter_digest is None
            and record.execution_selection is None
            and record.final_execution_claim is None
            and record.final_execution_link is None
            and record.budget.attempted_runs == 0
            and record.budget.wall_time_seconds == 0
            and record.budget.cpu_time_seconds == 0
            and record.budget.peak_memory_bytes == 0
        )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lock_file_descriptor(descriptor: int) -> None:
    module = importlib.import_module("msvcrt" if os.name == "nt" else "fcntl")
    if os.name == "nt":  # pragma: no cover - exercised on Windows desktop builds
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        locking = cast(Callable[[int, int, int], None], module.locking)
        locking(descriptor, cast(int, module.LK_LOCK), 1)
        return
    flock = cast(Callable[[int, int], None], module.flock)
    flock(descriptor, cast(int, module.LOCK_EX))


def _unlock_file_descriptor(descriptor: int) -> None:
    module = importlib.import_module("msvcrt" if os.name == "nt" else "fcntl")
    if os.name == "nt":  # pragma: no cover - exercised on Windows desktop builds
        os.lseek(descriptor, 0, os.SEEK_SET)
        locking = cast(Callable[[int, int, int], None], module.locking)
        locking(descriptor, cast(int, module.LK_UNLCK), 1)
        return
    flock = cast(Callable[[int, int], None], module.flock)
    flock(descriptor, cast(int, module.LOCK_UN))
