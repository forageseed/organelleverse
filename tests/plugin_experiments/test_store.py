"""Atomic, restart-safe experiment persistence (Plugin-04, Task 2)."""

from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleContractError, OrganelleInputError
from organelleverse.plugin_experiments.models import (
    ExperimentRecord,
    ExperimentRequest,
    ExperimentTrial,
)
from organelleverse.plugin_experiments.store import ExperimentStore


def _record(
    experiment_id: str = "exp-1",
    status: str = "queued",
    submitted_at: datetime | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=experiment_id,
        capability_id="demo.experiment",
        status=status,  # type: ignore[arg-type]
        submitted_at=submitted_at or datetime.now(UTC),
        request=ExperimentRequest(
            capability_id="demo.experiment",
            inputs={"images": "/data/images"},
            candidates=({"threshold": 0.5},),
        ),
        trials=(ExperimentTrial(index=0, parameters={"threshold": 0.5}, status="queued"),),
    )


def test_create_persists_one_canonical_json_file_atomically(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    # A terminal record round-trips unchanged (non-terminal records are
    # converted to run_interrupted on reopen; that is covered separately).
    record = _record().model_copy(
        update={
            "status": "succeeded",
            "completed_at": datetime.now(UTC),
            "best_trial_index": 0,
            "trials": (
                ExperimentTrial(
                    index=0, parameters={"threshold": 0.5}, status="succeeded", score=0.5
                ),
            ),
        }
    )
    store.create(record)

    path = tmp_path / "experiments" / "exp-1.json"
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "exp-1"
    assert ExperimentStore(tmp_path / "experiments").get("exp-1") == record


def test_update_replaces_and_get_unknown_fails(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    store.create(_record())
    updated = store.get("exp-1").model_copy(
        update={
            "status": "succeeded",
            "completed_at": datetime.now(UTC),
            "best_trial_index": 0,
            "trials": (
                ExperimentTrial(
                    index=0, parameters={"threshold": 0.5}, status="succeeded", score=0.5
                ),
            ),
        }
    )
    store.update(updated)
    assert store.get("exp-1").status == "succeeded"

    with pytest.raises(OrganelleInputError) as captured:
        store.get("exp-missing")
    assert captured.value.code == "experiment.unknown_experiment"


def test_restart_converts_interrupted_runs_to_terminal_failure(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    running = _record(status="running").model_copy(
        update={
            "trials": (ExperimentTrial(index=0, parameters={"threshold": 0.5}, status="running"),)
        }
    )
    store.create(running)

    reopened = ExperimentStore(tmp_path / "experiments")
    recovered = reopened.get("exp-1")

    assert recovered.status == "failed"
    assert recovered.completed_at is not None
    assert recovered.trials[0].status == "failed"
    assert recovered.trials[0].error is not None
    assert recovered.trials[0].error.code == "experiment.run_interrupted"
    # The conversion is persisted, not just in memory.
    assert ExperimentStore(tmp_path / "experiments").get("exp-1").status == "failed"


def test_invalid_persisted_json_is_a_contract_error_not_silent_loss(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    root.mkdir()
    (root / "exp-broken.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(OrganelleContractError) as captured:
        ExperimentStore(root)

    assert captured.value.code == "experiment.history_invalid"


def test_list_returns_records_in_submission_order(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    store.create(_record("exp-2", submitted_at=datetime(2026, 8, 12, 10, 0, 0, tzinfo=UTC)))
    store.create(_record("exp-1", submitted_at=datetime(2026, 8, 12, 11, 0, 0, tzinfo=UTC)))
    assert [record.experiment_id for record in store.list()] == ["exp-2", "exp-1"]


def test_replacement_fsyncs_the_history_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import organelleverse.plugin_experiments.store as store_module

    observed: list[int] = []
    real_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        observed.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr(store_module.os, "fsync", record_fsync)
    ExperimentStore(tmp_path / "experiments").create(_record())

    assert any(stat.S_ISDIR(mode) for mode in observed)


def test_directory_fsync_failure_is_not_silently_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import organelleverse.plugin_experiments.store as store_module

    real_fsync = os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("synthetic directory sync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(store_module.os, "fsync", fail_directory_fsync)

    with pytest.raises(OSError, match="synthetic directory sync failure"):
        ExperimentStore(tmp_path / "experiments").create(_record())


def test_terminal_history_with_a_nonterminal_trial_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    root.mkdir()
    payload = _record(status="queued").model_dump(mode="json")
    payload["status"] = "succeeded"
    payload["completed_at"] = datetime.now(UTC).isoformat()
    payload["best_trial_index"] = 0
    payload["trials"][0]["status"] = "running"
    (root / "exp-1.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OrganelleContractError) as captured:
        ExperimentStore(root)

    assert captured.value.code == "experiment.history_invalid"
