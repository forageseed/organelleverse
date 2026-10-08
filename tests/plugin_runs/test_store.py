"""Atomic, fail-closed persistence for shared plugin runs."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

import organelleverse.plugin_runs.store as store_module
from organelleverse.capabilities.plugin_descriptor import PluginDescriptor
from organelleverse.core.errors import OrganelleContractError
from organelleverse.plugin_runs import PluginRunRecord, PluginRunRequest, PluginRunStore


def _descriptor_payload() -> dict[str, object]:
    return {
        "capability_id": "demo.experiment",
        "title": "Demo",
        "summary": "",
        "inputs": [],
        "outputs": [],
        "parameters": [],
        "agent_task_description": "",
        "agent_examples": [],
        "gui_page": None,
        "optimization": None,
    }


def _descriptor() -> PluginDescriptor:
    return PluginDescriptor.model_validate(_descriptor_payload())


def _record(
    run_id: str,
    *,
    status: Literal["queued", "running", "succeeded", "failed"] = "queued",
) -> PluginRunRecord:
    completed_at = datetime.now(UTC) if status in {"succeeded", "failed"} else None
    return PluginRunRecord(
        run_id=run_id,
        l6_run_id=None,
        capability_id="demo.experiment",
        contract_identity="contract-v1",
        status=status,
        submitted_at=datetime.now(UTC),
        completed_at=completed_at,
        request=PluginRunRequest(capability_id="demo.experiment", inputs={}),
        descriptor=_descriptor(),
    )


def _legacy_payload(*, status: str) -> dict[str, object]:
    now = datetime.now(UTC).isoformat()
    return {
        "run_id": "legacy-run",
        "capability_id": "demo.experiment",
        "status": status,
        "submitted_at": now,
        "completed_at": now if status in {"succeeded", "failed"} else None,
        "request": {"capability_id": "demo.experiment", "inputs": {}, "parameters": {}},
        "descriptor": _descriptor_payload(),
        "result": None,
        "summary": {},
        "artifacts": [],
        "error": None,
    }


def test_store_round_trips_records_and_lists_in_submission_order(tmp_path: Path) -> None:
    store = PluginRunStore(tmp_path / "runs")
    first = store.create(_record("run-1"))
    second = store.create(_record("run-2"))

    assert store.get(first.run_id) == first
    assert store.list() == (first, second)


def test_terminal_record_is_immutable(tmp_path: Path) -> None:
    store = PluginRunStore(tmp_path / "runs")
    terminal = store.create(_record("run-1", status="failed"))

    assert store.update(terminal.model_copy(update={"status": "succeeded"})) == terminal
    assert store.get(terminal.run_id) == terminal


def test_raw_legacy_terminal_json_loads_without_fabricated_identities(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    (root / "legacy-run.json").write_text(
        json.dumps(_legacy_payload(status="failed")) + "\n", encoding="utf-8"
    )

    restored = PluginRunStore(root).get("legacy-run")

    assert restored.status == "failed"
    assert restored.contract_identity is None
    assert restored.l6_run_id is None


def test_raw_legacy_nonterminal_json_is_interrupted_once(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    (root / "legacy-run.json").write_text(
        json.dumps(_legacy_payload(status="running")) + "\n", encoding="utf-8"
    )

    restored = PluginRunStore(root).get("legacy-run")
    restored_again = PluginRunStore(root).get("legacy-run")

    assert restored.status == "failed"
    assert restored.error is not None
    assert restored.error.error_code == "plugin_run.run_interrupted"
    assert restored.contract_identity is None
    assert restored_again == restored


def test_explicit_legacy_root_skips_trust_and_migrates_terminal_run(
    tmp_path: Path,
) -> None:
    state = tmp_path / "desktop-state"
    state.mkdir()
    (state / "trust.json").write_text(
        '{"schema":"organelleverse.trust.v2","trusted":[]}\n', encoding="utf-8"
    )
    (state / "legacy-run.json").write_text(
        json.dumps(_legacy_payload(status="failed")) + "\n", encoding="utf-8"
    )

    restored = PluginRunStore(state / "runs", legacy_root=state).get("legacy-run")
    restored_again = PluginRunStore(state / "runs", legacy_root=state).get("legacy-run")

    assert restored.status == "failed"
    assert restored.contract_identity is None
    assert restored_again == restored
    assert (state / "runs" / "legacy-run.json").is_file()


def test_malformed_explicit_legacy_run_fails_closed(tmp_path: Path) -> None:
    state = tmp_path / "desktop-state"
    state.mkdir()
    (state / "trust.json").write_text(
        '{"schema":"organelleverse.trust.v2","trusted":[]}\n', encoding="utf-8"
    )
    (state / "broken-run.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(OrganelleContractError) as captured:
        PluginRunStore(state / "runs", legacy_root=state)

    assert captured.value.code == "plugin_run.history_invalid"


def test_invalid_history_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    (root / "broken.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(OrganelleContractError) as captured:
        PluginRunStore(root)

    assert captured.value.code == "plugin_run.history_invalid"


@pytest.mark.skipif(os.name == "nt", reason="Windows does not fsync directories")
def test_replacement_fsyncs_the_history_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptors: list[int] = []
    real_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        descriptors.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(store_module.os, "fsync", record_fsync)
    PluginRunStore(tmp_path / "runs").create(_record("run-1"))

    assert len(descriptors) == 2
