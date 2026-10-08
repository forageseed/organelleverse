"""Two-level immutable notebook revision store."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from organelleverse.core.errors import OrganelleContractError
from organelleverse.notebooks import (
    CellExecutionRecord,
    CellRevision,
    InlineOutput,
    KernelEnvironmentRefs,
    NotebookRevision,
    NotebookStore,
)

_BASE = datetime(2026, 8, 14, tzinfo=UTC)


def _cell(
    cell_revision_id: str,
    notebook_id: str,
    *,
    parent: str | None = None,
    status: str = "draft",
    language: str = "python",
    source: str = "1 + 1",
    diagnostics: tuple[str, ...] = (),
    offset: int = 0,
) -> CellRevision:
    terminal = status in {"succeeded", "failed", "cancelled"}
    return CellRevision(
        cell_revision_id=cell_revision_id,
        notebook_id=notebook_id,
        parent_cell_revision_id=parent,
        language=language,  # type: ignore[arg-type]
        source=source,
        status=status,  # type: ignore[arg-type]
        execution=CellExecutionRecord(
            kernel_session_id="session-1",
            started_at=_BASE + timedelta(seconds=offset),
            finished_at=_BASE + timedelta(seconds=offset + 1) if terminal else None,
        )
        if status != "draft"
        else None,
        inline_outputs=(InlineOutput(media_type="text/plain", text="2"),)
        if status == "succeeded"
        else (),
        artifact_output_ids=(),
        evidence_links=(),
        diagnostics=diagnostics,
        created_at=_BASE + timedelta(seconds=offset),
    )


def _notebook(
    notebook_revision_id: str,
    notebook_id: str,
    cell_revisions: tuple[str, ...],
    *,
    parents: tuple[str, ...] = (),
    milestone: str | None = None,
    offset: int = 10,
) -> NotebookRevision:
    return NotebookRevision(
        notebook_revision_id=notebook_revision_id,
        notebook_id=notebook_id,
        parent_notebook_revision_ids=parents,
        cell_revisions=cell_revisions,
        kernel_environment_refs=KernelEnvironmentRefs(python="env-py-1", r="env-r-1"),
        milestone_name=milestone,
        created_at=_BASE + timedelta(seconds=offset),
    )


def test_two_level_append_preserves_parents_and_order(tmp_path: Path) -> None:
    store = NotebookStore(tmp_path)
    store.register_cell(_cell("cell-1", "nb-1", status="succeeded"))
    store.register_cell(_cell("cell-2", "nb-1", parent="cell-1", offset=2))
    store.register_notebook(_notebook("nr-1", "nb-1", ("cell-1",)))
    store.register_notebook(_notebook("nr-2", "nb-1", ("cell-1", "cell-2"), parents=("nr-1",), offset=20))

    assert store.get_cell("cell-2").parent_cell_revision_id == "cell-1"
    assert store.get_notebook("nr-1").cell_revisions == ("cell-1",)
    assert store.get_notebook("nr-2").cell_revisions == ("cell-1", "cell-2")
    assert [nr.notebook_revision_id for nr in store.lineage("nb-1")] == ["nr-1", "nr-2"]


def test_models_are_frozen_and_store_has_no_mutation_api(tmp_path: Path) -> None:
    store = NotebookStore(tmp_path)
    cell = _cell("cell-1", "nb-1")
    with pytest.raises(ValidationError):
        cell.source = "mutated"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        CellRevision.model_validate({**cell.model_dump(), "extra": 1})
    for forbidden in ("update", "delete", "patch", "remove", "edit"):
        assert not hasattr(store, forbidden)


def test_duplicate_and_cross_notebook_targets_are_rejected(tmp_path: Path) -> None:
    store = NotebookStore(tmp_path)
    store.register_cell(_cell("cell-1", "nb-1"))
    store.register_notebook(_notebook("nr-1", "nb-1", ("cell-1",)))

    with pytest.raises(OrganelleContractError) as info:
        store.register_cell(_cell("cell-1", "nb-1"))
    assert info.value.code == "notebook.duplicate_cell"

    with pytest.raises(OrganelleContractError) as info:
        store.register_notebook(_notebook("nr-1", "nb-1", ("cell-1",)))
    assert info.value.code == "notebook.duplicate_notebook"

    with pytest.raises(OrganelleContractError) as info:
        store.register_cell(_cell("cell-x", "nb-2", parent="cell-1"))
    assert info.value.code == "notebook.cell_parent_mismatch"

    with pytest.raises(OrganelleContractError) as info:
        store.register_notebook(_notebook("nr-3", "nb-2", ("cell-1",)))
    assert info.value.code == "notebook.cell_not_in_notebook"

    with pytest.raises(OrganelleContractError) as info:
        store.register_notebook(_notebook("nr-4", "nb-1", (), parents=("missing",)))
    assert info.value.code == "notebook.unknown_parent"


def test_failures_and_cancellations_stay_in_history(tmp_path: Path) -> None:
    store = NotebookStore(tmp_path)
    store.register_cell(
        _cell("cell-ok", "nb-1", status="succeeded", offset=1)
    )
    store.register_cell(
        _cell("cell-fail", "nb-1", status="failed", diagnostics=("kernel died",), offset=2)
    )
    store.register_cell(_cell("cell-cancel", "nb-1", status="cancelled", offset=3))
    store.register_notebook(
        _notebook("nr-1", "nb-1", ("cell-ok", "cell-fail", "cell-cancel"))
    )

    failed = store.get_cell("cell-fail")
    assert failed.status == "failed"
    assert failed.diagnostics == ("kernel died",)
    cancelled = store.get_cell("cell-cancel")
    assert cancelled.status == "cancelled"
    assert store.get_notebook("nr-1").cell_revisions == (
        "cell-ok",
        "cell-fail",
        "cell-cancel",
    )


def test_branch_restore_appends_never_rewrites(tmp_path: Path) -> None:
    store = NotebookStore(tmp_path)
    store.register_cell(_cell("cell-1", "nb-1", offset=1))
    store.register_cell(_cell("cell-2b", "nb-1", parent="cell-1", offset=2))
    store.register_notebook(_notebook("nr-1", "nb-1", ("cell-1",)))
    store.register_notebook(_notebook("nr-2", "nb-1", ("cell-1", "cell-2b"), parents=("nr-1",), offset=20))
    store.register_notebook(_notebook("nr-3", "nb-1", ("cell-1",), parents=("nr-1",), offset=30))

    restored = store.restore("nr-1")
    assert restored.notebook_revision_id != "nr-1"
    assert restored.parent_notebook_revision_ids == ("nr-1",)
    assert restored.cell_revisions == ("cell-1",)
    assert [nr.notebook_revision_id for nr in store.lineage("nb-1")] == [
        "nr-1",
        "nr-2",
        "nr-3",
        restored.notebook_revision_id,
    ]
    assert store.get_notebook("nr-1").created_at == _BASE + timedelta(seconds=10)


def test_milestones_append_names_without_touching_history(tmp_path: Path) -> None:
    store = NotebookStore(tmp_path)
    store.register_cell(_cell("cell-1", "nb-1"))
    store.register_notebook(_notebook("nr-1", "nb-1", ("cell-1",)))
    store.register_notebook(_notebook("nr-2", "nb-1", ("cell-1",), parents=("nr-1",), offset=20))

    named = store.name_milestone("nr-2", "reviewed-v1")

    assert named.milestone_name == "reviewed-v1"
    assert named.cell_revisions == ("cell-1",)
    assert store.get_notebook("nr-1").milestone_name is None
    assert len(store.lineage("nb-1")) == 3

    with pytest.raises(OrganelleContractError) as info:
        store.name_milestone("nr-1", "reviewed-v1")
    assert info.value.code == "notebook.milestone_duplicate"


def test_interrupted_executions_terminalize_on_restart(tmp_path: Path) -> None:
    store = NotebookStore(tmp_path)
    store.register_cell(_cell("cell-running", "nb-1", status="running", offset=1))
    store.register_cell(_cell("cell-queued", "nb-1", status="queued", offset=2))
    store.register_cell(_cell("cell-done", "nb-1", status="succeeded", offset=3))

    reopened = NotebookStore(tmp_path)

    for cell_id in ("cell-running", "cell-queued"):
        cell = reopened.get_cell(cell_id)
        assert cell.status == "failed"
        assert cell.diagnostics
        assert "interrupted" in " ".join(cell.diagnostics)
    assert reopened.get_cell("cell-done").status == "succeeded"


def test_persistence_restores_records(tmp_path: Path) -> None:
    store = NotebookStore(tmp_path)
    store.register_cell(_cell("cell-1", "nb-1"))
    store.register_notebook(_notebook("nr-1", "nb-1", ("cell-1",)))

    reopened = NotebookStore(tmp_path)
    assert reopened.get_notebook("nr-1").cell_revisions == ("cell-1",)
    cell_file = json.loads((tmp_path / "cell-1.json").read_text(encoding="utf-8"))
    assert {"cell_revision_id", "notebook_id", "language", "source", "status"} <= set(cell_file)


def test_corrupt_history_fails_closed(tmp_path: Path) -> None:
    store = NotebookStore(tmp_path)
    store.register_cell(_cell("cell-1", "nb-1"))
    (tmp_path / "cell-1.json").write_text("{ broken", encoding="utf-8")

    with pytest.raises(OrganelleContractError) as info:
        NotebookStore(tmp_path)
    assert info.value.code == "notebook.history_invalid"
