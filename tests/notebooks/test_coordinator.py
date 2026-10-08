"""Coordinator semantics: dependencies, exchange ledger, replay, terminal state."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

from organelleverse.notebooks import CellRevision, NotebookRevision, NotebookStore
from organelleverse.notebooks.coordinator import (
    CellDeclaration,
    KernelCoordinator,
)
from organelleverse.notebooks.models import KernelEnvironmentRefs

_CREATED = datetime(2026, 8, 14, tzinfo=UTC)


def _draft(cell_id: str, source: str, *, language: str = "python") -> CellRevision:
    return CellRevision(
        cell_revision_id=cell_id,
        notebook_id="nb-1",
        parent_cell_revision_id=None,
        language=language,  # type: ignore[arg-type]
        source=source,
        status="draft",
        execution=None,
        inline_outputs=(),
        artifact_output_ids=(),
        evidence_links=(),
        diagnostics=(),
        created_at=_CREATED,
    )


def _notebook(cell_revisions: tuple[str, ...]) -> NotebookRevision:
    return NotebookRevision(
        notebook_revision_id="nr-selected",
        notebook_id="nb-1",
        parent_notebook_revision_ids=(),
        cell_revisions=cell_revisions,
        kernel_environment_refs=KernelEnvironmentRefs(),
        milestone_name=None,
        created_at=_CREATED,
    )


def _coordinator(tmp_path: Path) -> tuple[NotebookStore, KernelCoordinator]:
    store = NotebookStore(tmp_path / "store")
    coordinator = KernelCoordinator(store, tmp_path / "artifacts")
    return store, coordinator


def test_outputs_record_against_exact_submitted_revision(tmp_path: Path) -> None:
    _, coordinator = _coordinator(tmp_path)
    first = coordinator.execute(_draft("cell-a", "print('one')\n1 + 1"))
    second = coordinator.execute(_draft("cell-b", "print('one')\n1 + 1"))

    assert first.status == "succeeded"
    assert second.status == "succeeded"
    assert first.parent_cell_revision_id == "cell-a"
    assert second.parent_cell_revision_id == "cell-b"
    assert first.cell_revision_id != second.cell_revision_id
    assert first.inline_outputs == second.inline_outputs


def test_failed_dependency_pauses_dependents_and_independents_run(tmp_path: Path) -> None:
    _, coordinator = _coordinator(tmp_path)

    failing = coordinator.execute(
        _draft("cell-1", "raise RuntimeError('upstream collapsed')"),
        CellDeclaration(writes=(("table", "csv"),)),
    )
    assert failing.status == "failed"
    assert failing.diagnostics

    independent = coordinator.execute(_draft("cell-2", "print('fine')"))
    assert independent.status == "succeeded"

    dependent = coordinator.execute(
        _draft("cell-3", "open(INPUT_PATHS['table']).read()"),
        CellDeclaration(reads=("art-does-not-exist",)),
    )
    assert dependent.status == "failed"
    assert any("artifact" in line for line in dependent.diagnostics)
    assert dependent.execution is not None
    assert dependent.execution.finished_at is not None


def test_exchange_ledger_records_production_and_consumption(tmp_path: Path) -> None:
    _, coordinator = _coordinator(tmp_path)

    producer = coordinator.execute(
        _draft(
            "cell-1",
            "open(OUTPUT_PATHS['table'], 'w').write('a,b\\n1,2\\n')\n'written'",
        ),
        CellDeclaration(writes=(("table", "csv"),)),
    )
    assert producer.status == "succeeded"
    artifact_id = producer.artifact_output_ids[0]

    consumer = coordinator.execute(
        _draft(
            "cell-2",
            f"rows = open(INPUT_PATHS[{artifact_id!r}]).read()\nrows.count(',')",
        ),
        CellDeclaration(reads=(artifact_id,)),
    )
    assert consumer.status == "succeeded"
    assert any(output.text == "2" for output in consumer.inline_outputs)

    production, consumption = coordinator.ledger()
    assert production.artifact_id == artifact_id
    assert production.producer_cell_revision_id == "cell-1"
    assert production.consumer_cell_revision_id == ""
    assert production.declared_format == "csv"
    assert consumption.artifact_id == artifact_id
    assert consumption.producer_cell_revision_id == "cell-1"
    assert consumption.consumer_cell_revision_id == "cell-2"


def test_undeclared_read_is_refused(tmp_path: Path) -> None:
    _, coordinator = _coordinator(tmp_path)
    producer = coordinator.execute(
        _draft("cell-1", "open(OUTPUT_PATHS['secret'], 'w').write('x')"),
        CellDeclaration(writes=(("secret", "json"),)),
    )
    known = producer.artifact_output_ids[0]

    refused = coordinator.execute(
        _draft("cell-2", "print('never reached')"),
        CellDeclaration(reads=(known, "art-unknown")),
    )
    assert refused.status == "failed"
    assert any("artifact" in line for line in refused.diagnostics)


def test_declared_output_not_written_fails_the_cell(tmp_path: Path) -> None:
    _, coordinator = _coordinator(tmp_path)
    lazy = coordinator.execute(
        _draft("cell-1", "print('forgot the file')"),
        CellDeclaration(writes=(("table", "csv"),)),
    )
    assert lazy.status == "failed"
    assert any("was not written" in line for line in lazy.diagnostics)


def test_replay_reproduces_content_without_snapshots(tmp_path: Path) -> None:
    store, coordinator = _coordinator(tmp_path)
    cell = _draft("cell-1", "open(OUTPUT_PATHS['table'], 'w').write('a,b\\n1,2\\n')\n'done'")
    declaration = CellDeclaration(writes=(("table", "csv"),))
    first = coordinator.execute(cell, declaration)
    store.register_notebook(_notebook((first.cell_revision_id,)))
    original_ref = coordinator.artifact_ref(first.artifact_output_ids[0])

    coordinator.restart()  # simulated restart: kernels destroyed
    replayed = coordinator.replay(
        store.get_notebook("nr-selected"),
        {first.cell_revision_id: cell},
        {first.cell_revision_id: declaration},
    )

    derived = replayed[first.cell_revision_id]
    assert derived.status == "succeeded"
    new_id = derived.artifact_output_ids[0]
    assert new_id != first.artifact_output_ids[0]
    assert coordinator.artifact_ref(new_id).sha256 == original_ref.sha256
    assert derived.inline_outputs == first.inline_outputs


def test_cancel_produces_durable_cancelled_terminal_state(tmp_path: Path) -> None:
    _, coordinator = _coordinator(tmp_path)
    outcome: dict[str, CellRevision] = {}

    def run() -> None:
        outcome["cell"] = coordinator.execute(_draft("cell-slow", "import time\ntime.sleep(30)"))

    worker = threading.Thread(target=run)
    worker.start()
    while not coordinator.running:
        threading.Event().wait(0.01)
    coordinator.cancel()
    worker.join(timeout=30)

    cancelled = outcome["cell"]
    assert cancelled.status == "cancelled"
    assert cancelled.diagnostics == ("cancelled by user",)
    assert cancelled.artifact_output_ids == ()


def test_process_loss_is_detected_and_never_reports_success(tmp_path: Path) -> None:
    _, coordinator = _coordinator(tmp_path)
    coordinator.execute(_draft("cell-warm", "1"))
    session = coordinator._session("python")
    process = session._ensure_started()
    process.kill()
    process.wait()

    after_loss = coordinator.execute(_draft("cell-next", "print('survivor')"))

    assert after_loss.status == "failed"
    assert any("died" in line for line in after_loss.diagnostics)
    recovered = coordinator.execute(_draft("cell-retry", "print('fresh kernel')"))
    assert recovered.status == "succeeded"


def test_replay_stops_at_first_failure(tmp_path: Path) -> None:
    store, coordinator = _coordinator(tmp_path)
    bad = _draft("cell-bad", "raise ValueError('nope')")
    good = _draft("cell-good", "print('should not run')")
    first = coordinator.execute(bad)
    second = coordinator.execute(good)
    store.register_notebook(
        NotebookRevision(
            notebook_revision_id="nr-fail",
            notebook_id="nb-1",
            parent_notebook_revision_ids=(),
            cell_revisions=(first.cell_revision_id, second.cell_revision_id),
            kernel_environment_refs=KernelEnvironmentRefs(),
            milestone_name=None,
            created_at=_CREATED,
        )
    )

    replayed = coordinator.replay(
        store.get_notebook("nr-fail"),
        {first.cell_revision_id: bad, second.cell_revision_id: good},
        {},
    )

    assert replayed[first.cell_revision_id].status == "failed"
    assert second.cell_revision_id not in replayed
