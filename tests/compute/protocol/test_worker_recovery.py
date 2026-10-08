from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest

import organelleverse.compute.worker as worker_module
from organelleverse.compute.protocol import RunStatus, RunToolRequest
from organelleverse.compute.worker import DurableWorkerBackend, PersistentRunRecord
from organelleverse.core.errors import OrganelleInputError
from organelleverse.operations.registry import registry

from .test_worker_execution import _request, _wait_terminal


def test_unknown_run_is_refused_without_creating_state(tmp_path: Path):
    root = tmp_path / "worker"
    backend = DurableWorkerBackend(registry=registry, run_root=root)

    with pytest.raises(OrganelleInputError) as raised:
        backend.get(RunToolRequest(provider_run_id="run-unknown"))

    assert raised.value.code == "compute.unknown_provider_run"
    assert not (root / "runs" / "run-unknown").exists()


def _large_fastq(path: Path, records: int = 750_000) -> None:
    record = b"@read\nACGTACGTACGTACGT\n+\nIIIIIIIIIIIIIIII\n"
    with path.open("wb") as handle:
        for _ in range(records):
            handle.write(record)


def _wait_running_pid(backend: DurableWorkerBackend, provider_run_id: str) -> int:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        record = backend.read_persistent_record(provider_run_id)
        if record.status.status == "running" and record.pid is not None:
            return record.pid
        time.sleep(0.01)
    pytest.fail("worker did not enter running state")


def test_new_backend_instance_recovers_same_completed_run(tmp_path: Path):
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r1\nACGT\n+\nIIII\n", encoding="utf-8")
    root = tmp_path / "worker"
    first = DurableWorkerBackend(registry=registry, run_root=root)
    submitted = first.submit(_request(first, reads)).run
    completed = _wait_terminal(first, submitted.provider_run_id)

    restarted = DurableWorkerBackend(registry=registry, run_root=root)
    recovered = restarted.get(RunToolRequest(provider_run_id=submitted.provider_run_id)).run

    assert recovered == completed


def test_cancel_remains_nonterminal_until_worker_acknowledges(tmp_path: Path):
    reads = tmp_path / "large.fastq"
    _large_fastq(reads)
    backend = DurableWorkerBackend(registry=registry, run_root=tmp_path / "worker")
    submitted = backend.submit(_request(backend, reads)).run
    _wait_running_pid(backend, submitted.provider_run_id)

    cancellation = backend.cancel(RunToolRequest(provider_run_id=submitted.provider_run_id)).run

    assert cancellation.status.status in {"queued", "running"}
    terminal = _wait_terminal(backend, submitted.provider_run_id)
    assert terminal.status.status == "cancelled"
    assert terminal.status.result is None
    assert terminal.status.error is None


def test_dead_worker_becomes_structured_provider_failure(tmp_path: Path):
    reads = tmp_path / "large.fastq"
    _large_fastq(reads)
    backend = DurableWorkerBackend(registry=registry, run_root=tmp_path / "worker")
    submitted = backend.submit(_request(backend, reads)).run
    pid = _wait_running_pid(backend, submitted.provider_run_id)
    os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)

    failed = _wait_terminal(backend, submitted.provider_run_id)

    assert failed.status.status == "failed"
    assert failed.status.result is None
    assert failed.status.error is not None
    assert failed.status.error["error_code"] == "compute.worker_lost"
    assert failed.status.error["kind"] == "provider_failure"


# === review regression: terminal-state overwrite race + orphan spin ========


def _seed_run_dir(root: Path, run_id: str, request, record: PersistentRunRecord) -> None:
    worker_module._run_dir(root, run_id).mkdir(mode=0o700, parents=True)
    worker_module._atomic_write_json(
        worker_module._request_path(root, run_id), request.model_dump(mode="json")
    )
    worker_module._write_state(root, record)


def test_cancellation_signal_never_overwrites_a_terminal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Regression: SIGTERM is delivered at an arbitrary bytecode boundary, so it
    # can land after _write_state(completed) but before process exit.  The
    # _CancellationRequested handler must leave the published terminal state
    # authoritative (mirroring _mark_worker_lost), not rewrite it to cancelled.
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r1\nACGT\n+\nIIII\n", encoding="utf-8")
    root = tmp_path / "worker"
    backend = DurableWorkerBackend(registry=registry, run_root=root)
    request = _request(backend, reads)
    run_id = "run-terminalrace"
    _seed_run_dir(
        root,
        run_id,
        request,
        PersistentRunRecord(
            provider_run_id=run_id,
            operation_id=request.operation_id,
            status=RunStatus(status="running", revision=2),
            pid=os.getpid(),
            expected_argv=("python", "-m", "organelleverse.compute.worker", "run", run_id),
        ),
    )

    real_write_state = worker_module._write_state

    def _write_then_signal(run_root, record):
        real_write_state(run_root, record)
        if record.status.status == "completed":
            # simulate SIGTERM arriving immediately after the completed write
            raise worker_module._CancellationRequested

    monkeypatch.setattr(worker_module, "_write_state", _write_then_signal)
    previous_handler = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        rc = worker_module.run_worker(run_id, run_root=root, registry=registry)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)

    assert rc == 0
    final = worker_module._load_state(root, run_id)
    assert final.status.status == "completed"
    assert final.status.result is not None


def test_orphan_child_exits_when_pid_is_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Regression: if the parent dies between Popen and the pid write, the
    # child's wait for the pid must be bounded — otherwise the orphan spins
    # at 200 Hz forever.  On timeout the child exits nonzero without writing
    # a terminal state (observable as compute.worker_lost by any survivor).
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r1\nACGT\n+\nIIII\n", encoding="utf-8")
    root = tmp_path / "worker"
    backend = DurableWorkerBackend(registry=registry, run_root=root)
    request = _request(backend, reads)
    run_id = "run-orphanwait"
    _seed_run_dir(
        root,
        run_id,
        request,
        PersistentRunRecord(
            provider_run_id=run_id,
            operation_id=request.operation_id,
            status=RunStatus(status="queued", revision=1),
            expected_argv=("python", "-m", "organelleverse.compute.worker", "run", run_id),
        ),
    )
    monkeypatch.setattr(worker_module, "_PID_PUBLISH_TIMEOUT_SECONDS", 0.05)

    started = time.monotonic()
    rc = worker_module.run_worker(run_id, run_root=root, registry=registry)
    elapsed = time.monotonic() - started

    assert rc == 2
    assert elapsed < 5
    final = worker_module._load_state(root, run_id)
    assert final.status.status == "queued"
