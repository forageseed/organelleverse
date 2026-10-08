"""Killable Linux process boundary for one evaluator attempt."""

from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import time
from contextlib import suppress
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import cast

from organelleverse.core.errors import OrganelleContractError
from organelleverse.core.result import ErrorDetail
from organelleverse.plugin_experiments.adaptive_models import RepeatEvidence

from .study_models import EvaluationRequest, EvaluatorRunner


def evaluate_under_grant(
    runner: EvaluatorRunner,
    dispatch: EvaluationRequest,
    running: RepeatEvidence,
) -> RepeatEvidence:
    """Run one attempt in a process group that can be killed and reaped."""

    if not sys.platform.startswith("linux"):
        raise _error("hard evaluator isolation is unavailable on this runtime")
    context = multiprocessing.get_context("forkserver")
    parent_connection, child_connection = context.Pipe(duplex=True)
    worker = context.Process(
        target=_isolated_evaluator_main,
        args=(child_connection, runner, dispatch),
        name=dispatch.run_id,
    )
    started_wall = time.monotonic()
    worker.start()
    child_connection.close()
    if not parent_connection.poll(running.granted_wall_time_seconds):
        _terminate_process_group(worker)
        parent_connection.close()
        return _resource_failure(
            running,
            violation="wall-time",
            wall_time=time.monotonic() - started_wall,
        )
    raw_startup: object = parent_connection.recv()
    if not isinstance(raw_startup, tuple):
        _terminate_process_group(worker)
        raise _error("evaluator isolation failed to start")
    startup = cast(tuple[object, ...], raw_startup)
    if len(startup) != 3 or startup[0] != "started" or not isinstance(startup[2], int):
        _terminate_process_group(worker)
        raise _error("evaluator isolation failed to start")
    baseline_cpu = _process_group_cpu_seconds(worker.pid)
    baseline_peak = int(startup[2])
    if time.monotonic() - started_wall >= running.granted_wall_time_seconds:
        _terminate_process_group(worker)
        parent_connection.close()
        return _resource_failure(
            running,
            violation="wall-time",
            wall_time=time.monotonic() - started_wall,
        )
    parent_connection.send("go")
    violation: str | None = None
    envelope: tuple[object, ...] | None = None
    measured_cpu = 0.0
    measured_peak = 0
    while True:
        elapsed_wall = time.monotonic() - started_wall
        measured_cpu = max(measured_cpu, _process_group_cpu_seconds(worker.pid) - baseline_cpu)
        measured_peak = max(
            measured_peak,
            _process_group_peak_rss_bytes(worker.pid) - baseline_peak,
        )
        if elapsed_wall > running.granted_wall_time_seconds:
            violation = "wall-time"
            break
        if measured_cpu > running.granted_cpu_time_seconds:
            violation = "CPU-time"
            break
        if measured_peak > running.granted_peak_memory_bytes:
            violation = "peak-memory"
            break
        if parent_connection.poll():
            try:
                envelope = parent_connection.recv()
            except EOFError:
                envelope = None
            break
        remaining = min(
            running.granted_wall_time_seconds - elapsed_wall,
            running.granted_cpu_time_seconds - measured_cpu,
        )
        time.sleep(max(0.0, min(0.01, remaining)))
    if violation is not None:
        measured_wall = time.monotonic() - started_wall
        _terminate_process_group(worker)
        parent_connection.close()
        return _resource_failure(
            running,
            violation=violation,
            wall_time=measured_wall,
            cpu_time=measured_cpu,
            peak_memory=measured_peak,
        )
    result = _collect_result(worker, parent_connection, envelope)
    return result.model_copy(
        update={
            "cpu_time_seconds": max(result.cpu_time_seconds, measured_cpu),
            "peak_memory_bytes": max(result.peak_memory_bytes, measured_peak),
        }
    )


def _collect_result(
    worker: BaseProcess,
    connection: Connection,
    envelope: tuple[object, ...] | None,
) -> RepeatEvidence:
    try:
        worker.join(timeout=1.0)
        if worker.is_alive():
            raise _error("evaluator did not exit cleanly")
        if envelope is None and connection.poll():
            envelope = cast(tuple[object, ...], connection.recv())
        if envelope is None or not envelope:
            raise _error("evaluator execution failed")
        kind = envelope[0]
        if kind == "interrupt":
            raise KeyboardInterrupt
        if (
            kind != "result"
            or len(envelope) != 5
            or not isinstance(envelope[1], RepeatEvidence)
            or not isinstance(envelope[2], int | float)
            or not isinstance(envelope[3], int | float)
            or not isinstance(envelope[4], int)
        ):
            raise _error("evaluator execution failed")
        result = envelope[1]
        return result.model_copy(
            update={
                "wall_time_seconds": max(float(result.wall_time_seconds), float(envelope[2])),
                "cpu_time_seconds": max(float(result.cpu_time_seconds), float(envelope[3])),
                "peak_memory_bytes": max(result.peak_memory_bytes, int(envelope[4])),
            }
        )
    finally:
        connection.close()
        _terminate_process_group(worker)


def _isolated_evaluator_main(
    connection: Connection,
    runner: EvaluatorRunner,
    dispatch: EvaluationRequest,
) -> None:
    import resource

    try:
        os.setsid()
        baseline_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        try:
            connection.send(("started", os.getpid(), baseline_peak))
        except (BrokenPipeError, EOFError, ConnectionError):
            return
        try:
            command = connection.recv()
        except (EOFError, ConnectionError):
            return
        if command != "go":
            return
        started_wall = time.monotonic()
        started_cpu = time.process_time()
        try:
            result = runner.evaluate(dispatch)
        except (KeyboardInterrupt, SystemExit):
            connection.send(("interrupt",))
            return
        except BaseException:
            connection.send(("error",))
            return
        cpu_seconds = time.process_time() - started_cpu
        wall_seconds = time.monotonic() - started_wall
        peak_bytes = max(
            0,
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - baseline_peak,
        )
        connection.send(("result", result, wall_seconds, cpu_seconds, peak_bytes))
    finally:
        connection.close()


def _resource_failure(
    running: RepeatEvidence,
    *,
    violation: str,
    wall_time: float,
    cpu_time: float = 0.0,
    peak_memory: int = 0,
) -> RepeatEvidence:
    return running.model_copy(
        update={
            "status": "failed",
            "wall_time_seconds": wall_time,
            "cpu_time_seconds": cpu_time,
            "peak_memory_bytes": peak_memory,
            "error": ErrorDetail(
                code="optimization.resource_constraint_failed",
                message=f"evaluator exceeded its hard {violation} grant",
            ),
        }
    )


def _process_group_cpu_seconds(pgid: int | None) -> float:
    if pgid is None:
        return 0.0
    ticks = sum(ticks for _, ticks in _process_group_members(pgid))
    return ticks / os.sysconf("SC_CLK_TCK")


def _process_group_peak_rss_bytes(pgid: int | None) -> int:
    if pgid is None:
        return 0
    return sum(_process_peak_rss_bytes(pid) for pid, _ in _process_group_members(pgid))


def _process_group_members(pgid: int) -> tuple[tuple[int, int], ...]:
    members: list[tuple[int, int]] = []
    try:
        entries = Path("/proc").iterdir()
    except OSError:
        return ()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            closing_parenthesis = raw.rfind(")")
            if closing_parenthesis < 0:
                continue
            pid = int(raw[: raw.find("(")].strip())
            fields = raw[closing_parenthesis + 1 :].split()
            if int(fields[2]) == pgid:
                ticks = sum(int(fields[index]) for index in (11, 12, 13, 14))
                members.append((pid, ticks))
        except (FileNotFoundError, OSError, ValueError, IndexError):
            continue
    return tuple(members)


def _process_peak_rss_bytes(pid: int) -> int:
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return 0
    for line in lines:
        if line.startswith("VmHWM:"):
            try:
                return int(line.split()[1]) * 1024
            except (ValueError, IndexError):
                return 0
    return 0


def _terminate_process_group(worker: BaseProcess) -> None:
    if worker.pid is None:
        return
    # Signal the known isolated PGID unconditionally. A /proc snapshot can
    # briefly miss a just-reparented descendant after the evaluator leader
    # exits; using that snapshot as a precondition lets the child escape and
    # perform delayed side effects outside the governed attempt.
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(worker.pid, signal.SIGTERM)
    if worker.is_alive():
        worker.terminate()
    worker.join(timeout=0.5)
    if _process_group_members(worker.pid):
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(worker.pid, signal.SIGKILL)
    if worker.is_alive():
        worker.kill()
    worker.join(timeout=1.0)


def _error(message: str) -> OrganelleContractError:
    return OrganelleContractError(
        code="optimization.evaluator_failed",
        message=message,
    )


__all__ = ["evaluate_under_grant"]
