"""In-process registry of tool-spawned subprocesses, killed on turn cancel.

Assembly and verification commands run for minutes; a user Stop (or the turn
watchdog) that only cancels the asyncio task would leave the spawned process
tree running. Spawning code registers each process under the current
execution owner; cancelling the owner kills the whole tree — the POSIX
process group (children are spawned with ``start_new_session``) or the
Windows task tree via ``taskkill /T``.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

__all__ = [
    "current_execution_owner",
    "execution_owner",
    "kill_by_owner",
    "register_process",
    "terminate_tree",
    "unregister_process",
]

_execution_owner: ContextVar[str | None] = ContextVar(
    "organelleverse_execution_owner", default=None
)


@contextmanager
def execution_owner(owner_id: str) -> Iterator[None]:
    """Bind the running turn's id so spawned subprocesses register under it."""
    token = _execution_owner.set(owner_id)
    try:
        yield
    finally:
        _execution_owner.reset(token)


def current_execution_owner() -> str | None:
    return _execution_owner.get()


_lock = threading.Lock()
_processes: dict[str, set[subprocess.Popen]] = {}


def register_process(popen: subprocess.Popen) -> None:
    """Track popen under the current execution owner; a no-op outside a turn."""
    owner = _execution_owner.get()
    if owner is None:
        return
    with _lock:
        _processes.setdefault(owner, set()).add(popen)


def unregister_process(popen: subprocess.Popen) -> None:
    with _lock:
        for owner, owned in list(_processes.items()):
            if popen in owned:
                owned.discard(popen)
                if not owned:
                    del _processes[owner]
                return


def kill_by_owner(owner_id: str) -> int:
    """Kill every live tree registered under owner_id; returns how many."""
    with _lock:
        targets = list(_processes.get(owner_id, ()))
    killed = 0
    for popen in targets:
        try:
            if popen.poll() is not None:
                continue
            terminate_tree(popen)
        except OSError:
            continue  # the process died between the liveness check and the kill
        killed += 1
    return killed


def terminate_tree(popen: subprocess.Popen) -> None:
    """Kill the process tree: the POSIX group or the Windows task tree.

    A two-second grace separates SIGTERM from SIGKILL so pipelines flush
    their partial outputs before the hard kill.
    """
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(popen.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
    except OSError:
        return  # already dead
    deadline = time.monotonic() + 2.0
    while popen.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if popen.poll() is None:
        try:
            os.killpg(os.getpgid(popen.pid), signal.SIGKILL)
        except OSError:
            pass  # died during the grace period
