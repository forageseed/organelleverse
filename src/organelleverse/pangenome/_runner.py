"""Controlled external-process boundary for the pangenome graph builders.

Every verified external pangenome backend (minigraph and pggb) is spawned
through :func:`run_command`: an argument vector executed without a shell, an
explicit working directory, and stderr captured for failure reporting. No
``>`` redirection token can appear in ``argv`` — minigraph emits its GFA on
stdout, so the caller hands over a ``stdout_path`` and the runner streams
process stdout into that file.

The returned :class:`CommandRecord` is the immutable evidence of one spawn.
Whether a build *succeeded* is decided by the caller from the record plus
validation of the produced output — never from "the backend was invoked".
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

__all__ = ["CommandRecord", "run_command"]


@dataclass(frozen=True)
class CommandRecord:
    """Immutable outcome of one external command spawn."""

    argv: tuple[str, ...]
    returncode: int
    stdout_path: str | None
    stderr: str
    wall_seconds: float | None = None
    cpu_seconds: float | None = None
    peak_memory_bytes: int | None = None

    @property
    def resource_usage(self) -> dict[str, float | int | str | None]:
        """Measured command rusage; never a simultaneous process-tree RSS sum."""
        return {
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "measurement": (
                "wait4_child_rusage"
                if self.cpu_seconds is not None
                else "wall_clock_only"
                if self.wall_seconds is not None
                else "unmeasured"
            ),
            "memory_scope": "OS child maximum RSS; not aggregate concurrent process-tree memory",
        }

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_command(
    argv: list[str] | tuple[str, ...],
    *,
    stdout_path: str | Path | None = None,
    cwd: str | Path | None = None,
    env_overrides: Mapping[str, str] | None = None,
) -> CommandRecord:
    """Run ``argv`` without a shell; stream stdout into ``stdout_path`` when given.

    Spawn failures propagate to the caller. Wall time includes launch and wait.
    On wait4 platforms, CPU/RSS belong to this child (and reaped descendants
    represented by OS accounting). RSS is a maximum, not summed concurrent
    process-tree memory. Windows records wall time and explicit None CPU/RSS.

    References: https://man7.org/linux/man-pages/man2/wait4.2.html and
    https://github.com/apple/darwin-xnu/blob/main/bsd/man/man2/getrusage.2
    """
    argv_tuple = tuple(str(item) for item in argv)
    workdir = None if cwd is None else str(cwd)
    environment = None if env_overrides is None else {**os.environ, **env_overrides}
    executable = Path(argv_tuple[0]).expanduser()
    if executable.is_absolute():
        environment = dict(os.environ) if environment is None else environment
        environment["PATH"] = os.pathsep.join(
            (str(executable.parent), environment.get("PATH", os.defpath))
        )
    target = Path(stdout_path) if stdout_path is not None else None
    with tempfile.TemporaryFile() as stderr_log, ExitStack() as stack:
        sink = subprocess.DEVNULL
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            sink = stack.enter_context(target.open("wb"))
        started = time.perf_counter()
        cpu_seconds = None
        peak_memory_bytes = None
        with subprocess.Popen(
            argv_tuple,
            cwd=workdir,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=sink,
            stderr=stderr_log,
        ) as process:
            if hasattr(os, "wait4"):
                # Wait for this PID only: process-global RUSAGE_CHILDREN deltas
                # conflate concurrent commands. No Popen poll/wait may reap it first.
                _pid, status, usage = os.wait4(process.pid, 0)
                process.returncode = os.waitstatus_to_exitcode(status)
                cpu_seconds = float(usage.ru_utime + usage.ru_stime)
                # Linux ru_maxrss is KiB; Darwin is bytes. Other platforms do
                # not claim a byte measurement without verified unit semantics.
                if sys.platform.startswith("linux"):
                    peak_memory_bytes = int(usage.ru_maxrss) * 1024
                elif sys.platform == "darwin":
                    peak_memory_bytes = int(usage.ru_maxrss)
            else:
                process.wait()
            wall_seconds = time.perf_counter() - started
        stderr = _decode_tail(stderr_log)
    return CommandRecord(
        argv=argv_tuple,
        returncode=process.returncode,
        stdout_path=str(target) if target is not None else None,
        stderr=stderr,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        peak_memory_bytes=peak_memory_bytes,
    )


def _decode_tail(handle: BinaryIO, *, max_bytes: int = 16_384) -> str:
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    handle.seek(max(0, size - max_bytes))
    return handle.read(max_bytes).decode("utf-8", errors="replace")
