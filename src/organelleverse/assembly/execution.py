"""Shell-free command execution and ordered process observations."""

from __future__ import annotations

import contextlib
import os
import subprocess
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from organelleverse.operations.spec import StrictSpecModel

__all__ = ["CommandOutcome", "CommandRunner", "managed_prefix_environment"]


Termination = Literal["exit", "signal", "timeout", "launch_error"]


class CommandOutcome(StrictSpecModel):
    """Frozen, side-effect-free record of one executed command."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", revalidate_instances="always", arbitrary_types_allowed=True
    )

    stage: str
    argv: tuple[str, ...]
    cwd: str
    started: bool
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0)
    returncode: int | None = None
    termination: Termination
    signal: int | None = None
    stdout_path: str
    stderr_path: str
    launch_error: str = ""


_PopenFactory = Callable[..., "subprocess.Popen[bytes]"]


def managed_prefix_environment(
    prefix: Path,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Activate one managed prefix without inheriting outer Python state."""
    env = dict(os.environ if base is None else base)
    prefix_text = str(prefix)
    env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")
    env["CONDA_PREFIX"] = prefix_text
    env["CONDA_DEFAULT_ENV"] = prefix_text
    env["CONDA_SHLVL"] = "1"
    for name in tuple(env):
        if name in {
            "VIRTUAL_ENV",
            "PYTHONHOME",
            "PYTHONPATH",
            "CONDA_PYTHON_EXE",
            "_CE_CONDA",
            "_CE_M",
            "_CONDA_PYTHON_SYSCONFIGDATA_NAME",
            "_PYTHON_SYSCONFIGDATA_NAME",
        } or name.startswith("CONDA_PREFIX_"):
            env.pop(name, None)
    return env


class CommandRunner:
    """Run argv sequences with ``subprocess`` and ``shell=False``."""

    def __init__(self, *, popen_factory: _PopenFactory | None = None) -> None:
        self._popen_factory = popen_factory

    def run(
        self,
        argv: tuple[str, ...],
        *,
        stage: str,
        cwd: Path,
        timeout_seconds: float,
        stdout_path: Path,
        stderr_path: Path,
        env: Mapping[str, str] | None = None,
        stdin_path: Path | None = None,
    ) -> CommandOutcome:
        started_at = datetime.now(UTC)
        # File handles stay open across Popen.communicate(); they are closed in the
        # finally block so cancellation still propagates after handles release.
        stdout_handle = Path(stdout_path).open("wb")  # noqa: SIM115
        stderr_handle = Path(stderr_path).open("wb")  # noqa: SIM115
        # ``stdin_path`` lets a staged pipeline feed one process from a verified
        # temporary file without a shell or a host pipeline executor; when it is
        # absent the process inherits no stdin (``DEVNULL``), preserving every
        # existing caller's behavior.
        stdin_handle = Path(stdin_path).open("rb") if stdin_path is not None else None  # noqa: SIM115
        popen: subprocess.Popen[bytes] | None = None
        try:
            factory = self._popen_factory if self._popen_factory is not None else subprocess.Popen
            try:
                popen = factory(
                    list(argv),
                    cwd=str(cwd),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    stdin=stdin_handle if stdin_handle is not None else subprocess.DEVNULL,
                    shell=False,
                    env=_resolved_env(env),
                    close_fds=True,
                )
            except (OSError, FileNotFoundError, ValueError) as error:
                return _launch_error_outcome(
                    argv, stage, cwd, stdout_path, stderr_path, started_at, str(error)
                )

            assert popen is not None
            try:
                popen.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                popen.kill()
                # Best-effort reap after kill; the process may already be dead.
                with contextlib.suppress(Exception):
                    popen.communicate(timeout=5)
                signal = _negative_signal(getattr(popen, "returncode", None))
                return _outcome(
                    argv,
                    stage,
                    cwd,
                    stdout_path,
                    stderr_path,
                    started_at,
                    started=True,
                    returncode=None,
                    termination="timeout",
                    signal=signal,
                )

            returncode = popen.returncode
            if returncode is not None and returncode < 0:
                return _outcome(
                    argv,
                    stage,
                    cwd,
                    stdout_path,
                    stderr_path,
                    started_at,
                    started=True,
                    returncode=returncode,
                    termination="signal",
                    signal=returncode,
                )
            return _outcome(
                argv,
                stage,
                cwd,
                stdout_path,
                stderr_path,
                started_at,
                started=True,
                returncode=returncode,
                termination="exit",
                signal=None,
            )
        finally:
            # Close file handles even on cancellation/KeyboardInterrupt; the exception
            # then propagates unchanged.
            stdout_handle.close()
            stderr_handle.close()
            if stdin_handle is not None:
                stdin_handle.close()


def _outcome(
    argv: tuple[str, ...],
    stage: str,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    started_at: datetime,
    *,
    started: bool,
    returncode: int | None,
    termination: Termination,
    signal: int | None,
) -> CommandOutcome:
    finished_at = datetime.now(UTC)
    duration = (finished_at - started_at).total_seconds()
    return CommandOutcome(
        stage=stage,
        argv=argv,
        cwd=str(cwd),
        started=started,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration if duration >= 0 else 0.0,
        returncode=returncode,
        termination=termination,
        signal=signal,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        launch_error="",
    )


def _launch_error_outcome(
    argv: tuple[str, ...],
    stage: str,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    started_at: datetime,
    message: str,
) -> CommandOutcome:
    finished_at = datetime.now(UTC)
    duration = (finished_at - started_at).total_seconds()
    return CommandOutcome(
        stage=stage,
        argv=argv,
        cwd=str(cwd),
        started=False,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration if duration >= 0 else 0.0,
        returncode=None,
        termination="launch_error",
        signal=None,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        launch_error=message,
    )


def _negative_signal(returncode: object) -> int | None:
    if isinstance(returncode, int) and returncode < 0:
        return returncode
    return None


def _resolved_env(env: Mapping[str, str] | None) -> dict[str, str]:
    if env is None:
        # Minimal inherited environment; callers that resolved a locked prefix PATH
        # pass an explicit env. Falling back to the host env here is an observation-only
        # default used by probes that do not depend on a managed prefix.
        return dict(os.environ)
    return dict(env)
