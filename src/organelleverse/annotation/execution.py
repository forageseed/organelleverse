"""Fail-closed executable preflight and subprocess evidence capture."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from shutil import which
from threading import Lock
from time import monotonic
from typing import Literal

from pydantic import Field

from organelleverse.core.base import StrictFrozenModel
from organelleverse.core.errors import OrganelleDependencyError, OrganelleExecutionError
from organelleverse.core.external import MESSAGE_TAIL_LINES, tail_lines

_VERSION_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "aragorn": ("-h",),
    "barrnap": ("--version",),
    "blastn": ("-version",),
    "cmsearch": ("-h",),
    "makeblastdb": ("-version",),
    "tblastn": ("-version",),
    "tRNAscan-SE": ("-h",),
}
_VERSION_PREFIXES: dict[str, tuple[str, ...]] = {
    "cmsearch": ("INFERNAL ",),
    "tRNAscan-SE": ("tRNAscan-SE ",),
}


class ResolvedTool(StrictFrozenModel[Literal["resolved_tool"]]):
    """One executable resolved to an exact path and runtime version."""

    kind: Literal["resolved_tool"] = "resolved_tool"
    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    version: str = Field(min_length=1)
    version_argv: tuple[str, ...] = Field(min_length=1)


class CommandEvidence(StrictFrozenModel[Literal["command_evidence"]]):
    """Serializable evidence for one shell-free subprocess invocation."""

    kind: Literal["command_evidence"] = "command_evidence"
    sequence: int = Field(ge=1)
    stage: str = Field(min_length=1)
    argv: tuple[str, ...] = Field(min_length=1)
    cwd: str = ""
    started_at: str = Field(min_length=1)
    duration_seconds: float = Field(ge=0)
    timeout_seconds: int = Field(gt=0)
    returncode: int | None
    termination: Literal["exit", "signal", "timeout", "launch_error"]
    signal: int | None = Field(default=None, ge=1)
    stdout: str = ""
    stderr: str = ""


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _version_from_evidence(name: str, evidence: CommandEvidence) -> str:
    lines = [
        line.strip().removeprefix("# ")
        for line in (evidence.stdout + "\n" + evidence.stderr).splitlines()
        if line.strip()
    ]
    prefixes = _VERSION_PREFIXES.get(name, ())
    return next(
        (line for line in lines if line.startswith(prefixes)),
        lines[0] if lines else "unknown",
    )


def _validate_tool_identity(name: str, evidence: CommandEvidence) -> None:
    lines = [
        line.strip().removeprefix("# ")
        for line in (evidence.stdout + "\n" + evidence.stderr).splitlines()
        if line.strip()
    ]
    expected = name.casefold()
    if any(line.casefold().startswith(expected) for line in lines):
        return
    raise OrganelleDependencyError(
        code="dependency_identity_mismatch",
        message=f"resolved executable does not identify as {name}",
        details={"name": name, "path": evidence.argv[0], "output": lines[:5]},
        suggested_action={"verify_executable": name},
    )


class CommandRunner:
    """Run commands without a shell and append ordered JSONL evidence."""

    def __init__(self, log_dir: str | Path | None = None) -> None:
        self._log_dir = Path(log_dir) if log_dir is not None else None
        if self._log_dir is not None:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        self._records: list[CommandEvidence] = []
        self._lock = Lock()

    @property
    def records(self) -> tuple[CommandEvidence, ...]:
        """Return immutable command evidence in invocation order."""

        with self._lock:
            return tuple(self._records)

    @property
    def log_path(self) -> Path | None:
        """Return the JSONL evidence path when persistent logging is enabled."""

        return self._log_dir / "commands.jsonl" if self._log_dir is not None else None

    def _record(
        self,
        *,
        stage: str,
        argv: tuple[str, ...],
        cwd: Path | None,
        started_at: str,
        duration_seconds: float,
        timeout: int,
        returncode: int | None,
        termination: Literal["exit", "signal", "timeout", "launch_error"],
        signal_number: int | None = None,
        stdout: str | bytes | None = None,
        stderr: str | bytes | None = None,
    ) -> CommandEvidence:
        with self._lock:
            evidence = CommandEvidence(
                sequence=len(self._records) + 1,
                stage=stage,
                argv=argv,
                cwd=str(cwd.resolve()) if cwd is not None else "",
                started_at=started_at,
                duration_seconds=duration_seconds,
                timeout_seconds=timeout,
                returncode=returncode,
                termination=termination,
                signal=signal_number,
                stdout=_decode_output(stdout),
                stderr=_decode_output(stderr),
            )
            self._records.append(evidence)
            if self.log_path is not None:
                payload = json.dumps(
                    evidence.model_dump(mode="json"),
                    sort_keys=True,
                    allow_nan=False,
                    ensure_ascii=False,
                )
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(payload + "\n")
            return evidence

    def run(
        self,
        argv: tuple[str, ...],
        *,
        stage: str,
        timeout: int,
        cwd: Path | None = None,
    ) -> CommandEvidence:
        """Run one command, record it, and raise on every unsuccessful exit."""

        if not argv or any(not item for item in argv):
            raise ValueError("argv must contain non-empty command arguments")
        if not stage:
            raise ValueError("stage must not be blank")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        started_at = datetime.now(UTC).isoformat()
        started = monotonic()
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            evidence = self._record(
                stage=stage,
                argv=argv,
                cwd=cwd,
                started_at=started_at,
                duration_seconds=monotonic() - started,
                timeout=timeout,
                returncode=None,
                termination="timeout",
                stdout=error.stdout,
                stderr=error.stderr,
            )
            message = f"{Path(argv[0]).name} timed out during {stage}"
            tail = tail_lines(error.stderr, MESSAGE_TAIL_LINES)
            if tail:
                message = f"{message}: {tail}"
            raise OrganelleExecutionError(
                code="backend_execution_failed",
                message=message,
                details=evidence.model_dump(mode="json"),
            ) from error
        except OSError as error:
            evidence = self._record(
                stage=stage,
                argv=argv,
                cwd=cwd,
                started_at=started_at,
                duration_seconds=monotonic() - started,
                timeout=timeout,
                returncode=None,
                termination="launch_error",
                stderr=str(error),
            )
            raise OrganelleExecutionError(
                code="backend_execution_failed",
                message=f"{Path(argv[0]).name} could not start during {stage}: {error}",
                details=evidence.model_dump(mode="json"),
            ) from error

        returncode = completed.returncode
        signal_number = -returncode if returncode < 0 else None
        evidence = self._record(
            stage=stage,
            argv=argv,
            cwd=cwd,
            started_at=started_at,
            duration_seconds=monotonic() - started,
            timeout=timeout,
            returncode=returncode,
            termination="signal" if signal_number is not None else "exit",
            signal_number=signal_number,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if returncode != 0:
            message = f"{Path(argv[0]).name} failed during {stage}"
            tail = tail_lines(completed.stderr, MESSAGE_TAIL_LINES)
            if tail:
                message = f"{message}: {tail}"
            raise OrganelleExecutionError(
                code="backend_execution_failed",
                message=message,
                details=evidence.model_dump(mode="json"),
            )
        return evidence


def resolve_required_tools(
    names: tuple[str, ...],
    *,
    runner: CommandRunner | None = None,
    version_timeout: int = 30,
    paths: Mapping[str, str] | None = None,
) -> tuple[ResolvedTool, ...]:
    """Resolve every required executable before capturing runtime versions.

    ``paths`` overrides the PATH lookup for individual tool names (executables
    resolved from an explicit location, such as a vendored LOSAT build).
    """

    unique_names = tuple(dict.fromkeys(names))
    overrides = dict(paths or {})
    resolved_paths = {name: overrides.get(name) or which(name) for name in unique_names}
    missing = [name for name, path in resolved_paths.items() if path is None]
    if missing:
        raise OrganelleDependencyError(
            code="dependency_missing",
            message="required annotation executables are unavailable",
            details={"missing": missing},
            suggested_action={"install_executables": missing},
        )

    active_runner = runner if runner is not None else CommandRunner()
    tools: list[ResolvedTool] = []
    for name in unique_names:
        resolved = resolved_paths[name]
        if resolved is None:  # narrowed by the complete missing check above
            raise AssertionError("resolved executable unexpectedly disappeared")
        # Keep the final symlink name: multicall wrappers dispatch from argv[0].
        exact_path = str(Path(resolved).expanduser().absolute())
        version_argv = (exact_path, *_VERSION_ARGUMENTS.get(name, ("--version",)))
        evidence = active_runner.run(
            version_argv,
            stage=f"dependency_preflight:{name}",
            timeout=version_timeout,
        )
        _validate_tool_identity(name, evidence)
        version = _version_from_evidence(name, evidence)
        tools.append(
            ResolvedTool(
                name=name,
                path=exact_path,
                version=version,
                version_argv=version_argv,
            )
        )
    return tuple(tools)
