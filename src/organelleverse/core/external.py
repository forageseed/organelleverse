"""Uniform subprocess execution for external tools with stderr-rich failures.

External tools (assemblers, IQ-TREE, codeml, BLAST, ...) usually explain their
failure on stderr. Callers that only surface "tool failed" hide that cause;
this module runs one argv without a shell, captures both streams with safe
decoding, and raises the shared :mod:`organelleverse.core.errors` types with
bounded output tails in ``details`` and the stderr tail in ``message``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

from .errors import OrganelleError, OrganelleExecutionError

STDERR_TAIL_LINES = 50
STDOUT_TAIL_LINES = 20
MESSAGE_TAIL_LINES = 5

_STDERR_TAIL_CHARS = 20_000
_STDOUT_TAIL_CHARS = 8_000


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def tail_lines(text: str | bytes | None, limit: int, *, max_chars: int = _STDERR_TAIL_CHARS) -> str:
    """Return the last ``limit`` lines of captured output, bounded to ``max_chars``."""
    if limit <= 0:
        return ""
    decoded = _decode_output(text)
    if not decoded:
        return ""
    tail = "\n".join(decoded.splitlines()[-limit:])
    if len(tail) > max_chars:
        return tail[-max_chars:]
    return tail


def failure_details(
    argv: Sequence[str],
    returncode: int | None,
    *,
    stdout: str | bytes | None = None,
    stderr: str | bytes | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the bounded ``details`` payload for a failed external command."""
    details: dict[str, Any] = {
        "argv": [str(item) for item in argv],
        "returncode": returncode,
        "stderr_tail": tail_lines(stderr, STDERR_TAIL_LINES),
        "stdout_tail": tail_lines(stdout, STDOUT_TAIL_LINES, max_chars=_STDOUT_TAIL_CHARS),
    }
    if extra:
        details.update(dict(extra))
    return details


def failure_message(tool: str, returncode: int | None, stderr: str | bytes | None) -> str:
    """Build a one-glance message naming the tool and the stderr cause."""
    base = f"{tool} failed with exit code {returncode}"
    if returncode is None:
        base = f"{tool} failed"
    tail = tail_lines(stderr, MESSAGE_TAIL_LINES)
    if tail:
        return f"{base}: {tail}"
    return base


def _raise_failure(
    error_class: type[OrganelleError],
    *,
    code: str,
    message: str,
    argv: Sequence[str],
    returncode: int | None,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
    extra_details: Mapping[str, Any] | None,
) -> NoReturn:
    raise error_class(
        code=code,
        message=message,
        details=failure_details(
            argv, returncode, stdout=stdout, stderr=stderr, extra=extra_details
        ),
    )


def run_external(
    argv: Sequence[str],
    *,
    timeout: float | None = None,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    input_data: str | None = None,
    code: str = "external_command_failed",
    tool: str | None = None,
    error_class: type[OrganelleError] = OrganelleExecutionError,
    extra_details: Mapping[str, Any] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` without a shell and return the completed process on exit 0.

    On non-zero exit, timeout, or launch failure, raise ``error_class`` (an
    :class:`~organelleverse.core.errors.OrganelleError` subclass) whose
    ``details`` carry the argv, return code, and bounded stdout/stderr tails.
    ``code`` defaults to ``external_command_failed``; callers that already
    surface a stable error code must pass theirs to keep it unchanged.
    """
    if not argv:
        raise ValueError("argv must not be empty")
    name = tool or Path(str(argv[0])).name
    try:
        completed = subprocess.run(
            [str(item) for item in argv],
            cwd=cwd,
            env=dict(env) if env is not None else None,
            input=input_data,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as error:
        _raise_failure(
            error_class,
            code=code,
            message=f"{name} timed out after {timeout} seconds",
            argv=argv,
            returncode=None,
            stdout=error.stdout,
            stderr=error.stderr,
            extra_details=extra_details,
        )
    except OSError as error:
        _raise_failure(
            error_class,
            code=code,
            message=f"{name} could not be started: {error}",
            argv=argv,
            returncode=None,
            stdout=None,
            stderr=str(error),
            extra_details=extra_details,
        )
    if completed.returncode != 0:
        _raise_failure(
            error_class,
            code=code,
            message=failure_message(name, completed.returncode, completed.stderr),
            argv=argv,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            extra_details=extra_details,
        )
    return completed


def raise_for_failure(
    result: subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes],
    *,
    argv: Sequence[str] | None = None,
    code: str = "external_command_failed",
    tool: str | None = None,
    error_class: type[OrganelleError] = OrganelleExecutionError,
    extra_details: Mapping[str, Any] | None = None,
) -> None:
    """Raise an enriched error when ``result`` did not exit with code 0.

    For call sites that must keep their own ``subprocess.run`` invocation
    (custom streams, injected runners, partial checks): pass the completed
    process here at the point where the legacy code branched on
    ``returncode != 0`` to keep the failure reporting uniform.
    """
    if result.returncode == 0:
        return
    if argv is None:
        raw_args = result.args
        command: Sequence[str] = (
            (raw_args,) if isinstance(raw_args, str) else tuple(str(item) for item in raw_args)
        )
    else:
        command = argv
    if not command:
        raise ValueError("argv must not be empty")
    name = tool or Path(command[0]).name
    _raise_failure(
        error_class,
        code=code,
        message=failure_message(name, result.returncode, result.stderr),
        argv=command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        extra_details=extra_details,
    )
