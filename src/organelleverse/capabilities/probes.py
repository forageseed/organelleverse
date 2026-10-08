"""Explicit verification-stage executable probes; never used during discovery/admission."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from pydantic import Field

from organelleverse.core.errors import OrganelleDependencyError
from organelleverse.core.external import MESSAGE_TAIL_LINES, STDERR_TAIL_LINES, tail_lines
from organelleverse.operations.spec import StrictSpecModel

from .hashing import hash_file
from .models import ProbeSpec

_MAX_CAPTURE = 1024 * 1024


class FastFileKey(StrictSpecModel):
    size: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)


class ProbeResult(StrictSpecModel):
    executable: Path
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    fast_key: FastFileKey
    observed_version: str = ""
    satisfied_requires: tuple[str, ...] = ()


def _run(executable: Path, argv: tuple[str, ...], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), *argv],
        shell=False,
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=executable.parent,
        env={"LC_ALL": "C", "LANG": "C", "PATH": os.defpath},
        start_new_session=os.name != "nt",
    )


def _combined_output(completed: subprocess.CompletedProcess[str]) -> str:
    return f"{completed.stdout or ''}\n{completed.stderr or ''}"[:_MAX_CAPTURE]


def probe_executable(executable: Path, spec: ProbeSpec, *, timeout: float = 30.0) -> ProbeResult:
    """Run a declared interface/version probe and freeze content identity evidence."""
    try:
        resolved = executable.resolve(strict=True)
    except OSError as error:
        raise OrganelleDependencyError(
            code="capability.executable_unavailable",
            message=f"executable is unavailable for {spec.dependency}: {executable}",
            details={"dependency": spec.dependency, "path": str(executable)},
        ) from error
    if not resolved.is_file():
        raise OrganelleDependencyError(
            code="capability.executable_unavailable",
            message=f"executable is not a regular file for {spec.dependency}: {resolved}",
            details={"dependency": spec.dependency, "path": str(resolved)},
        )
    help_result = _run(resolved, spec.help_argv, timeout=timeout)
    help_output = _combined_output(help_result)
    missing = tuple(token for token in spec.requires if token not in help_output)
    if missing:
        message = f"{spec.dependency} does not expose its declared interface"
        tail = tail_lines(help_result.stderr, MESSAGE_TAIL_LINES)
        if tail:
            message = f"{message}: {tail}"
        raise OrganelleDependencyError(
            code="capability.probe_requirements_missing",
            message=message,
            details={
                "dependency": spec.dependency,
                "missing": list(missing),
                "returncode": help_result.returncode,
                "stderr_tail": tail_lines(help_result.stderr, STDERR_TAIL_LINES),
            },
        )
    observed_version = ""
    if spec.version_argv:
        version_output = _combined_output(_run(resolved, spec.version_argv, timeout=timeout))
        if spec.version_capture:
            match = re.search(spec.version_capture, version_output)
            observed_version = match.group(1) if match is not None and match.groups() else ""
        else:
            observed_version = version_output.strip()
    stat = resolved.stat()
    return ProbeResult(
        executable=resolved,
        content_hash=hash_file(resolved),
        fast_key=FastFileKey(size=stat.st_size, mtime_ns=stat.st_mtime_ns),
        observed_version=observed_version,
        satisfied_requires=spec.requires,
    )


__all__ = ["FastFileKey", "ProbeResult", "probe_executable"]
