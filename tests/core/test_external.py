"""Unit tests for organelleverse.core.external uniform command failures."""

from __future__ import annotations

import subprocess
import sys

import pytest

from organelleverse.core.errors import OrganelleDependencyError, OrganelleExecutionError
from organelleverse.core.external import (
    STDERR_TAIL_LINES,
    STDOUT_TAIL_LINES,
    failure_details,
    failure_message,
    raise_for_failure,
    run_external,
    tail_lines,
)


def _failing_command(exit_code: int = 3) -> list[str]:
    script = f'import sys; sys.stderr.write("boom\\n"); sys.exit({exit_code})'
    return [sys.executable, "-c", script]


def test_run_external_success_returns_completed_process() -> None:
    result = run_external([sys.executable, "-c", 'print("ok")'])
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_run_external_failure_raises_with_code_and_details() -> None:
    argv = _failing_command()
    with pytest.raises(OrganelleExecutionError) as caught:
        run_external(argv, code="test_tool_failed", tool="boomtool")
    error = caught.value
    assert error.code == "test_tool_failed"
    assert "boom" in error.message
    assert "boomtool" in error.message
    assert "exit code 3" in error.message
    assert list(error.details["argv"]) == argv
    assert error.details["returncode"] == 3
    assert "boom" in error.details["stderr_tail"]


def test_run_external_default_code() -> None:
    with pytest.raises(OrganelleExecutionError) as caught:
        run_external(_failing_command())
    assert caught.value.code == "external_command_failed"


def test_run_external_dependency_error_class_keeps_code() -> None:
    with pytest.raises(OrganelleDependencyError) as caught:
        run_external(
            _failing_command(),
            code="assembly.environment_unavailable",
            tool="hmmpress",
            error_class=OrganelleDependencyError,
        )
    assert caught.value.code == "assembly.environment_unavailable"
    assert "boom" in caught.value.message


def test_run_external_stderr_tail_is_limited_to_50_lines() -> None:
    payload = "\n".join(f"line{i}" for i in range(100)) + "\n"
    script = f"import sys; sys.stderr.write({payload!r}); sys.exit(1)"
    with pytest.raises(OrganelleExecutionError) as caught:
        run_external([sys.executable, "-c", script])
    lines = caught.value.details["stderr_tail"].splitlines()
    assert len(lines) == STDERR_TAIL_LINES
    assert lines[0] == "line50"
    assert lines[-1] == "line99"


def test_run_external_stdout_tail_is_limited_to_20_lines() -> None:
    payload = "\n".join(f"out{i}" for i in range(60)) + "\n"
    script = f"import sys; sys.stdout.write({payload!r}); sys.stderr.write('err\\n'); sys.exit(1)"
    with pytest.raises(OrganelleExecutionError) as caught:
        run_external([sys.executable, "-c", script])
    out_lines = caught.value.details["stdout_tail"].splitlines()
    assert len(out_lines) == STDOUT_TAIL_LINES
    assert out_lines[0] == "out40"
    assert out_lines[-1] == "out59"


def test_run_external_non_utf8_stderr_decodes_safely() -> None:
    script = 'import os,sys; os.write(2, b"\\xff\\xfe bad bytes\\n"); sys.exit(1)'
    with pytest.raises(OrganelleExecutionError) as caught:
        run_external([sys.executable, "-c", script])
    assert "bad bytes" in caught.value.details["stderr_tail"]
    assert isinstance(caught.value.details["stderr_tail"], str)


def test_run_external_timeout_reports_tail() -> None:
    script = (
        'import sys,time; sys.stderr.write("partial\\n"); sys.stderr.flush(); time.sleep(30)'
    )
    with pytest.raises(OrganelleExecutionError) as caught:
        run_external([sys.executable, "-c", script], timeout=2.0, tool="sleeper")
    assert "timed out" in caught.value.message
    assert caught.value.details["returncode"] is None
    assert "partial" in caught.value.details["stderr_tail"]


def test_run_external_missing_executable_reports_launch_error() -> None:
    with pytest.raises(OrganelleExecutionError) as caught:
        run_external(["/nonexistent/definitely-missing-organelleverse-bin"])
    assert "could not be started" in caught.value.message
    assert caught.value.details["returncode"] is None


def test_run_external_empty_argv_rejected() -> None:
    with pytest.raises(ValueError):
        run_external([])


def test_run_external_extra_details_merged() -> None:
    with pytest.raises(OrganelleExecutionError) as caught:
        run_external(_failing_command(), extra_details={"profile": "test-profile"})
    assert caught.value.details["profile"] == "test-profile"
    assert caught.value.details["returncode"] == 3


def test_raise_for_failure_passes_on_success() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "print('fine')"],
        capture_output=True,
        text=True,
        check=False,
    )
    raise_for_failure(result, code="unused")  # must not raise


def test_raise_for_failure_enriches_foreign_completed_process() -> None:
    result = subprocess.run(
        _failing_command(),
        capture_output=True,
        text=True,
        check=False,
    )
    with pytest.raises(OrganelleExecutionError) as caught:
        raise_for_failure(result, code="legacy_failed", tool="legacytool")
    assert caught.value.code == "legacy_failed"
    assert "boom" in caught.value.message
    assert list(caught.value.details["argv"]) == list(result.args)
    assert caught.value.details["returncode"] == 3


def test_tail_lines_bounds_long_lines() -> None:
    huge = "x" * 100_000
    tail = tail_lines(huge, 10, max_chars=1_000)
    assert len(tail) == 1_000


def test_failure_message_prefers_stderr_tail() -> None:
    message = failure_message("tool", 2, "first\nsecond")
    assert message == "tool failed with exit code 2: first\nsecond"


def test_failure_message_without_stderr() -> None:
    assert failure_message("tool", 2, "") == "tool failed with exit code 2"
    assert failure_message("tool", None, None) == "tool failed"


def test_failure_details_accepts_bytes_and_paths() -> None:
    from pathlib import Path

    details = failure_details(
        [Path("/usr/bin/tool"), "-f"], 1, stdout=b"out", stderr=b"err"
    )
    assert details["argv"] == ["/usr/bin/tool", "-f"]
    assert details["stdout_tail"] == "out"
    assert details["stderr_tail"] == "err"
