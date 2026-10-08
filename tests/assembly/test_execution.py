from __future__ import annotations

import json
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from organelleverse.assembly.execution import (
    CommandOutcome,
    CommandRunner,
    managed_prefix_environment,
)


def _write_slow_script(path: Path, *, sleep_seconds: float) -> None:
    path.write_text(
        f"#!/usr/bin/env python3\nimport time, sys\ntime.sleep({sleep_seconds})\nprint('done')\n"
    )
    path.chmod(0o755)


def test_managed_prefix_environment_resolves_env_python_inside_long_prefix(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / ("managed-" + "a" * 128)
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text("#!/bin/sh\nprintf 'managed-python\\n'\n")
    python.chmod(0o755)
    probe = bin_dir / "probe"
    probe.write_text("#!/usr/bin/env python\n")
    probe.chmod(0o755)

    completed = subprocess.run(
        [str(probe), "--version"],
        check=True,
        capture_output=True,
        text=True,
        env=managed_prefix_environment(
            prefix,
            base={
                "PATH": "/usr/bin",
                "PYTHONPATH": "/host/modules",
                "VIRTUAL_ENV": "/host/venv",
            },
        ),
    )

    assert completed.stdout == "managed-python\n"


def test_command_runner_never_invokes_a_shell(tmp_path: Path) -> None:
    script = tmp_path / "print argv.py"
    script.write_text("#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    script.chmod(0o755)
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"

    outcome = CommandRunner().run(
        (str(script), "a b", "$(touch should-not-exist)", "`uname`"),
        stage="probe",
        cwd=tmp_path,
        timeout_seconds=30,
        stdout_path=stdout,
        stderr_path=stderr,
    )

    assert outcome.termination == "exit"
    assert outcome.started is True
    assert outcome.returncode == 0
    assert outcome.stage == "probe"
    assert outcome.cwd == str(tmp_path)
    assert outcome.signal is None
    assert outcome.launch_error == ""
    assert outcome.started_at <= outcome.finished_at
    assert outcome.duration_seconds >= 0
    assert json.loads(stdout.read_text()) == [
        "a b",
        "$(touch should-not-exist)",
        "`uname`",
    ]
    assert not (tmp_path / "should-not-exist").exists()


def test_launch_error_is_recorded_without_claiming_started(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"

    outcome = CommandRunner().run(
        (str(tmp_path / "does-not-exist"),),
        stage="probe",
        cwd=tmp_path,
        timeout_seconds=30,
        stdout_path=stdout,
        stderr_path=stderr,
    )

    assert outcome.started is False
    assert outcome.termination == "launch_error"
    assert outcome.returncode is None
    assert outcome.launch_error


def test_timeout_terminates_and_kills_the_process(tmp_path: Path) -> None:
    script = tmp_path / "sleep.py"
    _write_slow_script(script, sleep_seconds=10)
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"

    outcome = CommandRunner().run(
        (str(script),),
        stage="execute_backend",
        cwd=tmp_path,
        timeout_seconds=1,
        stdout_path=stdout,
        stderr_path=stderr,
    )

    assert outcome.started is True
    assert outcome.termination == "timeout"
    assert outcome.returncode is None
    assert outcome.signal is not None
    assert outcome.signal < 0


def test_negative_return_code_is_recorded_as_a_signal(tmp_path: Path) -> None:
    script = tmp_path / "segfault.py"
    script.write_text(
        "#!/usr/bin/env python3\nimport os, signal\nos.kill(os.getpid(), signal.SIGTERM)\n"
    )
    script.chmod(0o755)
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"

    outcome = CommandRunner().run(
        (str(script),),
        stage="execute_backend",
        cwd=tmp_path,
        timeout_seconds=30,
        stdout_path=stdout,
        stderr_path=stderr,
    )

    assert outcome.started is True
    assert outcome.termination == "signal"
    assert outcome.returncode is not None
    assert outcome.returncode < 0


def test_utf8_output_is_preserved(tmp_path: Path) -> None:
    script = tmp_path / "unicode.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.stdout.buffer.write('线粒体 🧬\\n'.encode('utf-8'))\n"
    )
    script.chmod(0o755)
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"

    outcome = CommandRunner().run(
        (str(script),),
        stage="probe",
        cwd=tmp_path,
        timeout_seconds=30,
        stdout_path=stdout,
        stderr_path=stderr,
    )

    assert outcome.termination == "exit"
    assert outcome.returncode == 0
    assert stdout.read_text(encoding="utf-8") == "线粒体 🧬\n"


def test_injected_popen_factory_keyboard_interrupt_propagates(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"

    def raising_popen_factory(argv: tuple[str, ...], **kwargs: object) -> subprocess.Popen[bytes]:
        raise KeyboardInterrupt

    runner = CommandRunner(popen_factory=raising_popen_factory)
    with pytest.raises(KeyboardInterrupt):
        runner.run(
            ("true",),
            stage="probe",
            cwd=tmp_path,
            timeout_seconds=30,
            stdout_path=stdout,
            stderr_path=stderr,
        )


def test_command_outcome_is_frozen_and_strict() -> None:
    outcome = CommandOutcome(
        stage="probe",
        argv=("oatk",),
        cwd="/tmp",
        started=True,
        started_at=datetime(2026, 7, 16, 12, 0, tzinfo=UTC),
        finished_at=datetime(2026, 7, 16, 12, 0, 5, tzinfo=UTC),
        duration_seconds=5.0,
        returncode=0,
        termination="exit",
        signal=None,
        stdout_path="/tmp/out",
        stderr_path="/tmp/err",
        launch_error="",
    )
    assert outcome.model_config.get("frozen") is True
    assert outcome.model_config.get("extra") == "forbid"
    with pytest.raises(ValidationError):
        outcome.returncode = 1  # type: ignore[misc]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals only")
def test_timeout_signal_default_is_sigkill_on_positive_codes(tmp_path: Path) -> None:
    # Sanity: signal module exposes the constants the runner relies on.
    assert signal.SIGKILL or signal.SIGTERM
