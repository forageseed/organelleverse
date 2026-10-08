"""Controlled pangenome subprocess environment tests."""

from __future__ import annotations

import sys
from pathlib import Path

from organelleverse.pangenome._runner import run_command


def test_absolute_environment_launcher_can_resolve_sibling_tools(tmp_path: Path) -> None:
    """A conda-style launcher must find tools installed beside itself."""
    helper = tmp_path / "backend-helper"
    helper.write_text("#!/usr/bin/env sh\nprintf sibling-ok\n", encoding="utf-8")
    helper.chmod(0o755)
    launcher = tmp_path / "backend-launcher"
    launcher.write_text("#!/usr/bin/env sh\nbackend-helper\n", encoding="utf-8")
    launcher.chmod(0o755)
    output = tmp_path / "stdout.log"

    record = run_command([str(launcher)], stdout_path=output, cwd=tmp_path)

    assert record.returncode == 0, record.stderr
    assert output.read_text(encoding="utf-8") == "sibling-ok"


def test_runner_keeps_only_a_bounded_stderr_tail(tmp_path: Path) -> None:
    script = tmp_path / "noisy.py"
    script.write_text(
        "import sys\nsys.stderr.write('x' * 20000 + 'END')\n",
        encoding="utf-8",
    )

    record = run_command([sys.executable, str(script)], cwd=tmp_path)

    assert record.returncode == 0
    assert len(record.stderr.encode("utf-8")) <= 16_384
    assert record.stderr.endswith("END")


def test_runner_measures_real_sleep_and_nonzero_exit_without_losing_streams(tmp_path: Path) -> None:
    import os

    output = tmp_path / "measured.stdout"
    record = run_command(
        [
            sys.executable,
            "-c",
            "import sys,time; print('retained stdout'); "
            "sys.stderr.write('retained stderr'); time.sleep(0.15); sys.exit(7)",
        ],
        stdout_path=output,
    )
    assert record.returncode == 7 and not record.ok
    assert output.read_text() == "retained stdout\n"
    assert record.stderr == "retained stderr"
    assert record.wall_seconds is not None and record.wall_seconds >= 0.15
    if hasattr(os, "wait4"):
        assert record.cpu_seconds is not None and record.cpu_seconds >= 0
        if sys.platform.startswith("linux") or sys.platform == "darwin":
            assert record.peak_memory_bytes is not None and record.peak_memory_bytes > 0
    else:
        assert record.cpu_seconds is None and record.peak_memory_bytes is None
        assert record.resource_usage["measurement"] == "wall_clock_only"


def test_runner_resource_defaults_do_not_fabricate_measurements() -> None:
    from organelleverse.pangenome._runner import CommandRecord

    record = CommandRecord(("stub",), 0, None, "")
    assert record.wall_seconds is record.cpu_seconds is record.peak_memory_bytes is None
    assert record.resource_usage["measurement"] == "unmeasured"


def test_concurrent_command_cpu_accounting_is_per_child(tmp_path: Path) -> None:
    import os
    from concurrent.futures import ThreadPoolExecutor

    import pytest

    if not hasattr(os, "wait4"):
        pytest.skip("Per-child CPU rusage requires wait4")
    burn = (
        "import time; a=bytearray(64*1024*1024); "
        "deadline=time.process_time()+0.35\n"
        "while time.process_time()<deadline: pass\n"
        "print('burn')"
    )
    sleep = "import time; time.sleep(0.6); print('sleep')"
    with ThreadPoolExecutor(max_workers=2) as pool:
        busy_future = pool.submit(
            run_command, [sys.executable, "-c", burn], stdout_path=tmp_path / "busy"
        )
        sleep_future = pool.submit(
            run_command, [sys.executable, "-c", sleep], stdout_path=tmp_path / "sleep"
        )
        busy_record, sleep_record = busy_future.result(), sleep_future.result()
    assert busy_record.ok and sleep_record.ok
    assert (tmp_path / "busy").read_text() == "burn\n"
    assert (tmp_path / "sleep").read_text() == "sleep\n"
    assert busy_record.cpu_seconds >= 0.35
    assert sleep_record.cpu_seconds < 0.2
    assert busy_record.cpu_seconds > sleep_record.cpu_seconds + 0.2
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        assert busy_record.peak_memory_bytes >= 64 * 1024 * 1024


def test_signal_exit_code_matches_subprocess_convention() -> None:
    import os
    import signal

    import pytest

    if os.name != "posix":
        pytest.skip("POSIX signal exit convention")
    record = run_command(
        [sys.executable, "-c", "import os,signal; os.kill(os.getpid(),signal.SIGTERM)"]
    )
    assert record.returncode == -signal.SIGTERM
