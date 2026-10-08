from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from organelleverse.capabilities.models import ProbeSpec
from organelleverse.capabilities.probes import probe_executable
from organelleverse.core.errors import OrganelleDependencyError
from organelleverse.environments import locate_executable


def test_locate_executable_is_path_only_and_returns_a_resolved_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "demo-tool"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    assert locate_executable("demo-tool") == executable.resolve()
    assert locate_executable("missing-tool") is None


def test_probe_uses_literal_argv_without_a_shell_and_records_interface_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "demo-tool"
    executable.write_bytes(b"executable bytes")
    calls: list[dict[str, Any]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"argv": argv, **kwargs})
        text = "--input ; touch /tmp/never --output" if "--help" in argv else "demo 4.2.1"
        return subprocess.CompletedProcess(argv, 0, stdout=text, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    probe = ProbeSpec(
        dependency="demo",
        help_argv=("--help", "; touch /tmp/never"),
        requires=("--input", "--output"),
        version_argv=("--version",),
        version_capture=r"demo ([0-9.]+)",
    )

    result = probe_executable(executable, probe)

    assert result.observed_version == "4.2.1"
    assert result.satisfied_requires == ("--input", "--output")
    assert result.content_hash.startswith("sha256:")
    assert calls[0]["argv"] == [str(executable.resolve()), "--help", "; touch /tmp/never"]
    assert calls[0]["shell"] is False
    assert calls[0]["stdin"] is subprocess.DEVNULL
    assert calls[0]["env"]["LC_ALL"] == "C"


def test_probe_missing_a_required_interface_token_fails_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "demo-tool"
    executable.write_bytes(b"tool")

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, stdout="--input", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    probe = ProbeSpec(
        dependency="demo",
        help_argv=("--help",),
        requires=("--input", "--output"),
    )

    with pytest.raises(OrganelleDependencyError) as captured:
        probe_executable(executable, probe)

    assert captured.value.code == "capability.probe_requirements_missing"


def test_probe_rejects_a_missing_executable_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess must not run")

    monkeypatch.setattr(subprocess, "run", fail_run)
    probe = ProbeSpec(dependency="demo", help_argv=("--help",))

    with pytest.raises(OrganelleDependencyError) as captured:
        probe_executable(tmp_path / "missing", probe)

    assert captured.value.code == "capability.executable_unavailable"


def test_probe_environment_does_not_inherit_python_startup_or_locale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "demo-tool"
    executable.write_bytes(b"tool")
    monkeypatch.setenv("PYTHONPATH", "/malicious")
    monkeypatch.setenv("LC_ALL", "fr_FR")
    observed: dict[str, str] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed.update(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    probe_executable(executable, ProbeSpec(dependency="demo", help_argv=("--help",)))

    assert observed == {"LC_ALL": "C", "LANG": "C", "PATH": os.defpath}
