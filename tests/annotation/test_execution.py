from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from organelleverse.annotation import execution
from organelleverse.annotation.execution import CommandRunner, resolve_required_tools
from organelleverse.core.errors import OrganelleDependencyError, OrganelleExecutionError


def test_missing_tool_names_every_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_tool(_name: str) -> None:
        return None

    monkeypatch.setattr(execution, "which", missing_tool)

    with pytest.raises(OrganelleDependencyError) as raised:
        resolve_required_tools(("blastn", "makeblastdb", "tblastn"))

    assert raised.value.code == "dependency_missing"
    assert raised.value.as_dict()["details"]["missing"] == [
        "blastn",
        "makeblastdb",
        "tblastn",
    ]


def test_nonzero_command_preserves_evidence_and_raises(tmp_path: Path) -> None:
    runner = CommandRunner(log_dir=tmp_path)

    with pytest.raises(OrganelleExecutionError) as raised:
        runner.run(
            (sys.executable, "-c", "import sys; print('bad'); sys.exit(2)"),
            stage="pcg_refinement",
            timeout=10,
        )

    assert raised.value.code == "backend_execution_failed"
    assert raised.value.as_dict()["details"]["returncode"] == 2
    records = _read_records(tmp_path / "commands.jsonl")
    assert records[0]["stage"] == "pcg_refinement"
    assert records[0]["stdout"] == "bad\n"
    assert records[0]["termination"] == "exit"


def test_timeout_is_recorded_before_structured_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_timeout(argv: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        timeout = cast(int, kwargs["timeout"])
        raise subprocess.TimeoutExpired(argv, timeout, output=b"partial\xff", stderr=b"late")

    monkeypatch.setattr(execution.subprocess, "run", fake_timeout)
    runner = CommandRunner(log_dir=tmp_path)

    with pytest.raises(OrganelleExecutionError) as raised:
        runner.run(("blastn", "-version"), stage="preflight", timeout=5)

    assert raised.value.code == "backend_execution_failed"
    assert raised.value.as_dict()["details"]["termination"] == "timeout"
    record = _read_records(tmp_path / "commands.jsonl")[0]
    assert record["returncode"] is None
    assert record["stdout"] == "partial�"


def test_signal_exit_is_named_and_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_signal(argv: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, -15, "", "terminated")

    monkeypatch.setattr(execution.subprocess, "run", fake_signal)
    runner = CommandRunner(log_dir=tmp_path)

    with pytest.raises(OrganelleExecutionError) as raised:
        runner.run(("tblastn", "-version"), stage="preflight", timeout=5)

    details = raised.value.as_dict()["details"]
    assert details["termination"] == "signal"
    assert details["signal"] == 15
    assert _read_records(tmp_path / "commands.jsonl")[0]["returncode"] == -15


def test_undecodable_output_is_replaced_not_lost(tmp_path: Path) -> None:
    runner = CommandRunner(log_dir=tmp_path)

    evidence = runner.run(
        (sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\xffok')"),
        stage="encoding",
        timeout=10,
    )

    assert evidence.stdout == "�ok"
    assert _read_records(tmp_path / "commands.jsonl")[0]["stdout"] == "�ok"


def test_version_capture_order_and_shell_free_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_which(name: str) -> str:
        return f"/tools/{name}"

    def fake_run(argv: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        name = Path(argv[0]).name
        return subprocess.CompletedProcess(argv, 0, f"{name} 1.2.3\n", "")

    monkeypatch.setattr(execution, "which", fake_which)
    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    runner = CommandRunner(log_dir=tmp_path)

    tools = resolve_required_tools(("blastn", "makeblastdb"), runner=runner)

    assert [(tool.name, tool.path, tool.version) for tool in tools] == [
        ("blastn", "/tools/blastn", "blastn 1.2.3"),
        ("makeblastdb", "/tools/makeblastdb", "makeblastdb 1.2.3"),
    ]
    assert [record.sequence for record in runner.records] == [1, 2]
    assert [record.stage for record in runner.records] == [
        "dependency_preflight:blastn",
        "dependency_preflight:makeblastdb",
    ]
    assert all(isinstance(argv, tuple) for argv, _kwargs in calls)
    assert all(kwargs["shell"] is False for _argv, kwargs in calls)
    assert all(kwargs["check"] is False for _argv, kwargs in calls)
    assert all(kwargs["text"] is True for _argv, kwargs in calls)


def test_tool_resolution_preserves_multicall_symlink_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "cmalign"
    executable.write_text(
        "#!/bin/sh\n"
        'name="$(basename "$0")"\n'
        'printf "# %s :: test program\\n" "$name"\n'
        'printf "# INFERNAL 1.1.4 (Dec 2020)\\n"\n'
    )
    executable.chmod(0o755)
    symlink = tmp_path / "cmsearch"
    symlink.symlink_to(executable.name)

    def locate_symlink(_name: str) -> str:
        return str(symlink)

    monkeypatch.setattr(execution, "which", locate_symlink)

    tool = resolve_required_tools(("cmsearch",))[0]

    assert tool.path == str(symlink.absolute())
    assert tool.version == "INFERNAL 1.1.4 (Dec 2020)"


def test_tool_resolution_rejects_wrong_program_banner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "cmsearch"
    executable.write_text(
        "#!/bin/sh\n"
        'printf "# cmalign :: wrong program\\n"\n'
        'printf "# INFERNAL 1.1.4 (Dec 2020)\\n"\n'
    )
    executable.chmod(0o755)

    def locate_executable(_name: str) -> str:
        return str(executable)

    monkeypatch.setattr(execution, "which", locate_executable)

    with pytest.raises(OrganelleDependencyError) as raised:
        resolve_required_tools(("cmsearch",))

    assert raised.value.code == "dependency_identity_mismatch"


def test_trnascan_version_probe_uses_supported_help_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_which(name: str) -> str:
        return f"/tools/{name}"

    monkeypatch.setattr(execution, "which", fake_which)

    def fake_run(argv: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv,
            0,
            "Usage: tRNAscan-SE [-options] <FASTA file(s)>\n",
            "tRNAscan-SE 2.0.12 (Nov 2022)\n",
        )

    monkeypatch.setattr(execution.subprocess, "run", fake_run)

    tools = resolve_required_tools(("tRNAscan-SE",), runner=CommandRunner(log_dir=tmp_path))

    assert calls == [("/tools/tRNAscan-SE", "-h")]
    assert tools[0].version == "tRNAscan-SE 2.0.12 (Nov 2022)"
    assert tools[0].version_argv == ("/tools/tRNAscan-SE", "-h")


def test_cmsearch_version_probe_records_infernal_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_which(name: str) -> str:
        return f"/tools/{name}"

    monkeypatch.setattr(execution, "which", fake_which)

    def fake_run(argv: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv,
            0,
            "# cmsearch :: search CM(s) against a sequence database\n# INFERNAL 1.1.4 (Dec 2020)\n",
            "",
        )

    monkeypatch.setattr(execution.subprocess, "run", fake_run)

    tools = resolve_required_tools(("cmsearch",), runner=CommandRunner(log_dir=tmp_path))

    assert tools[0].version == "INFERNAL 1.1.4 (Dec 2020)"


def test_released_trnascan_failure_cannot_be_converted_to_empty_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from organelleverse.annotation.mitochondrion.trna import annotate_trna

    def fake_failure(argv: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 3, "", "tRNAscan failed")

    monkeypatch.setattr(execution.subprocess, "run", fake_failure)
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">r1\nACGT\n")
    runner = CommandRunner(log_dir=tmp_path / "evidence")

    with pytest.raises(OrganelleExecutionError) as raised:
        annotate_trna(
            fasta,
            tmp_path / "output",
            threads=2,
            engine="trnascan_se",
            tool_paths={
                "tRNAscan-SE": "/tools/tRNAscan-SE",
                "cmsearch": "/tools/cmsearch",
            },
            command_runner=runner,
        )

    assert raised.value.code == "backend_execution_failed"
    assert runner.records[0].stage == "trna:trnascan_se"
    assert runner.records[0].returncode == 3


def _read_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        value: object = json.loads(line)
        assert isinstance(value, dict)
        records.append(cast(dict[str, object], value))
    return records
