"""Tests for the ``verify-all`` batch CLI (the owner-approved 'verify once'
model applied to every discovered capability).

Runs the real CLI entry point against the real core bundles in an isolated
ORGANELLEVERSE_HOME, then proves the four counters move: verification records
persist, admission flips rejected -> admitted, and ``ov.operations`` grows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from organelleverse.tools import capability_cli


@pytest.fixture()
def _iso_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    return home


def _parse_cli_json(captured: str) -> dict:
    """Parse the pretty-printed JSON document the CLI prints after its NOTICE."""
    lines = captured.splitlines()
    start = next(i for i, line in enumerate(lines) if line == "{")
    return json.loads("\n".join(lines[start:]))


def test_verify_all_admits_every_discovered_capability(
    _iso_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # before: all core bundles are rejected for lack of verification records
    from organelleverse.capabilities import discover_capabilities

    before = discover_capabilities()
    assert len(before.entries) > 0
    assert all(str(entry.status) == "rejected" for entry in before.entries)

    exit_code = capability_cli.main(["verify-all", "--json"])
    captured = capsys.readouterr().out
    assert exit_code == 0
    payload = _parse_cli_json(captured)
    assert payload["summary"]["discovered"] == len(before.entries)
    assert payload["summary"]["verified"] == len(before.entries)
    assert payload["summary"]["failed"] == 0
    assert len(payload["verified"]) == len(before.entries)

    # after: the same discovery now admits everything, with real records on disk
    after = discover_capabilities()
    assert all(str(entry.status) == "admitted" for entry in after.entries)
    records = list((_iso_home / "verifications").rglob("*.json"))
    assert len(records) == len(before.entries)


def test_verify_all_grows_the_operation_registry(_iso_home: Path) -> None:
    # The in-process default registry is constructed at import time, before the
    # records exist, so growth is asserted in a fresh session (subprocess) —
    # the same semantics a real owner gets after running verify-all once.
    import subprocess
    import sys

    from tests._paths import child_env

    exit_code = capability_cli.main(["verify-all"])
    assert exit_code == 0
    probe = "import organelleverse as ov;print(len(ov.operations.list()))"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=child_env(),
        check=True,
    )
    after = int(result.stdout.strip())
    assert after >= 170  # 4 residual + the core bundle surface


def test_verify_all_stop_on_first_error_without_flag(
    _iso_home: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # make the very first capability fail structurally: an unknown id cannot be
    # described, so a broken index yields a controlled failure path
    from organelleverse.core.errors import OrganelleContractError

    def _boom(*args, **kwargs):
        raise OrganelleContractError(
            code="capability.verification_missing", message="injected failure"
        )

    monkeypatch.setattr("organelleverse.tools.capability_cli.verify_capability", _boom)
    exit_code = capability_cli.main(["verify-all"])
    assert exit_code == 1


def test_verify_all_continue_on_error_reports_failures(
    _iso_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # scaffold a broken third-party bundle next to the core surface: a native
    # bundle whose callable is absent fails verification with a structured error
    from organelleverse.capabilities.scaffold import scaffold

    root = Path(_iso_home).parent / "search"
    scaffold(
        root / "broken",
        capability_id="thirdparty.broken",
        title="Broken",
        description="a deliberately broken bundle that fails verification",
    )
    # break the generated implementation at import time (scaffold's default
    # verification inspects the module — a body-level raise would not trip it)
    impl = next((root / "broken" / "code").glob("*/impl.py"))
    impl.write_text("raise RuntimeError('boom at import')\n", encoding="utf-8")

    exit_code = capability_cli.main(
        ["verify-all", "--search", str(root), "--continue-on-error", "--json"]
    )
    assert exit_code == 1  # failures present -> nonzero even when continuing
    captured = capsys.readouterr().out
    payload = _parse_cli_json(captured)
    assert payload["summary"]["failed"] >= 1
    assert any(f["capability_id"] == "thirdparty.broken" for f in payload["failures"])
    assert payload["summary"]["verified"] == 0  # only the broken bundle is in scope
