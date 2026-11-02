"""Packaged LOSAT/ovasm binaries are found after the env override and before PATH."""

from __future__ import annotations

import sys

import pytest

from organelleverse import _bundled, _losat, _ovasm
from organelleverse.annotation.plastome import tools as plastome_tools


@pytest.mark.parametrize(
    ("platform_name", "machine", "expected"),
    [
        ("linux", "x86_64", "linux-x86_64"),
        ("linux", "aarch64", "linux-arm64"),
        ("darwin", "arm64", "macos-arm64"),
        ("darwin", "x86_64", "macos-x86_64"),
        ("win32", "AMD64", "windows-x86_64"),
        ("freebsd14", "amd64", None),
        ("linux", "riscv64", None),
    ],
)
def test_platform_tag(monkeypatch, platform_name, machine, expected):
    monkeypatch.setattr(_bundled.sys, "platform", platform_name)
    monkeypatch.setattr(_bundled.platform, "machine", lambda: machine)
    assert _bundled.platform_tag() == expected


@pytest.fixture
def packaged(tmp_path, monkeypatch):
    tag = _bundled.platform_tag()
    if tag is None:
        pytest.skip("no packaged builds for this platform")
    monkeypatch.setattr(_bundled, "_BIN_ROOT", tmp_path)
    folder = tmp_path / tag
    folder.mkdir()
    suffix = ".exe" if sys.platform == "win32" else ""
    made = {}
    for name in ("LOSAT", "ovasm"):
        binary = folder / f"{name}{suffix}"
        binary.write_bytes(b"")
        binary.chmod(0o644)  # resolution restores the mode bits archives drop
        made[name] = str(binary)
    for var in ("ORG_VERSE_LOSAT_BIN", "ORG_VERSE_OVASM_BIN"):
        monkeypatch.delenv(var, raising=False)
    return made


def test_packaged_binaries_win_over_path(packaged):
    assert _losat.resolve_losat() == packaged["LOSAT"]
    assert plastome_tools._resolve_losat(None) == packaged["LOSAT"]
    assert _ovasm.resolve_ovasm() == packaged["ovasm"]


def test_env_override_and_opt_out_still_win(packaged, tmp_path, monkeypatch):
    own = tmp_path / "own-ovasm"
    own.write_bytes(b"")
    monkeypatch.setenv("ORG_VERSE_OVASM_BIN", str(own))
    assert _ovasm.resolve_ovasm() == str(own)
    monkeypatch.setenv("ORG_VERSE_LOSAT_BIN", "none")
    assert _losat.resolve_losat() is None
    assert plastome_tools._resolve_losat(None) is None


def test_missing_packaged_build_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(_bundled, "_BIN_ROOT", tmp_path)
    assert _bundled.bundled_executable("ovasm") is None
