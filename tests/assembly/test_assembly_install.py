"""Tests for the assembly backend installer."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

import organelleverse as ov
from organelleverse.assembly.install import (
    BACKEND_INSTALL_INFO,
    check_all_backends,
    check_backend,
    install_backend,
    install_hint,
)

install_module = importlib.import_module("organelleverse.assembly.install")


@pytest.fixture(autouse=True)
def no_host_backend_discovery(monkeypatch, tmp_path):
    monkeypatch.setenv("ORGANELLEVERSE_TOOL_ROOT", str(tmp_path / "tools"))
    monkeypatch.setattr(install_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(install_module, "_list_conda_envs", lambda: [])


# -- check_backend --------------------------------------------------------


def test_check_backend_known():
    """check_backend returns a status dict for a known backend."""
    status = check_backend("getorganelle")
    assert status["name"] == "getorganelle"
    assert "installed" in status
    assert isinstance(status["installed"], bool)


def test_check_backend_unknown():
    status = check_backend("nonexistent_tool")
    assert status["installed"] is False
    assert "Unknown" in status["note"]


def test_check_all_backends():
    """check_all_backends returns status for the released backends."""
    statuses = check_all_backends()
    assert len(statuses) == 8
    for _name, status in statuses.items():
        assert "installed" in status
        assert "path" in status


# -- BACKEND_INSTALL_INFO -------------------------------------------------


def test_install_info_complete():
    """Every backend has at least one install method."""
    for name, info in BACKEND_INSTALL_INFO.items():
        has_method = any(k in info for k in ("pip", "conda", "source"))
        assert has_method, f"{name} has no install method"
        assert "cli" in info, f"{name} missing cli"
        assert "note" in info, f"{name} missing note"


def test_tier_assignment():
    """Each backend has a tier (pip/conda/source)."""
    for name, info in BACKEND_INSTALL_INFO.items():
        assert info["tier"] in ("pip", "conda", "source"), f"{name} bad tier"


# -- install_backend (dry-run, no actual install) -------------------------


def test_install_dry_run_pip():
    """dry-run shows the command without executing."""
    result = install_backend("getorganelle", dry_run=True, skip_if_installed=False)
    assert result["success"] is None  # dry-run
    assert "pip" in result["command"]
    assert "getorganelle" in result["command"]


def test_install_dry_run_conda():
    result = install_backend("oatk", method="conda", dry_run=True)
    assert result["success"] is None
    assert "conda" in result["command"]
    assert "oatk" in result["command"]


def test_himt_install_hint_preserves_specific_channel_and_package():
    hint = install_hint("himt")

    assert (
        "  conda:  conda install -c shuyuan_tang -c bioconda -c conda-forge himt=1.1.3=0"
    ) in hint


def test_himt_conda_dry_run_preserves_specific_channel_and_package():
    result = install_backend(
        "himt",
        method="conda",
        dry_run=True,
        skip_if_installed=False,
    )

    assert result["command"] == [
        "conda",
        "install",
        "-y",
        "-c",
        "shuyuan_tang",
        "-c",
        "bioconda",
        "-c",
        "conda-forge",
        "himt=1.1.3=0",
    ]


def test_conda_dry_run_deduplicates_default_channel():
    result = install_backend(
        "oatk",
        method="conda",
        dry_run=True,
        skip_if_installed=False,
    )

    assert result["command"] == [
        "conda",
        "install",
        "-y",
        "-c",
        "bioconda",
        "-c",
        "conda-forge",
        "oatk",
    ]


def test_himt_conda_install_preserves_specific_channel_and_package(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(install_module.subprocess, "run", fake_run)

    result = install_backend(
        "himt",
        method="conda",
        skip_if_installed=False,
    )

    assert result["success"] is True
    assert calls == [
        [
            "conda",
            "install",
            "-y",
            "-c",
            "shuyuan_tang",
            "-c",
            "bioconda",
            "-c",
            "conda-forge",
            "himt=1.1.3=0",
        ]
    ]


def test_install_dry_run_source():
    result = install_backend("pmat", method="source", dry_run=True)
    assert result["success"] is None
    cmd = result["command"]
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
    assert "git" in cmd_str
    assert "PMAT" in cmd_str


def test_install_unknown_backend():
    result = install_backend("nonexistent", dry_run=True)
    assert result["success"] is False
    assert "Unknown" in result["message"]


def test_install_auto_method():
    """auto picks pip when available (force skip check)."""
    result = install_backend("getorganelle", dry_run=True, skip_if_installed=False)
    assert result["method"] == "pip"


# -- Integration: public API access ---------------------------------------


def test_public_api_access():
    """The canonical assembly facade exposes installer functions."""
    assert hasattr(ov.assembly, "check_backend")
    assert hasattr(ov.assembly, "check_all_backends")
    assert hasattr(ov.assembly, "install_backend")
    statuses = ov.assembly.check_all_backends()
    assert len(statuses) == 8


# -- scan_envs: conda env detection ---------------------------------------


def test_check_backend_finds_mocked_path(monkeypatch, tmp_path):
    executable = tmp_path / "get_organelle_from_reads.py"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setattr(
        install_module.shutil,
        "which",
        lambda name: str(executable) if name == executable.name else None,
    )
    s = check_backend("getorganelle", scan_envs=False)
    assert s["installed"] is True
    assert s["path"] == str(executable)


def test_check_backend_env_field():
    """status dict includes 'env' field (may be None for PATH tools)."""
    s = check_backend("getorganelle")
    assert "env" in s


def test_check_backend_no_envs_scan():
    """scan_envs=False only checks PATH (faster)."""
    s = check_backend("getorganelle", scan_envs=False)
    assert s["installed"] is False


def test_check_backend_finds_mocked_conda_environment(monkeypatch, tmp_path):
    env_bin = tmp_path / "envs" / "assembly" / "bin"
    env_bin.mkdir(parents=True)
    executable = env_bin / "himt"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setattr(install_module, "_list_conda_envs", lambda: [env_bin])

    status = check_backend("himt")

    assert status["installed"] is True
    assert status["path"] == str(executable)
    assert status["env"] == "assembly"


# -- skip_if_installed flow -----------------------------------------------


def test_install_skips_if_found(monkeypatch, tmp_path):
    """install_backend skips when the mocked executable is found."""
    executable = tmp_path / "get_organelle_from_reads.py"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setattr(
        install_module.shutil,
        "which",
        lambda name: str(executable) if name == executable.name else None,
    )

    result = install_backend("getorganelle")

    assert result["method"] == "already_installed"
    assert result["success"] is True
    assert result["path"] == str(executable)


def test_install_forces_recheck():
    """skip_if_installed=False forces dry-run command."""
    result = install_backend("getorganelle", dry_run=True, skip_if_installed=False)
    assert result["method"] == "pip"
    assert result["success"] is None  # dry-run


def test_check_backend_finds_registered_managed_prefix(tmp_path):
    from organelleverse.assembly.environment_contracts import (
        ProviderComponentIdentity,
        ResolvedProvider,
    )
    from organelleverse.assembly.environment_registry import InstallationRegistry

    prefix = tmp_path / "managed-prefix"
    executable = prefix / "bin" / "get_organelle_from_reads.py"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    registry = InstallationRegistry()
    registry.register_verified(
        ResolvedProvider(
            requested_source="managed",
            discovery_source="managed",
            carrier="conda",
            platform="linux-64",
            prefix=prefix,
            capability_contract_digest="sha256:" + "a" * 64,
            components=(
                ProviderComponentIdentity(
                    role="getorganelle",
                    kind="executable",
                    path=executable,
                    sha256="b" * 64,
                    version="1.7.7.1",
                ),
            ),
        )
    )
    registry_path = registry.tool_root / "environments.json"
    before = registry_path.stat().st_mtime_ns
    status = check_backend("getorganelle")
    assert status["installed"] is True
    assert status["path"] == str(executable)
    assert status["env"] == str(prefix)
    assert status["method"] == "managed"
    assert registry_path.stat().st_mtime_ns == before
    assert check_backend("getorganelle", scan_envs=False)["installed"] is False
    executable.chmod(0o644)
    assert check_backend("getorganelle")["installed"] is False
    executable.unlink()
    assert check_backend("getorganelle")["installed"] is False


def test_check_backend_does_not_create_registry(tmp_path):
    assert check_backend("getorganelle")["installed"] is False
    assert not (tmp_path / "tools").exists()
