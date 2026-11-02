"""Deployment paths follow the execution account while preserving explicit settings."""

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from organelleverse.morphology import fusion3, orgseg_remote, orgseg_wsl


def test_local_delivery_defaults_use_the_current_account():
    assert orgseg_remote.DEFAULT_ORGSEG_LOCAL_ROOT == str(Path.home() / "organelleverse-orgseg-local")
    assert orgseg_remote.DEFAULT_ORGSEG_LOCAL_PYTHON == str(Path.home() / "orgsegnet-local-venv/bin/python")


def test_fusion_delivery_keeps_explicit_remote_configuration():
    assert fusion3.fusion3_env({"ORGANELLEVERSE_FUSION3_HOST": "lab", "ORGANELLEVERSE_FUSION3_ROOT": "/srv/science"}) == {"host": "lab", "root": "/srv/science"}
    assert not Path(fusion3.fusion3_env({})["root"]).is_absolute()


def test_wsl_defaults_resolve_the_wsl_account_instead_of_the_windows_account():
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        return CompletedProcess(argv, 0, "/srv/users/researcher\n", "")
    settings = orgseg_wsl._settings({}, _runner=runner)
    assert settings["root"] == "/srv/users/researcher/organelleverse-orgseg-local"
    assert settings["python"] == "/srv/users/researcher/orgsegnet-local-venv/bin/python"
    assert calls[0][:5] == ["wsl.exe", "--distribution", "Ubuntu", "--exec", "python3"]
    assert "--user" not in calls[0]


def test_explicit_wsl_settings_do_not_probe_or_replace_the_target():
    def runner(*args, **kwargs):
        raise AssertionError("Explicit paths do not need a discovery subprocess")
    settings = orgseg_wsl._settings({
        "ORGANELLEVERSE_WSL_DISTRIBUTION": "Science",
        "ORGANELLEVERSE_WSL_USER": "analyst",
        "ORGANELLEVERSE_WSL_ORGSEG_ROOT": "/opt/orgseg",
        "ORGANELLEVERSE_WSL_ORGSEG_PYTHON": "/opt/env/bin/python",
    }, _runner=runner)
    assert settings == {"distribution": "Science", "user": "analyst", "root": "/opt/orgseg", "python": "/opt/env/bin/python"}
    assert orgseg_wsl._prefix(settings) == ["wsl.exe", "--distribution", "Science", "--user", "analyst", "--exec"]


def test_wsl_home_discovery_failure_is_explicit():
    def runner(argv, **kwargs):
        return CompletedProcess(argv, 1, "", "distribution unavailable")
    with pytest.raises(RuntimeError, match="Cannot resolve"):
        orgseg_wsl._settings({}, _runner=runner)


def test_failed_wsl_home_probe_reports_not_ready(monkeypatch):
    monkeypatch.setattr(orgseg_wsl.sys, "platform", "win32")
    def runner(argv, **kwargs):
        return CompletedProcess(argv, 1, "", "offline")
    result = orgseg_wsl.check_orgseg_wsl(environ={}, _runner=runner)
    assert result["installed"] is False
    assert result["path"] is None
    assert "Cannot resolve" in result["note"]
