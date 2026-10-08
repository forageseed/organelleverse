"""Published third-party wheels are discoverable without a core checkout.

The fast discovery tests use a small ``DistributionLike`` double.  This module
proves the distribution boundary separately: it builds both packages, installs
them into a newly-created virtual environment, and asks a fresh interpreter to
discover the third-party capability through Python entry points.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._paths import PROJECT_ROOT

_PLUGIN_ID = "wheel.demo"
_PIP_INSTALL_TIMEOUT_SECONDS = 180


def _run(
    command: list[str], *, cwd: Path, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _write_plugin_project(project: Path) -> None:
    package = project / "src" / "wheel_demo"
    bundle = package / "bundles" / "wheel-demo"
    code = bundle / "code" / "wheel_impl"
    code.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "wheel-demo-plugin"
version = "0.1.0"
requires-python = ">=3.11"

[project.entry-points."organelleverse.capabilities"]
bundles = "wheel_demo:bundles"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
wheel_demo = ["bundles/**/*.toml", "bundles/**/*.py"]
""",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "bundles" / "__init__.py").write_text("", encoding="utf-8")
    (code / "__init__.py").write_text("", encoding="utf-8")
    (code / "impl.py").write_text(
        "def run(*, value: int = 1) -> dict[str, object]:\n    return {'value': value}\n",
        encoding="utf-8",
    )
    (bundle / "capability.toml").write_text(
        f"""\
schema = "organelleverse.capability.v1"

[capability]
id = "{_PLUGIN_ID}"
bundle_version = "1.0.0"
implementation = "native"

[contract]
contract_version = "1.0"
title = "Wheel discovery demo"
description = "A third-party capability installed from a real wheel."
keywords = ["discovery", "package", "wheel"]
execution_mode = "inline"
stage = "analyze"
input_kind = "none"
output_kind = "result"
callable_locator = "wheel_impl.impl:run"

[contract.binding]
argument_mode = "named_parameters"
result_codec = "canonical_json"

[[contract.binding.parameters]]
name = "value"
codec = "json"
""",
        encoding="utf-8",
    )


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


@pytest.mark.slow
def test_pip_installed_plugin_wheel_is_discovered_in_a_fresh_environment(tmp_path: Path) -> None:
    """A real package entry point reaches discovery after normal ``pip install``."""
    distributions = tmp_path / "distributions"
    distributions.mkdir()
    plugin_project = tmp_path / "wheel-demo-plugin"
    _write_plugin_project(plugin_project)

    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(distributions),
            str(PROJECT_ROOT),
        ],
        cwd=tmp_path,
    )
    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(distributions),
            str(plugin_project),
        ],
        cwd=tmp_path,
    )
    core_wheel = next(distributions.glob("organelleverse-*.whl"))
    plugin_wheel = next(distributions.glob("wheel_demo_plugin-*.whl"))

    environment = tmp_path / "fresh-environment"
    _run([sys.executable, "-m", "venv", str(environment)], cwd=tmp_path)
    python = _venv_python(environment)
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            str(core_wheel),
            str(plugin_wheel),
        ],
        cwd=tmp_path,
        timeout=_PIP_INSTALL_TIMEOUT_SECONDS,
    )

    probe = """
import json
from pathlib import Path
import organelleverse
from organelleverse.capabilities.discovery import discover_capabilities

entry = discover_capabilities(paths=()).describe("wheel.demo")
cache_paths = tuple((entry.bundle_root / "code").rglob("__pycache__/*.pyc"))
print(json.dumps({
    "core_path": str(Path(organelleverse.__file__).resolve()),
    "diagnostic": None if entry.diagnostic is None else entry.diagnostic.code,
    "origin": entry.origins[0].channel,
    "status": entry.status.value,
    "has_bytecode_cache": bool(cache_paths),
}))
"""
    child_environment = os.environ.copy()
    child_environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [str(python), "-I", "-c", probe],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env=child_environment,
    )
    observed = json.loads(completed.stdout)

    assert Path(observed["core_path"]).is_relative_to(environment)
    assert observed == {
        "core_path": observed["core_path"],
        "diagnostic": "capability.verification_missing",
        "origin": "package:wheel-demo-plugin",
        "status": "rejected",
        "has_bytecode_cache": True,
    }
