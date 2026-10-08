"""Release portability checks."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tomllib
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from tests._paths import PROJECT_ROOT

ROOT = PROJECT_ROOT


def test_package_sources_do_not_reference_local_user_paths():
    """Published package files must not bake in one developer's filesystem."""
    checked = [
        *sorted((ROOT / "src").rglob("*.py")),
        ROOT / "README.md",
        ROOT / "pyproject.toml",
    ]
    forbidden = (
        "/" + "home/",
        "/" + "Users/",
        "C:" + "\\Users\\",
        "/tmp/ov_",
        "data16t",
        "jiazc",
    )
    offenders: list[str] = []
    for path in checked:
        text = path.read_text(errors="replace")
        for pattern in forbidden:
            if pattern in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {pattern!r}")

    assert offenders == []


def test_tests_do_not_reference_local_user_paths_or_fixed_temp_outputs():
    """Tests should be runnable from any checkout without developer paths."""
    checked = sorted((ROOT / "tests").rglob("test_*.py"))
    forbidden = (
        "/" + "home/" + "jiazc",
        "data16t",
        "/tmp/ov_",
        "software/paml",
        "software/KaKs_Calculator",
    )
    offenders: list[str] = []
    for path in checked:
        if path.name == "test_portability.py":
            continue
        text = path.read_text(errors="replace")
        for pattern in forbidden:
            if pattern in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {pattern!r}")

    assert offenders == []


def test_pytest_import_mode_handles_duplicate_test_basenames():
    """The default pytest command must collect duplicate test module names."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pytest_config = pyproject["tool"]["pytest"]["ini_options"]
    addopts = pytest_config.get("addopts", [])
    if isinstance(addopts, str):
        addopts = addopts.split()

    test_basenames = Counter(path.name for path in (ROOT / "tests").rglob("test_*.py"))
    duplicate_basenames = {name for name, count in test_basenames.items() if count > 1}

    assert duplicate_basenames
    assert "--import-mode=importlib" in addopts


def test_release_archives_exclude_local_demo_artifacts():
    """Source releases exclude tests and internal development documents.

    Internal design docs (specs, plans) are allowed — expected, even — to
    live under version control at ``docs/superpowers/``: that is where the
    approved L7.1/L5.1 architecture and capability-migration plans this
    project is executing against actually live. What must not happen is
    those documents leaking into a *published* wheel or sdist. This is a
    build-configuration check only; ``test_release_archives_exclude_them``
    below builds the real archives and inspects their contents.
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    excludes = set(pyproject["tool"]["hatch"]["build"].get("exclude", []))

    assert "tests/**" in excludes
    assert "docs/superpowers/**" in excludes


@pytest.mark.slow
def test_release_archives_exclude_internal_plans(tmp_path: Path) -> None:
    """Build a real wheel and sdist; neither may contain docs/superpowers/.

    This inspects actual build output rather than trusting the
    pyproject.toml exclude declaration in isolation: the wheel is unaffected
    regardless (it only packages ``src/organelleverse``), but the sdist
    packages the project source tree and previously leaked exactly the 12
    ``docs/superpowers/**`` files this rule exists to keep out.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--sdist", "--outdir", str(tmp_path), str(ROOT)],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = sorted(tmp_path.glob("*.whl"))
    sdists = sorted(tmp_path.glob("*.tar.gz"))
    assert wheels, f"no wheel built:\n{completed.stdout}\n{completed.stderr}"
    assert sdists, f"no sdist built:\n{completed.stdout}\n{completed.stderr}"

    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = archive.namelist()
    with tarfile.open(sdists[0]) as archive:
        sdist_names = archive.getnames()

    assert wheel_names, "built wheel is unexpectedly empty"
    assert sdist_names, "built sdist is unexpectedly empty"
    assert not any("docs/superpowers" in name for name in wheel_names)
    assert not any("docs/superpowers" in name for name in sdist_names)
