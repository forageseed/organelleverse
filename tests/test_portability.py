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
        text = path.read_text(encoding="utf-8", errors="replace")
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
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in forbidden:
            if pattern in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {pattern!r}")

    assert offenders == []


def test_pytest_import_mode_handles_duplicate_test_basenames():
    """The default pytest command must collect duplicate test module names."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
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

    The build backend is setuptools, so MANIFEST.in decides what the sdist
    carries: tests are pruned and no rule includes anything under docs/.
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"
    rules = [
        line.split()
        for line in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert ["prune", "tests"] in rules
    including = {"include", "recursive-include", "graft", "global-include"}
    assert not any(
        rule[0] in including and any(arg.startswith("docs") for arg in rule[1:]) for rule in rules
    )
    # The documents this rule protects against leaking must actually exist in
    # the source tree; otherwise the exclude rule would be untested by the
    # build check below.
    assert any((ROOT / "docs" / "superpowers").rglob("*.md"))


@pytest.mark.slow
def test_release_archives_exclude_internal_plans(tmp_path: Path) -> None:
    """Build a real wheel and sdist; neither may contain docs/superpowers/.

    This inspects actual build output rather than trusting the
    pyproject.toml exclude declaration in isolation: the wheel is unaffected
    regardless (it only packages ``src``), but the sdist
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


def test_readme_assembly_quick_start_uses_released_contract():
    """The README assembly example must use the structured released API."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    snippet = readme[readme.index("### Assemble and assess a HiFi organelle genome") :]

    required = (
        "from pathlib import Path",
        "LongReadLibrary(",
        'technology="pacbio_hifi"',
        'quality_state="ccs"',
        'reads=Path("sample.hifi.fastq.gz")',
        "data = ov.read(",
        "ov.assembly.assemble(",
        'organelle="mitochondrion"',
        'method="oatk"',
        'ov.write(assembly_result, Path("results/sample/assembly"))',
    )
    for fragment in required:
        assert fragment in snippet

    assert "ov.read(" in snippet
    assert "ov.asm" not in snippet
