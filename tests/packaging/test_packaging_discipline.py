"""T-G5 packaging discipline: static gates that must hold forever.

These encode the owner-ruled exclusions so a future dependency addition or
helper fails loudly in CI instead of silently shipping in the installer.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from tests._paths import PROJECT_ROOT

#: Second GUI toolkit + friends: our own workbench is the UI; these must
#: never enter any dependency list or the bundle.
BANNED_BUNDLE_PACKAGES = ("napari", "PyQt6", "magicgui", "superqt")

#: License-restricted resources the app must never download for the user.
LICENSE_RESTRICTED_HOSTS = ("zenodo.org",)


def _pyproject() -> dict:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())


def test_banned_gui_toolkit_packages_are_nowhere_in_dependency_lists() -> None:
    pyproject = _pyproject()
    all_deps = list(pyproject["project"]["dependencies"])
    for extra in pyproject["project"]["optional-dependencies"].values():
        all_deps.extend(extra)
    for requirement in all_deps:
        for banned in BANNED_BUNDLE_PACKAGES:
            assert banned.lower() not in requirement.lower(), (
                f"{banned} appeared in dependencies: {requirement!r}"
            )


def test_no_license_restricted_download_path_exists() -> None:
    """No code path downloads the OrgSegNet weights (or any Zenodo artifact)."""
    src = PROJECT_ROOT / "src" / "organelleverse"
    download_call = re.compile(r"urlretrieve|urlopen|requests\.get|httpx\.(get|stream)|urlretrieve")
    offenders: list[str] = []
    for path in src.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        text = path.read_text(encoding="utf-8")
        for host in LICENSE_RESTRICTED_HOSTS:
            for match in re.finditer(re.escape(host), text):
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                line = text[line_start:line_end if line_end != -1 else len(text)]
                if "http" in line and download_call.search(text):
                    # A URL string next to a download call in the same file:
                    # only fail when the URL is actually fed to a downloader.
                    if download_call.search(line):
                        offenders.append(f"{path.name}: {line.strip()[:80]}")
    assert offenders == []


def test_no_auto_install_framework_exists() -> None:
    """No downloader/managed-install machinery for advanced capabilities."""
    src = PROJECT_ROOT / "src" / "organelleverse"
    banned_patterns = (
        re.compile(r"def\s+\w*auto_?install", re.IGNORECASE),
        re.compile(r"pip\.main|pip_install|subprocess.*\bpip\b.*\binstall\b"),
    )
    offenders: list[str] = []
    for path in src.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in banned_patterns:
            for match in pattern.finditer(text):
                line_start = text.rfind("\n", 0, match.start()) + 1
                line = text[line_start : text.find("\n", match.end())]
                if line.strip().startswith("#"):
                    continue
                offenders.append(f"{path.relative_to(src)}: {line.strip()[:80]}")
    assert offenders == []


def test_packaging_code_never_targets_user_environments() -> None:
    """Nothing may write into a system Python or a user's conda env."""
    src = PROJECT_ROOT / "src" / "organelleverse"
    offenders: list[str] = []
    for path in src.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"site-packages", text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line = text[line_start : text.find("\n", match.end())]
            if re.search(r"write|mkdir|open\([^)]*[\"']w", line):
                offenders.append(f"{path.relative_to(src)}: {line.strip()[:80]}")
    assert offenders == []
