"""Executables shipped inside the package (``_bin/<platform>/``).

Published OrganelleVerse sources carry prebuilt LOSAT and ovasm for each
supported platform (see ``_bin/MANIFEST.json``). Development checkouts
usually have none and fall through to the vendored build or ``PATH``.
"""

from __future__ import annotations

import os
import platform
import stat
import sys
from pathlib import Path

__all__ = ["bundled_executable", "platform_tag"]

_MACHINES = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}
_SYSTEMS = {"linux": "linux", "darwin": "macos", "win32": "windows"}
_BIN_ROOT = Path(__file__).resolve().parent / "_bin"


def platform_tag() -> str | None:
    """``<system>-<machine>`` of this interpreter (``linux-x86_64``, ``macos-arm64`` ...)."""

    system = _SYSTEMS.get("linux" if sys.platform.startswith("linux") else sys.platform)
    machine = _MACHINES.get(platform.machine().lower())
    if system is None or machine is None:
        return None
    return f"{system}-{machine}"


def bundled_executable(name: str) -> str | None:
    """Path of the packaged ``name`` binary for this platform, or ``None``."""

    tag = platform_tag()
    if tag is None:
        return None
    filename = f"{name}.exe" if sys.platform == "win32" else name
    candidate = _BIN_ROOT / tag / filename
    if not candidate.is_file():
        return None
    if sys.platform != "win32" and not os.access(candidate, os.X_OK):
        # Archives and some installers drop the mode bits; restore them once.
        try:
            mode = candidate.stat().st_mode
            candidate.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError:
            return None
        if not os.access(candidate, os.X_OK):
            return None
    return str(candidate)
