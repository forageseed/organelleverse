"""User-space micromamba and side-by-side versioned tool environments.

Everything lives under the persistent tool root (``default_tool_root()``), so
nothing here needs root or writes outside the user's home:

    <tool_root>/bin/micromamba          pinned static binary, sha256-checked
    <tool_root>/mamba/                  MAMBA_ROOT_PREFIX (package cache)
    <tool_root>/envs/<tool>/<version>/  one prefix per installed version

Several versions of one tool coexist as sibling prefixes. Discovery lists them
newest first, so callers that take the first hit get the newest version unless
a caller asks for a specific one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .environment_registry import default_tool_root
from .environment_specs import normalize_platform

__all__ = [
    "MICROMAMBA_VERSION",
    "bootstrap_micromamba",
    "carrier_environment",
    "find_micromamba",
    "installed_versions",
    "managed_envs_root",
    "managed_micromamba_path",
    "package_version",
    "valid_version",
    "version_sort_key",
]

MICROMAMBA_VERSION = "2.8.1-1"
_RELEASE_URL = (
    "https://github.com/mamba-org/micromamba-releases/releases/download/"
    f"{MICROMAMBA_VERSION}/micromamba-{{platform}}"
)
# Published next to each release asset as micromamba-<platform>.sha256.
_MICROMAMBA_SHA256 = {
    "linux-64": "77b7790ec97f64581118f103585b175df4306f95829b0fa6bfe4a19cc88a1182",
    "linux-aarch64": "49aa29008cafa6cd6027ebd643fc43ca32c20740e60b0a05378c4e5bb837c217",
    "osx-64": "d6fce18e56d7c6bf2331b0ee1b372a581c70f09b509cc9e924cdd131e053b58a",
    "osx-arm64": "9618a2866a2ffb3d36b55e9520f64d63dcd6dc2e622a351ca3cbe8e2cc90c757",
}

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}$")


def managed_micromamba_path(tool_root: Path | None = None) -> Path:
    return (tool_root or default_tool_root()) / "bin" / "micromamba"


def managed_envs_root(tool_root: Path | None = None) -> Path:
    return (tool_root or default_tool_root()) / "envs"


def find_micromamba(tool_root: Path | None = None) -> str | None:
    """A micromamba already on PATH wins (it keeps the user's own config);
    otherwise the managed copy, if it has been bootstrapped."""
    on_path = shutil.which("micromamba")
    if on_path:
        return on_path
    managed = managed_micromamba_path(tool_root)
    if managed.is_file() and os.access(managed, os.X_OK):
        return str(managed)
    return None


def carrier_environment(executable: str, tool_root: Path | None = None) -> dict[str, str]:
    """Environment for running *executable*. Only the managed binary gets a
    root prefix of ours; a user's own micromamba keeps its configured one."""
    env = dict(os.environ)
    if Path(executable) == managed_micromamba_path(tool_root):
        env.setdefault("MAMBA_ROOT_PREFIX", str((tool_root or default_tool_root()) / "mamba"))
    return env


def bootstrap_micromamba(
    log: Callable[[str], None],
    *,
    tool_root: Path | None = None,
    platform: str | None = None,
    opener: Callable[[str], object] | None = None,
) -> Path | None:
    """Download the pinned micromamba into the tool root without root rights.

    The binary is staged next to its destination, checked against the pinned
    sha256, probed with ``--version`` and only then moved into place.
    """
    resolved = platform or normalize_platform()
    expected = _MICROMAMBA_SHA256.get(resolved)
    if expected is None:
        log(f"no pinned micromamba build for platform {resolved}")
        return None
    destination = managed_micromamba_path(tool_root)
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = _RELEASE_URL.format(platform=resolved)
    log(f"downloading micromamba {MICROMAMBA_VERSION} ({resolved}) into {destination.parent}")
    fd, staged_name = tempfile.mkstemp(prefix=".micromamba.", dir=destination.parent)
    staged = Path(staged_name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as target, (opener or _open)(url) as source:  # type: ignore[attr-defined]
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected:
            log("micromamba checksum mismatch; bootstrap stopped")
            return None
        staged.chmod(staged.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        probe = subprocess.run([str(staged), "--version"], capture_output=True, text=True, timeout=30)
        if probe.returncode != 0:
            log(f"micromamba did not pass its version probe: {probe.stderr.strip()[:200]}")
            return None
        os.replace(staged, destination)
        log(f"micromamba {probe.stdout.strip()} ready")
        return destination
    except (OSError, subprocess.TimeoutExpired) as error:
        log(f"micromamba bootstrap failed: {error}")
        return None
    finally:
        staged.unlink(missing_ok=True)


def _open(url: str):  # pragma: no cover - network
    return urllib.request.urlopen(url, timeout=60)


def valid_version(value: str) -> bool:
    return bool(_VERSION_RE.match(value)) and value not in {".", ".."}


def version_sort_key(version: str) -> tuple[tuple[int, int | str], ...]:
    """Numeric-aware key: 1.10 sorts after 1.9; text parts sort before numbers."""
    parts = re.split(r"[._+-]", version)
    return tuple((1, int(part)) if part.isdigit() else (0, part) for part in parts)


def installed_versions(tool: str, tool_root: Path | None = None) -> list[tuple[str, Path]]:
    """(version, prefix) for each managed install of *tool*, newest first.
    Staging directories (dot-prefixed) are ignored."""
    root = managed_envs_root(tool_root) / tool
    if not root.is_dir():
        return []
    found = [
        (entry.name, entry)
        for entry in root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".") and (entry / "conda-meta").is_dir()
    ]
    return sorted(found, key=lambda item: version_sort_key(item[0]), reverse=True)


def package_version(prefix: Path, package: str) -> str | None:
    """The installed version of *package* according to the prefix's conda-meta."""
    for record in (prefix / "conda-meta").glob(f"{package}-*.json"):
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("name") == package and isinstance(data.get("version"), str):
            return data["version"]
    return None
