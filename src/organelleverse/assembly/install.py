"""Backend installer + dependency manager for assembly tools.

Each assembly backend has a different install method. This module provides:
  - ``check_backend(name)``     → {installed, path, version, method}
  - ``install_backend(name)``   → auto-installs via the best method
  - ``check_all_backends()``    → status of all released backends
  - ``BACKEND_INSTALL_INFO``    → install metadata for each backend

Three install tiers:
  1. **pip**    — pure Python packages (pip install name)
  2. **conda**  — channel-qualified Conda packages (conda install -c bioconda name)
  3. **source** — git clone + manual setup (GitHub repos)

OrganelleVerse never hard-requires any of these at import time.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.core.external import run_external

from .backends.registry import get_backend, install_info
from .environment_registry import InstallationRegistry

__all__ = [
    "BACKEND_INSTALL_INFO",
    "check_all_backends",
    "check_backend",
    "install_backend",
    "install_hint",
    "list_conda_envs",
]


def list_conda_envs() -> list[str]:
    """List all conda/micromamba environment bin/ directory paths.

    Convenience public wrapper around the internal scanner.
    """
    return [str(p) for p in _list_conda_envs()]


def _parse_conda_locator(locator: str) -> tuple[str, list[str]]:
    """Return a package and ordered channels from ``channel::package``."""
    specific_channel, separator, package = locator.partition("::")
    channels = [specific_channel] if separator else []
    channels.extend(["bioconda", "conda-forge"])
    return (package if separator else specific_channel), list(dict.fromkeys(channels))


def _build_conda_install_command(locator: str, *, assume_yes: bool) -> list[str]:
    """Build one conda install command from a package locator."""
    package, channels = _parse_conda_locator(locator)
    command = ["conda", "install"]
    if assume_yes:
        command.append("-y")
    for channel in channels:
        command.extend(["-c", channel])
    command.append(package)
    return command


def install_hint(name: str) -> str:
    """Return a human-readable install guide for a backend, including URLs.

    Example output::

        Backend 'getorganelle' not found. Install options:
          pip:    pip install getorganelle
          conda:  conda install -c bioconda -c conda-forge getorganelle
          docs:   https://github.com/Kinggerm/GetOrganelle
    """
    info = BACKEND_INSTALL_INFO.get(name)
    if not info:
        return f"Unknown backend: {name}"

    url = info.get("url", "?")
    pip = info.get("pip")
    conda = info.get("conda")
    source = info.get("source")

    lines = [f"Backend '{name}' not found. Install options:"]
    lines.append(f"  pip:    pip install {pip}" if pip else "  pip:    (not available)")
    if conda:
        command = _build_conda_install_command(conda, assume_yes=False)
        lines.append(f"  conda:  {' '.join(command)}")
    if source:
        lines.append(f"  source: git clone {source}")
    lines.append(f"  docs:   {url}")
    if info.get("note"):
        lines.append(f"  note:   {info['note']}")
    return "\n".join(lines)


BACKEND_INSTALL_INFO = install_info()


# =========================================================================
# Check: is a backend installed? (scan PATH + all conda/micromamba envs)
# =========================================================================


def _list_conda_envs() -> list[Path]:
    """Find all conda/micromamba environment bin/ directories.

    Searches:
    1. ``conda env list`` output
    2. ``micromamba env list`` output
    3. Common default locations (~/.conda, ~/miniconda3, ~/micromamba)
    """
    env_bins: list[Path] = []
    seen: set[str] = set()

    def _add_env_bin(env_path: str | Path) -> None:
        bin_dir = Path(env_path) / "bin"
        key = str(bin_dir)
        if bin_dir.is_dir() and key not in seen:
            seen.add(key)
            env_bins.append(bin_dir)

    # 1. conda env list
    conda_exe = shutil.which("conda")
    if conda_exe:
        try:
            result = subprocess.run(
                [conda_exe, "env", "list"], capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split()
                    # last column is the path (or '*' for active env)
                    env_path = (
                        parts[-1] if parts[-1] != "*" else (parts[-2] if len(parts) > 1 else "")
                    )
                    if env_path and env_path != "*":
                        _add_env_bin(env_path)
        except Exception:
            pass

    # 2. micromamba env list
    mm_exe = shutil.which("micromamba")
    if mm_exe:
        try:
            result = subprocess.run(
                [mm_exe, "env", "list"], capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "/" in line:
                    env_path = line.split()[-1]
                    _add_env_bin(env_path)
        except Exception:
            pass

    # 3. Common default locations (fallback if conda not on PATH)
    home = Path.home()
    for candidate in [
        home / "miniconda3",
        home / "anaconda3",
        home / "miniforge3",
        home / "micromamba",
        home / ".conda",
    ]:
        if candidate.is_dir():
            _add_env_bin(candidate)  # base env
            envs_dir = candidate / "envs"
            if envs_dir.is_dir():
                for env_dir in envs_dir.iterdir():
                    _add_env_bin(env_dir)

    return env_bins


def check_backend(name: str, *, scan_envs: bool = True) -> dict[str, Any]:
    """Check if an assembly backend is installed and findable.

    Scans in order:
    1. Current ``$PATH`` (fastest — tools in active env)
    2. Registered providers (if ``scan_envs=True``)
    3. All conda/micromamba environment ``bin/`` dirs (if ``scan_envs=True``)

    This is read-only executable discovery, not a runtime readiness check;
    assembly still verifies the provider and prepares its database.

    Returns::

        {
            "name": str,
            "installed": bool,
            "path": str | None,  # full path to the binary
            "env": str | None,  # conda env name if found in one
            "method": str,  # install tier (pip/conda/source)
            "note": str,
        }
    """
    info = BACKEND_INSTALL_INFO.get(name)
    if not info:
        return {
            "name": name,
            "installed": False,
            "path": None,
            "env": None,
            "method": "unknown",
            "note": f"Unknown backend: {name}",
        }

    cli = info.get("cli", name)

    # 1. Check current PATH (active env); the native binary the way it is run (env
    # override, checkout build, PATH)
    if get_backend(name).environment.carrier.value == "native":
        from organelleverse._ovasm import resolve_ovasm

        path = resolve_ovasm()
    else:
        path = shutil.which(cli)
    if path:
        return {
            "name": name,
            "installed": True,
            "path": path,
            "env": None,
            "method": info.get("tier", "unknown"),
            "note": info.get("note", ""),
        }

    # Managed prefixes are published outside conda's named environment list.
    if scan_envs:
        for provider in InstallationRegistry().locate(name):
            for component in provider.components:
                if (
                    component.role == name
                    and component.kind == "executable"
                    and component.path.is_file()
                    and os.access(component.path, os.X_OK)
                ):
                    return {
                        "name": name,
                        "installed": True,
                        "path": str(component.path),
                        "env": str(provider.prefix) if provider.prefix is not None else None,
                        "method": provider.source,
                        "note": info.get("note", ""),
                    }

        # Scan all conda/micromamba envs.
        for env_bin in _list_conda_envs():
            candidate = env_bin / cli
            if candidate.exists() and candidate.is_file():
                # derive env name from path: .../envs/assembly/bin → assembly
                parts = env_bin.parts
                env_name = None
                if "envs" in parts:
                    idx = parts.index("envs")
                    if idx + 1 < len(parts):
                        env_name = parts[idx + 1]
                elif "bin" in parts:
                    # base env: .../miniconda3/bin
                    env_name = parts[-2] if len(parts) >= 2 else None
                return {
                    "name": name,
                    "installed": True,
                    "path": str(candidate),
                    "env": env_name,
                    "method": info.get("tier", "unknown"),
                    "note": f"{info.get('note', '')} [found in env: {env_name}]".strip(),
                }

    return {
        "name": name,
        "installed": False,
        "path": None,
        "env": None,
        "method": info.get("tier", "unknown"),
        "note": info.get("note", ""),
    }


def check_all_backends(*, scan_envs: bool = True) -> dict[str, dict[str, Any]]:
    """Check all released assembly backends (scans PATH + conda envs).

    Returns {name: status_dict}.
    """
    return {name: check_backend(name, scan_envs=scan_envs) for name in BACKEND_INSTALL_INFO}


# =========================================================================
# Install: auto-install a backend via the best available method
# =========================================================================


def install_backend(
    name: str,
    *,
    method: str = "auto",
    dry_run: bool = False,
    skip_if_installed: bool = True,
) -> dict[str, Any]:
    """Install an assembly backend automatically.

    First **scans PATH + all conda envs** to check if already installed.
    If found, skips installation (unless ``skip_if_installed=False``).
    Otherwise tries methods in order: pip > conda > source (git clone).

    Parameters
    ----------
    name : backend name (e.g. "getorganelle", "himt")
    method : "auto" (default) | "pip" | "conda" | "source"
    dry_run : if True, print the command without executing
    skip_if_installed : if True (default), skip install when already found

    Returns {name, method, success, command, message}.
    """
    info = BACKEND_INSTALL_INFO.get(name)
    if not info:
        return {
            "name": name,
            "method": "unknown",
            "success": False,
            "command": "",
            "message": f"Unknown backend: {name}",
        }

    # 1. Scan first: already installed?
    if skip_if_installed:
        status = check_backend(name, scan_envs=True)
        if status["installed"]:
            env_info = f" (in env: {status['env']})" if status.get("env") else ""
            return {
                "name": name,
                "method": "already_installed",
                "success": True,
                "command": "",
                "path": status["path"],
                "message": f"Already installed at {status['path']}{env_info}. "
                f"Activate that env or use the full path.",
            }

    # decide method
    if method == "auto":
        method = _best_install_method(info)

    if dry_run:
        cmd, msg = _build_install_command(name, info, method)
        return {
            "name": name,
            "method": method,
            "success": None,
            "command": cmd,
            "message": f"[dry-run] {msg}",
        }

    if method == "pip" and info.get("pip"):
        return _install_pip(name, info)
    if method == "conda" and info.get("conda"):
        return _install_conda(name, info)
    if method == "source" and info.get("source"):
        return _install_source(name, info)
    return {
        "name": name,
        "method": method,
        "success": False,
        "command": "",
        "message": f"No install info for method={method!r}",
    }


def _best_install_method(info: dict[str, str]) -> str:
    """Pick the best install method for a backend."""
    if info.get("pip"):
        return "pip"
    if info.get("conda") and shutil.which("conda"):
        return "conda"
    return "source"


def _build_install_command(name: str, info: dict[str, str], method: str) -> tuple[list[str], str]:
    """Build the install command for a given method (for dry-run display)."""
    if method == "pip" and info.get("pip"):
        return [sys.executable, "-m", "pip", "install", info["pip"]], f"pip install {info['pip']}"
    if method == "conda" and info.get("conda"):
        command = _build_conda_install_command(info["conda"], assume_yes=True)
        display_command = _build_conda_install_command(info["conda"], assume_yes=False)
        return command, " ".join(display_command)
    if method == "source" and info.get("source"):
        return [
            "git",
            "clone",
            info["source"],
            f"./{name}",
        ], f"git clone {info['source']} (then follow {name}'s install instructions)"
    return [], "no command"


def _install_pip(name: str, info: dict[str, str]) -> dict[str, Any]:
    """Install via pip."""
    pkg = info["pip"]
    cmd = [sys.executable, "-m", "pip", "install", pkg]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        success = result.returncode == 0
        msg = "installed" if success else f"failed: {result.stderr[:200]}"
    except Exception as exc:
        success = False
        msg = f"error: {exc}"
    return {
        "name": name,
        "method": "pip",
        "success": success,
        "command": " ".join(cmd),
        "message": msg,
    }


def _install_conda(name: str, info: dict[str, str]) -> dict[str, Any]:
    """Install via conda using channels declared by the package locator."""
    cmd = _build_conda_install_command(info["conda"], assume_yes=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        success = result.returncode == 0
        msg = "installed" if success else f"failed: {result.stderr[:200]}"
    except Exception as exc:
        success = False
        msg = f"error: {exc}"
    return {
        "name": name,
        "method": "conda",
        "success": success,
        "command": " ".join(cmd),
        "message": msg,
    }


def _install_source(name: str, info: dict[str, str]) -> dict[str, Any]:
    """Clone from source (git). User must complete setup manually."""
    url = info["source"]
    target = Path(".") / name
    cmd = ["git", "clone", url, str(target)]
    try:
        run_external(cmd, timeout=300, tool="git")
        success = True
        note = info.get("note", "")
        msg = (
            f"Cloned to ./{name}. {note}. "
            f"Follow the repository's install instructions to complete setup."
        )
    except OrganelleExecutionError as error:
        success = False
        msg = f"failed: {error.message}"
    return {
        "name": name,
        "method": "source",
        "success": success,
        "command": " ".join(cmd),
        "message": msg,
    }
