"""Backend installer + auto-locate for subcellular-localization tools.

Same pattern as the population/assembly installers: scan PATH + all conda
envs, and give install links when a tool is not found.

DeepLoc 2.1 (the current DTU release) ships on bioconda as
``deeploc2-2.1.0`` -- the conda package *bundles* the ESM1b (Fast) and ProtT5
(Accurate) model weights, so no separate weight download is needed. The DTU
academic license (non-commercial, no redistribution) means the weights/code
are NOT copied into OrganelleVerse; this module auto-locates an installed
``deeploc2`` on PATH or in any conda env.

TargetP 2.0 is a DTU standalone bundle (not on bioconda) with the same
academic-license restrictions.

The built-in ``heuristic`` backend needs none of these and runs offline.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..assembly.install import _list_conda_envs

__all__ = [
    "LOC_BACKEND_INFO",
    "check_backend",
    "check_all_backends",
    "install_hint",
]


LOC_BACKEND_INFO: dict[str, dict[str, str]] = {
    "deeploc": {
        "url": "https://services.healthtech.dtu.dk/services/DeepLoc-2.1/",
        "cli": "deeploc2",
        "note": (
            "DeepLoc 2.1 -- multi-label deep-learning predictor (13 "
            "compartments/descriptors incl. Plastid/Mitochondrion). Install via "
            "conda: `conda create -n deeploc -c bioconda deeploc2` (package "
            "deeploc2-2.1.0). The conda package bundles the ESM1b (Fast) and "
            "ProtT5 (Accurate) model weights, so no separate weight download is "
            "needed. Run: `deeploc2 -f <fa> -o <outdir> -m Fast|Accurate "
            "-d cpu|cuda|mps`. DTU academic license (non-commercial, no "
            "redistribution) -- model weights are NOT bundled into "
            "OrganelleVerse; the wrapper auto-locates an installed deeploc2."
        ),
        "tier": "conda",
    },
    "targetp": {
        "url": "https://services.healthtech.dtu.dk/services/TargetP-2.0/",
        "cli": "targetp",
        "note": (
            "TargetP 2.0 -- DTU N-terminal sorting-signal predictor "
            "(cTP/mTP/SP/OTHER). Standalone academic bundle from DTU (includes "
            "bundled models); run with `-org pl` for plants. Not on PyPI/"
            "bioconda. DTU academic license (non-commercial, no redistribution)."
        ),
        "tier": "manual",
    },
}


def check_backend(name: str, *, scan_envs: bool = True) -> dict[str, Any]:
    """Check if a localization backend is installed.

    Scans PATH + all conda/micromamba envs.
    Returns {name, installed, path, env, note}.
    """
    info = LOC_BACKEND_INFO.get(name)
    if not info:
        return {
            "name": name,
            "installed": False,
            "path": None,
            "env": None,
            "note": f"Unknown backend: {name}",
        }

    cli = info.get("cli", name)
    path = shutil.which(cli)
    if path:
        return {
            "name": name,
            "installed": True,
            "path": path,
            "env": None,
            "note": info.get("note", ""),
        }
    if scan_envs:
        for env_bin in _list_conda_envs():
            candidate = env_bin / cli
            if candidate.exists() and candidate.is_file():
                parts = env_bin.parts
                env_name = (
                    parts[parts.index("envs") + 1]
                    if "envs" in parts and parts.index("envs") + 1 < len(parts)
                    else None
                )
                note = f"{info.get('note', '')} [env: {env_name}]".strip()
                return {
                    "name": name,
                    "installed": True,
                    "path": str(candidate),
                    "env": env_name,
                    "note": note,
                }
    return {
        "name": name,
        "installed": False,
        "path": None,
        "env": None,
        "note": info.get("note", ""),
    }


def check_all_backends(*, scan_envs: bool = True) -> dict[str, dict[str, Any]]:
    """Check all localization backends. Returns {name: status_dict}."""
    return {name: check_backend(name, scan_envs=scan_envs) for name in LOC_BACKEND_INFO}


def install_hint(name: str) -> str:
    """Return a human-readable install guide with URLs."""
    info = LOC_BACKEND_INFO.get(name)
    if not info:
        return f"Unknown backend: {name}"
    url = info.get("url", "?")
    lines = [f"Backend '{name}' not found. Setup:"]
    lines.append(f"  docs: {url}")
    if name == "deeploc":
        lines.append("  conda: conda create -n deeploc -c bioconda deeploc2")
        lines.append(
            "  (conda package deeploc2-2.1.0 bundles the model weights -- no separate download)"
        )
        lines.append("  run:  deeploc2 -f <fa> -o <out> -m Fast|Accurate -d cpu|cuda|mps")
        lines.append(
            "  license: DTU academic (non-commercial, no "
            "redistribution -- weights are NOT bundled here)"
        )
    elif name == "targetp":
        lines.append(
            "  DTU standalone bundle (academic license); "
            "install per the page above, then use `-org pl`"
        )
    if info.get("note"):
        lines.append(f"  note: {info['note']}")
    lines.append("  fallback: backend='heuristic' runs offline, no install")
    return "\n".join(lines)
