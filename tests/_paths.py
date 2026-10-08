"""Stable repository paths shared by tests regardless of their directory depth."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def child_env(**overrides: str) -> dict[str, str]:
    """``os.environ`` with the repository's ``src/`` prepended to ``PYTHONPATH``.

    pytest puts ``src/`` on ``sys.path`` in-process (pyproject
    ``pythonpath=["src"]``), but that never reaches a spawned interpreter: a
    subprocess that imports organelleverse silently depends on an editable
    install being present, and otherwise resolves whatever ambient package
    happens to be importable — or fails outright on a pure source checkout.
    Prepending the checkout's ``src/`` makes every such subprocess exercise
    the tree under test, installed or not.
    """
    env = dict(os.environ)
    src = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    env.update(overrides)
    return env


__all__ = ["PROJECT_ROOT", "child_env"]
