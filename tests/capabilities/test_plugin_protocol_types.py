"""Plugin Plan 02, Task 1: PluginContext/PluginResult core types.

``organelleverse.plugin_protocol`` is the entire OrganelleVerse surface a
plugin author imports. It must stay tiny (no heavy third-party imports),
frozen, and free of any OrganelleVerse-internal objects so a plugin never
couples to registry or capability internals.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest


def test_plugin_context_and_result_are_tiny_frozen_models(tmp_path: Path) -> None:
    from organelleverse.plugin_protocol import PluginContext, PluginResult

    context = PluginContext(
        capability_id="em.segment",
        run_id="run-1",
        work_dir=tmp_path / "work",
        log_path=None,
    )
    result = PluginResult(
        summary={"masks": 3},
        outputs={"masks": str(tmp_path / "work" / "outputs" / "masks")},
    )

    assert context.capability_id == "em.segment"
    assert context.log_path is None
    assert result.score is None
    assert result.log_paths == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.capability_id = "other"


def test_plugin_result_rejects_non_json_summary() -> None:
    from organelleverse.plugin_protocol import PluginResult

    with pytest.raises(ValueError, match="JSON-safe"):
        PluginResult(summary={"handle": object()})


def test_plugin_protocol_module_has_no_heavy_imports() -> None:
    import os

    import organelleverse.plugin_protocol

    script = (
        "import sys\n"
        "import organelleverse.plugin_protocol\n"
        "heavy = [m for m in ('Bio', 'numpy', 'torch', 'tensorflow', 'matplotlib')\n"
        "         if m in sys.modules]\n"
        "print(heavy)\n"
        "sys.exit(1 if heavy else 0)\n"
    )
    # Pin the cold-import child to this checkout: the ambient interpreter may
    # resolve an editable install of a different checkout that lacks the
    # module (same failure class as detached worker spawning).
    env = os.environ.copy()
    import_root = str(Path(organelleverse.plugin_protocol.__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = import_root + (os.pathsep + existing if existing else "")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert completed.returncode == 0, (
        f"plugin_protocol imported heavy modules: {completed.stdout}\n{completed.stderr}"
    )


def test_optimization_score_metric_name_is_reserved_and_exported() -> None:
    from organelleverse.plugin_protocol import (
        PLUGIN_OPTIMIZATION_SCORE_METRIC,
        PluginResult,
    )

    assert PLUGIN_OPTIMIZATION_SCORE_METRIC == "plugin_optimization_score"
    # PluginResult.score stays optional scorer input and defaults to None;
    # it is never published on its own authority.
    assert PluginResult(summary={}).score is None
