"""The plugin protocol: the entire OrganelleVerse surface a plugin imports.

A plugin author writes exactly one callable::

    from organelleverse.plugin_protocol import PluginContext, PluginResult

    def run(
        inputs: dict[str, object],
        outputs: dict[str, object],
        parameters: dict[str, object],
        context: PluginContext,
    ) -> PluginResult: ...

This module deliberately uses only the standard library: the Worker executes
third-party plugin code under a closed import policy (private sources plus
stdlib), so the protocol types must be importable there - and cheap to
import anywhere else.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, cast


def _require_json_safe(value: object, *, path: str) -> None:
    """Reject anything outside the closed JSON value set, recursively."""

    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be a finite number")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(cast("list[object]", value)):
            _require_json_safe(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in cast("dict[object, object]", value).items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            _require_json_safe(item, path=f"{path}.{key}")
        return
    raise ValueError(f"{path} must be JSON-safe, got {type(value).__name__}")


@dataclasses.dataclass(frozen=True)
class PluginContext:
    """Run-scoped facts the runtime hands to the plugin callable."""

    capability_id: str
    run_id: str
    work_dir: Path
    log_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.capability_id.strip():
            raise ValueError("capability_id must not be blank")
        if not self.run_id.strip():
            raise ValueError("run_id must not be blank")


@dataclasses.dataclass(frozen=True)
class PluginResult:
    """The structured outcome a plugin returns from one run.

    ``outputs`` maps each declared output name to the path the plugin
    actually produced inside its allocated destination. ``score`` is
    optional data the plugin may expose to its own declared score callable
    (``contract.optimization.score_locator``); it is read there but never
    independently trusted or published. Only the finite float returned by
    the declared score callable becomes the run's score, published under
    the reserved ``PLUGIN_OPTIMIZATION_SCORE_METRIC`` L6 metric key.
    Plugins without an optimization declaration leave it ``None``.
    """

    summary: dict[str, Any] = dataclasses.field(default_factory=lambda: {})
    outputs: dict[str, str] = dataclasses.field(default_factory=lambda: {})
    log_paths: tuple[str, ...] = ()
    score: float | None = None

    def __post_init__(self) -> None:
        _require_json_safe(self.summary, path="summary")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("score must be a finite number")


#: The reserved L6 metric key carrying the declared optimization score.
#: A plugin summary must never emit this key itself; only the finite float
#: returned by the declared ``score_locator`` callable is published here.
PLUGIN_OPTIMIZATION_SCORE_METRIC = "plugin_optimization_score"

__all__ = ["PLUGIN_OPTIMIZATION_SCORE_METRIC", "PluginContext", "PluginResult"]
