"""Hermetic fixtures: an admitted + trusted scored v2 demo plugin.

The demo plugin declares one directory input, one managed directory output,
and three JSON value parameters:

- ``threshold`` — the declared optimization parameter (number, 0..1);
- ``probe_log`` — optional path the plugin appends start/end interval lines
  to, letting tests measure real cross-process invocation concurrency;
- ``fail_at`` — when equal to ``threshold``, the run callable raises.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from organelleverse.plugin_experiments import ExperimentRecord, ExperimentService

from organelleverse.capabilities.code_identity import inspect_bundle_code
from organelleverse.capabilities.hashing import hash_bundle
from organelleverse.capabilities.index import (
    CapabilityEntry,
    CapabilityIndex,
    CapabilityOrigin,
    CapabilityStatus,
)
from organelleverse.capabilities.models import PluginCapabilityBundle
from organelleverse.capabilities.trust import TrustStore, trust
from organelleverse.capabilities.worker import OneShotBundleWorkerExecutor
from organelleverse.operations.python_binding import worker_parameter_schema
from tests.capabilities.test_plugin_v2_models import v2_payload

TRIAL_PLUGIN = """\
import os
import time
from pathlib import Path

from organelleverse.plugin_protocol import PluginContext, PluginResult


def run(inputs: dict, outputs: dict, parameters: dict, context: PluginContext) -> PluginResult:
    threshold = parameters["threshold"]
    probe = parameters.get("probe_log") or ""
    if probe:
        with open(probe, "a", encoding="utf-8") as handle:
            handle.write(f"start {time.monotonic()}\\n")
    if threshold == parameters.get("fail_at", -1.0):
        raise RuntimeError("synthetic trial failure")
    time.sleep(0.4 if probe else 0.0)
    if probe:
        with open(probe, "a", encoding="utf-8") as handle:
            handle.write(f"end {time.monotonic()}\\n")
    masks = Path(outputs["masks"])
    (masks / "mask.txt").write_text(str(threshold), encoding="utf-8")
    return PluginResult(
        summary={"threshold": threshold},
        outputs={"masks": str(masks)},
    )


def score(result: PluginResult) -> float:
    return float(result.summary["threshold"])
"""


def build_plugin_entry(
    root: Path,
    source: str = TRIAL_PLUGIN,
    *,
    capability_id: str = "demo.experiment",
    optimization: bool = True,
    max_trials: int = 3,
    parallelism: int = 2,
) -> CapabilityEntry:
    """Build one verified-signature bundle entry for the scored demo plugin."""

    payload = v2_payload()
    capability = cast(dict[str, object], payload["capability"])
    capability["id"] = capability_id
    contract = cast(dict[str, object], payload["contract"])
    contract["operation_id"] = capability_id
    contract["callable_locator"] = "demo_experiment.plugin:run"
    contract.pop("optimization")
    if optimization:
        contract["optimization"] = {
            "score_locator": "demo_experiment.plugin:score",
            "parameters": ["threshold"],
            "max_trials": max_trials,
            "parallelism": parallelism,
        }
    binding = cast(dict[str, object], contract["binding"])
    binding["parameters"] = [
        {
            "name": "images",
            "codec": "directory",
            "path_role": "input",
            "json_schema": {"type": "string"},
        },
        {
            "name": "masks_dir",
            "codec": "directory",
            "path_role": "output",
            "json_schema": {"type": "string"},
        },
        {
            "name": "threshold",
            "codec": "json",
            "json_schema": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5},
        },
        {"name": "probe_log", "codec": "json", "json_schema": {"type": "string", "default": ""}},
        {"name": "fail_at", "codec": "json", "json_schema": {"type": "number", "default": -1.0}},
    ]
    contract["outputs"] = [
        {"name": "masks", "kind": "directory", "parameter": "masks_dir", "description": "Masks."},
    ]
    bundle = PluginCapabilityBundle.model_validate(payload)
    package = root / "code" / "demo_experiment"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "plugin.py").write_text(source, encoding="utf-8")
    (root / "capability.toml").write_text("schema = 'fixture'\n", encoding="utf-8")
    content_hash = hash_bundle(root)
    identity = inspect_bundle_code(
        root,
        capability_id=bundle.capability.id,
        bundle_content_hash=content_hash,
        callable_locator=cast(str, bundle.contract.callable_locator),
    )
    entry = CapabilityEntry(
        capability_id=bundle.capability.id,
        content_hash=content_hash,
        bundle_root=root,
        bundle=bundle,
        origins=(
            CapabilityOrigin(channel="local", source_path=str(root / "capability.toml"), search_root=str(root)),
        ),
        execution_identity=identity,
    )
    inspection = OneShotBundleWorkerExecutor().inspect(entry)
    return entry.model_copy(update={"worker_parameters": inspection.parameters})


def admit(entry: CapabilityEntry) -> CapabilityEntry:
    """Return the admitted snapshot with its frozen parameter schema."""

    assert entry.worker_parameters is not None
    return entry.model_copy(
        update={
            "status": CapabilityStatus.ADMITTED,
            "parameter_schema": worker_parameter_schema(entry.bundle, entry.worker_parameters),
        }
    )


class PluginEnvironment:
    """One admitted demo plugin, its trust store, and a images input dir."""

    def __init__(self, tmp_path: Path, **entry_kwargs: object) -> None:
        self.entry = admit(build_plugin_entry(tmp_path / "bundle", **entry_kwargs))  # type: ignore[arg-type]
        self.trust_store = TrustStore(tmp_path / "trust.json")
        assert self.entry.execution_identity is not None
        trust(self.entry.capability_id, self.entry.execution_identity, store=self.trust_store)
        self.index = CapabilityIndex(entries=(self.entry,))
        self.images = tmp_path / "images"
        self.images.mkdir()
        (self.images / "one.tif").write_bytes(b"image")


@pytest.fixture
def plugin_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PluginEnvironment:
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    return PluginEnvironment(tmp_path)


def wait_for_experiment(
    service: ExperimentService, experiment_id: str, timeout: float = 120.0
) -> ExperimentRecord:
    """Poll until the experiment reaches a terminal status."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = service.get(experiment_id)
        if record.status in {"succeeded", "failed"}:
            return record
        time.sleep(0.05)
    raise AssertionError(f"experiment {experiment_id} did not become terminal")
