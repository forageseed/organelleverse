"""The v2 scaffold must produce an immediately usable scientific plugin."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from organelleverse.capabilities import scaffold_plugin
from organelleverse.capabilities.admission import admit_capabilities
from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.models import PluginCapabilityBundle
from organelleverse.capabilities.parser import parse_capability_bundle
from organelleverse.capabilities.plugin_descriptor import describe_plugin
from organelleverse.capabilities.trust import TrustStore, trust
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.capabilities.worker import OneShotBundleWorkerExecutor
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.registry import OperationRegistry
from organelleverse.operations.spec import PluginOperationSpec
from tests._paths import PROJECT_ROOT


def test_scaffold_plugin_generates_a_valid_v2_bundle_and_its_form(tmp_path: Path) -> None:
    root = scaffold_plugin(
        tmp_path / "plugin",
        capability_id="demo.segment",
        plugin_name="Demo segmentation",
        author="Example Laboratory",
        summary="Counts images and writes a result directory.",
    )

    bundle = parse_capability_bundle(root / "capability.toml")

    assert isinstance(bundle, PluginCapabilityBundle)
    descriptor = describe_plugin(bundle)
    assert [field.name for field in descriptor.inputs] == ["input_dir"]
    assert [field.name for field in descriptor.outputs] == ["results"]
    assert [field.name for field in descriptor.parameters] == ["threshold"]
    assert descriptor.optimization is not None
    assert (root / "code" / "demo_segment_plugin" / "plugin.py").is_file()
    assert (root / "tests" / "test_plugin.py").is_file()


def test_scaffolded_plugin_runs_the_complete_authoring_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    search_root = tmp_path / "plugins"
    root = scaffold_plugin(
        search_root / "demo-segment",
        capability_id="demo.segment",
        plugin_name="Demo segmentation",
        author="Example Laboratory",
        summary="Counts images and writes a result directory.",
    )
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    (input_dir / "one.tif").write_bytes(b"image")

    discovered = discover_capabilities(paths=[search_root])
    entry = discovered.describe("demo.segment")
    assert entry.execution_identity is not None
    verification_store = VerificationStore(tmp_path / "verifications")
    verify_capability(
        "demo.segment",
        store=verification_store,
        environment=LocalVerificationEnvironment(discovered),
    )
    trust_store = TrustStore(tmp_path / "trust.json")
    trust("demo.segment", entry.execution_identity, store=trust_store)
    admitted = admit_capabilities(discovered, store=verification_store)
    registry = OperationRegistry(
        capability_source=admitted.binding_source(
            trust_store=trust_store,
            executor=OneShotBundleWorkerExecutor(),
        )
    )
    bound = registry.require("demo.segment")
    assert isinstance(bound.spec, PluginOperationSpec)
    assert [output.name for output in bound.spec.outputs] == ["results"]
    assert bound.spec.optimization is not None

    result = bound.invoke(None, {"input_dir": str(input_dir), "threshold": 0.75})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert result.metrics["input_file_count"] == 1
    assert result.metrics["threshold"] == 0.75
    # The scaffold declares a threshold experiment whose score hook scores the
    # count summary; the declared score must flow into the reserved L6 metric.
    assert result.metrics["plugin_optimization_score"] == 1.0
    assert {artifact.kind for artifact in result.artifacts} == {"results"}
    assert root.is_dir()


@pytest.mark.slow
def test_scaffolded_plugin_runs_from_an_installed_wheel_in_a_clean_venv(tmp_path: Path) -> None:
    distributions = tmp_path / "distributions"
    distributions.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(distributions),
            str(PROJECT_ROOT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(distributions.glob("organelleverse-*.whl"))
    environment = tmp_path / "clean-environment"
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--force-reinstall", str(wheel)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    driver = '''\
import json
import os
import sys
from pathlib import Path

from organelleverse.capabilities.admission import admit_capabilities
from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.scaffold import scaffold_plugin
from organelleverse.capabilities.trust import TrustStore, trust
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.capabilities.worker import OneShotBundleWorkerExecutor
from organelleverse.operations.registry import OperationRegistry

root = Path(sys.argv[1])
os.environ["ORGANELLEVERSE_HOME"] = str(root / "home")
os.environ["ORGANELLEVERSE_CACHE_ROOT"] = str(root / "cache")
plugin_root = scaffold_plugin(
    root / "plugins" / "demo-segment",
    capability_id="demo.segment",
    plugin_name="Demo segmentation",
    author="Example Laboratory",
    summary="Counts images and writes a result directory.",
)
inputs = root / "images"
inputs.mkdir()
(inputs / "one.tif").write_bytes(b"image")
index = discover_capabilities(paths=[plugin_root.parent])
entry = index.describe("demo.segment")
store = VerificationStore(root / "verifications")
verify_capability("demo.segment", store=store, environment=LocalVerificationEnvironment(index))
trust_store = TrustStore(root / "trust.json")
trust("demo.segment", entry.execution_identity, store=trust_store)
admitted = admit_capabilities(index, store=store)
registry = OperationRegistry(capability_source=admitted.binding_source(
    trust_store=trust_store, executor=OneShotBundleWorkerExecutor()
))
result = registry.invoke(
    "demo.segment", input=None,
    parameters={"input_dir": str(inputs), "threshold": 0.75},
)
print(json.dumps({"status": result.status, "count": result.metrics["input_file_count"]}))
'''
    child_environment = os.environ.copy()
    child_environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [str(python), "-I", "-c", driver, str(tmp_path / "run")],
        check=True,
        capture_output=True,
        text=True,
        env=child_environment,
        timeout=180,
    )

    assert json.loads(completed.stdout) == {"status": "ok", "count": 1}
