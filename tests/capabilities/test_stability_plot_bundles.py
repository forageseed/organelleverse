"""The organelle stability figure capabilities are admitted by their immutable fixtures."""

import tomllib
from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.result import OrganelleResult
from tests._paths import PROJECT_ROOT

_IDS = (
    "visualization.plot_depth_profile",
    "visualization.plot_variant_frequencies",
    "visualization.plot_variant_site_heatmap",
)


@pytest.mark.parametrize("capability", _IDS)
def test_stability_bundle_is_admitted_and_renders(capability, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    index = discover_capabilities()
    verify_capability(
        capability,
        store=VerificationStore(home / "verifications"),
        environment=LocalVerificationEnvironment(index),
    )
    index = discover_capabilities()
    assert index.describe(capability).status is CapabilityStatus.ADMITTED
    manifest = PROJECT_ROOT / "src/capabilities" / capability.replace(".", "-").replace(
        "_", "-"
    ) / "capability.toml"
    parameters = tomllib.loads(manifest.read_text(encoding="utf-8"))["fixture"][0]["parameters"]
    result = index.binding_source().resolve(capability).invoke(None, parameters)
    assert isinstance(result, OrganelleResult) and result.status == "ok"
    assert not result.artifacts
    published = ov.write(result, Path(tmp_path) / "capability.svg")
    assert {a.kind for a in published.artifacts} == {"figure", "figure_data"}
