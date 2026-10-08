"""The structure-map capability is admitted by a real immutable fixture."""

from pathlib import Path

import organelleverse as ov
from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.result import OrganelleResult


def test_structure_map_bundle(tmp_path, monkeypatch):
    from organelleverse.capabilities.adapters.visualization import _STRUCTURE_GENBANK

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    capability = "visualization.plot_structure_map"
    index = discover_capabilities()
    verify_capability(
        capability,
        store=VerificationStore(home / "verifications"),
        environment=LocalVerificationEnvironment(index),
    )
    index = discover_capabilities()
    assert index.describe(capability).status is CapabilityStatus.ADMITTED
    source = Path("sample.gb")
    source.write_text(_STRUCTURE_GENBANK.content)
    result = (
        index.binding_source()
        .resolve(capability)
        .invoke(None, {"genbank_path": str(source.resolve())})
    )
    assert isinstance(result, OrganelleResult)
    assert result.metrics["cis_transcripts"] == 2
    assert result.metrics["trans_transcripts"] == 1
    assert not result.artifacts

    source.unlink()
    published = ov.write(result, tmp_path / "capability.svg")
    assert {a.kind for a in published.artifacts} == {"figure", "figure_data"}


def test_structure_map_bundle_takes_several_files_on_one_figure(tmp_path, monkeypatch):
    from organelleverse.capabilities.adapters.visualization import _STRUCTURE_GENBANK

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    capability = "visualization.plot_structure_map"
    index = discover_capabilities()
    verify_capability(
        capability,
        store=VerificationStore(home / "verifications"),
        environment=LocalVerificationEnvironment(index),
    )
    index = discover_capabilities()
    first, second = Path("a.gb"), Path("b.gb")
    first.write_text(_STRUCTURE_GENBANK.content)
    second.write_text(_STRUCTURE_GENBANK.content.replace("fixture", "other  ", 1))
    result = (
        index.binding_source()
        .resolve(capability)
        .invoke(None, {"genbank_path": [str(first.resolve()), str(second.resolve())]})
    )
    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
