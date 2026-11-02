"""The assembly-graph capability is admitted by a real immutable fixture."""

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


def test_assembly_graph_bundle(tmp_path, monkeypatch):
    from organelleverse.capabilities.adapters.visualization import _ASSEMBLY_GFA

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    capability = "visualization.plot_assembly_graph"
    index = discover_capabilities()
    verify_capability(
        capability,
        store=VerificationStore(home / "verifications"),
        environment=LocalVerificationEnvironment(index),
    )
    index = discover_capabilities()
    assert index.describe(capability).status is CapabilityStatus.ADMITTED
    source = Path("graph.gfa")
    source.write_text(_ASSEMBLY_GFA.content)
    result = index.binding_source().resolve(capability).invoke(None, {"gfa": str(source.resolve())})
    assert isinstance(result, OrganelleResult)
    assert result.metrics["nodes"] == 3
    assert result.metrics["repeats"][0]["status"] == "candidate_only"
    assert not result.artifacts

    source.unlink()
    published = ov.write(result, tmp_path / "capability.svg")
    assert {a.kind for a in published.artifacts} == {"figure", "figure_data"}
