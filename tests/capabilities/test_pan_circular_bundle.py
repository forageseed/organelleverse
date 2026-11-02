"""The pan-circle capability is admitted by its real Citrus fixture and redraws from metrics."""

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


def test_pan_circular_bundle(tmp_path, monkeypatch):
    from organelleverse.capabilities.adapters.visualization import (
        _PAN_CIRCULAR_GENBANK,
        _PAN_CIRCULAR_GFA,
        _PAN_CIRCULAR_VCF,
    )

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    capability = "visualization.plot_pan_circular"
    index = discover_capabilities()
    verify_capability(
        capability,
        store=VerificationStore(home / "verifications"),
        environment=LocalVerificationEnvironment(index),
    )
    index = discover_capabilities()
    assert index.describe(capability).status is CapabilityStatus.ADMITTED

    for item in (_PAN_CIRCULAR_GENBANK, _PAN_CIRCULAR_VCF, _PAN_CIRCULAR_GFA):
        Path(item.relative_path).write_text(item.content)
    result = (
        index.binding_source()
        .resolve(capability)
        .invoke(
            None,
            {
                "reference_gbk": str(Path("reference.gb").resolve()),
                "variants": str(Path("variants.vcf").resolve()),
                "pangenome_gfa": str(Path("graph.gfa").resolve()),
                "bins": 30,
                "graph_max_blocks": 12,
            },
        )
    )
    assert isinstance(result, OrganelleResult)
    assert result.metrics["center_image"] == "ov_graph"
    assert len(result.metrics["heatmap"]["samples"]) == 4
    assert not result.artifacts

    for name in ("reference.gb", "variants.vcf", "graph.gfa"):
        Path(name).unlink()
    published = ov.write(result, tmp_path / "capability.svg")
    assert "figure" in {a.kind for a in published.artifacts}
