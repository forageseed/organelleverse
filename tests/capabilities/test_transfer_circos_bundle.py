"""The transfer chord-diagram capability is admitted by a real immutable fixture."""

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


def test_transfer_circos_bundle(tmp_path, monkeypatch):
    from organelleverse.capabilities.adapters.visualization import (
        _CIRCOS_FRAGMENTS,
        _CIRCOS_MITO_GENBANK,
        _CIRCOS_PLASTID_GENBANK,
    )

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    capability = "visualization.plot_transfer_circos"
    index = discover_capabilities()
    verify_capability(
        capability,
        store=VerificationStore(home / "verifications"),
        environment=LocalVerificationEnvironment(index),
    )
    index = discover_capabilities()
    assert index.describe(capability).status is CapabilityStatus.ADMITTED

    mito, plastid = Path("mito.gb"), Path("plastid.gb")
    mito.write_text(_CIRCOS_MITO_GENBANK.content)
    plastid.write_text(_CIRCOS_PLASTID_GENBANK.content)
    result = (
        index.binding_source()
        .resolve(capability)
        .invoke(
            None,
            {
                "transfers": _CIRCOS_FRAGMENTS,
                "recipient": str(mito.resolve()),
                "donor": str(plastid.resolve()),
                "min_length": 150,
            },
        )
    )
    assert isinstance(result, OrganelleResult)
    assert result.metrics["transfer_kind"] == "MTPT"
    assert result.metrics["link_count"] == 2  # the 100 bp IR trnN copy (two HSPs) is filtered
    assert result.metrics["dropped_below_min_length"] == 2
    assert not result.artifacts

    # the canonical Result renders after its input files are gone
    mito.unlink()
    plastid.unlink()
    published = ov.write(result, tmp_path / "capability.svg")
    assert {a.kind for a in published.artifacts} == {"figure", "figure_data"}
