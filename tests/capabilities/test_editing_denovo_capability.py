"""Public discovery, binding and real BAM fixture for de novo editing."""

from pathlib import Path

import pytest

from organelleverse.capabilities.adapters.rna_editing import OVERRIDES
from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.rna_editing import detect_editing_sites

pysam = pytest.importorskip("pysam")
ID = "rna_editing.detect_editing_sites"
BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "src/organelleverse/capabilities/rna-editing-detect-editing-sites"
)


def test_discovery_verification_binding_and_all_parameter_codecs(tmp_path, monkeypatch):
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    index = discover_capabilities()
    entry = index.describe(ID)
    assert entry.origins[0].channel == "core"
    overrides = {p.name: p.codec for p in OVERRIDES[ID].parameters}
    assert overrides == {p.name: p.codec for p in entry.bundle.contract.binding.parameters}
    verify_capability(
        ID,
        store=VerificationStore(tmp_path / "home/verifications"),
        environment=LocalVerificationEnvironment(index),
    )
    admitted = discover_capabilities()
    assert admitted.describe(ID).status is CapabilityStatus.ADMITTED
    binding = admitted.binding_source().resolve(ID)
    inputs = BUNDLE / "fixtures/basic/input"
    parameters = {
        "bam_path": str(inputs / "reads.bam"),
        "reference_fasta": str(inputs / "reference.fa"),
        "scope": "plastid",
    }
    actual = binding.invoke(None, parameters)
    expected = detect_editing_sites(**parameters)
    assert actual.metrics == expected.metrics
    assert actual.metrics["site_count"] == 2
