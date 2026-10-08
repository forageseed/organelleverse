"""New scientific capabilities bind and verify their committed portable fixtures."""

import pytest

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)


@pytest.mark.parametrize(
    "capability",
    [
        "comparative.detect_structural_variants",
        "phylogeny.reconstruct_ancestral_states",
        "phylogeny.gene_presence_traits",
    ],
)
def test_committed_science_fixture_verifies(tmp_path, monkeypatch, capability):
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    index = discover_capabilities()
    record = verify_capability(
        capability,
        store=VerificationStore(tmp_path / "verification"),
        environment=LocalVerificationEnvironment(index),
    )
    assert record.equivalence and all(e.verdict == "pass" for e in record.equivalence), record
