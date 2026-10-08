"""The new canonical capability verifies its committed fixture and binds inputs."""

from pathlib import Path

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)


def test_orientation_bundle_verifies_and_invokes(tmp_path, monkeypatch):
    from organelleverse.capabilities.adapters.comparative import _ORIENTATION_FASTA

    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(project)
    capability_id = "comparative.normalize_plastome_orientation"
    index = discover_capabilities()
    store = VerificationStore(home / "verifications")
    verify_capability(capability_id, store=store, environment=LocalVerificationEnvironment(index))
    index = discover_capabilities()
    assert index.describe(capability_id).status is CapabilityStatus.ADMITTED
    binding = index.binding_source().resolve(capability_id)
    path = Path("plastomes.fa")
    path.write_text(_ORIENTATION_FASTA.content)
    result = binding.invoke(None, {"input_fasta": str(path.resolve())})
    assert result.status == "ok"
    assert result.metrics["samples"][2]["ssc_reverse_complemented"]
    assert len({seq for _, seq in result.metrics["sequences"]}) == 1
