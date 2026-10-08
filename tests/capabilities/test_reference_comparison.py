"""New QC entry point is discoverable, bound, and verifies its real fixture."""

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.operations.python_binding import bind_python_capability
from organelleverse.qc import compare_assembly_to_reference


def test_reference_comparison_binding_and_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    index = discover_capabilities()
    entry = index.describe("qc.compare_assembly_to_reference")
    assert bind_python_capability(entry.bundle, compare_assembly_to_reference, None) is not None
    assert (
        verify_capability(
            "qc.compare_assembly_to_reference",
            store=VerificationStore(tmp_path / "verify"),
            environment=LocalVerificationEnvironment(index),
        )
        is not None
    )
