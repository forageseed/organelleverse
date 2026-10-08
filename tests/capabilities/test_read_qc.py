"""Read-QC discovery, parameter bindings and the frozen statistics fixture."""

import importlib

import pytest

from organelleverse.capabilities.adapters.qc import FIXTURES
from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.operations.python_binding import bind_python_capability


@pytest.mark.parametrize("name", ["read_statistics", "filter_short_reads", "filter_long_reads"])
def test_bindings_match_public_api(name):
    entry = discover_capabilities().describe("qc." + name)
    module = importlib.import_module("organelleverse.quality_control.api")
    bound = bind_python_capability(entry.bundle, getattr(module, name), None)
    assert bound is not None


def test_statistics_fixture_verifies(tmp_path, monkeypatch):
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    record = verify_capability(
        "qc.read_statistics",
        store=VerificationStore(tmp_path / "verification"),
        environment=LocalVerificationEnvironment(discover_capabilities()),
    )
    assert record is not None


def test_long_read_fixture_through_binding(tmp_path, monkeypatch):
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    case = FIXTURES["qc.filter_long_reads"][0]
    path = tmp_path / "reads.fastq"
    path.write_text(case.files[0].content)
    entry = discover_capabilities().describe("qc.filter_long_reads")
    api = importlib.import_module("organelleverse.quality_control.api")
    bound = bind_python_capability(entry.bundle, api.filter_long_reads, None)
    result = bound.invoke(None, {**case.parameters, "reads": str(path)})
    assert result.metrics["after"]["reads"] == 2
    assert result.metrics["after"]["bases"] == 10
