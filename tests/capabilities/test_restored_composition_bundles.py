"""Capability Plan 03, Task 3: the restored ``composition`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated the two bundles
under ``src/organelleverse/capabilities/composition-*`` from
``docs/operations/restored-capabilities.toml`` - see the generator's and
``organelleverse.capabilities.adapters.composition``'s module docstrings for
the per-capability reasoning, and why ``composition.write_content`` was
deliberately not generated (no codec can decode its
``OrganelleResult | Mapping`` parameter honestly).

Mirrors ``tests/capabilities/test_restored_format_conversion_bundles.py``.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.parser import parse_capability_bundle
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.result import OrganelleResult
from tests._paths import PROJECT_ROOT
from tests.capabilities._bundle_regeneration_check import (
    assert_regenerated_domain_matches_committed,
)

_GC_CONTENT_ID = "composition.gc_content"
_COMPUTE_GC_CONTENT_ID = "composition.compute_gc_content"
_RESTORED_IDS = (_GC_CONTENT_ID, _COMPUTE_GC_CONTENT_ID)
_EXCLUDED_ID = "composition.write_content"

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {
    "composition-compute-gc-content",
    "composition-gc-content",
}

_FASTA_TEXT = ">seq1\nGGGGCCCCAAAATTTT\n"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_restored_bundles", _GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _isolate_non_core_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    project = tmp_path / "project"
    (project / ".organelleverse" / "capabilities").mkdir(parents=True)
    home.mkdir()

    def no_entry_points(**kwargs: object) -> tuple[()]:
        return ()

    monkeypatch.setattr(importlib.metadata, "entry_points", no_entry_points)
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(project)
    return home


def test_the_two_bindable_capabilities_are_discovered_from_the_real_core_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_non_core_roots(tmp_path, monkeypatch)

    index = discover_capabilities()

    discovered_ids = {entry.capability_id for entry in index.entries}
    for capability_id in _RESTORED_IDS:
        entry = index.describe(capability_id)
        assert entry.origins[0].channel == "core"
        assert entry.execution_identity is None
    assert _EXCLUDED_ID not in discovered_ids


def test_the_generator_produces_exactly_these_ids_and_skips_write_content(
    tmp_path: Path,
) -> None:
    generator = _load_generator()
    isolated_output_root = tmp_path / "capabilities"

    report = generator.build_domain(
        domain="composition",
        ledger_path=_LEDGER_PATH,
        current_root=_CURRENT_ROOT,
        output_root=isolated_output_root,
    )

    generated_ids = {item.capability_id for item in report.generated}
    assert generated_ids == set(_RESTORED_IDS)

    skip_codes = {item.capability_id: item.code for item in report.skipped}
    assert skip_codes == {_EXCLUDED_ID: "capability.no_adapter_override"}

    generated_dirs = {item.path.parent.name for item in report.generated}
    assert generated_dirs == _EXPECTED_BUNDLE_DIRS

    # What is committed under src/organelleverse/capabilities/composition-*
    # is exactly what today's generator produces - proved by real,
    # byte-for-byte comparison against a fresh, isolated regeneration,
    # never by writing into the committed tree to find out.
    assert_regenerated_domain_matches_committed(
        isolated_output_root, _BUNDLE_ROOT, _EXPECTED_BUNDLE_DIRS
    )

    # Capability Plan 04 Task 1: both bindable capabilities in this domain
    # now carry exactly one real, generator-captured fixture each - see
    # ``adapters.composition.FIXTURES`` and
    # ``test_generated_fixtures_end_to_end.py`` for the full discover ->
    # verify -> admit -> invoke proof.
    fixture_cases_by_id = {
        item.capability_id: {
            fixture.case for fixture in parse_capability_bundle(item.path).fixtures
        }
        for item in report.generated
    }
    assert fixture_cases_by_id == {
        _GC_CONTENT_ID: {"basic"},
        _COMPUTE_GC_CONTENT_ID: {"basic"},
    }


def test_every_generated_bundle_reaches_admitted_through_the_real_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _isolate_non_core_roots(tmp_path, monkeypatch)

    discovered = discover_capabilities()
    for capability_id in _RESTORED_IDS:
        assert discovered.describe(capability_id).status is CapabilityStatus.REJECTED

    store = VerificationStore(home / "verifications")
    for capability_id in _RESTORED_IDS:
        record = verify_capability(
            capability_id,
            store=store,
            environment=LocalVerificationEnvironment(discovered),
        )
        assert record.execution_identity is None
        assert record.worker_parameters == ()

    admitted = discover_capabilities()
    for capability_id in _RESTORED_IDS:
        entry = admitted.describe(capability_id)
        assert entry.status is CapabilityStatus.ADMITTED, entry.diagnostic
    admitted_ids = {item.capability_id for item in admitted.list()}
    assert admitted_ids == set(_RESTORED_IDS)


def _admit_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = _isolate_non_core_roots(tmp_path, monkeypatch)
    discovered = discover_capabilities()
    store = VerificationStore(home / "verifications")
    for capability_id in _RESTORED_IDS:
        verify_capability(
            capability_id, store=store, environment=LocalVerificationEnvironment(discovered)
        )
    return discover_capabilities()


def test_gc_content_resolves_and_invokes_on_a_real_fasta_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_GC_CONTENT_ID)
    assert binding is not None

    fasta_path = tmp_path / "genome.fasta"
    fasta_path.write_text(_FASTA_TEXT)

    result = binding.invoke(None, {"genome_fasta": str(fasta_path)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _GC_CONTENT_ID
    assert result.status == "ok"
    assert result.metrics["gc_content"] == 0.5


def test_compute_gc_content_resolves_and_invokes_on_a_real_fasta_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_COMPUTE_GC_CONTENT_ID)
    assert binding is not None

    fasta_path = tmp_path / "genome.fasta"
    fasta_path.write_text(_FASTA_TEXT)

    result = binding.invoke(None, {"fasta_path": str(fasta_path)})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    metrics = result.metrics["gc_metrics"]
    assert metrics["total_length"] == 16


def test_write_content_the_generator_refused_to_produce_is_simply_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_non_core_roots(tmp_path, monkeypatch)
    index = discover_capabilities()

    with pytest.raises(Exception) as excinfo:
        index.describe(_EXCLUDED_ID)
    assert "unknown_capability" in str(excinfo.value) or "unknown capability" in str(excinfo.value)
