"""Capability Plan 03, Task 3: the restored ``hgt`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated 2 bundles under
``src/organelleverse/capabilities/hgt-*`` from
``docs/operations/restored-capabilities.toml`` - see
``organelleverse.capabilities.adapters.hgt``'s module docstring for the
full reasoning, including a newly discovered reason for exclusion:
``run_hgt_alignments``/``run_blastn_confirmation`` return plain
``@dataclass`` instances that no implemented ``ResultCodec`` can honestly
represent (not JSON-safe, not an ``OrganelleResult``, not an artifact).

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

_DETECT_HGT_ID = "hgt.detect_hgt"
_COMPUTE_HGT_ID = "hgt.compute_hgt"
_RESTORED_IDS = (_DETECT_HGT_ID, _COMPUTE_HGT_ID)
_EXCLUDED_IDS = ("hgt.run_blastn_confirmation", "hgt.run_hgt_alignments")

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {"hgt-detect-hgt", "hgt-compute-hgt"}


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


def _admit_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = _isolate_non_core_roots(tmp_path, monkeypatch)
    discovered = discover_capabilities()
    store = VerificationStore(home / "verifications")
    for capability_id in _RESTORED_IDS:
        verify_capability(
            capability_id, store=store, environment=LocalVerificationEnvironment(discovered)
        )
    return discover_capabilities()


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in records))


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
    for excluded_id in _EXCLUDED_IDS:
        assert excluded_id not in discovered_ids


def test_the_generator_produces_exactly_these_ids_and_skips_the_rest(tmp_path: Path) -> None:
    generator = _load_generator()
    isolated_output_root = tmp_path / "capabilities"

    report = generator.build_domain(
        domain="hgt",
        ledger_path=_LEDGER_PATH,
        current_root=_CURRENT_ROOT,
        output_root=isolated_output_root,
    )

    generated_ids = {item.capability_id for item in report.generated}
    assert generated_ids == set(_RESTORED_IDS)

    skip_codes = {item.capability_id: item.code for item in report.skipped}
    assert skip_codes == {
        excluded_id: "capability.no_adapter_override" for excluded_id in _EXCLUDED_IDS
    }

    generated_dirs = {item.path.parent.name for item in report.generated}
    assert generated_dirs == _EXPECTED_BUNDLE_DIRS

    assert_regenerated_domain_matches_committed(
        isolated_output_root, _BUNDLE_ROOT, _EXPECTED_BUNDLE_DIRS
    )


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
            capability_id, store=store, environment=LocalVerificationEnvironment(discovered)
        )
        assert record.execution_identity is None
        assert record.worker_parameters == ()

    admitted = discover_capabilities()
    for capability_id in _RESTORED_IDS:
        entry = admitted.describe(capability_id)
        assert entry.status is CapabilityStatus.ADMITTED, entry.diagnostic
    admitted_ids = {item.capability_id for item in admitted.list()}
    assert admitted_ids == set(_RESTORED_IDS)


def test_a_capability_the_generator_refused_to_produce_is_simply_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_non_core_roots(tmp_path, monkeypatch)
    index = discover_capabilities()

    for excluded_id in _EXCLUDED_IDS:
        with pytest.raises(Exception) as excinfo:
            index.describe(excluded_id)
        assert "unknown_capability" in str(excinfo.value) or "unknown capability" in str(
            excinfo.value
        )


def test_detect_hgt_and_compute_hgt_resolve_and_invoke_on_real_no_hit_fastas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real donor/recipient FASTAs with no shared sequence - an honest negative."""
    admitted = _admit_all(tmp_path, monkeypatch)

    donor_path = tmp_path / "donor.fasta"
    recipient_path = tmp_path / "recipient.fasta"
    _write_fasta(donor_path, [("donor1", "ACGTACGTACGTACGTACGT" * 5)])
    _write_fasta(recipient_path, [("recipient1", "TTTTGGGGCCCCAAAATTTT" * 5)])

    detect_binding = admitted.binding_source().resolve(_DETECT_HGT_ID)
    assert detect_binding is not None
    detect_result = detect_binding.invoke(
        None, {"donor_fasta": str(donor_path), "recipient_fasta": str(recipient_path)}
    )
    assert isinstance(detect_result, OrganelleResult)
    assert detect_result.operation_id == _DETECT_HGT_ID
    assert detect_result.status in {"ok", "failed"}

    compute_binding = admitted.binding_source().resolve(_COMPUTE_HGT_ID)
    assert compute_binding is not None
    compute_result = compute_binding.invoke(
        None, {"donor_fasta": str(donor_path), "recipient_fasta": str(recipient_path)}
    )
    assert isinstance(compute_result, OrganelleResult)
    assert compute_result.status == "ok"
    assert "hgt_candidates" in compute_result.metrics
