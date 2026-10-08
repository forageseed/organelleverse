"""Capability Plan 03, Task 3: the restored ``barcode`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated all four
bundles under ``src/organelleverse/capabilities/barcode-*`` from
``docs/operations/restored-capabilities.toml`` - see the generator's and
``organelleverse.capabilities.adapters.barcode``'s module docstrings for the
per-capability reasoning. Every ledger record in this domain resolves; none
is skipped.

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

_DESIGN_BARCODE_ID = "barcode.design_barcode"
_IDENTIFY_ID = "barcode.identify"
_COMPUTE_BARCODE_CANDIDATES_ID = "barcode.compute_barcode_candidates"
_COMPUTE_JACCARD_IDENTITY_ID = "barcode.compute_jaccard_identity"
_RESTORED_IDS = (
    _DESIGN_BARCODE_ID,
    _IDENTIFY_ID,
    _COMPUTE_BARCODE_CANDIDATES_ID,
    _COMPUTE_JACCARD_IDENTITY_ID,
)

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {
    "barcode-compute-barcode-candidates",
    "barcode-compute-jaccard-identity",
    "barcode-design-barcode",
    "barcode-identify",
}

_ALIGNMENT_FASTA = ">a\nACGTACGTACGTACGTACGTGGGGCCCC\n>b\nACGAACGTACGTACGAACGTGGGGCCCA\n"
# identify()/compute_jaccard_identity() default to k=31: the sequences need
# to be longer than that for any k-mer overlap to exist at all.
_QUERY_FASTA = ">query\nACGTACGTACGTACGTACGTGGGGCCCCAAAATTTTACGTACGTACGT\n"
_REFERENCE_FASTA = (
    ">species_a\nACGTACGTACGTACGTACGTGGGGCCCCAAAATTTTACGTACGTACGT\n"
    ">species_b\nTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n"
)


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


def test_the_four_bindable_capabilities_are_discovered_from_the_real_core_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_non_core_roots(tmp_path, monkeypatch)

    index = discover_capabilities()

    for capability_id in _RESTORED_IDS:
        entry = index.describe(capability_id)
        assert entry.origins[0].channel == "core"
        assert entry.execution_identity is None


def test_the_generator_produces_exactly_these_ids_with_no_skips(tmp_path: Path) -> None:
    generator = _load_generator()
    isolated_output_root = tmp_path / "capabilities"

    report = generator.build_domain(
        domain="barcode",
        ledger_path=_LEDGER_PATH,
        current_root=_CURRENT_ROOT,
        output_root=isolated_output_root,
    )

    generated_ids = {item.capability_id for item in report.generated}
    assert generated_ids == set(_RESTORED_IDS)
    assert report.skipped == ()

    generated_dirs = {item.path.parent.name for item in report.generated}
    assert generated_dirs == _EXPECTED_BUNDLE_DIRS

    assert_regenerated_domain_matches_committed(
        isolated_output_root, _BUNDLE_ROOT, _EXPECTED_BUNDLE_DIRS
    )

    # Capability Plan 04 Task 1: only the two json_metric capabilities carry
    # a real, generator-captured fixture - design_barcode/identify embed
    # real wall-clock timestamps in their provenance (see
    # adapters.barcode's module docstring) so no honest exact fixture is
    # possible for either.
    fixture_cases_by_id = {
        item.capability_id: {
            fixture.case for fixture in parse_capability_bundle(item.path).fixtures
        }
        for item in report.generated
    }
    assert fixture_cases_by_id == {
        _DESIGN_BARCODE_ID: set(),
        _IDENTIFY_ID: set(),
        _COMPUTE_BARCODE_CANDIDATES_ID: {"basic"},
        _COMPUTE_JACCARD_IDENTITY_ID: {"basic"},
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


def test_design_barcode_resolves_and_invokes_on_a_real_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_DESIGN_BARCODE_ID)
    assert binding is not None

    fasta_path = tmp_path / "aln.fasta"
    fasta_path.write_text(_ALIGNMENT_FASTA)

    result = binding.invoke(None, {"alignment_fasta": str(fasta_path)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _DESIGN_BARCODE_ID
    assert result.status == "ok"


def test_identify_resolves_and_invokes_on_real_query_and_reference_fastas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_IDENTIFY_ID)
    assert binding is not None

    query_path = tmp_path / "query.fasta"
    query_path.write_text(_QUERY_FASTA)
    reference_path = tmp_path / "reference.fasta"
    reference_path.write_text(_REFERENCE_FASTA)

    result = binding.invoke(
        None,
        {"query_fasta": str(query_path), "reference_fasta": str(reference_path)},
    )

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert result.metrics["best_match"] == "species_a"


def test_compute_barcode_candidates_resolves_and_invokes_on_a_real_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_COMPUTE_BARCODE_CANDIDATES_ID)
    assert binding is not None

    fasta_path = tmp_path / "aln.fasta"
    fasta_path.write_text(_ALIGNMENT_FASTA)

    result = binding.invoke(None, {"alignment_fasta": str(fasta_path)})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert result.metrics["barcode_candidates"]["n_sequences"] == 2


def test_compute_jaccard_identity_resolves_and_invokes_on_real_fastas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_COMPUTE_JACCARD_IDENTITY_ID)
    assert binding is not None

    query_path = tmp_path / "query.fasta"
    query_path.write_text(_QUERY_FASTA)
    reference_path = tmp_path / "reference.fasta"
    reference_path.write_text(_REFERENCE_FASTA)

    result = binding.invoke(
        None,
        {"query_fasta": str(query_path), "reference_fasta": str(reference_path)},
    )

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert result.metrics["jaccard_identity"]["best_match"] == "species_a"
