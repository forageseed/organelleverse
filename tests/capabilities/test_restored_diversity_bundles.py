"""Capability Plan 03, Task 3: the restored ``diversity`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated the four
bundles under ``src/organelleverse/capabilities/diversity-*`` from
``docs/operations/restored-capabilities.toml`` - see the generator's and
``organelleverse.capabilities.adapters.diversity``'s module docstrings for
the per-capability reasoning, and why ``diversity.write_nucleotide`` /
``.write_neutral_tests`` were deliberately not generated (no codec can
decode their ``OrganelleResult | Mapping`` parameter honestly).

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

_NUCLEOTIDE_DIVERSITY_ID = "diversity.nucleotide_diversity"
_NEUTRAL_TESTS_ID = "diversity.neutral_tests"
_COMPUTE_NUCLEOTIDE_DIVERSITY_ID = "diversity.compute_nucleotide_diversity"
_COMPUTE_NEUTRAL_TESTS_ID = "diversity.compute_neutral_tests"
_RESTORED_IDS = (
    _NUCLEOTIDE_DIVERSITY_ID,
    _NEUTRAL_TESTS_ID,
    _COMPUTE_NUCLEOTIDE_DIVERSITY_ID,
    _COMPUTE_NEUTRAL_TESTS_ID,
)
_EXCLUDED_IDS = ("diversity.write_nucleotide", "diversity.write_neutral_tests")

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {
    "diversity-compute-neutral-tests",
    "diversity-compute-nucleotide-diversity",
    "diversity-neutral-tests",
    "diversity-nucleotide-diversity",
}

_ALIGNMENT_FASTA = ">a\nACGTACGTACGTACGTACGT\n>b\nACGTACGTACGTACGTACGA\n"
# compute_neutral_tests's raw dict does not sanitize NaN the way its
# OrganelleResult-returning sibling neutral_tests() does (diversity.py:278-279
# converts NaN to None; diversity_core.py's json twin does not) - Tajima's D
# is mathematically undefined below 3 segregating sites, so this fixture
# needs enough real polymorphism to keep it finite for the json_metric codec.
_NEUTRAL_TESTS_FASTA = (
    ">a\nACGTACGTACGTACGTACGTACGT\n"
    ">b\nACGAACGAACGAACGTACGAACGT\n"
    ">c\nACGCACGCACGCACGTACGCACGA\n"
    ">d\nACGTACGTACGTACGTACGTACGC\n"
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

    discovered_ids = {entry.capability_id for entry in index.entries}
    for capability_id in _RESTORED_IDS:
        entry = index.describe(capability_id)
        assert entry.origins[0].channel == "core"
        assert entry.execution_identity is None
    for excluded_id in _EXCLUDED_IDS:
        assert excluded_id not in discovered_ids


def test_the_generator_produces_exactly_these_ids_and_skips_the_writers(tmp_path: Path) -> None:
    generator = _load_generator()
    isolated_output_root = tmp_path / "capabilities"

    report = generator.build_domain(
        domain="diversity",
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

    # Capability Plan 04 Task 1: all four bindable capabilities carry a
    # real, generator-captured fixture over the same 6-sequence alignment
    # (see adapters.diversity's module docstring for why it has 4
    # segregating sites, not fewer).
    fixture_cases_by_id = {
        item.capability_id: {
            fixture.case for fixture in parse_capability_bundle(item.path).fixtures
        }
        for item in report.generated
    }
    assert fixture_cases_by_id == {capability_id: {"basic"} for capability_id in _RESTORED_IDS}


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


def test_nucleotide_diversity_resolves_and_invokes_on_a_real_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_NUCLEOTIDE_DIVERSITY_ID)
    assert binding is not None

    fasta_path = tmp_path / "aln.fasta"
    fasta_path.write_text(_ALIGNMENT_FASTA)

    result = binding.invoke(None, {"alignment_fasta": str(fasta_path)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _NUCLEOTIDE_DIVERSITY_ID
    assert result.status == "ok"
    assert result.metrics["n_sequences"] == 2


def test_neutral_tests_resolves_and_invokes_on_a_real_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_NEUTRAL_TESTS_ID)
    assert binding is not None

    fasta_path = tmp_path / "aln.fasta"
    fasta_path.write_text(_ALIGNMENT_FASTA)

    result = binding.invoke(None, {"alignment_fasta": str(fasta_path)})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"


def test_compute_nucleotide_diversity_resolves_and_invokes_on_a_real_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_COMPUTE_NUCLEOTIDE_DIVERSITY_ID)
    assert binding is not None

    fasta_path = tmp_path / "aln.fasta"
    fasta_path.write_text(_ALIGNMENT_FASTA)

    result = binding.invoke(None, {"alignment_fasta": str(fasta_path)})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    metrics = result.metrics["diversity_metrics"]
    assert metrics["n_sequences"] == 2


def test_compute_neutral_tests_resolves_and_invokes_on_a_real_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_COMPUTE_NEUTRAL_TESTS_ID)
    assert binding is not None

    fasta_path = tmp_path / "aln.fasta"
    fasta_path.write_text(_NEUTRAL_TESTS_FASTA)

    result = binding.invoke(None, {"alignment_fasta": str(fasta_path)})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert "neutral_tests" in result.metrics


def test_the_writers_the_generator_refused_to_produce_are_simply_absent(
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
