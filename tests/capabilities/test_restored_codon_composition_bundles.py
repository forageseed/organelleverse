"""Capability Plan 03, Task 3: the restored ``codon_composition`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated 6 bundles under
``src/organelleverse/capabilities/codon-composition-*`` from
``docs/operations/restored-capabilities.toml`` - see
``organelleverse.capabilities.adapters.codon_composition``'s module
docstring for the full per-capability reasoning.

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

_AMINO_ACID_ID = "codon_composition.amino_acid"
_CODON_USAGE_ID = "codon_composition.codon_usage"
_COMPUTE_AA_ID = "codon_composition.compute_amino_acid_composition"
_COMPUTE_CODON_USAGE_ID = "codon_composition.compute_codon_usage"
_COMPUTE_ENC_ID = "codon_composition.compute_enc"
_RESOLVE_GENETIC_CODE_ID = "codon_composition.resolve_genetic_code"
_RESTORED_IDS = (
    _AMINO_ACID_ID,
    _CODON_USAGE_ID,
    _COMPUTE_AA_ID,
    _COMPUTE_CODON_USAGE_ID,
    _COMPUTE_ENC_ID,
    _RESOLVE_GENETIC_CODE_ID,
)
_EXCLUDED_IDS = (
    "codon_composition.write_amino_acid",
    "codon_composition.write_usage",
)

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {
    "codon-composition-amino-acid",
    "codon-composition-codon-usage",
    "codon-composition-compute-amino-acid-composition",
    "codon-composition-compute-codon-usage",
    "codon-composition-compute-enc",
    "codon-composition-resolve-genetic-code",
}


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


def _write_cds_fasta(path: Path) -> None:
    # ATG start, three sense codons, TAA stop - a minimal but real CDS.
    path.write_text(">gene1\nATGGCTGCTGCTTAA\n>gene2\nATGGCTGCTGCTTAA\n")


def test_the_six_bindable_capabilities_are_discovered_from_the_real_core_channel(
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
        domain="codon_composition",
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

    # Capability Plan 04 Task 1: every one of the six bindable capabilities
    # carries a real, generator-captured fixture - none uses a timestamp, a
    # random seed, or an external tool (see adapters.codon_composition's
    # module docstring).
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


def test_amino_acid_and_codon_usage_resolve_and_invoke_on_a_real_cds_fasta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    cds_path = tmp_path / "cds.fasta"
    _write_cds_fasta(cds_path)

    codon_binding = admitted.binding_source().resolve(_CODON_USAGE_ID)
    assert codon_binding is not None
    codon_result = codon_binding.invoke(None, {"cds_fasta": str(cds_path)})
    assert isinstance(codon_result, OrganelleResult)
    assert codon_result.status == "ok"
    assert codon_result.metrics["total_codons"] == 10  # 2 * 5 codons

    aa_binding = admitted.binding_source().resolve(_AMINO_ACID_ID)
    assert aa_binding is not None
    aa_result = aa_binding.invoke(None, {"cds_fasta": str(cds_path)})
    assert isinstance(aa_result, OrganelleResult)
    assert aa_result.status == "ok"
    assert aa_result.metrics["total"] > 0


def test_compute_codon_usage_and_compute_amino_acid_composition_invoke_json_metric_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    cds_path = tmp_path / "cds.fasta"
    _write_cds_fasta(cds_path)

    usage_binding = admitted.binding_source().resolve(_COMPUTE_CODON_USAGE_ID)
    assert usage_binding is not None
    usage_result = usage_binding.invoke(None, {"cds_fasta": str(cds_path)})
    assert isinstance(usage_result, OrganelleResult)
    assert usage_result.status == "ok"
    assert usage_result.metrics["codon_usage"]["total_codons"] == 10

    aa_binding = admitted.binding_source().resolve(_COMPUTE_AA_ID)
    assert aa_binding is not None
    aa_result = aa_binding.invoke(None, {"cds_fasta": str(cds_path)})
    assert isinstance(aa_result, OrganelleResult)
    assert aa_result.status == "ok"
    assert aa_result.metrics["amino_acid_composition"]["total"] > 0


def test_compute_enc_and_resolve_genetic_code_resolve_and_invoke_real_json_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)

    resolve_binding = admitted.binding_source().resolve(_RESOLVE_GENETIC_CODE_ID)
    assert resolve_binding is not None
    resolve_result = resolve_binding.invoke(None, {})
    assert isinstance(resolve_result, OrganelleResult)
    assert resolve_result.status == "ok"
    assert resolve_result.metrics["genetic_code_id"] == 1  # default organelle="mito"

    enc_binding = admitted.binding_source().resolve(_COMPUTE_ENC_ID)
    assert enc_binding is not None
    counts = {"TTT": 4, "TTC": 4, "ATG": 2}
    codon_to_aa = {"TTT": "F", "TTC": "F", "ATG": "M"}
    enc_result = enc_binding.invoke(None, {"counts": counts, "codon_to_aa": codon_to_aa})
    assert isinstance(enc_result, OrganelleResult)
    assert enc_result.status == "ok"
    assert enc_result.metrics["enc"] > 0
