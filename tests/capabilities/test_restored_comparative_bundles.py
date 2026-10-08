"""Capability Plan 03, Task 3: the restored ``comparative`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated 2 bundles under
``src/organelleverse/capabilities/comparative-*`` from
``docs/operations/restored-capabilities.toml`` - see
``organelleverse.capabilities.adapters.comparative``'s module docstring for
the full reasoning. This domain confirms, by a real bind attempt, the
``list[OrganelleGenome]`` blocker class first identified in the
``ir_boundary`` adapter: attempting a ``json`` override for
``comparative.compare_genes``'s ``genomes: list[OrganelleGenome]``
parameter raises ``OrganelleContractError: parameter 'genomes' has no
JSON-safe annotation for codec 'json'`` at ``verify_capability`` time (that
attempt is not committed to the adapter module - the module correctly
carries no entry for it).

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

_NORMALIZE_GENE_NAMES_ID = "comparative.normalize_gene_names"
_NORMALIZE_GENES_ID = "comparative.normalize_genes"
_COMPUTE_GENOME_IDENTITY_ID = "comparative.compute_genome_identity"
_RESTORED_IDS = (
    _NORMALIZE_GENE_NAMES_ID,
    _NORMALIZE_GENES_ID,
    _COMPUTE_GENOME_IDENTITY_ID,
    "comparative.compare_genes",
    "comparative.compare_genomes",
    "comparative.gene_table",
    "comparative.synteny",
)
_EXCLUDED_IDS = (
    "comparative.compute_gene_intersection",
    "comparative.compute_gene_sets",
    "comparative.compute_synteny_score",
    "comparative.write_gene_comparison",
    "comparative.write_gene_table",
    "comparative.write_genome_comparison",
    "comparative.write_synteny",
)

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {"comparative-normalize-gene-names", "comparative-normalize-genes",
    "comparative-compare-genes",
    "comparative-compare-genomes",
    "comparative-compute-genome-identity",
    "comparative-gene-table",
    "comparative-synteny",
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
        domain="comparative",
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

    # Capability Plan 04 Task 1: both bindable capabilities carry a real,
    # generator-captured fixture over the same gene-name list.
    fixture_cases_by_id = {
        item.capability_id: {
            fixture.case for fixture in parse_capability_bundle(item.path).fixtures
        }
        for item in report.generated
    }
    # The three bindable compute/normalize capabilities carry a real
    # generator-captured fixture; the four Ruling-3 sequence capabilities are
    # canonical-core (inspect-only, no fixture) like every other canonical
    # bundle.
    fixture_expected = {
        capability_id: (
            {"basic"}
            if ("normalize" in capability_id or "compute_genome_identity" in capability_id)
            else set()
        )
        for capability_id in _RESTORED_IDS
    }
    fixture_expected[_COMPUTE_GENOME_IDENTITY_ID] = {"basic", "single_query"}
    assert fixture_cases_by_id == fixture_expected


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


def test_normalize_gene_names_and_normalize_genes_resolve_and_invoke_real_json_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)

    names_binding = admitted.binding_source().resolve(_NORMALIZE_GENE_NAMES_ID)
    assert names_binding is not None
    names_result = names_binding.invoke(None, {"names": ["ATP1", "atp1", "COX1"]})
    assert isinstance(names_result, OrganelleResult)
    assert names_result.status == "ok"
    assert names_result.metrics["gene_name_map"]["ATP1"] == "atp1"

    genes_binding = admitted.binding_source().resolve(_NORMALIZE_GENES_ID)
    assert genes_binding is not None
    genes_result = genes_binding.invoke(None, {"gene_names": ["NAD1", "nad1"]})
    assert isinstance(genes_result, OrganelleResult)
    assert genes_result.status == "ok"
    assert "NAD1" in genes_result.metrics["gene_name_map"]


def test_compute_genome_identity_resolves_and_invokes_on_real_fasta_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mappy+edlib anchor-then-align pipeline binds and runs end to end."""
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_COMPUTE_GENOME_IDENTITY_ID)
    assert binding is not None

    from organelleverse.capabilities.adapters import comparative as comparative_adapter

    input_dir = tmp_path / "identity_inputs"
    input_dir.mkdir()
    reference_path = input_dir / "reference.fasta"
    query_a_path = input_dir / "query_a.fasta"
    query_b_path = input_dir / "query_b.fasta"
    reference_path.write_text(comparative_adapter._IDENTITY_REFERENCE_FASTA.content)
    query_a_path.write_text(comparative_adapter._IDENTITY_QUERY_A_FASTA.content)
    query_b_path.write_text(comparative_adapter._IDENTITY_QUERY_B_FASTA.content)

    result = binding.invoke(
        None,
        {
            "reference_fasta": str(reference_path),
            "query_fastas": [str(query_a_path), str(query_b_path)],
        },
    )

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    identity = result.metrics["genome_identity"]
    assert identity["reference_length"] == 2400
    assert identity["samples"] == ("query_a", "query_b")
    assert len(identity["windows"]) == 48
    assert {segment["strand"] for segment in identity["segments"]} == {1, -1}
    for sample in identity["samples"]:
        identities = [
            row["identity"]
            for row in identity["windows"]
            if row["sample"] == sample
        ]
        assert all(80.0 <= value <= 100.0 for value in identities)
