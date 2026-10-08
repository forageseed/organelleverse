"""Capability Plan 03, Task 3: the restored ``structure`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated 6 bundles under
``src/organelleverse/capabilities/structure-*`` from
``docs/operations/restored-capabilities.toml`` - see
``organelleverse.capabilities.adapters.structure``'s module docstring for
the full per-capability reasoning. ``structure.introns`` matches the
generator's generic ``canonical_core`` shape on its own (no adapter entry
needed for it); the other five required an explicit override.

Mirrors ``tests/capabilities/test_restored_format_conversion_bundles.py``
and ``test_restored_trans_splicing_bundles.py`` (canonical_core + adapter
mixed in one domain).
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
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OrganelleResult
from tests._paths import PROJECT_ROOT
from tests.capabilities._bundle_regeneration_check import (
    assert_regenerated_domain_matches_committed,
)

_COMPUTE_MULTICONF_ID = "structure.compute_multiconf"
_COMPUTE_REPEATS_ID = "structure.compute_repeats"
_INTRONS_ID = "structure.introns"
_MULTICONF_ID = "structure.multiconf"
_REPEATS_ID = "structure.repeats"
_RESOLVE_CONFIGS_ID = "structure.resolve_configs"
_RESTORED_IDS = (
    _COMPUTE_MULTICONF_ID,
    _COMPUTE_REPEATS_ID,
    _INTRONS_ID,
    _MULTICONF_ID,
    _REPEATS_ID,
    _RESOLVE_CONFIGS_ID,
)
_EXCLUDED_IDS = (
    "structure.write_introns",
    "structure.write_multiconf",
    "structure.write_repeats",
    "structure.write_resolve_configs",
)

_MITO_GBK = PROJECT_ROOT / "tests" / "data" / "mito.gbk"

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {
    "structure-compute-multiconf",
    "structure-compute-repeats",
    "structure-introns",
    "structure-multiconf",
    "structure-repeats",
    "structure-resolve-configs",
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


def _write_fasta(path: Path) -> None:
    # A direct-repeat pair (>= 50bp) at both ends, so multiconf/repeats find something.
    repeat = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC"
    path.write_text(f">contig1\n{repeat}TTTTGGGGCCCC{repeat}\n")


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
        domain="structure",
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

    # Capability Plan 04 Task 1: only the two json_metric capabilities carry
    # a real, generator-captured fixture - multiconf/repeats/resolve_configs
    # (canonical) embed real wall-clock timestamps in their provenance, and
    # introns is canonical_core (genome-input, out of the fixture
    # evaluator's supported input_kind=none scope) - see adapters.structure's
    # module docstring.
    fixture_cases_by_id = {
        item.capability_id: {
            fixture.case for fixture in parse_capability_bundle(item.path).fixtures
        }
        for item in report.generated
    }
    assert fixture_cases_by_id == {
        _COMPUTE_MULTICONF_ID: {"basic"},
        _COMPUTE_REPEATS_ID: {"basic"},
        _INTRONS_ID: set(),
        _MULTICONF_ID: set(),
        _REPEATS_ID: set(),
        _RESOLVE_CONFIGS_ID: set(),
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


def test_introns_resolves_and_invokes_the_canonical_core_binding_on_a_real_genome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_INTRONS_ID)
    assert binding is not None

    genome = OrganelleGenome(
        organelle="mitochondrion",
        annotation=ArtifactRef.from_path(_MITO_GBK, kind="annotation", format="genbank"),
    )
    result = binding.invoke(genome, {})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _INTRONS_ID
    assert result.status == "ok"


def test_multiconf_and_repeats_resolve_and_invoke_on_a_real_fasta_with_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    fasta_path = tmp_path / "genome.fasta"
    _write_fasta(fasta_path)

    multiconf_binding = admitted.binding_source().resolve(_MULTICONF_ID)
    assert multiconf_binding is not None
    multiconf_result = multiconf_binding.invoke(None, {"genome_fasta": str(fasta_path)})
    assert isinstance(multiconf_result, OrganelleResult)
    assert multiconf_result.operation_id == _MULTICONF_ID
    assert multiconf_result.status == "ok"

    repeats_binding = admitted.binding_source().resolve(_REPEATS_ID)
    assert repeats_binding is not None
    repeats_result = repeats_binding.invoke(None, {"genome_fasta": str(fasta_path)})
    assert isinstance(repeats_result, OrganelleResult)
    assert repeats_result.operation_id == _REPEATS_ID
    assert repeats_result.status == "ok"


def test_resolve_configs_resolves_and_invokes_the_fasta_only_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    fasta_path = tmp_path / "genome.fasta"
    _write_fasta(fasta_path)

    binding = admitted.binding_source().resolve(_RESOLVE_CONFIGS_ID)
    assert binding is not None
    result = binding.invoke(None, {"genome_fasta": str(fasta_path)})
    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _RESOLVE_CONFIGS_ID
    assert result.status == "ok"


def test_compute_multiconf_and_compute_repeats_resolve_and_invoke_json_metric_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    fasta_path = tmp_path / "genome.fasta"
    _write_fasta(fasta_path)

    multiconf_binding = admitted.binding_source().resolve(_COMPUTE_MULTICONF_ID)
    assert multiconf_binding is not None
    multiconf_result = multiconf_binding.invoke(None, {"fasta_path": str(fasta_path)})
    assert isinstance(multiconf_result, OrganelleResult)
    assert multiconf_result.status == "ok"
    assert "multiconf" in multiconf_result.metrics

    repeats_binding = admitted.binding_source().resolve(_COMPUTE_REPEATS_ID)
    assert repeats_binding is not None
    repeats_result = repeats_binding.invoke(None, {"fasta_path": str(fasta_path)})
    assert isinstance(repeats_result, OrganelleResult)
    assert repeats_result.status == "ok"
    assert "repeats" in repeats_result.metrics
