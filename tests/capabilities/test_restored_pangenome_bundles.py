"""Capability Plan 03, Task 3: the restored ``pangenome`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated 3 bundles under
``src/organelleverse/capabilities/pangenome-*`` from
``docs/operations/restored-capabilities.toml`` - see
``organelleverse.capabilities.adapters.pangenome``'s module docstring for
the full reasoning. Four of the six exclusions are the ``list[OrganelleGenome]``
blocker class (first identified in ``organelleverse.capabilities.adapters
.ir_boundary``); one is an unannotated parameter; one is the charter's
``LEGACY_RESULT`` gap.

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

_CHECK_ALL_BACKENDS_ID = "pangenome.check_all_backends"
_CHECK_BACKEND_ID = "pangenome.check_backend"
_INSTALL_HINT_ID = "pangenome.install_hint"
_BUILD_GRAPH_ID = "pangenome.build_graph"
_RECOMMEND_PARAMETERS_ID = "pangenome.recommend_parameters"
_GENERATED_IDS = (
    _CHECK_ALL_BACKENDS_ID,
    _CHECK_BACKEND_ID,
    _INSTALL_HINT_ID,
    _BUILD_GRAPH_ID,
    "pangenome.classify_sequences",
    "pangenome.gene_pav",
    "pangenome.pan_repeats",
)
_RESTORED_IDS = (*_GENERATED_IDS, _RECOMMEND_PARAMETERS_ID)
_EXCLUDED_IDS = (
    "pangenome.compute_gene_pav",
    "pangenome.write_pav",
)

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {
    "pangenome-check-all-backends",
    "pangenome-check-backend",
    "pangenome-install-hint",
    "pangenome-classify-sequences",
    "pangenome-gene-pav",
    "pangenome-pan-repeats",
    "pangenome-build-graph",
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


def test_the_three_bindable_capabilities_are_discovered_from_the_real_core_channel(
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
        domain="pangenome",
        ledger_path=_LEDGER_PATH,
        current_root=_CURRENT_ROOT,
        output_root=isolated_output_root,
    )

    generated_ids = {item.capability_id for item in report.generated}
    assert generated_ids == set(_GENERATED_IDS)

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


def test_build_graph_contract_is_sequence_core_and_declares_real_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _isolate_non_core_roots(tmp_path, monkeypatch)

    discovered = discover_capabilities()
    contract = discovered.describe(_BUILD_GRAPH_ID).bundle.contract

    assert contract.input_sequence is True
    assert contract.input_kind.value == "genome"
    assert {effect.value for effect in contract.side_effects} == {
        "read_files",
        "write_files",
        "subprocess",
    }
    record = verify_capability(
        _BUILD_GRAPH_ID,
        store=VerificationStore(home / "verifications"),
        environment=LocalVerificationEnvironment(discovered),
    )
    parameter_names = set(record.parameter_schema["properties"])
    assert parameter_names == {
        "method",
        "k",
        "threads",
        "n_haplotypes",
        "segment_length",
        "identity",
        "recommendation",
        "auto_adopt_recommendation",
        "reference_index",
        "pantools_memory_mb",
    }
    assert {"output_dir", "workspace", "executor", "runner"}.isdisjoint(parameter_names)


def test_recommendation_contract_is_agent_visible_and_path_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _isolate_non_core_roots(tmp_path, monkeypatch)
    discovered = discover_capabilities()
    contract = discovered.describe(_RECOMMEND_PARAMETERS_ID).bundle.contract
    assert contract.input_sequence is True
    assert contract.callable_locator == (
        "organelleverse.pangenome.tuning_service:recommend_parameters"
    )
    record = verify_capability(
        _RECOMMEND_PARAMETERS_ID,
        store=VerificationStore(home / "verifications"),
        environment=LocalVerificationEnvironment(discovered),
    )
    assert set(record.parameter_schema["properties"]) == {
        "threads",
        "run_repeatmasker",
        "species",
        "identity_margin",
        "repeat_multiplier",
        "round_to",
        "no_repeat_fallback",
        "include_rna",
    }
    assert {"path", "output_dir", "runner", "callable"}.isdisjoint(
        record.parameter_schema["properties"]
    )


def test_admitted_build_graph_invokes_managed_service_and_publishes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from organelleverse.core.artifacts import ArtifactRef
    from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
    from organelleverse.operations.adapters.json import invoke_json
    from organelleverse.operations.registry import OperationRegistry
    from organelleverse.pangenome import service
    from organelleverse.pangenome._contract import make_provenance

    first_path = tmp_path / "invoke-first.fa"
    first_path.write_text(">chr1\nACGTACGT\n", encoding="utf-8")
    second_path = tmp_path / "invoke-second.fa"
    second_path.write_text(">chr1\nACGTACGA\n", encoding="utf-8")
    genomes = [
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=ArtifactRef.from_path(first_path, kind="sequence", format="fasta"),
            metadata=OrganelleMetadata(accession="NC_001284"),
        ),
        OrganelleGenome(
            organelle="mitochondrion",
            sequence=ArtifactRef.from_path(second_path, kind="sequence", format="fasta"),
            metadata=OrganelleMetadata(accession="NC_001285"),
        ),
    ]

    def fake_core_build_graph(*args: object, **kwargs: object) -> OrganelleResult:
        graph = Path(kwargs["output_dir"]) / "pangenome.gfa"
        graph.write_text("H\tVN:Z:1.1\nS\tseg1\tACGT\n", encoding="utf-8")
        return OrganelleResult(
            operation_id=_BUILD_GRAPH_ID,
            operation_version="1.0",
            scope="mitochondrion",
            status="ok",
            summary_text="managed graph",
            flags=("graph_built",),
            artifacts=(ArtifactRef.from_path(graph, kind="pangenome_graph", format="gfa"),),
            provenance=make_provenance(
                operation_id=_BUILD_GRAPH_ID,
                parameters={},
                requested_backend="minigraph",
                actual_backend="minigraph",
                attempted_backends=("minigraph",),
            ),
        )

    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(service, "_core_build_graph", fake_core_build_graph)
    admitted = _admit_all(tmp_path, monkeypatch)
    registry = OperationRegistry()
    registry.attach_capability_source(admitted.binding_source())

    response = invoke_json(
        {
            "operation_id": _BUILD_GRAPH_ID,
            "input": [genome.model_dump(mode="json") for genome in genomes],
            "parameters": {"method": "minigraph", "threads": 2},
        },
        registry=registry,
        granted_side_effects=["read_files", "write_files", "subprocess"],
    )

    assert response["ok"] is True, response.get("error")
    result = response["result"]
    artifacts = result.artifacts if isinstance(result, OrganelleResult) else result["artifacts"]
    assert {item.kind if isinstance(item, ArtifactRef) else item["kind"] for item in artifacts} == {
        "pangenome_graph",
        "pangenome_run_record",
    }


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


def test_check_backend_and_install_hint_resolve_and_invoke_real_json_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)

    check_binding = admitted.binding_source().resolve(_CHECK_BACKEND_ID)
    assert check_binding is not None
    check_result = check_binding.invoke(None, {"name": "minigraph"})
    assert isinstance(check_result, OrganelleResult)
    assert check_result.status == "ok"
    assert check_result.metrics["backend_status"]["name"] == "minigraph"

    hint_binding = admitted.binding_source().resolve(_INSTALL_HINT_ID)
    assert hint_binding is not None
    hint_result = hint_binding.invoke(None, {"name": "minigraph"})
    assert isinstance(hint_result, OrganelleResult)
    assert hint_result.status == "ok"
    assert isinstance(hint_result.metrics["hint_text"], str)


def test_check_all_backends_resolves_and_invokes_with_zero_agent_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_CHECK_ALL_BACKENDS_ID)
    assert binding is not None

    result = binding.invoke(None, {})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert "minigraph" in result.metrics["backend_status"]
