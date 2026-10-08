"""Capability Plan 03, Task 3: the restored ``visualization`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated 10 bundles
under ``src/organelleverse/capabilities/visualization-*`` from
``docs/operations/restored-capabilities.toml`` - see
``organelleverse.capabilities.adapters.visualization``'s module docstring
for the full reasoning. This is Task 3's largest domain by ledger count
(40) and its heaviest exclusion rate: a newly discovered fifth blocker
class - this generator's own ``_looks_json_safe`` only recognizes the bare
container spellings ``list``/``dict``/``tuple``, never ``typing``/
``collections.abc`` aliases like ``Sequence``/``Mapping`` - accounts for 19
of the 30 exclusions on its own (confirmed empirically:
``_looks_json_safe`` on ``plot_erc_distribution``'s ``Sequence[Mapping[str,
Any]]`` parameter returns ``False``).

``plot_rscu_usage`` and ``plot_gfa_graph`` both resolve via the single-arm
``str | Path`` union reasoning (first used in ``morphology``/``transfer``),
confirmed against the *real* runtime type via ``get_type_hints`` +
``get_args`` (not just the AST), even though each union also contains an
unresolvable ``Sequence``/``Mapping``/custom-type arm.

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

_CLASSIFY_GENE_ID = "visualization.classify_gene"
_PLOT_GENOME_MAP_ID = "visualization.plot_genome_map"
_PLOT_GENOME_IDENTITY_ID = "visualization.plot_genome_identity"
_PLOT_GFA_GRAPH_ID = "visualization.plot_gfa_graph"
_PLOT_HEATMAP_ID = "visualization.plot_heatmap"
_PLOT_OGDRAW_MAP_ID = "visualization.plot_ogdraw_map"
_PLOT_PAN_CIRCULAR_ID = "visualization.plot_pan_circular"
_PLOT_RSCU_USAGE_ID = "visualization.plot_rscu_usage"
_PLOT_SCATTER_ID = "visualization.plot_scatter"
_PLOT_TREE_ID = "visualization.plot_tree"
_SPREAD_LABELS_ID = "visualization.spread_labels"
_RESTORED_IDS = (
    _CLASSIFY_GENE_ID,
    _PLOT_GENOME_MAP_ID,
    _PLOT_GENOME_IDENTITY_ID,
    _PLOT_GFA_GRAPH_ID,
    _PLOT_HEATMAP_ID,
    _PLOT_OGDRAW_MAP_ID,
    _PLOT_PAN_CIRCULAR_ID,
    _PLOT_RSCU_USAGE_ID,
    _PLOT_SCATTER_ID,
    _PLOT_TREE_ID,
    _SPREAD_LABELS_ID,
    "visualization.plot_collinearity",
    "visualization.plot_gbdraw",
    "visualization.plot_gene_structure",
    "visualization.plot_erc_distribution",
    "visualization.plot_erc_group_ridges",
    "visualization.plot_erc_network",
    "visualization.plot_erc_pair_scatter",
    "visualization.plot_erc_significance",
    "visualization.plot_network",
    "visualization.plot_rna_editing_summary",
    "visualization.plot_selection_summary",
    "visualization.plot_track_density",
    "visualization.plot_transfer_schematic",
    "visualization.plot_transfer_tracks",
    "visualization.plot_qc_dashboard",
    "visualization.plot_matrix_heatmap",
    "visualization.plot_splicing_schematic",
    "visualization.plot_localization_bars",
    "visualization.summarize_synteny_matrix",
    "visualization.ideogram",
    "visualization.nuclear_transfer_ideogram",
    "visualization.plot_synteny_matrix",
    "visualization.write_erc_visualization_report",
    "visualization.write_visualization_preview_report",
    "visualization.save_plot",
    "visualization.write_ideogram",
    "visualization.write_nuclear_transfer_ideogram",
)
_EXCLUDED_IDS = (
    "visualization.draw_mito_map",
    "visualization.parse_genbank",
    "visualization.read_gfa_graph",
)

_MITO_GBK = PROJECT_ROOT / "tests" / "data" / "mito.gbk"

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {
    "visualization-classify-gene",
    "visualization-plot-collinearity",
    "visualization-plot-erc-distribution",
    "visualization-plot-erc-group-ridges",
    "visualization-plot-erc-network",
    "visualization-plot-erc-pair-scatter",
    "visualization-plot-erc-significance",
    "visualization-plot-network",
    "visualization-plot-rna-editing-summary",
    "visualization-plot-selection-summary",
    "visualization-plot-track-density",
    "visualization-plot-transfer-schematic",
    "visualization-plot-transfer-tracks",
    "visualization-plot-qc-dashboard",
    "visualization-plot-matrix-heatmap",
    "visualization-plot-splicing-schematic",
    "visualization-plot-localization-bars",
    "visualization-summarize-synteny-matrix",
    "visualization-ideogram",
    "visualization-nuclear-transfer-ideogram",
    "visualization-plot-gbdraw",
    "visualization-plot-gene-structure",
    "visualization-plot-genome-map",
    "visualization-plot-genome-identity",
    "visualization-plot-gfa-graph",
    "visualization-plot-heatmap",
    "visualization-plot-ogdraw-map",
    "visualization-plot-pan-circular",
    "visualization-plot-rscu-usage",
    "visualization-plot-scatter",
    "visualization-plot-tree",
    "visualization-spread-labels",
    "visualization-plot-synteny-matrix",
    "visualization-write-erc-visualization-report",
    "visualization-write-visualization-preview-report",

    "visualization-save-plot",
    "visualization-write-ideogram",
    "visualization-write-nuclear-transfer-ideogram",
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


def test_the_ten_bindable_capabilities_are_discovered_from_the_real_core_channel(
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
        domain="visualization",
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


def test_classify_gene_and_spread_labels_resolve_and_invoke_real_json_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)

    classify_binding = admitted.binding_source().resolve(_CLASSIFY_GENE_ID)
    assert classify_binding is not None
    classify_result = classify_binding.invoke(None, {"name": "atp1"})
    assert isinstance(classify_result, OrganelleResult)
    assert classify_result.status == "ok"
    assert isinstance(classify_result.metrics["gene_class"], str)

    spread_binding = admitted.binding_source().resolve(_SPREAD_LABELS_ID)
    assert spread_binding is not None
    spread_result = spread_binding.invoke(
        None, {"gene_mids": [100.0, 5000.0, 9000.0], "genome_length": 15000.0}
    )
    assert isinstance(spread_result, OrganelleResult)
    assert spread_result.status == "ok"
    assert len(spread_result.metrics["spread_labels"][0]) == 3


def test_plot_genome_map_and_plot_ogdraw_map_resolve_and_invoke_on_a_real_genbank_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)

    genome_map_binding = admitted.binding_source().resolve(_PLOT_GENOME_MAP_ID)
    assert genome_map_binding is not None
    genome_map_result = genome_map_binding.invoke(None, {"genbank_path": str(_MITO_GBK)})
    assert isinstance(genome_map_result, OrganelleResult)
    assert genome_map_result.status == "ok"
    assert len(genome_map_result.artifacts) == 1

    ogdraw_binding = admitted.binding_source().resolve(_PLOT_OGDRAW_MAP_ID)
    assert ogdraw_binding is not None
    ogdraw_result = ogdraw_binding.invoke(None, {"genbank_path": str(_MITO_GBK)})
    assert isinstance(ogdraw_result, OrganelleResult)
    assert ogdraw_result.status == "ok"
    assert len(ogdraw_result.artifacts) == 1


def test_genome_and_ogdraw_maps_take_several_genbank_files_on_one_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    admitted = _admit_all(tmp_path, monkeypatch)
    first, second = tmp_path / "a.gb", tmp_path / "b.gb"
    shutil.copy(_MITO_GBK, first)
    shutil.copy(_MITO_GBK, second)
    for capability in (_PLOT_GENOME_MAP_ID, _PLOT_OGDRAW_MAP_ID):
        binding = admitted.binding_source().resolve(capability)
        assert binding is not None, capability
        result = binding.invoke(None, {"genbank_path": [str(first), str(second)]})
        assert isinstance(result, OrganelleResult), capability
        assert result.status == "ok", capability
        # one request means one figure, however many files it draws
        assert len(result.artifacts) == 1, capability


def test_a_single_path_string_still_means_one_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_PLOT_OGDRAW_MAP_ID)
    assert binding is not None
    result = binding.invoke(None, {"genbank_path": str(_MITO_GBK)})
    assert result.status == "ok"


def test_plot_pan_circular_resolves_and_invokes_on_a_real_genbank_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_PLOT_PAN_CIRCULAR_ID)
    assert binding is not None

    result = binding.invoke(None, {"reference_gbk": str(_MITO_GBK)})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert len(result.artifacts) >= 1


def test_plot_gfa_graph_resolves_and_invokes_on_a_real_gfa_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_PLOT_GFA_GRAPH_ID)
    assert binding is not None

    gfa_path = tmp_path / "graph.gfa"
    gfa_path.write_text(
        "H\tVN:Z:1.0\n"
        "S\t1\tACGTACGTACGT\tLN:i:12\tdp:f:10.0\n"
        "S\t2\tTTTTGGGGCCCC\tLN:i:12\tdp:f:8.0\n"
        "L\t1\t+\t2\t+\t0M\n"
    )

    result = binding.invoke(None, {"gfa": str(gfa_path)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _PLOT_GFA_GRAPH_ID


def test_plot_heatmap_and_plot_scatter_resolve_and_invoke_real_json_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)

    heatmap_binding = admitted.binding_source().resolve(_PLOT_HEATMAP_ID)
    assert heatmap_binding is not None
    heatmap_result = heatmap_binding.invoke(
        None, {"data": {"gene1": {"gene2": 0.8}, "gene2": {"gene1": 0.8}}}
    )
    assert isinstance(heatmap_result, OrganelleResult)
    assert heatmap_result.status == "ok"
    assert len(heatmap_result.artifacts) == 1

    scatter_binding = admitted.binding_source().resolve(_PLOT_SCATTER_ID)
    assert scatter_binding is not None
    scatter_result = scatter_binding.invoke(None, {"x": [0.1, 0.2, 0.3], "y": [0.4, 0.5, 0.6]})
    assert isinstance(scatter_result, OrganelleResult)
    assert scatter_result.status == "ok"
    assert len(scatter_result.artifacts) == 1


def test_plot_tree_resolves_and_invokes_on_a_real_newick_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_PLOT_TREE_ID)
    assert binding is not None

    newick_path = tmp_path / "tree.nwk"
    newick_path.write_text("((A:1.0,B:1.0):1.0,C:2.0);\n")

    result = binding.invoke(None, {"newick_path": str(newick_path)})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert len(result.artifacts) == 1


def test_plot_rscu_usage_resolves_and_invokes_proving_the_single_arm_path_codec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the four-way union (two container-alias arms, str, Path) really binds."""
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_PLOT_RSCU_USAGE_ID)
    assert binding is not None

    rscu_path = tmp_path / "rscu.tsv"
    rscu_path.write_text("codon\tAA\tRSCU\nTTT\tF\t1.2\nTTC\tF\t0.8\nTTA\tL\t0.9\n")

    result = binding.invoke(None, {"rscu_rows": str(rscu_path)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _PLOT_RSCU_USAGE_ID


def test_plot_genome_identity_resolves_and_invokes_real_json_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mVISTA identity plot binds like the other row-driven suite plots."""
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_PLOT_GENOME_IDENTITY_ID)
    assert binding is not None

    windows = [
        {"sample": "query_a", "start": 0, "end": 100, "identity": 96.0},
        {"sample": "query_a", "start": 100, "end": 200, "identity": 55.0},
        {"sample": "query_b", "start": 0, "end": 100, "identity": 88.0},
        {"sample": "query_b", "start": 100, "end": 200, "identity": 99.5},
    ]
    features = [
        {"key": "rbcL|1", "name": "rbcL", "start": 10, "end": 180, "strand": 1, "category": "cds"},
        {"key": "trnH|-1", "name": "trnH", "start": 190, "end": 260, "strand": -1, "category": "nc_gene"},
    ]
    result = binding.invoke(
        None,
        {
            "windows": windows,
            "features": features,
            "reference_length": 400,
            "region": [0, 300],
        },
    )

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert result.operation_id == _PLOT_GENOME_IDENTITY_ID
    assert len(result.artifacts) == 1
