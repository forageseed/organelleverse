"""Capability Plan 03/05: the restored ``coevolution`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated 19 bundles
under ``src/organelleverse/capabilities/coevolution-*`` from
``docs/operations/restored-capabilities.toml`` - see
``organelleverse.capabilities.adapters.coevolution``'s module docstring for
the full per-capability reasoning, including a NaN caveat carried across
most of the numeric ERC2/BXB cores (same shape as the ``diversity``
adapter's Tajima's D caveat) that this test suite avoids by construction
(every real invoke below uses complete, concordant, non-degenerate inputs).

Three more, Capability Plan 05: ``phylogenomics``, ``run_orthofinder``, and
``run_coevolution`` were blocked solely on required directory parameters
until ``ParameterCodec.DIRECTORY`` shipped - ``proteome_dir``/
``orthofinder_dir`` get ``path_role = "input"`` (pre-existing trees), and
``run_orthofinder``/``run_coevolution``'s ``output_dir`` gets ``path_role =
"output"``.

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

_BXB_LENGTHS_DLCPAR_ID = "coevolution.bxb_lengths_dlcpar"
_BXB_LENGTHS_LCA_ID = "coevolution.bxb_lengths_lca"
_COMPARTMENT_MAP_FROM_PREFIX_ID = "coevolution.compartment_map_from_prefix"
_CORRELATION_MATRIX_ID = "coevolution.correlation_matrix"
_EXTRACT_BRANCH_LENGTHS_ID = "coevolution.extract_branch_lengths"
_FILTER_OUTLIER_TREE_ID = "coevolution.filter_outlier_tree"
_FISHER_ID = "coevolution.fisher"
_NORMALIZE_EDGE_MEAN_ID = "coevolution.normalize_edge_mean"
_NORMALIZE_GENE_SUM_ID = "coevolution.normalize_gene_sum"
_PERM_TEST_ID = "coevolution.perm_test"
_PROJECT_PATHS_ID = "coevolution.project_paths"
_RATE_COVARIATION_ID = "coevolution.rate_covariation"
_RECONCILE_TREES_ID = "coevolution.reconcile_trees"
_RESIDUAL_MATRIX_ID = "coevolution.residual_matrix"
_RUN_ERC_ID = "coevolution.run_erc"
_SPECIES_TREE_EDGES_ID = "coevolution.species_tree_edges"
_PHYLOGENOMICS_ID = "coevolution.phylogenomics"
_RUN_ORTHOFINDER_ID = "coevolution.run_orthofinder"
_RUN_COEVOLUTION_ID = "coevolution.run_coevolution"
_RESTORED_IDS = (
    _BXB_LENGTHS_DLCPAR_ID,
    _BXB_LENGTHS_LCA_ID,
    _COMPARTMENT_MAP_FROM_PREFIX_ID,
    _CORRELATION_MATRIX_ID,
    _EXTRACT_BRANCH_LENGTHS_ID,
    _FILTER_OUTLIER_TREE_ID,
    _FISHER_ID,
    _NORMALIZE_EDGE_MEAN_ID,
    _NORMALIZE_GENE_SUM_ID,
    _PERM_TEST_ID,
    _PROJECT_PATHS_ID,
    _RATE_COVARIATION_ID,
    _RECONCILE_TREES_ID,
    _RESIDUAL_MATRIX_ID,
    _RUN_ERC_ID,
    _SPECIES_TREE_EDGES_ID,
    _PHYLOGENOMICS_ID,
    _RUN_ORTHOFINDER_ID,
    _RUN_COEVOLUTION_ID,
    "coevolution.coevolution_network",
)
_EXCLUDED_IDS = (
    "coevolution.erc_correlation_matrix",
    "coevolution.erc_master_paths",
    "coevolution.erc_residual_matrix",
    "coevolution.fisher_transform",
    "coevolution.parse_newick",
    "coevolution.perm_test_matrix",
)

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {
    "coevolution-bxb-lengths-dlcpar",
    "coevolution-bxb-lengths-lca",
    "coevolution-compartment-map-from-prefix",
    "coevolution-correlation-matrix",
    "coevolution-extract-branch-lengths",
    "coevolution-filter-outlier-tree",
    "coevolution-fisher",
    "coevolution-normalize-edge-mean",
    "coevolution-normalize-gene-sum",
    "coevolution-perm-test",
    "coevolution-project-paths",
    "coevolution-rate-covariation",
    "coevolution-reconcile-trees",
    "coevolution-residual-matrix",
    "coevolution-run-erc",
    "coevolution-species-tree-edges",
    "coevolution-phylogenomics",
    "coevolution-run-orthofinder",
    "coevolution-run-coevolution",
    "coevolution-coevolution-network"
}

# A balanced, fully-resolved 3-species tree so downstream BXB/ERC helpers
# never hit the missing-edge NaN case for these inputs.
_SPECIES_TREE = "((A:1.0,B:1.0)N0:1.0,C:2.0)root;"
_GENE_TREE_1 = "((A:1.2,B:0.9)N0:0.8,C:1.9)root;"
_GENE_TREE_2 = "((A:1.1,B:1.0)N0:0.9,C:2.1)root;"
_GENE_TREES = {"gene1": _GENE_TREE_1, "gene2": _GENE_TREE_2}


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


def test_the_nineteen_bindable_capabilities_are_discovered_from_the_real_core_channel(
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
        domain="coevolution",
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


def test_species_tree_edges_and_reconcile_trees_resolve_and_invoke_real_newick_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)

    edges_binding = admitted.binding_source().resolve(_SPECIES_TREE_EDGES_ID)
    assert edges_binding is not None
    edges_result = edges_binding.invoke(None, {"species_tree_newick": _SPECIES_TREE})
    assert isinstance(edges_result, OrganelleResult)
    assert edges_result.status == "ok"
    assert len(edges_result.metrics["edges"]) > 0

    reconcile_binding = admitted.binding_source().resolve(_RECONCILE_TREES_ID)
    assert reconcile_binding is not None
    reconcile_result = reconcile_binding.invoke(
        None,
        {"gene_tree_newick": _GENE_TREE_1, "species_tree_newick": _SPECIES_TREE},
    )
    assert isinstance(reconcile_result, OrganelleResult)
    assert reconcile_result.operation_id == _RECONCILE_TREES_ID
    assert reconcile_result.status == "ok"


def test_run_erc_and_extract_branch_lengths_resolve_and_invoke_a_real_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)

    run_erc_binding = admitted.binding_source().resolve(_RUN_ERC_ID)
    assert run_erc_binding is not None
    run_erc_result = run_erc_binding.invoke(
        None, {"gene_trees": _GENE_TREES, "species_tree_newick": _SPECIES_TREE}
    )
    assert isinstance(run_erc_result, OrganelleResult)
    assert run_erc_result.operation_id == _RUN_ERC_ID
    assert run_erc_result.status == "ok"
    assert run_erc_result.metrics["n_genes"] == 2

    extract_binding = admitted.binding_source().resolve(_EXTRACT_BRANCH_LENGTHS_ID)
    assert extract_binding is not None
    extract_result = extract_binding.invoke(None, {"gene_trees": _GENE_TREES})
    assert isinstance(extract_result, OrganelleResult)
    assert extract_result.operation_id == _EXTRACT_BRANCH_LENGTHS_ID
    assert extract_result.status == "ok"


def test_project_paths_residual_matrix_and_correlation_matrix_chain_real_json_dicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercises the same 3-stage pipeline ``run_erc`` composes, one call at a time."""
    admitted = _admit_all(tmp_path, monkeypatch)

    project_binding = admitted.binding_source().resolve(_PROJECT_PATHS_ID)
    assert project_binding is not None
    project_result = project_binding.invoke(
        None, {"gene_trees": _GENE_TREES, "species_tree_newick": _SPECIES_TREE}
    )
    assert isinstance(project_result, OrganelleResult)
    assert project_result.status == "ok"
    path_matrix = dict(project_result.metrics["path_matrix"])
    assert set(path_matrix) == {"gene1", "gene2"}

    residual_binding = admitted.binding_source().resolve(_RESIDUAL_MATRIX_ID)
    assert residual_binding is not None
    residual_result = residual_binding.invoke(
        None, {"path_matrix": {k: list(v) for k, v in path_matrix.items()}}
    )
    assert isinstance(residual_result, OrganelleResult)
    assert residual_result.status == "ok"

    correlation_binding = admitted.binding_source().resolve(_CORRELATION_MATRIX_ID)
    assert correlation_binding is not None
    residuals = {k: list(v) for k, v in dict(residual_result.metrics["residual_matrix"]).items()}
    correlation_result = correlation_binding.invoke(None, {"residuals": residuals})
    assert isinstance(correlation_result, OrganelleResult)
    assert correlation_result.status == "ok"


def test_rate_covariation_resolves_and_invokes_a_real_union_typed_parameter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves ``dict[str, list[float] | dict[str, float]]`` is really JSON-bindable."""
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_RATE_COVARIATION_ID)
    assert binding is not None

    branch_lengths = {"gene1": [1.0, 2.0, 3.0], "gene2": [1.1, 2.1, 2.9], "gene3": [0.9, 1.9, 3.1]}
    result = binding.invoke(None, {"branch_lengths": branch_lengths})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _RATE_COVARIATION_ID
    assert result.status == "ok"


def test_fisher_and_perm_test_resolve_and_invoke_real_nested_json_dicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    cor = {"g1": {"g2": 0.8}, "g2": {"g1": 0.8}}
    count = {"g1": {"g2": 10}, "g2": {"g1": 10}}

    fisher_binding = admitted.binding_source().resolve(_FISHER_ID)
    assert fisher_binding is not None
    fisher_result = fisher_binding.invoke(None, {"cor": cor, "count": count})
    assert isinstance(fisher_result, OrganelleResult)
    assert fisher_result.status == "ok"
    assert fisher_result.metrics["fisher_z"]["g1"]["g2"] != 0.0

    perm_binding = admitted.binding_source().resolve(_PERM_TEST_ID)
    assert perm_binding is not None
    perm_result = perm_binding.invoke(None, {"gene_set": ["g1", "g2"], "cor": cor})
    assert isinstance(perm_result, OrganelleResult)
    assert perm_result.status == "ok"
    assert "p_value" in perm_result.metrics["perm_test"]


def test_normalize_gene_sum_and_normalize_edge_mean_resolve_and_invoke_real_matrices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    matrix = {"gene1": {"e1": 1.0, "e2": 2.0}, "gene2": {"e1": 1.5, "e2": 1.5}}

    sum_binding = admitted.binding_source().resolve(_NORMALIZE_GENE_SUM_ID)
    assert sum_binding is not None
    sum_result = sum_binding.invoke(None, {"matrix": matrix})
    assert isinstance(sum_result, OrganelleResult)
    assert sum_result.status == "ok"

    mean_binding = admitted.binding_source().resolve(_NORMALIZE_EDGE_MEAN_ID)
    assert mean_binding is not None
    mean_result = mean_binding.invoke(None, {"matrix": matrix})
    assert isinstance(mean_result, OrganelleResult)
    assert mean_result.status == "ok"


def test_filter_outlier_tree_and_compartment_map_from_prefix_resolve_and_invoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)

    filter_binding = admitted.binding_source().resolve(_FILTER_OUTLIER_TREE_ID)
    assert filter_binding is not None
    filter_result = filter_binding.invoke(None, {"edge_lengths": {"e1": 1.0, "e2": 1.2}})
    assert isinstance(filter_result, OrganelleResult)
    assert filter_result.status == "ok"
    assert filter_result.metrics["filtered_edges"] == {"e1": 1.0, "e2": 1.2}

    compartment_binding = admitted.binding_source().resolve(_COMPARTMENT_MAP_FROM_PREFIX_ID)
    assert compartment_binding is not None
    compartment_result = compartment_binding.invoke(
        None, {"genes": ["MT_atp1", "nad2"], "prefix_rules": {"MT_": "mito"}}
    )
    assert isinstance(compartment_result, OrganelleResult)
    assert compartment_result.status == "ok"
    assert compartment_result.metrics["compartment_map"]["MT_atp1"] == "mito"


def test_bxb_lengths_lca_resolves_and_invokes_a_real_lca_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    edges_binding = admitted.binding_source().resolve(_SPECIES_TREE_EDGES_ID)
    assert edges_binding is not None
    edges = list(
        edges_binding.invoke(None, {"species_tree_newick": _SPECIES_TREE}).metrics["edges"]
    )

    lca_binding = admitted.binding_source().resolve(_BXB_LENGTHS_LCA_ID)
    assert lca_binding is not None
    lca_result = lca_binding.invoke(
        None,
        {
            "gene_tree_newick": _GENE_TREE_1,
            "species_tree_newick": _SPECIES_TREE,
            "edges": edges,
        },
    )
    assert isinstance(lca_result, OrganelleResult)
    assert lca_result.status == "ok"


def test_bxb_lengths_dlcpar_resolves_and_invokes_reading_real_recon_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real (empty) recon files: an honest degenerate case, not a fabricated success.

    Empty ``.locus.recon``/``.coal.recon`` files map nothing, so every
    species-tree edge is genuinely unresolved - the implementation's own
    documented NaN-for-missing-data behavior (see the adapter module's NaN
    caveat), not a binding defect. This asserts that real, documented
    failure mode rather than forcing a fabricated non-NaN result.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    edges_binding = admitted.binding_source().resolve(_SPECIES_TREE_EDGES_ID)
    assert edges_binding is not None
    edges = list(
        edges_binding.invoke(None, {"species_tree_newick": _SPECIES_TREE}).metrics["edges"]
    )

    dlcpar_binding = admitted.binding_source().resolve(_BXB_LENGTHS_DLCPAR_ID)
    assert dlcpar_binding is not None
    locus_recon = tmp_path / "gene1.dlcpar.locus.recon"
    coal_recon = tmp_path / "gene1.dlcpar.coal.recon"
    locus_recon.write_text("")
    coal_recon.write_text("")

    from organelleverse.core.errors import OrganelleContractError

    with pytest.raises(OrganelleContractError, match="finite JSON"):
        dlcpar_binding.invoke(
            None,
            {
                "gene_tree_newick": _GENE_TREE_1,
                "locus_recon_path": str(locus_recon),
                "coal_recon_path": str(coal_recon),
                "edges": edges,
            },
        )


def test_phylogenomics_resolves_and_invokes_a_real_input_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``orthofinder_dir`` (``path_role="input"``), proven for real.

    ``executor`` is never Agent-exposed, so every real call takes the
    planned branch - ``orthofinder_dir`` is only ``str()``-embedded into the
    planned argv, but DIRECTORY-input containment still runs first.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_PHYLOGENOMICS_ID)
    assert binding is not None

    orthofinder_dir = tmp_path / "orthofinder_results"
    orthofinder_dir.mkdir()

    result = binding.invoke(None, {"orthofinder_dir": str(orthofinder_dir)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _PHYLOGENOMICS_ID
    assert result.status == "ok"
    assert "planned" in result.flags


def test_run_orthofinder_resolves_with_a_real_input_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent supplies inputs; intermediates use managed storage."""
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_RUN_ORTHOFINDER_ID)
    assert binding is not None

    proteome_dir = tmp_path / "proteomes"
    proteome_dir.mkdir()
    (proteome_dir / "species_a.fasta").write_text(">p1\nMAAAAL\n")
    result = binding.invoke(None, {"proteome_dir": str(proteome_dir)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _RUN_ORTHOFINDER_ID
    assert result.status == "ok"
    assert "planned" in result.flags


def test_run_coevolution_resolves_without_creating_a_plan_only_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan-only call does not create a managed output directory.

    Verified by real invocation: the returned ``OrganelleResult`` carries
    ``operation_id = "coevolution.run_orthofinder"``, not
    ``"coevolution.run_coevolution"`` - ``run_coevolution`` immediately
    returns ``run_orthofinder``'s own planned result when ``executor`` is
    ``None`` (always true for a real Agent call), the same cross-function
    ``operation_id`` quirk already documented for
    ``population.detect_numt``, left exactly as found.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_RUN_COEVOLUTION_ID)
    assert binding is not None

    proteome_dir = tmp_path / "proteomes"
    proteome_dir.mkdir()
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    result = binding.invoke(None, {"proteome_dir": str(proteome_dir)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _RUN_ORTHOFINDER_ID
    assert result.status == "ok"
    assert not (tmp_path / "cache" / "runs" / "coevolution.run_coevolution").exists()
