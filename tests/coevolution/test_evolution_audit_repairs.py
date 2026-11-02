"""Discriminating regressions for the October scientific correctness review."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from scipy import stats

from organelleverse.coevolution._bxb_extract import bxb_lengths_lca, species_tree_edges
from organelleverse.coevolution.coevolution import parse_newick, reconcile_trees
from organelleverse.coevolution.erc_engine import correlation_matrix, fisher, perm_test, project_paths, run_erc


@pytest.mark.parametrize(('r', 'state'), [(1.0, 'perfect_positive'), (-1.0, 'perfect_negative')])
def test_fisher_perfect_correlation_has_explicit_json_state(r, state):
    value = fisher({'a': {'b': r}}, {'a': {'b': 20}})['a']['b']
    assert value == {'z': None, 'p_value': 0.0, 'status': state}
    json.dumps(value, allow_nan=False)


def test_fisher_finite_and_untestable_are_distinct():
    estimate = fisher({'a': {'b': .8}}, {'a': {'b': 10}})['a']['b']
    assert estimate['z'] == pytest.approx(math.atanh(.8) * math.sqrt(7))
    assert estimate['p_value'] == pytest.approx(2 * stats.norm.sf(abs(estimate['z'])))
    assert fisher({'a': {'b': .8}}, {'a': {'b': 3}})['a']['b']['status'] == 'insufficient_overlap'


def test_min_overlap_excludes_pairs_from_matrix_and_permutation():
    cor, count = correlation_matrix({'a': [1., 2., 3.], 'b': [1., 2., 3.]}, min_overlap=20)
    assert cor['a']['b'] is None
    assert count['a']['b'] == 3
    result = perm_test(['a', 'b'], cor, n_perms=10, seed=3)
    assert result['status'] == 'no_testable_pairs'
    assert result['p_value'] is None
    json.dumps(result, allow_nan=False)


def test_standard_sdi_duplication_is_shared_by_bxb():
    gene = '((A__a:1,B__b:1)X:1,A__c:2)R;'
    species = '(A:1,B:1)S;'
    assert reconcile_trees(gene, species).metrics['duplications'] == 1
    # The duplicated R must not contribute the extra root-to-tip distance.
    assert bxb_lengths_lca(gene, species, species_tree_edges(species)) == {'S_to_A': 1., 'S_to_B': 1.}


def test_bxb_thresholds_complete_distances_not_each_constituent_branch():
    from organelleverse.coevolution._bxb_extract import _dist_nodes
    tree = parse_newick('((A:.000006,B:1)X:.000006,C:1)R;')
    nodes = {n.name: n for n in tree.traverse()}
    distances = _dist_nodes(tree)
    assert distances[(id(nodes['R']), id(nodes['A']))] == pytest.approx(.000012)
    assert distances[(id(nodes['X']), id(nodes['A']))] == 0.


def test_projection_contains_every_node_pair_and_direct_edge():
    species = '((A:1,B:1)X:1,C:1)R;'
    gene = '((A:1,B:2)X:10,C:3)R;'
    # TreeTools PathLengths semantics: X->A=1 and R->A=11 are distinct.
    assert project_paths({'g': gene}, species)['g'] == pytest.approx([10., 1., 11., 2., 12., 3.])


def test_projection_matches_real_treetools_endpoint_reference():
    from organelleverse.coevolution._erc_paths import layout
    fixture = json.loads((Path(__file__).parent / 'fixtures' / 'treetools-paths.json').read_text())
    def labels(value):
        return tuple(sorted(value if isinstance(value, list) else [value]))
    for example in fixture['examples']:
        newick = example['newick']
        paths = layout(parse_newick(newick))
        row = project_paths({'g': newick}, newick)['g']
        actual = {(tuple(sorted(paths.clades[a])), tuple(sorted(paths.clades[b]))): value
                  for (a, b), value in zip(paths.paths, row, strict=True)}
        expected = {(labels(p['start']), labels(p['end'])): p['length'] for p in example['paths']}
        assert actual == pytest.approx(expected)


def test_common_species_correlation_uses_independent_pruned_edges():
    from organelleverse.coevolution._erc_paths import layout
    species = '((A:1,B:1)X:1,(C:1,D:1)Y:1)R;'
    trees = {'a': species, 'b': species}
    paths = layout(parse_newick(species))
    n = len(paths.paths)
    cor, count = correlation_matrix({'a': list(range(n)), 'b': list(reversed(range(n)))},
                                    min_overlap=2, gene_trees=trees, species_tree_newick=species)
    assert n == 10
    assert count['a']['b'] == 6  # 2n-2 independent rooted edges, not all 10 paths
    assert cor['a']['b'] == pytest.approx(-1)


def test_projected_missing_paths_remain_json_null():
    species = '((A:1,B:1)X:1,(C:1,D:1)Y:1)R;'
    result = project_paths({'g': '(A:2,(C:1,D:1)Y:1)R;'}, species)
    assert any(v is None for v in result['g'])
    json.dumps(result, allow_nan=False)


def test_projection_rejects_paralogs_and_discordant_topology():
    species = '((A:1,B:1)X:1,C:1)R;'
    with pytest.raises(ValueError, match='one copy'):
        project_paths({'g': '((A__a:1,A__b:1)X:1,C:1)R;'}, species)
    with pytest.raises(ValueError, match='topology/root'):
        project_paths({'g': '((A:1,C:1)X:1,B:1)R;'}, species)


def test_ercnet_statistics_are_direct_correlations_of_normalized_lengths():
    from organelleverse.coevolution.coevolution import _ercnet_pairs
    x = [1., 3., 2., 5., 4., 8., 7., 9.]
    y = [2., 2., 4., 5., 7., 8., 6., 10.]
    pair = _ercnet_pairs({'a': x, 'b': y})[0]
    assert pair['r'] == pytest.approx(stats.pearsonr(x, y).statistic)
    assert pair['pval'] == pytest.approx(stats.pearsonr(x, y).pvalue)
    assert pair['spearman_pval'] == pytest.approx(stats.spearmanr(x, y).pvalue)
    assert pair['kendall_pval'] == pytest.approx(stats.kendalltau(x, y).pvalue)


def test_lowess_matches_independent_r_stats_reference():
    from organelleverse.coevolution._erc_paths import lowess
    # Generated by server R stats::lowess(f=.7, iter=2), 2026-10-25.
    x = [.1, .2, .4, .6, 1., 1.5, 2., 3., 5.]
    y = [-2., -1.2, -1.4, .3, 0., 1.2, 1.1, 2.4, 4.]
    expected = [-1.7611950703305377, -1.5267542961116918, -1.0714372520056568,
                -.6414093244355494, .18357716960938264, .7813925277514243,
                1.3193967993528051, 2.3052979151456108, 4.0306959964916285]
    xx, fitted = lowess(x, y)
    assert xx == x
    assert fitted == pytest.approx(expected, abs=1e-10)


def test_lowess_matches_dense_r_default_interpolation_reference():
    from organelleverse.coevolution._erc_paths import lowess
    fixture = json.loads((Path(__file__).parent / 'fixtures' / 'lowess-r-dense.json').read_text())
    xx, fitted = lowess(fixture['x'], fixture['y'])
    assert xx == pytest.approx(fixture['x'])
    assert fitted == pytest.approx(fixture['fitted'], abs=1e-10)


def test_log_transform_uses_one_global_offset_against_independent_r_lm():
    from organelleverse.coevolution.erc_engine import residual_matrix
    fixture = json.loads((Path(__file__).parent / 'fixtures' / 'global-log-offset-r.json').read_text())
    assert fixture['offset'] == 1
    observed = residual_matrix(fixture['input'], transform='log')
    for gene, expected in fixture['expected'].items():
        assert observed[gene] == pytest.approx(expected, abs=1e-12)


def test_supported_erc2_core_matches_original_official_r_and_cpp():
    from organelleverse.coevolution._erc_paths import layout
    from organelleverse.coevolution.erc_engine import residual_matrix
    fixture = json.loads((Path(__file__).parent / 'fixtures' / 'official-erc2-reference.json').read_text())
    master, trees = fixture['master'], fixture['gene_trees']
    paths = layout(parse_newick(master))
    def labels(value):
        return tuple(sorted(value if isinstance(value, list) else [value]))
    indices = {(tuple(sorted(paths.clades[a])), tuple(sorted(paths.clades[b]))): i
               for i, (a, b) in enumerate(paths.paths)}
    remap = [indices[(labels(p['start']), labels(p['end']))] for p in fixture['endpoints']]
    projected = project_paths(trees, master)
    residuals = residual_matrix(projected, transform='sqrt', gene_trees=trees,
                                species_tree_newick=master, variance_weight=True)
    for gene in trees:
        assert [projected[gene][i] for i in remap] == pytest.approx(fixture['paths'][gene], abs=1e-12)
        # Independent Armadillo inverse vs NumPy least-squares floating arithmetic.
        assert [residuals[gene][i] for i in remap] == pytest.approx(fixture['residuals'][gene], abs=1e-8)
    result = run_erc(trees, master)
    assert result.status == 'ok'
    genes = list(trees)
    for pair in result.metrics['pairs']:
        a, b = pair['gene_a'], pair['gene_b']
        assert pair['r'] == pytest.approx(fixture['correlations'][a][genes.index(b)], abs=1e-9)
        assert pair['n_branches'] == fixture['counts'][a][genes.index(b)]
    assert result.metrics['pairs_tested'] == 15


def test_run_erc_adjusts_all_pairwise_tests(monkeypatch):
    from organelleverse.coevolution._erc_paths import layout
    species = '((A:1,B:1)X:1,(C:1,(D:1,E:1)Z:1)Y:1)R;'
    trees = {g: species for g in ('a', 'b', 'c', 'd')}
    paths = layout(parse_newick(species))
    n = len(paths.paths)
    residuals = {g: [math.sin(i * (j + 1) * .31) for i in range(n)] for j, g in enumerate(trees)}
    import organelleverse.coevolution.erc_engine as module
    monkeypatch.setattr(module, 'residual_matrix', lambda *args, **kwargs: residuals)
    result = run_erc(trees, species)
    assert result.status == 'ok'
    pairs = result.metrics['pairs']
    assert len(pairs) == 6
    expected = stats.false_discovery_control([p['pval'] for p in pairs], method='bh')
    assert [p['fdr_pval'] for p in pairs] == pytest.approx(expected)
    assert result.metrics['multiple_testing'] == 'Benjamini-Hochberg'


@pytest.mark.parametrize('values,transform', [([1., -1., 2.], 'sqrt'), ([1., 2., 3.], 'unknown')])
def test_residual_transform_rejects_invalid_scientific_inputs(values, transform):
    from organelleverse.coevolution.erc_engine import residual_matrix
    with pytest.raises(ValueError):
        residual_matrix({'a': values}, transform=transform)


def test_matrices_reject_ragged_path_layouts():
    from organelleverse.coevolution.erc_engine import residual_matrix
    with pytest.raises(ValueError, match='same path layout'):
        residual_matrix({'a': [1., 2., 3.], 'b': [1., 2.]})
    with pytest.raises(ValueError, match='same path layout'):
        correlation_matrix({'a': [1., 2., 3.], 'b': [1., 2.]})


def test_ercnet_rejects_ragged_branch_vectors():
    from organelleverse.coevolution.coevolution import _ercnet_pairs
    with pytest.raises(ValueError, match='same branch layout'):
        _ercnet_pairs({'a': [1., 2., 3.], 'b': [1., 2.]})
