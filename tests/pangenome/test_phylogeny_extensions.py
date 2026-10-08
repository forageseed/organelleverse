import numpy as np
import pytest

from organelleverse.pangenome.phylogeny import jaccard_distances
from organelleverse.pangenome.sequence_phylogeny import RaxmlOptions, alignment_records
from organelleverse.pangenome.streaming_phylogeny import (
    _convergence,
    _distance,
    batch_node_pav_tree,
)


def test_streamed_distance_matches_dense_without_materializing_all_batches():
    values = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=bool)

    def source():
        return iter([values[:1], values[1:3], values[3:]])

    assert np.allclose(_distance(source, 4, 3), jaccard_distances(["a", "b", "c"], values))
    tree = batch_node_pav_tree(["a", "b", "c"], 4, source, bootstrap_replicates=15, seed=17)
    assert tree == batch_node_pav_tree(["a", "b", "c"], 4, source, bootstrap_replicates=15, seed=17)


def test_weighted_bootstrap_is_exact_explicit_resampling_distance():
    values = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=bool)

    class Draw:
        def multinomial(self, n, p):
            assert n == 3
            return np.array([2, 0, 1])

    weighted = _distance(lambda: iter([values]), 3, 3, Draw())
    assert np.allclose(weighted, jaccard_distances(["a", "b", "c"], values[[0, 0, 2]]))


def test_adaptive_bootstrap_strict_maximum_and_undefined_correlation():
    values = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]], dtype=bool)
    tree = batch_node_pav_tree(
        ["a", "b", "c", "d"],
        3,
        lambda: iter([values]),
        bootstrap_method="adaptive_pav",
        min_replicates=7,
        max_replicates=13,
        batch_size=5,
        seed=4,
        consecutive_converged=10,
    )
    assert tree["bootstrap_replicates"] == 13
    assert [h["replicates"] for h in tree["convergence_history"]] == [5, 10, 13]
    assert _convergence(np.array([0.1, 0.1]), np.array([0.2, 0.2])) == (None, "pearson_undefined")


def test_maf_retains_reverse_coordinates_and_missing_rows(tmp_path):
    source = tmp_path / "input.maf"
    source.write_text(
        "##maf version=1\n\na score=1\ns a.chr1 2 3 + 10 AC-G\ns b.chr2 1 4 - 12 ACTG\n\na score=2\ns a.chr1 5 2 + 10 TT\n\n"
    )
    records, coords = alignment_records(
        source, format="maf", source_mapping={"a.chr1": "a", "b.chr2": "b"}
    )
    assert records == [("a", "AC-GTT"), ("b", "ACTG--")]
    assert (coords[1]["source_start"], coords[1]["source_end"], coords[1]["strand"]) == (7, 11, "-")
    with pytest.raises(ValueError, match="Multiple MAF"):
        alignment_records(source, format="maf", source_mapping={"a.chr1": "x", "b.chr2": "x"})


def test_alignment_validation_and_model_boundary(tmp_path):
    source = tmp_path / "bad.fa"
    source.write_text(">a\nACGT\n>b\nACG\n")
    with pytest.raises(ValueError, match="equal"):
        alignment_records(source)
    with pytest.raises(ValueError, match="model expression"):
        RaxmlOptions(model="../partitions.txt")


def test_msa_tree_plot_retains_branch_lengths_and_never_invents_jaccard_table(tmp_path):
    from organelleverse.pangenome.visualization import render_tree

    newick = "((a:0.1,b:0.2)95:0.3,(c:0.4,d:0.5)90:0.1);"
    result = render_tree(
        {"mode": "msa", "model": "GTR+G", "paths": ["a", "b", "c", "d"], "newick": newick}, tmp_path
    )
    assert (tmp_path / "msa_tree.nwk").read_text().strip() == newick
    assert not (tmp_path / "jaccard_distances.tsv").exists()
    svg = (tmp_path / "msa_tree.svg").read_text()
    assert "Substitutions per site" in svg and "Jaccard" not in svg
    assert "a;b\t0.3\t95" in (tmp_path / "msa_tree_branches.tsv").read_text()
    assert "branch_table" in result


def test_stored_pav_tree_matches_original_graph_distances(tmp_path):
    from organelleverse.pangenome.graph import node_pav
    from organelleverse.pangenome.pav_store import write_node_pav
    from organelleverse.pangenome.streaming_phylogeny import stored_node_pav_tree

    graph = tmp_path / "graph.gfa"
    graph.write_text(
        "S\t1\tAC\nS\t2\tGG\nS\t3\tTT\nL\t1\t+\t2\t+\t0M\nL\t1\t+\t3\t+\t0M\nP\ta#1#1\t1+,2+\t*\nP\tb#1#1\t1+,3+\t*\nP\tc#1#1\t3+\t*\n"
    )
    write_node_pav(graph, tmp_path / "pav", row_group_size=1)
    result = stored_node_pav_tree(tmp_path / "pav", bootstrap_replicates=5, seed=23)
    dense = node_pav(graph)
    assert np.allclose(
        result["distances"], jaccard_distances(dense["samples"], dense["sample_matrix"])
    )
    assert result["bootstrap_replicates"] == 5


def test_adaptive_stop_records_exact_identical_support_convergence():
    values = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=bool)
    result = batch_node_pav_tree(
        ["a", "b", "c", "d"],
        4,
        lambda: iter([values]),
        bootstrap_method="adaptive_pav",
        min_replicates=2,
        max_replicates=50,
        batch_size=2,
        seed=19,
    )
    assert result["converged"]
    assert result["bootstrap_replicates"] < 50
    assert result["convergence_history"][-1]["criterion"] == "identical_support_vector"
