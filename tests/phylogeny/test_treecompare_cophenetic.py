"""Branch-length tree comparison: patristic/shared-path matrices, cophenetic r, pruning."""

from __future__ import annotations

import numpy as np
import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.phylogeny.treecompare import (
    compare_trees,
    cophenetic_correlation,
    parse_rooted_tree,
    robinson_foulds,
)

TREE = "((A:1,B:2)90:1,(C:1,D:1):2);"


def test_patristic_and_shared_path_by_hand() -> None:
    tree = parse_rooted_tree(TREE)
    taxa = ["A", "B", "C", "D"]
    shared = tree.shared_path_matrix(taxa)
    assert np.allclose(np.diag(shared), [2, 3, 3, 3])
    assert shared[0, 1] == 1 and shared[2, 3] == 2 and shared[0, 2] == 0
    patristic = tree.patristic_matrix(taxa)
    assert patristic[0, 1] == 3  # 1 + 2
    assert patristic[0, 2] == 5  # 1 + 1 + 2 + 1
    assert patristic[2, 3] == 2


def test_pruning_reroots_at_the_retained_mrca() -> None:
    tree = parse_rooted_tree("((A:1,B:1):3,C:4);")
    shared = tree.shared_path_matrix(["A", "B"])
    assert np.allclose(shared, [[1, 0], [0, 1]])
    assert tree.patristic_matrix(["A", "B"])[0, 1] == 2


def test_cophenetic_is_scale_free_and_needs_lengths() -> None:
    scaled = "((A:10,B:20):10,(C:10,D:10):20);"
    assert cophenetic_correlation(TREE, scaled) == pytest.approx(1.0)
    assert cophenetic_correlation(TREE, "((A,B),(C,D));") is None
    other = "((A:1,C:1):1,(B:3,D:1):2);"
    assert cophenetic_correlation(TREE, other) < 0.9


def test_rf_prune_to_shared_taxa() -> None:
    bigger = "(((A:1,B:2):1,E:1):1,(C:1,D:1):2);"
    with pytest.raises(OrganelleInputError):
        robinson_foulds(TREE, bigger)
    stats = robinson_foulds(TREE, bigger, prune_to_shared=True)
    assert stats["rf"] == 0 and stats["rf_similarity"] == 1.0
    assert list(stats["pruned_from_b"]) == ["E"] and not stats["pruned_from_a"]


def test_compare_trees_reports_both_measures(tmp_path) -> None:
    a = tmp_path / "mito.nwk"
    b = tmp_path / "nuclear.nwk"
    a.write_text(TREE)
    b.write_text("(((A:1,C:1):1,E:2):1,(B:3,D:1):2);")
    result = compare_trees(a, b, prune_to_shared=True)
    assert result.status == "ok"
    metrics = result.metrics
    assert metrics["n_taxa"] == 4 and metrics["rf"] == 2
    assert metrics["rf_similarity"] == pytest.approx(0.0)
    assert -1.0 <= metrics["cophenetic_correlation"] <= 1.0
    assert "pruned_to_shared_taxa" in result.flags
