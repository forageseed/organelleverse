"""project_paths must normalize species names on BOTH sides of the match."""

from __future__ import annotations

import math

from organelleverse.coevolution.erc_engine import project_paths

MASTER = "((Arabidopsis_thaliana:0.1,Oryza_sativa:0.2):0.05,Vitis_vinifera:0.3);"
# Same topology in the gene tree; two-word species names must match the
# master's descendant sets (pre-fix the gene side truncated to genus while
# the master side kept full names, so every edge came back NaN).
GENE = "((Arabidopsis_thaliana:0.2,Oryza_sativa:0.1):0.08,Vitis_vinifera:0.4);"


def test_two_word_species_names_project() -> None:
    matrix = project_paths({"geneA": GENE}, MASTER)
    row = matrix["geneA"]
    assert len(row) == 4  # rooted 3-tip master has 2*3-2 edges
    matched = sum(1 for value in row if not math.isnan(value))
    assert matched >= 3


def test_species_gene_labels_project() -> None:
    gene = "((Arabidopsis_thaliana__geneA:0.2,Oryza_sativa__geneA:0.1):0.08,Vitis_vinifera|geneA:0.4);"
    matrix = project_paths({"geneA": gene}, MASTER)
    row = matrix["geneA"]
    assert sum(1 for value in row if not math.isnan(value)) >= 3
