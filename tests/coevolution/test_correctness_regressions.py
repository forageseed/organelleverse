"""Scientific-correctness regressions for the coevolution suite.

Each test pins a defect that used to pass silently:

* congeneric species (``Arabidopsis_thaliana`` / ``Arabidopsis_lyrata``) were
  collapsed to the genus when a leaf label was cut at its first ``_``;
* the network admitted non-significant pairs whenever ``|r| >= r_threshold``
  (an ``or`` where an ``and`` was meant), and its result carried only counts;
* ``TreeNode`` used structural dataclass equality although reconciliation asks
  "is this the same node".
"""

from __future__ import annotations

import math
import random
import re

import pytest

from organelleverse.coevolution import (
    coevolution_network,
    extract_branch_lengths,
    rate_covariation,
    reconcile_trees,
)
from organelleverse.coevolution._species import species_resolver, split_species_label
from organelleverse.coevolution.coevolution import (  # direct entry: caller-chosen output_dir
    TreeNode,
    _coerce_branch_length_vectors,
    _network_from_pairs,
    _parse_orthofinder_outputs,
    run_coevolution,
)
from organelleverse.coevolution.erc_engine import project_paths

SPECIES_TREE = "((Arabidopsis_thaliana:1,Arabidopsis_lyrata:1)N1:1,Oryza_sativa:1)N0;"
# underscore-only labels: ``<Genus_species>_<gene>`` (no explicit delimiter)
GENE_TREE = "((Arabidopsis_thaliana_g:0.3,Arabidopsis_lyrata_g:0.2):0.1,Oryza_sativa_g:0.5);"


# --------------------------------------------------------------------------
# Species identity
# --------------------------------------------------------------------------


def test_resolver_keeps_congeners_apart() -> None:
    known = ["Arabidopsis_thaliana", "Arabidopsis_lyrata", "Oryza_sativa"]
    resolve = species_resolver(known)
    assert resolve("Arabidopsis_thaliana_AT1G01010") == "Arabidopsis_thaliana"
    assert resolve("Arabidopsis_lyrata__AL1G1") == "Arabidopsis_lyrata"
    assert resolve("Oryza_sativa|LOC1") == "Oryza_sativa"
    assert resolve("Oryza_sativa") == "Oryza_sativa"


def test_resolver_longest_known_name_wins() -> None:
    resolve = species_resolver(["Ath", "Ath_b"])
    assert resolve("Ath_b_gene1") == "Ath_b"
    assert resolve("Ath_gene1") == "Ath"


def test_resolver_species_map_overrides_and_unknown_falls_back() -> None:
    resolve = species_resolver(["Ath"], {"weird-label": "Ath"})
    assert resolve("weird-label") == "Ath"
    assert resolve("Zma_gene1") == "Zma"  # species code, first "_"
    assert resolve("Zea_mays__g1") == "Zea_mays"  # explicit delimiter beats "_"


@pytest.mark.parametrize(
    ("label", "species"),
    [
        ("Arabidopsis_thaliana__AT1G01010", "Arabidopsis_thaliana"),
        ("Arabidopsis_thaliana|AT1G01010", "Arabidopsis_thaliana"),
        ("Arabidopsis_thaliana@AT1G01010", "Arabidopsis_thaliana"),
        ("Brsc_Bra001", "Brsc"),
        ("Brsc", "Brsc"),
        ("", ""),
    ],
)
def test_split_species_label_without_species_set(label: str, species: str) -> None:
    assert split_species_label(label) == species


def test_project_paths_gives_each_congener_its_own_edge_value() -> None:
    master = (
        "(((Arabidopsis_thaliana:0.1,Arabidopsis_lyrata:0.1):0.05,Oryza_sativa:0.3):0.1,"
        "Vitis_vinifera:0.4);"
    )
    gene = (
        "(((Arabidopsis_thaliana_g:0.2,Arabidopsis_lyrata_g:0.15):0.05,Oryza_sativa_g:0.35):0.1,"
        "Vitis_vinifera_g:0.45);"
    )
    row = project_paths({"g": gene}, master)["g"]
    # Every ancestor/descendant node pair; nearest ancestor first.
    assert row == pytest.approx([.10, .05, .15, .20, .25, .35, .15, .20, .30, .35, .45, .45])
    assert row[3] != pytest.approx(row[6])  # the two Arabidopsis leaves differ


def test_reconcile_trees_resolves_congeners_to_their_own_leaves() -> None:
    result = reconcile_trees(GENE_TREE, SPECIES_TREE)
    assert result.status == "ok"
    assert result.metrics["duplications"] == 0
    assert result.metrics["speciations"] == 2


def test_r2t_keys_are_full_species_names_when_species_tree_given() -> None:
    result = extract_branch_lengths({"g": GENE_TREE}, species_tree=SPECIES_TREE)
    row = dict(result.metrics["branch_lengths"]["g"])
    assert row == pytest.approx(
        {"Arabidopsis_thaliana": 0.4, "Arabidopsis_lyrata": 0.3, "Oryza_sativa": 0.5}
    )


def test_r2t_without_species_tree_uses_explicit_delimiter() -> None:
    gene = "(Arabidopsis_thaliana__g:0.3,Arabidopsis_lyrata__g:0.2,Oryza_sativa__g:0.5);"
    result = extract_branch_lengths({"g": gene})
    assert sorted(result.metrics["branch_lengths"]["g"]) == [
        "Arabidopsis_lyrata",
        "Arabidopsis_thaliana",
        "Oryza_sativa",
    ]


def test_bxb_lca_maps_congener_leaves_onto_their_species_edges() -> None:
    result = extract_branch_lengths(
        {"g": GENE_TREE},
        method="BXB",
        species_tree=SPECIES_TREE,
        bxb_normalization="none",
    )
    assert result.status == "ok"
    row = dict(result.metrics["branch_lengths"]["g"])
    assert row == pytest.approx(
        {
            "N0_to_N1": 0.1,
            "N1_to_Arabidopsis_thaliana": 0.3,
            "N1_to_Arabidopsis_lyrata": 0.2,
            "N0_to_Oryza_sativa": 0.5,
        }
    )


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

PAIRS = [
    {"gene_a": "a", "gene_b": "b", "r": 0.9, "significant": False},  # strong, not significant
    {"gene_a": "c", "gene_b": "d", "r": 0.1, "significant": True},  # significant, weak
    {"gene_a": "e", "gene_b": "f", "r": 0.8, "significant": True},
    {"gene_a": "f", "gene_b": "g", "r": 0.7, "significant": True},
    {"gene_a": "h", "gene_b": "i", "r": -0.7, "significant": True},  # strong anti-correlation
]


def test_network_with_fdr_needs_significance_and_effect_size() -> None:
    result = coevolution_network(PAIRS, r_threshold=0.5, use_fdr=True)
    edges = {(e["gene_a"], e["gene_b"]) for e in result.metrics["edge_list"]}
    assert edges == {("e", "f"), ("f", "g"), ("h", "i")}
    assert ("a", "b") not in edges  # not significant
    assert ("c", "d") not in edges  # below the effect-size threshold


def test_network_without_fdr_uses_only_the_threshold() -> None:
    result = coevolution_network(PAIRS, r_threshold=0.5, use_fdr=False)
    edges = {(e["gene_a"], e["gene_b"]) for e in result.metrics["edge_list"]}
    assert edges == {("a", "b"), ("e", "f"), ("f", "g"), ("h", "i")}


def test_network_result_carries_nodes_edges_and_communities() -> None:
    result = coevolution_network(PAIRS, r_threshold=0.5, use_fdr=True)
    metrics = result.metrics
    assert list(metrics["node_ids"]) == ["e", "f", "g", "h", "i"]
    assert [list(c) for c in metrics["community_members"]] == [["e", "f", "g"], ["h", "i"]]
    assert metrics["nodes"] == 5
    assert metrics["edges"] == 3
    assert metrics["communities"] == 2
    assert {"gene_a": "h", "gene_b": "i", "r": -0.7} in [dict(e) for e in metrics["edge_list"]]


def test_network_skips_pairs_without_a_finite_r() -> None:
    pairs = [
        {"gene_a": "a", "gene_b": "b", "r": None, "significant": True},
        {"gene_a": "c", "gene_b": "d", "significant": True},
        {"gene_a": "e", "gene_b": "f", "r": float("nan"), "significant": True},
        {"gene_a": "g", "gene_b": "h", "r": 0.9, "significant": True},
    ]
    nodes, edges, communities = _network_from_pairs(pairs, r_threshold=0.5, use_fdr=True)
    assert nodes == ["g", "h"]
    assert edges == [("g", "h", 0.9)]
    assert communities == [["g", "h"]]


def test_network_output_is_order_independent() -> None:
    forward = _network_from_pairs(PAIRS, r_threshold=0.5, use_fdr=True)
    backward = _network_from_pairs(list(reversed(PAIRS)), r_threshold=0.5, use_fdr=True)
    assert forward[0] == backward[0]
    assert forward[2] == backward[2]


# --------------------------------------------------------------------------
# Node identity
# --------------------------------------------------------------------------


def test_tree_nodes_compare_and_hash_by_identity() -> None:
    a, b = TreeNode(name="x", branch_length=1.0), TreeNode(name="x", branch_length=1.0)
    assert a != b
    assert a not in [b]
    assert len({a, b}) == 2
    assert a == a


# --------------------------------------------------------------------------
# End to end: OrthoFinder-style results -> ERCnet -> network
# --------------------------------------------------------------------------

_TEMPLATE = (
    "(((Arabidopsis_thaliana_{g}:{ath},Arabidopsis_lyrata_{g}:{aly}):{N3},"
    "(Brassica_rapa_{g}:{br},Brassica_oleracea_{g}:{bo}):{N4}):{N2},"
    "(Oryza_sativa_{g}:{os},Zea_mays_{g}:{zm}):{N5});"
)
_KEYS = ("N2", "N3", "N4", "N5", "ath", "aly", "br", "bo", "os", "zm")
_ORTHOFINDER_SPECIES_TREE = (
    "(((Arabidopsis_thaliana:0.1,Arabidopsis_lyrata:0.1)N3:0.2,"
    "(Brassica_rapa:0.05,Brassica_oleracea:0.05)N4:0.25)N2:0.3,"
    "(Oryza_sativa:0.2,Zea_mays:0.2)N5:0.4)N1;"
)


def _gene_tree(name: str, lengths: dict[str, float]) -> str:
    return _TEMPLATE.format(g=name, **{k: f"{v:.5f}" for k, v in lengths.items()})


def _write_orthofinder_results(root, *, seed: int = 7):
    rng = random.Random(seed)
    results = root / "Results_Oct13"
    (results / "Species_Tree").mkdir(parents=True)
    (results / "Resolved_Gene_Trees").mkdir()
    (results / "Species_Tree" / "SpeciesTree_rooted_node_labels.txt").write_text(
        _ORTHOFINDER_SPECIES_TREE
    )
    modules = {m: {k: rng.uniform(0.1, 1.0) for k in _KEYS} for m in ("A", "B")}
    lines = []
    for module, members in (("A", 3), ("B", 3)):
        for i in range(members):
            lengths = {k: v * rng.uniform(0.97, 1.03) for k, v in modules[module].items()}
            lines.append(f"OG_{module}{i}: {_gene_tree(f'{module}{i}', lengths)}")
    for i in range(4):
        lengths = {k: rng.uniform(0.1, 1.0) for k in _KEYS}
        lines.append(f"OG_C{i}: {_gene_tree(f'C{i}', lengths)}")
    (results / "Resolved_Gene_Trees" / "Resolved_Gene_Trees.txt").write_text("\n".join(lines) + "\n")
    return results


def test_run_coevolution_ercnet_network_matches_its_pairs(tmp_path) -> None:
    results = _write_orthofinder_results(tmp_path)
    out = tmp_path / "out"
    result = run_coevolution(tmp_path, output_dir=out, results_dir=results, r_threshold=0.5)
    assert result.status == "ok", result.summary_text
    assert result.metrics["n_species"] == 6
    assert result.metrics["duplications"] == 0

    pairs = {}
    for line in (out / "erc_pairs.tsv").read_text().splitlines()[1:]:
        a, b, r, _rho, _p, _fdr, sig = line.split("\t")
        pairs[(a, b)] = (float(r), sig == "True")
    network = [
        tuple(line.split("\t"))
        for line in (out / "coevolution_network.tsv").read_text().splitlines()[1:]
        if line
    ]
    assert network, "the planted co-evolving modules should give at least one edge"
    for a, b, r in network:
        pair_r, significant = pairs[(a, b)]
        assert significant, f"{a}-{b} entered the network without being significant"
        assert abs(float(r)) >= 0.5
        assert float(r) == pytest.approx(pair_r)
    # planted modules come back as network edges inside A and inside B
    edges = {frozenset(e[:2]) for e in network}
    assert any({"OG_A0", "OG_A1"} == set(e) or {"OG_A0", "OG_A2"} == set(e) for e in edges)
    assert any({"OG_B0", "OG_B1"} == set(e) or {"OG_B0", "OG_B2"} == set(e) for e in edges)


# --------------------------------------------------------------------------
# Chaining stages: a result's frozen metrics feed the next stage directly
# --------------------------------------------------------------------------


def _bxb_result(root, *, drop_zea_from: int = 0):
    """BXB result for the planted dataset; the first genes lose *Zea_mays*."""
    results = _write_orthofinder_results(root)
    species_tree, gene_trees = _parse_orthofinder_outputs(results)
    for name in list(gene_trees)[:drop_zea_from]:
        label = name.removeprefix("OG_")
        gene_trees[name] = re.sub(rf",Zea_mays_{label}:[0-9.]+", "", gene_trees[name])
    return extract_branch_lengths(
        gene_trees, method="BXB", species_tree=species_tree, bxb_normalization="edge_mean"
    )


def _pairs(result) -> list[dict]:
    return [dict(p) for p in result.metrics["pairs"]]


def test_coerce_reads_none_and_absent_branches_as_nan() -> None:
    vectors = _coerce_branch_length_vectors(
        {"g1": {"a": None, "b": 1.0}, "g2": {"b": 2.0, "c": 3.0}}
    )
    assert math.isnan(vectors["g1"][0])  # a: None
    assert vectors["g1"][1] == 1.0  # b
    assert math.isnan(vectors["g1"][2])  # c: absent from g1
    assert math.isnan(vectors["g2"][0])  # a: absent from g2
    assert vectors["g2"][1:] == [2.0, 3.0]


def test_coerce_reads_none_in_list_rows() -> None:
    row = _coerce_branch_length_vectors({"g": [None, 1.5]})["g"]
    assert math.isnan(row[0])
    assert row[1] == 1.5


def test_bxb_result_chains_into_rate_covariation_and_network(tmp_path) -> None:
    frozen = _bxb_result(tmp_path).metrics["branch_lengths"]
    row = next(iter(frozen.values()))
    with pytest.raises(TypeError):  # the frozen form, not a mutable dict
        row["x"] = 1.0

    chained = rate_covariation(frozen, min_overlap=5)
    plain = rate_covariation({g: dict(row) for g, row in frozen.items()}, min_overlap=5)
    assert chained.status == "ok", chained.summary_text
    assert _pairs(chained)
    assert _pairs(chained) == _pairs(plain)

    network = coevolution_network(chained.metrics["pairs"], r_threshold=0.5)
    assert network.status == "ok"
    assert network.metrics["edges"] == len(network.metrics["edge_list"])


def test_chain_survives_missing_branches_reported_as_none(tmp_path) -> None:
    frozen = _bxb_result(tmp_path, drop_zea_from=3).metrics["branch_lengths"]
    assert any(v is None for row in frozen.values() for v in row.values()), (
        "dropping a species must leave missing (None) branches in the frozen result"
    )
    chained = rate_covariation(frozen, min_overlap=5)
    nan_form = {
        g: {k: math.nan if v is None else v for k, v in row.items()} for g, row in frozen.items()
    }
    assert chained.status == "ok", chained.summary_text
    assert _pairs(chained) == _pairs(rate_covariation({g: dict(row) for g, row in frozen.items()}, min_overlap=5))
    # Native and JSON entry points share finite JSON: None is missing; NaN is rejected.
    from organelleverse.core.errors import OrganelleParameterError
    with pytest.raises(OrganelleParameterError):
        rate_covariation(nan_form, min_overlap=5)
