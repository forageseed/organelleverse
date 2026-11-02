"""Read linkage, conditional graph order and physical identifiability are distinct."""

import pytest

from organelleverse.assembly.configuration import infer_configurations


def graph(tmp_path, names, edges):
    path = tmp_path / "graph.gfa"
    path.write_text("".join(f"S\t{x}\t{'ACGT' * 20}\n" for x in names)
                    + "".join(f"L\t{a[:-1]}\t{a[-1]}\t{b[:-1]}\t{b[-1]}\t0M\n"
                              for a, b in edges))
    return path


def evidence(edges, **extra):
    return {"links": [{"from": a, "to": b, "reads": 10} for a, b in edges],
            "repeat_segments": [], "traversals": [], "phased_paths": [],
            "observed_walks": [], **extra}


def test_supported_independent_cycles_do_not_certify_physical_shape(tmp_path):
    edges = [(f"C{i}+", f"C{i}+") for i in range(8)]
    result = infer_configurations(graph(tmp_path, [f"C{i}" for i in range(8)], edges), evidence(edges))
    assert len(result["components"]) == 8
    assert result["graph_order_identified"]
    assert not result["physical_topology_identified"]
    assert result["physical_molecule_count"] is None


def test_cycle_with_unobserved_closure_has_unresolved_order(tmp_path):
    edges = [("A+", "B+"), ("B+", "A+")]
    ev = evidence(edges)
    ev["links"][1]["reads"] = 0
    result = infer_configurations(graph(tmp_path, ["A", "B"], edges), ev)
    component = result["components"][0]
    assert component["topology"] == "simple_cycle"
    assert not component["graph_order_identified"]
    assert component["unobserved_junctions"] == ["B+ A+"]


@pytest.mark.parametrize("reverse_links", [False, True])
def test_supported_open_chain_identifies_order_without_certifying_ends(tmp_path, reverse_links):
    edges = [("A+", "B-"), ("B-", "C+")]
    if reverse_links:
        edges = [("C-", "B+"), ("B+", "A-")]
    result = infer_configurations(graph(tmp_path, ["A", "B", "C"], edges), evidence(edges))
    component = result["components"][0]
    assert component["topology"] == "open_chain"
    assert component["linear_order"] == "A+ B- C+"
    assert component["cyclic_order"] is None
    assert result["graph_order_identified"]
    assert not result["physical_topology_identified"]
    assert result["physical_molecule_count"] is None


def test_open_chain_with_unobserved_junction_keeps_order_unresolved(tmp_path):
    edges = [("A+", "B+"), ("B+", "C+")]
    ev = evidence(edges)
    ev["links"][1]["reads"] = 0
    result = infer_configurations(graph(tmp_path, ["A", "B", "C"], edges), ev)
    assert result["components"][0]["linear_order"] == "A+ B+ C+"
    assert not result["graph_order_identified"]


def test_isolated_unitig_has_known_retained_order_but_unknown_physical_ends(tmp_path):
    result = infer_configurations(graph(tmp_path, ["A"], []), evidence([]))
    assert result["components"][0]["linear_order"] == "A+"
    assert result["graph_order_identified"]
    assert not result["physical_topology_identified"]


def test_reverse_self_link_is_not_a_single_visit_linear_order(tmp_path):
    edges = [("A+", "A-")]
    result = infer_configurations(graph(tmp_path, ["A"], edges), evidence(edges))
    assert result["components"][0]["linear_order"] is None
    assert not result["graph_order_identified"]


def test_mirrored_links_are_one_physical_adjacency(tmp_path):
    edges = [("A+", "B-"), ("B+", "A-"), ("B-", "A+")]
    result = infer_configurations(graph(tmp_path, ["A", "B"], edges), evidence([edges[0], edges[2]]))
    assert result["graph_order_identified"]
    assert result["components"][0]["cyclic_order"] == "A+ B-"


def test_more_than_two_entries_preserves_all_supported_contexts(tmp_path):
    edges = [(x + "+", "R+") for x in "ABC"] + [("R+", x + "+") for x in "XY"]
    ev = evidence(edges, repeat_segments=["R"], traversals=[
        {"path": "A+ R+ X+", "reads": 40}, {"path": "A+ R+ Y+", "reads": 25},
        {"path": "Y- R- B-", "reads": 18}, {"path": "C+ R+ X+", "reads": 3},
    ])
    result = infer_configurations(graph(tmp_path, list("ABCRXY"), edges), ev)
    repeat = result["repeat_contexts"][0]
    assert repeat["status"] == "alternative_contexts_observed"
    assert len(repeat["contexts"]) == 6
    counts = {x["path"]: x["reads"] for x in repeat["contexts"]}
    assert counts["B+ R+ Y+"] == 18
    assert counts["C+ R+ Y+"] == 0
    assert not result["graph_order_identified"]


def test_joint_choices_are_retained_without_cartesian_global_isoforms(tmp_path):
    edges = [("A+", "R+"), ("R+", "B+"), ("R+", "C+"),
             ("B+", "S+"), ("C+", "S+"), ("S+", "D+"), ("S+", "E+")]
    ev = evidence(edges, repeat_segments=["R", "S"], observed_walks=[
        {"path": "A+ R+ B+ S+ D+", "reads": 30},
        {"path": "E- S- C- R- A-", "reads": 20},
    ])
    result = infer_configurations(graph(tmp_path, list("ARBCSDE"), edges), ev)
    assert {x["path"] for x in result["observed_walks"]} == {
        "A+ R+ B+ S+ D+", "A+ R+ C+ S+ E+"
    }
    assert "global_isoforms" not in result
    assert result["physical_molecule_count"] is None


def test_long_repeat_without_spanning_reads_remains_unspanned(tmp_path):
    edges = [("A+", "R+"), ("B+", "R+"), ("R+", "C+"), ("R+", "D+")]
    ev = evidence(edges, repeat_segments=["R"])
    result = infer_configurations(graph(tmp_path, list("ABRCD"), edges), ev)
    assert result["repeat_contexts"][0]["status"] == "unspanned"
    assert result["components"][0]["unspanned_repeats"] == ["R"]
    assert not result["graph_order_identified"]


@pytest.mark.parametrize("spanned", [False, True])
def test_separate_reads_across_adjacent_repeats_do_not_phase_the_chain(tmp_path, spanned):
    edges = [("A+", "R+"), ("R+", "S+"), ("S+", "B+")]
    ev = evidence(edges, repeat_segments=["R", "S"], observed_walks=[
        {"path": "A+ R+ S+", "reads": 10}, {"path": "R+ S+ B+", "reads": 10},
    ], traversals=[{"path": "A+ R+ S+", "reads": 10}, {"path": "R+ S+ B+", "reads": 10}])
    if spanned:
        ev["phased_paths"] = [{"path": "A+ R+ S+ B+", "reads": 3}]
        ev["observed_walks"].append({"path": "A+ R+ S+ B+", "reads": 3})
    result = infer_configurations(graph(tmp_path, ["A", "R", "S", "B"], edges), ev)
    linkage = result["repeat_linkage"][0]
    assert linkage["combined_repeat_length"] == 160
    assert linkage["status"] == ("anchor_linkage_observed" if spanned else "anchor_linkage_unresolved")
    assert bool(linkage["joint_paths"]) == spanned
    assert bool(result["components"][0]["unresolved_repeat_links"]) != spanned


def test_cyclic_read_order_is_not_a_circular_molecule_certificate(tmp_path):
    edges = [("A+", "R+"), ("R+", "A+")]
    ev = evidence(edges, observed_walks=[{"path": "A- R- A-", "reads": 4}])
    result = infer_configurations(graph(tmp_path, ["A", "R"], edges), ev)
    assert {x["cyclic_order"] for x in result["observed_cyclic_orders"]} == {"A+ R+"}
    assert all(x["reads"] == 4 and x["supporting_walk"] == "A+ R+ A+"
               for x in result["observed_cyclic_orders"])
    assert not result["physical_topology_identified"]


def test_internal_return_is_preserved_with_its_linkage_to_other_sequence(tmp_path):
    edges = [("A+", "R+"), ("R+", "A+"), ("A+", "B+")]
    ev = evidence(edges, observed_walks=[{"path": "A+ R+ A+ B+", "reads": 7}])
    result = infer_configurations(graph(tmp_path, ["A", "R", "B"], edges), ev)
    orders = result["observed_cyclic_orders"]
    assert len(orders) == 1
    assert orders[0] == {"cyclic_order": "A+ R+", "reads": 7, "supporting_walk": "A+ R+ A+ B+"}


def test_old_evidence_does_not_invent_joint_read_linkage(tmp_path):
    edges = [("A+", "A+")]
    ev = evidence(edges)
    del ev["observed_walks"]
    result = infer_configurations(graph(tmp_path, ["A"], edges), ev)
    assert not result["joint_read_linkage_available"]
    assert result["observed_walks"] == []
    assert result["read_length_summary"]["maximum_read_length"] is None


def test_long_reads_without_locus_support_do_not_resolve_repeat_linkage(tmp_path):
    edges = [("A+", "R+"), ("R+", "S+"), ("S+", "B+")]
    ev = evidence(edges, repeat_segments=["R", "S"], maximum_read_length=100000,
                  params={"min_anchor": 500})
    result = infer_configurations(graph(tmp_path, ["A", "R", "S", "B"], edges), ev)
    assert result["read_length_summary"]["maximum_read_length"] == 100000
    assert result["read_length_summary"]["junction_min_anchor"] == 500
    assert result["repeat_linkage"][0]["status"] == "anchor_linkage_unresolved"
    assert not result["physical_topology_identified"]


@pytest.mark.parametrize("failure", ["adjacency", "path", "count"])
def test_mismatched_evidence_is_rejected(tmp_path, failure):
    edges = [("A+", "B+")]
    ev = evidence(edges)
    if failure == "adjacency":
        ev["links"][0]["to"] = "A+"
    elif failure == "path":
        ev["observed_walks"] = [{"path": "A+ B+ A+", "reads": 1}]
    else:
        ev["links"][0]["reads"] = -1
    with pytest.raises(ValueError):
        infer_configurations(graph(tmp_path, ["A", "B"], edges), ev)
