"""Annotated trees and nuclear-versus-organelle tree conflict."""

from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.result import OrganelleResult
from organelleverse.visualization import compose_figure, plot_tree
from organelleverse.visualization.tree_plots import plot_phylo_tree, plot_tree_conflict

# A: ((a,b),(c,d)),(e,f); B swaps a and c. g and h are in one tree only and are pruned.
TREE_A = "(((a:1,b:1)95:1,(c:1,d:1)90:1)99:1,(e:1,f:1)80:1,g:1);"
TREE_B = "(((c:1,b:1)98:1,(a:1,d:1)85:1)97:1,e:1,f:1,h:1);"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text + "\n")
    return path


def _conflict(tmp_path, **kwargs):
    return plot_tree_conflict(
        {"Nuclear": _write(tmp_path, "a.nwk", TREE_A), "Plastid": _write(tmp_path, "b.nwk", TREE_B)},
        **kwargs,
    )


def test_conflict_classes_follow_bipartition_compatibility(tmp_path):
    plot = _conflict(tmp_path)
    m = plot.metrics
    assert m["taxa"] == ("a", "b", "c", "d", "e", "f")
    assert dict(m["dropped"]) == {"Nuclear": ("g",), "Plastid": ("h",)}
    pair = m["pairs"][0]
    nuclear = {tuple(c["taxa"]): c for c in pair["conflicts"]["Nuclear"]}
    # {a,b} and {c,d} contradict {c,b} and {a,d}: supported on both sides
    assert set(nuclear) == {("a", "b"), ("c", "d")}
    assert all(c["strongly_supported"] for c in nuclear.values())
    plastid = {tuple(c["taxa"]) for c in pair["conflicts"]["Plastid"]}
    assert plastid == {("b", "c"), ("a", "d")}
    # unrooted splits: A {a,b},{c,d},{e,f}; B {b,c},{a,d},{e,f} ({a,b,c,d} is {e,f})
    assert pair["shared_clades"] == 1
    assert pair["robinson_foulds"] == 4 and pair["robinson_foulds_max"] == 6
    assert "taxa_pruned_to_shared" in plot.flags and "supported_conflict" in plot.flags


def test_clade_missing_from_a_polytomy_is_unresolved_not_conflict(tmp_path):
    a = _write(tmp_path, "x.nwk", "((a,b)100,(c,d)100,(e,f)100);")
    b = _write(tmp_path, "y.nwk", "((a,b)100,c,d,(e,f)100);")
    pair = plot_tree_conflict([a, b]).metrics["pairs"][0]
    assert sorted(pair["left_classes"].values()) == ["shared", "shared", "unresolved"]
    assert not pair["conflicts"]["Tree 1"] and pair["robinson_foulds"] == 1


def test_support_threshold_decides_supported_conflict(tmp_path):
    pair = _conflict(tmp_path, support_threshold=96).metrics["pairs"][0]
    supported = {tuple(c["taxa"]) for c in pair["conflicts"]["Nuclear"] if c["strongly_supported"]}
    assert supported == set()  # 95 and 90 in A fall below 96
    strong_b = {tuple(c["taxa"]) for c in pair["conflicts"]["Plastid"] if c["strongly_supported"]}
    assert strong_b == set()  # B's 98 is opposed by A's 95 at most


def test_identical_topologies_untangle_without_crossings(tmp_path):
    a = _write(tmp_path, "x.nwk", "((a,b),((c,d),(e,f)),g);")
    b = _write(tmp_path, "y.nwk", "(g,((f,e),(d,c)),(b,a));")
    pair = plot_tree_conflict([a, b], names=["Nuclear", "Mitochondrion"]).metrics["pairs"][0]
    assert pair["crossings"] == 0
    assert pair["robinson_foulds"] == 0
    assert not pair["conflicts"]["Nuclear"] and not pair["conflicts"]["Mitochondrion"]


def test_rename_matches_differently_labelled_trees(tmp_path):
    a = _write(tmp_path, "x.nwk", "((NC_1,NC_2),(NC_3,NC_4),NC_5);")
    b = _write(tmp_path, "y.nwk", "((Alpha_one,Beta_two),(Gamma_three,Delta_four),Eps_five);")
    rename = {"NC_1": "Alpha one", "NC_2": "Beta two", "NC_3": "Gamma three",
              "NC_4": "Delta four", "NC_5": "Eps five"}
    m = plot_tree_conflict([a, b], rename=rename).metrics
    assert m["n_taxa"] == 5 and not any(m["dropped"].values())


def test_too_few_shared_taxa_is_an_input_error(tmp_path):
    a = _write(tmp_path, "x.nwk", "((a,b),(c,d));")
    b = _write(tmp_path, "y.nwk", "((a,b),(x,y));")
    with pytest.raises(OrganelleInputError):
        plot_tree_conflict([a, b])


def test_three_trees_give_two_pairs_and_render_from_serialized_metrics(tmp_path):
    a = _write(tmp_path, "n.nwk", "((a,b)100,((c,d)100,(e,f)100)100,g);")
    b = _write(tmp_path, "m.nwk", "((a,c)100,((b,d)100,(e,f)100)100,g);")
    c = _write(tmp_path, "p.nwk", "((a,b)100,((c,d)100,(e,f)100)100,g);")
    plot = plot_tree_conflict({"Nuclear": a, "Mitochondrion": b, "Plastid": c})
    pairs = plot.metrics["pairs"]
    assert [p["right"] for p in pairs] == [1, 2]
    assert pairs[1]["robinson_foulds"] == 0
    restored = OrganelleResult.model_validate_json(plot.model_dump_json())
    for suffix in ("png", "svg"):
        written = ov.write(restored, tmp_path / f"conflict.{suffix}")
        assert written.artifacts and (tmp_path / f"conflict.{suffix}").stat().st_size > 0


def test_annotations_are_matched_typed_and_reported(tmp_path):
    tree = _write(tmp_path, "t.nwk", "((Alpha_one:1,Beta_two:1)99:1,(Gamma_three:1,Delta_four:1)60:1,Eps:2);")
    plot = plot_phylo_tree(
        tree,
        tip_groups={"Alpha one": "G1", "Beta_two": "G1", "Gamma three": "G2", "Nope": "G3"},
        clades={"Pair": ["Alpha one", "Beta two"], "Wide": ["Alpha one", "Gamma three", "Eps"],
                "Odd": ["Alpha one", "Gamma three", "Delta four"]},
        tip_data={"Size": {"Alpha one": 10, "Eps": 30}, "gene": {"Alpha one": "yes", "Eps": "no"},
                  "Tech": {"Beta two": "HiFi", "Eps": "ONT"}},
    )
    m = plot.metrics
    assert m["support_scale"] == "bootstrap" and tuple(m["support_thresholds"]) == (70.0, 95.0)
    assert m["unmatched"]["tip_groups"] == ("Nope",)
    clades = {c["label"]: c for c in m["clades"]}
    assert clades["Pair"]["definition"] == "mrca" and clades["Pair"]["monophyletic"]
    assert clades["Odd"]["monophyletic"] is False  # three tips = full membership
    assert clades["Wide"]["level"] > clades["Pair"]["level"]  # nested brackets get their own column
    kinds = {s["name"]: s["kind"] for s in m["strips"]}
    assert kinds == {"Size": "numeric", "gene": "binary", "Tech": "categorical"}
    assert "clade_not_monophyletic" in plot.flags and "annotation_names_unmatched" in plot.flags
    assert "unrooted_tree_drawn_as_given" in plot.flags


@pytest.mark.parametrize("layout", ["rectangular", "circular"])
def test_layouts_render_at_journal_width(tmp_path, layout):
    tree = _write(tmp_path, "t.nwk", "((a:1,b:2)90:1,((c:1,d:1)100:1,(e:3,f:1)50:1)80:1,g:2);")
    plot = plot_phylo_tree(tree, layout=layout, outgroup=["g"], tip_data={"x": {"a": 1, "c": 2}})
    assert plot.metrics["width_mm"] in (89.0, 120.0, 183.0)
    assert plot.metrics["rooting"]["method"] == "outgroup"
    written = ov.write(plot, tmp_path / f"{layout}.png")
    assert written.artifacts


def test_outgroup_rooting_and_midpoint(tmp_path):
    tree = _write(tmp_path, "t.nwk", "((a:1,b:1):1,(c:1,d:5):1,e:1);")
    assert "outgroup_not_applied" in plot_phylo_tree(tree, outgroup=["zz"]).flags
    assert plot_phylo_tree(tree, midpoint_root=True).metrics["rooting"]["method"] == "midpoint"


def test_tree_plots_compose_into_one_figure(tmp_path):
    a = _write(tmp_path, "n.nwk", "((a,b)100,((c,d)100,(e,f)100)100,g);")
    b = _write(tmp_path, "m.nwk", "((a,c)100,((b,d)100,(e,f)100)100,g);")
    figure = compose_figure([plot_phylo_tree(a), plot_tree_conflict([a, b])], width="double", columns=1)
    assert ov.write(figure, tmp_path / "composed.pdf").artifacts


def test_plot_tree_circular_layout_is_drawn(tmp_path):
    tree = _write(tmp_path, "t.nwk", "((a:1,b:1)90:1,(c:1,d:1)80:1,e:1);")
    rect = ov.write(plot_tree(tree, method="biophylo"), tmp_path / "rect.png")
    circ = ov.write(plot_tree(tree, method="biophylo", layout="circular"), tmp_path / "circ.png")
    assert rect.artifacts and circ.artifacts
    assert (tmp_path / "rect.png").read_bytes() != (tmp_path / "circ.png").read_bytes()
    with pytest.raises(ValueError):
        plot_tree(tree, layout="radial")


def test_auto_method_is_the_same_renderer_everywhere(tmp_path):
    tree = _write(tmp_path, "t.nwk", "((a:1,b:1)90:1,(c:1,d:1)80:1,e:1);")
    assert plot_tree(tree).metrics["method"] == "biophylo"
    with pytest.raises(ValueError):
        plot_tree(tree, method="ete3")


@pytest.mark.parametrize("layout", ["rectangular", "circular"])
def test_toytree_writes_the_format_its_suffix_names(tmp_path, layout):
    pytest.importorskip("toytree")
    tree = _write(tmp_path, "t.nwk", "((Alpha_one:1,Beta_two:1)90:1,(Gamma:1,Delta:1)0.8:1,Eps:1);")
    plot = plot_tree(tree, method="toytree", layout=layout)
    magic = {"png": b"\x89PNG", "pdf": b"%PDF", "svg": b"<"}
    for suffix, head in magic.items():
        out = tmp_path / f"tree_{layout}.{suffix}"
        assert ov.write(plot, out).artifacts
        assert out.read_bytes().lstrip()[: len(head)] == head
    svg = (tmp_path / f"tree_{layout}.svg").read_text(encoding="utf-8")
    assert "Alpha one" in svg and "Alpha_one" not in svg
    with pytest.raises(ValueError):
        ov.write(plot, tmp_path / "tree.jpg")
