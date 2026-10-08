from pathlib import Path

import pytest

from organelleverse.pangenome.graph import (
    extract_subgraph,
    graph_statistics,
    inspect_topology,
    load_gfa,
    node_pav,
    path_sequences,
    path_steps,
)


def write_graph(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "input.gfa"
    path.write_text(text.replace(" ", "\t"))
    return path


@pytest.fixture
def diamond(tmp_path):
    return write_graph(
        tmp_path,
        """H VN:Z:1.1
S a AA
S b C
S c G
S d TT
S z NNN
L a + b + 0M
L a + c + 0M
L b + d + 0M
L c + d + 0M
P alpha#0#mito a+,b+,d+ *
P beta#0#mito a+,c+,d+ *
W alpha 0 plasmid 10 13 >a>b
""",
    )


def test_pav_distinguishes_paths_samples_and_unobserved(diamond):
    result = node_pav(diamond, cloud_threshold=0.5)
    assert result["nodes"] == ["a", "b", "c", "d", "z"]
    assert result["matrix"] == [[1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0]]
    assert result["classes"] == ["core", "shell", "cloud", "shell", "unobserved"]
    assert result["samples"] == ["alpha", "beta"]
    assert result["sample_matrix"] == [[1, 1], [1, 0], [0, 1], [1, 1], [0, 0]]
    assert result["sample_classes"] == ["core", "cloud", "cloud", "core", "unobserved"]


def test_statistics_and_walk_coordinates(diamond):
    result = graph_statistics(diamond)
    assert (result["nodes"], result["edges"], result["paths"], result["components"]) == (5, 4, 3, 2)
    assert result["total_bp"] == 9
    assert result["n50"] == 2
    assert result["node_lengths"] == {"a": 2, "b": 1, "c": 1, "d": 2, "z": 3}
    assert result["path_lengths"]["alpha#0#plasmid:10-13"] == 3
    assert path_steps(diamond)["alpha#0#plasmid:10-13"][0]["start"] == 10
    assert path_sequences(diamond)["beta#0#mito"] == "AAGTT"


def test_reverse_overlap_and_repeated_nodes(tmp_path):
    graph = write_graph(
        tmp_path,
        """S x ACCTT
S y TCAAGG
S z CTTGATT
L x + y - 4M
L y - z + 5M
P forward x+,y-,z+ 4M,5M
P reverse z-,y+,x- *
""",
    )
    assert path_sequences(graph) == {"forward": "ACCTTGATT", "reverse": "AATCAAGGT"}
    assert [(s["start"], s["end"]) for s in path_steps(graph)["forward"]] == [
        (0, 5),
        (1, 7),
        (2, 9),
    ]


def test_subgraph_retains_only_complete_selected_paths(diamond, tmp_path):
    text = extract_subgraph(diamond, ["alpha#0#mito"])
    out = tmp_path / "sub.gfa"
    out.write_text(text)
    graph = load_gfa(out)
    assert list(graph.segments) == ["a", "b", "d"]
    assert list(graph.paths) == ["alpha#0#mito"]
    assert path_sequences(out) == {"alpha#0#mito": "AACTT"}
    assert inspect_topology(diamond)["bubbles"][0]["branches"] == ["b+", "c+"]


@pytest.mark.parametrize(
    "body,match",
    [
        ("S a A\nS a C\n", "duplicate"),
        ("S a A\nL a + absent + 0M\n", "unknown segment"),
        ("S a A\nS b C\nP p a+,b+ *\n", "link"),
        ("S a A\nS b C\nL a + b + 1M\nW s 0 m 0 2 >a>b\n", "zero-overlap"),
        ("S a A\nW s 0 m 0 2 >a\n", "length"),
        ("S a A\nP a a+ *\n", "namespace"),
        ("S a A\nS b C\nL a ? b + 0M\n", "orientation"),
    ],
)
def test_invalid_graphs_fail_at_boundary(tmp_path, body, match):
    with pytest.raises(ValueError, match=match):
        load_gfa(write_graph(tmp_path, body))


def test_unknown_and_nonexact_overlap_are_not_fabricated(tmp_path):
    graph = write_graph(tmp_path, "S a AC\nS b CT\nL a + b + *\nP p a+,b+ *\n")
    assert graph_statistics(graph)["path_lengths"]["p"] is None
    with pytest.raises(ValueError, match="unknown overlap"):
        path_sequences(graph)
    graph = write_graph(tmp_path, "S a AC\nS b GT\nL a + b + 1M\nP p a+,b+ *\n")
    with pytest.raises(ValueError, match="disagree"):
        path_sequences(graph)


def test_unknown_sequence_length_not_zero(tmp_path):
    graph = write_graph(tmp_path, "S a *\nP p a+ *\n")
    assert graph_statistics(graph)["total_bp"] is None
    assert graph_statistics(graph)["n50"] is None
    assert node_pav(graph)["matrix"] == [[1]]
    with pytest.raises(ValueError, match="sequence"):
        path_sequences(graph)


def test_repeated_node_visit_counts_binary_presence_once(tmp_path):
    graph = write_graph(tmp_path, "S a AC\nL a + a + 0M\nP p a+,a+,a+ *\n")
    assert node_pav(graph)["matrix"] == [[1]]
    assert graph_statistics(graph)["node_traversal_counts"] == {"a": 3}
    assert graph_statistics(graph)["degree"] == {"a": 2}
    assert path_sequences(graph) == {"p": "ACACAC"}


def test_unconnected_zero_overlap_and_missing_sequence_length(tmp_path):
    graph = write_graph(tmp_path, "S a * LN:i:5\nP p a+ *\n")
    assert graph_statistics(graph)["path_lengths"] == {"p": 5}
    assert path_steps(graph)["p"][0]["end"] == 5
    with pytest.raises(ValueError, match="stored sequence"):
        path_sequences(graph)


def test_parallel_overlap_requires_explicit_disambiguation(tmp_path):
    graph = write_graph(tmp_path, "S a AAA\nS b AAA\nL a + b + 1M\nL a + b + 2M\nP p a+,b+ *\n")
    with pytest.raises(ValueError, match="ambiguous"):
        path_sequences(graph)
    assert inspect_topology(graph)["parallel_edges"][0]["records"] == 2
    graph = write_graph(tmp_path, "S a AAA\nS b AAA\nL a + b + 1M\nL a + b + 2M\nP p a+,b+ 1M\n")
    assert path_sequences(graph) == {"p": "AAAAA"}


def test_node_and_interval_subgraphs_are_valid_without_partial_paths(diamond, tmp_path):
    selected = tmp_path / "selected.gfa"
    selected.write_text(extract_subgraph(diamond, path_interval=("alpha#0#mito", 2, 3)))
    assert list(load_gfa(selected).segments) == ["b"]
    assert load_gfa(selected).paths == {}
    with pytest.raises(ValueError, match="outside"):
        extract_subgraph(diamond, path_interval=("alpha#0#mito", 0, 100))
    with pytest.raises(ValueError, match="limited"):
        inspect_topology(diamond, max_nodes=2)


def test_walk_intervals_remain_separate_and_cannot_overlap(tmp_path):
    graph = write_graph(tmp_path, "S a AC\nW s 0 m 0 2 >a\nW s 0 m 3 5 <a\n")
    assert path_sequences(graph) == {"s#0#m:0-2": "AC", "s#0#m:3-5": "GT"}
    assert node_pav(graph)["sample_matrix"] == [[1]]
    graph = write_graph(tmp_path, "S a AC\nW s 0 m 0 2 >a\nW s 0 m 1 3 <a\n")
    with pytest.raises(ValueError, match=r"intervals.*overlap"):
        load_gfa(graph)


@pytest.mark.parametrize("overlap", ["1I", "1D", "1X", "1M1I"])
def test_gapped_or_mismatched_cigar_does_not_invent_consensus(tmp_path, overlap):
    graph = write_graph(tmp_path, f"S a AA\nS b AA\nL a + b + {overlap}\nP p a+,b+ *\n")
    assert node_pav(graph)["matrix"] == [[1], [1]]
    with pytest.raises(ValueError, match="non-exact"):
        path_sequences(graph)


def test_cloud_threshold_and_unknown_paths(tmp_path):
    graph = write_graph(tmp_path, "S a A\nS b C\nP p a+ *\nP q b+ *\n")
    assert node_pav(graph, cloud_threshold=0.5)["classes"] == ["cloud", "cloud"]
    assert node_pav(graph, cloud_threshold=0.49)["classes"] == ["shell", "shell"]
    with pytest.raises(ValueError, match="cloud_threshold"):
        node_pav(graph, cloud_threshold=1)
    with pytest.raises(ValueError, match="unknown paths"):
        extract_subgraph(graph, ["absent"])


def test_no_paths_cannot_stand_in_for_sample_presence(tmp_path):
    graph = write_graph(tmp_path, "S a AAA\nS b TTT\n")
    assert graph_statistics(graph)["paths"] == 0
    with pytest.raises(ValueError, match="requires P or W"):
        node_pav(graph)
    with pytest.raises(ValueError, match="no paths"):
        path_sequences(graph)


def test_graph_slice_is_bounded_and_oriented(diamond):
    from organelleverse.pangenome.graph import graph_slice

    result = graph_slice(diamond, offset=0, limit=3)
    assert result["total_nodes"] == 5
    assert result["nodes"] == [
        {"id": "a", "length": 2},
        {"id": "b", "length": 1},
        {"id": "c", "length": 1},
    ]
    assert [(edge["source"], edge["target"]) for edge in result["edges"]] == [
        ("a", "b"),
        ("a", "c"),
    ]
    assert result["paths"][0] == {"name": "alpha#0#mito", "sample": "alpha", "steps": 3}
    assert graph_slice(diamond, offset=5)["nodes"] == []
    for bad in [0, 501, True, 2.5]:
        with pytest.raises(ValueError, match="limit"):
            graph_slice(diamond, limit=bad)


def test_equivalent_cigar_runs_and_zero_equal_walk(tmp_path):
    graph = write_graph(tmp_path, "S a AAA\nS b AAA\nL a + b + 2M\nP p a+,b+ 1M1M\n")
    assert path_sequences(graph) == {"p": "AAAA"}
    graph = write_graph(tmp_path, "S a A\nS b C\nL a + b + 0=\nW s 0 m 0 2 >a>b\n")
    assert path_sequences(graph) == {"s#0#m:0-2": "AC"}
    assert path_steps(graph)["s#0#m:0-2"][0]["coordinate_system"] == "molecule"


def test_explicit_exact_match_and_link_match_share_overlap_geometry(tmp_path):
    graph = write_graph(tmp_path, "S a AAA\nS b AAA\nL a + b + 2M\nP p a+,b+ 2=\n")
    assert path_sequences(graph) == {"p": "AAAA"}
