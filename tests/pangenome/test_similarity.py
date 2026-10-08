from pathlib import Path

import pytest

from organelleverse.pangenome.similarity import graph_shared_windows, render_shared_windows


def gfa(tmp_path, text):
    path = tmp_path / "graph.gfa"
    path.write_text(text)
    return path


def overlap_graph(tmp_path, *, extra_comparison_molecule=False):
    return gfa(
        tmp_path,
        "S\t1\tAAAA\nS\t2\tAAAA\nS\t3\tCC\nL\t1\t+\t2\t+\t2M\nL\t2\t+\t3\t+\t0M\nP\ta#1#chr\t1+,2+,3+\t*\nP\ta#1#other\t3+\t*\nP\tb#1#chr\t1+,2+\t*\n"
        + ("P\tb#1#other\t3+\t*\n" if extra_comparison_molecule else ""),
    )


def test_overlapping_node_spans_union_before_window_coverage(tmp_path):
    result = graph_shared_windows(overlap_graph(tmp_path), bin_size=5)
    rows = [r for r in result["rows"] if r["path"] == "a#1#chr"]
    assert [(r["start"], r["end"], r["shared_bp"], r["window_bp"]) for r in rows] == [
        (0, 5, 5, 5),
        (5, 8, 1, 3),
    ]
    assert [r["shared_fraction"] for r in rows] == [1, 1 / 3]
    assert all(r["reference_sample"] != r["comparison_sample"] for r in result["rows"])
    assert result["samples"] == ["a", "b"]
    assert result["source_graph"]["format"] == "gfa"
    assert result["source_graph"]["sha256"]
    assert "not nucleotide identity" in result["interpretation"]


def test_comparison_sample_presence_unions_molecules(tmp_path):
    result = graph_shared_windows(
        overlap_graph(tmp_path, extra_comparison_molecule=True), bin_size=5
    )
    assert all(r["shared_fraction"] == 1 for r in result["rows"])
    assert {r["comparison_sample"] for r in result["rows"] if r["reference_sample"] == "a"} == {"b"}


def test_walk_native_offsets_and_intervals_remain_separate(tmp_path):
    path = gfa(
        tmp_path,
        "S\t1\tAAAA\nS\t2\tCC\nW\ta\t0\tchr\t100\t104\t>1\nW\ta\t0\tchr\t200\t202\t>2\nW\tb\t0\tchr\t0\t4\t<1\n",
    )
    result = graph_shared_windows(path, bin_size=3)
    rows = [r for r in result["rows"] if r["reference_sample"] == "a"]
    assert [(r["start"], r["end"], r["shared_bp"]) for r in rows] == [
        (100, 103, 3),
        (103, 104, 1),
        (200, 202, 0),
    ]
    assert len({r["path"] for r in rows}) == 2


def test_repeated_node_visits_cover_each_reference_occurrence_once(tmp_path):
    path = gfa(
        tmp_path,
        "S\t1\tAA\nS\t2\tCC\nL\t1\t+\t2\t+\t0M\nL\t2\t+\t1\t+\t0M\nP\ta#1#chr\t1+,2+,1+\t*\nP\tb#1#chr\t1-\t*\n",
    )
    rows = graph_shared_windows(path, bin_size=6)["rows"]
    assert rows[0]["shared_bp"] == 4
    assert rows[0]["shared_fraction"] == 2 / 3
    assert rows[1]["shared_fraction"] == 1


@pytest.mark.parametrize(
    "text,reason",
    [
        ("S\t1\t*\nP\ta\t1+\t*\nP\tb\t1+\t*\n", "unknown segment length"),
        ("S\t1\tAAAA\nW\ta\t0\tchr\t*\t*\t>1\nP\tb\t1+\t*\n", "coordinates are unknown"),
        (
            "S\t1\tAAAA\nS\t2\tCCCC\nL\t1\t+\t2\t+\t2M\nP\ta\t1+,2+\t*\nP\tb\t1+\t*\n",
            "sequences disagree",
        ),
        ("S\t1\tAAAA\nS\t2\tAAAA\nL\t1\t+\t2\t+\t*\nP\ta\t1+,2+\t*\nP\tb\t1+\t*\n", "overlap"),
    ],
)
def test_unknown_or_inexact_coordinate_inputs_fail_explicitly(tmp_path, text, reason):
    with pytest.raises(ValueError, match=reason):
        graph_shared_windows(gfa(tmp_path, text))


def test_known_length_zero_overlap_allows_unstored_bases(tmp_path):
    path = gfa(tmp_path, "S\t1\t*\tLN:i:5\nP\ta\t1+\t*\nP\tb\t1+\t*\n")
    assert [r["shared_fraction"] for r in graph_shared_windows(path)["rows"]] == [1, 1]


def test_output_row_bound_and_no_within_sample_pseudoreplication(tmp_path):
    with pytest.raises(ValueError, match="max_rows"):
        graph_shared_windows(overlap_graph(tmp_path), bin_size=1, max_rows=2)
    path = gfa(tmp_path, "S\t1\tAAAA\nP\ta#1#chr\t1+\t*\nP\ta#1#other\t1+\t*\n")
    with pytest.raises(ValueError, match="two biological samples"):
        graph_shared_windows(path)


def test_profiles_export_exact_source_table_and_requested_formats(tmp_path):
    data = graph_shared_windows(overlap_graph(tmp_path), bin_size=5)
    outputs = render_shared_windows(data, tmp_path / "figures", formats=("svg", "pdf", "png"))
    table = Path(outputs["table"]).read_text()
    assert "shared_fraction" in table
    assert "a#1#chr\ta\tb\t5\t8\t1\t3\t0.3333333333333333" in table
    assert Path(outputs["path_0001_svg"]).read_text().find("Graph-shared fraction") > 0
    assert Path(outputs["path_0001_pdf"]).read_bytes().startswith(b"%PDF")
    assert Path(outputs["path_0001_png"]).read_bytes().startswith(b"\x89PNG")


def test_cohort_batches_keep_all_paths_and_comparison_samples(tmp_path, monkeypatch):
    from organelleverse.pangenome import similarity
    rows = [{"path": f"p{i}", "comparison_sample": sample} for i in range(101) for sample in ("a", "b")]
    captured = []
    def render(data, directory, **kwargs):
        assert len({row["path"] for row in data["rows"]}) <= kwargs["max_paths"]
        captured.extend(data["rows"])
        return {"table": str(directory / "graph_shared_windows.tsv")}
    monkeypatch.setattr(similarity, "render_shared_windows", render)
    batches = list(similarity.render_shared_window_batches({"rows": rows}, tmp_path))
    assert len(batches) == 2
    assert captured == rows
    assert len({batch["table"] for batch in batches}) == 2


def test_parallel_profile_rendering_preserves_exact_tables_and_all_outputs(tmp_path):
    data = graph_shared_windows(overlap_graph(tmp_path), bin_size=5)
    serial = render_shared_windows(data, tmp_path / "serial", formats=("svg",), workers=1)
    parallel = render_shared_windows(data, tmp_path / "parallel", formats=("svg",), workers=2)
    assert set(serial) == set(parallel)
    assert Path(serial["table"]).read_bytes() == Path(parallel["table"]).read_bytes()
    for key, path in parallel.items():
        assert Path(path).is_file()
        if key != "table":
            assert "Graph-shared fraction" in Path(path).read_text()
