"""Exact PAV equivalence, bounded storage, and row-group pagination."""

import csv
import random

import numpy as np
import pyarrow.parquet as pq
import pytest

from organelleverse.pangenome import pav_store
from organelleverse.pangenome.graph import node_pav
from organelleverse.pangenome.pav_store import (
    iter_pav_batches,
    iter_pav_rows,
    load_pav_metadata,
    pav_summary,
    read_pav_page,
    verify_pav_store,
    write_node_pav,
)


@pytest.fixture
def fixture(tmp_path):
    graph = tmp_path / "source.gfa"
    graph.write_text(
        "S\t1\tAC\nS\t2\tGG\nS\t3\tTT\nS\t4\tAA\nS\t5\t*\tLN:i:4\nL\t1\t+\t2\t+\t0M\nL\t1\t+\t3\t+\t0M\nL\t1\t+\t4\t+\t0M\nP\ta#1#chr\t1+,2+\t*\nP\ta#1#other\t2-\t*\nP\tb#1#chr\t1+,3+\t*\nW\tc\t0\tchr\t0\t2\t>1\nP\td#1#chr\t1+,4+\t*\n"
    )
    directory = tmp_path / "pav"
    metadata = write_node_pav(graph, directory, cloud_threshold=0.25, row_group_size=2)
    return graph, directory, metadata


def assert_equivalent(graph, directory):
    expected = node_pav(graph, cloud_threshold=load_pav_metadata(directory).cloud_threshold)
    for unit, prefix in (("sample", "sample_"), ("path", "")):
        actual = np.concatenate(list(iter_pav_batches(directory, unit=unit, batch_size=2)))
        np.testing.assert_array_equal(actual, expected[prefix + "matrix"])
        scalars = list(iter_pav_rows(directory, unit=unit))
        assert [r["node_id"] for r in scalars] == expected["nodes"]
        assert [r["count"] for r in scalars] == expected[prefix + "counts"]
        assert [r["frequency"] for r in scalars] == expected[prefix + "frequencies"]
        assert [r["category"] for r in scalars] == expected[prefix + "classes"]
        with (directory / f"node_{unit}_pav.tsv").open() as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
        assert rows[0][4:] == expected["samples" if unit == "sample" else "paths"]
        assert [[int(v) for v in row[4:]] for row in rows[1:]] == expected[prefix + "matrix"]


def test_parquet_tsv_and_streaming_exactly_match_dense_reference(fixture):
    graph, directory, metadata = fixture
    assert_equivalent(graph, directory)
    assert metadata.sample_class_counts == {"core": 1, "cloud": 3, "unobserved": 1}
    assert metadata.path_class_counts == {"shell": 2, "cloud": 2, "unobserved": 1}
    assert verify_pav_store(directory) == metadata
    assert "matrix" not in (directory / "pav-metadata.json").read_text()
    assert pav_summary(directory)["sample_frequency_counts"] == {4: 1, 1: 3, 0: 1}


def test_row_and_column_pages_preserve_global_frequency(fixture):
    _, directory, _ = fixture
    page = read_pav_page(
        directory, offset=1, limit=3, columns=["d", "a"], column_offset=1, column_limit=1
    )
    assert page["samples"] == page["columns"] == ["a"]
    assert [r["node_id"] for r in page["rows"]] == ["2", "3", "4"]
    assert [r["presence"] for r in page["rows"]] == [[1], [0], [0]]
    assert [r["frequency"] for r in page["rows"]] == [0.25, 0.25, 0.25]
    assert page["frequency_denominator"] == 4
    assert page["total_columns"] == 2 and page["total_graph_columns"] == 4
    selected = read_pav_page(directory, node_ids=["4", "1"], offset=1, limit=1)
    assert selected["total_rows"] == 2
    assert selected["rows"][0]["node_id"] == "4"
    assert read_pav_page(directory, node_ids=[])["rows"] == []
    assert read_pav_page(directory, offset=100)["rows"] == []
    assert read_pav_page(directory, column_offset=100)["rows"][0]["presence"] == []


def test_column_order_in_tree_batch_interface(fixture):
    _, directory, _ = fixture
    matrix = np.concatenate(list(iter_pav_batches(directory, columns=["d", "a"])))
    assert matrix.tolist() == [
        [True, True],
        [False, True],
        [False, False],
        [True, False],
        [False, False],
    ]


def test_invalid_labels_limits_and_changed_artifacts_fail(fixture):
    _, directory, _ = fixture
    for params in [
        {"limit": 501},
        {"column_limit": 501},
        {"offset": -1},
        {"columns": ["missing"]},
        {"node_ids": ["absent"]},
        {"unit": "contig"},
    ]:
        with pytest.raises(ValueError):
            read_pav_page(directory, **params)
    table = directory / "node_sample_pav.parquet"
    content = bytearray(table.read_bytes())
    content[12] ^= 1
    table.write_bytes(content)
    with pytest.raises(ValueError, match="changed"):
        verify_pav_store(directory)


def test_random_multimolecule_paths_match_dense_reference(tmp_path):
    rng = random.Random(19)
    graph = tmp_path / "random.gfa"
    graph.write_text(
        "".join(f"S\t{i}\tA\n" for i in range(19))
        + "".join(f"P\ts{i % 7}#1#chr{i}\t{rng.randrange(19)}+\t*\n" for i in range(37))
    )
    directory = tmp_path / "pav"
    write_node_pav(graph, directory, cloud_threshold=0.2, row_group_size=3)
    assert_equivalent(graph, directory)


def test_wide_store_bounds_row_groups_and_page_seeks_one_group(tmp_path, monkeypatch):
    graph = tmp_path / "wide.gfa"
    graph.write_text(
        "".join(f"S\t{i}\tA\n" for i in range(3000))
        + "".join(f"P\ts{i}#1#chr\t{i % 3000}+\t*\n" for i in range(1025))
    )
    directory = tmp_path / "pav"
    write_node_pav(graph, directory)
    parquet = pq.ParquetFile(directory / "node_sample_pav.parquet")
    assert all(
        parquet.metadata.row_group(i).num_rows * 1025 <= 1_000_000
        for i in range(parquet.num_row_groups)
    )
    real_open = pav_store._open
    groups = []

    class Reader:
        def __init__(self, source):
            self.source = source

        def __getattr__(self, name):
            return getattr(self.source, name)

        def read_row_group(self, group, **kwargs):
            groups.append(group)
            return self.source.read_row_group(group, **kwargs)

    monkeypatch.setattr(pav_store, "_open", lambda *args: Reader(real_open(*args)))
    page = read_pav_page(directory, offset=1100, limit=2, column_offset=1000, column_limit=5)
    assert len(page["rows"]) == 2 and all(len(row["presence"]) == 5 for row in page["rows"])
    assert groups == [1]
    assert page["rows"][0]["node_id"] == "1100"
