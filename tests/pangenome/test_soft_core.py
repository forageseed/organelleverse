"""Soft core boundaries retain sample OR aggregation and strict defaults."""

from collections import Counter

import pytest

from organelleverse.pangenome.graph import node_pav
from organelleverse.pangenome.pav_store import iter_pav_rows, load_pav_metadata, write_node_pav
from organelleverse.pangenome.workflow_models import WorkflowRequest


@pytest.mark.parametrize("backend,recommend", [("pggb", False), ("minigraph", True)])
def test_automatic_adoption_requires_pggb_and_recommendation(backend, recommend):
    with pytest.raises(ValueError, match="Automatic adoption requires"):
        WorkflowRequest(
            dataset_path="inputs",
            backend=backend,
            recommend_parameters=recommend,
            auto_adopt_recommendation=True,
        )


def test_soft_core_matches_independent_counts_and_stored_categories(tmp_path):
    graph = tmp_path / "graph.gfa"
    # 20 samples: exact 95% boundary, one below, cloud boundary and absent node.
    memberships = {"strict": 20, "soft": 19, "shell": 18, "cloud": 1, "absent": 0}
    lines = [f"S\t{node}\tAC" for node in memberships]
    # Separate single-node W intervals represent the same biological sample.
    for node, count in memberships.items():
        lines.extend(f"W\ts{i}\t0\t{node}\t0\t2\t>{node}" for i in range(count))
    graph.write_text("\n".join(lines) + "\n")
    strict = node_pav(graph)
    soft = node_pav(graph, core_threshold=0.95)
    assert strict["core_threshold"] == 1.0
    assert strict["sample_classes"] == ["core", "shell", "shell", "cloud", "unobserved"]
    expected = ["core", "core", "shell", "cloud", "unobserved"]
    assert soft["sample_classes"] == expected
    assert soft["sample_matrix"] == strict["sample_matrix"]
    directory = tmp_path / "pav"
    metadata = write_node_pav(graph, directory, core_threshold=0.95, row_group_size=2)
    assert metadata.sample_class_counts == dict(Counter(expected))
    assert load_pav_metadata(directory).core_threshold == 0.95
    assert [r["category"] for r in iter_pav_rows(directory)] == expected
    assert [r["category"] for r in iter_pav_rows(directory, unit="path")] == soft["classes"]


@pytest.mark.parametrize(
    "core,cloud", [(0, 0), (0.05, 0.05), (0.04, 0.05), (1.1, 0.05), (float("nan"), 0.05)]
)
def test_invalid_or_overlapping_thresholds_fail_before_writing(tmp_path, core, cloud):
    with pytest.raises(ValueError):
        node_pav(tmp_path / "missing", cloud, core_threshold=core)
    with pytest.raises(ValueError):
        write_node_pav(
            tmp_path / "missing", tmp_path / "out", cloud_threshold=cloud, core_threshold=core
        )
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError):
        WorkflowRequest(
            backend="existing", gfa_path="graph.gfa", cloud_threshold=cloud, core_threshold=core
        )
