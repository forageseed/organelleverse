import json

import pytest

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.result import OrganelleResult
from organelleverse.pangenome.benchmark_evaluator import (
    GraphBenchmarkTruth,
    evaluate_graph_benchmark,
    score_graph,
)


def truth():
    return GraphBenchmarkTruth.model_validate(
        {
            "schema_version": "organelleverse.pangenome.graph-truth.v1",
            "benchmark_id": "pangenome.conserved_homology",
            "benchmark_version": "1.0.0",
            "case_id": "two-identical-base-positions",
            "source_description": "Synthetic exact correspondence fixture; not biological validation",
            "sequences": {"a#1#1": "AA", "b#1#1": "AA"},
            "anchors": [
                {
                    "first": {"path": "a#1#1", "position": 0},
                    "second": {"path": "b#1#1", "position": 0},
                    "homologous": True,
                },
                {
                    "first": {"path": "a#1#1", "position": 0},
                    "second": {"path": "b#1#1", "position": 1},
                    "homologous": False,
                },
            ],
        }
    )


def test_exact_loci_distinguish_false_collapse_from_correct_sharing(tmp_path):
    correct = tmp_path / "correct.gfa"
    correct.write_text("S\t1\tAA\nP\ta#1#1\t1+\t*\nP\tb#1#1\t1+\t*\n")
    collapsed = tmp_path / "collapsed.gfa"
    collapsed.write_text("S\t1\tA\nL\t1\t+\t1\t+\t0M\nP\ta#1#1\t1+,1+\t0M\nP\tb#1#1\t1+,1+\t0M\n")
    good = score_graph(correct, truth())
    bad = score_graph(collapsed, truth())
    assert good["confusion"] == {"tp": 1, "tn": 1, "fp": 0, "fn": 0}
    assert good["metrics"]["homology_f1"] == 1.0
    assert bad["confusion"] == {"tp": 1, "tn": 0, "fp": 1, "fn": 0}
    assert bad["metrics"]["homology_f1"] == pytest.approx(2 / 3)
    assert good["metrics"]["exact_input_paths"] == bad["metrics"]["exact_input_paths"] == 1.0


def test_unshared_homology_is_not_rewarded_as_graph_correctness(tmp_path):
    graph = tmp_path / "unshared.gfa"
    graph.write_text("S\t1\tAA\nS\t2\tAA\nP\ta#1#1\t1+\t*\nP\tb#1#1\t2+\t*\n")
    scored = score_graph(graph, truth())
    assert scored["metrics"]["homology_f1"] == 0.0
    assert scored["confusion"] == {"tp": 0, "tn": 1, "fp": 0, "fn": 1}


def test_reverse_path_offsets_map_to_physical_graph_bases(tmp_path):
    graph = tmp_path / "reverse.gfa"
    graph.write_text("S\t1\tACGT\nP\ta#1#1\t1+\t*\nP\tb#1#1\t1-\t*\n")
    data = truth().model_dump(mode="json")
    data["sequences"] = {"a#1#1": "ACGT", "b#1#1": "ACGT"}
    data["anchors"][0]["second"]["position"] = 3
    data["anchors"][1]["second"]["position"] = 0
    scored = score_graph(graph, GraphBenchmarkTruth.model_validate(data))
    assert scored["metrics"]["homology_f1"] == 1.0


def test_changed_input_molecules_fail_before_any_scientific_score(tmp_path):
    graph = tmp_path / "changed.gfa"
    graph.write_text("S\t1\tAC\nP\ta#1#1\t1+\t*\nP\tb#1#1\t1+\t*\n")
    with pytest.raises(ValueError, match="exactly reconstruct"):
        score_graph(graph, truth())


def test_native_evaluator_preserves_ground_truth_and_raw_anchor_evidence(tmp_path):
    graph = tmp_path / "graph.gfa"
    graph.write_text("S\t1\tAA\nP\ta#1#1\t1+\t*\nP\tb#1#1\t1+\t*\n")
    reference = tmp_path / "truth.json"
    reference.write_text(truth().model_dump_json())
    graph_ref = ArtifactRef.from_path(graph, kind="pangenome_graph", format="gfa")
    truth_ref = ArtifactRef.from_path(reference, kind="benchmark_truth", format="json")
    source = OrganelleResult(
        operation_id="pangenome.build_graph",
        scope="mitochondrion",
        status="ok",
        artifacts=(graph_ref, truth_ref),
    )
    evaluated = evaluate_graph_benchmark(source, truth_artifact_id=truth_ref.object_id)
    assert evaluated.metrics["optimization_evaluation"]["homology_f1"] == 1.0
    assert set(evaluated.provenance.input_artifact_hashes) == {graph_ref.sha256, truth_ref.sha256}
    artifact = evaluated.artifacts[0]
    raw = json.loads(artifact.resolve().read_text())
    assert len(raw["anchors"]) == 2 and raw["anchors"][0]["first_graph_locus"] == ["1", 0]
    assert reference.read_text() == truth().model_dump_json()


def test_truth_with_conflicting_or_single_class_labels_is_rejected():
    data = truth().model_dump(mode="json")
    data["anchors"][1] = {**data["anchors"][0], "homologous": False}
    with pytest.raises(ValueError, match="distinct and unique"):
        GraphBenchmarkTruth.model_validate(data)
    data = truth().model_dump(mode="json")
    data["anchors"][1]["homologous"] = True
    with pytest.raises(ValueError, match="positive and negative"):
        GraphBenchmarkTruth.model_validate(data)
