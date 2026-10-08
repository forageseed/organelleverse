import pytest

from organelleverse.pangenome.conformation_benchmark import (
    ConformationTruth,
    ranking_metrics,
    score_conformations,
)


def truth():
    return ConformationTruth(
        schema_version="organelleverse.pangenome.conformation-truth.v1",
        benchmark_id="pangenome.fixture",
        benchmark_version="1.0.0",
        case_id="test",
        source_description="Hand-written edge topology",
        sequences={"a": "AACCGG", "b": "CCGGTT", "c": "AATTGG"},
        accessions={"a": "A", "b": "B", "c": "C"},
        conformations={"a": "one", "b": "one", "c": "two"},
    )


def test_bidirected_edges_are_orientation_invariant(tmp_path):
    graph = tmp_path / "graph.gfa"
    graph.write_text(
        "S\t1\tAA\nS\t2\tCC\nS\t3\tGG\nS\t4\tTT\n"
        "L\t1\t+\t2\t+\t0M\nL\t2\t+\t3\t+\t0M\nL\t1\t+\t4\t+\t0M\nL\t4\t+\t3\t+\t0M\n"
        "P\ta\t1+,2+,3+\t*\nP\tb\t3-,2-,1-\t*\nP\tc\t1+,4+,3+\t*\n"
    )
    score = score_conformations(graph, truth())
    assert score["metrics"]["conformation_auroc"] == 1
    assert score["metrics"]["conformation_average_precision"] == 1
    assert score["pairs"][0]["edge_jaccard"] == 1
    from organelleverse.core.artifacts import ArtifactRef
    from organelleverse.core.result import OrganelleResult
    from organelleverse.pangenome.benchmark_evaluator import evaluate_conformation_benchmark

    reference = tmp_path / "truth.json"
    reference.write_text(truth().model_dump_json())
    graph_ref = ArtifactRef.from_path(graph, kind="pangenome_graph", format="gfa")
    truth_ref = ArtifactRef.from_path(reference, kind="benchmark_truth", format="json")
    result = evaluate_conformation_benchmark(
        OrganelleResult(
            operation_id="pangenome.build_graph",
            scope="mitochondrion",
            status="ok",
            artifacts=(graph_ref, truth_ref),
        ),
        truth_artifact_id=truth_ref.object_id,
    )
    assert result.operation_id == "pangenome.evaluate_conformation_benchmark"
    assert result.metrics["optimization_evaluation"]["conformation_auroc"] == 1
    assert set(result.provenance.input_artifact_hashes) == {graph_ref.sha256, truth_ref.sha256}
    assert "homology_f1" not in result.metrics["optimization_evaluation"]
    altered = truth().model_dump()
    altered["sequences"]["a"] = "AAACGG"
    with pytest.raises(ValueError, match="exactly reconstruct"):
        score_conformations(graph, ConformationTruth.model_validate(altered))


def test_ties_do_not_depend_on_pair_order():
    pairs = [{"same_conformation": label, "edge_jaccard": 0.25} for label in (True, False, False)]
    assert ranking_metrics(pairs) == ranking_metrics(pairs[::-1])
    assert ranking_metrics(pairs)["conformation_auroc"] == 0.5
    assert ranking_metrics(pairs)["conformation_average_precision"] == pytest.approx(1 / 3)


def test_reverse_ranking_is_not_clipped_to_chance():
    pairs = [
        {"same_conformation": True, "edge_jaccard": 0},
        {"same_conformation": False, "edge_jaccard": 1},
    ]
    assert ranking_metrics(pairs)["conformation_auroc"] == 0
    assert ranking_metrics(pairs)["conformation_average_precision"] == 0.5


def test_accession_replicates_and_missing_pair_classes_are_rejected():
    payload = truth().model_dump()
    payload["accessions"]["b"] = "A"
    with pytest.raises(ValueError, match="independent accession"):
        ConformationTruth.model_validate(payload)
    payload = truth().model_dump()
    payload["conformations"]["b"] = "three"
    with pytest.raises(ValueError, match="same- and different"):
        ConformationTruth.model_validate(payload)
