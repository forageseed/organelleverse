import pytest

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.result import OrganelleResult
from organelleverse.pangenome.benchmark_evaluator import evaluate_taxonomy_benchmark
from organelleverse.pangenome.taxonomy_benchmark import TaxonomyTruth, score_taxonomy

SEGMENTS = {
    "common": "ACGTACGTAC",
    "fam1": "TTGACCAGTA",
    "fam2": "GGATCCATGC",
    "triX": "CATGCAGTCC",
    "triY": "ATCGGATTCA",
    "triZ": "GCTAGCTTAG",
    "ua1": "AAAACCCC",
    "ua2": "AAAAGGGG",
    "ub1": "TTTTCCCC",
    "ub2": "TTTTGGGG",
    "uc1": "CCCCAAAA",
    "uc2": "CCCCTTTT",
}
PATHS = {
    "a1": ("common", "fam1", "triX", "ua1"),
    "a2": ("common", "fam1", "triX", "ua2"),
    "b1": ("common", "fam1", "triY", "ub1"),
    "b2": ("common", "fam1", "triY", "ub2"),
    "c1": ("common", "fam2", "triZ", "uc1"),
    "c2": ("common", "fam2", "triZ", "uc2"),
}
FAMILIES = {"a1": "F1", "a2": "F1", "b1": "F1", "b2": "F1", "c1": "F2", "c2": "F2"}
TRIBES = {"a1": "X", "a2": "X", "b1": "Y", "b2": "Y", "c1": "Z", "c2": "Z"}


def gfa(tmp_path, paths=PATHS):
    lines = [f"S\t{name}\t{seq}" for name, seq in SEGMENTS.items()]
    links = {(a, b) for steps in paths.values() for a, b in zip(steps, steps[1:], strict=False)}
    lines += [f"L\t{a}\t+\t{b}\t+\t0M" for a, b in sorted(links)]
    lines += [
        "P\t" + name + "\t" + ",".join(f"{step}+" for step in steps) + "\t*"
        for name, steps in paths.items()
    ]
    path = tmp_path / "graph.gfa"
    path.write_text("\n".join(lines) + "\n")
    return path


def truth(**changes):
    payload = {
        "schema_version": "organelleverse.pangenome.taxonomy-truth.v1",
        "benchmark_id": "pangenome.fixture",
        "benchmark_version": "1.0.0",
        "case_id": "test",
        "source_description": "Hand-written node topology",
        "sequences": {
            name: "".join(SEGMENTS[step] for step in steps) for name, steps in PATHS.items()
        },
        "accessions": {name: name.upper() for name in PATHS},
        "families": FAMILIES,
        "tribes": TRIBES,
    }
    payload.update(changes)
    return TaxonomyTruth.model_validate(payload)


def test_nested_taxonomy_is_recovered_exactly(tmp_path):
    score = score_taxonomy(gfa(tmp_path), truth())
    metrics = score["metrics"]
    assert metrics["taxonomic_pair_auroc"] == 1
    assert metrics["taxonomic_pair_average_precision"] == 1
    assert metrics["taxonomic_clade_recovery"] == 1
    assert metrics["family_clade_recovery"] == 1
    assert metrics["tribe_clade_recovery"] == 1
    assert metrics["same_tribe_prevalence"] == pytest.approx(3 / 15)
    assert {item["rank"]: item["testable"] for item in score["recoveries"]} == {
        "family": 2,
        "tribe": 3,
    }


def test_mislabelled_tribe_is_not_recovered(tmp_path):
    wrong = dict(TRIBES, a2="Y", b1="X")
    score = score_taxonomy(gfa(tmp_path), truth(tribes=wrong))
    assert score["metrics"]["tribe_clade_recovery"] < 1
    assert score["metrics"]["taxonomic_pair_auroc"] < 1
    assert score["metrics"]["family_clade_recovery"] == 1


def test_rank_without_a_testable_group_reports_no_rate(tmp_path):
    single_family = dict.fromkeys(FAMILIES, "F")
    score = score_taxonomy(gfa(tmp_path), truth(families=single_family))
    assert "family_clade_recovery" not in score["metrics"]
    assert score["metrics"]["taxonomic_clade_recovery"] == 1


def test_graph_must_reconstruct_every_declared_molecule(tmp_path):
    altered = truth().model_dump()
    altered["sequences"]["a1"] = altered["sequences"]["a1"][:-1] + "T"
    with pytest.raises(ValueError, match="exactly reconstruct"):
        score_taxonomy(gfa(tmp_path), TaxonomyTruth.model_validate(altered))


def test_admitted_evaluator_binds_graph_and_truth_artifacts(tmp_path):
    reference = tmp_path / "truth.json"
    reference.write_text(truth().model_dump_json())
    graph_ref = ArtifactRef.from_path(gfa(tmp_path), kind="pangenome_graph", format="gfa")
    truth_ref = ArtifactRef.from_path(reference, kind="benchmark_truth", format="json")
    result = evaluate_taxonomy_benchmark(
        OrganelleResult(
            operation_id="pangenome.build_graph",
            scope="plastid",
            status="ok",
            artifacts=(graph_ref, truth_ref),
        ),
        truth_artifact_id=truth_ref.object_id,
    )
    assert result.operation_id == "pangenome.evaluate_taxonomy_benchmark"
    assert result.metrics["optimization_evaluation"]["taxonomic_clade_recovery"] == 1
    assert set(result.provenance.input_artifact_hashes) == {graph_ref.sha256, truth_ref.sha256}
    assert "homology_f1" not in result.metrics["optimization_evaluation"]


def test_replicates_and_degenerate_cohorts_are_rejected():
    with pytest.raises(ValueError, match="independent accession"):
        truth(accessions={name: "SAME" for name in PATHS})
    with pytest.raises(ValueError, match="same- and different-tribe"):
        truth(tribes={name: name for name in PATHS})
    with pytest.raises(ValueError, match="one accession, family and tribe"):
        truth(families={"a1": "F1"})
    with pytest.raises(ValueError, match="A/C/G/T"):
        truth(sequences={**truth().sequences, "a1": "ACGN"})
