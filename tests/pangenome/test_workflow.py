"""Scientific and recovery contracts for the managed pangenome workflow."""

import json
import os
import sys

import pytest

from organelleverse.pangenome import workflow
from organelleverse.pangenome.stage_worker import StageOutcome
from organelleverse.pangenome.workflow_inputs import prepare_inputs
from organelleverse.pangenome.workflow_models import WorkflowRequest
from organelleverse.pangenome.workflow_store import directory_for


@pytest.fixture
def gfa(tmp_path):
    path = tmp_path / "input.gfa"
    path.write_text(
        "H\tVN:Z:1.0\nS\t1\tACGT\nS\t2\tAA\nS\t3\tTT\n"
        "L\t1\t+\t2\t+\t0M\nL\t1\t+\t3\t+\t0M\n"
        "P\ta#1#1\t1+,2+\t0M\nP\tb#1#1\t1+,3+\t0M\n"
    )
    return path


def test_existing_graph_publishes_real_tables_and_canonical_result(gfa):
    request = WorkflowRequest(backend="existing", gfa_path=str(gfa), bootstrap_replicates=10)
    record = workflow.create_run(request)
    finished = workflow.execute_run(record["run_id"])
    assert finished["status"] == "succeeded", finished
    directory = directory_for(record["run_id"])
    from organelleverse.pangenome.pav_store import read_pav_page

    pav = read_pav_page(directory)
    assert pav["samples"] == ["a", "b"]
    assert [row["presence"] for row in pav["rows"]] == [[1, 1], [1, 0], [0, 1]]
    result = json.loads((directory / "result.json").read_text())
    assert result["operation_id"] == "pangenome.workflow"
    assert result["operation_version"] == finished["workflow_version"]
    assert result["provenance"]["operation_version"] == finished["workflow_version"]
    manifest = json.loads((directory / "inputs.json").read_text())
    expected_inputs = {
        source["object_id"] for source in [*manifest["sources"], *manifest["snapshots"]]
    }
    assert expected_inputs <= set(result["provenance"]["input_object_ids"])
    assert result["artifacts"]
    assert next(s for s in finished["stages"] if s["name"] == "annotation")["status"] == "skipped"


def test_resume_reuses_verified_stages_without_rebuilding(gfa, monkeypatch):
    record = workflow.create_run(WorkflowRequest(backend="existing", gfa_path=str(gfa)))
    original = workflow._execute_stage
    seen = []

    def fail_tree(name, *args):
        seen.append(name)
        if name == "phylogeny":
            raise RuntimeError("interrupted tree")
        return original(name, *args)

    monkeypatch.setattr(workflow, "_execute_stage", fail_tree)
    failed = workflow.execute_run(record["run_id"])
    assert failed["status"] == "failed"
    seen.clear()

    def record_stage(name, *args):
        seen.append(name)
        return original(name, *args)

    monkeypatch.setattr(workflow, "_execute_stage", record_stage)
    finished = workflow.execute_run(record["run_id"])
    assert finished["status"] == "succeeded", finished
    assert "prepare" not in seen and "graph" not in seen and "statistics" not in seen


def test_resume_rejects_modified_checkpoint(gfa, monkeypatch):
    record = workflow.create_run(WorkflowRequest(backend="existing", gfa_path=str(gfa)))
    original = workflow._execute_stage

    def fail_tree(name, *args):
        if name == "phylogeny":
            raise RuntimeError("interrupted")
        return original(name, *args)

    monkeypatch.setattr(workflow, "_execute_stage", fail_tree)
    workflow.execute_run(record["run_id"])
    workflow.graph_path(record["run_id"]).write_text("S\tx\tA\n")
    monkeypatch.setattr(workflow, "_execute_stage", original)
    resumed = workflow.execute_run(record["run_id"])
    assert resumed["status"] == "failed"
    assert "checkpoint artifact changed" in resumed["error"]


def test_input_preparation_preserves_multimolecule_ids_and_sequences(tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "a.fa").write_text(">chromosome\nACGT\n>minicircle\nTTAA\n")
    (dataset / "b.fa").write_text(">mito\nACGT\n")
    target = tmp_path / "run"
    target.mkdir()
    prepare_inputs(WorkflowRequest(dataset_path=str(dataset)), target)
    manifest = json.loads((target / "inputs.json").read_text())
    assert [m["source_id"] for m in manifest["samples"][0]["molecules"]] == [
        "chromosome",
        "minicircle",
    ]
    assert [m["path"] for m in manifest["samples"][0]["molecules"]] == ["a#1#1", "a#1#2"]


def test_duplicate_record_ids_are_rejected(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.fa").write_text(">same\nACGT\n>same\nTTAA\n")
    (data / "b.fa").write_text(">other\nACGT\n")
    target = tmp_path / "run"
    target.mkdir()
    with pytest.raises(ValueError, match="duplicate molecule"):
        prepare_inputs(WorkflowRequest(dataset_path=str(data)), target)


def test_input_snapshots_are_not_republished_and_source_changes_do_not_change_run(gfa):
    record = workflow.create_run(WorkflowRequest(backend="existing", gfa_path=str(gfa)))
    finished = workflow.execute_run(record["run_id"])
    assert finished["status"] == "succeeded", finished
    directory = directory_for(record["run_id"])
    snapshot = workflow.graph_path(record["run_id"])
    assert not snapshot.is_relative_to(directory)
    assert not (directory / "graph.gfa").exists()
    result = json.loads((directory / "result.json").read_text())
    source_hash = json.loads((directory / "inputs.json").read_text())["sources"][0]["sha256"]
    assert all(a["sha256"] != source_hash for a in result["artifacts"])
    gfa.write_text("S\tx\tA\n")
    assert workflow.execute_run(record["run_id"])["status"] == "succeeded"


def test_completed_checkpoint_verification_does_not_mutate_published_record(gfa):
    record = workflow.create_run(WorkflowRequest(backend="existing", gfa_path=str(gfa)))
    assert workflow.execute_run(record["run_id"])["status"] == "succeeded"
    directory = directory_for(record["run_id"])
    before = (directory / "workflow.json").read_bytes()
    (directory / "statistics.json").write_text("{}")
    inspected = workflow.execute_run(record["run_id"])
    assert inspected["status"] == "failed"
    assert "checkpoint artifact changed" in inspected["error"]
    assert (directory / "workflow.json").read_bytes() == before


def test_missing_sequence_is_explicit_optional_export_not_failed_pav(tmp_path):
    graph = tmp_path / "lengths.gfa"
    graph.write_text("S\ta\t*\tLN:i:4\nP\ts\ta+\t*\n")
    record = workflow.create_run(WorkflowRequest(backend="existing", gfa_path=str(graph)))
    finished = workflow.execute_run(record["run_id"])
    assert finished["status"] == "succeeded", finished
    directory = directory_for(record["run_id"])
    assert (directory / "pav-metadata.json").exists()
    diagnostic = json.loads((directory / "sequence-export.json").read_text())
    assert diagnostic["status"] == "unavailable"
    assert "stored sequence" in diagnostic["reason"]
    assert not (directory / "paths.fasta").exists()


@pytest.mark.parametrize("input_format", ["fasta", "genbank"])
def test_built_graph_remains_in_builder_run_and_referenced_once(
    tmp_path, gfa, monkeypatch, input_format
):
    from organelleverse.core.artifacts import ArtifactRef
    from organelleverse.core.result import OrganelleResult
    from organelleverse.pangenome import service
    from organelleverse.pangenome._contract import make_provenance

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    if input_format == "fasta":
        (dataset / "a.fa").write_text(">a\nACGTAA\n")
        (dataset / "b.fa").write_text(">b\nACGTTT\n")
    else:
        from Bio import SeqIO
        from Bio.Seq import Seq
        from Bio.SeqFeature import FeatureLocation, SeqFeature
        from Bio.SeqRecord import SeqRecord

        for sample, sequence in [("a", "ACGTAA"), ("b", "ACGTTT")]:
            sequence_record = SeqRecord(
                Seq(sequence), id="same_molecule", annotations={"molecule_type": "DNA"}
            )
            sequence_record.features = [
                SeqFeature(
                    FeatureLocation(0, 4, strand=1), type="gene", qualifiers={"gene": ["nad1"]}
                )
            ]
            SeqIO.write([sequence_record], dataset / f"{sample}.gb", "genbank")
    reference = ArtifactRef.from_path(gfa, kind="pangenome_graph", format="gfa")
    built = OrganelleResult(
        operation_id="pangenome.build_graph",
        operation_version="1.0",
        scope="mitochondrion",
        status="ok",
        summary_text="test builder",
        artifacts=(reference,),
        flags=("graph_built",),
        provenance=make_provenance(
            operation_id="pangenome.build_graph",
            parameters={},
            requested_backend="pggb",
            actual_backend="pggb",
            attempted_backends=("pggb",),
        ),
    )
    # The test injects a builder result; execute the scientific body in-process
    # at this test boundary, while release tests exercise real stage workers.
    monkeypatch.setattr(
        workflow,
        "_execute_stage",
        lambda *args: StageOutcome(*workflow._execute_stage_body(*args), None),
    )
    monkeypatch.setattr(service, "build_graph", lambda *args, **kwargs: built)
    record = workflow.create_run(WorkflowRequest(dataset_path=str(dataset), backend="pggb"))
    finished = workflow.execute_run(record["run_id"])
    assert finished["status"] == "succeeded", finished
    assert workflow.graph_path(record["run_id"]) == gfa
    if input_format == "genbank":
        assert (directory_for(record["run_id"]) / "annotation.json").exists()
    result = json.loads((directory_for(record["run_id"]) / "result.json").read_text())
    assert all(a["sha256"] != reference.sha256 for a in result["artifacts"])


def test_record_read_survives_atomic_publication_rename(gfa, monkeypatch):
    from pathlib import Path

    from organelleverse.pangenome import workflow_store as store
    from organelleverse.runtime import managed_run_path

    record = workflow.create_run(WorkflowRequest(backend="existing", gfa_path=str(gfa)))
    staging = store.directory_for(record["run_id"])
    target = managed_run_path(store.OPERATION, record["run_id"])
    original = Path.read_text
    moved = False

    def publish_during_read(path, *args, **kwargs):
        nonlocal moved
        if path == staging / "workflow.json" and not moved:
            staging.rename(target)
            moved = True
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", publish_during_read)
    assert store.load(record["run_id"])["run_id"] == record["run_id"]
    assert len(store.list_runs()) == 1


def test_explicit_repeat_and_validation_stages_keep_measured_resource_scope(gfa):
    from organelleverse.core.artifacts import ArtifactRef

    record = workflow.create_run(
        WorkflowRequest(backend="existing", gfa_path=str(gfa), formats=("svg",))
    )
    finished = workflow.execute_run(record["run_id"])
    assert finished["status"] == "succeeded", finished
    assert [stage["name"] for stage in finished["stages"]] == [
        "prepare",
        "repeat",
        "graph",
        "validate",
        "statistics",
        "overview",
        "phylogeny",
        "annotation",
        "report",
    ]
    repeat = next(stage for stage in finished["stages"] if stage["name"] == "repeat")
    assert repeat["status"] == "skipped" and "not requested" in repeat["note"]
    directory = directory_for(record["run_id"])
    assert json.loads((directory / "validation.json").read_text())["status"] == "passed"
    for stage in finished["stages"]:
        assert stage["python_thread_cpu_seconds"] >= 0
        assert "isolated stage process" in stage["cpu_scope"]
        if hasattr(os, "wait4"):
            assert stage["cpu_seconds"] > 0
            assert stage["measurement"] == "wait4_child_rusage"
        if sys.platform.startswith("linux") or sys.platform == "darwin":
            assert stage["peak_memory_bytes"] > 0
        assert "not aggregate" in stage["memory_scope"]
    result = json.loads((directory / "result.json").read_text())
    reference_artifacts = [
        row for row in result["artifacts"] if row["kind"] == "pangenome_graph_reference"
    ]
    assert len(reference_artifacts) == 1
    reference = json.loads(ArtifactRef.model_validate(reference_artifacts[0]).resolve().read_text())
    graph_ref = ArtifactRef.model_validate(reference["graph_ref"])
    assert graph_ref.resolve() == workflow.graph_path(record["run_id"])
    assert graph_ref.sha256 in result["provenance"]["input_artifact_hashes"]


@pytest.mark.parametrize("after_stage", ["graph", "report"])
def test_pause_is_at_stage_boundary_and_resumes_without_reexecution(gfa, monkeypatch, after_stage):
    from organelleverse.pangenome import workflow_store as store
    from organelleverse.runtime import managed_run_path

    record = workflow.create_run(
        WorkflowRequest(backend="existing", gfa_path=str(gfa), formats=("svg",))
    )
    run_id = record["run_id"]
    original = workflow._execute_stage
    seen = []

    def pause_at_boundary(name, *args):
        seen.append(name)
        output = original(name, *args)
        if name == after_stage:
            store.pause_path(run_id).touch()
        return output

    monkeypatch.setattr(workflow, "_execute_stage", pause_at_boundary)
    paused = workflow.execute_run(run_id)
    assert paused["status"] == "paused" and paused["error"] is None
    assert not paused["pause_requested"]
    assert not managed_run_path(store.OPERATION, run_id).exists()
    completed_before_pause = set(seen)
    seen.clear()

    def record_stage(name, *args):
        seen.append(name)
        return original(name, *args)

    monkeypatch.setattr(workflow, "_execute_stage", record_stage)
    resumed = workflow.execute_run(run_id)
    assert resumed["status"] == "succeeded", resumed
    assert not completed_before_pause.intersection(seen)
    assert not store.pause_path(run_id).exists()
    if after_stage == "report":
        assert seen == []


def test_pause_intent_is_visible_without_changing_checkpoint(gfa):
    from organelleverse.pangenome import workflow_store as store

    record = workflow.create_run(
        WorkflowRequest(backend="existing", gfa_path=str(gfa), formats=("svg",))
    )
    directory = directory_for(record["run_id"])
    before = (directory / "workflow.json").read_bytes()
    store.pause_path(record["run_id"]).touch()
    assert store.public_view(store.load(record["run_id"]))["pause_requested"]
    assert (directory / "workflow.json").read_bytes() == before
    paused = workflow.execute_run(record["run_id"])
    assert paused["status"] == "paused"
    assert all(stage["status"] == "pending" for stage in paused["stages"])


@pytest.mark.parametrize(
    "parameters",
    [
        {"run_repeatmasker": True},
        {"repeatmasker_species": "Oryza"},
        {"recommend_parameters": True, "backend": "existing", "gfa_path": "/tmp/input.gfa"},
        {"recommend_parameters": True, "threads": 65},
    ],
)
def test_recommendation_requires_explicit_supported_opt_in(parameters):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        WorkflowRequest(
            dataset_path=None if parameters.get("backend") == "existing" else "/tmp/data",
            **parameters,
        )


@pytest.mark.parametrize("automatic", [False, True])
def test_requested_recommendation_adoption_and_resume(tmp_path, gfa, monkeypatch, automatic):
    from organelleverse.core.artifacts import ArtifactRef
    from organelleverse.core.result import OrganelleResult
    from organelleverse.pangenome import service, tuning_service
    from organelleverse.pangenome._contract import make_provenance

    dataset = tmp_path / "recommendation-data"
    dataset.mkdir()
    (dataset / "a.fa").write_text(">a\nACGTAA\n")
    (dataset / "b.fa").write_text(">b\nACGTTT\n")
    recommendation_file = tmp_path / "external-recommendation.json"
    recommendation_file.write_text('{"identity": 70, "segment_length": 12345}')
    recommendation_ref = ArtifactRef.from_path(
        recommendation_file, kind="pangenome_parameter_recommendation", format="json"
    )
    calls = []

    def recommend(genomes, **kwargs):
        calls.append(("recommend", kwargs))
        assert len(genomes) == 2
        return OrganelleResult(
            operation_id="pangenome.recommend_parameters",
            operation_version="1.0",
            scope="mitochondrion",
            status="ok",
            summary_text="fixture recommendation",
            artifacts=(recommendation_ref,),
            metrics={"recommendation": {"identity": 70, "segment_length": 12345}},
            provenance=make_provenance(
                operation_id="pangenome.recommend_parameters",
                parameters={},
                requested_backend="mash",
                actual_backend="mash",
                attempted_backends=("mash",),
            ),
        )

    def build(genomes, **kwargs):
        calls.append(("build", kwargs))
        assert kwargs["identity"] == 93 and kwargs["segment_length"] == 5000
        assert kwargs["auto_adopt_recommendation"] is automatic
        assert kwargs["recommendation"] == (
            {"identity": 70, "segment_length": 12345} if automatic else None
        )
        return OrganelleResult(
            operation_id="pangenome.build_graph",
            operation_version="1.0",
            scope="mitochondrion",
            status="ok",
            summary_text="fixture graph",
            artifacts=(ArtifactRef.from_path(gfa, kind="pangenome_graph", format="gfa"),),
            flags=("graph_built",),
            provenance=make_provenance(
                operation_id="pangenome.build_graph",
                parameters={},
                requested_backend="pggb",
                actual_backend="pggb",
                attempted_backends=("pggb",),
            ),
        )

    monkeypatch.setattr(
        workflow,
        "_execute_stage",
        lambda *args: StageOutcome(*workflow._execute_stage_body(*args), None),
    )
    monkeypatch.setattr(tuning_service, "recommend_parameters", recommend)
    monkeypatch.setattr(service, "build_graph", build)
    request = WorkflowRequest(
        dataset_path=str(dataset),
        backend="pggb",
        recommend_parameters=True,
        auto_adopt_recommendation=automatic,
        run_repeatmasker=True,
        repeatmasker_species="Oryza",
        identity=93,
        formats=("svg",),
    )
    record = workflow.create_run(request)
    original = workflow._execute_stage

    def fail_validate(name, *args):
        if name == "validate":
            raise RuntimeError("interrupted validation")
        return original(name, *args)

    monkeypatch.setattr(workflow, "_execute_stage", fail_validate)
    assert workflow.execute_run(record["run_id"])["status"] == "failed"
    monkeypatch.setattr(workflow, "_execute_stage", original)
    finished = workflow.execute_run(record["run_id"])
    assert finished["status"] == "succeeded", finished
    assert [name for name, _kwargs in calls] == ["recommend", "build"]
    assert calls[0][1] == {"threads": 4, "run_repeatmasker": True, "species": "Oryza"}
    directory = directory_for(record["run_id"])
    evidence = json.loads((directory / "recommendation.json").read_text())
    assert evidence["adopted"] is False
    assert evidence["graph_parameters"] == {"identity": 93, "segment_length": 5000}
    assert "baseline prior" in evidence["source"]
    if automatic:
        adopted = json.loads((directory / "parameter-adoption.json").read_text())
        assert adopted["adopted"] is True
        assert adopted["graph_parameters"] == {"identity": 70, "segment_length": 12345}
    result = json.loads((directory / "result.json").read_text())
    assert all(artifact["sha256"] != recommendation_ref.sha256 for artifact in result["artifacts"])
