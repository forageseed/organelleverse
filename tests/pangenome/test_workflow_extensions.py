"""Integration invariants for explicit coordinates and columnar reports."""

import csv
import json
from pathlib import Path

from Bio import SeqIO

from organelleverse.pangenome import workflow
from organelleverse.pangenome.pav_store import read_pav_page, write_node_pav
from organelleverse.pangenome.stored_report import write_stored_report
from organelleverse.pangenome.workflow_inputs import prepare_inputs
from organelleverse.pangenome.workflow_models import WorkflowRequest
from organelleverse.pangenome.workflow_science import annotation_stage
from organelleverse.pangenome.workflow_store import directory_for


def test_source_coordinates_transform_with_sequence_and_remain_auditable(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.fa").write_text(">original\nAACCGT\n")
    (data / "b.fa").write_text(">other\nAACCGT\n")
    bed = tmp_path / "genes.bed"
    bed.write_text("a#1#1\t0\t2\tNAD1\t0\t+\n")
    request = WorkflowRequest(
        dataset_path=str(data),
        annotations_path=str(bed),
        normalization={"a#1#1": {"orientation": "-", "origin": 1}},
        molecule_topologies={"a#1#1": "circular"},
        gene_synonyms={"NAD1": "nad1"},
        expected_genes=("nad1", "cox1"),
    )
    directory = tmp_path / "run"
    directory.mkdir()
    prepare_inputs(request, directory)
    manifest = json.loads((directory / "inputs.json").read_text())
    transformed = str(next(SeqIO.parse(manifest["samples"][0]["fasta"], "fasta")).seq)
    assert transformed == "CGGTTA"
    assert manifest["coordinate_mappings"]
    graph = tmp_path / "normalized.gfa"
    graph.write_text(f"S\t1\t{transformed}\nS\t2\tAACCGT\nP\ta#1#1\t1+\t*\nP\tb#1#1\t2+\t*\n")
    outputs, note = annotation_stage(graph, directory, request)
    assert note is None
    result = json.loads(outputs[0].read_text())
    arrow = result["gene_arrows"][0]
    assert (arrow["gene"], arrow["start"], arrow["end"], arrow["strand"]) == ("nad1", 3, 5, "-")
    assert result["annotation_audit"]
    assert (data / "a.fa").read_text().endswith("AACCGT\n")


def test_bounded_preview_keeps_all_nodes_in_frequency_statistics(tmp_path):
    graph = tmp_path / "graph.gfa"
    # 501 disconnected real nodes; two paths can traverse the same singleton.
    graph.write_text(
        "".join(f"S\t{index}\tA\n" for index in range(501)) + "P\ta\t0+\t*\nP\tb\t0+\t*\n"
    )
    store = tmp_path / "store"
    write_node_pav(graph, store)
    report = store / "report"
    files = write_stored_report(store, report)
    policy = json.loads((report / "matrix-preview.json").read_text())
    assert (policy["shown_nodes"], policy["total_nodes"]) == (500, 501)
    with (report / "node_frequency.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 501
    assert sum(float(row["frequency"]) for row in rows) == 1
    assert {row["class"] for row in rows} == {"core", "unobserved"}
    assert len(read_pav_page(store, offset=500)["rows"]) == 1
    assert all(Path(path).exists() for path in files)
    assert not (store / "pav.json").exists()


def test_new_workflow_publishes_columnar_tables_bubbles_and_tree_mode(tmp_path):
    graph = tmp_path / "graph.gfa"
    graph.write_text(
        "S\t1\tAC\nS\t2\tG\nS\t3\tT\nS\t4\tCA\n"
        "L\t1\t+\t2\t+\t0M\nL\t1\t+\t3\t+\t0M\n"
        "L\t2\t+\t4\t+\t0M\nL\t3\t+\t4\t+\t0M\n"
        "P\ta\t1+,2+,4+\t0M,0M\nP\tb\t1+,3+,4+\t0M,0M\n"
    )
    request = WorkflowRequest(
        backend="existing",
        gfa_path=str(graph),
        tree_mode="none",
        generate_overview=False,
        formats=("svg",),
    )
    created = workflow.create_run(request)
    result = workflow.execute_run(created["run_id"])
    assert result["status"] == "succeeded", result
    directory = directory_for(created["run_id"])
    assert (directory / "pav-metadata.json").exists()
    assert (directory / "superbubbles.json").exists()
    assert not (directory / "tree.json").exists()
    assert result["parameters"]["tree_mode"] == "none"
    assert result["summary"]["samples"] == 2
    assert workflow.execute_run(created["run_id"])["status"] == "succeeded"


def test_legacy_unfinished_checkpoints_require_a_new_analysis_without_false_resume(tmp_path):
    from organelleverse.pangenome import workflow_store

    graph = tmp_path / "graph.gfa"
    graph.write_text("S\t1\tAC\nP\ta\t1+\t*\n")
    created = workflow.create_run(WorkflowRequest(backend="existing", gfa_path=str(graph)))
    record = workflow_store.load(created["run_id"])
    del record["workflow_version"]
    workflow_store.save(directory_for(created["run_id"]), record)
    resumed = workflow.execute_run(created["run_id"])
    assert resumed["status"] == "failed"
    assert "older checkpoint format" in resumed["error"]
    assert all(stage["status"] == "pending" for stage in resumed["stages"])


def test_failed_stage_retains_actual_child_resource_measurement(tmp_path):
    import os

    import pytest

    from organelleverse.pangenome.stage_worker import StageFailure, execute_stage

    directory = tmp_path / "run"
    directory.mkdir()
    request = WorkflowRequest(dataset_path=str(tmp_path / "missing-dataset"))
    with pytest.raises(StageFailure) as caught:
        execute_stage("prepare", directory, request)
    assert caught.value.resources["wall_seconds"] > 0
    if hasattr(os, "wait4"):
        assert caught.value.resources["cpu_seconds"] > 0


def test_stage_thread_limit_is_applied_before_numerical_library_import(tmp_path):
    from organelleverse.pangenome.stage_worker import execute_stage

    directory = tmp_path / "run"
    directory.mkdir()
    graph = tmp_path / "input.gfa"
    graph.write_text("S\t1\tAC\n")
    outcome = execute_stage(
        "prepare", directory, WorkflowRequest(backend="existing", gfa_path=str(graph), threads=2)
    )
    assert outcome.resources["numerical_thread_limits"] == {
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
    }


def test_controlled_runner_environment_override_is_child_local(tmp_path, monkeypatch):
    import os
    import sys

    from organelleverse.pangenome._runner import run_command

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    output = tmp_path / "threads.txt"
    result = run_command(
        [sys.executable, "-c", "import os; print(os.environ['OPENBLAS_NUM_THREADS'])"],
        stdout_path=output,
        env_overrides={"OPENBLAS_NUM_THREADS": "2"},
    )
    assert result.ok and output.read_text().strip() == "2"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "7"


def test_valid_gfa_with_changed_input_sequence_fails_reconstruction_check(tmp_path):
    import pytest

    from organelleverse.pangenome.workflow_inputs import validate_project_paths

    data = tmp_path / "data"
    data.mkdir()
    for name in ("a", "b"):
        (data / f"{name}.fa").write_text(">molecule\nACGT\n")
    directory = tmp_path / "run"
    directory.mkdir()
    prepare_inputs(WorkflowRequest(dataset_path=str(data)), directory)
    manifest = json.loads((directory / "inputs.json").read_text())
    graph = tmp_path / "graph.gfa"
    graph.write_text("S\t1\tACGA\nP\ta#1#1\t1+\t*\nP\tb#1#1\t1+\t*\n")
    with pytest.raises(ValueError, match="exactly reconstruct"):
        validate_project_paths(graph, manifest)
    graph.write_text("S\t1\tACGT\nP\ta#1#1\t1+\t*\n")
    with pytest.raises(ValueError, match="path identities differ"):
        validate_project_paths(graph, manifest)
    graph.write_text("S\t1\tACGT\nP\ta#1#1\t1+\t*\nP\tb#1#1\t1+\t*\n")
    assert validate_project_paths(graph, manifest)["molecules"] == 2


def test_resume_rejects_changed_scientific_parameters(tmp_path, monkeypatch):
    from organelleverse.pangenome import workflow_store
    graph = tmp_path / "graph.gfa"
    graph.write_text("S\t1\tACGT\nP\ta\t1+\t*\nP\tb\t1+\t*\n")
    created = workflow.create_run(WorkflowRequest(backend="existing", gfa_path=str(graph), generate_overview=False, formats=("svg",)))
    original = workflow._execute_stage
    def interrupt(name, *args):
        if name == "phylogeny":
            raise RuntimeError("test interruption")
        return original(name, *args)
    monkeypatch.setattr(workflow, "_execute_stage", interrupt)
    assert workflow.execute_run(created["run_id"])["status"] == "failed"
    record = workflow_store.load(created["run_id"])
    record["request"]["cloud_threshold"] = 0.2
    workflow_store.save(directory_for(created["run_id"]), record)
    monkeypatch.setattr(workflow, "_execute_stage", original)
    resumed = workflow.execute_run(created["run_id"])
    assert resumed["status"] == "failed"
    assert "parameters changed" in resumed["error"]
