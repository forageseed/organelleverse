import json

from organelleverse.pangenome import workflow, workflow_store
from organelleverse.pangenome.workflow_models import WorkflowRequest
from organelleverse.results import build_result_dag


def test_native_workflow_links_original_inputs_and_outputs_without_mutating_store(tmp_path):
    graph = tmp_path / "input.gfa"
    graph.write_text("S\t1\tACGT\nP\ta\t1+\t*\n")
    record = workflow.create_run(
        WorkflowRequest(
            backend="existing",
            gfa_path=str(graph),
            formats=("svg",),
            generate_overview=False,
            tree_mode="none",
        )
    )
    assert workflow.execute_run(record["run_id"])["status"] == "succeeded"
    directory = workflow_store.directory_for(record["run_id"])
    before = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    result = json.loads((directory / "result.json").read_text())
    dag = build_result_dag(
        history_root=tmp_path / "desktop-state", native_runs=workflow_store.dag_runs()
    )
    producer = f"run:{record['run_id']}"
    assert (
        next(node for node in dag.nodes if node.node_id == producer).label == "pangenome.workflow"
    )
    expected = {f"sha256:{digest}" for digest in result["provenance"]["input_artifact_hashes"]}
    assert {
        edge.source for edge in dag.edges if edge.target == producer and edge.kind == "consumed"
    } == expected
    assert any(edge.source == producer and edge.kind == "produced" for edge in dag.edges)
    assert all(node.path is None for node in dag.nodes)
    assert all(path.read_bytes() == content for path, content in before.items())
