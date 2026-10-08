# ruff: noqa: F811
import json

import pytest

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.adapters.json import invoke_json
from organelleverse.pangenome.graph import load_gfa
from organelleverse.pangenome.graph_operations import extract_subgraph
from tests.pangenome.test_graph_operations import registry, source  # noqa: F401


def annotated(source, tmp_path, *, wrong_graph=False, wrong_node=False):
    graph = source.artifacts[0]
    ref = graph.model_dump(mode="json")
    if wrong_graph:
        ref["sha256"] = "0" * 64
    path = tmp_path / "annotation.json"
    path.write_text(
        json.dumps(
            {
                "graph_ref": ref,
                "projection": [
                    {
                        "gene": "rbcL",
                        "path": "a#1#1",
                        "step": 1,
                        "node": "3" if wrong_node else "2",
                        "orientation": "+",
                        "path_start": 4,
                        "path_end": 6,
                        "node_start": 0,
                        "node_end": 2,
                    }
                ],
            }
        )
    )
    annotation = ArtifactRef.from_path(path, kind="pangenome_annotation", format="json")
    pointer = tmp_path / "graph-reference.json"
    pointer.write_text(json.dumps({"graph_ref": graph.model_dump(mode="json")}))
    return source.model_copy(
        update={
            "artifacts": (
                ArtifactRef.from_path(pointer, kind="pangenome_graph_reference", format="json"),
                annotation,
            )
        }
    )


def test_exact_gene_selector_admitted_with_verified_pointer(registry, source, tmp_path):
    source = annotated(source, tmp_path)
    response = invoke_json(
        {
            "operation_id": "pangenome.extract_subgraph",
            "input": source.model_dump(mode="json"),
            "parameters": {"gene_names": ["rbcL"]},
        },
        registry=registry,
        granted_side_effects=["read_files", "write_files"],
    )
    assert response["ok"], response.get("error")
    result = response["result"]
    if not isinstance(result, OrganelleResult):
        result = OrganelleResult.model_validate(result)
    graph = next(a for a in result.artifacts if a.kind == "pangenome_graph")
    assert list(load_gfa(graph.resolve()).segments) == ["2"]
    assert set(a.sha256 for a in source.artifacts) <= set(result.provenance.input_artifact_hashes)
    assert result.provenance.input_object_ids == (source.object_id,)


@pytest.mark.parametrize(
    "change", ["graph", "node", "annotation_bytes", "pointer_bytes", "missing_gene", "case"]
)
def test_gene_evidence_rejects_mismatch_tamper_and_nonexact_names(source, tmp_path, change):
    source = annotated(source, tmp_path, wrong_graph=change == "graph", wrong_node=change == "node")
    if change == "annotation_bytes":
        source.artifacts[1].resolve().write_text("{}")
    if change == "pointer_bytes":
        source.artifacts[0].resolve().write_text("{}")
    names = ["RbcL"] if change == "case" else ["absent"] if change == "missing_gene" else ["rbcL"]
    with pytest.raises(OrganelleInputError):
        extract_subgraph(source, gene_names=names)


def test_gene_selection_requires_annotation(source):
    with pytest.raises(OrganelleInputError, match="annotation evidence"):
        extract_subgraph(source, gene_names=["rbcL"])


def test_gene_selector_unions_nodes(source, tmp_path):
    result = extract_subgraph(annotated(source, tmp_path), gene_names=["rbcL"], node_ids=["1"])
    graph = next(a for a in result.artifacts if a.kind == "pangenome_graph")
    assert list(load_gfa(graph.resolve()).segments) == ["1", "2"]
