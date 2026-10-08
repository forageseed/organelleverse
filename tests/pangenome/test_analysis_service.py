"""Canonical graph analysis is discoverable, admitted, and scientifically executed."""

from organelleverse.capabilities.admission import admit_capabilities
from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.registry import OperationRegistry
from organelleverse.pangenome.analysis_service import analyze_graph


def test_native_graph_analysis_uses_same_workflow(tmp_path):
    path = tmp_path / "graph.gfa"
    path.write_text("S\t1\tACGT\nP\ta#1#1\t1+\t*\nP\tb#1#1\t1+\t*\n")
    source = OrganelleResult(
        operation_id="pangenome.build_graph",
        scope="mitochondrion",
        status="ok",
        artifacts=(ArtifactRef.from_path(path, kind="pangenome_graph", format="gfa"),),
    )
    result = analyze_graph(source, bootstrap_replicates=3, core_threshold=0.95)
    assert result.status == "ok", result
    assert result.metrics["samples"] == 2
    assert result.metrics["core_threshold"] == 0.95
    import json

    metadata = next(a for a in result.artifacts if a.uri.endswith("pav-metadata.json"))
    assert json.loads(metadata.resolve().read_text())["core_threshold"] == 0.95
    assert any(a.uri.endswith("node_sample_pav.tsv") for a in result.artifacts)


def test_graph_analysis_capability_verifies(tmp_path, monkeypatch):
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    index = discover_capabilities()
    index.describe("pangenome.analyze_graph")
    store = VerificationStore(tmp_path / "home" / "verifications")
    verified = verify_capability(
        "pangenome.analyze_graph", store=store, environment=LocalVerificationEnvironment(index)
    )
    assert verified.capability_id == "pangenome.analyze_graph"
    admitted = admit_capabilities(index, store=store)
    entry = admitted.describe("pangenome.analyze_graph")
    assert entry.status is CapabilityStatus.ADMITTED, entry.diagnostic
    registry = OperationRegistry(capability_source=admitted.binding_source())
    graph = tmp_path / "native.gfa"
    graph.write_text("S\t1\tACGT\nP\ta#1#1\t1+\t*\nP\tb#1#1\t1+\t*\n")
    source = OrganelleResult(
        operation_id="pangenome.build_graph",
        scope="mitochondrion",
        status="ok",
        artifacts=(ArtifactRef.from_path(graph, kind="pangenome_graph", format="gfa"),),
    )
    result = registry.invoke(
        "pangenome.analyze_graph", input=source, parameters={"bootstrap_replicates": 0}
    )
    assert result.status == "ok", result
    assert result.metrics["samples"] == 2
    assert result.provenance.input_object_ids == (source.object_id,)


def test_analysis_parameters_share_annotation_and_adaptive_tree_contract(tmp_path):
    import json

    graph = tmp_path / "graph.gfa"
    graph.write_text("S\t1\tACGT\nP\ta#1#1\t1+\t*\nP\tb#1#1\t1+\t*\n")
    bed = tmp_path / "input.bed"
    bed.write_text("a#1#1\t0\t4\tNAD1\t0\t+\n")
    annotation = ArtifactRef.from_path(bed, kind="annotation", format="bed")
    source = OrganelleResult(
        operation_id="pangenome.build_graph",
        scope="mitochondrion",
        status="ok",
        artifacts=(ArtifactRef.from_path(graph, kind="pangenome_graph", format="gfa"), annotation),
    )
    result = analyze_graph(
        source,
        annotation_artifact_id=annotation.object_id,
        gene_synonyms={"NAD1": "nad1"},
        expected_genes=["nad1", "cox1"],
        bootstrap_method="adaptive_pav",
        adaptive_min_replicates=10,
        adaptive_max_replicates=20,
        adaptive_batch_size=5,
        generate_overview=False,
        formats=["svg"],
    )
    assert result.status == "ok", result
    assert annotation.sha256 in result.provenance.input_artifact_hashes
    recorded = json.loads(
        next(a for a in result.artifacts if a.uri.endswith("workflow.json")).resolve().read_text()
    )
    assert recorded["request"]["gene_synonyms"] == {"NAD1": "nad1"}
    data = json.loads(
        next(a for a in result.artifacts if a.uri.endswith("annotation.json")).resolve().read_text()
    )
    assert data["genes"] == ["nad1"]
    assert data["annotation_audit"]


def test_analysis_rejects_unrecorded_annotation_paths(tmp_path):
    import pytest

    from organelleverse.core.errors import OrganelleInputError

    graph = tmp_path / "graph.gfa"
    graph.write_text("S\t1\tACGT\nP\ta\t1+\t*\n")
    source = OrganelleResult(
        operation_id="pangenome.build_graph",
        scope="mitochondrion",
        status="ok",
        artifacts=(ArtifactRef.from_path(graph, kind="pangenome_graph", format="gfa"),),
    )
    with pytest.raises(OrganelleInputError, match="existing Result artifact"):
        analyze_graph(source, annotation_artifact_id=str(tmp_path / "unrecorded.bed"))
