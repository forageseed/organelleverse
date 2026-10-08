"""Canonical graph consumers must be admitted and preserve scientific lineage."""

import os
from pathlib import Path

import pytest

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.result import ErrorDetail, OrganelleResult
from organelleverse.operations.adapters.json import invoke_json
from organelleverse.operations.registry import OperationRegistry
from organelleverse.pangenome.graph import load_gfa, path_sequences
from organelleverse.pangenome.graph_operations import convert_graph_result, extract_subgraph


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.gfa"
    path.write_text(
        "S\t1\tACGT\nS\t2\tAA\nS\t3\tTT\nL\t1\t+\t2\t+\t0M\nL\t1\t+\t3\t+\t0M\nP\ta#1#1\t1+,2+\t*\nP\tb#1#1\t1+,3+\t*\n"
    )
    return OrganelleResult(
        operation_id="pangenome.build_graph",
        scope="plastid",
        status="ok",
        artifacts=(ArtifactRef.from_path(path, kind="pangenome_graph", format="gfa"),),
    )


@pytest.fixture
def registry(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    index = discover_capabilities()
    store = VerificationStore(home / "verifications")
    for operation in ["pangenome.convert_graph_result", "pangenome.extract_subgraph"]:
        verify_capability(operation, store=store, environment=LocalVerificationEnvironment(index))
    admitted = discover_capabilities()
    for operation in ["pangenome.convert_graph_result", "pangenome.extract_subgraph"]:
        entry = admitted.describe(operation)
        assert entry.status is CapabilityStatus.ADMITTED, entry.diagnostic
    registry = OperationRegistry()
    registry.attach_capability_source(admitted.binding_source())
    return registry


def test_both_graph_consumers_reach_admission(registry):
    assert registry is not None


def test_admitted_extraction_uses_real_subgraph_and_parent_lineage(registry, source):
    response = invoke_json(
        {
            "operation_id": "pangenome.extract_subgraph",
            "input": source.model_dump(mode="json"),
            "parameters": {"path_names": ["a#1#1"]},
        },
        registry=registry,
        granted_side_effects=["read_files", "write_files"],
    )
    assert response["ok"], response.get("error")
    result = response["result"]
    if not isinstance(result, OrganelleResult):
        result = OrganelleResult.model_validate(result)
    assert result.scope == "plastid"
    assert result.operation_id == "pangenome.extract_subgraph"
    assert result.provenance.input_object_ids == (source.object_id,)
    graph = next(a for a in result.artifacts if a.kind == "pangenome_graph")
    assert path_sequences(graph.uri) == {"a#1#1": "ACGTAA"}
    assert list(load_gfa(graph.uri).segments) == ["1", "2"]


def test_exact_noop_reuses_source_reference(source):
    result = extract_subgraph(source, path_names=["a#1#1", "b#1#1"])
    assert result.flags == ("graph_reused",)
    assert result.artifacts == source.artifacts
    assert result.provenance.input_object_ids == (source.object_id,)


def test_interval_and_node_selection_keep_whole_nodes(source):
    result = extract_subgraph(source, interval_path="a#1#1", interval_start=4, interval_end=5)
    graph = next(a for a in result.artifacts if a.kind == "pangenome_graph")
    assert list(load_gfa(graph.uri).segments) == ["2"]
    assert load_gfa(graph.uri).paths == {}
    with pytest.raises(OrganelleInputError, match="together"):
        extract_subgraph(source, interval_path="a#1#1")
    with pytest.raises(OrganelleInputError, match="integers"):
        extract_subgraph(source, interval_path="a#1#1", interval_start=True, interval_end=2)


def test_failed_unknown_scope_changed_inputs_are_rejected(source):
    for invalid in [
        source.model_copy(update={"scope": "mixed"}),
        source.model_copy(
            update={
                "status": "failed",
                "errors": (ErrorDetail(code="test.failed", message="failed graph"),),
            }
        ),
    ]:
        with pytest.raises(OrganelleInputError):
            extract_subgraph(invalid, node_ids=["1"])
    source.artifacts[0].resolve().write_text("S\t1\tAAAA\n")
    with pytest.raises(OrganelleInputError, match="changed"):
        convert_graph_result(source, target_format="og")


@pytest.mark.skipif(
    os.environ.get("ORGANELLEVERSE_TEST_GRAPH_CONVERTERS") != "1",
    reason="explicit real backend opt-in",
)
def test_admitted_real_conversion_retains_parent_scope_and_evidence(registry, source):
    response = invoke_json(
        {
            "operation_id": "pangenome.convert_graph_result",
            "input": source.model_dump(mode="json"),
            "parameters": {"target_format": "og", "threads": 1},
        },
        registry=registry,
        granted_side_effects=["read_files", "write_files", "subprocess"],
    )
    assert response["ok"], response.get("error")
    converted = response["result"]
    if not isinstance(converted, OrganelleResult):
        converted = OrganelleResult.model_validate(converted)
    assert converted.scope == "plastid"
    assert converted.operation_id == "pangenome.convert_graph_result"
    assert converted.provenance.input_object_ids == (source.object_id,)
    assert converted.provenance.actual_backend == "odgi"
    assert any(a.format == "og" and Path(a.uri).is_file() for a in converted.artifacts)
    restored = convert_graph_result(converted, target_format="gfa")
    graph = next(a for a in restored.artifacts if a.kind == "pangenome_graph")
    assert path_sequences(graph.uri) == path_sequences(source.artifacts[0].uri)
    assert restored.provenance.input_object_ids == (converted.object_id,)


def test_admitted_noop_reuses_original_without_duplicate_publication(registry, source):
    response = invoke_json(
        {
            "operation_id": "pangenome.extract_subgraph",
            "input": source.model_dump(mode="json"),
            "parameters": {"path_names": ["a#1#1", "b#1#1"]},
        },
        registry=registry,
        granted_side_effects=["read_files", "write_files"],
    )
    assert response["ok"], response.get("error")
    result = response["result"]
    if not isinstance(result, OrganelleResult):
        result = OrganelleResult.model_validate(result)
    assert result.artifacts == source.artifacts
    assert "graph_reused" in result.flags
