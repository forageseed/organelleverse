"""Output path grammar (spec §1) and the Result DAG (spec §2)."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from organelleverse.results import (
    build_result_dag,
    is_canonical,
    paths,
    relative_address,
)

# -- path grammar -----------------------------------------------------------


def test_builders_construct_the_canonical_layout() -> None:
    assert paths.run_receipt("run-1").as_posix() == "runs/run-1.json"
    assert (
        paths.run_artifact("run-1", "tree.newick").as_posix() == "runs/run-1/artifacts/tree.newick"
    )
    assert paths.experiment_record("exp-1").as_posix() == "experiments/exp-1.json"
    assert paths.revision_record("rev-abc").as_posix() == "revisions/rev-abc.json"
    assert paths.image_artifact("img-1", "png").as_posix() == "images/img-1.png"
    assert paths.image_child("img-1", "deadbeef", "png").as_posix() == "images/img-1-deadbeef.png"
    assert paths.notebook_record("nb-1").as_posix() == "notebooks/nb-1.json"
    assert paths.conversation_record("conv-1").as_posix() == "conversations/conv-1.json"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        ".",
        "..",
        "Up",
        "has space",
        "semi;colon",
        "-leading-dash",
        "a" * 130,
        "trailing/slash/",
    ],
)
def test_segments_reject_spec_violations(bad: str) -> None:
    with pytest.raises(paths.PathSpecError):
        paths.validate_segment(bad)


def test_image_ids_must_be_prefixed_and_hex8_is_pinned() -> None:
    with pytest.raises(paths.PathSpecError):
        paths.validate_image_id("not-prefixed")
    with pytest.raises(paths.PathSpecError):
        paths.validate_hex8("XYZ12345")
    with pytest.raises(paths.PathSpecError):
        paths.validate_suffix("PNG")


def test_is_canonical_owns_only_producer_roots() -> None:
    assert is_canonical("runs/run-1.json")
    assert is_canonical("images/img-1.png")
    assert not is_canonical("settings.json")  # single segment, no owner
    assert not is_canonical("../outside/run.json")
    assert not is_canonical("/absolute/path.json")
    assert not is_canonical("random-dir/file.txt")


def test_relative_address_hides_paths_outside_the_state_root(tmp_path: Path) -> None:
    history = tmp_path / "state"
    (history / "images").mkdir(parents=True)
    inside = history / "images" / "img-1.png"
    inside.write_bytes(b"x")
    assert relative_address(str(inside), history) == "images/img-1.png"
    # Legacy absolute addresses elsewhere on disk: no address, never a host path.
    assert relative_address(str(tmp_path / "elsewhere.bin"), history) is None
    assert relative_address(str(inside), tmp_path / "other-state") is None


# -- Result DAG -------------------------------------------------------------


def _revision(revision_id: str, sha: str, *, parent: str | None = None, uri: str = "") -> object:
    from organelleverse.core.artifacts import ArtifactRef
    from organelleverse.revisions.models import ArtifactRevision

    return ArtifactRevision(
        revision_id=revision_id,
        artifact_id="img-1",
        parent_revision_id=parent,
        kind="image",
        content_ref=ArtifactRef(
            kind="image",
            uri=uri or "memory:placeholder",
            format="png",
            sha256=sha,
            size_bytes=4,
        ),
        created_at=__import__("datetime").datetime(2026, 8, 15),
    )


def test_image_operation_chain_becomes_derived_from(tmp_path: Path) -> None:
    upload = _revision("rev-a", "a" * 64, uri=str(tmp_path / "images" / "img-1.png"))
    filtered = _revision("rev-b", "b" * 64, parent="rev-a")

    dag = build_result_dag(history_root=tmp_path, revisions=(upload, filtered))

    nodes = {node.node_id: node for node in dag.nodes}
    assert "revision:rev-a" in nodes and "revision:rev-b" in nodes
    assert nodes["sha256:" + "a" * 64].path == "images/img-1.png"
    edges = {edge.edge_id for edge in dag.edges}
    assert "derived_from:revision:rev-b->revision:rev-a" in edges
    assert "produced:revision:rev-a->sha256:" + "a" * 64 in edges


def test_dedupes_content_and_hides_external_addresses(tmp_path: Path) -> None:
    first = _revision("rev-a", "a" * 64, uri=str(tmp_path / "images" / "img-1.png"))
    second = _revision("rev-b", "a" * 64, parent="rev-a", uri="/elsewhere/img.png")

    dag = build_result_dag(history_root=tmp_path, revisions=(first, second))

    content = [node for node in dag.nodes if node.kind == "content"]
    assert len(content) == 1  # same sha, one node (R3)
    assert content[0].path == "images/img-1.png"  # first address wins; no host leak


def test_empty_state_is_an_empty_dag(tmp_path: Path) -> None:
    dag = build_result_dag(history_root=tmp_path)
    assert dag.nodes == () and dag.edges == ()
    assert isinstance(dag.model_dump(mode="json"), dict)


def _run_record(run_id: str, *, input_hashes: tuple[str, ...], output_sha: str) -> object:
    from datetime import UTC, datetime

    from organelleverse.capabilities.plugin_descriptor import PluginDescriptor
    from organelleverse.core.artifacts import ArtifactRef
    from organelleverse.core.provenance import ResultProvenance
    from organelleverse.core.result import OrganelleResult
    from organelleverse.plugin_runs import PluginRunRecord, PluginRunRequest

    return PluginRunRecord(
        run_id=run_id,
        capability_id="demo.experiment",
        contract_identity="contract-v1",
        status="succeeded",
        submitted_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        request=PluginRunRequest(capability_id="demo.experiment", inputs={}),
        descriptor=PluginDescriptor.model_validate(
            {
                "capability_id": "demo.experiment",
                "title": "Demo",
                "summary": "",
                "inputs": [],
                "outputs": [],
                "parameters": [],
                "agent_task_description": "",
                "agent_examples": [],
                "gui_page": None,
                "optimization": None,
            }
        ),
        result=OrganelleResult(
            operation_id="demo.experiment",
            scope="none",
            status="ok",
            artifacts=(
                ArtifactRef(
                    kind="tree",
                    uri="/workspace/tree.newick",
                    format="newick",
                    sha256=output_sha,
                    size_bytes=9,
                ),
            ),
            provenance=ResultProvenance(
                run_id="l6-1",
                operation_id="demo.experiment",
                operation_version="1.0",
                package_version="0.1.0",
                git_commit="",
                input_artifact_hashes=input_hashes,
                parameters_hash="0" * 64,
            ),
        ),
    )


def test_run_input_hashes_flow_through_the_producer(tmp_path: Path) -> None:
    genome_sha = "c" * 64
    tree_sha = "d" * 64
    run = _run_record("run-1", input_hashes=(genome_sha,), output_sha=tree_sha)

    dag = build_result_dag(history_root=tmp_path, runs=(run,))

    edges = {edge.edge_id for edge in dag.edges}
    assert f"consumed:sha256:{genome_sha}->run:run-1" in edges
    assert f"produced:run:run-1->sha256:{tree_sha}" in edges
    nodes = {node.node_id: node for node in dag.nodes}
    # Unknown-input nodes keep the edge alive with no address (R7/R3).
    assert nodes[f"sha256:{genome_sha}"].path is None
    assert nodes[f"sha256:{tree_sha}"].path is None  # workspace bytes, not state bytes
    assert nodes["run:run-1"].label == "demo.experiment"
