"""The Result DAG: one read-only graph over every producer store (spec §2).

Nodes are content (keyed by sha256, R3) or producers (runs, revisions,
experiments, conversations). Edges follow data movement: consumed
(content → producer), produced (producer → content), derived_from (revision →
parent revision), cites (verified/evidence references). Assembly never
mutates a store, and the projection carries only relative addresses (R7).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from organelleverse.operations.spec import StrictSpecModel

from .paths import relative_address

if TYPE_CHECKING:
    from typing import Any as ConversationRecord
    from organelleverse.core.result import OrganelleResult
    from organelleverse.plugin_experiments import ExperimentRecord
    from organelleverse.plugin_runs import PluginRunRecord
    from organelleverse.revisions.models import ArtifactRevision

DagNodeKind = Literal[
    "content",
    "run",
    "revision",
    "experiment",
    "conversation",
]
DagEdgeKind = Literal["consumed", "produced", "derived_from", "cites"]


class DagNode(StrictSpecModel):
    node_id: str
    kind: DagNodeKind
    label: str = ""
    status: str = ""
    """Canonical relative address (R7); null when bytes sit outside the state."""
    path: str | None = None
    formats: tuple[str, ...] = ()
    size_bytes: int | None = None
    created_at: str | None = None


class DagEdge(StrictSpecModel):
    edge_id: str
    kind: DagEdgeKind
    source: str
    target: str


class ResultDag(StrictSpecModel):
    nodes: tuple[DagNode, ...] = ()
    edges: tuple[DagEdge, ...] = ()


@dataclass(frozen=True)
class NativeRun:
    """Read-only snapshot of an existing native producer; no second run store."""

    run_id: str
    operation_id: str
    status: str
    created_at: str | None = None
    result: OrganelleResult | None = None


def _sha_node_id(digest: str) -> str:
    return f"sha256:{digest}"


class _Builder:
    def __init__(self, history_root: Path) -> None:
        self._root = history_root
        self._nodes: dict[str, DagNode] = {}
        self._edges: dict[str, DagEdge] = {}

    # -- node helpers ------------------------------------------------------

    def _node(self, node: DagNode) -> None:
        existing = self._nodes.get(node.node_id)
        if existing is None:
            self._nodes[node.node_id] = node
            return
        # Content merges: first registration wins on path; kinds/formats union.
        self._nodes[node.node_id] = existing.model_copy(
            update={
                "formats": tuple(dict.fromkeys((*existing.formats, *node.formats))),
                "label": existing.label or node.label,
            }
        )

    def _content(
        self,
        digest: str,
        *,
        path: str | None,
        format_name: str = "",
        label: str = "",
        size_bytes: int | None = None,
    ) -> str:
        node_id = _sha_node_id(digest)
        self._node(
            DagNode(
                node_id=node_id,
                kind="content",
                label=label,
                path=path,
                formats=(format_name,) if format_name else (),
                size_bytes=size_bytes,
            )
        )
        return node_id

    def _producer(
        self,
        node_id: str,
        *,
        kind: DagNodeKind,
        label: str,
        status: str = "",
        created_at: str | None = None,
    ) -> str:
        self._node(
            DagNode(node_id=node_id, kind=kind, label=label, status=status, created_at=created_at)
        )
        return node_id

    def _edge(self, kind: DagEdgeKind, source: str, target: str) -> None:
        edge_id = f"{kind}:{source}->{target}"
        self._edges[edge_id] = DagEdge(edge_id=edge_id, kind=kind, source=source, target=target)

    # -- assembly per producer kind ----------------------------------------

    def add_run(self, record: PluginRunRecord) -> None:
        node_id = self._producer(
            f"run:{record.run_id}",
            kind="run",
            label=record.capability_id,
            status=record.status,
            created_at=record.submitted_at.isoformat() if record.submitted_at else None,
        )
        result = record.result
        if result is None:
            return
        for artifact in result.artifacts:
            content = self._content(
                artifact.sha256,
                path=relative_address(artifact.uri, self._root),
                format_name=artifact.format,
                size_bytes=artifact.size_bytes,
            )
            self._edge("produced", node_id, content)
        provenance = result.provenance
        if provenance is None:
            return
        for digest in provenance.input_artifact_hashes:
            # An input the state never registered still gets its node (with
            # no address) so the dependency edge survives instead of being
            # silently dropped.
            content = self._content(digest, path=None)
            self._edge("consumed", content, node_id)

    def add_native_run(self, record: NativeRun) -> None:
        node_id = self._producer(
            f"run:{record.run_id}",
            kind="run",
            label=record.operation_id,
            status=record.status,
            created_at=record.created_at,
        )
        if record.result is None:
            return
        for artifact in record.result.artifacts:
            content = self._content(
                artifact.sha256,
                path=relative_address(artifact.uri, self._root),
                format_name=artifact.format,
                size_bytes=artifact.size_bytes,
            )
            self._edge("produced", node_id, content)
        if record.result.provenance is not None:
            for digest in record.result.provenance.input_artifact_hashes:
                content = self._content(digest, path=None)
                self._edge("consumed", content, node_id)

    def add_revision(self, revision: ArtifactRevision) -> None:
        # When the revision records the operation that produced it, the
        # producer node's label carries that operation name: the graph reads
        # as a Galaxy-style tool history (e.g. "img-… · gaussian_filter")
        # without needing a new DagNode field.
        label = (
            f"{revision.artifact_id} · {revision.operation}"
            if revision.operation
            else f"{revision.artifact_id} · {revision.kind}"
        )
        node_id = self._producer(
            f"revision:{revision.revision_id}",
            kind="revision",
            label=label,
            created_at=revision.created_at.isoformat(),
        )
        ref = revision.content_ref
        content = self._content(
            ref.sha256,
            path=relative_address(ref.uri, self._root),
            format_name=ref.format,
            size_bytes=ref.size_bytes,
        )
        self._edge("produced", node_id, content)
        for link in revision.evidence_links:
            if link.kind == "run":
                self._edge("cites", node_id, f"run:{link.target_id}")
            elif link.kind == "artifact":
                self._edge("cites", node_id, f"sha256:{link.target_id}")

    def add_experiment(self, record: ExperimentRecord) -> None:
        self._producer(
            f"experiment:{record.experiment_id}",
            kind="experiment",
            label=record.capability_id,
            status=record.status,
        )

    def add_conversation(self, record: ConversationRecord) -> None:
        node_id = self._producer(
            f"conversation:{record.conversation_id}",
            kind="conversation",
            label=record.title or record.conversation_id,
            status=record.status,
        )
        for turn in record.turns:
            for run_id in turn.verified_run_ids:
                target = f"experiment:{run_id}" if run_id.startswith("exp-") else f"run:{run_id}"
                self._edge("cites", node_id, target)

    def link_revision_parents(self, revisions: Iterable[ArtifactRevision]) -> None:
        """derived_from edges for revision chains (the image-op history)."""
        by_revision = {revision.revision_id: revision for revision in revisions}
        for revision in revisions:
            parent_id = revision.parent_revision_id
            if parent_id is None or parent_id not in by_revision:
                continue
            self._edge(
                "derived_from",
                f"revision:{revision.revision_id}",
                f"revision:{parent_id}",
            )

    def build(self) -> ResultDag:
        return ResultDag(
            nodes=tuple(sorted(self._nodes.values(), key=lambda node: node.node_id)),
            edges=tuple(sorted(self._edges.values(), key=lambda edge: edge.edge_id)),
        )


def build_result_dag(
    *,
    history_root: Path,
    runs: Sequence[PluginRunRecord] = (),
    native_runs: Sequence[NativeRun] = (),
    revisions: Sequence[ArtifactRevision] = (),
    experiments: Sequence[ExperimentRecord] = (),
    conversations: Sequence[ConversationRecord] = (),
) -> ResultDag:
    """Assemble the graph read-only from store snapshots."""
    builder = _Builder(history_root)
    for record in runs:
        builder.add_run(record)
    for record in native_runs:
        builder.add_native_run(record)
    for revision in revisions:
        builder.add_revision(revision)
    builder.link_revision_parents(revisions)
    for record in experiments:
        builder.add_experiment(record)
    for record in conversations:
        builder.add_conversation(record)
    return builder.build()


__all__ = ["DagEdge", "DagNode", "ResultDag", "build_result_dag"]
