"""Canonical Result consumers for managed graph conversion and extraction."""

from __future__ import annotations

import json
from typing import Literal
from uuid import uuid4

from ..core.artifacts import ArtifactRef
from ..core.errors import OrganelleInputError
from ..core.result import OrganelleResult
from ..runtime import create_staged_run, managed_run_path, publish_staged_result
from ._contract import make_provenance, utc_now
from .graph import extract_subgraph as _extract_subgraph
from .graph import graph_statistics
from .graph_formats import convert_graph
from .graph_selection import gene_nodes, verified_artifact


def _graph_input(result: OrganelleResult, formats: set[str]) -> ArtifactRef:
    if result.scope not in {"mitochondrion", "plastid"}:
        raise OrganelleInputError(
            code="pangenome.scope_required",
            message="Graph operations require an explicit mitochondrion or plastid scope",
        )
    graphs = [
        artifact
        for artifact in result.artifacts
        if artifact.kind == "pangenome_graph" and artifact.format in formats
    ]
    references = [
        a for a in result.artifacts if a.kind == "pangenome_graph_reference" and a.format == "json"
    ]
    if not graphs and len(references) == 1:
        try:
            payload = json.loads(verified_artifact(references[0]).read_text())
            referenced = ArtifactRef.model_validate(payload["graph_ref"])
            if referenced.kind != "pangenome_graph" or referenced.format not in formats:
                raise ValueError("Graph reference has an unsupported kind or format")
            graphs = [referenced]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise OrganelleInputError(
                code="pangenome.invalid_graph_reference", message=str(error)
            ) from error
    elif references:
        raise OrganelleInputError(
            code="pangenome.ambiguous_graph_reference",
            message="Result contains ambiguous graph artifacts/references",
        )
    if result.status == "failed" or len(graphs) != 1:
        raise OrganelleInputError(
            code="pangenome.graph_required",
            message=f"Expected a non-failed Result with exactly one graph in {sorted(formats)}",
        )
    graph = graphs[0]
    try:
        actual = ArtifactRef.from_path(graph.resolve(), kind=graph.kind, format=graph.format)
    except OSError as error:
        raise OrganelleInputError(
            code="pangenome.graph_unavailable",
            message="Graph input cannot be read",
            details={"reason": str(error)},
        ) from error
    if actual.sha256 != graph.sha256 or actual.size_bytes != graph.size_bytes:
        raise OrganelleInputError(
            code="pangenome.graph_changed", message="Graph bytes changed after capture"
        )
    return graph


def convert_graph_result(
    result: OrganelleResult, *, target_format: Literal["gfa", "og", "vg"], threads: int = 1
) -> OrganelleResult:
    """Convert GFA→OG, OG→GFA or GFA→VG, retaining source Result lineage.

    Uses the same real-backend conversion and roundtrip validation as the Python
    artifact API. No output directory or executable override crosses the boundary.
    """
    graph = _graph_input(result, {"gfa", "og"})
    converted = convert_graph(graph, target_format=target_format, threads=threads)
    if converted.provenance is None:
        raise RuntimeError("Managed graph converter omitted provenance")
    return converted.model_copy(
        update={
            "operation_id": "pangenome.convert_graph_result",
            "scope": result.scope,
            "provenance": converted.provenance.model_copy(
                update={
                    "operation_id": "pangenome.convert_graph_result",
                    "operation_version": "1.0",
                    "input_object_ids": (result.object_id,),
                    "input_artifact_hashes": tuple(
                        dict.fromkeys(
                            [graph.sha256]
                            + [
                                a.sha256
                                for a in result.artifacts
                                if a.kind == "pangenome_graph_reference"
                            ]
                        )
                    ),
                }
            ),
        }
    )


def extract_subgraph(
    result: OrganelleResult,
    *,
    path_names: list[str] | None = None,
    node_ids: list[str] | None = None,
    gene_names: list[str] | None = None,
    interval_path: str | None = None,
    interval_start: int | None = None,
    interval_end: int | None = None,
) -> OrganelleResult:
    """Extract an induced graph by paths, nodes, or a half-open path interval.

    Intervals select complete intersecting nodes, not clipped nucleotide strings.
    Only complete traversals are retained. Gene names match exactly against this
    Result's verified annotation evidence; their projected nodes are selected.
    Multiple selector kinds are combined by union. An unchanged graph reuses its original
    ArtifactRef without copying trusted input bytes into a new output artifact.
    """
    graph = _graph_input(result, {"gfa"})
    annotation = None
    selected_nodes = list(node_ids or [])
    if gene_names is not None:
        projected, annotation = gene_nodes(result, graph, gene_names)
        selected_nodes = sorted(set(selected_nodes) | set(projected))
    input_hashes = tuple(
        dict.fromkeys(
            [graph.sha256]
            + [a.sha256 for a in result.artifacts if a.kind == "pangenome_graph_reference"]
            + ([annotation.sha256] if annotation else [])
        )
    )
    interval = None
    coordinates = (interval_path, interval_start, interval_end)
    if any(value is not None for value in coordinates):
        if any(value is None for value in coordinates):
            raise OrganelleInputError(
                code="pangenome.incomplete_interval",
                message="interval_path, interval_start and interval_end must be supplied together",
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (interval_start, interval_end)
        ):
            raise OrganelleInputError(
                code="pangenome.invalid_interval", message="Interval coordinates must be integers"
            )
        interval = (interval_path, interval_start, interval_end)
    started = utc_now()
    try:
        text = _extract_subgraph(
            graph.resolve(), path_names, node_names=selected_nodes or None, path_interval=interval
        )
    except ValueError as error:
        raise OrganelleInputError(
            code="pangenome.invalid_subgraph_selection", message=str(error)
        ) from error
    operation = "pangenome.extract_subgraph"
    parameters = {
        "path_names": path_names,
        "node_ids": node_ids,
        "gene_names": gene_names,
        "resolved_node_ids": selected_nodes,
        "interval_path": interval_path,
        "interval_start": interval_start,
        "interval_end": interval_end,
    }
    provenance = make_provenance(
        operation_id=operation,
        parameters=parameters,
        input_object_ids=(result.object_id,),
        input_artifact_hashes=input_hashes,
        requested_backend="organelleverse",
        actual_backend="organelleverse",
        attempted_backends=("organelleverse",),
        started_at=started,
        finished_at=utc_now(),
    ).model_copy(update={"operation_version": "1.0"})
    if text.encode("utf-8") == graph.resolve().read_bytes():
        stats = graph_statistics(graph.resolve())
        return OrganelleResult(
            operation_id=operation,
            operation_version="1.0",
            scope=result.scope,
            status="ok",
            summary_text="Selection retains the complete graph; the original graph reference is reused.",
            artifacts=(graph,),
            flags=("graph_reused",),
            metrics={"nodes": stats["nodes"], "paths": stats["paths"]},
            provenance=provenance,
        )
    run_id = uuid4().hex
    staging = create_staged_run(operation, run_id)
    output = staging / "subgraph.gfa"
    output.write_text(text, encoding="utf-8")
    stats = graph_statistics(output)
    evidence = staging / "selection.json"
    evidence.write_text(
        json.dumps(
            {
                "source_result_id": result.object_id,
                "source_graph": graph.model_dump(mode="json"),
                "annotation_evidence": annotation.model_dump(mode="json") if annotation else None,
                "selection": parameters,
                "interval_semantics": "half-open coordinates select complete intersecting nodes",
                "path_semantics": "only complete paths are retained",
                "statistics": stats,
            },
            indent=2,
        )
        + "\n"
    )
    artifacts = tuple(
        ArtifactRef.from_path(
            path,
            kind="pangenome_graph" if path == output else "subgraph_evidence",
            format=path.suffix.lstrip("."),
            media_type="text/plain" if path == output else "application/json",
        ).model_copy(update={"uri": path.name})
        for path in (output, evidence)
    )
    extracted = OrganelleResult(
        operation_id=operation,
        operation_version="1.0",
        scope=result.scope,
        status="ok",
        summary_text="Extracted a validated induced subgraph from the selected paths, nodes or interval.",
        artifacts=artifacts,
        flags=("subgraph_extracted",),
        metrics={"nodes": stats["nodes"], "edges": stats["edges"], "paths": stats["paths"]},
        provenance=provenance,
    )
    return publish_staged_result(
        extracted,
        staging,
        managed_run_path(operation, run_id),
        trusted_input_hashes=frozenset(input_hashes),
    )
