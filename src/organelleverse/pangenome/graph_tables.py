"""Bounded path and exact superbubble summaries for shared graph browsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .graph import _steps, load_gfa


def path_page(
    gfa_path: str | Path, *, offset: int = 0, limit: int = 100, sample_name: str | None = None
) -> dict[str, Any]:
    if offset < 0 or not 1 <= limit <= 500:
        raise ValueError("Path pagination requires offset>=0 and limit in [1,500]")
    graph = load_gfa(gfa_path)
    if sample_name is not None and sample_name not in {
        record.sample for record in graph.paths.values()
    }:
        raise ValueError(f"Unknown biological sample: {sample_name}")
    selected = [
        record
        for record in graph.paths.values()
        if sample_name is None or record.sample == sample_name
    ]
    rows = []
    for record in selected[offset : offset + limit]:
        error = None
        try:
            spans = _steps(graph, record)
            length = max((step["end"] for step in spans), default=record.start or 0) - (
                record.start or 0
            )
        except ValueError as exception:
            length, error = None, str(exception)
        start = record.start if record.kind == "W" else 0
        rows.append(
            {
                "name": record.name,
                "sample": record.sample,
                "molecule": record.molecule,
                "steps": len(record.steps),
                "unique_nodes": len({node for node, _ in record.steps}),
                "length_bp": length,
                "start": start,
                "end": start + length if start is not None and length is not None else None,
                "coordinate_system": "molecule"
                if record.kind == "W" and start is not None
                else "path_local"
                if record.kind == "P"
                else "unknown_origin",
                "coordinate_error": error,
            }
        )
    return {
        "paths": rows,
        "total_paths": len(selected),
        "total_graph_paths": len(graph.paths),
        "offset": offset,
        "limit": limit,
    }


def bubble_page(
    data: dict[str, Any],
    *,
    offset: int = 0,
    limit: int = 100,
    sample_name: str | None = None,
    path_name: str | None = None,
) -> dict[str, Any]:
    """Bound list members; selection uses the complete backend record by ID."""
    if offset < 0 or not 1 <= limit <= 500:
        raise ValueError("Bubble pagination requires offset>=0 and limit in [1,500]")
    selected = [
        row
        for row in data["bubbles"]
        if (sample_name is None or sample_name in row["sample_names"])
        and (path_name is None or path_name in row["path_names"])
    ]
    rows = []
    for row in selected[offset : offset + limit]:
        summary = {
            key: value
            for key, value in row.items()
            if key not in {"oriented_nodes", "node_ids", "path_names", "sample_names"}
        }
        summary["node_count"] = len(row["node_ids"])
        for key in ("node_ids", "path_names", "sample_names"):
            summary[key] = row[key][:500]
            summary[f"{key}_total"] = len(row[key])
            summary[f"{key}_truncated"] = len(row[key]) > 500
        rows.append(summary)
    return {
        "bubbles": rows,
        "total_bubbles": len(selected),
        "total_graph_bubbles": data["total_bubbles"],
        "offset": offset,
        "limit": limit,
        "definition": data["definition"],
        "algorithm": data["algorithm"],
        "graph_model": data["graph_model"],
        "source_graph": data["source_graph"],
        "analysis_origin": data["analysis_origin"],
        "path_membership_semantics": data["path_membership_semantics"],
    }
