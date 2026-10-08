"""Exact gene selectors backed by verified graph-coordinate annotation evidence."""

from __future__ import annotations

import json
from pathlib import Path

from ..core.artifacts import ArtifactRef
from ..core.errors import OrganelleInputError
from ..core.result import OrganelleResult
from .graph import path_steps


def verified_artifact(artifact: ArtifactRef) -> Path:
    path = artifact.resolve()
    actual = ArtifactRef.from_path(path, kind=artifact.kind, format=artifact.format)
    if (actual.sha256, actual.size_bytes) != (artifact.sha256, artifact.size_bytes):
        raise ValueError("Artifact bytes changed after capture")
    return path


def gene_nodes(
    result: OrganelleResult, graph: ArtifactRef, names: list[str]
) -> tuple[list[str], ArtifactRef]:
    """Resolve exact, case-sensitive names; validate every selected projected span.

    Evidence must be an annotation artifact in this Result, carrying the graph's
    captured hash and size. No gene synonyms or homology are inferred here.
    """
    try:
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("gene_names must contain nonempty exact gene names")
        candidates = [
            a
            for a in result.artifacts
            if a.format == "json"
            and (a.kind == "pangenome_annotation" or Path(a.uri).name == "annotation.json")
        ]
        if len(candidates) != 1:
            raise ValueError("Exactly one graph annotation evidence artifact is required")
        artifact = candidates[0]
        payload = json.loads(verified_artifact(artifact).read_text())
        reference = ArtifactRef.model_validate(payload["graph_ref"])
        if reference.format != "gfa" or (reference.sha256, reference.size_bytes) != (
            graph.sha256,
            graph.size_bytes,
        ):
            raise ValueError("Annotation graph_ref does not match the selected graph")
        steps = path_steps(graph.resolve())
        wanted = set(names)
        found = set()
        nodes = set()
        for row in payload["projection"]:
            if row["gene"] not in wanted:
                continue
            index = row["step"]
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("Invalid projected step index")
            step = steps[row["path"]][index]
            if row["node"] != step["node"] or row["orientation"] != step["orientation"]:
                raise ValueError("Projected node/orientation does not match its graph path step")
            start, end, ns, ne = (
                row[key] for key in ("path_start", "path_end", "node_start", "node_end")
            )
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end, ns, ne)):
                raise ValueError("Projected coordinates must be integers")
            if not step["start"] <= start < end <= step["end"]:
                raise ValueError("Projected interval falls outside its graph node")
            expected = (
                (start - step["start"], end - step["start"])
                if step["orientation"] == "+"
                else (step["end"] - end, step["end"] - start)
            )
            if (ns, ne) != expected:
                raise ValueError("Projected node coordinates disagree with path coordinates")
            found.add(row["gene"])
            nodes.add(row["node"])
        if wanted != found:
            raise ValueError(
                f"Requested genes have no projected graph nodes: {sorted(wanted - found)}"
            )
        return sorted(nodes), artifact
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        raise OrganelleInputError(
            code="pangenome.invalid_gene_selection", message=str(error)
        ) from error
