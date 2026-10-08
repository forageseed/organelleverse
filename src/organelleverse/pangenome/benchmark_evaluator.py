"""Independent, supervised conserved-base homology benchmark for path graphs.

This is a benchmark-specific measurement of graph representation, not a claim
that compactness, node count, or a tool's own score measures biological accuracy.
Ground-truth loci and complete input molecules are supplied in a versioned,
captured benchmark artifact independently of the graph constructor.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from ..core.artifacts import ArtifactRef
from ..core.errors import OrganelleInputError
from ..core.result import OrganelleResult
from ..runtime import create_staged_run, managed_run_path, publish_staged_result
from ._contract import make_provenance
from .graph import path_sequences, path_steps
from .graph_operations import _graph_input
from .graph_selection import verified_artifact


class Locus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(min_length=1)
    position: StrictInt = Field(ge=0)


class HomologyAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    first: Locus
    second: Locus
    homologous: StrictBool


class GraphBenchmarkTruth(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["organelleverse.pangenome.graph-truth.v1"]
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    case_id: str = Field(min_length=1)
    source_description: str = Field(min_length=1)
    sequences: dict[str, str] = Field(min_length=2)
    anchors: tuple[HomologyAnchor, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_truth(self) -> Self:
        seen = set()
        for anchor in self.anchors:
            loci = tuple(
                sorted(
                    (
                        (anchor.first.path, anchor.first.position),
                        (anchor.second.path, anchor.second.position),
                    )
                )
            )
            if loci in seen or loci[0] == loci[1]:
                raise ValueError("Benchmark pairs must be distinct and unique")
            seen.add(loci)
            bases = []
            for locus in (anchor.first, anchor.second):
                if locus.path not in self.sequences or locus.position >= len(
                    self.sequences[locus.path]
                ):
                    raise ValueError("Benchmark locus is outside its declared input molecule")
                base = self.sequences[locus.path][locus.position].upper()
                if base not in "ACGT":
                    raise ValueError("Homology anchors require unambiguous A/C/G/T bases")
                bases.append(base)
            if anchor.homologous and bases[0] not in {
                bases[1],
                bases[1].translate(str.maketrans("ACGT", "TGCA")),
            }:
                raise ValueError("Positive anchors must describe conserved homologous bases")
        if {anchor.homologous for anchor in self.anchors} != {False, True}:
            raise ValueError("The benchmark must contain positive and negative homology anchors")
        return self


def score_graph(graph_path: Path, truth: GraphBenchmarkTruth) -> dict:
    """Compare exact physical graph loci against independently supplied labels."""
    observed = path_sequences(graph_path)
    expected = {name: sequence.upper() for name, sequence in truth.sequences.items()}
    if {name: sequence.upper() for name, sequence in observed.items()} != expected:
        raise ValueError(
            "Benchmark graph must exactly reconstruct every declared molecule and identity"
        )
    paths = path_steps(graph_path)
    if any(row["overlap_previous"] for rows in paths.values() for row in rows):
        raise ValueError("Conserved-base homology benchmark requires zero-overlap graph paths")
    if any(rows and rows[0]["start"] != 0 for rows in paths.values()):
        raise ValueError("Benchmark graph must contain complete paths starting at zero")
    starts = {name: [row["start"] for row in rows] for name, rows in paths.items()}

    def locate(locus: Locus) -> tuple[str, int]:
        rows = paths[locus.path]
        index = bisect_right(starts[locus.path], locus.position) - 1
        row = rows[index]
        if not row["start"] <= locus.position < row["end"]:
            raise ValueError("Benchmark locus has no unique represented graph base")
        offset = locus.position - row["start"]
        return row["node"], offset if row["orientation"] == "+" else row["node_length"] - 1 - offset

    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    observations = []
    for anchor in truth.anchors:
        first, second = locate(anchor.first), locate(anchor.second)
        prediction = first == second
        category = (
            ("tp" if prediction else "fn") if anchor.homologous else ("fp" if prediction else "tn")
        )
        counts[category] += 1
        observations.append(
            {
                **anchor.model_dump(mode="json"),
                "first_graph_locus": first,
                "second_graph_locus": second,
                "predicted_homologous": prediction,
                "category": category,
            }
        )
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    # Zero precision is the declared convention when no homology is predicted.
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    return {
        "benchmark_id": truth.benchmark_id,
        "benchmark_version": truth.benchmark_version,
        "case_id": truth.case_id,
        "source_description": truth.source_description,
        "definition": "Conserved-base pair homology represented by the same physical graph base; oriented path offsets are resolved exactly",
        "confusion": counts,
        "anchors": observations,
        "metrics": {
            "homology_precision": precision,
            "homology_recall": recall,
            "homology_f1": 2 * tp / (2 * tp + fp + fn),
            "exact_input_paths": 1.0,
        },
    }


def evaluate_graph_benchmark(result: OrganelleResult, *, truth_artifact_id: str) -> OrganelleResult:
    """Evaluate a graph against a versioned truth artifact in the same Result.

    The selector is an existing opaque ArtifactRef ID. Neither Agent text nor
    constructor metrics supply the ground-truth labels or the resulting scores.
    """
    return _evaluate_benchmark(
        result,
        truth_artifact_id,
        GraphBenchmarkTruth,
        score_graph,
        "pangenome.evaluate_graph_benchmark",
        "Independent conserved-base homology benchmark",
    )


def evaluate_conformation_benchmark(
    result: OrganelleResult, *, truth_artifact_id: str
) -> OrganelleResult:
    """Measure published-conformation discrimination with a fixed edge predictor."""
    from .conformation_benchmark import ConformationTruth, score_conformations

    return _evaluate_benchmark(
        result,
        truth_artifact_id,
        ConformationTruth,
        score_conformations,
        "pangenome.evaluate_conformation_benchmark",
        "Independent published-conformation discrimination",
    )


def _evaluate_benchmark(result, truth_artifact_id, truth_model, scorer, operation, title):
    graph = _graph_input(result, {"gfa"})
    matches = [
        artifact
        for artifact in result.artifacts
        if artifact.object_id == truth_artifact_id and artifact.format == "json"
    ]
    if len(matches) != 1:
        raise OrganelleInputError(
            code="pangenome.benchmark_truth_required",
            message="Select exactly one captured JSON truth artifact by artifact_id",
        )
    source = matches[0]
    truth = truth_model.model_validate_json(verified_artifact(source).read_text())
    evaluation = scorer(verified_artifact(graph), truth)
    run_id = uuid4().hex
    staging = create_staged_run(operation, run_id)
    evidence = staging / "benchmark-evaluation.json"
    evidence.write_text(json.dumps(evaluation, indent=2, allow_nan=False) + "\n")
    evaluated = OrganelleResult(
        operation_id=operation,
        operation_version="1.0",
        scope=result.scope,
        status="ok",
        summary_text=f"{title}: {truth.case_id}",
        metrics={"optimization_evaluation": evaluation["metrics"]},
        artifacts=(
            ArtifactRef.from_path(
                evidence, kind="benchmark_evaluation", format="json", media_type="application/json"
            ).model_copy(update={"uri": evidence.name}),
        ),
        provenance=make_provenance(
            operation_id=operation,
            parameters={"truth_artifact_id": truth_artifact_id},
            input_object_ids=(result.object_id,),
            input_artifact_hashes=(graph.sha256, source.sha256),
            requested_backend="organelleverse",
            actual_backend="organelleverse",
            attempted_backends=("organelleverse",),
        ).model_copy(update={"operation_version": "1.0"}),
    )
    destination = managed_run_path(operation, run_id)
    published = publish_staged_result(
        evaluated,
        staging,
        destination,
        trusted_input_hashes=frozenset((graph.sha256, source.sha256)),
    )
    (destination / "result.json").write_text(published.model_dump_json(indent=2) + "\n")
    return published
