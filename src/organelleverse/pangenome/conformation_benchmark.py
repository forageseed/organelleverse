"""Independent published-conformation discrimination, not base-homology accuracy.

The fixed predictor is Jaccard similarity of canonical bidirected path edges.
Published conformation labels supply pair classes; no threshold is fit to labels.
Small variants also alter edges, so this endpoint is not a structural-variant
caller or a claim that graph edges alone recover all biological rearrangements.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations, pairwise
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .graph import path_sequences, path_steps


class ConformationTruth(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["organelleverse.pangenome.conformation-truth.v1"]
    benchmark_id: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    benchmark_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    case_id: str = Field(min_length=1)
    source_description: str = Field(min_length=1)
    sequences: dict[str, str] = Field(min_length=3)
    accessions: dict[str, str]
    conformations: dict[str, str]

    @model_validator(mode="after")
    def validate_cohort(self) -> Self:
        if set(self.sequences) != set(self.accessions) or set(self.sequences) != set(
            self.conformations
        ):
            raise ValueError("Every benchmark path requires exactly one accession and conformation")
        if len(set(self.accessions.values())) != len(self.accessions):
            raise ValueError("Conformation benchmark uses one molecule per independent accession")
        if any(not value for value in (*self.accessions.values(), *self.conformations.values())):
            raise ValueError("Accession and conformation labels must not be empty")
        if any(not seq or set(seq.upper()) - set("ACGT") for seq in self.sequences.values()):
            raise ValueError("Benchmark molecules must contain only A/C/G/T")
        counts = Counter(self.conformations.values())
        if len(counts) < 2 or max(counts.values()) < 2:
            raise ValueError("Benchmark requires same- and different-conformation accession pairs")
        return self


def ranking_metrics(observations: list[dict]) -> dict:
    """Tie-aware ROC AUC and stepwise average precision, with no fitted cutoff."""
    buckets = {}
    for row in observations:
        score = row["edge_jaccard"]
        bucket = buckets.setdefault(score, [0, 0])
        bucket[0 if row["same_conformation"] else 1] += 1
    positives = sum(row[0] for row in buckets.values())
    negatives = sum(row[1] for row in buckets.values())
    if not positives or not negatives:
        raise ValueError("Both pair classes are required for discrimination metrics")
    below = wins = 0
    for score in sorted(buckets):
        pos, neg = buckets[score]
        wins += pos * (below + 0.5 * neg)
        below += neg
    tp = fp = 0
    ap = 0.0
    for score in sorted(buckets, reverse=True):
        pos, neg = buckets[score]
        tp += pos
        fp += neg
        ap += pos / positives * tp / (tp + fp)
    return {
        "conformation_auroc": wins / (positives * negatives),
        "conformation_average_precision": ap,
        "same_conformation_prevalence": positives / (positives + negatives),
    }


def _edge(first: dict, second: dict) -> tuple:
    forward = (first["node"], first["orientation"], second["node"], second["orientation"])
    reverse = (
        second["node"],
        "-" if second["orientation"] == "+" else "+",
        first["node"],
        "-" if first["orientation"] == "+" else "+",
    )
    return min(forward, reverse)


def score_conformations(graph_path: Path, truth: ConformationTruth) -> dict:
    observed = path_sequences(graph_path)
    if {key: seq.upper() for key, seq in observed.items()} != {
        key: seq.upper() for key, seq in truth.sequences.items()
    }:
        raise ValueError(
            "Benchmark graph must exactly reconstruct every declared molecule and identity"
        )
    paths = path_steps(graph_path)
    if any(row["overlap_previous"] for rows in paths.values() for row in rows) or any(
        rows[0]["start"] != 0 for rows in paths.values()
    ):
        raise ValueError("Conformation benchmark requires complete zero-overlap paths")
    edges = {name: {_edge(a, b) for a, b in pairwise(rows)} for name, rows in paths.items()}
    observations = []
    for first, second in combinations(sorted(paths), 2):
        union = edges[first] | edges[second]
        if not union:
            raise ValueError("Pair has no represented path edges; edge similarity is undefined")
        observations.append(
            {
                "first": first,
                "second": second,
                "same_conformation": truth.conformations[first] == truth.conformations[second],
                "edge_jaccard": len(edges[first] & edges[second]) / len(union),
            }
        )
    return {
        "benchmark_id": truth.benchmark_id,
        "benchmark_version": truth.benchmark_version,
        "case_id": truth.case_id,
        "source_description": truth.source_description,
        "definition": "Published same-conformation pair discrimination using canonical bidirected internal-edge Jaccard; no inferred circular closing edge, no fitted threshold, pair observations are not independent replicates",
        "pairs": observations,
        "accession_count": len(truth.accessions),
        "metrics": {**ranking_metrics(observations), "exact_input_paths": 1.0},
    }
