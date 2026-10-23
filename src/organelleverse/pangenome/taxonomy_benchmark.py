"""Published-taxonomy recovery from a path graph; not a phylogenetic accuracy score.

The fixed predictor is the unweighted Jaccard similarity of graph-node presence
(one molecule per accession) and its UPGMA tree. Curated taxonomic labels supply
the pair classes and the testable groups. No threshold is fitted to labels.

Taxonomic ranks are working classifications, not an exact species tree: a
recognised group may itself be paraphyletic, so a perfect score is not expected
and only candidates evaluated against the same frozen truth are comparable.
Pair observations share accessions and are not independent replicates.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformation_benchmark import ranking_metrics
from .graph import path_sequences, path_steps
from .phylogeny import node_pav_tree

RANKS = ("family", "tribe")


class TaxonomyTruth(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["organelleverse.pangenome.taxonomy-truth.v1"]
    benchmark_id: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    benchmark_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    case_id: str = Field(min_length=1)
    source_description: str = Field(min_length=1)
    sequences: dict[str, str] = Field(min_length=4)
    accessions: dict[str, str]
    families: dict[str, str]
    tribes: dict[str, str]

    @model_validator(mode="after")
    def validate_cohort(self) -> Self:
        names = set(self.sequences)
        for field in (self.accessions, self.families, self.tribes):
            if set(field) != names:
                raise ValueError("Every benchmark path requires one accession, family and tribe")
        if len(set(self.accessions.values())) != len(self.accessions):
            raise ValueError("Taxonomy benchmark uses one molecule per independent accession")
        if any(not v for f in (self.accessions, self.families, self.tribes) for v in f.values()):
            raise ValueError("Accession, family and tribe labels must not be empty")
        if any(not seq or set(seq.upper()) - set("ACGT") for seq in self.sequences.values()):
            raise ValueError("Benchmark molecules must contain only A/C/G/T")
        counts = Counter(self.tribes.values())
        if len(counts) < 2 or max(counts.values()) < 2:
            raise ValueError("Benchmark requires same- and different-tribe accession pairs")
        return self


def _testable_groups(labels: dict[str, str]) -> dict[str, frozenset[str]]:
    """Groups with at least two members that leave at least two outside."""
    members: dict[str, set[str]] = {}
    for path, label in labels.items():
        members.setdefault(label, set()).add(path)
    total = len(labels)
    return {
        label: frozenset(group)
        for label, group in sorted(members.items())
        if 2 <= len(group) <= total - 2
    }


def _recovery(rank: str, labels: dict[str, str], tree: dict) -> dict:
    """A group is recovered when it or its complement is a clade (unrooted).

    The UPGMA root is an arbitrary midpoint of average linkage, so requiring the
    group itself to be the rooted clade would penalise root placement, not the
    graph.
    """
    paths = frozenset(labels)
    clades = {frozenset(c["paths"]) for c in tree["clades"]}
    groups = _testable_groups(labels)
    return {
        "rank": rank,
        "testable": len(groups),
        "recovered": sorted(
            label for label, group in groups.items() if group in clades or paths - group in clades
        ),
        "missed": sorted(
            label
            for label, group in groups.items()
            if group not in clades and paths - group not in clades
        ),
        "sizes": {label: len(group) for label, group in groups.items()},
    }


def _node_presence(paths: dict[str, list[dict]]) -> tuple[list[str], np.ndarray]:
    names = sorted(paths)
    nodes = sorted({row["node"] for rows in paths.values() for row in rows})
    index = {node: i for i, node in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(names)), dtype=np.int64)
    for column, name in enumerate(names):
        for row in paths[name]:
            matrix[index[row["node"]], column] = 1
    return names, matrix


def score_taxonomy(graph_path: Path, truth: TaxonomyTruth) -> dict:
    observed = path_sequences(graph_path)
    if {key: seq.upper() for key, seq in observed.items()} != {
        key: seq.upper() for key, seq in truth.sequences.items()
    }:
        raise ValueError(
            "Benchmark graph must exactly reconstruct every declared molecule and identity"
        )
    paths = path_steps(graph_path)
    names, matrix = _node_presence(paths)
    if int(matrix.sum(axis=0).min()) == 0:
        raise ValueError("Every benchmark path must traverse at least one graph node")
    tree = node_pav_tree(names, matrix)
    similarity = 1.0 - np.asarray(tree["distances"])
    position = {name: i for i, name in enumerate(names)}
    observations = [
        {
            "first": first,
            "second": second,
            "same_tribe": truth.tribes[first] == truth.tribes[second],
            "same_family": truth.families[first] == truth.families[second],
            "node_jaccard": float(similarity[position[first], position[second]]),
        }
        for first, second in combinations(names, 2)
    ]
    ranked = ranking_metrics(
        [
            {"same_conformation": row["same_tribe"], "edge_jaccard": row["node_jaccard"]}
            for row in observations
        ]
    )
    recoveries = [
        _recovery("family", truth.families, tree),
        _recovery("tribe", truth.tribes, tree),
    ]
    testable = sum(item["testable"] for item in recoveries)
    if not testable:
        raise ValueError("Benchmark has no testable taxonomic group")
    recovered = sum(len(item["recovered"]) for item in recoveries)
    return {
        "benchmark_id": truth.benchmark_id,
        "benchmark_version": truth.benchmark_version,
        "case_id": truth.case_id,
        "source_description": truth.source_description,
        "definition": "Recovery of curated taxonomic groups from unweighted graph-node-presence Jaccard similarity: same-tribe pair AUROC/average precision, and the fraction of testable family and tribe groups that are clades (or complements of clades) in the UPGMA tree; no fitted threshold; ranks are working classifications and pair observations are not independent replicates",
        "pairs": observations,
        "recoveries": recoveries,
        "accession_count": len(truth.accessions),
        "newick": tree["newick"],
        "metrics": {
            "taxonomic_pair_auroc": ranked["conformation_auroc"],
            "taxonomic_pair_average_precision": ranked["conformation_average_precision"],
            "same_tribe_prevalence": ranked["same_conformation_prevalence"],
            "taxonomic_clade_recovery": recovered / testable,
            # A rank with no testable group has no recovery rate; omit it, never report 1.0.
            **{
                f"{item['rank']}_clade_recovery": len(item["recovered"]) / item["testable"]
                for item in recoveries
                if item["testable"]
            },
            "exact_input_paths": 1.0,
        },
    }
