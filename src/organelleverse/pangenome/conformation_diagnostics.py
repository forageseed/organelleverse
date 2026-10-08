"""Post hoc accession-level diagnostics for fixed conformation graph scores.

Deletion ranges are influence diagnostics, not confidence intervals. Label
permutation is a conditional random-label reference, not proof of generalization.
Graphs are kept fixed; none of these diagnostics changes the adoption decision.
"""

from collections import Counter
from itertools import combinations
from math import isfinite
from random import Random

from .conformation_benchmark import ranking_metrics

METRICS = ("conformation_auroc", "conformation_average_precision")


def diagnose_pairs(pairs: list[dict], labels: dict[str, str], *, permutations=1999, seed=20260909):
    names = sorted(labels)
    expected = set(combinations(names, 2))
    observed = {}
    for row in pairs:
        key = tuple(sorted((row["first"], row["second"])))
        if key not in expected or key in observed:
            raise ValueError("Diagnostics require each unordered accession pair exactly once")
        score = row["edge_jaccard"]
        if not isinstance(score, (int, float)) or not isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Pair score must be finite and in [0, 1]")
        if row["same_conformation"] != (labels[key[0]] == labels[key[1]]):
            raise ValueError("Pair classes differ from the bound published labels")
        observed[key] = row
    if set(observed) != expected:
        raise ValueError("Diagnostics require the complete accession pair matrix")
    if not isinstance(permutations, int) or permutations < 1:
        raise ValueError("A positive permutation count is required")
    metrics = ranking_metrics(pairs)
    deletion = []
    retrieval = []
    sizes = Counter(labels.values())
    for name in names:
        remaining = [r for r in pairs if name not in (r["first"], r["second"])]
        both = {r["same_conformation"] for r in remaining} == {True, False}
        deletion.append(
            {
                "removed": name,
                "group": labels[name],
                "scorable": both,
                "metrics": ranking_metrics(remaining) if both else None,
                "reason": None if both else "Deleting this accession removes a pair class",
            }
        )
        neighbors = [r for r in pairs if name in (r["first"], r["second"])]
        top = max(r["edge_jaccard"] for r in neighbors)
        tied = [r for r in neighbors if r["edge_jaccard"] == top]
        eligible = sizes[labels[name]] > 1
        retrieval.append(
            {
                "sample": name,
                "group": labels[name],
                "eligible": eligible,
                "group_size": sizes[labels[name]],
                "top_score": top,
                "top_neighbors": [r["second"] if r["first"] == name else r["first"] for r in tied],
                "top1_tie_fraction": sum(r["same_conformation"] for r in tied) / len(tied),
                "average_precision": ranking_metrics(neighbors)[METRICS[1]] if eligible else None,
            }
        )
    groups = []
    for group in sorted(sizes):
        members = [r for r in retrieval if r["group"] == group]
        eligible = sizes[group] > 1
        groups.append(
            {
                "group": group,
                "accessions": sizes[group],
                "retrieval_eligible": eligible,
                "mean_query_ap": sum(r["average_precision"] for r in members) / len(members)
                if eligible
                else None,
                "mean_top1_tie_fraction": sum(r["top1_tie_fraction"] for r in members)
                / len(members)
                if eligible
                else None,
            }
        )
    group_deletion = []
    for group in sorted(sizes):
        remaining = [
            r for r in pairs if labels[r["first"]] != group and labels[r["second"]] != group
        ]
        both = {r["same_conformation"] for r in remaining} == {True, False}
        group_deletion.append(
            {
                "removed_group": group,
                "removed_accessions": sizes[group],
                "scorable": both,
                "metrics": ranking_metrics(remaining) if both else None,
            }
        )
    rng = Random(seed)
    original_labels = [labels[n] for n in names]
    null = {m: [] for m in METRICS}
    for _ in range(permutations):
        shuffled = original_labels.copy()
        rng.shuffle(shuffled)
        mapping = dict(zip(names, shuffled, strict=True))
        permuted = [
            {**r, "same_conformation": mapping[r["first"]] == mapping[r["second"]]} for r in pairs
        ]
        result = ranking_metrics(permuted)
        for metric in METRICS:
            null[metric].append(result[metric])
    references = {}
    for metric, values in null.items():
        ordered = sorted(values)
        references[metric] = {
            "mean": sum(values) / len(values),
            "lower_order_statistic": ordered[int(0.025 * (len(values) - 1))],
            "upper_order_statistic": ordered[int(0.975 * (len(values) - 1))],
            "upper_tail_fraction_plus_one": (1 + sum(v >= metrics[metric] for v in values))
            / (len(values) + 1),
        }
    return {
        "metrics": metrics,
        "accessions": len(names),
        "groups": groups,
        "positive_pairs": sum(r["same_conformation"] for r in pairs),
        "total_pairs": len(pairs),
        "retrieval_eligible_accessions": sum(r["eligible"] for r in retrieval),
        "retrieval": retrieval,
        "leave_one_accession_out": deletion,
        "leave_one_group_out": group_deletion,
        "random_label_reference": {
            "permutations": permutations,
            "seed": seed,
            "metrics": references,
        },
        "same_group_pairs_lowest_first": sorted(
            (r for r in pairs if r["same_conformation"]), key=lambda r: r["edge_jaccard"]
        ),
        "different_group_pairs_highest_first": sorted(
            (r for r in pairs if not r["same_conformation"]), key=lambda r: -r["edge_jaccard"]
        ),
    }


def compare_deletions(baseline: dict, candidate: dict):
    """Paired changes after deleting the same accession from two fixed graphs."""
    left = {r["removed"]: r for r in baseline["leave_one_accession_out"]}
    right = {r["removed"]: r for r in candidate["leave_one_accession_out"]}
    if left.keys() != right.keys() or any(left[n]["group"] != right[n]["group"] for n in left):
        raise ValueError("Paired deletion comparison requires the same accessions and labels")
    return [
        {
            "removed": n,
            "candidate_minus_baseline": {
                m: right[n]["metrics"][m] - left[n]["metrics"][m] for m in METRICS
            }
            if left[n]["scorable"] and right[n]["scorable"]
            else None,
        }
        for n in sorted(left)
    ]
