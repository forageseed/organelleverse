from itertools import combinations

import pytest

from organelleverse.pangenome.conformation_diagnostics import compare_deletions, diagnose_pairs


def pairs(labels, constant=False):
    return [
        {
            "first": a,
            "second": b,
            "same_conformation": labels[a] == labels[b],
            "edge_jaccard": 0.5 if constant else float(labels[a] == labels[b]),
        }
        for a, b in combinations(sorted(labels), 2)
    ]


def test_perfect_retrieval_singletons_and_accession_deletion():
    labels = dict(a="A", b="A", c="B", d="B", e="C")
    result = diagnose_pairs(pairs(labels), labels, permutations=19)
    assert result["metrics"]["conformation_auroc"] == 1
    assert result["retrieval_eligible_accessions"] == 4
    assert all(r["average_precision"] == 1 for r in result["retrieval"] if r["eligible"])
    singleton = next(r for r in result["groups"] if r["group"] == "C")
    assert singleton["mean_query_ap"] is None
    assert len(result["leave_one_accession_out"]) == 5
    assert all(r["scorable"] for r in result["leave_one_accession_out"])
    assert len(result["leave_one_group_out"]) == 3
    assert all(r["metrics"]["conformation_auroc"] == 1 for r in result["leave_one_group_out"])
    assert all(
        all(v == 0 for v in r["candidate_minus_baseline"].values())
        for r in compare_deletions(result, result)
    )


def test_ties_do_not_depend_on_sample_order_and_null_is_reproducible():
    labels = dict(a="A", b="A", c="B", d="B")
    result = diagnose_pairs(pairs(labels, True), labels, permutations=19)
    assert result["metrics"]["conformation_auroc"] == 0.5
    assert all(r["top1_tie_fraction"] == pytest.approx(1 / 3) for r in result["retrieval"])
    assert all(r["average_precision"] == pytest.approx(1 / 3) for r in result["retrieval"])
    assert all(
        r["upper_tail_fraction_plus_one"] == 1
        for r in result["random_label_reference"]["metrics"].values()
    )
    again = diagnose_pairs(pairs(labels, True), labels, permutations=19)
    assert result == again


def test_deleting_only_positive_group_member_reports_unscorable():
    labels = dict(a="A", b="A", c="B", d="C")
    result = diagnose_pairs(pairs(labels), labels, permutations=9)
    assert [r["removed"] for r in result["leave_one_accession_out"] if not r["scorable"]] == [
        "a",
        "b",
    ]
    assert compare_deletions(result, result)[0]["candidate_minus_baseline"] is None
    assert result["leave_one_group_out"][0]["metrics"] is None


def test_paired_deletion_direction_and_different_group_binding():
    labels = dict(a="A", b="A", c="B", d="B", e="C")
    baseline = diagnose_pairs(pairs(labels), labels, permutations=9)
    reversed_rows = [{**r, "edge_jaccard": 1 - r["edge_jaccard"]} for r in pairs(labels)]
    candidate = diagnose_pairs(reversed_rows, labels, permutations=9)
    assert all(
        r["candidate_minus_baseline"]["conformation_auroc"] == -1
        for r in compare_deletions(baseline, candidate)
    )
    candidate["leave_one_accession_out"][0]["group"] = "wrong"
    with pytest.raises(ValueError, match="same accessions and labels"):
        compare_deletions(baseline, candidate)


@pytest.mark.parametrize("change", ["missing", "duplicate", "label", "nan"])
def test_invalid_pair_evidence_is_rejected(change):
    labels = dict(a="A", b="A", c="B", d="B")
    rows = pairs(labels)
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "label":
        rows[0]["same_conformation"] = False
    else:
        rows[0]["edge_jaccard"] = float("nan")
    with pytest.raises(ValueError):
        diagnose_pairs(rows, labels, permutations=9)
