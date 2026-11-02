import importlib.util
from pathlib import Path

import pytest

path = (
    Path(__file__).resolve().parents[2]
    / "scripts/pangenome_benchmarks/prepare_conformation_cohort.py"
)
spec = importlib.util.spec_from_file_location("prepare_conformation_cohort", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_representatives_keep_rare_groups_and_independent_accessions():
    partition = {
        "partitions": {
            "search": {"Alpha": ["S1", "S2"], "Beta": ["S3"], "Gamma": ["S4"], "Rare": ["S5"]},
            "validation": {"Delta": ["V1"], "Epsilon": ["V2"], "Zeta": ["V3"]},
        }
    }
    structures = [
        {"sample_index": sample, "group": group}
        for sample, group in [
            ("S1", "common"),
            ("S2", "common"),
            ("S3", "common"),
            ("S4", "common"),
            ("S5", "singleton"),
            ("V1", "common"),
            ("V2", "common"),
            ("V3", "singleton"),
        ]
    ]
    selected = module.select_samples(partition, structures)
    assert selected["search"] == [
        ("common", "Alpha", "S1"),
        ("common", "Beta", "S3"),
        ("singleton", "Rare", "S5"),
    ]
    assert {row[1] for row in selected["search"]}.isdisjoint(
        row[1] for row in selected["validation"]
    )


def test_conflicting_replicates_require_review_instead_of_selecting_one_label():
    partition = {"partitions": {"search": {"Alpha": ["S1", "S2"]}, "validation": {"Beta": ["V1"]}}}
    structures = [
        {"sample_index": sample, "group": group}
        for sample, group in [("S1", "one"), ("S2", "two"), ("V1", "one")]
    ]
    with pytest.raises(ValueError, match="Replicates disagree"):
        module.select_samples(partition, structures)


def test_accession_overlap_is_not_hidden_by_different_sample_ids():
    partition = {"partitions": {"search": {"Same": ["S1"]}, "validation": {"Same": ["V1"]}}}
    structures = [{"sample_index": sample, "group": "one"} for sample in ("S1", "V1")]
    with pytest.raises(ValueError, match="overlap"):
        module.select_samples(partition, structures)
