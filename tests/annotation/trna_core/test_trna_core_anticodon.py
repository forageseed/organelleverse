from __future__ import annotations

import csv
import re

import pytest

from organelleverse.annotation.trna_core.anticodon import (
    isotypes_for_anticodon,
    normalize_anticodon,
)
from tests._paths import PROJECT_ROOT

ROOT = PROJECT_ROOT


def test_normalize_anticodon_uses_dna_lowercase():
    assert normalize_anticodon("GAA") == "gaa"
    assert normalize_anticodon("uUg") == "ttg"
    assert normalize_anticodon(" nNn ") == "nnn"


@pytest.mark.parametrize("value", ["GA", "GAAA", "BAX", ""])
def test_normalize_anticodon_rejects_invalid_values(value: str):
    with pytest.raises(ValueError):
        normalize_anticodon(value)


def test_isotypes_for_known_plant_mitochondrial_anticodons():
    assert "F" in isotypes_for_anticodon("gaa", organelle="mitochondrion")
    assert "K" in isotypes_for_anticodon("ttt", organelle="mitochondrion")
    assert "L" in isotypes_for_anticodon("taa", organelle="mitochondrion")
    assert {"M", "fM", "I"}.issubset(set(isotypes_for_anticodon("cat", organelle="mitochondrion")))
    assert isotypes_for_anticodon("nnn", organelle="mitochondrion") == ()


def test_isotype_lookup_rejects_unknown_organelle():
    with pytest.raises(ValueError):
        isotypes_for_anticodon("gaa", organelle="nucleus")


def test_pmga_positive_fixture_anticodons_are_explainable():
    rows = list(
        csv.DictReader(
            (ROOT / "tests" / "fixtures" / "trna_core" / "positives.tsv").open(), delimiter="\t"
        )
    )
    missing = []
    for row in rows:
        match = re.match(r"trn(.+)\([acgtn]{3}\)", row["gene"])
        assert match is not None
        amino_acid = match.group(1)
        if amino_acid not in isotypes_for_anticodon(row["anticodon"], organelle="mitochondrion"):
            missing.append((row["gene"], row["anticodon"]))

    assert missing == []
