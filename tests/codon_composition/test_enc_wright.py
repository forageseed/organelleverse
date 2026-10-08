"""Wright's ENC: extremes, monotonicity, and the real-data regression.

The pre-fix implementation divided class sizes by a bias-corrected
homozygosity in [0, 1] — near zero for the moderately even codon usage of
real organelle genes — so every family hit its ceiling and ENC always
collapsed to 61. These tests pin the corrected formula (Wright 1990).
"""

from __future__ import annotations

from organelleverse._bio import get_codon_table
from organelleverse.codon_composition.codon_core import _synonymous_families, compute_enc

_CODE = get_codon_table(11)


def _counts(by_codon: dict[str, int]) -> dict[str, int]:
    return {codon: by_codon.get(codon, 0) for codon in _CODE}


def test_extreme_bias_gives_twenty() -> None:
    one_codon_each = {}
    for family in _synonymous_families(_CODE, include_stop=False).values():
        one_codon_each[family[0]] = 10
    assert compute_enc(_counts(one_codon_each), _CODE) == 20.0


def test_perfect_uniformity_gives_sixty_one() -> None:
    every_codon = {codon: 10 for codon in _CODE}
    assert compute_enc(_counts(every_codon), _CODE) == 61.0


def test_moderate_bias_lands_between_and_moves_monotonically() -> None:
    # Skew just the 2-fold Phe family 30/70; everything else uniform.
    mild = {codon: 10 for codon in _CODE}
    mild["TTT"], mild["TTC"] = 7, 3
    mild_enc = compute_enc(_counts(mild), _CODE)
    assert 55.0 < mild_enc < 61.0
    heavier = dict(mild)
    heavier["TTT"], heavier["TTC"] = 95, 5
    heavier_enc = compute_enc(_counts(heavier), _CODE)
    assert heavier_enc < mild_enc
    assert 45.0 < heavier_enc < 61.0
