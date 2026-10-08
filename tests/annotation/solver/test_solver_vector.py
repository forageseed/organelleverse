"""``solve_vector`` must reproduce ``dp.solve``, the solver's correctness oracle.

Random genes are planted in random DNA -- with and without introns of every
phase, with and without a start codon -- and both implementations must agree
on the score (to floating-point rounding) and on every exon coordinate.
"""

from __future__ import annotations

import math
import random

import pytest

from organelleverse.annotation.solver.dp import AMINO_ACIDS, Profile, solve
from organelleverse.annotation.solver.vector import solve_vector

CODONS: dict[str, list[str]] = {}
for _b1 in "TCAG":
    for _b2 in "TCAG":
        for _b3 in "TCAG":
            _codon = _b1 + _b2 + _b3
            _aa = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"[
                "TCAG".index(_b1) * 16 + "TCAG".index(_b2) * 4 + "TCAG".index(_b3)
            ]
            CODONS.setdefault(_aa, []).append(_codon)

KW = {"min_intron": 20, "max_intron": 120}


def _profile(rng: random.Random, protein: str) -> Profile:
    scores = [
        [5.0 if a == aa else rng.choice([-2.0, -1.0, 0.5]) for a in AMINO_ACIDS] for aa in protein
    ]
    return Profile(name="t", scores=scores, intron_open=-9.5, intron_extend=-0.02)


def _case(seed: int) -> tuple[Profile, str]:
    rng = random.Random(seed)
    protein = "".join(rng.choice(AMINO_ACIDS) for _ in range(rng.randint(5, 14)))
    cds = "ATG" + "".join(rng.choice(CODONS[a]) for a in protein) + "TAA"
    if rng.random() < 0.7:  # plant an intron at a random phase
        cut = rng.randint(4, len(cds) - 4)
        intron = "GT" + "".join(rng.choice("ACGT") for _ in range(rng.randint(25, 80))) + "AT"
        cds = cds[:cut] + intron + cds[cut:]
    if rng.random() < 0.15:
        cds = "CCC" + cds[3:]  # no initiator: often no solution at all
    flank = lambda n: "".join(rng.choice("ACGT") for _ in range(n))  # noqa: E731
    return _profile(rng, protein), flank(rng.randint(0, 40)) + cds + flank(rng.randint(0, 40))


@pytest.mark.parametrize("seed", range(120))
def test_vector_solver_matches_the_oracle(seed: int) -> None:
    profile, seq = _case(seed)
    for kwargs in (KW, {**KW, "require_stop": False}, {**KW, "allow_introns": False}):
        expected = solve(profile, seq, **kwargs)
        got = solve_vector(profile, seq, **kwargs)
        if expected is None:
            assert got is None
            continue
        assert got is not None
        assert math.isclose(got.score, expected.score, rel_tol=1e-9, abs_tol=1e-9)
        assert [(e.start, e.end) for e in got.exons] == [(e.start, e.end) for e in expected.exons]
