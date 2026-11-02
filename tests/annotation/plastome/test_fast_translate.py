"""translate11 returns exactly what Biopython's table-11 translation returns."""

import random
import warnings

import pytest
from Bio import BiopythonWarning
from Bio.Seq import Seq

from organelleverse.annotation.plastome.fast_translate import translate11


def _bio(nt: str) -> str:
    return str(Seq(nt).translate(table=11))


def test_all_64_codons() -> None:
    for a in "ACGT":
        for b in "ACGT":
            for c in "ACGT":
                assert translate11(a + b + c) == _bio(a + b + c)


@pytest.mark.parametrize("alphabet", ["ACGT", "ACGTN", "acgtACGT", "ACGTRYKMSWN"])
def test_random_sequences(alphabet: str) -> None:
    rng = random.Random(alphabet)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonWarning)  # partial codons warn in both paths alike
        for n in [*range(13), 99, 300, 1200, 3001]:
            for _ in range(20):
                nt = "".join(rng.choice(alphabet) for _ in range(n))
                assert translate11(nt) == _bio(nt), nt
