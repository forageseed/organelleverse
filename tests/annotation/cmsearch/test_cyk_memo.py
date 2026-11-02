"""The memoised CYK kernel call returns exactly what the kernel returns, on first and repeated calls."""

import random

import pytest

from organelleverse.annotation.cmsearch import search
from organelleverse.annotation.cmsearch.genome import default_plastid_cm_path
from organelleverse.annotation.cmsearch.model import parse_cm


@pytest.mark.skipif(search._rust_cm_cyk is None, reason="Rust kernel not built")
def test_memo_equals_kernel() -> None:
    model = parse_cm(default_plastid_cm_path())
    flat = search._flatten_model(model)
    rng = random.Random(7)
    seqs = ["".join(rng.choice("ACGT") for _ in range(rng.randint(40, 220))) for _ in range(30)]
    for bits in (float("-inf"), 0.0, 20.0):
        for s in seqs + seqs[:10]:  # the last ten repeat
            assert search.rust_cyk_memo(model, flat, s, bits) == search._rust_cm_cyk(s, *flat, bits)
    fresh = parse_cm(default_plastid_cm_path())
    assert "_cyk_memo" not in fresh.__dict__  # the memo belongs to one model object
