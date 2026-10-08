from __future__ import annotations

import random

import pytest

from organelleverse.annotation.cmsearch.model import parse_cm
from organelleverse.annotation.cmsearch.search import cm_search
from tests._paths import PROJECT_ROOT

_PACKAGED_CM = PROJECT_ROOT / "src/organelleverse/annotation/data/models/trna/trna.cm"
pytestmark = pytest.mark.skipif(not _PACKAGED_CM.exists(), reason="packaged tRNA CM not built")

_CORE = "AAAGCCGTTATCAAGTGGCAATGAGGCTCAACTTGCGATTGAGAAATCTCGGTATGAGTCCGTGCGGCTTTG"


def _rc(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def test_cm_search_forward_strand_coordinates():
    model = parse_cm(_PACKAGED_CM)
    random.seed(3)
    left = "".join(random.choice("ACGT") for _ in range(120))
    right = "".join(random.choice("ACGT") for _ in range(140))
    seq = left + _CORE + right
    hits = cm_search(seq, model, min_bits=20.0)
    assert hits
    best = max(hits, key=lambda h: h.score)
    assert best.strand == 1
    assert best.start == 121
    assert best.end == 120 + len(_CORE)


def test_cm_search_reverse_strand_coordinates():
    model = parse_cm(_PACKAGED_CM)
    random.seed(9)
    left = "".join(random.choice("ACGT") for _ in range(90))
    right = "".join(random.choice("ACGT") for _ in range(110))
    seq = left + _rc(_CORE) + right
    hits = cm_search(seq, model, min_bits=20.0)
    assert hits
    best = max(hits, key=lambda h: h.score)
    assert best.strand == -1
    assert best.start == 91
    assert best.end == 90 + len(_CORE)


def test_cm_search_threshold_filters_noise():
    model = parse_cm(_PACKAGED_CM)
    random.seed(1)
    seq = "".join(random.choice("ACGT") for _ in range(300))
    hits = cm_search(seq, model, min_bits=30.0)
    assert all(h.score >= 30.0 for h in hits)
