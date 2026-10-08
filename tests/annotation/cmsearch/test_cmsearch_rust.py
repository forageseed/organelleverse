from __future__ import annotations

import random

import pytest

from organelleverse.annotation.cmsearch import search as search_mod
from organelleverse.annotation.cmsearch.cyk import cyk_align
from organelleverse.annotation.cmsearch.model import parse_cm
from tests._paths import PROJECT_ROOT

_PACKAGED_CM = PROJECT_ROOT / "src/organelleverse/annotation/data/models/trna/trna.cm"

pytestmark = pytest.mark.skipif(
    search_mod._rust_cm_cyk is None or not _PACKAGED_CM.exists(),
    reason="Rust cm_cyk kernel or packaged CM not available",
)

_CORE = "AAAGCCGTTATCAAGTGGCAATGAGGCTCAACTTGCGATTGAGAAATCTCGGTATGAGTCCGTGCGGCTTTG"


def test_rust_cyk_matches_python_on_real_model():
    model = parse_cm(_PACKAGED_CM)
    flat = search_mod._flatten_model(model)
    random.seed(5)
    seq = "".join(random.choice("ACGT") for _ in range(40)) + _CORE + "AAAA"

    py = cyk_align(model, seq, min_score=0.0)
    r = search_mod._rust_cm_cyk(seq, *flat, 0.0)
    assert py is not None and r is not None
    assert (r[0], r[1]) == (py.start, py.end)
    assert abs(r[2] - py.score) < 1e-6


def test_rust_traceback_matches_python():
    from organelleverse.annotation.cmsearch import cyk as cyk_mod

    if cyk_mod._rust_cm_cyk_trace is None:
        pytest.skip("Rust cm_cyk_trace kernel not available")
    model = parse_cm(_PACKAGED_CM)
    hit_r, align_r = cyk_mod.cyk_trace(model, _CORE)
    hit_p, align_p = cyk_mod._cyk_trace_py(model, _CORE)
    assert (hit_r.start, hit_r.end) == (hit_p.start, hit_p.end)
    assert abs(hit_r.score - hit_p.score) < 1e-6
    assert align_r == align_p


def test_rust_cm_search_localizes_trna_exactly():
    model = parse_cm(_PACKAGED_CM)
    random.seed(2)
    left = "".join(random.choice("ACGT") for _ in range(130))
    seq = left + _CORE + "".join(random.choice("ACGT") for _ in range(70))
    hits = search_mod.cm_search(seq, model, min_bits=20.0)
    assert hits
    best = max(hits, key=lambda h: h.score)
    assert best.strand == 1
    assert best.start == 131
    assert best.end == 130 + len(_CORE)
