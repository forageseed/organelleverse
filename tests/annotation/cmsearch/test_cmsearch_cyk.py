from __future__ import annotations

import pytest

from organelleverse.annotation.cmsearch.cyk import cyk_align
from organelleverse.annotation.cmsearch.model import parse_cm
from organelleverse.annotation.cmsearch.models import CMState, CovarianceModel
from tests._paths import PROJECT_ROOT

_PACKAGED_CM = PROJECT_ROOT / "src/organelleverse/annotation/data/models/trna/trna.cm"


def _single_pair_model() -> CovarianceModel:
    """ROOT S -> MP -> E. MP favours a G-C pair; everything else is penalised."""
    emissions = [-10.0] * 16
    emissions[4 * 2 + 1] = 5.0  # left=G(2), right=C(1)
    states = [
        CMState("S", 0, plast=-1, pnum=0, cfirst=1, cnum=1, transitions=[0.0], emissions=[]),
        CMState("MP", 1, plast=0, pnum=1, cfirst=2, cnum=1, transitions=[0.0], emissions=emissions),
        CMState("E", 2, plast=1, pnum=1, cfirst=-1, cnum=0, transitions=[], emissions=[]),
    ]
    return CovarianceModel(name="tiny_pair", clen=1, states=states)


def _stem_model() -> CovarianceModel:
    """ROOT S -> MP(outer) -> MP(inner) -> E. Two stacked G-C pairs (4 nt, no loop)."""
    e = [-10.0] * 16
    e[4 * 2 + 1] = 5.0  # G-C
    e[4 * 1 + 2] = 5.0  # C-G
    states = [
        CMState("S", 0, plast=-1, pnum=0, cfirst=1, cnum=1, transitions=[0.0], emissions=[]),
        CMState("MP", 1, plast=0, pnum=1, cfirst=2, cnum=1, transitions=[0.0], emissions=list(e)),
        CMState("MP", 2, plast=1, pnum=1, cfirst=3, cnum=1, transitions=[0.0], emissions=list(e)),
        CMState("E", 3, plast=2, pnum=1, cfirst=-1, cnum=0, transitions=[], emissions=[]),
    ]
    return CovarianceModel(name="tiny_stem", clen=2, states=states)


def test_cyk_single_pair_coordinates():
    model = _single_pair_model()
    hit = cyk_align(model, "AAGCAA")  # only G-C adjacent pair at positions 3-4
    assert hit is not None
    assert (hit.start, hit.end) == (3, 4)
    assert hit.score == 5.0


def test_cyk_stem_coordinates_and_score():
    model = _stem_model()
    # outer G..C with inner C-G:  G C G C  -> positions 3..6
    hit = cyk_align(model, "TTGCGCTT")
    assert hit is not None
    assert (hit.start, hit.end) == (3, 6)
    assert hit.score == 10.0


def test_cyk_returns_none_when_no_positive_alignment():
    model = _single_pair_model()
    hit = cyk_align(model, "AAAAAA", min_score=0.0)
    assert hit is None


@pytest.mark.skipif(not _PACKAGED_CM.exists(), reason="packaged tRNA CM not built")
def test_cyk_localizes_real_trna_on_packaged_model():
    model = parse_cm(_PACKAGED_CM)
    # A tRNA emitted from the packaged Rfam RF00005 model (U->T), embedded in
    # 10-nt A flanks; the native CYK must recover the exact span (11..82).
    core = "AAAGCCGTTATCAAGTGGCAATGAGGCTCAACTTGCGATTGAGAAATCTCGGTATGAGTCCGTGCGGCTTTG"
    seq = "A" * 10 + core + "A" * 10
    hit = cyk_align(model, seq, min_score=0.0)
    assert hit is not None
    assert (hit.start, hit.end) == (11, 10 + len(core))
    assert hit.score > 20.0
