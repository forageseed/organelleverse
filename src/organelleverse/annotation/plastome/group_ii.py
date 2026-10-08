"""Group II intron 3' splice sites from domains V-VI (Rfam RF00029, Intron_gpII).

Domains V and VI sit at the 3' end of every plastid group II intron; RF00029
models them, and the end of its alignment lies a fixed, intron-specific
distance from the 3' splice site (rpl2 intron 1 +1, clpP intron 1 +13, most
introns 0; conserved across land plants). The distance per gene and intron is
measured on the shipped references (``group_ii_offsets.json``, built by
``scripts/build_group_ii_offsets.py``); a target's 3' splice site is predicted
as its own alignment end minus the most common distance among the references
actually loaded, so a reference removed from the set never contributes. This needs no
close relative of the target, unlike reference-based placement.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

logger = logging.getLogger(__name__)

TAIL = 300  # intron bases before the 3' splice site searched for domains V-VI
FLANK = 30  # downstream exon bases included
SLACK = 20  # extra bases either side: the transferred 3' end may be off
MIN_REFERENCES = 2  # references needed to calibrate a gene's intron
# Share of references agreeing on an intron's offset above which the prediction
# may overrule reference votes: on 381 RNA-verified development introns the
# prediction was right in 192 of 194 at >= 0.7 agreement and 67% below it.
RELIABLE_SHARE = 0.75
RELIABLE_MIN_REFERENCES = 5


def data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "plastome" / "group_ii"


@lru_cache(maxsize=1)
def _model():
    from organelleverse.annotation.cmsearch.model import parse_cm

    path = data_dir() / "RF00029.cm"
    return parse_cm(path) if path.exists() else None


@lru_cache(maxsize=1)
def _offsets() -> dict:
    path = data_dir() / "group_ii_offsets.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())["offsets"]


def alignment_end(window: str) -> int | None:
    """1-based end of the best RF00029 alignment in ``window`` (transcript sense), or None."""
    model = _model()
    if model is None or len(window) < 80:
        return None
    from organelleverse.annotation.cmsearch.cyk import cyk_align

    hit = cyk_align(model, window.upper(), min_score=0.0)
    return hit.end if hit is not None else None


def reference_calibration(gene: str, join_index: int, reference_files: set[str]) -> tuple[int, bool] | None:
    """(offset, reliable) for a gene's intron from the loaded references, or None when uncalibrated.

    ``reliable``: at least RELIABLE_MIN_REFERENCES references and RELIABLE_SHARE of
    them on the most common offset.
    """
    values = [
        entry[gene][str(join_index)]
        for name, entry in _offsets().items()
        if name in reference_files and gene in entry and str(join_index) in entry[gene]
    ]
    if len(values) < MIN_REFERENCES:
        return None
    counts = Counter(values).most_common()
    top = counts[0][1]
    offset = min((v for v, n in counts if n == top), key=abs)
    return offset, len(values) >= RELIABLE_MIN_REFERENCES and top / len(values) >= RELIABLE_SHARE


SUSPECT_TOLERANCE = 3  # nt a reference's offset may differ from a reliable intron's consensus


def suspect_references(reference_files: set[str]) -> set[tuple[str, str]]:
    """(reference file, gene) pairs whose annotated 3' splice site domains V-VI contradict.

    At a reliable intron the references' offsets agree; a reference more than
    SUSPECT_TOLERANCE away has the intron end somewhere else than its own
    domain VI places it, i.e. a misannotated junction. Four grass references
    put petB exon 2 57 nt into the intron (offsets 26-30 against a consensus of
    0) and carried Sorghum and Brachypodium with them; RNA-curated petB
    (CPGAVAS2) and 37 other references agree with domain VI.
    """
    offsets = _offsets()
    joins = {
        (gene, j)
        for name, entry in offsets.items()
        if name in reference_files
        for gene, by_join in entry.items()
        for j in by_join
    }
    out: set[tuple[str, str]] = set()
    for gene, j in joins:
        calibration = reference_calibration(gene, int(j), reference_files)
        if calibration is None or not calibration[1] or calibration[0] is None:
            continue
        for name, entry in offsets.items():
            value = entry.get(gene, {}).get(j)
            if name in reference_files and value is not None and abs(value - calibration[0]) > SUSPECT_TOLERANCE:
                out.add((name, gene))
    return out


def reference_offset(gene: str, join_index: int, reference_files: set[str]) -> int | None:
    """Most common alignment-end minus 3'-splice-site distance over the loaded references, or None.

    The mode, not the median: references annotate some joins two ways (atpF
    0/+1, ycf3 intron 1 0/-2), and a median can fall between the two.
    """
    values = [
        entry[gene][str(join_index)]
        for name, entry in _offsets().items()
        if name in reference_files and gene in entry and str(join_index) in entry[gene]
    ]
    if len(values) < MIN_REFERENCES:
        return None
    counts = Counter(values).most_common()
    top = counts[0][1]
    return min((v for v, n in counts if n == top), key=abs)


def _predict_once(seq: str, approx_end: int, offset: int) -> int | None:
    lo = max(0, approx_end - TAIL - SLACK)
    hi = min(len(seq), approx_end + FLANK + SLACK)
    end = alignment_end(seq[lo:hi])
    if end is None:
        return None
    return lo + end - offset


def predict_intron_end(seq: str, approx_end: int, offset: int) -> int | None:
    """Predicted last intron base (1-based, forward-oriented ``seq``) near ``approx_end``.

    The window is re-centred on the prediction until it is stable: a transferred
    acceptor far upstream of the real one (Zea petB 57 nt) cut domain VI off at
    the window's edge and moved the prediction with it.
    """
    position = _predict_once(seq, approx_end, offset)
    for _ in range(3):
        if position is None:
            return None
        again = _predict_once(seq, position, offset)
        if again == position:
            break
        position = again
    return position


def equivalent_intron_ends(seq: str, end: int, start: int) -> set[int]:
    """Last-intron-base positions (1-based) of the placements equivalent to intron seq[end:start-1]."""
    lo, hi = end, start - 1  # 0-based intron [lo, hi)
    out = {hi}
    a, b = lo, hi
    while a > 0 and b - 1 > a and seq[a - 1] == seq[b - 1]:
        a, b = a - 1, b - 1
        out.add(b)
    a, b = lo, hi
    while b < len(seq) and b - 1 > a and seq[a] == seq[b]:
        a, b = a + 1, b + 1
        out.add(b)
    return out


def structure_agreement(seq: str, exons: list[tuple[int, int]], predicted: list) -> tuple[int, int]:
    """(reliable, other) joins whose 3' splice site (or an equivalent placement) is at the prediction.

    ``predicted`` holds (position, reliable) per join, or None.
    """
    strong = weak = 0
    for ((_, end), (start, _)), p in zip(pairwise(exons), predicted, strict=False):
        if p is not None and p[0] in equivalent_intron_ends(seq, end, start):
            if p[1]:
                strong += 1
            else:
                weak += 1
    return strong, weak


def predicted_ends(seq: str, exons: list[tuple[int, int]], calibrations: list | None) -> list:
    """(predicted last intron base, reliable) per join of a forward-oriented model; None where unavailable.

    ``calibrations`` holds (offset, reliable) per join, or None.
    """
    if not calibrations:
        return []
    out: list = []
    for (_, (start, _)), cal in zip(pairwise(exons), calibrations, strict=False):
        if cal is None:
            out.append(None)
            continue
        position = predict_intron_end(seq, start - 1, cal[0])
        out.append(None if position is None else (position, cal[1]))
    return out
