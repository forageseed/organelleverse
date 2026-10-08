"""Windowed covariance-model search over both strands.

``cm_search`` scans a sequence with overlapping windows, runs the CYK aligner in
each, maps hits back to 1-based genome coordinates, filters by bit-score, and
resolves overlaps. Windows bound the O(W^2 .. W^3) CYK cost; overlap is kept at
least as long as the model so a hit is never split across a boundary.

The pure-Python CYK is the correctness reference; when the Rust kernel is
available (Phase 2 Task 6) ``search`` will prefer it transparently.
"""

from __future__ import annotations

from .cyk import cyk_align
from .models import CMHit, CovarianceModel

try:  # optional Rust acceleration
    from ...accel import cm_cyk as _rust_cm_cyk
except Exception:  # pragma: no cover - accel import is best-effort
    _rust_cm_cyk = None

_RC = str.maketrans("ACGTacgt", "TGCAtgca")

_TYPE_CODE = {"E": 0, "S": 1, "D": 2, "ML": 3, "MR": 4, "IL": 5, "IR": 6, "MP": 7, "B": 8, "EL": 9}


def _reverse_complement(seq: str) -> str:
    return seq.translate(_RC)[::-1]


def _flatten_model(model: CovarianceModel):
    """Flatten a CovarianceModel into arrays for the Rust ``cm_cyk`` kernel."""
    types: list[int] = []
    cfirst: list[int] = []
    cnum: list[int] = []
    trans_flat: list[float] = []
    trans_off: list[int] = [0]
    emit_flat: list[float] = []
    emit_off: list[int] = [0]
    for st in model.states:
        types.append(_TYPE_CODE[st.type])
        cfirst.append(st.cfirst)
        cnum.append(st.cnum)
        trans_flat.extend(st.transitions)
        trans_off.append(len(trans_flat))
        emit_flat.extend(st.emissions)
        emit_off.append(len(emit_flat))
    return types, cfirst, cnum, trans_flat, trans_off, emit_flat, emit_off


def _default_window(model: CovarianceModel) -> int:
    # tRNAs are short; cap the window so CYK stays cheap but always covers a full
    # model instance plus insertions.
    return max(200, int(model.clen * 2.5))


def _cyk_window(model: CovarianceModel, flat, chunk: str, min_bits: float):
    """Best CYK hit in a window as (start, end, score); Rust kernel if available."""
    if _rust_cm_cyk is not None and flat is not None:
        types, cfirst, cnum, tf, to, ef, eo = flat
        res = _rust_cm_cyk(chunk, types, cfirst, cnum, tf, to, ef, eo, min_bits)
        if res is None:
            return None
        return res[0], res[1], res[2]
    hit = cyk_align(model, chunk, min_score=min_bits)
    if hit is None:
        return None
    return hit.start, hit.end, hit.score


def _scan_strand(
    seq: str, model: CovarianceModel, flat, strand: int, window: int, step: int, min_bits: float
) -> list[CMHit]:
    hits: list[CMHit] = []
    L = len(seq)
    start = 0
    while start < L:
        chunk = seq[start : start + window]
        res = _cyk_window(model, flat, chunk, min_bits)
        if res is not None:
            hstart, hend, score = res
            hits.append(
                CMHit(
                    start=start + hstart,
                    end=start + hend,
                    strand=strand,
                    score=score,
                    cm_from=1,
                    cm_to=model.clen,
                )
            )
        if start + window >= L:
            break
        start += step
    return hits


def _map_reverse(hits: list[CMHit], L: int) -> list[CMHit]:
    out: list[CMHit] = []
    for h in hits:
        gstart = L - h.end + 1
        gend = L - h.start + 1
        out.append(
            CMHit(
                start=gstart, end=gend, strand=-1, score=h.score, cm_from=h.cm_from, cm_to=h.cm_to
            )
        )
    return out


def _intervals(hit: CMHit) -> tuple[tuple[int, int], ...]:
    return hit.exons or ((hit.start, hit.end),)


def _hits_overlap(a: CMHit, b: CMHit) -> bool:
    return any(
        not (a_end < b_start or a_start > b_end)
        for a_start, a_end in _intervals(a)
        for b_start, b_end in _intervals(b)
    )


def _resolve_overlaps(hits: list[CMHit]) -> list[CMHit]:
    """Keep the highest-scoring hit among any set of overlapping hits.

    Spliced hits overlap others only through their exons, so a hit lying inside
    an intron is not discarded because of the spliced gene's span.
    """
    chosen: list[CMHit] = []
    for h in sorted(hits, key=lambda x: x.score, reverse=True):
        if any(_hits_overlap(h, c) for c in chosen):
            continue
        chosen.append(h)
    return sorted(chosen, key=lambda x: (x.start, x.end))


def cm_search(
    seq: str,
    model: CovarianceModel,
    *,
    min_bits: float = 20.0,
    window: int | None = None,
    step: int | None = None,
) -> list[CMHit]:
    """Search both strands of ``seq`` with ``model``; return score-filtered hits."""
    seq = seq.upper()
    L = len(seq)
    if L == 0:
        return []
    win = window if window is not None else _default_window(model)
    win = min(win, L)
    # Overlap must exceed the model length so a hit is never split.
    stp = step if step is not None else max(1, win - max(model.clen + 20, win // 2))

    flat = _flatten_model(model) if _rust_cm_cyk is not None else None
    fwd = _scan_strand(seq, model, flat, 1, win, stp, min_bits)
    rc = _scan_strand(_reverse_complement(seq), model, flat, -1, win, stp, min_bits)
    rev = _map_reverse(rc, L)

    return _resolve_overlaps(fwd + rev)
