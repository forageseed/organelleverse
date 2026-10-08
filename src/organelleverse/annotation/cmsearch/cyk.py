"""Pure-Python CYK alignment over a covariance model.

Computes, for a covariance model rooted at state 0, the best-scoring alignment
to any subsequence ``[i..j]`` of the target and returns 1-based inclusive
coordinates. This is the correctness reference for the native cmsearch engine;
the Rust port mirrors this recursion for speed.

Recursion (log-odds, maximisation) per state type over span ``[i..j]`` (``d`` =
span length):

- ``E``  : 0 if the span is empty, else -inf
- ``S``/``D`` : max over children ``c`` of ``t[v->c] + dp[c][i][j]``
- ``ML``/``IL`` : ``e(x_i) + max_c(t + dp[c][i+1][j])``   (d >= 1)
- ``MR``/``IR`` : ``e(x_j) + max_c(t + dp[c][i][j-1])``   (d >= 1)
- ``MP`` : ``e(x_i, x_j) + max_c(t + dp[c][i+1][j-1])``   (d >= 2)
- ``B``  : ``max_k dp[left][i][k] + dp[right][k+1][j]``   (bifurcation split)

States are ordered so every child index exceeds its parent, so filling ``v``
from last to first guarantees children are ready.
"""

from __future__ import annotations

import logging

from .models import CMHit, CovarianceModel

try:  # optional Rust acceleration
    from ...accel import cm_cyk_trace as _rust_cm_cyk_trace
except Exception:  # pragma: no cover - accel import is best-effort
    _rust_cm_cyk_trace = None

logger = logging.getLogger(__name__)
_warned_python_dp = False


def _warn_python_dp() -> None:
    """Say once that CM alignment runs on the pure-Python reference DP.

    Without the Rust kernel one plastome tRNA search can take tens of minutes
    at 100% CPU with no other sign of progress.
    """
    global _warned_python_dp
    if not _warned_python_dp:
        _warned_python_dp = True
        logger.warning(
            "organelleverse_rust is not installed: CM/CYK alignment uses the pure-Python "
            "reference DP, which is orders of magnitude slower (a plastome tRNA search can "
            "take tens of minutes). Build and install rust/ to accelerate it."
        )

_NEG_INF = float("-inf")
_BASE = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
_TYPE_CODE = {"E": 0, "S": 1, "D": 2, "ML": 3, "MR": 4, "IL": 5, "IR": 6, "MP": 7, "B": 8, "EL": 9}


def _encode(seq: str) -> list[int]:
    return [_BASE.get(ch, -1) for ch in seq.upper()]


def _children(state) -> list[int]:
    if state.type == "B":
        # Bifurcation: the two subtree roots are stored in cfirst and cnum.
        return [state.cfirst, state.cnum]
    return [state.cfirst + k for k in range(state.cnum)]


def _forward_dp(model: CovarianceModel, x: list[int]) -> list[list[list[float]]]:
    """Fill the CYK score table ``dp[v][i][j]`` for the encoded sequence ``x``."""
    L = len(x)
    states = model.states
    n = len(states)
    dp = [[[_NEG_INF] * (L + 2) for _ in range(L + 2)] for _ in range(n)]

    for v in range(n - 1, -1, -1):
        st = states[v]
        t = st.type
        dpv = dp[v]
        kids = _children(st)
        trans = st.transitions
        emit = st.emissions

        for i in range(L + 1, 0, -1):
            row = dpv[i]
            for j in range(i - 1, L + 1):
                d = j - i + 1
                if t == "E" or t == "EL":
                    row[j] = 0.0 if d == 0 else _NEG_INF
                elif t == "S" or t == "D":
                    best = _NEG_INF
                    for k, c in enumerate(kids):
                        val = trans[k] + dp[c][i][j]
                        if val > best:
                            best = val
                    row[j] = best
                elif t == "ML" or t == "IL":
                    if d < 1:
                        row[j] = _NEG_INF
                        continue
                    a = x[i - 1]
                    es = emit[a] if a >= 0 else _NEG_INF
                    best = _NEG_INF
                    for k, c in enumerate(kids):
                        val = trans[k] + dp[c][i + 1][j]
                        if val > best:
                            best = val
                    row[j] = es + best
                elif t == "MR" or t == "IR":
                    if d < 1:
                        row[j] = _NEG_INF
                        continue
                    b = x[j - 1]
                    es = emit[b] if b >= 0 else _NEG_INF
                    best = _NEG_INF
                    for k, c in enumerate(kids):
                        val = trans[k] + dp[c][i][j - 1]
                        if val > best:
                            best = val
                    row[j] = es + best
                elif t == "MP":
                    if d < 2:
                        row[j] = _NEG_INF
                        continue
                    a = x[i - 1]
                    b = x[j - 1]
                    es = emit[4 * a + b] if (a >= 0 and b >= 0) else _NEG_INF
                    best = _NEG_INF
                    for k, c in enumerate(kids):
                        val = trans[k] + dp[c][i + 1][j - 1]
                        if val > best:
                            best = val
                    row[j] = es + best
                elif t == "B":
                    left, right = kids
                    best = _NEG_INF
                    dl = dp[left]
                    dr = dp[right]
                    for k in range(i - 1, j + 1):
                        val = dl[i][k] + dr[k + 1][j]
                        if val > best:
                            best = val
                    row[j] = best
                else:
                    row[j] = _NEG_INF
    return dp


def cyk_align(model: CovarianceModel, seq: str, *, min_score: float = 0.0) -> CMHit | None:
    """Return the best CM alignment to a subsequence of ``seq`` (1-based coords).

    Returns ``None`` if no alignment scores strictly greater than ``min_score``.
    Uses the Rust kernel when available — same result as the pure-Python DP,
    orders of magnitude faster; the Python path stays as the reference
    fallback (and mirrors the kernel's recursion).
    """
    if _rust_cm_cyk_trace is not None:
        res = _rust_cm_cyk_trace(seq.upper(), *_flatten_for_trace(model))
        if res is None:
            return None
        start, end, score, _align = res
        if score <= min_score:
            return None
        return CMHit(start=start, end=end, strand=1, score=score, cm_from=1, cm_to=model.clen)
    _warn_python_dp()
    x = _encode(seq)
    L = len(x)
    dp = _forward_dp(model, x)

    # Best subsequence [i..j] under the root (state 0).
    root = dp[0]
    best_score = _NEG_INF
    best_i = best_j = -1
    for i in range(1, L + 1):
        for j in range(i, L + 1):
            s = root[i][j]
            if s > best_score:
                best_score = s
                best_i, best_j = i, j

    if best_i < 0 or best_score <= min_score:
        return None
    return CMHit(start=best_i, end=best_j, strand=1, score=best_score, cm_from=1, cm_to=model.clen)


def _flatten_for_trace(model: CovarianceModel):
    """Flatten a model into arrays for the Rust ``cm_cyk_trace`` kernel."""
    types: list[int] = []
    cfirst: list[int] = []
    cnum: list[int] = []
    trans_flat: list[float] = []
    trans_off: list[int] = [0]
    emit_flat: list[float] = []
    emit_off: list[int] = [0]
    state_node: list[int] = []
    for st in model.states:
        types.append(_TYPE_CODE[st.type])
        cfirst.append(st.cfirst)
        cnum.append(st.cnum)
        trans_flat.extend(st.transitions)
        trans_off.append(len(trans_flat))
        emit_flat.extend(st.emissions)
        emit_off.append(len(emit_flat))
        state_node.append(st.node)
    node_lcol = [nd.lcol for nd in model.nodes]
    node_rcol = [nd.rcol for nd in model.nodes]
    return (
        types,
        cfirst,
        cnum,
        trans_flat,
        trans_off,
        emit_flat,
        emit_off,
        state_node,
        node_lcol,
        node_rcol,
    )


def cyk_trace(model: CovarianceModel, seq: str):
    """Return (CMHit, alignment) for the best CM alignment to a subsequence.

    ``alignment`` is a list of ``(consensus_column, seq_pos)`` pairs for match
    emissions, sorted by consensus column (5'->3'); ``seq_pos`` is 1-based within
    ``seq``. Insertions are not part of the consensus and are omitted. Returns
    ``(None, [])`` if no alignment exists. Uses the Rust kernel when available.
    """
    if _rust_cm_cyk_trace is not None:
        flat = _flatten_for_trace(model)
        res = _rust_cm_cyk_trace(seq.upper(), *flat)
        if res is None:
            return None, []
        start, end, score, align = res
        hit = CMHit(start=start, end=end, strand=1, score=score, cm_from=1, cm_to=model.clen)
        return hit, [(int(c), int(p)) for c, p in align]
    _warn_python_dp()
    return _cyk_trace_py(model, seq)


def _cyk_trace_py(model: CovarianceModel, seq: str):
    """Pure-Python traceback reference (used when the Rust kernel is absent)."""
    x = _encode(seq)
    L = len(x)
    dp = _forward_dp(model, x)
    root = dp[0]
    best = _NEG_INF
    bi = bj = -1
    for i in range(1, L + 1):
        for j in range(i, L + 1):
            if root[i][j] > best:
                best = root[i][j]
                bi, bj = i, j
    if bi < 0:
        return None, []

    states = model.states
    nodes = model.nodes
    eps = 1e-6
    emissions: list[tuple[int, int]] = []
    stack = [(0, bi, bj)]
    while stack:
        v, i, j = stack.pop()
        st = states[v]
        t = st.type
        kids = _children(st)
        trans = st.transitions
        emit = st.emissions
        node = nodes[st.node] if 0 <= st.node < len(nodes) else None
        target = dp[v][i][j]

        if t == "E" or t == "EL":
            continue
        if t == "S" or t == "D":
            for k, c in enumerate(kids):
                if abs(trans[k] + dp[c][i][j] - target) < eps:
                    stack.append((c, i, j))
                    break
        elif t == "ML" or t == "IL":
            es = emit[x[i - 1]] if x[i - 1] >= 0 else _NEG_INF
            if t == "ML" and node is not None and node.lcol >= 0:
                emissions.append((node.lcol, i))
            for k, c in enumerate(kids):
                if abs(es + trans[k] + dp[c][i + 1][j] - target) < eps:
                    stack.append((c, i + 1, j))
                    break
        elif t == "MR" or t == "IR":
            es = emit[x[j - 1]] if x[j - 1] >= 0 else _NEG_INF
            if t == "MR" and node is not None and node.rcol >= 0:
                emissions.append((node.rcol, j))
            for k, c in enumerate(kids):
                if abs(es + trans[k] + dp[c][i][j - 1] - target) < eps:
                    stack.append((c, i, j - 1))
                    break
        elif t == "MP":
            es = emit[4 * x[i - 1] + x[j - 1]] if (x[i - 1] >= 0 and x[j - 1] >= 0) else _NEG_INF
            if node is not None:
                if node.lcol >= 0:
                    emissions.append((node.lcol, i))
                if node.rcol >= 0:
                    emissions.append((node.rcol, j))
            for k, c in enumerate(kids):
                if abs(es + trans[k] + dp[c][i + 1][j - 1] - target) < eps:
                    stack.append((c, i + 1, j - 1))
                    break
        elif t == "B":
            left, right = kids
            for k in range(i - 1, j + 1):
                if abs(dp[left][i][k] + dp[right][k + 1][j] - target) < eps:
                    stack.append((left, i, k))
                    stack.append((right, k + 1, j))
                    break

    emissions.sort(key=lambda e: e[0])
    hit = CMHit(start=bi, end=bj, strand=1, score=best, cm_from=1, cm_to=model.clen)
    return hit, emissions
