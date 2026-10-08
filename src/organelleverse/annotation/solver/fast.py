"""JIT-compiled genome-wide scan, numerically identical to the pure-Python one.

The reference implementation needs 6.5-23 minutes per gene to scan a 368 kb
genome on both strands, which puts a fair comparison against tools that scan
whole genomes -- miniprot does the same genome in 16 s -- out of reach until the
Rust kernel exists. This closes that gap in Python: the recursion is a tight
numeric loop over arrays, which is what numba is for.

Correctness is the point, not speed: :func:`scan_fast` must agree with
``dp.scan`` score-for-score, and :func:`verify_against_reference` asserts it.
Everything Python-level is hoisted out -- the genome becomes an int8 array, the
profile a 2-D float array, and codon translation a 64-entry lookup -- so the
kernel touches no Python objects at all.
"""

from __future__ import annotations

import numpy as np

from .dp import (
    AMINO_ACIDS,
    Profile,
    START_CODONS,
    STOP_CODONS,
    _ACCEPTOR_BONUS,
    _DONOR_BONUS,
    translate,
)

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is optional
    njit = None

NEG = -1e18

_BASE = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_genome(seq: str) -> np.ndarray:
    """Genome as int8 with 4 for anything ambiguous."""
    out = np.full(len(seq), 4, dtype=np.int8)
    for base, code in _BASE.items():
        out[np.frombuffer(seq.encode(), dtype=np.uint8) == ord(base)] = code
    return out


def _codon_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lookups keyed by ``16*b1 + 4*b2 + b3``; index 64 means ambiguous."""
    aa_index = np.full(65, -2, dtype=np.int8)      # -2 ambiguous, -1 stop
    stop = np.zeros(65, dtype=np.bool_)
    start = np.zeros(65, dtype=np.bool_)
    bases = "ACGT"
    for i, b1 in enumerate(bases):
        for j, b2 in enumerate(bases):
            for k, b3 in enumerate(bases):
                codon = b1 + b2 + b3
                idx = 16 * i + 4 * j + k
                if codon in STOP_CODONS:
                    aa_index[idx] = -1
                    stop[idx] = True
                else:
                    aa_index[idx] = AMINO_ACIDS.index(translate(codon))
                start[idx] = codon in START_CODONS
    return aa_index, stop, start


def _dinuc_tables() -> tuple[np.ndarray, np.ndarray]:
    """Splice bonuses keyed by ``4*b1 + b2``; index 16 means ambiguous."""
    donor = np.zeros(17, dtype=np.float64)
    acceptor = np.zeros(17, dtype=np.float64)
    bases = "ACGT"
    for i, b1 in enumerate(bases):
        for j, b2 in enumerate(bases):
            donor[4 * i + j] = _DONOR_BONUS.get(b1 + b2, 0.0)
            acceptor[4 * i + j] = _ACCEPTOR_BONUS.get(b1 + b2, 0.0)
    return donor, acceptor


AA_INDEX, IS_STOP, IS_START, = _codon_tables()
DONOR, ACCEPTOR = _dinuc_tables()


def _kernel(gen, scores, aa_index, is_stop, is_start, donor, acceptor,
            gap_open, gap_extend, insert_open, insert_extend,
            intron_open, intron_extend, min_intron, max_intron,
            out_score, out_end):
    L = gen.shape[0]
    M = scores.shape[0]
    W = 4

    mat = np.full((W, M + 1), NEG)
    dele = np.full((W, M + 1), NEG)
    ins = np.full((W, M + 1), NEG)
    intr = np.full((3, W, M + 1), NEG)
    istart = np.zeros((3, W, M + 1), dtype=np.int64)
    n_out = 0

    for j in range(L + 1):
        c = j % W
        for i in range(M + 1):
            mat[c, i] = NEG
            dele[c, i] = NEG
            ins[c, i] = NEG
            for p in range(3):
                intr[p, c, i] = NEG

        # entry: only at an initiator
        cs = 64
        if j + 3 <= L:
            b1, b2, b3 = gen[j], gen[j + 1], gen[j + 2]
            if b1 < 4 and b2 < 4 and b3 < 4:
                cs = 16 * b1 + 4 * b2 + b3
        mat[c, 0] = 0.0 if cs < 64 and is_start[cs] else NEG

        cc = 64
        if j >= 3:
            b1, b2, b3 = gen[j - 3], gen[j - 2], gen[j - 1]
            if b1 < 4 and b2 < 4 and b3 < 4:
                cc = 16 * b1 + 4 * b2 + b3
        legal = cc < 64 and not is_stop[cc]
        aa = aa_index[cc] if legal else -2
        c3 = (j - 3) % W

        for i in range(1, M + 1):
            if legal:
                sc = scores[i - 1, aa] if aa >= 0 else -4.0
                best = mat[c3, i - 1]
                if dele[c3, i - 1] > best:
                    best = dele[c3, i - 1]
                if ins[c3, i - 1] > best:
                    best = ins[c3, i - 1]
                if best > NEG:
                    mat[c, i] = best + sc

                for p in range(3):
                    tail = 3 - p
                    ca = j - tail
                    if ca < 1:
                        continue
                    cj = ca % W
                    s = intr[p, cj, i - 1]
                    if s <= NEG:
                        continue
                    st = istart[p, cj, i - 1]
                    ln = ca - st + 1
                    if ln < min_intron or ln > max_intron:
                        continue
                    # rebuild the codon split across the intron
                    ok = True
                    idx = 0
                    for q in range(p):
                        b = gen[st - p - 1 + q]
                        if b > 3:
                            ok = False
                            break
                        idx = idx * 4 + b
                    if not ok:
                        continue
                    for q in range(tail):
                        b = gen[ca + q]
                        if b > 3:
                            ok = False
                            break
                        idx = idx * 4 + b
                    if not ok or is_stop[idx]:
                        continue
                    a2 = aa_index[idx]
                    v = s + (scores[i - 1, a2] if a2 >= 0 else -4.0)
                    d1, d2 = gen[st - 1], gen[st]
                    if d1 < 4 and d2 < 4:
                        v += donor[4 * d1 + d2]
                    a3, a4 = gen[ca - 2], gen[ca - 1]
                    if a3 < 4 and a4 < 4:
                        v += acceptor[4 * a3 + a4]
                    if v > mat[c, i]:
                        mat[c, i] = v

                if mat[c3, i] > NEG:
                    v = mat[c3, i] + insert_open
                    if v > ins[c, i]:
                        ins[c, i] = v
                if ins[c3, i] > NEG:
                    v = ins[c3, i] + insert_extend
                    if v > ins[c, i]:
                        ins[c, i] = v

            v = mat[c, i - 1] + gap_open
            w = dele[c, i - 1] + gap_extend
            dele[c, i] = v if v > w else w

            for p in range(3):
                base = NEG
                if j - p >= 0:
                    base = mat[(j - p) % W, i]
                if p == 0 and dele[c, i] > base:
                    base = dele[c, i]
                if base > NEG and base + intron_open > intr[p, c, i]:
                    intr[p, c, i] = base + intron_open
                    istart[p, c, i] = j + 1
                pv = intr[p, (j - 1) % W, i]
                if pv > NEG and pv + intron_extend > intr[p, c, i]:
                    intr[p, c, i] = pv + intron_extend
                    istart[p, c, i] = istart[p, (j - 1) % W, i]

        if cs < 64 and is_stop[cs]:
            e = mat[c, M]
            if dele[c, M] > e:
                e = dele[c, M]
            if ins[c, M] > e:
                e = ins[c, M]
            if e > 0.0 and n_out < out_score.shape[0]:
                out_score[n_out] = e
                out_end[n_out] = j
                n_out += 1
    return n_out


_kernel_jit = njit(cache=True, fastmath=False)(_kernel) if njit else None


def scan_fast(
    profile: Profile,
    seq: str,
    *,
    min_intron: int = 400,
    max_intron: int = 6000,
    top_n: int = 20,
    min_separation: int | None = None,
) -> list[tuple[float, int]]:
    """Same contract as ``dp.scan``, compiled."""
    gen = encode_genome(seq.upper())
    scores = np.asarray(profile.scores, dtype=np.float64)
    M = scores.shape[0]
    cap = len(seq) // 3 + 16
    out_score = np.zeros(cap, dtype=np.float64)
    out_end = np.zeros(cap, dtype=np.int64)

    run = _kernel_jit if _kernel_jit is not None else _kernel
    n = run(gen, scores, AA_INDEX, IS_STOP, IS_START, DONOR, ACCEPTOR,
            profile.gap_open, profile.gap_extend,
            profile.insert_open, profile.insert_extend,
            profile.intron_open, profile.intron_extend,
            min_intron, max_intron, out_score, out_end)

    hits = sorted(zip(out_score[:n].tolist(), out_end[:n].tolist()), reverse=True)
    sep = min_separation if min_separation is not None else 3 * M
    kept: list[tuple[float, int]] = []
    for s, e in hits:
        if all(abs(e - k[1]) > sep for k in kept):
            kept.append((s, e))
        if len(kept) >= top_n:
            break
    return kept


def verify_against_reference(profile: Profile, seq: str, **kw) -> None:
    """Assert the compiled kernel reproduces the reference scores exactly."""
    from .dp import scan

    a = scan(profile, seq, **kw)
    b = scan_fast(profile, seq, **kw)
    assert len(a) == len(b), f"hit count differs: {len(a)} vs {len(b)}"
    for (sa, ea), (sb, eb) in zip(a, b):
        assert ea == eb, f"end differs: {ea} vs {eb}"
        assert abs(sa - sb) < 1e-6, f"score differs at {ea}: {sa} vs {sb}"
