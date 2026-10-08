"""Vectorised equivalent of :func:`organelleverse.annotation.solver.dp.solve`.

``dp.solve`` is the correctness oracle and builds a Python object per cell,
which is fine for tests but not for a mitochondrial gene window: a 530-residue
cox1 profile over a 10 kb window is ~5 million cells in each of six layers.
This walks the genome one column at a time and updates every profile position
of that column with numpy, keeping only the last four columns of scores plus
compact back-pointers for the traceback.

Same recursion, same tie-breaking (strictly greater wins, in the oracle's
order), same traceback -- ``tests/annotation/solver/test_solver_vector.py``
holds it to ``dp.solve`` structure-for-structure. The one arithmetic difference
is the within-column deletion chain, computed as a running maximum instead of a
loop, so scores agree to floating-point rounding rather than bit-for-bit.
"""

from __future__ import annotations

import numpy as np

from .dp import (
    _ACCEPTOR_BONUS,
    _DONOR_BONUS,
    AMINO_ACIDS,
    START_CODONS,
    STOP_CODONS,
    Exon,
    Profile,
    Solution,
    translate,
)

_NEG = -np.inf
_BASE_CODE = {b: n for n, b in enumerate("ACGT")}  # anything else -> 4
_LETTERS = "ACGTN"


def _codon_tables() -> tuple[np.ndarray, np.ndarray]:
    """aa column (20 = unknown/X) and stop flag for each 5-letter codon code."""
    aa = np.full(125, 20, dtype=np.int64)
    stop = np.zeros(125, dtype=bool)
    for code in range(125):
        codon = _LETTERS[code // 25] + _LETTERS[(code // 5) % 5] + _LETTERS[code % 5]
        stop[code] = codon in STOP_CODONS
        letter = translate(codon)
        if letter in AMINO_ACIDS:
            aa[code] = AMINO_ACIDS.index(letter)
    return aa, stop


def _dinuc_table(bonus: dict[str, float]) -> np.ndarray:
    table = np.zeros(25)
    for pair, value in bonus.items():
        table[_BASE_CODE[pair[0]] * 5 + _BASE_CODE[pair[1]]] = value
    return table


_CODON_AA, _CODON_STOP = _codon_tables()
_DONOR = _dinuc_table(_DONOR_BONUS)
_ACCEPTOR = _dinuc_table(_ACCEPTOR_BONUS)


def _encode(seq: str) -> np.ndarray:
    codes = np.full(len(seq), 4, dtype=np.int64)
    raw = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
    for base, code in _BASE_CODE.items():
        codes[raw == ord(base)] = code
    return codes


def solve_vector(
    profile: Profile,
    seq: str,
    *,
    min_intron: int = 400,
    max_intron: int = 6000,
    allow_introns: bool = True,
    require_start: bool = True,
    require_stop: bool = True,
) -> Solution | None:
    """Best gene structure for ``profile`` in ``seq`` (coding strand); see ``dp.solve``."""
    seq = seq.upper()
    L, M = len(seq), len(profile)
    if L < 3 or M == 0:
        return None

    scores = np.full((M + 1, 21), -4.0)
    scores[1:, :20] = np.asarray(profile.scores, dtype=float)
    g = _encode(seq)
    starts = np.array([seq[j : j + 3] in START_CODONS for j in range(L + 1)])
    stops_next = np.array([seq[j : j + 3] in STOP_CODONS for j in range(L + 1)])

    ring = 4
    mat = np.full((ring, M + 1), _NEG)
    dele = np.full((ring, M + 1), _NEG)
    ins = np.full((ring, M + 1), _NEG)
    intr = np.full((3, ring, M + 1), _NEG)
    intr_start = np.zeros((3, ring, M + 1), dtype=np.int64)

    # back-pointers: match 0/1/2 contiguous from M/D/I, 3+p split; delete/insert 0 M, 1 D|I
    mcode = np.full((L + 1, M + 1), -1, dtype=np.int8)
    mstart = np.zeros((L + 1, M + 1), dtype=np.int32)
    dcode = np.full((L + 1, M + 1), -1, dtype=np.int8)
    icode = np.full((L + 1, M + 1), -1, dtype=np.int8)

    def delete_column(m_col: np.ndarray, c: int, j: int) -> None:
        """delete[i] = max(match[i-1] + open, delete[i-1] + extend), i = 1..M."""
        opened = m_col[:-1] + profile.gap_open  # candidate for rows 1..M
        ext = profile.gap_extend
        # running maximum of (opened[k] - k*ext), re-offset by the extension count
        k = np.arange(M)
        running = np.maximum.accumulate(opened - k * ext)
        d = running + k * ext
        d[~np.isfinite(running)] = _NEG
        col = np.full(M + 1, _NEG)
        col[1:] = d
        dele[c] = col
        from_d = np.zeros(M + 1, dtype=bool)
        from_d[2:] = (col[1:-1] + ext) > opened[1:]
        dcode[j] = np.where(np.isfinite(col), np.where(from_d, 1, 0), -1)

    # column 0
    mat[0] = _NEG
    if not require_start or starts[0]:
        mat[0][0] = 0.0
    delete_column(mat[0], 0, 0)

    best_score, best_pos, best_layer = _NEG, (0, 0), "M"

    for j in range(1, L + 1):
        c = j % ring
        m_new = np.full(M + 1, _NEG)
        m_code = np.full(M + 1, -1, dtype=np.int8)
        m_start = np.zeros(M + 1, dtype=np.int32)
        if not require_start or starts[j]:
            m_new[0] = 0.0

        codon_ok = False
        if j >= 3:
            code = g[j - 3] * 25 + g[j - 2] * 5 + g[j - 1]
            codon_ok = not _CODON_STOP[code]
            if codon_ok:
                c3 = (j - 3) % ring
                src_m, src_d, src_i = mat[c3][:-1], dele[c3][:-1], ins[c3][:-1]
                cand = src_m.copy()
                which = np.zeros(M, dtype=np.int8)
                take = src_d > cand
                cand[take], which[take] = src_d[take], 1
                take = src_i > cand
                cand[take], which[take] = src_i[take], 2
                ok = np.isfinite(cand)
                m_new[1:][ok] = cand[ok] + scores[1:, _CODON_AA[code]][ok]
                m_code[1:][ok] = which[ok]

        if allow_introns:
            for p in (0, 1, 2):
                tail = 3 - p
                close_at = j - tail
                if close_at < 1:
                    continue
                cc = close_at % ring
                cell = intr[p][cc][:-1]  # rows 0..M-1 feed rows 1..M
                st = intr_start[p][cc][:-1]
                length = close_at - st + 1
                ok = np.isfinite(cell) & (length >= min_intron) & (length <= max_intron)
                ok &= (st - p - 1) >= 0
                if not ok.any():
                    continue
                idx = np.nonzero(ok)[0]
                s = st[idx]
                bases = [g[s - p - 1 + t] for t in range(p)] + [
                    np.full(len(idx), g[close_at + t]) for t in range(tail)
                ]
                code = bases[0] * 25 + bases[1] * 5 + bases[2]
                legal = ~_CODON_STOP[code]
                idx, s, code = idx[legal], s[legal], code[legal]
                if len(idx) == 0:
                    continue
                donor_ok = s < L  # seq[start-1:start+1] needs two bases
                donor = np.zeros(len(idx))
                donor[donor_ok] = _DONOR[g[s[donor_ok] - 1] * 5 + g[s[donor_ok]]]
                acceptor = (
                    _ACCEPTOR[g[close_at - 2] * 5 + g[close_at - 1]] if close_at >= 2 else 0.0
                )
                total = cell[idx] + scores[idx + 1, _CODON_AA[code]] + donor + acceptor
                better = total > m_new[idx + 1]
                rows_i = idx[better] + 1
                m_new[rows_i] = total[better]
                m_code[rows_i] = 3 + p
                m_start[rows_i] = s[better]

        mat[c] = m_new
        mcode[j], mstart[j] = m_code, m_start

        i_new = np.full(M + 1, _NEG)
        i_code = np.full(M + 1, -1, dtype=np.int8)
        if j >= 3 and codon_ok:
            c3 = (j - 3) % ring
            opened = mat[c3][1:] + profile.insert_open
            ok = opened > i_new[1:]
            i_new[1:][ok], i_code[1:][ok] = opened[ok], 0
            ext = ins[c3][1:] + profile.insert_extend
            ok = ext > i_new[1:]
            i_new[1:][ok], i_code[1:][ok] = ext[ok], 1
        ins[c] = i_new
        icode[j] = i_code

        delete_column(mat[c], c, j)

        if allow_introns:
            for p in (0, 1, 2):
                col = np.full(M + 1, _NEG)
                col_start = np.zeros(M + 1, dtype=np.int64)
                if j - p >= 0:
                    opened = mat[(j - p) % ring].copy()
                    if p == 0:
                        opened = np.where(dele[c] > opened, dele[c], opened)
                    opened[0] = _NEG  # the oracle's row loop starts at 1
                    ok = np.isfinite(opened) & (opened + profile.intron_open > col)
                    col[ok] = opened[ok] + profile.intron_open
                    col_start[ok] = j + 1
                prev = intr[p][(j - 1) % ring]
                ext = prev + profile.intron_extend
                ok = np.isfinite(prev) & (ext > col)
                col[ok] = ext[ok]
                col_start[ok] = intr_start[p][(j - 1) % ring][ok]
                intr[p][c] = col
                intr_start[p][c] = col_start

        if not require_stop or stops_next[j]:
            for layer, value in (("M", mat[c][M]), ("D", dele[c][M]), ("I", ins[c][M])):
                if value > best_score:
                    best_score, best_pos, best_layer = value, (M, j), layer

    if not best_score > 0:
        return None
    exons = _traceback(mcode, mstart, dcode, icode, best_pos, best_layer)
    sol = Solution(score=float(best_score), exons=exons, tier=1)
    if require_stop and sol.exons:
        sol.exons[-1] = Exon(sol.exons[-1].start, sol.exons[-1].end + 3)
    return sol


def _traceback(mcode, mstart, dcode, icode, pos, layer) -> list[Exon]:
    """Replay the stored back-pointers exactly as ``dp._traceback`` does."""
    i, j = pos
    spans: list[tuple[int, int]] = []
    while i > 0 or layer == "I":
        if layer == "M":
            code = int(mcode[j][i])
            if code < 0:
                break
            if code < 3:
                spans.append((j - 2, j))
                layer, i, j = "MDI"[code], i - 1, j - 3
            else:
                p = code - 3
                tail = 3 - p
                close_at = j - tail
                start = int(mstart[j][i])
                if p:
                    spans.append((start - p, start - 1))
                spans.append((close_at + 1, close_at + tail))
                layer, i, j = "M", i - 1, start - p - 1 if p else start - 1
        elif layer == "D":
            code = int(dcode[j][i])
            if code < 0:
                break
            layer, i = ("M" if code == 0 else "D"), i - 1
        else:  # insert
            code = int(icode[j][i])
            if code < 0:
                break
            spans.append((j - 2, j))
            layer, j = ("M" if code == 0 else "I"), j - 3
    spans.sort()
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [Exon(a, b) for a, b in merged if b >= a and a >= 1]
