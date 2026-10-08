"""Intron-removal detection for tRNAs with an anticodon-loop intron.

Group-I / anticodon-loop introns (e.g. plant mitochondrial ``trnL-UAA``) break
the covariance-model structure, so RF00005 scores the unspliced gene poorly and
misses it. This module recovers such tRNAs the way tRNAscan-SE does
conceptually: splice out a candidate intron, re-score the spliced *mature* tRNA
with the CM, and if that scores well, report the two mature exons and the intron
as a separate feature.

Coordinates are 1-based inclusive within the supplied region; the caller adds
the genome offset. Mature-exon coordinates are preserved (per project decision);
the intron is reported as its own span, not merged into the tRNA.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cyk import cyk_align
from .models import CovarianceModel
from .naming import name_trna
from .search import _cyk_window, _flatten_model, _rust_cm_cyk


@dataclass
class SplicedTRNAHit:
    exon1_start: int
    exon1_end: int
    intron_start: int
    intron_end: int
    exon2_start: int
    exon2_end: int
    score: float
    # Transcription strand. Exon/intron fields are forward-strand coordinates.
    strand: int = 1


_RC = str.maketrans("ACGTacgt", "TGCAtgca")


def _rc(seq: str) -> str:
    return seq.translate(_RC)[::-1]


def _score(model: CovarianceModel, flat, seq: str) -> float | None:
    if _rust_cm_cyk is not None and flat is not None:
        res = _rust_cm_cyk(seq, *flat, float("-inf"))
        return res[2] if res is not None else None
    hit = cyk_align(model, seq, min_score=float("-inf"))
    return hit.score if hit is not None else None


def detect_spliced_trna(
    region: str,
    model: CovarianceModel,
    *,
    min_bits: float = 25.0,
    min_intron: int = 100,
    min_mature: int = 65,
    max_mature: int = 95,
    e1_range: tuple[int, int] = (28, 50),
    e2_range: tuple[int, int] = (28, 46),
) -> SplicedTRNAHit | None:
    """Detect an anticodon-loop intron in a region bracketing a tRNA gene.

    The region is assumed to start at the 5' exon and end at the 3' exon (the
    gene span). Returns the best-scoring intron splice, or ``None`` if no splice
    yields a mature tRNA scoring at least ``min_bits`` with an intron of at least
    ``min_intron`` nt.
    """
    region = region.upper()
    L = len(region)
    if min_mature + min_intron > L:
        return None
    flat = _flatten_model(model) if _rust_cm_cyk is not None else None

    best: tuple[int, int, float] | None = None
    for e1 in range(e1_range[0], e1_range[1] + 1):
        for e2 in range(e2_range[0], e2_range[1] + 1):
            mature_len = e1 + e2
            if not (min_mature <= mature_len <= max_mature):
                continue
            intron_len = L - e1 - e2
            if intron_len < min_intron:
                continue
            spliced = region[:e1] + region[L - e2 :]
            sc = _score(model, flat, spliced)
            if sc is None:
                continue
            if best is None or sc > best[2]:
                best = (e1, e2, sc)

    if best is None or best[2] < min_bits:
        return None
    e1, e2, sc = best
    return SplicedTRNAHit(
        exon1_start=1,
        exon1_end=e1,
        intron_start=e1 + 1,
        intron_end=L - e2,
        exon2_start=L - e2 + 1,
        exon2_end=L,
        score=sc,
    )


def recover_spliced_trna_near(
    genome: str,
    anchor_start: int,
    anchor_end: int,
    model: CovarianceModel,
    *,
    min_bits: float = 25.0,
    min_intron: int = 100,
    max_intron: int = 2700,
    acceptor_k: int = 7,
) -> SplicedTRNAHit | None:
    """Recover an intron tRNA seeded by a single-exon anchor (1-based, forward).

    An HMM filter anchors only one exon of an intron tRNA. Using acceptor-stem
    complementarity (the mature 5' end pairs with the 3' end), this finds the
    opposite exon, brackets the gene, and runs :func:`detect_spliced_trna`.
    Returns a hit with 1-based genome coordinates, or ``None``.
    """
    genome = genome.upper()
    L = len(genome)
    min_gene = min_intron + 65
    max_gene = max_intron + 95
    regions: set[tuple[int, int]] = set()

    # Interpretation A: the anchor is the 3' exon; the gene lies upstream.
    for off in range(0, 6):
        tail = anchor_end - off
        acc3 = genome[tail - acceptor_k : tail]
        if len(acc3) < acceptor_k:
            continue
        want = _rc(acc3)
        lo = max(0, anchor_end - max_gene)
        hi = max(0, anchor_end - min_gene)
        for p in range(lo, hi):
            if genome[p : p + acceptor_k] == want:
                regions.add((p, anchor_end))

    # Interpretation B: the anchor is the 5' exon; the gene lies downstream.
    for off in range(0, 6):
        head = anchor_start - 1 + off
        acc5 = genome[head : head + acceptor_k]
        if len(acc5) < acceptor_k:
            continue
        want = _rc(acc5)
        lo = anchor_start - 1 + min_gene
        hi = min(L, anchor_start - 1 + max_gene)
        for p in range(lo, hi):
            if genome[p : p + acceptor_k] == want:
                regions.add((anchor_start - 1, p + acceptor_k))

    best: tuple[SplicedTRNAHit, int] | None = None
    for a, b in regions:
        hit = detect_spliced_trna(genome[a:b], model, min_bits=min_bits, min_intron=min_intron)
        if hit is None:
            continue
        if best is None or hit.score > best[0].score:
            best = (hit, a)

    if best is None:
        return None
    hit, a = best
    return SplicedTRNAHit(
        exon1_start=a + hit.exon1_start,
        exon1_end=a + hit.exon1_end,
        intron_start=a + hit.intron_start,
        intron_end=a + hit.intron_end,
        exon2_start=a + hit.exon2_start,
        exon2_end=a + hit.exon2_end,
        score=hit.score,
    )


def recover_spliced_trna_from_exon_pair(
    genome: str,
    exon1: tuple[int, int],
    exon2: tuple[int, int],
    model: CovarianceModel,
    *,
    min_bits: float = 25.0,
    pad: int = 6,
) -> SplicedTRNAHit | None:
    """Recover an intron tRNA from two HMM-anchored exon windows (1-based, forward).

    When a filter HMM anchors *both* exons of an intron tRNA (typical for
    plastid tRNAs), splice the two exon windows directly and score the mature
    tRNA, scanning a small boundary padding on each side. Returns a hit with
    1-based genome coordinates (mature exon coords + the intron between), or
    ``None``. ``exon1`` must be 5' of ``exon2`` on the forward strand.
    """
    genome = genome.upper()
    a1s, a1e = exon1
    a2s, a2e = exon2
    if a2s <= a1e:
        return None

    from .search import _flatten_model, _rust_cm_cyk

    flat = _flatten_model(model) if _rust_cm_cyk is not None else None
    candidates = []  # (score, e1s, e1e, e2s, e2e, strand)
    for p1 in range(-pad, pad + 1):
        for p2 in range(-pad, pad + 1):
            e1s, e1e = a1s, a1e + p1
            e2s, e2e = a2s - p2, a2e
            if e1e < e1s or e2e < e2s or e2s <= e1e:
                continue
            ex1 = genome[e1s - 1 : e1e]
            ex2 = genome[e2s - 1 : e2e]
            mature_fwd = ex1 + ex2
            mature_rev = _rc(ex2) + _rc(ex1)
            for strand, mat in ((1, mature_fwd), (-1, mature_rev)):
                if not (60 <= len(mat) <= 95):
                    continue
                sc = _score(model, flat, mat)
                if sc is not None and sc >= min_bits:
                    candidates.append((sc, e1s, e1e, e2s, e2e, strand))

    # The outer ends are HMM window edges; a local alignment places them.
    splices = []
    for _, e1s, e1e, e2s, e2e, strand in sorted(candidates, reverse=True)[:_NAMING_TRIES]:
        refined = _splice_with_local_ends(genome, e1e, e2s, e1s, e2e, strand, model, flat)
        if refined is not None and refined[0] >= min_bits:
            splices.append((*refined, strand))
    chosen = _pick_named(splices, model)
    if chosen is None:
        return None
    sc, (e1s, e1e), (e2s, e2e), _, strand = chosen
    return SplicedTRNAHit(
        exon1_start=e1s,
        exon1_end=e1e,
        intron_start=e1e + 1,
        intron_end=e2s - 1,
        exon2_start=e2s,
        exon2_end=e2e,
        score=sc,
        strand=strand,
    )


def _splice_with_local_ends(
    genome: str,
    low_end: int,
    high_start: int,
    outer_lo: int,
    outer_hi: int,
    strand: int,
    model: CovarianceModel,
    flat,
    *,
    flank: int = 15,
    min_exon: int = 10,
    mature_range: tuple[int, int] = (60, 95),
) -> tuple[float, tuple[int, int], tuple[int, int], str] | None:
    """Splice at fixed inner sites and let a local CYK alignment place the ends.

    ``genome[outer_lo - flank .. low_end]`` and ``genome[high_start .. outer_hi +
    flank]`` (1-based, forward strand) are joined, oriented to ``strand`` and
    aligned locally, so the outer ends need only be approximately known. The
    mature tRNA must cross the junction with at least ``min_exon`` nt per exon.
    Returns ``(score, low_exon, high_exon, mature)``: exons in forward genome
    coordinates, the mature tRNA in transcription orientation.
    """
    lo = max(1, outer_lo - flank)
    hi = min(len(genome), outer_hi + flank)
    if not (lo <= low_end < high_start <= hi):
        return None
    low = genome[lo - 1 : low_end]
    spliced = low + genome[high_start - 1 : hi]
    oriented = spliced if strand == 1 else _rc(spliced)
    res = _cyk_window(model, flat, oriented, float("-inf"))
    if res is None:
        return None
    hit_start, hit_end, score = res
    n, split = len(spliced), len(low)
    fs, fe = (hit_start, hit_end) if strand == 1 else (n - hit_end + 1, n - hit_start + 1)
    if fs > split - min_exon + 1 or fe < split + min_exon:
        return None
    if not (mature_range[0] <= fe - fs + 1 <= mature_range[1]):
        return None
    mature = oriented[hit_start - 1 : hit_end]
    return score, (lo + fs - 1, low_end), (high_start, high_start + (fe - split) - 1), mature


# Splices tried, best CM score first, before settling for one whose anticodon
# cannot be read. Inner splice sites sit in weakly constrained loops, so a
# splice shifted by a codon can outscore the true one while breaking the loop.
_NAMING_TRIES = 8


# Named splices scoring within this many bits of the best named splice are
# treated as equally supported by the CM and ranked by exon lengths instead.
_LENGTH_TIE_BITS = 5.0


def _exon_lengths(splice) -> tuple[int, int]:
    _, (low_start, low_end), (high_start, high_end), _, strand = splice
    low, high = low_end - low_start + 1, high_end - high_start + 1
    return (low, high) if strand == 1 else (high, low)


def _pick_named(
    splices,
    model: CovarianceModel,
    expected_aa: str | None = None,
    exon_lengths: tuple[int, int] | None = None,
):
    """Choose a splice whose mature tRNA names (to ``expected_aa`` if given).

    ``splices`` are ``(score, low_exon, high_exon, mature, strand)``. Among named
    splices within ``_LENGTH_TIE_BITS`` of the best, the one closest to the
    reference ``exon_lengths`` wins (intron positions are conserved per gene).
    Falls back to the best splice that names at all; a splice whose anticodon
    cannot be read is not reported (every true intron tRNA in the benchmark
    plastomes names, while unreadable splices were spurious).
    """
    ranked = sorted(splices, key=lambda item: item[0], reverse=True)[:_NAMING_TRIES]
    matching, named_any = [], None
    for splice in ranked:
        named = name_trna(model, splice[3])
        if named is None:
            continue
        if expected_aa is None or named[1] == expected_aa:
            matching.append(splice)
        named_any = named_any or splice
    if matching:
        if exon_lengths is None:
            return matching[0]
        top = matching[0][0]
        close = [splice for splice in matching if splice[0] >= top - _LENGTH_TIE_BITS]
        return min(
            close,
            key=lambda splice: sum(
                abs(a - b) for a, b in zip(_exon_lengths(splice), exon_lengths, strict=True)
            ),
        )
    return named_any


def splice_trna_in_span(
    genome: str,
    start: int,
    end: int,
    strand: int,
    exon_lengths: tuple[int, int],
    model: CovarianceModel,
    *,
    min_bits: float = 20.0,
    slack: int = 5,
    min_intron: int = 50,
    expected_aa: str | None = None,
) -> SplicedTRNAHit | None:
    """Splice an intron tRNA whose gene span (1-based, forward) is roughly known.

    ``exon_lengths`` are the (5', 3') exon lengths in transcription order from
    reference annotations of the same tRNA. Inner splice sites are scanned
    within ``slack`` nt of the positions they imply, and the mature ends come
    from a local alignment, so a reference-transferred span suffices even when
    the filter HMM anchored neither exon (short exons, e.g. trnG-UCC's 23 nt).
    Among splices, the best-scoring one whose anticodon reads as
    ``expected_aa`` (e.g. ``"V"``) wins.
    """
    genome = genome.upper()
    five, three = exon_lengths
    low_len, high_len = (five, three) if strand == 1 else (three, five)
    flat = _flatten_model(model) if _rust_cm_cyk is not None else None
    splices = []
    low_guess = start + low_len - 1
    high_guess = end - high_len + 1
    for low_end in range(low_guess - slack, low_guess + slack + 1):
        for high_start in range(high_guess - slack, high_guess + slack + 1):
            if high_start - low_end - 1 < min_intron:
                continue
            got = _splice_with_local_ends(
                genome, low_end, high_start, start, end, strand, model, flat
            )
            if got is not None and got[0] >= min_bits:
                splices.append((*got, strand))
    chosen = _pick_named(splices, model, expected_aa, exon_lengths)
    if chosen is None:
        return None
    score, (e1s, e1e), (e2s, e2e), _, _ = chosen
    return SplicedTRNAHit(
        exon1_start=e1s,
        exon1_end=e1e,
        intron_start=e1e + 1,
        intron_end=e2s - 1,
        exon2_start=e2s,
        exon2_end=e2e,
        score=score,
        strand=strand,
    )
