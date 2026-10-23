"""Genome-scale native tRNA search.

Two-stage, fully native (no external program):

1. **Fast filter** — a packaged CC0 tRNA profile HMM (Rfam RF00005 seed) is run
   with ``pyhmmer`` ``nhmmer`` to find candidate windows across a whole genome in
   ~0.1 s.
2. **Accurate localization** — the covariance-model CYK aligner runs on each
   candidate window (Rust-accelerated) to give coordinate-exact mature tRNA
   boundaries.

This makes the coordinate-parity CM engine practical at genome scale, where a
naive windowed CYK sweep would be far too slow. Divergent plant-mitochondrial
intron tRNAs that RF00005 cannot model are a known limitation (they need a
plant-mito-specific model) and are handled separately by the intron-removal
detector where an anchor exists.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pyhmmer

from .model import parse_cm
from .models import CMHit, CovarianceModel
from .parallel import pmap
from .search import _resolve_overlaps, cm_search

logger = logging.getLogger(__name__)

_DATA = Path(__file__).resolve().parent.parent / "data" / "models" / "trna"

#: Seconds the covariance-model alignment of one genome may take; unset means no limit.
#: Windows not started before the deadline are skipped (their tRNAs are missing, said in a warning).
BUDGET_ENV = "ORGANELLEVERSE_CM_TIME_BUDGET_S"


def _deadline() -> float | None:
    raw = os.environ.get(BUDGET_ENV, "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number of seconds; no time budget applied", BUDGET_ENV, raw)
        return None
    return time.monotonic() + max(0.0, seconds)


def default_cm_path() -> Path:
    """Plant-mito-specific CM if packaged, else the general RF00005 CM."""
    plant_mito = _DATA / "plant_mito_trna.cm"
    return plant_mito if plant_mito.exists() else _DATA / "trna.cm"


def default_hmm_path() -> Path:
    """Plant-mito-specific HMM filter if packaged, else the general RF00005 HMM."""
    plant_mito = _DATA / "plant_mito_trna.hmm"
    return plant_mito if plant_mito.exists() else _DATA / "trna.hmm"


def default_plastid_cm_path() -> Path:
    """Plastid-specific CM if packaged, else the general RF00005 CM."""
    plastid = _DATA / "plastid_trna.cm"
    return plastid if plastid.exists() else _DATA / "trna.cm"


def default_plastid_hmm_path() -> Path:
    """Plastid-specific HMM filter if packaged, else the general RF00005 HMM."""
    plastid = _DATA / "plastid_trna.hmm"
    return plastid if plastid.exists() else _DATA / "trna.hmm"


def _nhmmer_windows(genome: str, hmm_path: Path, evalue: float) -> list[tuple[int, int]]:
    alphabet = pyhmmer.easel.Alphabet.dna()
    with pyhmmer.plan7.HMMFile(str(hmm_path)) as fh:
        hmm = fh.read()
    target = pyhmmer.easel.TextSequence(name=b"genome", sequence=genome).digitize(alphabet)
    windows: list[tuple[int, int]] = []
    for top in pyhmmer.nhmmer([hmm], [target], E=evalue):
        for hit in top:
            for dom in hit.domains:
                ali = dom.alignment
                s = min(ali.target_from, ali.target_to)
                e = max(ali.target_from, ali.target_to)
                windows.append((s, e))
    return windows


def find_trnas(
    genome: str,
    *,
    cm_model: CovarianceModel | None = None,
    hmm_path: Path | None = None,
    min_bits: float = 20.0,
    filter_evalue: float = 10.0,
    pad: int = 20,
    recover_intron_pairs: bool = False,
    min_intron: int = 100,
    max_intron: int = 3000,
) -> list[CMHit]:
    """Find tRNAs genome-wide: HMM pre-filter then CM CYK localization.

    Returns coordinate-exact mature tRNA hits (1-based inclusive, genome
    coordinates), de-overlapped and filtered to ``min_bits``. When
    ``recover_intron_pairs`` is set (recommended for plastids, whose filter HMM
    anchors both exons of intron tRNAs), pairs of nearby filter windows that
    yield no mature hit individually are spliced and re-scored to recover
    intron-containing tRNAs; recovered hits carry both exons in ``CMHit.exons``.
    """
    genome = genome.upper().replace("U", "T")
    L = len(genome)
    if L == 0:
        return []
    if hmm_path is None:
        hmm_path = default_hmm_path()
    if cm_model is None:
        cm_model = parse_cm(default_cm_path())

    windows = _nhmmer_windows(genome, hmm_path, filter_evalue)
    deadline = _deadline()
    started = time.monotonic()
    skipped: list[int] = []
    logger.info(
        "tRNA search: %d candidate windows to align on %d bp%s",
        len(windows),
        L,
        "" if deadline is None else f" (time budget {deadline - started:.0f} s)",
    )

    def search_window(window: tuple[int, int]) -> tuple[int, list[CMHit]]:
        s, e = window
        a = max(0, s - 1 - pad)
        if deadline is not None and time.monotonic() > deadline:
            skipped.append(a)
            return a, []
        b = min(L, e + pad)
        return a, cm_search(genome[a:b], cm_model, min_bits=min_bits)

    hits: list[CMHit] = []
    for a, found in pmap(search_window, windows):  # windows are independent; order kept
        for h in found:
            hits.append(
                CMHit(
                    start=a + h.start,
                    end=a + h.end,
                    strand=h.strand,
                    score=h.score,
                    cm_from=h.cm_from,
                    cm_to=h.cm_to,
                )
            )

    logger.info(
        "tRNA search: %d of %d windows aligned in %.1f s",
        len(windows) - len(skipped),
        len(windows),
        time.monotonic() - started,
    )
    if skipped:
        logger.warning(
            "tRNA search stopped by the %s time budget: %d of %d candidate windows were not "
            "aligned, so tRNAs in them are missing from the result",
            BUDGET_ENV,
            len(skipped),
            len(windows),
        )
    if recover_intron_pairs and not (deadline is not None and time.monotonic() > deadline):
        hits.extend(
            _recover_intron_pairs(
                genome,
                windows,
                hits,
                cm_model,
                min_bits=min_bits,
                min_intron=min_intron,
                max_intron=max_intron,
                deadline=deadline,
            )
        )
    return _resolve_overlaps(hits)


def _recover_intron_pairs(
    genome, windows, mature_hits, cm_model, *, min_bits, min_intron, max_intron, deadline=None
):
    from .introns import recover_spliced_trna_from_exon_pair

    def _covered(win):
        s, e = win
        return any(not (e < h.start or s > h.end) for h in mature_hits)

    free = sorted((w for w in windows if not _covered(w)), key=lambda w: w[0])
    recovered: list[CMHit] = []
    used: set[tuple[int, int]] = set()
    for i, w1 in enumerate(free):
        if w1 in used:
            continue
        for w2 in free[i + 1 :]:
            if w2 in used:
                continue
            gap = w2[0] - w1[1]
            if gap < min_intron:
                continue
            if gap > max_intron:
                break
            if deadline is not None and time.monotonic() > deadline:
                logger.warning("intron-tRNA recovery stopped by the %s time budget", BUDGET_ENV)
                return recovered
            hit = recover_spliced_trna_from_exon_pair(genome, w1, w2, cm_model, min_bits=min_bits)
            if hit is not None:
                recovered.append(
                    CMHit(
                        start=hit.exon1_start,
                        end=hit.exon2_end,
                        strand=hit.strand,
                        score=hit.score,
                        cm_from=1,
                        cm_to=cm_model.clen,
                        exons=(
                            (hit.exon1_start, hit.exon1_end),
                            (hit.exon2_start, hit.exon2_end),
                        ),
                    )
                )
                used.add(w1)
                used.add(w2)
                break
    return recovered
