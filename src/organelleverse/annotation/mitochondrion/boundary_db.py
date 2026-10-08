"""Reference-anchored exon/intron boundary finder for mitochondrial CDS.

Plant-mito intron boundaries are frequently non-canonical (group II introns end
in AY, not AG), so canonical GT..AG search is unreliable. Following the MGAVAS
approach, we instead pin each junction by aligning it against a database of
conserved boundary flanks (20 bp exon + 10 bp intron, mRNA-sense) collected from
hundreds of reference species (see ``scripts/annotation_benchmarks/
build_mito_boundary_db.py``). This replaces hardcoded per-gene coordinate offsets
with a data-driven, species-general refinement.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from Bio.Seq import Seq

EXON_FLANK = 20
INTRON_FLANK = 10
_FLANK_LEN = EXON_FLANK + INTRON_FLANK
_DEFAULT_MARGIN = 30
# Only move a boundary when the new position is a strong consensus match AND
# clearly better than the current one — otherwise leave an already-good boundary
# alone (moving correct boundaries to nearby peaks is a net regression).
_MIN_IDENTITY = 0.85
_MIN_IMPROVEMENT = 0.10


def default_boundary_db_path() -> Path:
    # boundary_db.py lives in annotation/mitochondrion/; the data dir is the
    # sibling annotation/data/mitochondrion/ (one level up).
    return (
        Path(__file__).resolve().parent.parent
        / "data"
        / "mitochondrion"
        / "boundary_db"
        / "mito_exon_intron_boundaries.fasta"
    )


def _normalize_gene(raw: str) -> str:
    g = re.sub(r"[^a-z0-9]", "", (raw or "").split()[0].lower()) if raw else ""
    m = re.match(r"nd(\d.*)", g)
    if m:
        g = "nad" + m.group(1)
    if g == "cox21":
        g = "cox2"
    return g


def _build_pwm(flanks: list[str]) -> tuple[tuple[float, float, float, float], ...]:
    """Position frequency matrix (A,C,G,T) over anchored flanks, per column."""
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    counts = [[0, 0, 0, 0] for _ in range(_FLANK_LEN)]
    n = 0
    for flank in flanks:
        if len(flank) != _FLANK_LEN:
            continue
        n += 1
        for pos, base in enumerate(flank):
            b = idx.get(base)
            if b is not None:
                counts[pos][b] += 1
    if n == 0:
        return ()
    return tuple(tuple(c / n for c in col) for col in counts)


@lru_cache(maxsize=4)
def _load_db(path_str: str) -> dict[tuple[str, str], tuple[tuple[float, float, float, float], ...]]:
    """Map (gene, side) -> PWM over the boundary flanks. Side is 'R'/'L'."""
    raw: dict[tuple[str, str], list[str]] = defaultdict(list)
    path = Path(path_str)
    if not path.exists():
        return {}
    name = None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:]
        elif name:
            fields = name.split("|")
            if len(fields) >= 3 and len(line) == _FLANK_LEN:
                raw[(fields[0], fields[2])].append(line.upper())
            name = None
    return {k: _build_pwm(v) for k, v in raw.items() if v}


def _best_junction(
    window: str,
    pwm: tuple[tuple[float, float, float, float], ...],
    exon_left: bool,
    center_offset: int,
) -> tuple[int, float, float] | None:
    """Slide the PWM over window; return (junction_index, best_score, center_score).

    All flanks are anchored at the junction, so the PWM captures the conserved
    boundary motif; the best-scoring window offset locates the true junction.
    ``center_offset`` is the flank offset corresponding to the *current* boundary
    (no change) so the caller can require the move to be a real improvement.
    ``exon_left`` True for donor flanks ([exon|intron], junction after EXON_FLANK),
    False for acceptor flanks ([intron|exon], junction after INTRON_FLANK).
    """
    if not pwm:
        return None
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    junction_in_flank = EXON_FLANK if exon_left else INTRON_FLANK
    max_score = sum(max(col) for col in pwm) or 1.0

    def score_at(k: int) -> float:
        seg = window[k : k + _FLANK_LEN]
        s = 0.0
        for pos, base in enumerate(seg):
            b = idx.get(base)
            if b is not None:
                s += pwm[pos][b]
        return s / max_score

    best = None
    span = len(window) - _FLANK_LEN
    for k in range(span + 1):
        norm = score_at(k)
        if best is None or norm > best[1]:
            best = (k + junction_in_flank, norm)
    center_score = score_at(center_offset)
    return (best[0], best[1], center_score)


def refine_exon_boundaries(
    gene: str,
    exons: list[tuple[int, int]],
    genome: str,
    strand: int,
    *,
    db_path: str | None = None,
    margin: int = _DEFAULT_MARGIN,
) -> list[tuple[int, int]] | None:
    """Refine internal exon boundaries of a multi-exon mito CDS.

    ``exons`` are 1-based (start, end) genomic coordinates in **transcription
    order** (as GenBank CompoundLocation lists them). Returns refined exons in the
    same order, or None if the gene has no boundary flanks / nothing was refined.
    The 5' start of the first exon and 3' end of the last exon are left untouched
    (start/stop-codon refinement handles those).
    """
    if len(exons) < 2:
        return None
    db = _load_db(db_path or str(default_boundary_db_path()))
    g = _normalize_gene(gene)
    r_flanks = db.get((g, "R"))
    l_flanks = db.get((g, "L"))
    if not r_flanks and not l_flanks:
        return None

    L = len(genome)
    # Work in mRNA-sense (forward) coordinates: use rc space for the minus strand.
    if strand == -1:
        seq = str(Seq(genome).reverse_complement())
        fwd = [(L - e + 1, L - s + 1) for s, e in exons]  # ascending, transcription order
    else:
        seq = genome
        fwd = [(s, e) for s, e in exons]

    def accept(hit) -> bool:
        # hit = (junction_index, best_score, center_score)
        return hit is not None and hit[1] >= _MIN_IDENTITY and hit[1] - hit[2] >= _MIN_IMPROVEMENT

    refined = [list(x) for x in fwd]
    changed = False
    for j in range(len(refined) - 1):
        # Donor: 3' end of exon j (mRNA-sense forward coordinate).
        if r_flanks:
            e_end = refined[j][1]
            w0 = e_end - EXON_FLANK - margin  # 0-based window start
            if w0 >= 0 and w0 + _FLANK_LEN + 2 * margin <= len(seq):
                window = seq[w0 : w0 + _FLANK_LEN + 2 * margin]
                hit = _best_junction(window, r_flanks, True, margin)
                if accept(hit):
                    new_end = w0 + hit[0]  # 1-based genomic (fwd)
                    if refined[j][0] < new_end < refined[j + 1][1]:
                        refined[j][1] = new_end
                        changed = True
        # Acceptor: 5' start of exon j+1.
        if l_flanks:
            s_start = refined[j + 1][0]
            w0 = s_start - 1 - INTRON_FLANK - margin
            if w0 >= 0 and w0 + _FLANK_LEN + 2 * margin <= len(seq):
                window = seq[w0 : w0 + _FLANK_LEN + 2 * margin]
                hit = _best_junction(window, l_flanks, False, margin)
                if accept(hit):
                    new_start = w0 + hit[0] + 1  # 1-based genomic (fwd)
                    if refined[j][1] < new_start < refined[j + 1][1]:
                        refined[j + 1][0] = new_start
                        changed = True
    if not changed:
        return None

    fwd_out = [(a, b) for a, b in refined]
    if strand == -1:
        return [(L - b + 1, L - a + 1) for a, b in fwd_out]
    return fwd_out
