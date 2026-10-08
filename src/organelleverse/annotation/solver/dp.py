"""Reference profile-to-genome dynamic program with reading-frame constraints.

This is the correctness oracle for the Rust kernel that will replace it, and it
is written for clarity over speed: full matrices, no Hirschberg, no SIMD.

Why a custom DP rather than an off-the-shelf aligner: an external aligner returns
*its* optimal alignment, which cannot be made to satisfy a constraint afterwards.
Here the constraint lives in the recursion, so a structure that violates it is
not scored lower -- it is not in the search space.

The state space, for profile position ``i`` and genome position ``j``:

===========  ==================================================================
``M[i][j]``  profile position ``i`` matched to the codon ending at ``j``
``I[i][j]``  a codon inserted relative to the profile
``D[i][j]``  profile position consumed with no genome codon
``T{p}``     inside an intron, ``p in {0,1,2}`` bases of the interrupted codon
             already read before the intron opened
===========  ==================================================================

Splitting the intron state three ways by phase is what makes phase-preserving
introns fall out of the recursion instead of being repaired afterwards, and it is
also why arbitrarily short exons cost nothing here: a read aligner cannot place
an exon shorter than its anchor length -- HISAT2 reports no evidence at all for
either intron flanking the 69 bp exon 4 of *Arabidopsis* nad7, even at 69x -- but
a per-position DP has no such notion.

What is hard and what is not:

* **Hard** -- no in-frame stop codon before the terminal one. C-to-U editing only
  *creates* stops (CAA/CAG/CGA -> UAA/UAG/UGA; TAA/TAG/TGA contain no editable
  C), so this holds on unedited DNA with no editing prediction involved.
  Measured: zero internal stops across 132 genes rebuilt from RNA-seq-corrected
  coordinates.
* **Soft** -- splice signals. Plant organelle introns are group II and the
  mitochondrial ones are degenerate. Of ten RNA-seq-verified *Arabidopsis*
  cis-introns, the spliceosomal ``GT..AG`` rule matches **none** (they end AT/AC,
  not AG) and even a strict group II ``GT/GC..AY`` rule matches only four. A
  signal rule used as a hard constraint would reject most real introns, so
  signals only add score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NEG_INF = float("-inf")

STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})

#: Plant mitochondrial genes do not restrict themselves to ATG. ACG/ATA/ATT/GTG
#: are all documented initiators, and C-to-U editing creates further ones, so the
#: set is permissive -- its job is to stop the alignment drifting by whole codons,
#: not to decide which initiator is right.
START_CODONS = frozenset({"ATG", "ACG", "ATA", "ATT", "GTG", "ATC", "TTG"})

#: Group II 5' consensus is GTGYG; plant mitochondrial copies are degenerate, so
#: these only score. Keyed by (donor dinucleotide, acceptor dinucleotide).
_DONOR_BONUS = {"GT": 3.0, "GC": 2.0, "GG": 1.0}
_ACCEPTOR_BONUS = {"AT": 3.0, "AC": 3.0, "AG": 1.0}

_CODON_TABLE = {}
for _b1 in "TCAG":
    for _b2 in "TCAG":
        for _b3 in "TCAG":
            _codon = _b1 + _b2 + _b3
            _CODON_TABLE[_codon] = (
                "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
            )["TCAG".index(_b1) * 16 + "TCAG".index(_b2) * 4 + "TCAG".index(_b3)]

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_AA_INDEX = {a: n for n, a in enumerate(AMINO_ACIDS)}


def translate(codon: str) -> str:
    """Return the single-letter amino acid for ``codon``, ``X`` if ambiguous."""
    return _CODON_TABLE.get(codon.upper(), "X")


@dataclass(frozen=True)
class Profile:
    """Position-specific scores for one gene family.

    ``scores[i][a]`` is the log-odds of amino acid ``a`` at profile position
    ``i``. Built from an HMM profile or a PSSM; the DP only needs the numbers.
    """

    name: str
    scores: list[list[float]]
    gap_open: float = -8.0
    gap_extend: float = -1.0
    #: Cost of a codon present in the genome but absent from the profile.
    #: Without an insert layer the only way to absorb extra codons is to jump
    #: them with a minimum-length intron, which is what the DP was doing:
    #: the spurious introns it produced were 402, 406, 422 and 569 bp against a
    #: 400 bp floor -- the cheapest legal jump, not a biological feature.
    insert_open: float = -9.0
    insert_extend: float = -2.0
    #: Cost of opening an intron, as the log-odds of one being there at all.
    #: Introns are rare per position -- the ten RNA-seq-verified cis-introns of
    #: Arabidopsis fall across roughly 8,300 codon positions, a prior of ~0.0012,
    #: so 2*log2(0.0012) ~= -19.4 in these units. Derived from that density, not
    #: fitted to any gene: at -12 the implied prior is 0.016, thirteen times too
    #: generous, which is what let the DP buy small score gains with fake introns.
    intron_open: float = -19.4
    intron_extend: float = -0.02

    def __len__(self) -> int:
        return len(self.scores)

    def match(self, i: int, aa: str) -> float:
        """Score of aligning amino acid ``aa`` to profile position ``i`` (1-based)."""
        n = _AA_INDEX.get(aa)
        return -4.0 if n is None else self.scores[i - 1][n]


@dataclass
class Exon:
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start + 1


@dataclass
class Solution:
    """A gene structure and the evidence for it."""

    score: float
    exons: list[Exon] = field(default_factory=list)
    tier: int = 1
    notes: list[str] = field(default_factory=list)

    @property
    def cds_length(self) -> int:
        return sum(len(e) for e in self.exons)


class _Cell:
    """Score plus the back-pointer needed to rebuild the structure."""

    __slots__ = ("score", "prev")

    def __init__(self, score: float = NEG_INF, prev: object = None) -> None:
        self.score = score
        self.prev = prev


def _splice_bonus(seq: str, donor_at: int, acceptor_end: int) -> float:
    """Score, never reject, the dinucleotides bounding a candidate intron.

    ``donor_at`` is the 0-based index of the intron's first base; ``acceptor_end``
    is the 0-based index one past its last base.
    """
    donor = seq[donor_at : donor_at + 2]
    acceptor = seq[acceptor_end - 2 : acceptor_end]
    return _DONOR_BONUS.get(donor, 0.0) + _ACCEPTOR_BONUS.get(acceptor, 0.0)


def solve(
    profile: Profile,
    seq: str,
    *,
    min_intron: int = 400,
    max_intron: int = 6000,
    allow_introns: bool = True,
    require_start: bool = True,
    require_stop: bool = True,
) -> Solution | None:
    """Find the best-scoring gene structure for ``profile`` in ``seq``.

    ``seq`` is the coding strand: callers reverse-complement before calling
    rather than the DP carrying a strand flag.

    Returns ``None`` when no structure scores above zero.
    """
    seq = seq.upper()
    L = len(seq)
    M = len(profile)
    if L < 3 or M == 0:
        return None

    # match/insert/delete/intron layers, indexed [i][j]; i in 0..M, j in 0..L
    match = [[_Cell() for _ in range(L + 1)] for _ in range(M + 1)]
    delete = [[_Cell() for _ in range(L + 1)] for _ in range(M + 1)]
    insert = [[_Cell() for _ in range(L + 1)] for _ in range(M + 1)]
    # intron[p][i][j]: intron open through j, p bases of the next codon already
    # read; prev carries the intron's first base so its length can be checked on
    # close without another dimension.
    intron = [[[_Cell() for _ in range(L + 1)] for _ in range(M + 1)] for _ in range(3)]

    # Entry: the gene may begin anywhere, but it must begin *at an initiator*.
    # Without this the alignment is free to drop or add whole codons wherever the
    # profile is indifferent at its ends -- measured on eight single-exon
    # Arabidopsis genes, every terminal error was exactly +-3 or +-6 bp, i.e. a
    # whole codon, never a partial one.
    for j in range(L + 1):
        if not require_start or seq[j : j + 3] in START_CODONS:
            match[0][j] = _Cell(0.0, None)

    # Column 0 of the delete layer: the j loop below starts at 1, so without this
    # the first profile position could never be deleted and a gene whose N-terminus
    # is shorter than the profile was unreachable rather than merely penalised.
    for i in range(1, M + 1):
        src = match[0][0] if i == 1 else delete[i - 1][0]
        pen = profile.gap_open if i == 1 else profile.gap_extend
        if src.score > NEG_INF:
            delete[i][0] = _Cell(src.score + pen,
                                 ("M", 0, 0) if i == 1 else ("D", i - 1, 0))

    def codon_at(end: int) -> str:
        """Codon whose last base is 1-based position ``end``."""
        return seq[end - 3 : end]

    best = _Cell()
    best_pos = (0, 0)
    best_layer = "M"

    for i in range(1, M + 1):
        for j in range(1, L + 1):
            # ---- match: a whole codon ending at j, contiguous in the genome
            if j >= 3:
                cod = codon_at(j)
                if cod not in STOP_CODONS:                 # hard constraint
                    sc = profile.match(i, translate(cod))
                    src = match[i - 1][j - 3]
                    cand, prev = src.score, ("contig", "M", i - 1, j - 3, j)
                    if delete[i - 1][j - 3].score > cand:
                        cand = delete[i - 1][j - 3].score
                        prev = ("contig", "D", i - 1, j - 3, j)
                    if insert[i - 1][j - 3].score > cand:
                        cand = insert[i - 1][j - 3].score
                        prev = ("contig", "I", i - 1, j - 3, j)
                    if cand > NEG_INF:
                        match[i][j] = _Cell(cand + sc, prev)

            # ---- match: codon interrupted by an intron
            # p == 0 is the intron sitting between two codons; it was written and
            # extended but never closed, so every gene carrying one was simply
            # unreachable. Measured across the 29 Table S2 species, phase 0 is 8%
            # of cis-introns (11/141) -- uncommon, but each occurrence loses the
            # whole gene, and they cluster in nad5, nad2 and nad1.
            if allow_introns:
                for p in (0, 1, 2):
                    tail = 3 - p
                    close_at = j - tail
                    if close_at < 1:
                        continue
                    cell = intron[p][i - 1][close_at]
                    if cell.score == NEG_INF:
                        continue
                    start = cell.prev[1] if cell.prev else None
                    if start is None or not (min_intron <= close_at - start + 1 <= max_intron):
                        continue
                    head = seq[start - p - 1 : start - 1]   # bases read before the intron
                    cod = head + seq[close_at : close_at + tail]
                    if len(cod) != 3 or cod in STOP_CODONS:
                        continue
                    sc = profile.match(i, translate(cod))
                    sc += _splice_bonus(seq, start - 1, close_at)
                    if cell.score + sc > match[i][j].score:
                        match[i][j] = _Cell(
                            cell.score + sc,
                            ("split", i - 1, close_at, start, p, tail),
                        )

            # ---- insert: consume a genome codon without a profile position
            if j >= 3 and codon_at(j) not in STOP_CODONS:
                src = match[i][j - 3]
                if src.score + profile.insert_open > insert[i][j].score:
                    insert[i][j] = _Cell(src.score + profile.insert_open,
                                         ("contig", "M", i, j - 3, j))
                if insert[i][j - 3].score + profile.insert_extend > insert[i][j].score:
                    insert[i][j] = _Cell(insert[i][j - 3].score + profile.insert_extend,
                                         ("contig", "I", i, j - 3, j))

            # ---- delete: consume a profile position without genome
            src = match[i - 1][j]
            if src.score + profile.gap_open > delete[i][j].score:
                delete[i][j] = _Cell(src.score + profile.gap_open, ("M", i - 1, j))
            if delete[i - 1][j].score + profile.gap_extend > delete[i][j].score:
                delete[i][j] = _Cell(delete[i - 1][j].score + profile.gap_extend,
                                     ("D", i - 1, j))

            # ---- intron: open after p bases of the next codon, or extend
            if allow_introns:
                for p in (0, 1, 2):
                    # an intron may open after a match or after a deletion; only
                    # the former was wired, so a deletion immediately followed by
                    # an intron had no path
                    opened = NEG_INF
                    if j - p >= 0:
                        opened = match[i][j - p].score
                        if p == 0 and delete[i][j].score > opened:
                            opened = delete[i][j].score
                    cur = intron[p][i][j]
                    if opened > NEG_INF and opened + profile.intron_open > cur.score:
                        intron[p][i][j] = _Cell(opened + profile.intron_open,
                                                ("open", j + 1))
                    prev_cell = intron[p][i][j - 1]
                    if prev_cell.score > NEG_INF:
                        ext = prev_cell.score + profile.intron_extend
                        if ext > intron[p][i][j].score:
                            intron[p][i][j] = _Cell(ext, prev_cell.prev)

            # Exit: the last matched codon must be followed by a terminator.
            # Exiting from the delete layer as well is what lets a gene be
            # genuinely shorter than the profile at its C-terminus. Without it the
            # DP had to spend those deletions *before* the last matched codon,
            # which does not fail loudly -- it silently aligns the final codon to
            # the wrong profile position and still scores positive.
            if i == M and (not require_stop or seq[j : j + 3] in STOP_CODONS):
                if match[i][j].score > best.score:
                    best, best_pos, best_layer = match[i][j], (i, j), "M"
                if delete[i][j].score > best.score:
                    best, best_pos, best_layer = delete[i][j], (i, j), "D"
                if insert[i][j].score > best.score:
                    best, best_pos, best_layer = insert[i][j], (i, j), "I"

    if best.score <= 0:
        return None
    sol = _traceback(match, delete, insert, intron, best_pos, best.score, profile, best_layer)
    if require_stop and sol.exons:
        # GenBank CDS coordinates include the terminator, and so does the truth
        # this is validated against -- 27 of 32 Arabidopsis genes in PMGA Table S2
        # end on TAA/TAG/TGA. The recursion cannot align a residue to a stop, so
        # the terminator is appended here rather than being matched.
        sol.exons[-1] = Exon(sol.exons[-1].start, sol.exons[-1].end + 3)
    return sol


def _traceback(match, delete, insert, intron, pos, score, profile, layer="M") -> Solution:
    """Rebuild exon coordinates by replaying the back-pointers.

    Each pointer records the genome interval its codon consumed, so this only
    has to collect intervals and merge the adjacent ones; a codon split by an
    intron contributes two intervals, which is what keeps a phase-1 or phase-2
    boundary from being rounded to a codon edge.
    """
    i, j = pos
    spans: list[tuple[int, int]] = []

    while i > 0 or layer == "I":
        cell = {"M": match, "D": delete, "I": insert}[layer][i][j]
        prev = cell.prev
        if prev is None:
            break
        kind = prev[0]
        if kind == "contig":
            _, src_layer, pi, pj, end = prev
            spans.append((end - 2, end))
            layer, i, j = src_layer, pi, pj
        elif kind == "split":
            _, pi, close_at, start, p, tail = prev
            if p:
                spans.append((start - p, start - 1))
            spans.append((close_at + 1, close_at + tail))
            layer, i, j = "M", pi, start - p - 1 if p else start - 1
        elif kind in ("M", "D"):
            _, pi, pj = prev
            layer, i, j = ("M" if kind == "M" else "D"), pi, pj
        else:
            break

    spans.sort()
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    exons = [Exon(a, b) for a, b in merged if b >= a and a >= 1]
    return Solution(score=score, exons=exons, tier=1)


def scan(
    profile: Profile,
    seq: str,
    *,
    min_intron: int = 400,
    max_intron: int = 6000,
    top_n: int = 20,
    min_separation: int | None = None,
) -> list[tuple[float, int]]:
    """Score every position of ``seq`` without a window, returning the best ends.

    This exists to test the claim the whole design rests on: that the true locus
    wins an exhaustive genome-wide search, with no anchor or seed telling the
    solver where to look. Every result measured so far was obtained inside a
    +-1500 bp window around the known gene, which is precisely the prior the
    design forbids -- so those numbers cannot support the claim.

    Only scores are computed, not structures: the question is whether the true
    locus is the top-scoring one, and full traceback over a whole genome needs
    the linear-memory formulation the Rust kernel will carry. All transitions
    reach at most three columns back and the intron start travels inside its own
    cell, so four rolling columns suffice here.

    ``min_separation`` is how far apart two reported hits must be. It defaults to
    three profile lengths, which is enough to stop one locus being listed twice
    but *not* enough to make the runners-up independent: measured on four genes,
    the second-best hit sat 423-1265 bp from the best and scored within 1.4-2.3%
    of it, i.e. it was the same locus shifted. To measure discrimination against
    the rest of the genome rather than against a shifted copy of itself, pass a
    separation well beyond the gene span.

    Returns ``[(score, end_position), ...]`` for the best non-overlapping ends,
    highest first.
    """
    seq = seq.upper()
    L, M = len(seq), len(profile)
    if L < 3 or M == 0:
        return []

    width = 4
    mat = [[NEG_INF] * (M + 1) for _ in range(width)]
    dele = [[NEG_INF] * (M + 1) for _ in range(width)]
    ins = [[NEG_INF] * (M + 1) for _ in range(width)]
    intr = [[[NEG_INF] * (M + 1) for _ in range(width)] for _ in range(3)]
    intr_start = [[[0] * (M + 1) for _ in range(width)] for _ in range(3)]

    hits: list[tuple[float, int]] = []

    for j in range(L + 1):
        c = j % width
        mat[c] = [NEG_INF] * (M + 1)
        dele[c] = [NEG_INF] * (M + 1)
        ins[c] = [NEG_INF] * (M + 1)
        for p in range(3):
            intr[p][c] = [NEG_INF] * (M + 1)

        mat[c][0] = 0.0 if seq[j : j + 3] in START_CODONS else NEG_INF

        c3 = (j - 3) % width
        cod = seq[j - 3 : j] if j >= 3 else ""
        legal = len(cod) == 3 and cod not in STOP_CODONS
        aa = translate(cod) if legal else "X"

        for i in range(1, M + 1):
            if legal:
                sc = profile.match(i, aa)
                best_src = max(mat[c3][i - 1], dele[c3][i - 1], ins[c3][i - 1])
                if best_src > NEG_INF:
                    mat[c][i] = best_src + sc
                for p in (0, 1, 2):
                    tail = 3 - p
                    cj = (j - tail) % width
                    if j - tail < 1:
                        continue
                    s = intr[p][cj][i - 1]
                    if s == NEG_INF:
                        continue
                    st = intr_start[p][cj][i - 1]
                    if not (min_intron <= (j - tail) - st + 1 <= max_intron):
                        continue
                    head = seq[st - p - 1 : st - 1]
                    split = head + seq[j - tail : j]
                    if len(split) != 3 or split in STOP_CODONS:
                        continue
                    v = s + profile.match(i, translate(split)) + _splice_bonus(
                        seq, st - 1, j - tail)
                    if v > mat[c][i]:
                        mat[c][i] = v
                if mat[c3][i] > NEG_INF:
                    ins[c][i] = max(ins[c][i], mat[c3][i] + profile.insert_open)
                if ins[c3][i] > NEG_INF:
                    ins[c][i] = max(ins[c][i], ins[c3][i] + profile.insert_extend)

            dele[c][i] = max(mat[c][i - 1] + profile.gap_open,
                             dele[c][i - 1] + profile.gap_extend)

            for p in (0, 1, 2):
                base = mat[(j - p) % width][i] if j - p >= 0 else NEG_INF
                if p == 0:
                    base = max(base, dele[c][i])
                if base > NEG_INF and base + profile.intron_open > intr[p][c][i]:
                    intr[p][c][i] = base + profile.intron_open
                    intr_start[p][c][i] = j + 1
                prev = intr[p][(j - 1) % width][i]
                if prev > NEG_INF and prev + profile.intron_extend > intr[p][c][i]:
                    intr[p][c][i] = prev + profile.intron_extend
                    intr_start[p][c][i] = intr_start[p][(j - 1) % width][i]

        if seq[j : j + 3] in STOP_CODONS:
            end_score = max(mat[c][M], dele[c][M], ins[c][M])
            if end_score > 0:
                hits.append((end_score, j))

    hits.sort(reverse=True)
    sep = min_separation if min_separation is not None else 3 * M
    kept: list[tuple[float, int]] = []
    for s, e in hits:
        if all(abs(e - k[1]) > sep for k in kept):
            kept.append((s, e))
        if len(kept) >= top_n:
            break
    return kept
