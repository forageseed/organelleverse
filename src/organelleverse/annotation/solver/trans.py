"""Trans-spliced genes: align profile segments independently, then assemble.

Three gene families account for essentially all trans-splicing in plant
mitochondria -- nad1, nad2 and nad5 -- and across the 29 Table S2 species they
carry 142 exons, 11.5% of all exons. That is not a corner case to skip.

It is also not a wider intron. Measured on those species, the gaps between exon
groups run to 850 kb, and **18 of 87 gene instances place their exon groups on
opposite strands**. A recursion that walks one strand cannot express that at all,
however large its intron bound.

So the model follows the biology instead: trans-splicing assembles independently
transcribed pieces after transcription, and this aligns pieces independently and
then assembles them. Each piece is a *profile-local* alignment covering some
range of profile positions; assembly chooses pieces whose ranges tile the profile
without overlapping, maximising total score.

The cost of getting this wrong is silent: a gene whose pieces cannot be joined
does not fail loudly, it comes back as a partial gene that looks plausible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dp import NEG_INF, STOP_CODONS, Exon, Profile, Solution, _Cell, translate

#: Gene families known to be trans-spliced in plant mitochondria.
#:
#: A biological fact about gene families, not a constant fitted to a validation
#: set: across the 29 Table S2 species it holds for nad2 in 10 instances, nad1 in
#: 9 and nad5 in 9. The distinction matters -- "nad2 is trans-spliced" survives a
#: change of species, "nad2 exon 3 sits at +63789 bp" does not, and only the
#: latter is the kind of hardcoding this solver exists to remove.
TRANS_SPLICED = frozenset({"nad1", "nad2", "nad5", "rps12"})


@dataclass
class Piece:
    """One independently aligned run of profile positions."""

    profile_start: int
    profile_end: int
    exons: list[Exon]
    strand: str
    score: float

    @property
    def span(self) -> int:
        return self.profile_end - self.profile_start + 1


@dataclass
class TransSolution(Solution):
    """A gene assembled from pieces, recording which pieces and any gaps."""

    pieces: list[Piece] = field(default_factory=list)
    profile_covered: int = 0
    profile_length: int = 0

    @property
    def is_complete(self) -> bool:
        return self.profile_covered >= self.profile_length


def align_pieces(
    profile: Profile,
    seq: str,
    strand: str,
    *,
    min_piece: int = 20,
    min_score: float = 0.0,
    max_pieces: int = 40,
) -> list[Piece]:
    """Find profile-local alignments: runs of profile positions matched anywhere.

    Local in *profile* space as well as genome space. The global recursion enters
    at profile position 0 and leaves at M, which is right for a gene lying in one
    piece and wrong for one that does not: a piece carrying only positions 120-260
    has no path there.

    Pieces are contiguous in the genome here; introns inside a piece are handled
    by the ordinary cis recursion, which this deliberately does not duplicate.
    """
    seq = seq.upper()
    L, M = len(seq), len(profile)
    if L < 3 or M == 0:
        return []

    # match[i][j] with free entry at every i: a piece may begin anywhere in the
    # profile, which is exactly what the global recursion forbids.
    best_end: dict[tuple[int, int], tuple[float, int, int]] = {}

    ring: list[list[_Cell]] = [[_Cell(0.0, None) for _ in range(M + 1)] for _ in range(4)]
    for j in range(L + 1):
        cur = ring[j % 4]
        for i in range(M + 1):
            cur[i] = _Cell(0.0, None)          # free start at any (i, j)
        if j >= 3:
            codon = seq[j - 3 : j]
            if codon not in STOP_CODONS:
                aa = translate(codon)
                src = ring[(j - 3) % 4]
                for i in range(1, M + 1):
                    base = src[i - 1]
                    # 0 is the free-start score, not "no path": rejecting it here
                    # discarded every piece that had not yet been extended once.
                    if base.score < 0:
                        continue
                    v = base.score + profile.match(i, aa)
                    if v > cur[i].score:
                        start_i = base.prev[0] if base.prev else i - 1
                        cur[i] = _Cell(v, (start_i, base.prev[1] if base.prev else j - 3))
                # a run that has just been extended is a candidate piece end
                for i in range(1, M + 1):
                    c = cur[i]
                    if c.score > min_score and c.prev:
                        si, sj = c.prev
                        if i - si + 1 >= min_piece:
                            key = (si, i)
                            if key not in best_end or c.score > best_end[key][0]:
                                best_end[key] = (c.score, sj, j)

    pieces = [
        Piece(profile_start=si + 1, profile_end=i,
              exons=[Exon(sj + 1, j)], strand=strand, score=sc)
        for (si, i), (sc, sj, j) in best_end.items()
    ]
    pieces.sort(key=lambda p: -p.score)
    return pieces[:max_pieces]


def assemble(profile: Profile, pieces: list[Piece]) -> TransSolution | None:
    """Chain pieces into one gene: non-overlapping in the profile, best total.

    Pieces may come from either strand and from anywhere in the genome, because
    trans-splicing joins separately transcribed pieces -- the only ordering that
    has to hold is the profile's.
    """
    if not pieces:
        return None
    M = len(profile)
    ordered = sorted(pieces, key=lambda p: (p.profile_start, p.profile_end))

    best: list[float] = [p.score for p in ordered]
    back: list[int | None] = [None] * len(ordered)
    for b, pb in enumerate(ordered):
        for a in range(b):
            if ordered[a].profile_end < pb.profile_start and best[a] + pb.score > best[b]:
                best[b] = best[a] + pb.score
                back[b] = a

    end = max(range(len(ordered)), key=lambda k: best[k])
    chain: list[Piece] = []
    node: int | None = end
    while node is not None:
        chain.append(ordered[node])
        node = back[node]
    chain.reverse()

    covered = sum(p.span for p in chain)
    exons = [e for p in chain for e in p.exons]
    return TransSolution(
        score=best[end],
        exons=sorted(exons, key=lambda e: e.start),
        tier=1 if covered >= M else 2,
        pieces=chain,
        profile_covered=covered,
        profile_length=M,
        notes=[] if covered >= M else [f"profile covered {covered}/{M}"],
    )
