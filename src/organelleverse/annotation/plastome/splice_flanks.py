"""Reference junction flanks: choose among splice placements that protein scores cannot separate.

Protein-guided splice refinement scores each candidate's spliced translation
against reference proteins. Those proteins are translations of the *edited*
mRNA, so two placements whose spliced sequences differ only at a C/T position
(an editing site at the junction) score alike, or the one carrying the genomic
T scores higher. Placing the intron one base off this way is common at
ndhA/rps12/clpP joins: ``...CTAC|GTGCG...AT|TATC`` versus
``...CTA|CGTGCG...ATT|ATC``.

Protein scores against another species' protein also cannot place a join to
the codon: residues next to plastid introns are poorly conserved, and the true
placement often scores a few BLOSUM points below a neighbour (rpl2 +3 nt,
rpl16 -3 nt, rpoC1 in the development set).

The reference annotations decide instead. Each reference CDS of the gene votes
for the candidate(s) whose genomic join flanks match its own exons most
closely (reference exons are genomic, unedited sequence). Among edit-equivalent
candidates (identical spliced sequence once C and T are not distinguished) the
most-voted wins; among other candidates within PROTEIN_TOLERANCE of the best
protein score, one replaces the protein-best only with a clear majority. Votes
are needed rather than the single closest reference because GenBank annotates
some joins both ways (ndhA intron 1: 19 versus 16 shipped plastomes; a few
records extend rpl2 exon 1 into the intron's GTG). Remaining ties go to the
placement with group II intron boundaries, then to the protein score.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from itertools import pairwise

FLANK = 10

# Per gene: one entry per reference CDS, each a list of (left, right) join flanks
# in transcription order (left = last FLANK nt of exon i, right = first FLANK of exon i+1).
JunctionFlanks = dict[str, list[list[tuple[str, str]]]]


def reference_junction_flanks(
    references: Iterable, flank: int = FLANK, skip: set[tuple[str, str]] | None = None
) -> JunctionFlanks:
    """Join flanks of every multi-exon reference CDS, from its per-exon features.

    Uses the ``CDS_exon`` features (sequence in transcription sense, exon_index
    1 = transcription-first), so it works for references parsed from GenBank
    and for the preprocessed cache alike. ``skip`` holds (reference file, gene)
    pairs left out (junctions the reference audit found misannotated).
    """
    exons: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    counts: dict[tuple[str, str], int] = {}
    genes: dict[tuple[str, str], str] = {}
    for reference in references:
        for feat in reference.features:
            if feat.feature_type != "CDS_exon" or not feat.gene:
                continue
            if skip and (reference.path.name, feat.gene) in skip:
                continue
            key = (feat.reference_name, feat.feature_id.rsplit(":exon", 1)[0])
            exons[key][feat.exon_index] = feat.sequence.upper()
            counts[key] = feat.exon_count
            genes[key] = feat.gene
    out: JunctionFlanks = defaultdict(list)
    for key, by_index in exons.items():
        n = counts[key]
        if sorted(by_index) != list(range(1, n + 1)):
            continue
        seqs = [by_index[i] for i in range(1, n + 1)]
        out[genes[key]].append([(seqs[i][-flank:], seqs[i + 1][:flank]) for i in range(n - 1)])
    return dict(out)


# Group II intron boundaries: 5' GUGYG, 3' AY (Michel & Ferat 1995).
_GROUP_II_5 = re.compile(r"GTG[CT]G")

# Candidates within this share of the best protein score are compared by
# reference votes; a non-equivalent one must collect more than VOTE_RATIO times
# the protein-best's votes and at least VOTE_MARGIN more.
PROTEIN_TOLERANCE = 0.06
VOTE_RATIO = 2.0
VOTE_MARGIN = 2.0


def _boundary_score(seq: str, end: int, start: int) -> int:
    """Group II boundary score (0-2) of the intron seq[end:start-1], best over equivalent slides."""
    best = 0
    lo, hi = end, start - 1  # 0-based intron [lo, hi)
    # Slide left/right while the spliced product is unchanged.
    placements = [(lo, hi)]
    a, b = lo, hi
    while a > 0 and b - 1 > a and seq[a - 1] == seq[b - 1]:
        a, b = a - 1, b - 1
        placements.append((a, b))
    a, b = lo, hi
    while b < len(seq) and b - 1 > a and seq[a] == seq[b]:
        a, b = a + 1, b + 1
        placements.append((a, b))
    for a, b in placements:
        intron = seq[a:b]
        if len(intron) < 7:
            continue
        score = int(bool(_GROUP_II_5.match(intron))) + int(intron[-2] == "A" and intron[-1] in "CT")
        best = max(best, score)
    return best


def group_ii_joins(seq: str, exons: list[tuple[int, int]]) -> int:
    """Summed group II boundary score of all joins (forward-oriented ``seq``, ascending 1-based exons)."""
    return sum(_boundary_score(seq, end, start) for (_, end), (start, _) in pairwise(exons))


def collapse_edits(seq: str) -> str:
    """Spliced sequence with C and T not distinguished (C-to-U and U-to-C editing)."""
    return seq.upper().replace("T", "C")


def _side_mismatches(a: str, b: str, *, right_aligned: bool) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    if right_aligned:
        a, b = a[-n:], b[-n:]
    else:
        a, b = a[:n], b[:n]
    return sum(x != y for x, y in zip(a, b, strict=True))


def _distance_to(product: str, join_offsets: Sequence[int], ref: list[tuple[str, str]], flank: int = FLANK) -> int:
    d = 0
    for offset, (left, right) in zip(join_offsets, ref, strict=True):
        d += _side_mismatches(product[max(0, offset - flank):offset], left, right_aligned=True)
        d += _side_mismatches(product[offset:offset + flank], right, right_aligned=False)
    return d


def flank_distance(product: str, join_offsets: Sequence[int], references: list[list[tuple[str, str]]] | None,
                   flank: int = FLANK) -> int | None:
    """Mismatches between the candidate's join flanks and the closest reference.

    ``join_offsets[i]`` is the number of spliced bases before join ``i``. Only
    references with the same number of joins are compared; None when none is.
    """
    usable = [ref for ref in references or [] if len(ref) == len(join_offsets)]
    if not usable:
        return None
    return min(_distance_to(product, join_offsets, ref, flank) for ref in usable)


def reference_votes(candidates, indices, references) -> dict[int, float]:
    """Each usable reference gives one vote, shared among the candidates closest to it."""
    votes = dict.fromkeys(indices, 0.0)
    if not indices:
        return votes
    n_joins = len(candidates[indices[0]][3])
    for ref in references or []:
        if len(ref) != n_joins:
            continue
        dist = {i: _distance_to(candidates[i][2], candidates[i][3], ref) for i in indices}
        best = min(dist.values())
        winners = [i for i in indices if dist[i] == best]
        for i in winners:
            votes[i] += 1 / len(winners)
    return votes


def choose_candidate(candidates: list[tuple], references: list[list[tuple[str, str]]] | None,
                     tolerance: float | None = None):
    """Pick a splice placement: protein score, overruled by reference votes where proteins cannot decide.

    ``candidates``: (protein score, exons, spliced sequence, join offsets[, group II boundary score
    [, joins agreeing with the domain V-VI 3' splice site prediction]]),
    in generation order (the first is the original model). Returns the chosen
    exons, or None for no candidates. Ties keep the earlier candidate.
    """
    if not candidates:
        return None
    tolerance = PROTEIN_TOLERANCE if tolerance is None else tolerance
    top = max(range(len(candidates)), key=lambda i: (candidates[i][0], -i))
    n_joins = len(candidates[top][3])
    if not references or not any(len(ref) == n_joins for ref in references):
        return candidates[top][1]
    target = collapse_edits(candidates[top][2])
    group = [i for i, c in enumerate(candidates) if len(c[3]) == n_joins and collapse_edits(c[2]) == target]
    floor = candidates[top][0] - tolerance * abs(candidates[top][0])
    close = [i for i, c in enumerate(candidates)
             if tolerance > 0 and c[0] >= floor and i not in group and len(c[3]) == n_joins]
    if len(group) == 1 and not close:
        return candidates[top][1]
    votes = reference_votes(candidates, group + close, references)

    def rank(i):
        strong, weak = _structure(candidates[i])
        return (strong, votes[i], weak, _motif(candidates[i]), candidates[i][0], -i)

    chosen = max(group, key=rank)
    # A clear reference majority replaces the protein-best. So does agreement with
    # the intron's own domains V-VI: always where the references agree on that
    # intron's calibration (reliable), otherwise only where votes do not oppose it.
    challengers = [
        i for i in close
        if (votes[i] > VOTE_RATIO * votes[chosen] and votes[i] >= votes[chosen] + VOTE_MARGIN)
        or _structure(candidates[i])[0] > _structure(candidates[chosen])[0]
        or (_structure(candidates[i])[1] > _structure(candidates[chosen])[1] and votes[i] >= votes[chosen])
    ]
    if challengers:
        chosen = max(challengers, key=rank)
    return candidates[chosen][1]


def _motif(candidate) -> int:
    return candidate[4] if len(candidate) > 4 else 0


def _structure(candidate) -> tuple[int, int]:
    """(reliable, other) joins agreeing with the domain V-VI prediction."""
    value = candidate[5] if len(candidate) > 5 else (0, 0)
    return value if isinstance(value, tuple) else (0, value)
