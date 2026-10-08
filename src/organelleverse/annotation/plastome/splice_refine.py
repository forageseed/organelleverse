"""Splice-site-aware refinement of two-exon CDS internal boundaries.

BLAST places exon boundaries approximately, so the intron rarely starts/ends at
the canonical GT..AG splice sites and the reading frame across the splice is
often wrong (which also throws off the 3' stop). For a two-exon CDS this searches
donor (GT/GC) and acceptor (AG or group II AY) positions near the approximate intron boundary
and keeps the combination whose spliced translation best matches the reference
protein — recovering the exact internal boundary and thus the correct frame and
terminal stop.
"""

from __future__ import annotations

import contextlib
from itertools import product

from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Seq import Seq

from .cds_terminals import _refine_multi_forward
from .group_ii import predicted_ends, structure_agreement
from .splice_flanks import choose_candidate, group_ii_joins

_STOP = {"TAA", "TAG", "TGA"}
_START = ("ATG", "ACG", "GTG")
_DONOR = ("GT", "GC")
_ACCEPTOR = ("AG", "AC", "AT")

_AL = PairwiseAligner()
_AL.mode = "local"
_AL.open_gap_score = -11
_AL.extend_gap_score = -1
with contextlib.suppress(Exception):  # pragma: no cover
    _AL.substitution_matrix = substitution_matrices.load("BLOSUM62")

_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _san(p: str) -> str:
    return "".join(c if c in _AA else "X" for c in p)


def _score(a: str, b: str) -> float:
    try:
        return _AL.score(a, b)
    except Exception:
        return float("-inf")


def _refine_forward_two_exon(
    seq: str, e1: tuple[int, int], e2: tuple[int, int], ref_protein: str,
    window: int, terminal_ext: int | None = None, junction_flanks=None, structure_offsets=None,
):
    """Score splice candidates in their own reading frame and terminal ORF."""
    ref = _san(ref_protein.rstrip("*"))
    (s1, e1_end), (e2_start, e2_end) = e1, e2

    def evaluate(exons):
        # Transfer leaves both splice and terminal coordinates approximate.
        # A candidate's terminal stop must be found in its own spliced frame
        # before testing phase or protein, rather than rejecting a correct
        # splice because the transferred terminal end has an extra base.
        if terminal_ext is not None:
            # An HSP may cover only a fragment of the last exon. Include
            # the reference's missing coding span as well as the existing
            # terminal slack, rather than bounding every gene by that slack.
            transferred_nt = sum(e - s + 1 for s, e in exons)
            missing_reference_nt = max(0, 3 * len(ref) - transferred_nt)
            exons = _refine_multi_forward(
                seq, list(exons), window, terminal_ext + missing_reference_nt
            )
        coding = "".join(seq[s - 1:e] for s, e in exons)
        if len(coding) < 60 or len(coding) % 3:
            return None
        protein = str(Seq(coding).translate(table=11))
        if "*" in protein[:-1]:
            return None
        return (_score(ref, _san(protein)), list(exons), coding, [exons[0][1] - exons[0][0] + 1],
                group_ii_joins(seq, exons), structure_agreement(seq, list(exons), predicted))

    # Keep a valid original model when a motif candidate scores no better.
    # Domains V-VI predict the 3' splice site independently of any reference.
    predicted = predicted_ends(seq, [e1, e2], structure_offsets)
    # Candidates in generation order; the original model first, so it is kept on ties.
    candidates = []
    original = evaluate([e1, e2])
    if original is not None:
        candidates.append(original)
    for donor in range(e1_end - window, e1_end + window + 1):
        if donor < s1 or seq[donor:donor + 2] not in _DONOR:
            continue
        for acceptor in range(e2_start - window, e2_start + window + 1):
            if acceptor >= e2_end or acceptor < 2:
                continue
            if seq[acceptor - 2:acceptor] not in _ACCEPTOR:
                continue
            candidate = evaluate([(s1, donor), (acceptor + 1, e2_end)])
            if candidate is not None:
                candidates.append(candidate)
    # A reliable domain V-VI 3' splice site outside the window is a candidate
    # too: the transferred acceptor is only as good as the primary reference,
    # and Zea/Hordeum/Sorghum put petB exon 2 42-57 nt into the intron.
    if predicted and predicted[0] is not None and predicted[0][1]:
        acceptor = predicted[0][0]
        if abs(acceptor - e2_start) > window and 2 <= acceptor < e2_end:
            for donor in range(e1_end - window, e1_end + window + 1):
                if donor < s1 or seq[donor:donor + 2] not in _DONOR:
                    continue
                candidate = evaluate([(s1, donor), (acceptor + 1, e2_end)])
                if candidate is not None:
                    candidates.append(candidate)
    chosen = choose_candidate(candidates, junction_flanks)
    return tuple(chosen) if chosen is not None else None


def refine_two_exon_cds(
    genome: str, exons, ref_protein: str, strand: int, *, window: int = 15, terminal_ext: int | None = None,
    junction_flanks=None, structure_offsets=None,
):
    """Return refined 2-exon coords (5'->3' reading order) or None.

    ``exons`` is a list of two (start, end) 1-based tuples in reading order.
    With ``terminal_ext``, score each candidate after terminal ORF refinement;
    otherwise only internal boundaries are refined.
    """
    if len(exons) != 2 or not ref_protein:
        return None
    L = len(genome)
    if strand == -1:
        rc = str(Seq(genome).reverse_complement())
        # reading-order exons -> forward coords in rc space, ascending
        r1 = (L - exons[0][1] + 1, L - exons[0][0] + 1)
        r2 = (L - exons[1][1] + 1, L - exons[1][0] + 1)
        res = _refine_forward_two_exon(
            rc, r1, r2, ref_protein, window, terminal_ext, junction_flanks, structure_offsets
        )
        if res is None:
            return None
        (a1, b1), (a2, b2) = res
        # back to genome coords, keep reading order
        return [(L - b1 + 1, L - a1 + 1), (L - b2 + 1, L - a2 + 1)]
    res = _refine_forward_two_exon(
        genome, exons[0], exons[1], ref_protein, window, terminal_ext, junction_flanks, structure_offsets
    )
    if res is None:
        return None
    return [res[0], res[1]]


def _adaptive_window(window: int, n_boundaries: int, cap: int = 40000) -> int:
    """Shrink the per-boundary window so the joint search stays under ``cap`` combos."""
    while window > 1 and (2 * window + 1) ** (2 * n_boundaries) > cap:
        window -= 1
    return window


def _refine_internal_forward(
    seq: str, exons: list[tuple[int, int]], ref_protein: str, window: int, junction_flanks=None,
    structure_offsets=None,
):
    """Protein-guided internal-boundary search on a forward-oriented sequence.

    ``exons`` are 1-based (start, end), ascending. The 5' start (exons[0][0]) and
    3' end (exons[-1][1]) are held fixed; every internal donor (exon end) and
    acceptor (exon start) is searched within ``+-window``. A combination is kept
    only if the spliced translation is a clean in-frame ORF (length a multiple of
    3, no internal stop); among those the best BLOSUM score vs the reference wins.
    Splice-agnostic, so it also recovers group II introns (3' end AY, not AG).
    """
    n = len(exons)
    if n < 2:
        return None
    ref = _san(ref_protein.rstrip("*"))
    L = len(seq)
    w = _adaptive_window(window, n - 1)
    ends = [range(exons[j][1] - w, exons[j][1] + w + 1) for j in range(n - 1)]
    starts = [range(exons[j][0] - w, exons[j][0] + w + 1) for j in range(1, n)]
    predicted = predicted_ends(seq, list(exons), structure_offsets)
    candidates = []
    for de in product(*ends):
        for st in product(*starts):
            ex: list[tuple[int, int]] = []
            ok = True
            for j in range(n):
                s = exons[0][0] if j == 0 else st[j - 1]
                e = exons[-1][1] if j == n - 1 else de[j]
                if s < 1 or e > L or e < s:
                    ok = False
                    break
                ex.append((s, e))
            if not ok:
                continue
            if sum(b - a + 1 for a, b in ex) % 3:
                continue
            spliced = "".join(seq[a - 1 : b] for a, b in ex)
            prot = str(Seq(spliced).translate(table=11))
            if "*" in prot[:-1]:
                continue
            sc = _score(ref, _san(prot))
            offsets, acc = [], 0
            for a, b in ex[:-1]:
                acc += b - a + 1
                offsets.append(acc)
            candidates.append(
                (sc, ex, spliced, offsets, group_ii_joins(seq, ex), structure_agreement(seq, ex, predicted))
            )
    return choose_candidate(candidates, junction_flanks)


def refine_internal_boundaries(
    genome: str, exons, ref_protein: str, strand: int, *, window: int = 6, junction_flanks=None,
    structure_offsets=None,
):
    """Refine internal exon boundaries of a multi-exon CDS (reading order in, out).

    Unlike :func:`refine_two_exon_cds` (which uses GT/GC..AG/AY boundaries), this
    is protein-guided and splice-agnostic, so it handles the group II introns of
    genes like clpP/ycf3 whose 3' boundary is AY. Returns refined reading-order
    coords or None.
    """
    if len(exons) < 2 or not ref_protein:
        return None
    L = len(genome)
    if strand == -1:
        rc = str(Seq(genome).reverse_complement())
        fwd = [(L - e + 1, L - s + 1) for s, e in exons]  # reading order -> ascending rc
        res = _refine_internal_forward(rc, fwd, ref_protein, window, junction_flanks, structure_offsets)
        if res is None:
            return None
        return [(L - b + 1, L - a + 1) for a, b in res]  # back to genome, reading order
    res = _refine_internal_forward(genome, list(exons), ref_protein, window, junction_flanks, structure_offsets)
    return list(res) if res is not None else None


def refine_cis_block(
    genome: str,
    block: list[tuple[int, int]],
    ref_protein: str,
    strand: int,
    *,
    phase_skip: int,
    final: bool,
    window: int = 6,
    junction_flanks=None,
    structure_offsets=None,
):
    """Refine the internal joins of a cis-spliced block inside a trans-spliced CDS.

    ``block`` holds two or more consecutive exons on one strand (reading order,
    1-based genome coordinates), e.g. rps12 exons 2-3; the trans-spliced exons
    outside it are untouched. ``phase_skip`` bases at the block's start finish
    a codon begun upstream. The block's outer ends stay fixed; its length may
    change only in steps of three (the downstream frame is kept), and a final
    block must end in frame. ``junction_flanks`` holds the reference flanks of
    this block's joins only.
    """
    if len(block) < 2 or not ref_protein:
        return None
    L = len(genome)
    if strand == -1:
        rc = str(Seq(genome).reverse_complement())
        fwd = [(L - e + 1, L - s + 1) for s, e in block]
        res = _refine_block_forward(
            rc, fwd, ref_protein, window, phase_skip, final, junction_flanks, structure_offsets
        )
        return None if res is None else [(L - b + 1, L - a + 1) for a, b in res]
    res = _refine_block_forward(
        genome, list(block), ref_protein, window, phase_skip, final, junction_flanks, structure_offsets
    )
    return None if res is None else list(res)


def _refine_block_forward(seq, exons, ref_protein, window, phase_skip, final, junction_flanks, structure_offsets=None):
    n = len(exons)
    ref = _san(ref_protein.rstrip("*"))
    L = len(seq)
    original_len = sum(b - a + 1 for a, b in exons)
    w = _adaptive_window(window, n - 1)
    ends = [range(exons[j][1] - w, exons[j][1] + w + 1) for j in range(n - 1)]
    starts = [range(exons[j][0] - w, exons[j][0] + w + 1) for j in range(1, n)]
    predicted = predicted_ends(seq, list(exons), structure_offsets)
    candidates = []
    for de in product(*ends):
        for st in product(*starts):
            ex = []
            for j in range(n):
                s = exons[0][0] if j == 0 else st[j - 1]
                e = exons[-1][1] if j == n - 1 else de[j]
                if s < 1 or e > L or e < s:
                    break
                ex.append((s, e))
            else:
                total = sum(b - a + 1 for a, b in ex)
                if (total - original_len) % 3:
                    continue
                spliced = "".join(seq[a - 1 : b] for a, b in ex)
                coding = spliced[phase_skip:]
                if final and len(coding) % 3:
                    continue
                coding = coding[: len(coding) - len(coding) % 3]
                prot = str(Seq(coding).translate(table=11))
                if "*" in (prot[:-1] if final else prot):
                    continue
                offsets, acc = [], 0
                for a, b in ex[:-1]:
                    acc += b - a + 1
                    offsets.append(acc)
                candidates.append((_score(ref, _san(prot)), ex, spliced, offsets, group_ii_joins(seq, ex),
                                   structure_agreement(seq, ex, predicted)))
    # The unchanged block is generated first (offset 0 in every range is visited
    # in order), so a tie keeps it.
    candidates.sort(key=lambda c: c[1] != list(exons))
    return choose_candidate(candidates, junction_flanks)
