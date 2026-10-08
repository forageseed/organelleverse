"""Terminal ORF refinement shared by splice candidate scoring and CDS refinement."""


_STOP = {"TAA", "TAG", "TGA"}
_START = ("ATG", "ACG", "GTG")


def _find_start(seq: str, anchor: int, window: int) -> int:
    """5'-most/closest in-frame start codon near ``anchor`` (1-based)."""
    L = len(seq)
    best, best_rank = None, None
    for off in range(-window, window + 1):
        s = anchor + off
        if s < 1 or s + 2 > L:
            continue
        codon = seq[s - 1 : s + 2]
        if codon in _START:
            rank = (abs(off), _START.index(codon))
            if best_rank is None or rank < best_rank:
                best_rank, best = rank, s
    return best if best is not None else anchor


def _refine_multi_forward(
    seq: str, exons: list[tuple[int, int]], window: int, max_ext: int
) -> list[tuple[int, int]]:
    """Refine terminal boundaries of a forward-strand multi-exon CDS.

    ``exons`` are 1-based (start, end) in 5'->3' reading order (ascending). The
    5' exon start is snapped to a start codon and the 3' exon end is extended to
    the first in-frame stop (reading across the actual splice, so the frame is
    correct). Internal splice boundaries are left unchanged.
    """
    L = len(seq)
    # Refine the 5' start on the first exon.
    ns0 = _find_start(seq, exons[0][0], window)
    exons = [(ns0, exons[0][1]), *exons[1:]]

    # Genome positions in spliced reading order, plus a 3' extension region.
    positions: list[int] = []
    for s, e in exons:
        positions.extend(range(s, e + 1))
    last_s, last_e = exons[-1]
    positions.extend(range(last_e + 1, min(L, last_e + max_ext) + 1))

    # Read codons in-frame from the start; stop at the first stop codon.
    stop_last = None
    i = 0
    while i + 3 <= len(positions):
        codon = seq[positions[i] - 1] + seq[positions[i + 1] - 1] + seq[positions[i + 2] - 1]
        if codon in _STOP:
            stop_last = positions[i + 2]
            break
        i += 3
    if stop_last is not None and stop_last >= last_s:
        exons[-1] = (last_s, stop_last)
    return exons


