"""Recovery of micro-exons in plastome CDS (petB/petD/rpl16).

Genes such as petB (6 bp), petD (8 bp) and rpl16 (9 bp) have a tiny 5' first exon
carrying the start codon, then a long second exon after a ~0.5-1.5 kb intron. A
blastn query for a 6-9 bp exon is below the word size, so reference transfer finds
only the second exon and the gene collapses to a frameshifted single exon.

Given the (found) second exon and the reference protein, this searches upstream
for the first exon: a start codon, a canonical GT/GC donor, and a spliced
translation that matches the reference N-terminus — recovering the exact
micro-exon and restoring the correct reading frame.
"""

from __future__ import annotations

import contextlib

from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

_START = ("ATG", "ACG", "GTG")  # ACG/GTG are edited/alternative organellar starts
_DONOR = ("GT", "GC")
_MAX_MICRO_EXON = 15  # first exon length that counts as a micro-exon
# A 5-9 nt exon encodes two or three residues, so candidates a few BLOSUM points
# apart are not told apart by the protein. Within this margin of the best, the
# candidate whose intron has group II boundaries (5' GUGYG, 3' AY) wins: without
# it the search kept the first candidate, the shortest intron, and put rpl16
# exon 1 nine bases downstream of the real one (Pinus, Abies, Welwitschia).
_MICRO_SCORE_MARGIN = 5.0
_INTRON_MIN = 400
_INTRON_MAX = 1600

_AL = PairwiseAligner()
_AL.mode = "local"
_AL.open_gap_score = -11
_AL.extend_gap_score = -1
with contextlib.suppress(Exception):  # pragma: no cover
    _AL.substitution_matrix = substitution_matrices.load("BLOSUM62")

_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _san(p: str) -> str:
    return "".join(c if c in _AA else "X" for c in p)


def find_micro_exon(
    genome: str, exon2: tuple[int, int], strand: int, exon1_len: int, ref_proteins: list[str]
):
    """Return the 1-based (start, end) of the recovered 5' micro-exon, or None.

    ``exon2`` is the (approximate) 1-based span of the found second exon.
    """
    if not ref_proteins or exon1_len < 3 or exon1_len > _MAX_MICRO_EXON:
        return None
    L = len(genome)
    b = exon2[0] if strand == 1 else exon2[1]  # exon2 5' genomic boundary
    refs = [_san(p.rstrip("*")) for p in ref_proteins]

    def spliced(e1s: int) -> tuple[str, str] | None:
        if strand == 1:
            end1 = e1s - 1 + exon1_len
            if e1s < 1 or end1 + 2 > L:
                return None
            e1 = genome[e1s - 1 : end1]
            donor = genome[end1 : end1 + 2]
            seq = e1 + genome[exon2[0] - 1 : exon2[1]]
        else:
            start1 = e1s - exon1_len
            if start1 - 2 < 0 or e1s > L:
                return None
            e1 = str(Seq(genome[start1:e1s]).reverse_complement())
            donor = str(Seq(genome[start1 - 2 : start1]).reverse_complement())
            seq = e1 + str(Seq(genome[exon2[0] - 1 : exon2[1]]).reverse_complement())
        return e1, donor, seq

    from .splice_flanks import group_ii_joins

    exon2_seq = genome[exon2[0] - 1 : exon2[1]] if strand == 1 else str(Seq(genome[exon2[0] - 1 : exon2[1]]).reverse_complement())
    candidates = []
    for gap in range(_INTRON_MIN, _INTRON_MAX):
        e1s = b - gap if strand == 1 else b + gap
        parts = spliced(e1s)
        if parts is None:
            continue
        e1, donor, seq = parts
        if e1[:3] not in _START or donor not in _DONOR:
            continue
        prot = _san(str(Seq(seq).translate(table=11)))
        if "*" in prot[:-1]:
            continue
        score = max((_AL.score(r, prot) for r in refs), default=float("-inf"))
        # transcript-sense intron between the candidate exon 1 and exon 2
        if strand == 1:
            intron = genome[e1s - 1 + exon1_len : exon2[0] - 1]
        else:
            intron = str(Seq(genome[exon2[1] : e1s - exon1_len]).reverse_complement())
        local = e1 + intron + exon2_seq[:30]
        motif = group_ii_joins(local, [(1, len(e1)), (len(e1) + len(intron) + 1, len(local))])
        candidates.append((score, motif, len(candidates), e1s))
    if not candidates:
        return None
    top = max(c[0] for c in candidates)
    near = [c for c in candidates if c[0] >= top - _MICRO_SCORE_MARGIN]
    e1s = max(near, key=lambda c: (c[1], c[0], -c[2]))[3]
    return (e1s, e1s + exon1_len - 1) if strand == 1 else (e1s - exon1_len + 1, e1s)


def recover_micro_exons(
    genome: str,
    features: list[SeqFeature],
    ref_proteins: dict[str, list[str]],
    micro_map: dict[str, int],
) -> list[SeqFeature]:
    """Prepend recovered micro first exons to single-exon CDS in ``micro_map``.

    ``micro_map`` maps gene -> reference first-exon length. Operates in place.
    """
    genome = genome.upper()
    for feat in features:
        if feat.type != "CDS" or len(feat.location.parts) != 1:
            continue
        gene = feat.qualifiers.get("gene", [""])[0]
        exon1_len = micro_map.get(gene)
        if not exon1_len or gene not in ref_proteins:
            continue
        strand = feat.location.strand or 1
        exon2 = (int(feat.location.start) + 1, int(feat.location.end))
        e1 = find_micro_exon(genome, exon2, strand, exon1_len, ref_proteins[gene])
        if e1 is None:
            continue
        # Assemble a two-exon compound location in transcription order.
        loc1 = FeatureLocation(e1[0] - 1, e1[1], strand=strand)
        loc2 = FeatureLocation(exon2[0] - 1, exon2[1], strand=strand)
        ordered = [loc1, loc2] if strand == 1 else [loc2, loc1]
        feat.location = CompoundLocation(ordered)
    return features


# ---------------------------------------------------------------- 3' terminal micro-exons

_MAX_TERMINAL_EXON = 40  # rps12 exon 3 is 26-30 nt: below the blastn word size, like petB exon 1
_STOP = ("TAA", "TAG", "TGA")
_ACCEPTOR = ("AG", "AC", "AT")  # group II introns end AY
_DONOR_SLACK = 9
_LENGTH_SLACK = 6


def _tx(genome: str, start: int, end: int, strand: int) -> str:
    """Transcript-sense sequence of 1-based genome span [start, end]."""
    s = genome[start - 1 : end]
    return s if strand == 1 else str(Seq(s).reverse_complement())


def find_terminal_micro_exon(
    genome: str,
    upstream: str,
    last_part: tuple[int, int],
    strand: int,
    penultimate_len: int,
    terminal_len: int,
    ref_proteins: list[str],
):
    """Return ((exon start, exon end) of the trimmed last exon, (start, end) of the new 3' exon), or None.

    ``upstream`` is the spliced coding sequence before ``last_part`` (transcript
    sense, from the start codon). The last part's 5' end stays; its 3' end is
    moved to a GT/GC donor near the reference exon length, and a short final exon
    ending in an in-frame stop is sought 400-1600 bp downstream behind an AG/AY
    acceptor. The spliced translation must be an open reading frame; the best
    match to a reference protein wins.
    """
    if not ref_proteins or terminal_len > _MAX_TERMINAL_EXON:
        return None
    L = len(genome)
    refs = [_san(p.rstrip("*")) for p in ref_proteins]
    a, b = last_part
    five = a if strand == 1 else b  # transcript 5' end of the last part

    def pos(offset: int) -> int:  # genome position `offset` nt downstream (transcript sense) of `five`
        return five + offset if strand == 1 else five - offset

    best = None
    for exon_len in range(penultimate_len - _DONOR_SLACK, penultimate_len + _DONOR_SLACK + 1):
        if exon_len < 10:
            continue
        end2 = pos(exon_len - 1)
        if not 1 <= end2 <= L:
            continue
        donor = _tx(genome, min(pos(exon_len), pos(exon_len + 1)), max(pos(exon_len), pos(exon_len + 1)), strand)
        if donor not in _DONOR:
            continue
        body = upstream + _tx(genome, min(five, end2), max(five, end2), strand)
        for gap in range(_INTRON_MIN, _INTRON_MAX):
            start3 = pos(exon_len + gap)  # first base of the terminal exon
            if not 1 <= start3 <= L:
                break
            acc_lo, acc_hi = sorted((pos(exon_len + gap - 2), pos(exon_len + gap - 1)))
            if _tx(genome, acc_lo, acc_hi, strand) not in _ACCEPTOR:
                continue
            for n3 in range(terminal_len - _LENGTH_SLACK, terminal_len + _LENGTH_SLACK + 1):
                end3 = pos(exon_len + gap + n3 - 1)
                if not 1 <= end3 <= L or (len(body) + n3) % 3:
                    continue
                coding = body + _tx(genome, min(start3, end3), max(start3, end3), strand)
                if coding[-3:] not in _STOP:
                    continue
                prot = str(Seq(coding).translate(table=11))
                if "*" in prot[:-1]:
                    continue
                score = max((_AL.score(r, _san(prot)) for r in refs), default=float("-inf"))
                if best is None or score > best[0]:
                    best = (score, (min(five, end2), max(five, end2)), (min(start3, end3), max(start3, end3)))
    if best is None:
        return None
    current = upstream + _tx(genome, a, b, strand)
    current = current[: len(current) - len(current) % 3]
    current_score = max((_AL.score(r, _san(str(Seq(current).translate(table=11)))) for r in refs), default=float("-inf"))
    return (best[1], best[2]) if best[0] > current_score else None


def recover_terminal_micro_exons(
    genome: str,
    features: list[SeqFeature],
    ref_proteins: dict[str, list[str]],
    terminal_map: dict[str, tuple[int, int, int]],
) -> list[SeqFeature]:
    """Append a missing short 3' exon (rps12 exon 3) to CDS one exon short of their references.

    ``terminal_map``: gene -> (reference exon count, penultimate exon length,
    terminal exon length). Operates in place.
    """
    genome = genome.upper()
    for feat in features:
        if feat.type != "CDS":
            continue
        gene = feat.qualifiers.get("gene", [""])[0]
        spec = terminal_map.get(gene)
        if spec is None or gene not in ref_proteins:
            continue
        n_exons, penultimate_len, terminal_len = spec
        parts = list(getattr(feat.location, "parts", [feat.location]))
        if len(parts) != n_exons - 1:
            continue
        last = parts[-1]
        strand = last.strand or 1
        upstream = "".join(_tx(genome, int(p.start) + 1, int(p.end), p.strand or 1) for p in parts[:-1])
        found = find_terminal_micro_exon(
            genome, upstream, (int(last.start) + 1, int(last.end)), strand,
            penultimate_len, terminal_len, ref_proteins[gene],
        )
        if found is None:
            continue
        (s2, e2), (s3, e3) = found
        feat.location = CompoundLocation(
            [*parts[:-1], FeatureLocation(s2 - 1, e2, strand=strand), FeatureLocation(s3 - 1, e3, strand=strand)]
        )
    return features
