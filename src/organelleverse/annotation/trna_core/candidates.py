"""Candidate window generation for the clean-room tRNA core."""

from __future__ import annotations

from .anticodon import isotypes_for_anticodon, known_anticodons
from .models import TRNACandidate

_DNA_COMPLEMENT = str.maketrans("ACGTUacgtuNn", "TGCAAtgcaaNn")


def scan_candidate_windows(
    sequence: str,
    *,
    organelle: str = "mitochondrion",
    circular: bool = False,
    min_len: int = 65,
    max_len: int = 130,
) -> list[TRNACandidate]:
    genome = sequence.upper().replace("U", "T")
    anticodons = set(known_anticodons(organelle))
    lengths = _candidate_lengths(min_len, max_len)
    anticodon_offsets = _candidate_anticodon_offsets()
    candidates: list[TRNACandidate] = []
    for strand, subject in ((1, genome), (-1, _reverse_complement(genome))):
        for pos in range(len(subject) - 2):
            anticodon = subject[pos : pos + 3].lower()
            if anticodon not in anticodons:
                continue
            for amino_acid in isotypes_for_anticodon(anticodon, organelle):
                for length in lengths:
                    if circular and length > len(subject):
                        continue
                    for anticodon_offset in anticodon_offsets:
                        start0 = pos - anticodon_offset
                        if circular:
                            window, segments = _window_circular(
                                subject, start0, length, strand, len(genome)
                            )
                        else:
                            if start0 < 0 or start0 + length > len(subject):
                                continue
                            window = subject[start0 : start0 + length]
                            segments = (_linear_segment(start0, length, strand, len(genome)),)
                        if "N" in window:
                            continue
                        start = min(segment[0] for segment in segments)
                        end = max(segment[1] for segment in segments)
                        candidates.append(
                            TRNACandidate(
                                sequence=window,
                                start=start,
                                end=end,
                                strand=strand,
                                anticodon=anticodon,
                                amino_acid=amino_acid,
                                organelle=organelle,
                                anticodon_offset=anticodon_offset,
                                segments=segments,
                            )
                        )
    return deduplicate_candidates(candidates)


def deduplicate_candidates(candidates: list[TRNACandidate]) -> list[TRNACandidate]:
    kept: list[TRNACandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.start,
            item.end,
            item.gene_name,
            abs(item.length - 73),
            abs((item.anticodon_offset if item.anticodon_offset is not None else 30) - 30),
            item.segments[0][0],
        ),
    ):
        if any(_same_locus(candidate, existing) for existing in kept):
            continue
        kept.append(candidate)
    return kept


def _same_locus(a: TRNACandidate, b: TRNACandidate) -> bool:
    if a.strand != b.strand or a.gene_name != b.gene_name:
        return False
    overlap = max(0, min(a.end, b.end) - max(a.start, b.start) + 1)
    shorter = min(a.end - a.start + 1, b.end - b.start + 1)
    return shorter > 0 and overlap / shorter >= 0.8


def _window_circular(
    subject: str, start0: int, length: int, strand: int, genome_length: int
) -> tuple[str, tuple[tuple[int, int], ...]]:
    n = len(subject)
    start_mod = start0 % n
    chars = [subject[(start_mod + offset) % n] for offset in range(length)]
    if start_mod + length <= n:
        segments = (_linear_segment(start_mod, length, strand, genome_length),)
    else:
        first_len = n - start_mod
        if strand == 1:
            segments = ((start_mod + 1, n), (1, length - first_len))
        else:
            segments = (
                _linear_segment(start_mod, first_len, strand, genome_length),
                _linear_segment(0, length - first_len, strand, genome_length),
            )
    return "".join(chars), segments


def _linear_segment(start0: int, length: int, strand: int, genome_length: int) -> tuple[int, int]:
    if strand == 1:
        return start0 + 1, start0 + length
    return genome_length - start0 - length + 1, genome_length - start0


def _reverse_complement(seq: str) -> str:
    return seq.translate(_DNA_COMPLEMENT)[::-1].upper().replace("U", "T")


def _candidate_lengths(min_len: int, max_len: int) -> tuple[int, ...]:
    preferred = (65, 70, 73, 75, 80, 90, 110, 130)
    values = [length for length in preferred if min_len <= length <= max_len]
    if values:
        return tuple(values)
    return (min_len,) if min_len == max_len else (min_len, max_len)


def _candidate_anticodon_offsets() -> tuple[int, ...]:
    return tuple(range(28, 39))
