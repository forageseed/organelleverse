"""Clean-room tRNA structural feature search."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from .models import StemFeature, StemPair, TRNACandidate, TRNAFeatureSet

PairClass = Literal["wc", "wobble", "mismatch", "gap"]

_WC = {("a", "t"), ("t", "a"), ("g", "c"), ("c", "g")}
_WOBBLE = {("g", "t"), ("t", "g")}
_PAIR_SCORES = {"wc": 2.0, "wobble": 1.0, "mismatch": -2.0, "gap": -3.0}


def classify_pair(left: str, right: str) -> PairClass:
    left_base = left.lower().replace("u", "t")
    right_base = right.lower().replace("u", "t")
    if left_base not in {"a", "c", "g", "t"} or right_base not in {"a", "c", "g", "t"}:
        return "gap"
    if (left_base, right_base) in _WC:
        return "wc"
    if (left_base, right_base) in _WOBBLE:
        return "wobble"
    return "mismatch"


def find_structural_models(
    sequence: str, anticodon_offset: int, *, organelle: str = "mitochondrion"
) -> list[TRNAFeatureSet]:
    if organelle not in {"mitochondrion", "plastome"}:
        raise ValueError(f"Unsupported organelle for tRNA structure search: {organelle!r}")
    seq = sequence.upper().replace("U", "T")
    n = len(seq)
    if anticodon_offset < 15 or anticodon_offset > n - 15:
        return []

    stems = [
        _best_stem(
            seq,
            "acceptor",
            left_starts=range(0, 3),
            right_starts=_range(n - 9, n - 4),
            lengths=(5, 6, 7),
            min_score=6.0,
        ),
        _best_stem(
            seq,
            "d_arm",
            left_starts=_range(8, min(16, anticodon_offset - 11)),
            right_starts=_range(18, min(anticodon_offset - 5, 29)),
            lengths=(3, 4),
            min_score=3.0,
        ),
        _best_stem(
            seq,
            "anticodon",
            left_starts=_range(anticodon_offset - 9, anticodon_offset - 3),
            right_starts=_range(anticodon_offset + 5, anticodon_offset + 11),
            lengths=(4, 5),
            min_score=4.0,
        ),
        _best_stem(
            seq,
            "t_arm",
            left_starts=_range(max(anticodon_offset + 10, n - 29), n - 17),
            right_starts=_range(n - 16, n - 5),
            lengths=(4, 5),
            min_score=4.0,
        ),
    ]
    accepted = tuple(stem for stem in stems if stem is not None)
    if len(accepted) < 3 or not any(stem.name == "anticodon" for stem in accepted):
        return []
    score = sum(stem.score for stem in accepted)
    if score < 8.0:
        return []
    return [
        TRNAFeatureSet(
            anticodon_offset=anticodon_offset,
            stems=accepted,
            structure_score=score,
        )
    ]


def best_structural_model(
    sequence: str, anticodon_offset: int, *, organelle: str = "mitochondrion"
) -> TRNAFeatureSet | None:
    models = find_structural_models(sequence, anticodon_offset, organelle=organelle)
    if not models:
        return None
    return max(models, key=lambda model: model.structure_score)


def offset_to_genomic_span(
    candidate: TRNACandidate, local_span: tuple[int, int]
) -> tuple[int, int]:
    left, right = sorted(local_span)
    if left < 0 or right >= len(candidate.sequence):
        raise ValueError("Local tRNA span falls outside candidate sequence")
    if candidate.strand == 1:
        return candidate.start + left, candidate.start + right
    return candidate.end - right, candidate.end - left


def _best_stem(
    seq: str,
    name: str,
    *,
    left_starts: Iterable[int],
    right_starts: Iterable[int],
    lengths: Iterable[int],
    min_score: float,
) -> StemFeature | None:
    candidates = []
    n = len(seq)
    for length in lengths:
        for left_start in left_starts:
            for right_start in right_starts:
                if left_start < 0 or right_start < 0:
                    continue
                if left_start + length > n or right_start + length > n:
                    continue
                if left_start + length >= right_start:
                    continue
                stem = _build_stem(seq, name, left_start, right_start, length)
                if _accepted_stem(stem, min_score):
                    candidates.append(stem)
    if not candidates:
        return None
    return max(candidates, key=lambda stem: (stem.score, -stem.mismatches, len(stem.pairs)))


def _build_stem(seq: str, name: str, left_start: int, right_start: int, length: int) -> StemFeature:
    pairs = []
    mismatches = 0
    gaps = 0
    score = 0.0
    for offset in range(length):
        left_offset = left_start + offset
        right_offset = right_start + length - offset - 1
        pair_class = classify_pair(seq[left_offset], seq[right_offset])
        score += _PAIR_SCORES[pair_class]
        if pair_class == "mismatch":
            mismatches += 1
        elif pair_class == "gap":
            gaps += 1
        pairs.append(
            StemPair(
                left_offset=left_offset,
                right_offset=right_offset,
                left_base=seq[left_offset],
                right_base=seq[right_offset],
                pair_class=pair_class,
            )
        )
    return StemFeature(
        name=name,
        left=(left_start, left_start + length - 1),
        right=(right_start, right_start + length - 1),
        pairs=tuple(pairs),
        mismatches=mismatches,
        gaps=gaps,
        score=score,
    )


def _accepted_stem(stem: StemFeature, min_score: float) -> bool:
    if stem.gaps:
        return False
    if stem.mismatches > max(1, len(stem.pairs) // 2):
        return False
    return stem.score >= min_score


def _range(start: int, stop: int) -> range:
    return range(max(0, start), max(0, stop))
