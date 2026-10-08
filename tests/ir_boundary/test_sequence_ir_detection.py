"""Sequence-level inverted-repeat detection behind ir_boundary()."""

from __future__ import annotations

import random

from organelleverse.ir_boundary.ir_boundary import _detect_ir_from_sequence


def _random_dna(length: int, seed: int) -> str:
    return "".join(random.Random(seed).choices("ACGT", k=length))


_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def test_detects_a_synthetic_inverted_repeat_pair() -> None:
    ir = _random_dna(3000, seed=7)
    genome = (
        _random_dna(20000, seed=1)
        + ir
        + _random_dna(5000, seed=2)
        + ir.translate(_COMPLEMENT)[::-1]
        + _random_dna(20000, seed=3)
    )
    detected = _detect_ir_from_sequence(genome)
    assert detected is not None
    ira, irb = detected
    # Random flanks let the mismatch-run walk drift a few tens of bases;
    # real genomes anchor junctions far tighter (published IR lengths were
    # recovered to within ~1% on the reference panel).
    assert abs(irb["start"] - 20000) <= 60
    assert abs(irb["end"] - 23000) <= 60
    assert abs(ira["start"] - 28000) <= 60
    assert abs(ira["end"] - 31000) <= 60


def test_detects_ir_with_diverged_copies() -> None:
    copy1 = list(_random_dna(5000, seed=11))
    copy2 = list(copy1)
    rng = random.Random(12)
    for index in range(0, 5000, 50):  # ~2% divergence
        copy2[index] = rng.choice("ACGT")
    genome = (
        _random_dna(15000, seed=13)
        + "".join(copy1)
        + _random_dna(4000, seed=14)
        + "".join(copy2).translate(_COMPLEMENT)[::-1]
        + _random_dna(15000, seed=15)
    )
    detected = _detect_ir_from_sequence(genome)
    assert detected is not None
    ira, irb = detected
    length = min(ira["end"] - ira["start"], irb["end"] - irb["start"])
    assert 4400 <= length <= 5060


def test_irless_sequence_returns_none() -> None:
    genome = _random_dna(60000, seed=21)
    assert _detect_ir_from_sequence(genome) is None


def test_stride_aliasing_cannot_hide_the_ir() -> None:
    """Both sides striding by the same k would miss IRs whose offset invariant
    is incongruent with the stride; the detector must probe independently."""
    for seed in range(4):
        ir = _random_dna(4000, seed=100 + seed)
        genome = (
            _random_dna(10000 + seed * 37, seed=200 + seed)
            + ir
            + _random_dna(3000, seed=300 + seed)
            + ir.translate(_COMPLEMENT)[::-1]
            + _random_dna(10000, seed=400 + seed)
        )
        assert _detect_ir_from_sequence(genome) is not None


def test_circular_ir_is_invariant_when_origin_splits_either_copy():
    ir = _random_dna(2000, seed=52)
    genome = (
        _random_dna(9000, seed=53)
        + ir
        + _random_dna(4000, seed=54)
        + ir.translate(_COMPLEMENT)[::-1]
    )

    def coordinates(offset):
        rotated = genome[offset:] + genome[:offset]
        pair = _detect_ir_from_sequence(rotated, circular=True)
        assert pair is not None
        return sorted(((r["start"] + offset) % len(genome), r["end"] - r["start"]) for r in pair)

    expected = coordinates(0)
    for offset in (100, 9500, 12000, 15500):
        assert coordinates(offset) == expected


def test_ir_extension_discards_terminal_mismatch_run():
    from organelleverse.ir_boundary.ir_boundary import _extend_arm

    ir = _random_dna(2000, seed=55)
    # AAAAA vs reverse-complement AAAAA gives exactly five mismatches.
    seq = "AAAAA" + ir + "AAAAA" + ir.translate(_COMPLEMENT)[::-1] + "AAAAA"
    assert _extend_arm(seq, 100, 1900, 2115, 3915) == ((5, 2005), (2010, 4010))
