"""transfer._merge_overlapping must not bridge unrelated organelle regions.

It used to merge on nuclear overlap alone and then report the organelle side as
the min..max hull, so two hits from different organelle loci that happened to
overlap on the nuclear genome became one candidate spanning everything between
them; it also kept the *first* hit's strand and the *maximum* identity.
"""

from __future__ import annotations

import pytest

from organelleverse.transfer.transfer import TransferCandidate, _merge_overlapping


def _cand(
    nuc: tuple[int, int],
    org: tuple[int, int],
    *,
    identity: float = 95.0,
    strand: str = "+",
    chrom: str = "chr1",
    org_id: str = "mt",
    bitscore: float = 100.0,
) -> TransferCandidate:
    return TransferCandidate(
        nuclear_seqid=chrom,
        nuclear_start=nuc[0],
        nuclear_end=nuc[1],
        organelle_seqid=org_id,
        organelle_start=org[0],
        organelle_end=org[1],
        identity=identity,
        length=nuc[1] - nuc[0] + 1,
        evalue=1e-30,
        bitscore=bitscore,
        strand=strand,
    )


def test_nuclear_overlap_with_distant_organelle_regions_stays_separate() -> None:
    first = _cand((1000, 1300), (10_000, 10_300))
    second = _cand((1200, 1500), (250_000, 250_300))

    merged = _merge_overlapping([first, second])

    assert len(merged) == 2
    spans = sorted((c.organelle_start, c.organelle_end) for c in merged)
    assert spans == [(10_000, 10_300), (250_000, 250_300)]  # no 240 kb hull


def test_collinear_overlapping_hits_merge_with_weighted_identity() -> None:
    long_hit = _cand((1000, 1999), (5000, 5999), identity=90.0)  # 1000 bp
    short_hit = _cand((1900, 2099), (5900, 6099), identity=100.0)  # 200 bp, 100 bp overlap

    (merged,) = _merge_overlapping([long_hit, short_hit])

    assert (merged.nuclear_start, merged.nuclear_end) == (1000, 2099)
    assert (merged.organelle_start, merged.organelle_end) == (5000, 6099)
    # span-weighted mean (1000 @ 90, 200 @ 100), not max(90, 100)
    assert merged.identity == pytest.approx((1000 * 90 + 200 * 100) / 1200, abs=0.01)


def test_opposite_strands_are_never_merged() -> None:
    forward = _cand((1000, 1300), (5000, 5300), strand="+")
    reverse = _cand((1200, 1500), (5100, 5400), strand="-")

    merged = _merge_overlapping([forward, reverse])

    assert sorted(c.strand for c in merged) == ["+", "-"]


def test_different_organelle_records_are_not_merged() -> None:
    first = _cand((1000, 1300), (100, 400), org_id="ctgA")
    second = _cand((1200, 1500), (100, 400), org_id="ctgB")

    assert len(_merge_overlapping([first, second])) == 2


def test_an_unrelated_hit_between_two_mergeable_hits_does_not_block_them() -> None:
    a = _cand((100, 300), (1000, 1200))
    unrelated = _cand((150, 250), (5000, 5100))
    c = _cand((280, 400), (1180, 1300))

    merged = _merge_overlapping([a, unrelated, c])

    assert len(merged) == 2
    joined = next(m for m in merged if m.organelle_start == 1000)
    assert (joined.nuclear_start, joined.nuclear_end) == (100, 400)
    assert (joined.organelle_start, joined.organelle_end) == (1000, 1300)


def test_other_chromosomes_and_single_candidates_pass_through() -> None:
    only = _cand((10, 200), (1, 190))
    assert _merge_overlapping([only]) == [only]
    one = _cand((10, 200), (1, 190), chrom="chr1")
    two = _cand((10, 200), (1, 190), chrom="chr2")
    assert len(_merge_overlapping([one, two])) == 2
