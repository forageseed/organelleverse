"""Intron tRNAs found from conserved reference exon sequences.

The HMM filter cannot anchor trnG-UCC's 23 nt 5' exon, so without a
transferred gene span only its 3' exon was reported (43-51 nt, unspliced) in
Arabidopsis, Marchantia, Pteridium and Podocarpus. Near-exact matches of
the intron-adjacent ends of reference exons, an intron apart, now bound the
gene for splicing.
"""

from __future__ import annotations

import random

from Bio.Seq import Seq

from organelleverse.annotation.cmsearch.models import CMHit
from organelleverse.annotation.plastome.native_features import (
    _encode,
    _near_matches,
    _seed_hints,
)

# Arabidopsis trnV-UAC exons (transcription order)
FIVE = "AGGGCTATAGCTCAGTTAGGTAGAGCACCTCGTTTACAC"
THREE = "CGAGCAGGTCTACGGTTCGAGTCCGTATAGCCCTA"
SEEDS = {"V": [(FIVE, THREE)]}
LENGTHS = {"V": (39, 35)}


def _genome(seed: int = 7) -> tuple[str, int, int]:
    rng = random.Random(seed)
    rand = lambda n: "".join(rng.choice("ACGT") for _ in range(n))  # noqa: E731
    five = FIVE[:10] + ("A" if FIVE[10] != "A" else "C") + FIVE[11:]  # one mismatch
    seq = rand(400) + five + rand(600) + THREE + rand(400)
    return seq, 401, 400 + len(FIVE) + 600 + len(THREE)


def test_near_matches_tolerate_a_few_mismatches() -> None:
    seq, start, _ = _genome()
    assert _near_matches(_encode(seq), FIVE) == [start - 1]
    assert _near_matches(_encode(seq), "ACGTACGTACGTACGTACGTACGT") == []


def test_seed_hints_bound_the_gene_on_both_strands() -> None:
    seq, start, end = _genome()

    (hint,) = _seed_hints(seq, SEEDS, LENGTHS, [])
    assert (hint.start, hint.end, hint.strand, hint.amino_acid) == (start, end, 1, "V")

    rev = str(Seq(seq).reverse_complement())
    n = len(seq)
    (hint,) = _seed_hints(rev, SEEDS, LENGTHS, [])
    assert (hint.start, hint.end, hint.strand) == (n - end + 1, n - start + 1, -1)


def test_an_indel_in_the_three_prime_exon_still_seeds_the_gene() -> None:
    seq, start, end = _genome()
    shortened = seq.replace(THREE, THREE[:-8] + THREE[-7:])  # T-arm deletion

    (hint,) = _seed_hints(shortened, SEEDS, LENGTHS, [])
    assert (hint.start, hint.strand) == (start, 1)
    assert abs(hint.end - (end - 1)) <= 1


def test_a_mis_annotated_reference_exon_does_not_move_the_start() -> None:
    seq, start, end = _genome()
    slipped = {"V": [(FIVE, THREE), ("A" * 90 + FIVE, THREE)]}  # exon 1 ran into flank

    (hint,) = _seed_hints(seq, slipped, LENGTHS, [])
    assert (hint.start, hint.end) == (start, end)


def test_a_five_prime_seed_inside_a_complete_trna_is_ignored() -> None:
    seq, start, _ = _genome()
    complete = CMHit(start=start - 20, end=start + 55, strand=1, score=50.0)

    assert _seed_hints(seq, SEEDS, LENGTHS, [complete]) == []
