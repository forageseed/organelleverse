"""Correctness tests for the reference profile-to-genome DP.

The DP's job is to recover a gene structure that a scoring-and-patching pipeline
cannot, so the tests plant known structures in flanking sequence and require the
exact coordinates back -- not merely a high score.
"""

from __future__ import annotations

import random

import pytest

from organelleverse.annotation.solver.dp import (
    Profile,
    START_CODONS,
    STOP_CODONS,
    solve,
    translate,
)

AA = "ACDEFGHIKLMNPQRSTVWY"


def make_profile(protein: str, *, name: str = "test") -> Profile:
    """A profile that rewards the intended residue and penalises the rest."""
    scores = []
    for aa in protein:
        row = [-2.0] * len(AA)
        if aa in AA:
            row[AA.index(aa)] = 4.0
        scores.append(row)
    return Profile(name=name, scores=scores)


def random_dna(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def coding_dna(protein: str, rng: random.Random) -> str:
    """Back-translate into a gene shaped like a real one.

    The solver requires an initiator and a terminator, so a planted gene that has
    neither is not a gene -- it is a stretch of codons. Fixtures therefore open on
    ATG and are followed by a stop by the callers below.
    """
    out = ["ATG"]
    for aa in protein[1:]:
        while True:
            codon = "".join(rng.choice("ACGT") for _ in range(3))
            if codon not in STOP_CODONS and translate(codon) == aa:
                out.append(codon)
                break
    return "".join(out)


@pytest.fixture
def rng() -> random.Random:
    return random.Random(20260804)


def test_recovers_single_exon_gene_exactly(rng: random.Random) -> None:
    protein = "M" + "".join(rng.choice(AA) for _ in range(39))
    cds = coding_dna(protein, rng)
    left, right = random_dna(120, rng), "TAA" + random_dna(120, rng)
    seq = left + cds + right

    sol = solve(make_profile(protein), seq, allow_introns=False)

    assert sol is not None
    assert len(sol.exons) == 1
    # the emitted CDS carries its terminator, as GenBank coordinates do
    assert (sol.exons[0].start, sol.exons[0].end) == (
        len(left) + 1,
        len(left) + len(cds) + 3,
    )


def test_never_places_an_in_frame_stop(rng: random.Random) -> None:
    """The stop-free reading frame is a constraint, not a preference.

    A stop codon is planted mid-gene with the profile still rewarding the residues
    on both sides, so a scoring approach would happily run through it.
    """
    protein = "M" + "".join(rng.choice(AA) for _ in range(29))
    cds = coding_dna(protein, rng)
    poisoned = cds[:45] + "TAA" + cds[48:]
    seq = random_dna(90, rng) + poisoned + "TAG" + random_dna(90, rng)

    sol = solve(make_profile(protein), seq, allow_introns=False)

    if sol is not None:
        for exon in sol.exons:
            block = seq[exon.start - 1 : exon.end]
            codons = [block[k : k + 3] for k in range(0, len(block) - 2, 3)]
            assert not (set(codons[:-1]) & STOP_CODONS)


def test_recovers_a_phase_preserving_intron(rng: random.Random) -> None:
    """An intron interrupting a codon must not break the reading frame.

    The intron is planted after one base of a codon -- phase 1 -- which is the
    case a contiguous-ORF finder cannot represent at all.
    """
    protein = "M" + "".join(rng.choice(AA) for _ in range(35))
    cds = coding_dna(protein, rng)
    cut = 3 * 18 + 1                      # one base into codon 19
    # the intron penalty is set from real intron density, so a planted intron has
    # to be worth having: the interrupted codons must not also align contiguously
    intron = "GT" + random_dna(996, rng) + "AT"
    seq = (random_dna(100, rng) + cds[:cut] + intron + cds[cut:]
           + "TAA" + random_dna(100, rng))

    sol = solve(make_profile(protein), seq, min_intron=400, max_intron=6000)

    assert sol is not None
    assert sol.cds_length == len(cds) + 3, "spliced CDS is the planted one plus its stop"
    assert len(sol.exons) == 2


def test_returns_none_when_the_gene_is_absent(rng: random.Random) -> None:
    protein = "M" + "".join(rng.choice(AA) for _ in range(39))
    unrelated = "M" + "".join(rng.choice(AA) for _ in range(39))
    seq = random_dna(80, rng) + coding_dna(unrelated, rng) + "TAA" + random_dna(80, rng)

    sol = solve(make_profile(protein), seq, allow_introns=False)

    assert sol is None or sol.score < len(protein) * 2.0


def test_recovers_a_phase_zero_intron(rng: random.Random) -> None:
    """An intron falling between two codons must be reachable.

    Phase 0 is the case where the intron does not interrupt a codon at all. The
    intron layer was opened and extended for it but never closed, so genes
    carrying one returned no solution rather than a worse one. It is 8% of
    cis-introns across the 29 Table S2 species (11/141), concentrated in nad5,
    nad2 and nad1 -- uncommon, but each occurrence loses the whole gene.
    """
    protein = "M" + "".join(rng.choice(AA) for _ in range(35))
    cds = coding_dna(protein, rng)
    cut = 3 * 18                          # exactly on a codon boundary
    intron = "GT" + random_dna(996, rng) + "AT"
    seq = (random_dna(100, rng) + cds[:cut] + intron + cds[cut:]
           + "TAA" + random_dna(100, rng))

    sol = solve(make_profile(protein), seq, min_intron=400, max_intron=6000)

    assert sol is not None, "phase-0 intron left the gene unreachable"
    assert sol.cds_length == len(cds) + 3
    assert len(sol.exons) == 2


def test_gene_shorter_than_profile_at_c_terminus(rng: random.Random) -> None:
    """Trailing profile positions must be deletable at the exit.

    Otherwise the deletions have to be spent before the last matched codon, which
    does not fail loudly -- it aligns the final codon to the wrong profile
    position and still scores positive.
    """
    protein = "M" + "".join(rng.choice(AA) for _ in range(29))
    cds = coding_dna(protein, rng)
    truncated = cds[: 3 * 25]             # gene stops five residues early
    seq = random_dna(90, rng) + truncated + "TAA" + random_dna(90, rng)

    sol = solve(make_profile(protein), seq, allow_introns=False)

    assert sol is not None
    assert sol.exons[0].start == 91
    assert sol.exons[-1].end == 90 + len(truncated) + 3


def test_gene_longer_than_profile_at_c_terminus(rng: random.Random) -> None:
    """Genome codons with no profile position must be absorbable.

    Without an insert state the only way to pass over them is a minimum-length
    intron, and that is what the DP did: on four Arabidopsis genes whose true
    length exceeds their profile it emitted introns of 402, 406, 422 and 569 bp
    against a 400 bp floor -- the cheapest legal jump rather than a feature.
    """
    protein = "M" + "".join(rng.choice(AA) for _ in range(29))
    cds = coding_dna(protein, rng)
    extra = coding_dna("M" + "".join(rng.choice(AA) for _ in range(5)), rng)[3:]
    seq = random_dna(90, rng) + cds + extra + "TAA" + random_dna(90, rng)

    sol = solve(make_profile(protein), seq, min_intron=400, max_intron=6000)

    assert sol is not None
    assert len(sol.exons) == 1, "extra codons must not be jumped with a fake intron"
    assert sol.exons[0].end == 90 + len(cds) + len(extra) + 3
