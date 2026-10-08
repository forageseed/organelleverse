"""Deterministic tests for the reference-anchored mito boundary finder."""

from __future__ import annotations

import random
from pathlib import Path

from organelleverse.annotation.mitochondrion.boundary_db import (
    EXON_FLANK,
    INTRON_FLANK,
    refine_exon_boundaries,
)

_FLANK = EXON_FLANK + INTRON_FLANK


def _synthetic(tmp_path: Path):
    """Build a genome with a known 2-exon/1-intron structure and a matching DB.

    The 30 bp donor and acceptor flanks are distinctive; the rest of the genome is
    random, so the boundary PWM peaks uniquely at the true junctions.
    """
    rng = random.Random(42)
    bases = "ACGT"
    g = [rng.choice(bases) for _ in range(400)]
    # exon1 = 1..100, intron = 101..200, exon2 = 201..300 (1-based, + strand)
    donor = "ATCGATCGATCGATCGATCG" + "TTAGGGCTAA"  # 20 exon + 10 intron
    acceptor = "CTAACCTTAG" + "GGCCAATTGGCCAATTGGCC"  # 10 intron + 20 exon
    # donor spans genome positions 81..110 (0-based 80..110)
    for i, ch in enumerate(donor):
        g[80 + i] = ch
    # acceptor spans genome positions 191..220 (0-based 190..220)
    for i, ch in enumerate(acceptor):
        g[190 + i] = ch
    genome = "".join(g)

    db = tmp_path / "boundary_db.fasta"
    db.write_text(
        f">testgene|exon1|R|Sp_a\n{donor}\n"
        f">testgene|exon1|R|Sp_b\n{donor}\n"
        f">testgene|exon2|L|Sp_a\n{acceptor}\n"
        f">testgene|exon2|L|Sp_b\n{acceptor}\n"
    )
    return genome, str(db)


def test_finder_recovers_perturbed_boundary(tmp_path):
    genome, db_path = _synthetic(tmp_path)
    # True internal junction: exon1 ends at 100, exon2 starts at 201.
    perturbed = [(1, 103), (197, 300)]
    refined = refine_exon_boundaries("testgene", perturbed, genome, 1, db_path=db_path)
    assert refined is not None
    assert refined[0][1] == 100  # donor snapped back
    assert refined[1][0] == 201  # acceptor snapped back


def test_finder_leaves_correct_boundary_untouched(tmp_path):
    genome, db_path = _synthetic(tmp_path)
    # Already-correct boundaries: the conservative gate must not move them.
    correct = [(1, 100), (201, 300)]
    refined = refine_exon_boundaries("testgene", correct, genome, 1, db_path=db_path)
    # None (no change) or unchanged coordinates are both acceptable.
    assert refined is None or (refined[0][1] == 100 and refined[1][0] == 201)


def test_finder_returns_none_for_unknown_gene(tmp_path):
    genome, db_path = _synthetic(tmp_path)
    assert (
        refine_exon_boundaries("no_such_gene", [(1, 100), (201, 300)], genome, 1, db_path=db_path)
        is None
    )
