"""Intron-containing plastid tRNAs are emitted spliced, with exact mature ends.

Before the fix, pair recovery kept only the exon1..exon2 span (strand 0), the
plastome writer re-guessed the intron inside fixed exon-length windows and
emitted the unspliced span (often unnamed), and intron tRNAs whose short exons
the HMM filter cannot anchor (trnK-UUU, trnG-UCC, trnL-UAA) were lost.

The exons below are Arabidopsis trnV-UAC (NC_000932, complement(join(
51199..51233, 51833..51871))) in transcription order; the intron is synthetic.
"""

from __future__ import annotations

import random

import pytest
from Bio.SeqFeature import CompoundLocation

from organelleverse.annotation.cmsearch.genome import default_plastid_cm_path
from organelleverse.annotation.cmsearch.introns import splice_trna_in_span
from organelleverse.annotation.cmsearch.model import parse_cm
from organelleverse.annotation.cmsearch.models import CMHit
from organelleverse.annotation.cmsearch.search import _resolve_overlaps
from organelleverse.annotation.plastome.native_features import (
    IntronTRNAHint,
    native_trna_features,
)

FIVE = "AGGGCTATAGCTCAGTTAGGTAGAGCACCTCGTTTACAC"  # 39 nt
THREE = "CGAGCAGGTCTACGGTTCGAGTCCGTATAGCCCTA"  # 35 nt
FLANK, INTRON = 300, 600


def _rc(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _genome() -> tuple[str, tuple[int, int], tuple[int, int]]:
    rng = random.Random(7)

    def rand(n: int) -> str:
        return "".join(rng.choice("ACGT") for _ in range(n))

    seq = rand(FLANK) + FIVE + rand(INTRON) + THREE + rand(FLANK)
    exon1 = (FLANK + 1, FLANK + len(FIVE))
    exon2 = (exon1[1] + INTRON + 1, exon1[1] + INTRON + len(THREE))
    return seq, exon1, exon2


@pytest.fixture(scope="module")
def model():
    return parse_cm(default_plastid_cm_path())


def test_span_splice_returns_exact_exons_from_an_approximate_span(model) -> None:
    seq, exon1, exon2 = _genome()

    # Transferred span off by a few nt; exon lengths are the reference median.
    hit = splice_trna_in_span(seq, exon1[0] - 4, exon2[1] + 3, 1, (39, 35), model, expected_aa="V")

    assert hit is not None
    assert hit.strand == 1
    assert ((hit.exon1_start, hit.exon1_end), (hit.exon2_start, hit.exon2_end)) == (exon1, exon2)


def test_span_splice_on_the_minus_strand(model) -> None:
    seq, exon1, exon2 = _genome()
    rev = _rc(seq)
    n = len(rev)
    low = (n - exon2[1] + 1, n - exon2[0] + 1)  # 3' exon, lower coordinates
    high = (n - exon1[1] + 1, n - exon1[0] + 1)  # 5' exon

    hit = splice_trna_in_span(rev, low[0] - 2, high[1] + 5, -1, (39, 35), model, expected_aa="V")

    assert hit is not None
    assert hit.strand == -1
    assert ((hit.exon1_start, hit.exon1_end), (hit.exon2_start, hit.exon2_end)) == (low, high)


def test_native_features_emit_a_spliced_named_trna(model) -> None:
    seq, exon1, exon2 = _genome()
    hint = IntronTRNAHint(
        start=exon1[0], end=exon2[1], strand=1, exon_lengths=(39, 35), amino_acid="V"
    )

    features = native_trna_features(seq, intron_hints=[hint], intron_exon_lengths={"V": (39, 35)})

    spliced = [f for f in features if isinstance(f.location, CompoundLocation)]
    assert len(spliced) == 1
    feature = spliced[0]
    assert feature.qualifiers["gene"] == ["trnV-UAC"]
    assert [(int(p.start) + 1, int(p.end)) for p in feature.location.parts] == [exon1, exon2]
    assert "intron-containing tRNA" in feature.qualifiers["note"]
    assert not any(
        not isinstance(f.location, CompoundLocation)
        and int(f.location.start) < exon2[1]
        and int(f.location.end) > exon1[0]
        and len(f.location) > 100
        for f in features
    ), "the unspliced exon1..exon2 span must not be emitted"


def test_hit_inside_an_intron_survives_overlap_resolution() -> None:
    spliced = CMHit(start=100, end=2700, strand=1, score=60.0, exons=((100, 135), (2665, 2700)))
    inside = CMHit(start=1000, end=1072, strand=1, score=40.0)
    clash = CMHit(start=2660, end=2730, strand=1, score=25.0)

    kept = _resolve_overlaps([inside, clash, spliced])

    assert spliced in kept
    assert inside in kept
    assert clash not in kept


def test_reference_identity_overrides_a_misread_anticodon(model) -> None:
    # trnV-UAC's loop (CGUUUACAC) spliced two nt off reads UUU (trnK); a
    # reference span on the same strand decides the identity instead.
    from organelleverse.annotation.plastome.native_features import _expected_identity

    seq, exon1, exon2 = _genome()
    hit = CMHit(start=exon1[0], end=exon2[1], strand=1, score=60.0, exons=(exon1, exon2))
    hint = IntronTRNAHint(
        start=exon1[0] - 3, end=exon2[1] + 2, strand=1, exon_lengths=(39, 35), amino_acid="V"
    )
    other_strand = IntronTRNAHint(
        start=exon1[0], end=exon2[1], strand=-1, exon_lengths=(37, 35), amino_acid="K"
    )

    assert _expected_identity(seq, hit, [other_strand, hint], model) == "V"
    assert _expected_identity(seq, hit, [other_strand], model) == "V"  # read from the splice
