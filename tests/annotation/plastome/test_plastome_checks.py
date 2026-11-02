"""Rotation to trnH-GUG and the reported curation checks (short CDS, overlapping products)."""

import random

from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from organelleverse.annotation.plastome.models import BlastHit
from organelleverse.annotation.plastome.pipeline import (
    _overlap_warnings,
    _short_cds_warnings,
    rotate_to_trnh,
)


def _hit(start: int, end: int, strand: int) -> BlastHit:
    return BlastHit(query_id="q", pident=1.0, qcov=1.0, start=start, end=end, strand=strand, bitscore=100.0,
                    evalue=0.0, align_length=end - start)


def test_rotation_keeps_trnh_on_the_reverse_strand_at_the_start() -> None:
    seq = "".join(random.Random(1).choice("ACGT") for _ in range(300))
    out, flipped, shift = rotate_to_trnh(seq, _hit(100, 175, -1))
    assert (out, flipped, shift) == (seq[100:] + seq[:100], False, 100)
    out, flipped, shift = rotate_to_trnh(seq, _hit(100, 175, 1))
    assert flipped and shift == 300 - 175
    assert out[:75] == str(Seq(seq[100:175]).reverse_complement())
    assert len(out) == len(seq)


def _f(kind: str, gene: str, *spans: tuple[int, int], strand: int = 1, **q) -> SeqFeature:
    parts = [FeatureLocation(s, e, strand=strand) for s, e in spans]
    loc = parts[0] if len(parts) == 1 else CompoundLocation(parts)
    return SeqFeature(loc, type=kind, qualifiers={"gene": [gene], **q})


def test_short_cds_is_reported() -> None:
    feats = [_f("CDS", "psbH", (100, 109)), _f("CDS", "petN", (300, 390)),
             _f("CDS", "rps12", (500, 520), trans_splicing=[""])]
    warnings = _short_cds_warnings(feats)
    assert len(warnings) == 1 and "psbH 9 bp" in warnings[0]


def test_overlaps_are_reported_except_introns_and_usual_neighbours() -> None:
    feats = [
        _f("tRNA", "trnK-UUU", (1000, 1035), (3500, 3540)),  # matK lies in its intron: no exon overlap
        _f("CDS", "matK", (1500, 3000)),
        _f("CDS", "psbD", (5000, 6062)),
        _f("CDS", "psbC", (6010, 7400)),  # usual 52 bp overlap
        _f("CDS", "ycf3", (8000, 8500)),
        _f("CDS", "ycf4", (8470, 9000)),  # 30 bp overlap: reported
        _f("rRNA", "rrn5S", (10000, 10121)),
        _f("rRNA", "rrn5", (10000, 10115)),  # inside
    ]
    warnings = _overlap_warnings(feats)
    assert len(warnings) == 2
    assert warnings[0].startswith("overlap: CDS ycf3") and "share 30 bp" in warnings[0]
    assert warnings[1].startswith("inside:") and "rrn5" in warnings[1]
