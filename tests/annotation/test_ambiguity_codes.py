"""Ambiguity codes in an input genome keep their position (as N) instead of shifting later coordinates."""

from __future__ import annotations

from organelleverse.annotation.mitochondrion.rrna_native import _clean_sequence as rrna_clean
from organelleverse.annotation.mitochondrion.trna_native import _clean_sequence as trna_clean
from organelleverse.annotation.plastome.references import clean_seq


def test_plastome_clean_keeps_length_and_positions():
    seq = "acgtRacgtYNacgu\nac"
    out = clean_seq(seq)
    assert out == "ACGTNACGTNNACGTAC"
    assert len(out) == len(seq.replace("\n", ""))
    assert out.index("ACGTN", 1) == 5  # the base after the first ambiguity code stays at its position


def test_mitochondrial_clean_keeps_length():
    seq = "ACGTRYKMSWBDHVNacgu"
    for clean in (trna_clean, rrna_clean):
        out = clean(seq)
        assert len(out) == len(seq)
        assert out == "ACGT" + "N" * 11 + "ACGT"
