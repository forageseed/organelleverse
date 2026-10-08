"""A micro first exon is placed before a group II intron boundary when protein scores cannot decide."""

from __future__ import annotations

from organelleverse.annotation.plastome import micro_exon


def _genome():
    # true exon 1 (9 nt, ATG...) then the intron GTGCG...AT; a decoy 9 nt further on also starts with a
    # start codon (GTG) and is followed by GT, i.e. it is a valid but wrong candidate closer to exon 2.
    exon1 = "ATGACTATA"
    intron = "GTGCGAGTGGT" + "ATCTTTTCAAAGCTAAT" * 30 + "TTTAT"
    exon2 = "CAGCCAAAACGTACTCGTTTTCGTAAACAACATAGAGGAAGAATGAAAGGAATTTCTTATCGTGGAAATCGTATTTGTTTTGGAA"
    pad = "C" * 50
    genome = pad + exon1 + intron + exon2 + pad
    s1 = len(pad) + 1
    s2 = s1 + len(exon1) + len(intron)
    return genome, (s1, s1 + len(exon1) - 1), (s2, s2 + len(exon2) - 1)


def test_tied_scores_choose_the_group_ii_boundary(monkeypatch):
    genome, true_exon1, exon2 = _genome()

    class Flat:
        @staticmethod
        def score(a, b):
            return 10.0  # every candidate alike, as for a 2-3 residue exon

    monkeypatch.setattr(micro_exon, "_AL", Flat())
    monkeypatch.setattr(micro_exon, "_INTRON_MIN", 400)
    got = micro_exon.find_micro_exon(genome, exon2, 1, 9, ["MTIQPKRTRFRKQHRGRMKGISYRGNRICFG"])
    assert got == true_exon1


def test_clearly_better_protein_still_wins(monkeypatch):
    genome, true_exon1, exon2 = _genome()
    decoy_start = true_exon1[1] + 3  # within the intron: its GTG start codon

    class Prefers:
        @staticmethod
        def score(ref, prot):
            return 50.0 if prot.startswith("V") else 10.0  # the decoy's GTG translates to V

    monkeypatch.setattr(micro_exon, "_AL", Prefers())
    monkeypatch.setattr(micro_exon, "_INTRON_MIN", 400)
    got = micro_exon.find_micro_exon(genome, exon2, 1, 9, ["VRVQPKRTRF"])
    assert got is not None and got != true_exon1
    assert got[0] >= decoy_start - 3
