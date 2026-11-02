"""The start codon search keeps the open reading frame of the anchored span.

A transfer anchor a few nt off once let the search take a start codon in another frame and stop a few codons later
(psbH 9 bp, matK 9 bp, rpl22 6 bp in the blind-C species)."""

from Bio.Seq import Seq

from organelleverse.annotation.plastome.cds_refine import refine_cds_boundaries

# GTA AGT GAC: open in frame 0; frame 1 reads TAA and frame 2 TGA. Its GTG (index 4) is a start codon in frame 1.
UNIT = "GTAAGTGAC"
UPSTREAM = "CCCCCCCCCC"
GENE = "ATG" + UNIT * 40 + "TAA"
SEQ = UPSTREAM + GENE + "TTTTTTTTTT"
START = len(UPSTREAM) + 1
END = len(UPSTREAM) + len(GENE)


def test_off_frame_decoy_start_is_ignored_forward() -> None:
    # Anchor 6 nt into the gene: the GTG decoy at +1 is closer than the true ATG at -6, but it is in another frame.
    assert refine_cds_boundaries(SEQ, START + 6, END, 1) == (START, END)


def test_off_frame_decoy_start_is_ignored_reverse() -> None:
    rc = str(Seq(SEQ).reverse_complement())
    n = len(SEQ)
    start, end = n - END + 1, n - START + 1
    # On the minus strand the gene's 5' end is the high coordinate; move that anchor 6 nt into the gene.
    assert refine_cds_boundaries(rc, start, end - 6, -1) == (start, end)


def test_anchor_on_the_true_start_is_unchanged() -> None:
    assert refine_cds_boundaries(SEQ, START, END, 1) == (START, END)
