from __future__ import annotations

import random

import pytest

from organelleverse.annotation.cmsearch.introns import (
    detect_spliced_trna,
    recover_spliced_trna_near,
)
from organelleverse.annotation.cmsearch.model import parse_cm
from tests._paths import PROJECT_ROOT

_PLANT_MITO_CM = PROJECT_ROOT / "src/organelleverse/annotation/data/models/trna/plant_mito_trna.cm"

_PACKAGED_CM = PROJECT_ROOT / "src/organelleverse/annotation/data/models/trna/trna.cm"
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _PACKAGED_CM.exists(), reason="packaged tRNA CM not built"),
]

# Real Ranunculus trnL-UAA gene span (599832..600430, +): a 40 nt 5' exon, a
# 514 nt group-I intron in the anticodon loop, and a 45 nt 3' exon. RF00005
# scores the unspliced region ~11 bits (undetected) but the spliced mature tRNA
# ~46 bits.
_TRNL_UAA_REGION = (
    "GGGGATATGGCGAAATTGGTAGACGCTACGGACTTGATTGGATTGAGCCTTGGTATGGAAACATACTAAG"
    "TGATAACTTTCAAATTCAGAGAAACCCTGGAATAAAAAATGGGCAATCCTGAGCCAAATCCTGCTTTCAG"
    "AAAACAAAAAAAAGGGGGGATTCAGAAAGCGAAAATAGGGATAGGTGCAGAGACTCAATGGAAGCTGTTC"
    "TAACGAATAGAGTTGACTGCGTTGATCGAGGAATCCTTCTATTTGAATTCCCGAAAGGATGAACCAATCC"
    "ATATACGTACGTACAAATATGTAATAAAATAGGAAAGAAGCAACCCAAATCTTTATTTTTTATGTTATAT"
    "GCAAAACAAATAGAAGAATTCTTGTGAATCGATTCTAAGTTGAAGTAAGAATCGAATATTCAATATTCAT"
    "TGATCAAATCAGTTATTCTCTAACCTGGTAGATTTTTTAAAGAACTGACTGGTTGGACGAGAATAAAGAT"
    "AGAGTCCCATTATACATGTCAATATCAATACCGACAAAAAGGAAATTTATAGTAAGAGGAAAATCCGTCG"
    "ACTTTTAAAATCGTGAGGGTTCAAGTCCCTCTATCCCCA"
)


@pytest.mark.slow
def test_detect_intron_in_trnl_uaa():
    model = parse_cm(_PACKAGED_CM)
    hit = detect_spliced_trna(_TRNL_UAA_REGION, model, min_bits=25.0, min_intron=100)
    assert hit is not None
    # 1-based inclusive coordinates within the region. The exact intron boundary
    # is inherently ambiguous by ~1 nt (a CM cannot pin a group-I splice site to
    # the exact nucleotide), so assert the biologically-meaningful structure.
    assert hit.exon1_start == 1
    assert 38 <= hit.exon1_end <= 42  # 5' exon ~40 nt
    assert hit.exon2_end == 599  # 3' exon ends at the region end
    assert 43 <= (hit.exon2_end - hit.exon2_start + 1) <= 47  # 3' exon ~45 nt
    assert 508 <= (hit.intron_end - hit.intron_start + 1) <= 520  # intron ~514 nt
    # exons and intron partition the region contiguously
    assert hit.intron_start == hit.exon1_end + 1
    assert hit.exon2_start == hit.intron_end + 1
    assert hit.score > 40.0


def test_no_intron_call_on_mature_trna():
    # A plain mature tRNA (no intron) must not produce a spliced call.
    model = parse_cm(_PACKAGED_CM)
    mature = "AAAGCCGTTATCAAGTGGCAATGAGGCTCAACTTGCGATTGAGAAATCTCGGTATGAGTCCGTGCGGCTTTG"
    hit = detect_spliced_trna(mature, model, min_bits=25.0, min_intron=100)
    assert hit is None


@pytest.mark.slow
def test_no_intron_call_on_random_sequence():
    model = parse_cm(_PACKAGED_CM)
    random.seed(4)
    seq = "".join(random.choice("ACGT") for _ in range(600))
    hit = detect_spliced_trna(seq, model, min_bits=25.0, min_intron=100)
    assert hit is None


@pytest.mark.slow
@pytest.mark.skipif(not _PLANT_MITO_CM.exists(), reason="plant-mito CM not built")
def test_recover_intron_trna_from_single_exon_anchor():
    # trnL-UAA embedded in flanks; the plant-mito HMM anchors only the 3' exon
    # (positions 655-699 here). Recovery must bracket the full gene via the
    # acceptor stem and localize the intron, ending exactly at the 3' exon end.
    model = parse_cm(_PLANT_MITO_CM)
    genome = "A" * 100 + _TRNL_UAA_REGION + "A" * 100
    gene_end = 100 + len(_TRNL_UAA_REGION)  # 699
    hit = recover_spliced_trna_near(genome, gene_end - 44, gene_end, model, min_bits=25.0)
    assert hit is not None
    assert hit.exon1_start == 101  # gene starts at region start
    assert hit.exon2_end == gene_end  # gene ends at the 3' exon end
    assert 508 <= (hit.intron_end - hit.intron_start + 1) <= 540
    assert hit.score > 40.0
