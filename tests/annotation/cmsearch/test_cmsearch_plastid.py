from __future__ import annotations

import random

import pytest

from organelleverse.annotation.cmsearch.genome import (
    default_plastid_cm_path,
    default_plastid_hmm_path,
)
from organelleverse.annotation.cmsearch.introns import recover_spliced_trna_from_exon_pair
from organelleverse.annotation.cmsearch.model import parse_cm
from tests._paths import PROJECT_ROOT

_PLASTID_CM = default_plastid_cm_path()
_HAS_PLASTID = _PLASTID_CM.name == "plastid_trna.cm" and _PLASTID_CM.exists()

# A real tRNA (recognised by the plastid/RF00005 CMs).
_CORE = "GGGTGTATAGCTCAGTTGGTAGAGCATTGGGCTTTTAACCTAATGGTCGCAGGTTCAAGTCCTGCTATACCCA"


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_PLASTID, reason="plastid CM not packaged")
def test_recover_from_exon_pair_splices_two_anchored_exons():
    # Build a genome where a real tRNA is split by an intron: exon1 (first 38 nt)
    # + a 700 nt intron + exon2 (rest). The two exon windows are the "anchors".
    model = parse_cm(_PLASTID_CM)
    random.seed(5)
    e1_len = 38
    exon1 = _CORE[:e1_len]
    exon2 = _CORE[e1_len:]
    intron = "".join(random.choice("ACGT") for _ in range(700))
    left = "".join(random.choice("ACGT") for _ in range(200))
    genome = left + exon1 + intron + exon2 + "".join(random.choice("ACGT") for _ in range(200))

    a1s = len(left) + 1
    a1e = a1s + e1_len - 1
    a2s = a1e + len(intron) + 1
    a2e = a2s + len(exon2) - 1
    hit = recover_spliced_trna_from_exon_pair(genome, (a1s, a1e), (a2s, a2e), model, min_bits=25.0)
    assert hit is not None
    assert hit.exon1_start == a1s
    assert hit.exon2_end == a2e
    assert hit.intron_end - hit.intron_start + 1 == len(intron)
    assert hit.score > 40.0


_NIC_CP = (
    PROJECT_ROOT
    / "src/organelleverse/annotation/data/plastome/references/Nicotiana_tabacum_chloroplast.gb"
)


@pytest.mark.slow
@pytest.mark.skipif(
    not _HAS_PLASTID or not _NIC_CP.exists(),
    reason="plastid CM or packaged chloroplast reference missing",
)
def test_plastid_genome_recall_with_intron_recovery():
    from Bio import SeqIO

    from organelleverse.annotation.cmsearch.genome import find_trnas

    rec = next(SeqIO.parse(str(_NIC_CP), "genbank"))
    genome = str(rec.seq).upper().replace("U", "T")
    gb = [
        (int(f.location.start) + 1, int(f.location.end)) for f in rec.features if f.type == "tRNA"
    ]

    hits = find_trnas(
        genome,
        cm_model=parse_cm(_PLASTID_CM),
        hmm_path=default_plastid_hmm_path(),
        min_bits=20.0,
        recover_intron_pairs=True,
    )
    cand = [(h.start, h.end) for h in hits]

    used = [False] * len(cand)
    rec_n = 0
    for bs, be in gb:
        for i, (cs, ce) in enumerate(cand):
            if not used[i] and abs(bs - cs) <= 8 and abs(be - ce) <= 8:
                used[i] = True
                rec_n += 1
                break
    # Plastid model + intron recovery reaches ~0.92 recall vs GenBank tRNAs.
    assert rec_n / len(gb) >= 0.85
