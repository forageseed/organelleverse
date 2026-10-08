"""Mirroring IR genes into the other inverted repeat.

Reference transfer keeps one best hit per query, so genes wholly inside an IR
(ycf2, rps7, rpl23, ndhB, ...) came out with one copy: 17 of 41 missed CDS in
the leave-one-species-out plastome benchmark were lost second IR copies.
"""

from __future__ import annotations

import random

from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from organelleverse.annotation.plastome.ir_mirror import cohere_ir_exons, mirror_ir_copies

LSC, IR, SSC = 3000, 2000, 1000


def _rand(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _rc(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


def _plastome(extra_base_in_irb: bool = False):
    rng = random.Random(3)
    ir = _rand(rng, IR)
    irb = _rc(ir)
    if extra_base_in_irb:
        # One-base indel between the copies, in IRb ahead of the mirrored gene
        # (which lands at IRb offsets 32-500), so arithmetic mapping is off by one.
        irb = irb[:10] + "A" + irb[10:]
    seq = _rand(rng, LSC) + ir + _rand(rng, SSC) + irb
    ira_region = (LSC, LSC + IR, 1)
    irb_start = LSC + IR + SSC
    irb_region = (irb_start, irb_start + len(irb), -1)
    return seq, (ira_region, irb_region)


def _cds(gene: str, parts: list[tuple[int, int, int]]) -> SeqFeature:
    locs = [FeatureLocation(s, e, strand=st) for s, e, st in parts]
    location = locs[0] if len(locs) == 1 else CompoundLocation(locs)
    return SeqFeature(location, type="CDS", qualifiers={"gene": [gene], "translation": ["MX"]})


def _extract(seq: str, feature: SeqFeature) -> str:
    return str(feature.location.extract(Seq(seq)))


def test_two_exon_gene_is_mirrored_with_the_same_spliced_sequence() -> None:
    seq, irs = _plastome()
    # rpl2-like: two exons on the plus strand inside IRa.
    original = _cds("rpl2", [(LSC + 400, LSC + 790, 1), (LSC + 1450, LSC + 1880, 1)])
    features = [original]

    assert mirror_ir_copies(features, seq, irs) == 1

    copy = features[1]
    assert [p.strand for p in copy.location.parts] == [-1, -1]
    assert _extract(seq, copy) == _extract(seq, original)
    assert "translation" not in copy.qualifiers  # recomputed downstream
    irb_start = irs[1][0]
    assert all(irb_start <= int(p.start) and int(p.end) <= irs[1][1] for p in copy.location.parts)


def test_copies_differing_by_an_indel_are_relocated() -> None:
    seq, irs = _plastome(extra_base_in_irb=True)
    original = _cds("rps7", [(LSC + 1500, LSC + 1968, 1)])
    features = [original]

    assert mirror_ir_copies(features, seq, irs) == 1
    assert _extract(seq, features[1]) == _extract(seq, original)


def test_existing_copy_is_not_duplicated_and_boundary_genes_are_skipped() -> None:
    seq, irs = _plastome()
    gene = _cds("rpl23", [(LSC + 100, LSC + 385, 1)])
    features = [gene]
    mirror_ir_copies(features, seq, irs)
    # Second pass: both copies exist now, and one of them is the source.
    assert mirror_ir_copies(features, seq, irs) == 0

    crossing = _cds("ycf1", [(LSC + IR - 300, LSC + IR + 400, 1)])  # IRa/SSC junction
    assert mirror_ir_copies([crossing], seq, irs) == 0


def test_ir_running_past_the_origin() -> None:
    seq, irs = _plastome()
    # Rotate so that IRb wraps from the end of the record to its start.
    shift = len(seq) - 700
    rotated = seq[shift:] + seq[:shift]
    ira = ((irs[0][0] - shift) % len(seq), (irs[0][0] - shift) % len(seq) + IR, 1)
    irb_start = (irs[1][0] - shift) % len(seq)
    irb = (irb_start, irb_start + IR, -1)  # end runs past len(rotated)
    original = _cds("ndhB", [(ira[0] + 1600, ira[0] + 1700, 1)])  # mirror lands after the origin
    features = [original]

    assert mirror_ir_copies(features, rotated, (ira, irb)) == 1
    assert _extract(rotated, features[1]) == _extract(rotated, original)


def test_trna_gene_features_are_not_mirrored() -> None:
    seq, irs = _plastome()
    trna_gene = SeqFeature(
        FeatureLocation(LSC + 50, LSC + 122, strand=1),
        type="gene",
        qualifiers={"gene": ["trnN-GUU"]},
    )

    assert mirror_ir_copies([trna_gene], seq, irs) == 0


def test_exons_from_opposite_ir_hits_form_two_coherent_copies() -> None:
    seq, irs = _plastome()
    original = _cds("ndhB", [(LSC + 400, LSC + 790, 1), (LSC + 1450, LSC + 1880, 1)])
    copies = [original]
    mirror_ir_copies(copies, seq, irs)
    mixed = _cds(
        "ndhB",
        [
            (int(p.start), int(p.end), p.strand)
            for p in [original.location.parts[0], copies[1].location.parts[1]]
        ],
    )
    before = _extract(seq, mixed)
    assert cohere_ir_exons([mixed], seq, irs) == 1
    assert mixed.location == original.location
    assert _extract(seq, mixed) == before
    features = [mixed]
    assert mirror_ir_copies(features, seq, irs) == 1
    assert all(_extract(seq, f) == before for f in features)
    assert cohere_ir_exons(features, seq, irs) == 0


def test_trans_spliced_gene_with_exon_outside_ir_is_preserved() -> None:
    seq, irs = _plastome()
    feature = _cds(
        "rps12", [(100, 200, 1), (LSC + 400, LSC + 600, 1), (irs[1][0] + 400, irs[1][0] + 600, -1)]
    )
    original = feature.location
    assert cohere_ir_exons([feature], seq, irs) == 0
    assert feature.location == original
