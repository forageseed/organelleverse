"""Read-through of U-to-C edited stop codons in fern-type plastomes.

Ferns edit U to C, so 52 of 165 RefSeq CDS in Adiantum and Pteridium carry
in-frame genomic stops; CDS refinement stopped at the first one and cut rpoB
to 129 of 3216 bp. Read-through is enabled only for a genome where several
genes show such early stops, so a seed-plant pseudogene is not "repaired".
"""

from __future__ import annotations

import random

from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature

from organelleverse.annotation.plastome.cds_refine import (
    refine_cds_boundaries_with_edits,
    refine_cds_features,
)
from organelleverse.annotation.plastome.pipeline import _apply_transl_except

SENSE = ["GCT", "GAA", "AAA", "CTT", "GGT", "TCT", "CCT", "ATT", "GTT", "TTT"]


def _gene(rng: random.Random, codons: int, stop_at: int | None) -> str:
    body = [rng.choice(SENSE) for _ in range(codons)]
    if stop_at is not None:
        body[stop_at] = "TAA"  # a genomic stop that editing removes (-> CAA, Gln)
    return "ATG" + "".join(body) + "TGA"


def test_read_through_stops_near_the_anchored_end() -> None:
    rng = random.Random(1)
    gene = _gene(rng, 200, stop_at=40)
    seq = "C" * 60 + gene + "C" * 60
    start, end = 61, 60 + len(gene)

    plain = refine_cds_boundaries_with_edits(seq, start, end, 1)
    through = refine_cds_boundaries_with_edits(seq, start, end, 1, tolerate_edits=True)

    assert plain == (start, start + 3 + 40 * 3 + 2, [])  # cut at the edited stop
    edit = start + 3 + 40 * 3
    assert through == (start, end, [(edit, edit + 2)])

    rc = str(Seq(seq).reverse_complement())
    n = len(seq)
    minus = refine_cds_boundaries_with_edits(
        rc, n - end + 1, n - start + 1, -1, tolerate_edits=True
    )
    assert minus == (n - end + 1, n - start + 1, [(n - edit - 1, n - edit + 1)])


def _plastome(early_stop_genes: int):
    rng = random.Random(2)
    seq, features = "", []
    for k in range(6):
        seq += "C" * 90
        gene = _gene(rng, 150, stop_at=30 if k < early_stop_genes else None)
        start = len(seq)
        seq += gene
        features.append(
            SeqFeature(
                FeatureLocation(start, start + len(gene), strand=1),
                type="CDS",
                qualifiers={"gene": [f"g{k}"]},
            )
        )
    return seq + "C" * 90, features


def test_an_editing_genome_keeps_full_length_cds_with_transl_except() -> None:
    seq, features = _plastome(early_stop_genes=5)
    ends = [int(f.location.end) for f in features]

    refine_cds_features(seq, features)

    assert [int(f.location.end) for f in features] == ends
    edited = features[0]
    assert edited.qualifiers["exception"] == ["RNA editing"]
    assert edited.qualifiers["transl_except"][0].endswith(",aa:Gln)")
    assert "transl_except" not in features[5].qualifiers


def test_a_lone_early_stop_is_a_pseudogene_not_an_edit() -> None:
    seq, features = _plastome(early_stop_genes=1)

    refine_cds_features(seq, features)

    first = features[0]
    assert int(first.location.end) - int(first.location.start) == 3 + 30 * 3 + 3
    assert "transl_except" not in first.qualifiers


def test_retranslation_applies_transl_except_on_both_strands() -> None:
    cds = "ATGGCTTAAGGTTGA"  # M A * G
    genome = "CC" + cds + "CC"
    forward = SeqFeature(
        FeatureLocation(2, 2 + len(cds), strand=1),
        type="CDS",
        qualifiers={"transl_except": ["(pos:9..11,aa:Gln)"]},
    )
    protein = str(forward.extract(Seq(genome)).translate(table=11)).removesuffix("*")
    assert _apply_transl_except(forward, protein) == "MAQG"

    rc = str(Seq(genome).reverse_complement())
    n = len(genome)
    reverse = SeqFeature(
        FeatureLocation(n - 2 - len(cds), n - 2, strand=-1),
        type="CDS",
        qualifiers={"transl_except": [f"(pos:complement({n - 10}..{n - 8}),aa:Gln)"]},
    )
    protein = str(reverse.extract(Seq(rc)).translate(table=11)).removesuffix("*")
    assert _apply_transl_except(reverse, protein) == "MAQG"


def test_mirrored_copy_gets_its_own_edited_positions() -> None:
    # IR mirroring places a copy on the other strand; its transl_except must
    # point at its own codons, or retranslation keeps the stop and drops the
    # translation (which failed the whole Pteridium annotation).
    from organelleverse.annotation.plastome.cds_refine import reannotate_edited_stops

    rng = random.Random(3)
    gene = _gene(rng, 60, stop_at=20)
    genome = "C" * 50 + str(Seq(gene).reverse_complement()) + "C" * 50
    feature = SeqFeature(
        FeatureLocation(50, 50 + len(gene), strand=-1),
        type="CDS",
        qualifiers={"transl_except": ["(pos:1..3,aa:Gln)"], "exception": ["RNA editing"]},
    )

    reannotate_edited_stops(feature, genome)

    assert len(feature.qualifiers["transl_except"]) == 1
    protein = str(feature.extract(Seq(genome)).translate(table=11)).removesuffix("*")
    assert "*" in protein
    assert "*" not in _apply_transl_except(feature, protein)
