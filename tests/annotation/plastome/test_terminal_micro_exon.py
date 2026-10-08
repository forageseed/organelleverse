"""A missing short 3' exon (rps12 exon 3) is appended when transfer ended the gene inside the intron."""

from __future__ import annotations

from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from organelleverse.annotation.plastome.micro_exon import recover_terminal_micro_exons

EXON1 = "ATGCCAACTATTAAACAACTTATTAGAAAT"  # 30 nt (trans-spliced, other strand)
EXON2 = "GCAAAAGAAGGTCTGGCAAAAGTTGAAGGCGAAATCGCAGCTAAAGAAGGTCTGGCAAAAGTTGAAGGCGAAATCGCAGCA"  # 81 nt
EXON3 = "GGTAAAGCTGAAGGCGTTCTGCGTTAA"  # 27 nt, ends in TAA
# Intron: GT.. (opens with an in-frame read-through to TGA after 12 nt) ... AT; 480 nt.
INTRON = "GTGCGACTTCAATGA" + "C" * 450 + "TTTTCCAAAAAT"


def _genome():
    pad = "G" * 200
    far = "C" * 3000
    genome = pad + str(Seq(EXON1).reverse_complement()) + far + EXON2 + INTRON + EXON3 + pad
    e1 = (len(pad) + 1, len(pad) + len(EXON1))
    s2 = e1[1] + len(far) + 1
    e2 = (s2, s2 + len(EXON2) - 1)
    s3 = e2[1] + len(INTRON) + 1
    e3 = (s3, s3 + len(EXON3) - 1)
    return genome, e1, e2, e3


def test_terminal_micro_exon_is_recovered():
    genome, e1, e2, e3 = _genome()
    protein = str(Seq(EXON1 + EXON2 + EXON3).translate(table=11)).rstrip("*")
    read_through_end = e2[1] + 15  # transfer ran on to the in-frame TGA inside the intron
    feat = SeqFeature(CompoundLocation([FeatureLocation(e1[0] - 1, e1[1], strand=-1),
                                        FeatureLocation(e2[0] - 1, read_through_end, strand=1)]),
                      type="CDS", qualifiers={"gene": ["rps12"]})
    recover_terminal_micro_exons(genome, [feat], {"rps12": [protein]}, {"rps12": (3, len(EXON2), len(EXON3))})
    got = [(int(p.start) + 1, int(p.end), p.strand) for p in feat.location.parts]
    assert got == [(e1[0], e1[1], -1), (e2[0], e2[1], 1), (e3[0], e3[1], 1)]


def test_complete_gene_is_left_alone():
    genome, e1, e2, e3 = _genome()
    protein = str(Seq(EXON1 + EXON2 + EXON3).translate(table=11)).rstrip("*")
    parts = [FeatureLocation(e1[0] - 1, e1[1], strand=-1), FeatureLocation(e2[0] - 1, e2[1], strand=1),
             FeatureLocation(e3[0] - 1, e3[1], strand=1)]
    feat = SeqFeature(CompoundLocation(parts), type="CDS", qualifiers={"gene": ["rps12"]})
    recover_terminal_micro_exons(genome, [feat], {"rps12": [protein]}, {"rps12": (3, len(EXON2), len(EXON3))})
    assert [(int(p.start) + 1, int(p.end)) for p in feat.location.parts] == [(e1[0], e1[1]), (e2[0], e2[1]), (e3[0], e3[1])]


def test_terminal_map_requires_the_usual_structure():
    from types import SimpleNamespace

    from organelleverse.annotation.plastome.pipeline import _terminal_micro_exon_map

    def cds(ref, gene, lengths):
        feats = [SimpleNamespace(feature_type="CDS", gene=gene, feature_id=f"{ref}:1", reference_name=ref,
                                 sequence="A" * sum(lengths), exon_index=1, exon_count=len(lengths))]
        if len(lengths) > 1:
            feats += [SimpleNamespace(feature_type="CDS_exon", gene=gene, feature_id=f"{ref}:1:exon{i}", reference_name=ref,
                                      sequence="A" * n, exon_index=i, exon_count=len(lengths))
                      for i, n in enumerate(lengths, start=1)]
        return feats

    refs = []
    for i in range(5):
        refs.append(SimpleNamespace(features=tuple(cds(f"r{i}", "rps12", [114, 232, 26]) + cds(f"r{i}", "rpl22", [450]))))
    refs.append(SimpleNamespace(features=tuple(cds("odd", "rpl22", [420, 30]))))  # one record's stray short exon
    assert _terminal_micro_exon_map(refs) == {"rps12": (3, 232, 26)}
