"""Spliced reference CDS parts on the wrong strand are repaired from the record's /translation."""

from __future__ import annotations

from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from organelleverse.annotation.data import plastome_reference_dir
from organelleverse.annotation.plastome.references import (
    _protein_similarity,
    _repair_part_strands,
    _spliced_translation,
    load_references,
)

EXON1 = "ATGCCAACTATTAAACAACTTATTAGAAATACAAGACAGCCAATCCGAAATGTCACGAAATCCCCAGCGCTTCGGGGATGCCCCCAACGACGAGGAACATGTACTAGGGTGTAT"
EXON2 = "ACTATAACCCCCAAAAAACCAAACTCTGCTTTACGTAAAGTTGCCAGAGTACGATTAACCTCTGGATTTGAAATTACTGCTTATATACCTGGTATTGGCCATAATTTACAAGAACATTCTGTAGTCTTAGTAAGAGGGGGAAGGGTTAAGGATTTACCCGGTGTGAGATATCACATTGTTCGAGGAACCCTAGATGCTGTCGGAGTAAAGGATCGTCAACAAGGGCGTTCTAAATATGGGGTCAAGAAGCCAAAGTAA"


def _record():
    """Exon 1 on the minus strand far from exons 2 (plus strand), like a trans-spliced rps12."""
    spacer = "A" * 300
    seq = Seq(spacer + str(Seq(EXON1).reverse_complement()) + spacer + EXON2 + spacer)
    e1 = FeatureLocation(300, 300 + len(EXON1), strand=-1)
    s2 = 600 + len(EXON1)
    e2 = FeatureLocation(s2, s2 + len(EXON2), strand=1)
    protein = str(Seq(EXON1 + EXON2).translate(table=11)).rstrip("*")
    return seq, e1, e2, protein


def test_part_on_wrong_strand_is_flipped():
    seq, e1, e2, protein = _record()
    wrong = FeatureLocation(e1.start, e1.end, strand=1)
    feat = SeqFeature(CompoundLocation([wrong, e2]), type="CDS", qualifiers={"translation": [protein]})
    _repair_part_strands(feat, seq)
    assert [p.strand for p in feat.location.parts] == [-1, 1]
    assert str(feat.extract(seq).translate(table=11)).rstrip("*") == protein


def test_correct_and_unverifiable_features_are_untouched():
    seq, e1, e2, protein = _record()
    good = SeqFeature(CompoundLocation([e1, e2]), type="CDS", qualifiers={"translation": [protein]})
    _repair_part_strands(good, seq)
    assert [p.strand for p in good.location.parts] == [-1, 1]
    wrong = FeatureLocation(e1.start, e1.end, strand=1)
    bare = SeqFeature(CompoundLocation([wrong, e2]), type="CDS")
    _repair_part_strands(bare, seq)  # no /translation: nothing to check against
    assert [p.strand for p in bare.location.parts] == [1, 1]


def test_packaged_trans_spliced_cds_have_no_reversed_part_after_loading():
    # Several NCBI records reverse rps12 exon 1 *and* compute /translation from it.
    # After loading, flipping any part of any rps12 copy must not improve its homology.
    refs = load_references(sorted(plastome_reference_dir().glob("*.gb")))
    others = [f.feature.qualifiers["translation"][0] for r in refs for f in r.features
              if f.feature_type == "CDS" and f.gene == "rps12" and f.feature.qualifiers.get("translation")][:12]
    reversed_parts = []
    for r in refs:
        for f in r.features:
            if f.feature_type != "CDS" or f.gene != "rps12" or f.exon_count < 2:
                continue
            parts = list(f.feature.location.parts)
            base = _protein_similarity(_spliced_translation(f.feature.location, r.record.seq), others)
            for k, part in enumerate(parts):
                flipped = CompoundLocation(
                    [*parts[:k], FeatureLocation(part.start, part.end, strand=-(part.strand or 1)), *parts[k + 1:]]
                )
                if _protein_similarity(_spliced_translation(flipped, r.record.seq), others) >= base + 0.15:
                    reversed_parts.append((r.path.stem, k + 1))
    assert reversed_parts == []


def test_copy_without_translation_uses_the_other_copy():
    seq, e1, e2, protein = _record()
    wrong = FeatureLocation(e1.start, e1.end, strand=1)
    bare = SeqFeature(CompoundLocation([wrong, e2]), type="CDS")
    _repair_part_strands(bare, seq, fallback_protein=protein)
    assert [p.strand for p in bare.location.parts] == [-1, 1]
