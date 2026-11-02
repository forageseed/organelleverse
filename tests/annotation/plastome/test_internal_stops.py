"""Internal stops of a spliced CDS are explained: edited sites in an editing genome, a note elsewhere; coordinates stay."""

from __future__ import annotations

from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from organelleverse.annotation.plastome.cds_refine import _UNEXPLAINED_STOP_NOTE, mark_internal_stops

# exon 1 ATG GCT TAA GCT (12 nt: an in-frame TAA at codon 3), intron, exon 2 GCT GCT TGA
EXON1, INTRON, EXON2 = "ATGGCTTAAGCT", "GT" + "A" * 30 + "AG", "GCTGCTTGA"
GENOME = "CCCCCC" + EXON1 + INTRON + EXON2 + "CCCCCC"


def _cds():
    a = 6
    b = a + len(EXON1) + len(INTRON)
    loc = CompoundLocation([FeatureLocation(a, a + len(EXON1), strand=1), FeatureLocation(b, b + len(EXON2), strand=1)])
    return SeqFeature(loc, type="CDS", qualifiers={"gene": ["clpP"]})


def test_editing_genome_gets_transl_except_and_keeps_its_location():
    f = _cds()
    before = [(int(p.start), int(p.end)) for p in f.location.parts]
    assert mark_internal_stops([f], GENOME, editing_genome=True) == 1
    assert f.qualifiers["exception"] == ["RNA editing"]
    assert len(f.qualifiers["transl_except"]) == 1 and "aa:" in f.qualifiers["transl_except"][0]
    assert [(int(p.start), int(p.end)) for p in f.location.parts] == before


def test_other_genomes_get_a_note_only():
    f = _cds()
    assert mark_internal_stops([f], GENOME, editing_genome=False) == 1
    assert "transl_except" not in f.qualifiers and _UNEXPLAINED_STOP_NOTE in f.qualifiers["note"]
    assert mark_internal_stops([f], GENOME, editing_genome=False) == 0  # idempotent


def test_clean_cds_and_single_exon_cds_are_left_alone():
    clean = "ATGGCT" + INTRON + "GCTTGA"
    g = "CCCCCC" + clean + "CCCCCC"
    loc = CompoundLocation([FeatureLocation(6, 12, strand=1), FeatureLocation(12 + len(INTRON), 12 + len(INTRON) + 6, strand=1)])
    f = SeqFeature(loc, type="CDS", qualifiers={"gene": ["ok"]})
    single = SeqFeature(FeatureLocation(6, 6 + len(EXON1), strand=1), type="CDS", qualifiers={"gene": ["x"]})
    assert mark_internal_stops([f, single], g, editing_genome=True) == 0
    assert "note" not in f.qualifiers and "transl_except" not in single.qualifiers


def test_acg_start_becomes_an_edited_met_and_only_qualifiers_change():
    from organelleverse.annotation.plastome.cds_refine import mark_edited_starts

    plus = SeqFeature(FeatureLocation(6, 6 + 9, strand=1), type="CDS", qualifiers={"gene": ["ndhD"]})
    g = "CCCCCC" + "ACGGCTTGA" + "CCCCCC"
    minus_seq = "CCCCCC" + "TCAAGCCGT" + "CCCCCC"  # reverse complement of ACGGCTTGA
    minus = SeqFeature(FeatureLocation(6, 15, strand=-1), type="CDS", qualifiers={"gene": ["psbL"]})
    atg = SeqFeature(FeatureLocation(6, 15, strand=1), type="CDS", qualifiers={"gene": ["x"]})
    assert mark_edited_starts([plus], g) == 1
    assert plus.qualifiers["transl_except"] == ["(pos:7..9,aa:Met)"] and plus.qualifiers["exception"] == ["RNA editing"]
    assert mark_edited_starts([plus], g) == 0  # idempotent
    assert mark_edited_starts([minus], minus_seq) == 1
    assert minus.qualifiers["transl_except"] == ["(pos:complement(13..15),aa:Met)"]
    assert mark_edited_starts([atg], "CCCCCC" + "ATGGCTTGA" + "CCCCCC") == 0
    assert (int(plus.location.start), int(plus.location.end)) == (6, 15)
