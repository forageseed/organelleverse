"""Group II domain V-VI 3' splice site prediction and its use in splice choice."""

from __future__ import annotations

from organelleverse.annotation.data import plastome_reference_dir
from organelleverse.annotation.plastome import group_ii
from organelleverse.annotation.plastome.splice_flanks import choose_candidate


def test_reference_offset_takes_the_mode_of_loaded_references(monkeypatch):
    table = {"a.gb": {"rpl2": {"1": 1}}, "b.gb": {"rpl2": {"1": 1}}, "c.gb": {"rpl2": {"1": 2}},
             "d.gb": {"rpl2": {"1": 2}}, "e.gb": {"rpl2": {"1": 2}}}
    monkeypatch.setattr(group_ii, "_offsets", lambda: table)
    assert group_ii.reference_offset("rpl2", 1, {"a.gb", "b.gb", "c.gb", "d.gb", "e.gb"}) == 2
    assert group_ii.reference_offset("rpl2", 1, {"a.gb", "b.gb", "c.gb"}) == 1
    assert group_ii.reference_offset("rpl2", 1, {"a.gb", "c.gb"}) == 1  # tie: the value nearer 0
    assert group_ii.reference_offset("rpl2", 1, {"a.gb"}) is None  # one reference cannot calibrate
    assert group_ii.reference_offset("rpl2", 1, {"x.gb", "y.gb"}) is None  # removed references never count


def test_equivalent_placements_of_an_intron_end():
    # exon ...AAG | GTGCGTTTTTTTAG | GTT...: the intron's last G equals the base before it, slide left once.
    seq = "AAG" + "GTGCGTTTTTTTAG" + "GTTCCC"
    ends = group_ii.equivalent_intron_ends(seq, 3, 18)
    assert 17 in ends and 16 in ends


def test_structure_breaks_vote_ties_but_not_a_clear_majority():
    cands = [(100.0, [(1, 9), (20, 29)], "ATGAAACCC" + "GGGTTTTAA", [9], 0, 0),
             (98.0, [(1, 12), (23, 29)], "ATGAAACCCAAA" + "TTTTAA", [12], 0, 1)]
    neutral = [[("ATGAAACCC", "GGGTTT")], [("GAAACCCAAA", "TTTTAA")]]  # one reference each: a 1:1 tie
    assert choose_candidate(cands, neutral, tolerance=0.05) == [(1, 12), (23, 29)]
    against = [[("TGAAACCC"[-10:], "GGGTTTTAA")]] * 4  # every reference prefers candidate 0
    assert choose_candidate(cands, against, tolerance=0.05) == [(1, 9), (20, 29)]
    assert choose_candidate(cands, neutral, tolerance=0.0) == [(1, 9), (20, 29)]  # outside the tolerance


def test_domain_v_vi_locates_a_reference_3prime_splice_site():
    from Bio import SeqIO

    path = next(p for p in sorted(plastome_reference_dir().glob("*.gb")) if p.name.startswith("Nicotiana_tabacum"))
    rec = SeqIO.read(path, "genbank")
    feat = next(f for f in rec.features if f.type == "CDS" and f.qualifiers.get("gene") == ["rpl16"])
    a, b = list(feat.location.parts)[:2]
    if a.strand == -1:
        intron = rec.seq[int(b.end):int(a.start)].reverse_complement()
    else:
        intron = rec.seq[int(a.end):int(b.start)]
    exon2 = b.extract(rec.seq)
    tail = str(intron[-group_ii.TAIL:]).upper()
    end = group_ii.alignment_end(tail + str(exon2[:group_ii.FLANK]).upper())
    assert end is not None and abs(end - len(tail)) <= 3


def test_reliable_structure_overrules_a_vote_majority():
    # Every reference prefers candidate 0, but candidate 1 matches a reliable domain V-VI prediction.
    cands = [(100.0, [(1, 9), (20, 29)], "ATGAAACCC" + "GGGTTTTAA", [9], 0, (0, 0)),
             (98.0, [(1, 12), (23, 29)], "ATGAAACCCAAA" + "TTTTAA", [12], 0, (1, 0))]
    against = [[("TGAAACCC"[-10:], "GGGTTTTAA")]] * 4
    assert choose_candidate(cands, against, tolerance=0.05) == [(1, 12), (23, 29)]


def test_calibration_reports_reliability(monkeypatch):
    table = {f"r{i}.gb": {"rpl16": {"1": 0}} for i in range(6)}
    table["odd.gb"] = {"rpl16": {"1": 3}}
    table.update({f"s{i}.gb": {"ycf3": {"1": (0 if i % 2 else -2)}} for i in range(6)})
    monkeypatch.setattr(group_ii, "_offsets", lambda: table)
    files = set(table)
    assert group_ii.reference_calibration("rpl16", 1, files) == (0, True)  # 6 of 7 agree
    assert group_ii.reference_calibration("ycf3", 1, files) == (0, False)  # split 3:3
    assert group_ii.reference_calibration("rpl16", 1, {"r0.gb", "r1.gb"}) == (0, False)  # too few references
