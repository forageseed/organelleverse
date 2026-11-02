from dataclasses import replace

from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from organelleverse.annotation.plastome import blast
from organelleverse.annotation.plastome.exon_models import ExonModel
from organelleverse.annotation.plastome.models import BlastTools, ReferenceFeature, ReferenceQuery
from organelleverse.annotation.plastome.pipeline import _promote_complete_exon_models


def candidates():
    first = FeatureLocation(0, 3, strand=1)
    left = CompoundLocation(
        [first, FeatureLocation(6, 9, strand=1), FeatureLocation(12, 15, strand=1)]
    )
    right = CompoundLocation(
        [first, FeatureLocation(24, 27, strand=-1), FeatureLocation(18, 21, strand=-1)]
    )
    template = SeqFeature(
        left, type="CDS", qualifiers={"gene": ["example"], "trans_splicing": [""]}
    )
    a = ExonModel("example", "a:1", template, left, "ATGAAATAA", "MK*", 0)
    return [a, replace(a, reference_product="b:1", location=right)]


def current(protein):
    return SeqFeature(
        FeatureLocation(0, 3, strand=1),
        type="CDS",
        qualifiers={"gene": ["example"], "translation": [protein]},
    )


def test_strict_improvement_retains_shared_first_copy_locations():
    features = [current("M")]
    assert _promote_complete_exon_models(features, candidates(), {"example": ["MK"]}, {}) == []
    assert len(features) == 2
    assert features[0].location.parts[0] == features[1].location.parts[0]
    assert all("trans_splicing" in f.qualifiers for f in features)


def test_equal_score_preserves_original_boundaries():
    old = current("MK")
    features = [old]
    _promote_complete_exon_models(features, candidates(), {"example": ["MK"]}, {})
    assert features == [old]


def test_different_mature_dna_ties_are_unresolved():
    a, b = candidates()
    old = current("M")
    features = [old]
    warnings = _promote_complete_exon_models(
        features, [a, replace(b, mature_dna="ATGAAGTAA")], {"example": ["MK"]}, {}
    )
    assert features == [old] and "mature-DNA" in warnings[0]


def test_overlapping_boundary_alternatives_are_not_extra_copies():
    a = candidates()[0]
    alternative = CompoundLocation(
        [a.location.parts[0], FeatureLocation(7, 10, strand=1), FeatureLocation(13, 16, strand=1)]
    )
    old = current("M")
    features = [old]
    warnings = _promote_complete_exon_models(
        features, [a, replace(a, location=alternative)], {"example": ["MK"]}, {}
    )
    assert features == [old] and "overlapping" in warnings[0]


def test_untranslated_current_product_is_not_a_lower_score():
    old = current("")
    old.qualifiers.pop("translation")
    old.qualifiers["exception"] = ["RNA editing"]
    features = [old]
    warnings = _promote_complete_exon_models(features, candidates(), {"example": ["MK"]}, {})
    assert features == [old]
    assert "unscored current" in warnings[0]


def test_untranslated_copy_is_not_deleted_when_other_copy_is_scored():
    scored = current("M")
    unscored = current("")
    unscored.qualifiers.pop("translation")
    features = [scored, unscored]
    warnings = _promote_complete_exon_models(features, candidates(), {"example": ["MK"]}, {})
    assert features == [scored, unscored]
    assert "unscored current" in warnings[0]


def test_absent_current_product_can_be_added():
    features = []
    assert _promote_complete_exon_models(features, candidates(), {"example": ["MK"]}, {}) == []
    assert len(features) == 2


def test_ncbi_complete_search_requests_all_hsps(tmp_path, monkeypatch):
    commands = []

    def command(cmd, **kwargs):
        commands.append(cmd)
        if "-out" in cmd:
            from pathlib import Path

            Path(cmd[cmd.index("-out") + 1]).write_text("")

    monkeypatch.setattr(blast, "run_command", command)
    template = candidates()[0].template
    feature = ReferenceFeature("a:1:exon1", template, "ATG", "example", "CDS_exon", "a", 1, 3)
    query = ReferenceQuery("q", "reference4", "M", feature)
    blast.run_plastome_blasts(
        "ATGAAA",
        (query,),
        work_dir=tmp_path,
        tools=BlastTools("blastn", "makeblastdb", "tblastn"),
        min_identity=0.4,
        qcoverage_range=(0.5, 1.1),
        retain_exon_hsps=True,
    )
    protein_command = next(cmd for cmd in commands if cmd[0] == "tblastn")
    assert "-max_hsps" not in protein_command
