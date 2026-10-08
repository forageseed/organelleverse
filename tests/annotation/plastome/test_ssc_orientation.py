"""Unit tests for SSC orientation measurement and the circular segment flip."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature

from organelleverse.annotation.plastome import ssc_orientation
from organelleverse.annotation.plastome.models import (
    BlastHit,
    ReferenceFeature,
    ReferenceQuery,
    ReferenceRecord,
)
from organelleverse.annotation.plastome.ssc_orientation import (
    _quadripartite,
    evaluate_ssc_correction,
    flip_ssc_segment,
    panel_convention,
    reference_relative_sign,
)

_LSC, _IR, _SSC = 2000, 600, 1200


@pytest.fixture(autouse=True)
def _small_ir_threshold(monkeypatch):
    """The synthetic circles use 600 bp repeats, far below a real IR."""
    monkeypatch.setattr(ssc_orientation, "_MIN_IR_LENGTH", 100)


def _random_dna(length: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


def _synthetic_genome(seed: int) -> str:
    """LSC + IRb + SSC + IRa, the textbook quadripartite circle."""
    lsc = _random_dna(_LSC, seed)
    ir = _random_dna(_IR, seed + 1)
    ssc = _random_dna(_SSC, seed + 2)
    return lsc + ir + ssc + str(Seq(ir).reverse_complement())


def _reference(name: str, genome: str, ssc_strand: int) -> ReferenceRecord:
    """A record whose LSC CDS sit on + and SSC CDS on ``ssc_strand``."""
    features: list[ReferenceFeature] = []

    def add(offset: int, length: int, strand: int, gene: str):
        feature = SeqFeature(
            FeatureLocation(offset, offset + length, strand=strand),
            type="CDS",
            qualifiers={"gene": [gene]},
        )
        features.append(
            ReferenceFeature(
                feature_id=f"{name}:{gene}",
                feature=feature,
                sequence="",
                gene=gene,
                feature_type="CDS",
                reference_name=name,
            )
        )

    for i in range(4):
        add(200 + i * 400, 120, 1, f"lsc{i}")
    for i in range(3):
        add(_LSC + _IR + 200 + i * 300, 100, ssc_strand, f"ssc{i}")
    from Bio.SeqRecord import SeqRecord

    return ReferenceRecord(
        path=Path(f"/synthetic/{name}.gb"),
        record=SeqRecord(Seq(genome), id=name),
        features=tuple(features),
    )


def test_flip_segment_preserves_flanks_and_reverses_interior():
    genome = _synthetic_genome(1)
    start, length = _LSC + _IR, _SSC
    flipped = flip_ssc_segment(genome, start, length)
    interior = str(Seq(genome[start : start + length]).reverse_complement())
    assert flipped[start : start + length] == interior
    assert flipped[:start] == genome[:start]
    assert flipped[start + length :] == genome[start + length :]


def test_flip_segment_wrapping_the_origin():
    genome = _synthetic_genome(2)
    # Rotate the circle so the SSC straddles the linear origin, then flip.
    rotated = genome[_LSC + _IR :] + genome[: _LSC + _IR]
    ssc_length = _SSC
    flipped = flip_ssc_segment(rotated, 0, ssc_length)
    interior = str(Seq(rotated[:ssc_length]).reverse_complement())
    assert flipped[:ssc_length] == interior
    assert flipped[ssc_length:] == rotated[ssc_length:]
    # Flipping commutes with rotation: the wrapped flip equals the unrotated
    # circle's flip rotated by the same offset.
    unrotated_flip = flip_ssc_segment(genome, _LSC + _IR, _SSC)
    assert flipped == unrotated_flip[_LSC + _IR :] + unrotated_flip[: _LSC + _IR]


def test_flip_segment_is_an_involution():
    genome = _synthetic_genome(3)
    once = flip_ssc_segment(genome, _LSC + _IR, _SSC)
    assert flip_ssc_segment(once, _LSC + _IR, _SSC) == genome


def test_flip_segment_rejects_degenerate_lengths():
    with pytest.raises(ValueError):
        flip_ssc_segment(_synthetic_genome(4), 0, 0)
    with pytest.raises(ValueError):
        flip_ssc_segment(_synthetic_genome(4), 0, len(_synthetic_genome(4)))


def test_quadripartite_finds_the_ssc_arc():
    genome = _synthetic_genome(5)
    regions = _quadripartite(genome)
    assert regions is not None
    start, length = regions["SSC"]
    # The shared IR detector reports boundaries within its seed stride.
    assert start == pytest.approx(_LSC + _IR, abs=12)
    assert length == pytest.approx(_SSC, abs=12)


def test_a_short_repeat_pair_is_not_a_quadripartite_genome(monkeypatch):
    """Conifer plastomes keep a ~0.5 kb IR remnant; the arc between is no SSC."""
    monkeypatch.setattr(ssc_orientation, "_MIN_IR_LENGTH", _IR + 100)
    assert _quadripartite(_synthetic_genome(5)) is None


def test_reference_relative_sign_reads_the_annotation():
    genome = _synthetic_genome(6)
    assert reference_relative_sign(_reference("plus.gb", genome, 1)) == 1
    assert reference_relative_sign(_reference("minus.gb", genome, -1)) == -1


def test_panel_convention_requires_a_majority():
    references = tuple(
        _reference(f"panel{i}", _synthetic_genome(10 + i), 1 if i < 5 else -1) for i in range(6)
    )
    convention, signs = panel_convention(references)
    assert convention == 1
    assert signs == {f"panel{i}.gb": 1 if i < 5 else -1 for i in range(6)}
    # Two references cannot establish a panel convention.
    assert panel_convention(references[:2])[0] is None


def _query_for(feature: ReferenceFeature) -> ReferenceQuery:
    return ReferenceQuery(
        query_id=f"q-{feature.gene}",
        group="cds",
        sequence="A" * 30,
        reference_feature=feature,
    )


def _hit(strand: int, offset: int = 0) -> BlastHit:
    return BlastHit(
        query_id="",
        pident=90.0,
        qcov=0.9,
        start=offset,
        end=offset + 100,
        strand=strand,
        bitscore=500.0,
        evalue=1e-40,
        align_length=100,
    )


def test_evaluate_decides_against_flipping_a_conventional_input():
    genome = _synthetic_genome(8)
    primary = _reference("plus.gb", genome, 1)
    # The panel needs enough members; replicate the primary layout.
    panel = tuple(_reference(f"ref{i}.gb", _synthetic_genome(20 + i), 1) for i in range(6))
    queries = tuple(_query_for(f) for f in primary.features)
    hits = {f"q-{f.gene}": _hit(f.feature.location.strand or 1, 10) for f in primary.features}
    decision = evaluate_ssc_correction(genome, primary, panel, hits, queries)
    assert not decision.flip
    assert decision.reason == "already_conventional"


def test_evaluate_flags_and_locates_a_flipped_ssc():
    genome = _synthetic_genome(9)
    primary = _reference("plus.gb", genome, 1)
    panel = tuple(_reference(f"ref{i}.gb", _synthetic_genome(30 + i), 1) for i in range(6))
    queries = tuple(_query_for(f) for f in primary.features)
    agreeing = {f"q-{f.gene}": _hit(f.feature.location.strand or 1, 10) for f in primary.features}
    opposed = {
        f"q-{f.gene}": _hit(
            -(f.feature.location.strand or 1)
            if f.gene.startswith("ssc")
            else (f.feature.location.strand or 1),
            10,
        )
        for f in primary.features
    }
    decision = evaluate_ssc_correction(genome, primary, panel, opposed, queries)
    assert decision.flip
    assert decision.ssc_start == pytest.approx(_LSC + _IR, abs=12)
    assert decision.ssc_length == pytest.approx(_SSC, abs=12)
    # After the flip the same genes present the reference way again, so a
    # remeasured transfer on the corrected genome stays conventional.
    flipped_genome = flip_ssc_segment(genome, decision.ssc_start, decision.ssc_length)
    redone = evaluate_ssc_correction(flipped_genome, primary, panel, agreeing, queries)
    assert not redone.flip
    assert redone.reason == "already_conventional"
    # And an input whose transfer agrees everywhere needs no correction.
    straight = evaluate_ssc_correction(genome, primary, panel, agreeing, queries)
    assert not straight.flip


def test_evaluate_fails_closed_without_a_panel_convention():
    genome = _synthetic_genome(10)
    primary = _reference("plus.gb", genome, 1)
    queries = tuple(_query_for(f) for f in primary.features)
    hits = {f"q-{f.gene}": _hit(f.feature.location.strand or 1, 10) for f in primary.features}
    decision = evaluate_ssc_correction(genome, primary, (primary,), hits, queries)
    assert not decision.flip
    assert decision.reason == "reference_panel_has_no_convention"
