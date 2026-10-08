"""Tests for initiators and terminators created by C-to-U editing.

The asymmetry these rest on is the whole reason the search can run without an
editing model: editing creates terminators and never removes them, so the DNA's
own stops are trustworthy and the candidates for a gained one are exactly the
three codons a single C-to-U turns into a stop. The first two tests pin that
premise itself rather than the code, because if it stopped holding the approach
would be wrong while every other test still passed.
"""

from __future__ import annotations

from organelleverse.annotation.mitochondrion.editing_gains import (
    STOP_CODONS,
    STOP_GAIN_CODONS,
    find_start_gain,
    find_stop_gain,
    scan_gains,
)


def test_every_stop_gain_candidate_really_is_one_edit_from_a_stop() -> None:
    """CAA/CAG/CGA each become a terminator by changing exactly one C to T."""
    for dna, edited in STOP_GAIN_CODONS.items():
        assert edited in STOP_CODONS
        diffs = [(a, b) for a, b in zip(dna, edited) if a != b]
        assert len(diffs) == 1, f"{dna}->{edited} is not a single substitution"
        assert diffs[0] == ("C", "T"), f"{dna}->{edited} is not C-to-U"


def test_no_terminator_can_be_edited_away() -> None:
    """The premise the whole approach rests on: stops contain no editable C."""
    for stop in STOP_CODONS:
        assert "C" not in stop, f"{stop} contains an editable C"


def test_stop_gain_found_at_the_first_in_frame_candidate() -> None:
    """A gained terminator truncates there, however much frame follows."""
    cds = "ATG" + "AAA" * 3 + "CAA" + "GGG" * 10
    gain = find_stop_gain(cds, cds_start=1000)

    assert gain is not None
    assert gain.dna_codon == "CAA"
    assert gain.edited_codon == "TAA"
    assert gain.position == 1000 + 3 * 4


def test_a_cds_that_already_terminates_is_left_alone() -> None:
    """Editing cannot remove a stop, so a real one ends the search."""
    assert find_stop_gain("ATG" + "AAA" * 3 + "TAA", cds_start=1) is None


def test_start_gain_takes_the_nearest_upstream_candidate() -> None:
    """Reaching past the nearest initiator invents N-terminal residues."""
    upstream = "ACG" + "AAA" * 5 + "ACG"      # candidates 6 and 1 codons back
    gain = find_start_gain(upstream, cds_start=500)

    assert gain is not None
    assert gain.offset_codons == -1
    assert gain.position == 497
    assert gain.edited_codon == "ATG"


def test_an_in_frame_stop_bounds_the_upstream_search() -> None:
    """An initiator beyond a terminator cannot belong to this reading frame."""
    assert find_start_gain("ACG" + "TAA" + "AAA" * 2, cds_start=500) is None


def test_scan_reports_both_gains_without_applying_them() -> None:
    """Reporting, not editing: moving a boundary also needs the alignment."""
    gains = scan_gains(
        cds="ATG" + "AAA" * 2 + "CGA" + "GGG" * 5,
        cds_start=1000,
        upstream="AAA" * 3 + "ACG",
    )

    assert [g.kind for g in gains] == ["start", "stop"]
    assert all(g.description for g in gains)
