"""Tests for RNA-editing annotation (annotate_rna_edits / edited_protein).

The pass records C-to-U gain sites on a CDS annotation: an ACG start (edited
to AUG) and a CAA/CGA/CAG terminal stop gain. Internal premature stops are
never editing sites — a stop has no editable C, so editing cannot remove one.
"""

from __future__ import annotations

import pytest

from organelleverse.annotation.mitochondrion.editing_gains import (
    annotate_rna_edits,
    edited_protein,
)
from organelleverse.annotation.mitochondrion.models.gene import (
    ExonRecord,
    GeneAnnotation,
    Strand,
)
from organelleverse.annotation.mitochondrion.models.genome import GenomeSequence


def _gene(name: str, start: int, end: int) -> GeneAnnotation:
    return GeneAnnotation(
        gene_name=name,
        gene_type="CDS",
        product="",
        strand=Strand.PLUS,
        exons=[ExonRecord(start=start, end=end, strand=Strand.PLUS)],
    )


def test_edited_protein_applies_c_to_u_at_given_codons() -> None:
    cds = "ATG" + "GGG" * 3 + "CAA"      # terminal CAA -> TAA
    assert edited_protein(cds, [4]) == "MGGG"


def test_edited_protein_start_gain() -> None:
    cds = "ACG" + "GGG" * 3 + "TAA"
    assert edited_protein(cds, [0]) == "MGGG"


def test_start_gain_annotated_for_listed_gene() -> None:
    genome = GenomeSequence(seqid="g", sequence="A" * 10 + "ACG" + "GGG" * 4 + "TAA")
    ann = _gene("atp1", 11, 28)

    class _DB:
        def is_start_gain_gene(self, name: str) -> bool:
            return name == "atp1"

        def is_stop_gain_gene(self, name: str) -> bool:
            return False

    repaired = annotate_rna_edits(ann, genome, _DB())

    assert repaired.rna_edits == [0]
    assert "RNA editing" in repaired.exceptions
    assert any("ACG->ATG" in note for note in repaired.notes)


def test_stop_gain_annotated_for_listed_gene() -> None:
    genome = GenomeSequence(seqid="g", sequence="A" * 10 + "ATG" + "GGG" * 4 + "CAA")
    ann = _gene("ccmC", 11, 28)

    class _DB:
        def is_start_gain_gene(self, name: str) -> bool:
            return False

        def is_stop_gain_gene(self, name: str) -> bool:
            return name == "ccmC"

    repaired = annotate_rna_edits(ann, genome, _DB())

    assert repaired.rna_edits == [5]
    assert "RNA editing" in repaired.exceptions


def test_unlisted_gene_without_gains_is_untouched() -> None:
    genome = GenomeSequence(seqid="g", sequence="A" * 10 + "ATG" + "GGG" * 4 + "TAA")
    ann = _gene("atp9", 11, 28)

    class _DB:
        def is_start_gain_gene(self, name: str) -> bool:
            return False

        def is_stop_gain_gene(self, name: str) -> bool:
            return False

    repaired = annotate_rna_edits(ann, genome, _DB())

    assert repaired is ann
    assert repaired.rna_edits == []


def test_internal_premature_stop_is_not_an_editing_site() -> None:
    # TAA mid-gene: editing cannot remove a stop (stops have no editable C).
    genome = GenomeSequence(seqid="g", sequence="A" * 10 + "ATG" + "GGG" + "TAA" + "GGG" * 2 + "TGA")
    ann = _gene("atp1", 11, 34)

    class _DB:
        def is_start_gain_gene(self, name: str) -> bool:
            return name == "atp1"

        def is_stop_gain_gene(self, name: str) -> bool:
            return False

    repaired = annotate_rna_edits(ann, genome, _DB())

    assert repaired is ann


def test_frame_shifted_gene_is_untouched() -> None:
    genome = GenomeSequence(seqid="g", sequence="A" * 10 + "ACG" + "GGG" * 4 + "TT" + "A" * 6)
    ann = _gene("atp1", 11, 29)

    class _DB:
        def is_start_gain_gene(self, name: str) -> bool:
            return True

        def is_stop_gain_gene(self, name: str) -> bool:
            return False

    repaired = annotate_rna_edits(ann, genome, _DB())

    assert repaired is ann
