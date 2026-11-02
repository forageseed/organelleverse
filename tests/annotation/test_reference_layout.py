"""Mitochondrial reference layout: two reference sets per gene must not collide.

Seven genes carry two different protein reference sets: a small,
length-consistent set for tblastn boundary refinement (``pcg_boundary/``) and a
broader set for reference-length statistics (``pcg/``, lower-case names). They
once sat side by side in ``pcg/`` differing only in letter case; a snapshot made
on a case-insensitive filesystem then deleted the boundary set, and boundary
refinement silently skipped those genes.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

import organelleverse
from organelleverse.annotation.mitochondrion.db import DBManager
from organelleverse.annotation.mitochondrion.fragment_filter import reference_protein_lengths

BOUNDARY_ONLY_GENES = ("ccmB", "ccmC", "ccmFC", "ccmFN", "matR", "mttB", "nad4L")


def test_no_source_files_differ_only_by_letter_case() -> None:
    """Windows and macOS checkouts can hold only one file of such a pair."""
    root = Path(organelleverse.__file__).resolve().parent
    by_folded: dict[str, list[str]] = defaultdict(list)
    for path in root.rglob("*"):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        by_folded[relative.casefold()].append(relative)
    collisions = {key: names for key, names in by_folded.items() if len(names) > 1}
    assert collisions == {}


@pytest.mark.parametrize("gene", BOUNDARY_ONLY_GENES)
def test_boundary_reference_comes_from_the_boundary_set(gene: str) -> None:
    database = DBManager()
    reference = database.boundary_reference(gene)
    assert reference == database.boundary_ref_dir / f"{gene}.Protein.fasta"
    assert reference.is_file()


def test_other_genes_fall_back_to_the_pcg_set() -> None:
    database = DBManager()
    assert database.boundary_reference("cox1") == database.blast_ref_dir / "cox1.Protein.fasta"
    assert database.boundary_reference("cox1").is_file()


def test_length_statistics_still_use_the_broader_pcg_set() -> None:
    # ccmFC: the pcg/ (length) set has a median of 446 aa, the boundary set 440 aa.
    lengths = reference_protein_lengths(DBManager())
    assert lengths["ccmfc"] == 446
    assert all(gene.lower() in lengths for gene in BOUNDARY_ONLY_GENES)
