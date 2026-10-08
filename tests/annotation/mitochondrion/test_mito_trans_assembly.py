"""Reference-protein-validated trans-splice assembly (nad1/nad2/nad5).

The end-to-end nad5 recovery test needs a real seed-plant mito GenBank, located
via the ``ORGANELLEVERSE_MITO_GENBANK`` environment variable so no developer path
is baked into the test; it skips when the variable is unset or blastn is absent.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from organelleverse.annotation.mitochondrion.db import DBManager
from organelleverse.annotation.mitochondrion.models.genome import GenomeSequence
from organelleverse.annotation.mitochondrion.trans_assembly import assemble_exons_by_protein

_NAD5_TRUTH = [
    (192149, 192378, 1),
    (193223, 194438, 1),
    (222166, 222187, -1),
    (172723, 173117, 1),
    (174209, 174355, 1),
]


def test_declines_without_reference(tmp_path):
    # No protein reference in an empty dir -> returns None (caller falls back).
    genome = GenomeSequence(seqid="g", sequence="ACGT" * 200)
    assert assemble_exons_by_protein("nad5", genome, {1: []}, 5, tmp_path) is None


@pytest.mark.slow
def test_assembles_nad5_cross_strand_exons_exactly():
    gb = os.environ.get("ORGANELLEVERSE_MITO_GENBANK")
    if not gb or not Path(gb).exists() or shutil.which("blastn") is None:
        pytest.skip("set ORGANELLEVERSE_MITO_GENBANK to a Nicotiana mito GenBank")

    from Bio import SeqIO

    from organelleverse.annotation.mitochondrion.trans_splicing import (
        find_exon_reference_file,
        search_exons_blastn,
    )

    rec = next(SeqIO.parse(gb, "genbank"))
    genome = GenomeSequence(seqid="g", sequence=str(rec.seq).upper())
    dbm = DBManager(None)
    exon_ref = find_exon_reference_file("nad5", dbm)
    assert exon_ref is not None
    exon_hits = search_exons_blastn("nad5", genome, exon_ref, shutil.which("blastn"))
    ann = assemble_exons_by_protein("nad5", genome, exon_hits, 5, dbm.blast_ref_dir)
    assert ann is not None
    got = [(e.start, e.end, int(e.strand)) for e in ann.exons]
    assert got == _NAD5_TRUTH  # all 5 exons exact, incl. 22bp -strand exon3
