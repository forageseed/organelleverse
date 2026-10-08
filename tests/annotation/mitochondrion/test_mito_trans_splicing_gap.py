"""Regression test: trans-spliced genes use max_exon_gap=None (no gap limit).

The fragmented-exon assembly path compared ``gap <= max_exon_gap`` without
handling the documented ``None`` sentinel, so any truly trans-spliced gene
(nad1/nad2/nad5/nad4/nad7) that entered that path — as they do on divergent
taxa such as Pinus, where per-exon BLASTn coverage is low — crashed with
``TypeError: '<=' not supported between instances of 'int' and 'NoneType'``.
"""

from __future__ import annotations

from organelleverse.annotation.mitochondrion.models.gene import Strand
from organelleverse.annotation.mitochondrion.models.genome import GenomeSequence
from organelleverse.annotation.mitochondrion.trans_splicing import merge_exons_to_gene


def test_merge_exons_handles_none_max_exon_gap():
    genome = GenomeSequence(seqid="t", sequence="ATGC" * 500)
    # A truly trans-spliced gene: no gap limit between exons.
    config = {"exons": 1, "max_span": 1_000_000, "min_exon_bp": 15, "max_exon_gap": None}
    # One expected exon with a fragmented best hit (coverage 100/300 < 0.85) plus a
    # second same-strand hit -> enters the fragmented-exon assembly branch, which
    # evaluated the gap against max_exon_gap.
    exon_hits = {
        1: [
            (100, 200, Strand.PLUS, 90.0, 100, 300),
            (260, 410, Strand.PLUS, 88.0, 150, 300),
        ]
    }
    # Must not raise (previously crashed comparing an int gap to None).
    result = merge_exons_to_gene("nad5", exon_hits, config, genome)
    assert result is None or hasattr(result, "gene_name")
