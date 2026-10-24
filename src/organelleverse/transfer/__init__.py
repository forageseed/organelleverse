"""Transfer suite: MTPT detection + three-evidence NUMT/NUPT pipeline.

Self-contained MTPT k-mer detection plus the paper-style BLASTN + depth +
long-read evidence pipeline (detect_transfers_blast /
validate_transfers_depth / validate_transfers_longread /
detect_transfers_evidence) for NUMT/NUPT.
"""

from __future__ import annotations

from ..writer import deprecated_alias

from ..inputs import make_reader
from .transfer import (
    detect,
    detect_mtpt,
    detect_transfers_blast,
    validate_transfers_depth,
    validate_transfers_longread,
    detect_transfers_evidence,
    annotate_organelle_genes,
    annotate_nuclear_locus,
    TransferCandidate,
)
from ..writer import write

mtpt = detect_mtpt
save = deprecated_alias(write, "save")

from .transfer_core import compute_transfer

read = make_reader("transfer", "fasta", "bam")

__all__ = [
    "read",
    "detect",
    "mtpt",
    "detect_mtpt",
    "detect_transfers_blast",
    "validate_transfers_depth",
    "validate_transfers_longread",
    "detect_transfers_evidence",
    "annotate_organelle_genes",
    "annotate_nuclear_locus",
    "TransferCandidate",
    "write",
    "save",
]
