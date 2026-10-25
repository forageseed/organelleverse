"""Deprecated: every module reads its own inputs with ``ov.<module>.read``.

``ov.read`` hands a file to a module's ``read``; ``ov.annotation.read`` gives a genome
and ``ov.assembly.read`` gives sequencing reads. This module only keeps the old names
importable.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "organelleverse.io is deprecated; use ov.read(...) or ov.<module>.read "
    "(ov.annotation.read for genomes, ov.assembly.read for sequencing reads)",
    DeprecationWarning,
    stacklevel=2,
)

from .io_genome import read_assembly_genome as read_assembly
from .io_genome import read_fasta_genome as read_fasta
from .io_genome import read_genbank_genome as read_genbank
from .io_reads import read_reads
from .read_facade import read

__all__ = ["read", "read_assembly", "read_fasta", "read_genbank", "read_reads"]
