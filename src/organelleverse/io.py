"""Canonical input readers and the human-facing unified dispatcher."""

from __future__ import annotations

from .io_genome import read_assembly_genome as read_assembly
from .io_genome import read_fasta_genome as read_fasta
from .io_genome import read_genbank_genome as read_genbank
from .io_reads import read_reads
from .read_facade import read

__all__ = ["read", "read_assembly", "read_fasta", "read_genbank", "read_reads"]
