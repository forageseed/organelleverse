"""Strict FASTA input shared by assembly-QC components."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Return unique ``(identifier, sequence)`` records from a non-empty FASTA."""

    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    records: list[tuple[str, str]] = []
    parsed = SeqIO.parse(candidate, "fasta")  # pyright: ignore[reportUnknownMemberType]
    for raw_record in parsed:  # pyright: ignore[reportUnknownVariableType]
        record = cast(SeqRecord, raw_record)
        records.append((str(record.id), str(record.seq)))
    if not records:
        raise ValueError(f"FASTA contains no records: {candidate}")
    identifiers = tuple(identifier for identifier, _ in records)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"FASTA identifiers must be unique: {candidate}")
    return records


__all__ = ["read_fasta"]
