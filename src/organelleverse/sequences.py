"""Sequence sets read from FASTA: the input of the alignment, diversity and selection steps.

``read`` states nothing about what the sequences are. Whether they are
aligned is a property of the records (``is_aligned``), and the step that needs an
alignment checks it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._bio import read_fasta
from .core.errors import OrganelleInputError


@dataclass(frozen=True)
class Sequences:
    """The records of one FASTA file, in file order."""

    path: Path
    records: tuple[tuple[str, str], ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.records)

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(len(sequence) for _, sequence in self.records)

    @property
    def is_aligned(self) -> bool:
        """Every record has the same length, as the columns of an alignment do."""
        return len(set(self.lengths)) == 1

    def __len__(self) -> int:
        return len(self.records)


def read(path: str | Path) -> Sequences:
    """Read a FASTA file of genes, proteins or an alignment into :class:`Sequences`."""
    candidate = Path(path)
    try:
        records = read_fasta(candidate)
    except FileNotFoundError:
        raise OrganelleInputError(
            code="input.missing_artifact",
            message=f"Artifact does not exist: {candidate.name}",
            details={"basename": candidate.name},
        ) from None
    except ValueError as error:
        raise OrganelleInputError(
            code="input.invalid_sequences",
            message=str(error),
            details={"basename": candidate.name},
        ) from error
    return Sequences(path=candidate, records=tuple(records))


def as_records(source: str | Path | Sequences) -> list[tuple[str, str]]:
    """The ``(name, sequence)`` records of a read result or a FASTA path."""
    if isinstance(source, Sequences):
        return list(source.records)
    return read_fasta(Path(source))


def as_path(source: Any) -> Any:
    """The file behind a read result, for code that takes a path; anything else passes through.

    Several images give the list of their files.
    """
    paths = getattr(source, "paths", None)
    if paths is not None:
        return paths[0] if len(paths) == 1 else list(paths)
    return getattr(source, "path", source)
