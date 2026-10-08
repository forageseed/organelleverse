"""Canonical readers for organelle sequence and annotation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata

ReaderOrganelle = Literal["mito", "chloro", "plastid", "mitochondrion"]


def _organelle_type(value: ReaderOrganelle) -> Literal["mitochondrion", "plastid"]:
    if value in {"mito", "mitochondrion"}:
        return "mitochondrion"
    return "plastid"


def _metadata(values: dict[str, Any]) -> OrganelleMetadata:
    return OrganelleMetadata.model_validate(values)


def read_fasta(
    path: str | Path,
    *,
    organelle: ReaderOrganelle,
    **metadata: Any,
) -> OrganelleGenome:
    """Read a FASTA file into the immutable v1 genome contract."""

    return OrganelleGenome(
        organelle=_organelle_type(organelle),
        sequence=ArtifactRef.from_path(
            path,
            kind="sequence",
            format="fasta",
            media_type="text/plain",
        ),
        metadata=_metadata(metadata),
    )


def read_genbank(
    path: str | Path,
    *,
    organelle: ReaderOrganelle,
    **metadata: Any,
) -> OrganelleGenome:
    """Read a GenBank file into the immutable v1 genome contract."""

    return OrganelleGenome(
        organelle=_organelle_type(organelle),
        annotation=ArtifactRef.from_path(
            path,
            kind="annotation",
            format="genbank",
            media_type="text/plain",
        ),
        metadata=_metadata(metadata),
    )


__all__ = ["ReaderOrganelle", "read_fasta", "read_genbank"]
