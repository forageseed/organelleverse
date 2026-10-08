"""Immutable v0.1 organelle genome contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .artifacts import ArtifactRef
from .base import StrictFrozenModel
from .data import LineageRecord

OrganelleType = Literal["mitochondrion", "plastid"]


class OrganelleMetadata(StrictFrozenModel[Literal["metadata"]]):
    """Scientific metadata associated with one organelle genome."""

    kind: Literal["metadata"] = "metadata"
    species: str = ""
    accession: str = ""
    genetic_code: int = Field(default=1, ge=1, le=33)
    assembly_type: str = ""
    plastid_type: str = ""
    source: str = ""


class OrganelleGenome(StrictFrozenModel[Literal["genome"]]):
    """A validated sequence and/or annotation input for one organelle."""

    schema_version: Literal["organelleverse.genome.v1"] = "organelleverse.genome.v1"
    kind: Literal["genome"] = "genome"
    organelle: OrganelleType
    sequence: ArtifactRef | None = None
    annotation: ArtifactRef | None = None
    metadata: OrganelleMetadata = Field(default_factory=OrganelleMetadata)
    lineage: tuple[LineageRecord, ...] = ()
    source_manifests: tuple[ArtifactRef, ...] = ()

    @field_validator("source_manifests")
    @classmethod
    def validate_unique_source_manifests(
        cls, manifests: tuple[ArtifactRef, ...]
    ) -> tuple[ArtifactRef, ...]:
        hashes = tuple(manifest.sha256 for manifest in manifests)
        if len(set(hashes)) != len(hashes):
            raise ValueError("source manifests must be unique by content SHA256")
        return manifests

    @model_validator(mode="after")
    def require_scientific_input(self) -> OrganelleGenome:
        if self.sequence is None and self.annotation is None:
            raise ValueError("OrganelleGenome requires sequence or annotation")
        return self

    def _identity_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "organelle": self.organelle,
            "sequence": self.sequence.object_id if self.sequence is not None else None,
            "annotation": self.annotation.object_id if self.annotation is not None else None,
            "metadata": self.metadata.model_dump(mode="json", exclude={"object_id"}),
        }
        if self.lineage:
            payload["lineage"] = [
                record.model_dump(mode="json", exclude={"object_id"}) for record in self.lineage
            ]
        if self.source_manifests:
            payload["source_manifests"] = [manifest.object_id for manifest in self.source_manifests]
        return payload
