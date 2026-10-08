"""Canonical immutable annotation document with exact compound locations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, field_serializer, field_validator, model_validator

from organelleverse._sequtil import reverse_complement
from organelleverse.core.base import StrictFrozenModel
from organelleverse.core.errors import OrganelleInternalError
from organelleverse.core.frozen import FrozenJson, FrozenMap, thaw_json

LocationOperator = Literal["single", "join", "order"]
PositionStatus = Literal["exact", "before", "after", "unknown"]


class LocationPart(StrictFrozenModel[Literal["location_part"]]):
    """One zero-based, half-open interval in biological feature order."""

    kind: Literal["location_part"] = "location_part"
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    strand: Literal[-1, 1]
    start_status: PositionStatus = "exact"
    end_status: PositionStatus = "exact"

    @model_validator(mode="after")
    def require_non_empty_interval(self) -> Self:
        if self.end <= self.start:
            raise ValueError("location part end must be greater than start")
        return self


class FeatureQualifier(StrictFrozenModel[Literal["qualifier"]]):
    """One lossless, ordered GenBank-style feature qualifier."""

    kind: Literal["qualifier"] = "qualifier"
    name: str = Field(min_length=1)
    values: tuple[str, ...] = ()


class AnnotationFeature(StrictFrozenModel[Literal["annotation_feature"]]):
    """One feature whose ordered location parts preserve splicing semantics."""

    kind: Literal["annotation_feature"] = "annotation_feature"
    feature_id: str = Field(min_length=1)
    seqid: str = Field(min_length=1)
    type: str = Field(min_length=1)
    operator: LocationOperator
    parts: tuple[LocationPart, ...]
    qualifiers: tuple[FeatureQualifier, ...]
    parents: tuple[str, ...]

    @model_validator(mode="after")
    def validate_parts(self) -> Self:
        if not self.parts:
            raise ValueError("annotation feature requires at least one location part")
        if self.operator == "single" and len(self.parts) != 1:
            raise ValueError("single location requires exactly one part")
        return self

    def qualifier_values(self, name: str) -> tuple[str, ...]:
        """Return all values for the first qualifier with ``name``."""

        return next((item.values for item in self.qualifiers if item.name == name), ())

    def extract(self, sequence: str) -> str:
        """Extract and concatenate parts in stored biological order."""

        chunks: list[str] = []
        for part in self.parts:
            chunk = sequence[part.start : part.end]
            chunks.append(reverse_complement(chunk) if part.strand == -1 else chunk)
        return "".join(chunks)

    def genomic_position(self, biological_offset: int) -> int:
        """Map a zero-based offset in the spliced feature to the genome."""

        if biological_offset < 0:
            raise IndexError("biological offset must be non-negative")
        remaining = biological_offset
        for part in self.parts:
            length = part.end - part.start
            if remaining < length:
                return part.start + remaining if part.strand == 1 else part.end - remaining - 1
            remaining -= length
        raise IndexError("biological offset exceeds feature length")


class AnnotationRecord(StrictFrozenModel[Literal["annotation_record"]]):
    """One annotated sequence and all of its canonical features."""

    kind: Literal["annotation_record"] = "annotation_record"
    seqid: str = Field(min_length=1)
    name: str
    description: str
    sequence: str = Field(min_length=1)
    features: tuple[AnnotationFeature, ...]

    @model_validator(mode="after")
    def validate_feature_ids(self) -> Self:
        feature_ids = tuple(feature.feature_id for feature in self.features)
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError("annotation record contains a duplicate feature ID")
        if any(feature.seqid != self.seqid for feature in self.features):
            raise ValueError("annotation feature seqid must match its record")
        return self


class AnnotationDocument(StrictFrozenModel[Literal["annotation_document"]]):
    """Versioned, immutable normalized annotation exchanged by all surfaces."""

    schema_version: Literal["organelleverse.annotation.v1"] = "organelleverse.annotation.v1"
    kind: Literal["annotation_document"] = "annotation_document"
    backend: str = Field(min_length=1)
    requested_stages: tuple[str, ...]
    completed_stages: tuple[str, ...]
    records: tuple[AnnotationRecord, ...]
    source_metadata: FrozenMap[FrozenJson] = Field(default_factory=FrozenMap)

    @field_validator("source_metadata", mode="before")
    @classmethod
    def freeze_source_metadata(cls, value: object) -> FrozenMap[FrozenJson]:
        try:
            return FrozenMap.from_json({} if value is None else value)
        except OrganelleInternalError as error:
            raise ValueError(str(error)) from error

    @field_serializer("source_metadata")
    def serialize_source_metadata(self, value: FrozenMap[FrozenJson]) -> object:
        return thaw_json(value)

    @model_validator(mode="after")
    def validate_record_ids(self) -> Self:
        record_ids = tuple(record.seqid for record in self.records)
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("annotation document contains a duplicate record ID")
        return self
