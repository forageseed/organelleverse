"""Immutable multi-sample and intermediate analysis state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

from pydantic import Field, ValidationInfo, field_serializer, field_validator

from .artifacts import ArtifactRef
from .base import ContractValidationContext, StrictFrozenModel
from .errors import OrganelleInternalError
from .frozen import FrozenJson, FrozenMap, thaw_json


def _empty_artifacts() -> FrozenMap[ArtifactRef]:
    return FrozenMap.from_items({})


class LineageRecord(StrictFrozenModel[Literal["lineage"]]):
    """One explicit transformation applied to an analysis object."""

    kind: Literal["lineage"] = "lineage"
    parent_object_ids: tuple[str, ...]
    operation_id: str = Field(min_length=3, pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    operation_version: str
    parameters_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class OrganelleData(StrictFrozenModel[Literal["data"]]):
    """Frozen data exchanged between multi-sample analysis operations."""

    schema_version: Literal["organelleverse.data.v1"] = "organelleverse.data.v1"
    kind: Literal["data"] = "data"
    modality: str = Field(min_length=2, pattern=r"^[a-z][a-z0-9_]*$")
    artifacts: FrozenMap[ArtifactRef] = Field(default_factory=_empty_artifacts)
    payload: FrozenMap[FrozenJson] = Field(default_factory=FrozenMap)
    dimensions: FrozenMap[FrozenJson] = Field(default_factory=FrozenMap)
    metadata: FrozenMap[FrozenJson] = Field(default_factory=FrozenMap)
    lineage: tuple[LineageRecord, ...] = ()

    @classmethod
    def _validation_context(cls, context: object | None) -> ContractValidationContext:
        return ContractValidationContext(
            allow_computed_object_id=True,
            allow_serialized_artifacts=True,
        )

    @field_validator("artifacts", mode="before")
    @classmethod
    def freeze_artifacts(cls, value: object, info: ValidationInfo) -> FrozenMap[ArtifactRef]:
        if not isinstance(value, Mapping):
            raise ValueError("artifacts must be an object")
        artifact_values = cast(Mapping[object, object], value)
        if not all(isinstance(key, str) for key in artifact_values):
            raise ValueError("artifacts keys must be strings")
        allow_serialized = (
            isinstance(info.context, ContractValidationContext)
            and info.context.allow_serialized_artifacts
        )
        if not allow_serialized and not all(
            isinstance(item, ArtifactRef) for item in artifact_values.values()
        ):
            raise ValueError("artifacts values must be ArtifactRef instances")
        artifacts: dict[str, ArtifactRef] = {}
        for key, item in artifact_values.items():
            if isinstance(item, ArtifactRef):
                artifact = ArtifactRef.model_validate(item)
            else:
                if not isinstance(item, Mapping):
                    raise ValueError("artifacts values must be ArtifactRef instances")
                artifact_data = dict(cast(Mapping[str, object], item))
                artifact = ArtifactRef.model_validate(artifact_data)
            artifacts[str(key)] = artifact
        return FrozenMap.from_items(artifacts)

    @field_validator("payload", "dimensions", "metadata", mode="before")
    @classmethod
    def freeze_objects(cls, value: object) -> FrozenMap[FrozenJson]:
        try:
            return FrozenMap.from_json({} if value is None else value)
        except OrganelleInternalError as error:
            raise ValueError(str(error)) from error

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: FrozenMap[FrozenJson]) -> FrozenMap[FrozenJson]:
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in value.values()
        ):
            raise ValueError("dimensions values must be non-negative integers")
        return value

    @field_serializer("artifacts")
    def serialize_artifacts(self, value: FrozenMap[ArtifactRef]) -> dict[str, Any]:
        return {key: item.model_dump(mode="json") for key, item in value.items()}

    @field_serializer("lineage")
    def serialize_lineage(self, value: tuple[LineageRecord, ...]) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json", exclude={"object_id"}) for item in value]

    @field_serializer("payload", "dimensions", "metadata")
    def serialize_objects(self, value: FrozenMap[FrozenJson]) -> object:
        return thaw_json(value)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "modality": self.modality,
            "artifacts": {key: item.object_id for key, item in self.artifacts.items()},
            "payload": thaw_json(self.payload),
            "dimensions": thaw_json(self.dimensions),
            "metadata": thaw_json(self.metadata),
            "lineage": [
                item.model_dump(mode="json", exclude={"object_id"}) for item in self.lineage
            ],
        }
