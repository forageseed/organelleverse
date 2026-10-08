"""Execution provenance for immutable results."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator, model_validator

from .base import StrictFrozenModel
from .errors import OrganelleInternalError
from .frozen import FrozenJson, FrozenMap, freeze_json, thaw_json


class PathTranslationRecord(StrictFrozenModel[Literal["path_translation"]]):
    """One cross-context path translation recorded for auditability."""

    kind: Literal["path_translation"] = "path_translation"
    parameter: str
    original_value: str
    source_context: str
    target_context: str
    translated_value: str


class InlineMaterializationRecord(StrictFrozenModel[Literal["inline_materialization"]]):
    """One inline parameter materialized to a file for worker execution."""

    kind: Literal["inline_materialization"] = "inline_materialization"
    parameter: str
    format: str
    sha256: str
    size_bytes: int
    materialized_path: str = ""


class ResultProvenance(StrictFrozenModel[Literal["provenance"]]):
    """Immutable record of the backend, inputs, and environment used for an operation."""

    kind: Literal["provenance"] = "provenance"
    run_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    operation_id: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    operation_version: str
    package_version: str
    git_commit: str
    input_object_ids: tuple[str, ...] = ()
    input_artifact_hashes: tuple[str, ...] = ()
    parameters_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    callable_locator: str | None = Field(default=None, pattern=r"^[a-zA-Z_][\w.]*:[a-zA-Z_]\w*$")
    requested_backend: str = ""
    actual_backend: str = ""
    attempted_backends: tuple[str, ...] = ()
    software_versions: FrozenMap[FrozenJson] = Field(default_factory=FrozenMap)
    model_hashes: FrozenMap[FrozenJson] = Field(default_factory=FrozenMap)
    database_hashes: FrozenMap[FrozenJson] = Field(default_factory=FrozenMap)
    argv: tuple[str, ...] = ()
    random_seed: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    run_manifest_id: str | None = None
    upstream_run_manifest_ids: tuple[str, ...] = ()
    path_translations: tuple[PathTranslationRecord, ...] = ()
    inline_materializations: tuple[InlineMaterializationRecord, ...] = ()

    @field_validator("software_versions", "model_hashes", "database_hashes", mode="before")
    @classmethod
    def freeze_objects(cls, value: object) -> FrozenMap[FrozenJson]:
        try:
            frozen = freeze_json({} if value is None else value)
        except OrganelleInternalError as error:
            raise ValueError(str(error)) from error
        if not isinstance(frozen, FrozenMap):
            raise ValueError("provenance mapping field must be a JSON object")
        return frozen

    @field_serializer("software_versions", "model_hashes", "database_hashes")
    def serialize_objects(self, value: FrozenMap[FrozenJson]) -> object:
        return thaw_json(value)

    @field_validator("upstream_run_manifest_ids")
    @classmethod
    def validate_upstream_run_manifest_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        pattern = re.compile(r"^[a-z][a-z0-9_-]*:sha256:[0-9a-f]{64}$")
        if any(pattern.fullmatch(value) is None for value in values):
            raise ValueError("upstream run manifests must be stable content IDs")
        if len(set(values)) != len(values):
            raise ValueError("upstream run manifest IDs must be unique")
        return values

    @model_validator(mode="after")
    def validate_consistency(self) -> ResultProvenance:
        if any(not backend for backend in self.attempted_backends):
            raise ValueError("attempted_backends entries must be non-empty")
        if self.requested_backend and self.requested_backend not in self.attempted_backends:
            raise ValueError("requested_backend must be included in attempted_backends")
        if self.actual_backend and (
            not self.attempted_backends or self.attempted_backends[-1] != self.actual_backend
        ):
            raise ValueError("actual_backend must be the final attempted backend")
        if self.run_manifest_id in self.upstream_run_manifest_ids:
            raise ValueError("current run_manifest_id cannot appear in upstream run manifests")
        if self.started_at is not None and self.finished_at is not None:
            started_is_aware = (
                self.started_at.tzinfo is not None and self.started_at.utcoffset() is not None
            )
            finished_is_aware = (
                self.finished_at.tzinfo is not None and self.finished_at.utcoffset() is not None
            )
            if started_is_aware != finished_is_aware:
                raise ValueError("started_at and finished_at must have matching timezone awareness")
            if self.finished_at < self.started_at:
                raise ValueError("finished_at cannot be earlier than started_at")
        return self

    def _identity_payload(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "operation_id": self.operation_id,
            "operation_version": self.operation_version,
            "package_version": self.package_version,
            "git_commit": self.git_commit,
            "input_object_ids": self.input_object_ids,
            "input_artifact_hashes": self.input_artifact_hashes,
            "parameters_hash": self.parameters_hash,
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
            "attempted_backends": self.attempted_backends,
            "software_versions": thaw_json(self.software_versions),
            "model_hashes": thaw_json(self.model_hashes),
            "database_hashes": thaw_json(self.database_hashes),
            "argv": self.argv,
            "random_seed": self.random_seed,
        }
        if self.upstream_run_manifest_ids:
            payload["upstream_run_manifest_ids"] = self.upstream_run_manifest_ids
        if self.callable_locator is not None:
            payload["callable_locator"] = self.callable_locator
        if self.path_translations:
            payload["path_translations"] = tuple(
                translation.model_dump(mode="json") for translation in self.path_translations
            )
        if self.inline_materializations:
            payload["inline_materializations"] = tuple(
                materialization.model_dump(mode="json")
                for materialization in self.inline_materializations
            )
        return payload
