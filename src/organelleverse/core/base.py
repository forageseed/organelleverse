"""Base behavior for immutable v0.1 contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, ValidationInfo, computed_field, model_validator
from typing_extensions import TypeVar

KindT = TypeVar("KindT", bound=str, default=str)


@dataclass
class ContractValidationContext:
    """Private validator capabilities passed through Pydantic validation."""

    allow_computed_object_id: bool = False
    allow_serialized_artifacts: bool = False
    serialized_object_ids: list[tuple[type[object], str | None]] | None = None

    def remember_object_id(self, model_type: type[object], object_id: str | None) -> None:
        if self.serialized_object_ids is None:
            self.serialized_object_ids = []
        self.serialized_object_ids.append((model_type, object_id))

    def pop_object_id(self, model_type: type[object]) -> str | None:
        if not self.serialized_object_ids:
            return None
        expected_type, object_id = self.serialized_object_ids.pop()
        if expected_type is not model_type:
            self.serialized_object_ids.append((expected_type, object_id))
            return None
        return object_id


def _is_mapping(value: object) -> bool:
    return isinstance(value, Mapping)


class StrictFrozenModel(BaseModel, Generic[KindT]):
    """Pydantic base with deterministic identity and validated evolution."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        revalidate_instances="always",
    )
    kind: KindT

    @classmethod
    def _validation_context(cls, context: object | None) -> ContractValidationContext:
        supplied = (
            context
            if isinstance(context, ContractValidationContext)
            else ContractValidationContext()
        )
        supplied.allow_computed_object_id = True
        return supplied

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: Literal["allow", "ignore", "forbid"] | None = None,
        from_attributes: bool | None = None,
        context: object | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        return super().model_validate(
            obj,
            strict=strict,
            extra=extra,
            from_attributes=from_attributes,
            context=cls._validation_context(context),
            by_alias=by_alias,
            by_name=by_name,
        )

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: Literal["allow", "ignore", "forbid"] | None = None,
        context: object | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        return super().model_validate_json(
            json_data,
            strict=strict,
            extra=extra,
            context=cls._validation_context(context),
            by_alias=by_alias,
            by_name=by_name,
        )

    @model_validator(mode="before")
    @classmethod
    def retain_serialized_object_id_for_validation(
        cls, value: object, info: ValidationInfo
    ) -> object:
        if not (
            _is_mapping(value)
            and isinstance(info.context, ContractValidationContext)
            and info.context.allow_computed_object_id
        ):
            return value
        restored = dict(cast(Mapping[str, object], value))
        if "object_id" in restored:
            serialized_object_id = restored.pop("object_id")
            if not isinstance(serialized_object_id, str):
                raise ValueError("serialized object_id must be a string")
        else:
            serialized_object_id = None
        info.context.remember_object_id(cls, serialized_object_id)
        return restored

    @model_validator(mode="after")
    def validate_serialized_object_id(self, info: ValidationInfo) -> Self:
        if not (
            isinstance(info.context, ContractValidationContext)
            and info.context.allow_computed_object_id
        ):
            return self
        serialized_object_id = info.context.pop_object_id(type(self))
        if serialized_object_id is not None and serialized_object_id != self.object_id:
            raise ValueError("serialized object_id does not match computed object_id")
        return self

    def _identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"object_id"}, exclude_none=True)

    @computed_field
    @property
    def object_id(self) -> str:
        payload = json.dumps(
            self._identity_payload(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return f"{self.kind}:sha256:{digest}"

    def evolve(self, **changes: Any) -> Self:
        values = {name: getattr(self, name) for name in type(self).model_fields}
        values.update(changes)
        return type(self).model_validate(values)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        values = self.model_dump(exclude={"object_id"})
        if update is not None:
            values.update(update)
        return type(self).model_validate(values)

    def copy(
        self,
        *,
        include: Any = None,
        exclude: Any = None,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        values = self.model_dump(include=include, exclude=exclude)
        values.pop("object_id", None)
        if update is not None:
            values.update(update)
        return type(self).model_validate(values)
