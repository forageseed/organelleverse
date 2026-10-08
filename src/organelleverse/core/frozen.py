"""Recursively immutable JSON-compatible values."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar, cast, overload

from .base import StrictFrozenModel
from .errors import OrganelleInternalError

JsonScalar: TypeAlias = None | bool | int | float | str


ValueT = TypeVar("ValueT", covariant=True)
if TYPE_CHECKING:
    from .artifacts import ArtifactRef


class FrozenMap(Mapping[str, ValueT], Generic[ValueT]):
    """Small deterministic immutable mapping used in public contracts."""

    __slots__ = ("_items",)
    _items: tuple[tuple[str, ValueT], ...]

    def __init__(self: FrozenMap[FrozenJson], values: Mapping[str, object] | None = None) -> None:
        frozen = FrozenMap.from_json({} if values is None else values)
        object.__setattr__(self, "_items", frozen._items)

    @overload
    @classmethod
    def from_items(cls, items: Mapping[str, ArtifactRef]) -> FrozenMap[ArtifactRef]: ...

    @overload
    @classmethod
    def from_items(cls, items: Mapping[str, FrozenJson]) -> FrozenMap[FrozenJson]: ...

    @classmethod
    def from_items(
        cls, items: Mapping[str, object]
    ) -> FrozenMap[FrozenJson] | FrozenMap[ArtifactRef]:
        """Copy a map after validating that its contents are immutable."""
        from .artifacts import ArtifactRef

        values = tuple(items.values())
        if all(type(value) is ArtifactRef for value in values):
            return cast(
                FrozenMap[ArtifactRef],
                FrozenMap.__from_validated_items(
                    tuple((str(key), cast(ArtifactRef, value)) for key, value in items.items())
                ),
            )
        if any(isinstance(value, StrictFrozenModel) for value in values):
            raise OrganelleInternalError(
                code="contract.non_frozen_value",
                message="FrozenMap contract-model values must be exact ArtifactRef instances",
            )
        return cast(
            FrozenMap[FrozenJson],
            FrozenMap.__from_validated_items(
                tuple((str(key), freeze_json(value)) for key, value in items.items())
            ),
        )

    @classmethod
    def from_json(cls, value: object) -> FrozenMap[FrozenJson]:
        """Freeze a JSON object, rejecting every non-object JSON value."""
        frozen = freeze_json(value)
        if not isinstance(frozen, FrozenMap):
            raise OrganelleInternalError(
                code="contract.non_object_json_value",
                message="FrozenMap requires a JSON object",
            )
        return frozen

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __getitem__(self, key: str) -> ValueT:
        for item_key, item_value in self._items:
            if item_key == key:
                return item_value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return f"FrozenMap({dict(self._items)!r})"

    @staticmethod
    def __from_validated_items(items: tuple[tuple[str, object], ...]) -> FrozenMap[object]:
        """Build a map after this module has fully validated every item."""
        frozen = cast(FrozenMap[object], object.__new__(FrozenMap))
        object.__setattr__(frozen, "_items", tuple(sorted(items)))
        return frozen

    @staticmethod
    def freeze_mapping(
        value: Mapping[object, object], ancestors: set[int]
    ) -> FrozenMap[FrozenJson]:
        _guard_container(value, ancestors)
        try:
            if not all(isinstance(key, str) for key in value):
                raise OrganelleInternalError(
                    code="contract.non_string_key",
                    message="JSON object keys must be strings",
                )
            return cast(
                FrozenMap[FrozenJson],
                FrozenMap.__from_validated_items(
                    tuple((str(key), _freeze_json(item, ancestors)) for key, item in value.items())
                ),
            )
        finally:
            ancestors.remove(id(value))


FrozenJson: TypeAlias = JsonScalar | tuple["FrozenJson", ...] | FrozenMap["FrozenJson"]


def freeze_json(value: object) -> FrozenJson:
    """Validate and recursively freeze one JSON-compatible value."""
    return _freeze_json(value, set())


def _freeze_json(value: object, ancestors: set[int]) -> FrozenJson:
    if value is None:
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OrganelleInternalError(
                code="contract.non_finite_number",
                message="JSON numbers must be finite",
            )
        return float(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, Mapping):
        return FrozenMap.freeze_mapping(cast(Mapping[object, object], value), ancestors)
    if isinstance(value, (list, tuple)):
        return _freeze_sequence(cast(list[object] | tuple[object, ...], value), ancestors)
    raise OrganelleInternalError(
        code="contract.non_json_value",
        message=f"{type(value).__name__} is not JSON-compatible",
    )


def _freeze_sequence(
    value: list[object] | tuple[object, ...], ancestors: set[int]
) -> tuple[FrozenJson, ...]:
    _guard_container(value, ancestors)
    try:
        return tuple(_freeze_json(item, ancestors) for item in value)
    finally:
        ancestors.remove(id(value))


def _guard_container(value: object, ancestors: set[int]) -> None:
    if id(value) in ancestors:
        raise OrganelleInternalError(
            code="contract.cyclic_json_value",
            message="Cyclic values are not JSON-compatible",
        )
    ancestors.add(id(value))


def thaw_json(value: FrozenJson) -> object:
    """Convert a frozen JSON value back to built-in JSON containers."""
    if isinstance(value, FrozenMap):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value
