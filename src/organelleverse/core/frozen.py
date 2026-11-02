"""Recursively immutable JSON-compatible values."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    NoReturn,
    TypeAlias,
    TypeVar,
    cast,
    get_args,
    overload,
)

from .base import StrictFrozenModel
from .errors import OrganelleInternalError

JsonScalar: TypeAlias = None | bool | int | float | str


ValueT = TypeVar("ValueT", covariant=True)
if TYPE_CHECKING:
    from .artifacts import ArtifactRef


def _immutable(self: object, *args: object, **kwargs: object) -> NoReturn:
    raise TypeError(f"{type(self).__name__} is immutable")


class FrozenMap(dict[str, Any], Generic[ValueT]):  # type: ignore[type-var]
    """Small deterministic immutable mapping used in public contracts.

    A ``dict`` subclass with every mutating method disabled, so that
    ``json.dumps(result.metrics)`` (whose C encoder accepts only dicts) writes
    the nested values as JSON instead of falling back to ``default=``; nested
    sequences are tuples, which ``json`` writes as arrays. Keys are kept sorted.
    """

    __slots__ = ()

    def __init__(self: FrozenMap[FrozenJson], values: Mapping[str, object] | None = None) -> None:
        frozen = FrozenMap.from_json({} if values is None else values)
        dict.__init__(self, frozen._items)

    @property
    def _items(self) -> tuple[tuple[str, ValueT], ...]:
        return tuple(dict.items(self))

    __setitem__ = __delitem__ = __ior__ = _immutable
    clear = pop = popitem = setdefault = update = _immutable

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        # pickle/copy would otherwise rebuild a dict subclass through __setitem__.
        return (FrozenMap._rebuild, (self._items,))

    @staticmethod
    def _rebuild(items: tuple[tuple[str, object], ...]) -> FrozenMap[object]:
        return FrozenMap.__from_validated_items(items)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: object, handler: Any) -> Any:
        # As a dict subclass with type parameters pydantic would only check
        # isinstance; keep accepting any mapping and freezing it, as it did
        # when FrozenMap was a plain Mapping.
        from pydantic_core import core_schema

        from .artifacts import ArtifactRef

        arguments = get_args(source)
        values = (
            handler.generate_schema(ArtifactRef)
            if arguments and arguments[0] is ArtifactRef
            else core_schema.any_schema()
        )
        from_mapping = core_schema.no_info_after_validator_function(
            FrozenMap.from_items,
            core_schema.dict_schema(keys_schema=core_schema.str_schema(), values_schema=values),
        )
        return core_schema.union_schema(
            [core_schema.is_instance_schema(FrozenMap), from_mapping],
            serialization=core_schema.plain_serializer_function_ser_schema(dict),
        )

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

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(self._items)

    def __repr__(self) -> str:
        return f"FrozenMap({dict(self._items)!r})"

    @staticmethod
    def __from_validated_items(items: tuple[tuple[str, object], ...]) -> FrozenMap[object]:
        """Build a map after this module has fully validated every item."""
        frozen = cast(FrozenMap[object], dict.__new__(FrozenMap))
        dict.__init__(frozen, sorted(items, key=lambda item: item[0]))
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
