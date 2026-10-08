"""Strict structured parameter models admitted by operation signatures.

The operation-signature framework (:mod:`organelleverse.operations.signature`) only
accepts JSON-safe primitive annotations as keyword parameters.  A small, controlled
nested-object family is required so that future backends can declare a closed
discriminator-keyed parameter block (for example Oatk backend parameters) inside the
Agent parameter JSON without reintroducing free-form mappings, ``**options``, or
arbitrary :class:`~pydantic.BaseModel` types.

This module defines :class:`OperationParameterModel`, the only structured parameter
base class the signature framework admits.  Admissibility is enforced by
:func:`assert_admissible_parameter_model`, which is:

- **post-build**: it reads the fully constructed model (``model_fields``,
  ``__pydantic_decorators__``, aliases, defaults, and resolved configuration) rather
  than the partial class available during ``__init_subclass__``;
- **idempotent and side-effect-free**: it validates the same facts every time it is
  called, so :mod:`signature` can re-audit a model at operation derivation/registration
  time and detect a model that was mutated and rebuilt after its initial definition;
- **exact**: the effective ``model_config`` must equal the approved configuration, not
  merely contain six matching keys.

The dependency direction is unchanged: ``operations`` does not import ``assembly`` or
any backend.  ``signature`` imports this module; this module imports only
``pydantic``/``annotated_types``/stdlib so it can validate fields without importing
``signature`` (which would create a cycle).
"""

from __future__ import annotations

import typing
from collections.abc import Mapping as AbcMapping
from collections.abc import Sequence as AbcSequence
from enum import Enum, Flag, StrEnum
from pathlib import Path
from types import NoneType, UnionType
from typing import Annotated, Any, Literal, Union, cast, get_args, get_origin

from annotated_types import Ge, Gt, Le, Lt, MultipleOf
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from typing_extensions import TypeAliasType

__all__ = ["OperationParameterModel", "assert_admissible_parameter_model"]

# Numeric annotated-types metadata that does not change a field's input semantics.
_SAFE_NUMERIC_METADATA: frozenset[type] = frozenset({Gt, Ge, Lt, Le, MultipleOf})

_PRIMITIVE_TYPES: frozenset[type] = frozenset({bool, int, float, str, NoneType, type(None)})

# Standard enum member constructors, matching signature._STANDARD_ENUM_MEMBER_CONSTRUCTORS.
# A non-standard constructor can change which inputs map to members, diverging runtime
# from the emitted enum Schema.
_STANDARD_ENUM_MEMBER_CONSTRUCTORS: frozenset[object] = frozenset(
    {str.__new__, StrEnum.__dict__["__new_member__"]}
)

# The single approved effective configuration. The effective model_config of an admitted
# subclass must equal this exactly (no extra or missing keys), so a subclass cannot add
# alias generators, schema extras, or any other behaviour-changing knob.
_APPROVED_CONFIG: AbcMapping[str, object] = {
    "frozen": True,
    "extra": "forbid",
    "strict": True,
    "validate_default": True,
    "allow_inf_nan": False,
    "revalidate_instances": "always",
}

_STRICT_PARAMETER_CONFIG: ConfigDict = ConfigDict(**_APPROVED_CONFIG)  # type: ignore[arg-type]

# Names Pydantic 2.13 places on every completed direct BaseModel subclass. Missing
# entries are harmless across Python versions; any additional entry is fail-closed as
# user-defined state/behaviour until the compatibility boundary is reviewed explicitly.
_GENERATED_MODEL_NAMESPACE: frozenset[str] = frozenset(
    {
        "__abstractmethods__",
        "__annotations__",
        "__class_vars__",
        "__doc__",
        "__firstlineno__",
        "__hash__",
        "__module__",
        "__private_attributes__",
        "__pydantic_complete__",
        "__pydantic_computed_fields__",
        "__pydantic_core_schema__",
        "__pydantic_custom_init__",
        "__pydantic_decorators__",
        "__pydantic_extra_info__",
        "__pydantic_fields__",
        "__pydantic_generic_metadata__",
        "__pydantic_parent_namespace__",
        "__pydantic_post_init__",
        "__pydantic_serializer__",
        "__pydantic_setattr_handlers__",
        "__pydantic_validator__",
        "__signature__",
        "__static_attributes__",
        "_abc_impl",
        "model_config",
    }
)

# Behaviour hooks whose presence on any non-base class in the MRO lets runtime decoding
# diverge from the generated JSON Schema. They must only be inherited from BaseModel.
_FORBIDDEN_BEHAVIOR_DUNDERS = (
    "model_post_init",
    "__init_subclass__",
    "__pydantic_init_subclass__",
    "__get_pydantic_core_schema__",
    "__get_pydantic_json_schema__",
)


def _is_safe_literal(annotation: object) -> bool:
    """Accept only all-string ``Literal`` values (mirrors the signature framework)."""
    return all(type(value) is str for value in get_args(annotation))


def _is_safe_enum(annotation: type[Enum]) -> bool:
    """Accept only plain string-valued enums consistent with the signature framework.

    Mirrors :func:`signature._is_supported_enum` exactly: a custom ``_missing_``
    coercion or a non-standard member constructor would let runtime accept inputs
    omitted from the emitted enum Schema, so such enums are rejected. The standard
    member constructors are ``str.__new__`` and ``StrEnum``'s ``__new_member__``.
    """
    if issubclass(annotation, Flag):
        return False
    if not issubclass(annotation, str):
        return False
    missing_handler = getattr(annotation, "_missing_", None)
    if getattr(missing_handler, "__func__", missing_handler) is not Enum._missing_.__func__:
        return False
    member_constructor = getattr(annotation, "__new_member__", None)
    if (
        member_constructor is not None
        and member_constructor not in _STANDARD_ENUM_MEMBER_CONSTRUCTORS
    ):
        return False
    if getattr(annotation, "__get_pydantic_core_schema__", None) is not None:
        return False
    if getattr(annotation, "__get_pydantic_json_schema__", None) is not None:
        return False
    return bool(annotation.__members__) and all(type(member.value) is str for member in annotation)


def _safe_annotated_metadata(metadata: object) -> bool:
    """Accept only numeric constraints or a FieldInfo without behaviour-changing knobs."""
    if type(metadata) is FieldInfo:
        if _field_info_is_unsafe(metadata):
            return False
        return all(type(item) in _SAFE_NUMERIC_METADATA for item in metadata.metadata)
    return type(metadata) in _SAFE_NUMERIC_METADATA


def _field_info_is_unsafe(field_info: FieldInfo) -> bool:
    """Return whether a field overrides the trusted model-level input contract."""
    forbidden = (
        field_info.alias,
        field_info.validation_alias,
        field_info.serialization_alias,
        field_info.discriminator,
        field_info.json_schema_extra,
        field_info.exclude,
        field_info.exclude_if,
        field_info.validate_default,
        field_info.alias_priority,
        field_info.field_title_generator,
        field_info.deprecated,
        field_info.frozen,
        field_info.init,
        field_info.init_var,
        field_info.kw_only,
    )
    if any(value is not None for value in forbidden):
        return True
    return field_info.default_factory is not None


def _is_safe_annotation(annotation: object, active: set[int]) -> bool:
    """Return True only for closed, schema-faithful JSON field annotations.

    Mirrors the operation-signature allow-list: primitives, ``Path``, string
    ``Literal``, plain string enums, ``Annotated`` with numeric constraints only,
    optional/union, homogeneous sequences, fixed/variadic tuples, and concrete
    :class:`OperationParameterModel` subclasses. Free-form mappings (``dict``/
    ``Mapping``) are rejected so backend parameter objects must use explicit fields.
    """
    if any(annotation is primitive for primitive in _PRIMITIVE_TYPES):
        return True
    if annotation is Path:
        return True
    if annotation is Any or annotation is object:
        return False

    if isinstance(annotation, TypeAliasType):
        marker = id(annotation)
        if marker in active:
            return False
        active.add(marker)
        try:
            return _is_safe_annotation(annotation.__value__, active)
        finally:
            active.remove(marker)

    origin = get_origin(annotation)
    arguments = get_args(annotation)

    if origin is Annotated:
        base, *metadata = arguments
        return (
            bool(metadata)
            and _is_safe_annotation(base, active)
            and all(_safe_annotated_metadata(item) for item in metadata)
        )
    if origin in {Union, UnionType}:
        return bool(arguments) and all(
            _is_safe_annotation(argument, active) for argument in arguments
        )
    if origin is Literal:
        return bool(arguments) and _is_safe_literal(annotation)
    if origin is tuple:
        if not arguments:
            return False
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return _is_safe_annotation(arguments[0], active)
        return all(_is_safe_annotation(argument, active) for argument in arguments)
    if origin in {list, set, frozenset, AbcSequence, typing.Sequence}:
        return len(arguments) == 1 and _is_safe_annotation(arguments[0], active)
    # Free-form string-keyed mappings are deliberately rejected: a backend parameter
    # object must declare explicit fields, not an arbitrary-key channel.
    if origin in {dict, AbcMapping, typing.Mapping}:
        return False

    if isinstance(annotation, type) and issubclass(annotation, OperationParameterModel):
        if annotation is OperationParameterModel:
            return False
        marker = id(annotation)
        if marker in active:
            return False
        active.add(marker)
        try:
            assert_admissible_parameter_model(annotation)
            return True
        finally:
            active.remove(marker)

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return _is_safe_enum(annotation)

    return False


class OperationParameterModel(BaseModel):
    """Base class for structured operation keyword parameters.

    Admissibility is verified post-build (via ``__pydantic_init_subclass__``) and
    re-verified by :func:`assert_admissible_parameter_model` whenever an operation
    signature is derived or registered. The strict configuration cannot be relaxed,
    and schema hooks, validators, serializers, computed fields, aliases, factories,
    behaviour mixins, free-form mappings, and recursive graphs are forbidden so the
    generated JSON Schema is the exact description of accepted runtime input.
    """

    model_config = _STRICT_PARAMETER_CONFIG

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)  # type: ignore[misc]
        if cls is not OperationParameterModel:
            assert_admissible_parameter_model(cls)


def assert_admissible_parameter_model(cls: type[OperationParameterModel]) -> None:
    """Re-assert that ``cls`` is an admissible structured parameter model.

    Idempotent and side-effect-free: safe to call at definition time and again at
    operation derivation/registration. Raises ``TypeError`` on any violation.
    """
    if cls is OperationParameterModel:
        return

    _assert_config_unchanged(cls)
    _assert_direct_parameter_model(cls)
    _assert_no_behavior_hooks(cls)
    _assert_no_decorators(cls)
    _assert_declarative_namespace(cls)
    _assert_fields_safe(cls)


def _assert_config_unchanged(cls: type[OperationParameterModel]) -> None:
    effective = dict(cls.model_config)
    if effective != dict(_APPROVED_CONFIG):
        # Identify the first differing key for a precise, reviewable message.
        keys = sorted(set(effective) | set(_APPROVED_CONFIG))
        difference = next(
            (key for key in keys if effective.get(key) != _APPROVED_CONFIG.get(key)),
            None,
        )
        raise TypeError(
            f"OperationParameterModel subclass {cls.__name__!r} has an inadmissible "
            f"model_config (differs at {difference!r}); it must equal the approved "
            "strict configuration exactly"
        )


def _assert_direct_parameter_model(cls: type[OperationParameterModel]) -> None:
    """Keep the trusted family flat; reuse is expressed through typed composition."""
    if cls.__bases__ != (OperationParameterModel,):
        bases = ", ".join(base.__name__ for base in cls.__bases__)
        raise TypeError(
            f"OperationParameterModel subclass {cls.__name__!r} must directly inherit "
            f"OperationParameterModel; mixins and concrete-model inheritance are "
            f"forbidden (received bases: {bases})"
        )


def _assert_no_behavior_hooks(cls: type[OperationParameterModel]) -> None:
    """Reject behaviour/schema hooks defined outside BaseModel/OperationParameterModel."""
    for name in _FORBIDDEN_BEHAVIOR_DUNDERS:
        for base in cls.__mro__:
            if base in {object}:
                continue
            if name in base.__dict__ and base not in {BaseModel, OperationParameterModel}:
                member = base.__dict__[name]
                raise TypeError(
                    f"OperationParameterModel subclass {cls.__name__!r} inherits a "
                    f"forbidden behaviour hook {name!r} from {base.__name__!r} "
                    f"({type(member).__name__})"
                )


def _assert_no_decorators(cls: type[OperationParameterModel]) -> None:
    """Reject validators, serializers, and computed fields."""
    decorators = cls.__pydantic_decorators__
    for category in (
        "field_validators",
        "model_validators",
        "field_serializers",
        "model_serializers",
        "computed_fields",
    ):
        members = getattr(decorators, category)
        if members:
            name = next(iter(members))
            kind = (
                "validator"
                if "validator" in category
                else "serializer"
                if "serializer" in category
                else "computed field"
            )
            raise TypeError(
                f"OperationParameterModel subclass {cls.__name__!r} cannot define a {kind} {name!r}"
            )


def _assert_declarative_namespace(cls: type[OperationParameterModel]) -> None:
    """Reject class state and executable members outside Pydantic's generated model state."""
    classvars = set(cls.__class_vars__)
    if classvars:
        name = sorted(classvars)[0]
        raise TypeError(
            f"OperationParameterModel subclass {cls.__name__!r} cannot declare "
            f"ClassVar member {name!r}"
        )
    if cls.__pydantic_custom_init__:
        raise TypeError(
            f"OperationParameterModel subclass {cls.__name__!r} cannot define custom "
            "initialization (__init__)"
        )

    unknown = sorted(set(vars(cls)) - _GENERATED_MODEL_NAMESPACE)
    if unknown:
        raise TypeError(
            f"OperationParameterModel subclass {cls.__name__!r} cannot define "
            f"non-field namespace member {unknown[0]!r}"
        )


def _assert_fields_safe(cls: type[OperationParameterModel]) -> None:
    """Validate every fully-built field's annotation, FieldInfo, and default."""
    field_hints = _resolved_field_hints(cls)
    active: set[int] = set()
    for name, field_info in cls.model_fields.items():
        annotation = field_hints.get(name)
        if annotation is None:
            raise TypeError(
                f"OperationParameterModel field {cls.__name__}.{name!r} has no "
                f"resolvable annotation"
            )
        if _field_info_is_unsafe(field_info):
            raise TypeError(
                f"OperationParameterModel field {cls.__name__}.{name!r} uses a "
                f"forbidden Field option (alias/default_factory/json_schema_extra/"
                f"discriminator/validate_default)"
            )
        unsupported_metadata = [
            item for item in field_info.metadata if type(item) not in _SAFE_NUMERIC_METADATA
        ]
        if unsupported_metadata:
            metadata_name = type(unsupported_metadata[0]).__name__
            raise TypeError(
                f"OperationParameterModel field {cls.__name__}.{name!r} uses unsupported "
                f"Field metadata {metadata_name!r}"
            )
        if not _is_safe_annotation(annotation, active):
            raise TypeError(
                f"OperationParameterModel field {cls.__name__}.{name!r} has a non-safe "
                f"JSON annotation: {annotation!r}"
            )
        _assert_default_is_valid(cls, name, field_info)


def _assert_default_is_valid(
    cls: type[OperationParameterModel], name: str, field_info: FieldInfo
) -> None:
    """Ensure a field default round-trips through validation (catches ``int='x'``)."""
    if field_info.default is PydanticUndefined:
        return
    if field_info.default_factory is not None:  # pragma: no cover - rejected earlier
        return
    try:
        cls.model_validate({name: field_info.default})
    except ValidationError as error:
        relevant = [
            detail
            for detail in error.errors(include_url=False)
            if detail["loc"] and detail["loc"][0] == name
        ]
        if not relevant:
            return
        raise TypeError(
            f"OperationParameterModel field {cls.__name__}.{name!r} has an invalid "
            f"default {field_info.default!r}: {relevant[0]['msg']}"
        ) from error


def _resolved_field_hints(cls: type[OperationParameterModel]) -> dict[str, Any]:
    """Resolve this class's field annotations, rejecting forward references."""
    module = sys_module(cls)
    globalns = cast(AbcMapping[str, object], getattr(module, "__dict__", {}) if module else {})
    hints: dict[str, Any] = {}
    for name, annotation in _raw_annotations(cls).items():
        if name not in cls.model_fields:
            continue
        hints[name] = _resolve_annotation(annotation, cls.__name__, name, globalns)
    return hints


def _raw_annotations(cls: type[OperationParameterModel]) -> dict[str, Any]:
    """Collect field annotations across the MRO, nearest-first."""
    merged: dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        for name, annotation in getattr(base, "__annotations__", {}).items():
            merged[name] = annotation
    return merged


def _resolve_annotation(
    annotation: Any,
    class_name: str,
    name: str,
    globalns: AbcMapping[str, object],
) -> Any:
    if isinstance(annotation, str):
        try:
            return eval(annotation, dict(globalns), None)
        except NameError as error:
            raise TypeError(
                f"OperationParameterModel subclass {class_name!r} has an unresolved "
                f"forward reference {name!r}"
            ) from error
    return annotation


def sys_module(cls: type[OperationParameterModel]) -> Any:
    """Return the module object that defines ``cls`` (used for forward-ref resolution)."""
    import sys

    return sys.modules.get(cls.__module__, None)
