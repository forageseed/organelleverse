"""Derive strict operation parameter contracts from callable signatures."""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, Flag, StrEnum
from pathlib import Path
from types import NoneType, UnionType
from typing import Annotated, Any, Literal, Self, Union, cast, get_args, get_origin, get_type_hints

from annotated_types import Ge, Gt, Le, Lt, MultipleOf
from pydantic import BaseModel, ConfigDict, ValidationError, create_model
from pydantic.errors import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode
from pydantic_core import PydanticUndefined, SchemaError
from typing_extensions import TypeAliasType

from organelleverse.core.base import ContractValidationContext
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleContractError
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.core.result import OrganelleResult

from .parameters import OperationParameterModel, assert_admissible_parameter_model
from .spec import CoreKind, OperationSpec, OperationStage

CoreType = type[OrganelleGenome] | type[OrganelleData] | type[OrganelleResult]

CORE_TYPES: dict[CoreKind, CoreType] = {
    CoreKind.GENOME: OrganelleGenome,
    CoreKind.DATA: OrganelleData,
    CoreKind.RESULT: OrganelleResult,
}

_NUMERIC_CONSTRAINT_METADATA_TYPES = frozenset({Gt, Ge, Lt, Le, MultipleOf})
_NUMERIC_CONSTRAINT_VALUE_ATTRIBUTES: dict[type[object], str] = {
    Gt: "gt",
    Ge: "ge",
    Lt: "lt",
    Le: "le",
    MultipleOf: "multiple_of",
}
_JSON_SCHEMA_CONTRACT_KEYS = frozenset({"type", "$ref", "enum", "const", "anyOf", "oneOf", "allOf"})
_JSON_PRIMITIVE_TYPES = frozenset({bool, int, float, str, NoneType})
_JSON_SEQUENCE_ORIGINS = frozenset({list, Sequence})
_JSON_MAPPING_ORIGINS = frozenset({dict, Mapping})
_STANDARD_ENUM_MEMBER_CONSTRUCTORS = frozenset({str.__new__, StrEnum.__dict__["__new_member__"]})
_PARAMETER_MODEL_TYPES: frozenset[type[BaseModel]] = frozenset({OrganelleMetadata})
_METADATA_OBJECT_ID_PATTERN = r"^metadata:sha256:[0-9a-f]{64}$"
_METADATA_SCHEMA_FIELD_NAMES = frozenset(OrganelleMetadata.model_fields)
_FIELD_INFO_ATTRIBUTES = {
    "default",
    "default_factory",
    "alias",
    "alias_priority",
    "validation_alias",
    "serialization_alias",
    "title",
    "field_title_generator",
    "description",
    "examples",
    "exclude",
    "exclude_if",
    "discriminator",
    "deprecated",
    "json_schema_extra",
    "frozen",
    "validate_default",
    "repr",
    "init",
    "init_var",
    "kw_only",
}


class StrictParameters(BaseModel):
    """Base model for parameters accepted by public operations."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )

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

    @classmethod
    def model_json_schema(
        cls,
        by_alias: bool = True,
        ref_template: str = "#/$defs/{model}",
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: JsonSchemaMode = "validation",
        *,
        union_format: Literal["any_of", "primitive_type_array"] = "any_of",
    ) -> dict[str, Any]:
        schema = super().model_json_schema(
            by_alias=by_alias,
            ref_template=ref_template,
            schema_generator=schema_generator,
            mode=mode,
            union_format=union_format,
        )
        _add_metadata_object_id_schema(cls, schema, ref_template)
        return schema


@dataclass(frozen=True)
class OperationSignature:
    """Callable metadata and its single generated parameter model."""

    python_signature: inspect.Signature
    input_name: str | None
    input_type: CoreType | None
    output_type: CoreType
    parameter_model: type[BaseModel]
    #: Ruling 3 (Decision 004 approval 2026-08-14): the core input is a
    #: sequence of ``input_type`` objects rather than a single one.
    input_sequence: bool = False


def derive_operation_signature(
    function: Callable[..., object], spec: OperationSpec
) -> OperationSignature:
    """Validate an operation callable and derive its strict parameter model."""
    signature, hints = _inspect_annotations(function)
    parameters = list(signature.parameters.values())
    expected_input = CORE_TYPES.get(spec.input_kind)
    input_name: str | None = None

    if expected_input is not None:
        if not parameters:
            raise _contract("operation first parameter does not match input_kind")
        input_parameter = parameters[0]
        _validate_core_input_kind(input_parameter)
        if input_parameter.default is not inspect.Parameter.empty:
            raise _contract("operation core input must be required")
        input_hint = hints.get(input_parameter.name)
        if spec.input_sequence:
            # Ruling 3 (Decision 004 approval 2026-08-14): the first parameter
            # is a one-argument list/Sequence/tuple of exactly the declared
            # core type; the Agent JSON input is an array of core payloads.
            origin = get_origin(input_hint)
            args = get_args(input_hint)
            element_ok = (
                input_hint is not None
                and origin in (list, Sequence, tuple)
                and len(args) == 1
                and args[0] is expected_input
            )
            if not element_ok:
                raise _contract(
                    "operation first parameter does not match input_kind "
                    "(input_sequence requires list[T]/Sequence[T]/tuple[T, ...] of T)"
                )
        elif input_hint is not expected_input:
            raise _contract("operation first parameter does not match input_kind")
        input_name = parameters.pop(0).name

    if spec.stage is OperationStage.READ and not parameters:
        raise _contract("read operation requires a positional-or-keyword source parameter")

    fields: dict[str, tuple[object, object]] = {}
    represented_parameters: dict[str, inspect.Parameter] = {}
    for index, parameter in enumerate(parameters):
        _validate_parameter_kind(parameter, spec.stage, index)
        annotation = hints.get(parameter.name)
        try:
            supported_annotation = annotation is not None and _is_supported_json_annotation(
                annotation
            )
        except TypeError as error:
            raise _contract(
                f"operation parameter '{parameter.name}' must have a JSON-safe annotation"
            ) from error
        if not supported_annotation:
            raise _contract(
                f"operation parameter '{parameter.name}' must have a JSON-safe annotation"
            )
        _validate_annotated_field_metadata(annotation, parameter.name)
        if isinstance(parameter.default, FieldInfo):
            raise _contract("operation parameter defaults cannot use Pydantic FieldInfo")
        if parameter.default is Ellipsis:
            raise _contract(
                f"operation parameter '{parameter.name}' cannot use Ellipsis as a default"
            )
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (annotation, default)
        represented_parameters[parameter.name] = parameter

    expected_output = CORE_TYPES[spec.output_kind]
    if hints.get("return") is not expected_output:
        raise _contract("operation return annotation does not match output_kind")

    parameter_model = _create_parameter_model(spec, fields)
    _validate_model_fields(parameter_model, fields)
    _validate_requiredness(parameter_model, represented_parameters)
    _validate_defaults(parameter_model)
    return OperationSignature(
        python_signature=signature,
        input_name=input_name,
        input_type=expected_input,
        output_type=expected_output,
        parameter_model=parameter_model,
        input_sequence=spec.input_sequence,
    )


def operation_parameter_schema(
    function: Callable[..., object], spec: OperationSpec
) -> dict[str, object]:
    """Return the JSON Schema generated by the operation's parameter model."""
    return derive_operation_signature(function, spec).parameter_model.model_json_schema()


def _inspect_annotations(
    function: Callable[..., object],
) -> tuple[inspect.Signature, dict[str, object]]:
    try:
        return inspect.signature(function), get_type_hints(function, include_extras=True)
    except (TypeError, ValueError, NameError) as error:
        raise _contract(
            "operation must be an inspectable callable with resolvable annotations"
        ) from error


def _validate_parameter_kind(
    parameter: inspect.Parameter,
    stage: OperationStage,
    index: int,
) -> None:
    if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
        raise _contract("operation variadic positional parameters are forbidden")
    if parameter.kind is inspect.Parameter.VAR_KEYWORD:
        raise _contract("operation variadic keyword parameters are forbidden")
    if stage is OperationStage.READ and index == 0:
        if parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            raise _contract("read source must be positional-or-keyword")
    elif parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
        raise _contract("operation parameters after input must be keyword-only")


def _validate_core_input_kind(parameter: inspect.Parameter) -> None:
    if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
        raise _contract("operation variadic positional parameters are forbidden")
    if parameter.kind is inspect.Parameter.VAR_KEYWORD:
        raise _contract("operation variadic keyword parameters are forbidden")
    if parameter.kind not in {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }:
        raise _contract("operation core input must be positional")


def _create_parameter_model(
    spec: OperationSpec,
    fields: dict[str, tuple[object, object]],
) -> type[BaseModel]:
    dynamic_fields: dict[str, Any] = fields
    try:
        # Pydantic's dynamic field-definition API cannot express generated fields statically.
        parameter_model = cast(
            type[BaseModel],
            create_model(
                f"{spec.operation_id.replace('.', '_')}_Parameters",
                __base__=StrictParameters,
                **dynamic_fields,
            ),
        )
        schema = parameter_model.model_json_schema()
    except (
        PydanticInvalidForJsonSchema,
        PydanticSchemaGenerationError,
        SchemaError,
        TypeError,
        ValueError,
    ) as error:
        raise _contract("operation parameter annotations must have a JSON Schema") from error
    try:
        json.dumps(schema, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise _contract("operation parameter JSON Schema must be strict JSON") from error
    if _has_unconstrained_json_schema(schema):
        raise _contract("operation parameters must be safely representable in Agent JSON")
    return parameter_model


def _has_unconstrained_json_schema(schema: dict[str, object]) -> bool:
    if not schema or not (_JSON_SCHEMA_CONTRACT_KEYS & schema.keys()):
        return True
    if schema.get("additionalProperties") is True:
        return True

    for key in ("items", "contains", "additionalProperties"):
        nested = schema.get(key)
        if isinstance(nested, dict) and _has_unconstrained_json_schema(
            cast(dict[str, object], nested)
        ):
            return True
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        nested = schema.get(key)
        if isinstance(nested, list) and any(
            isinstance(item, dict) and _has_unconstrained_json_schema(cast(dict[str, object], item))
            for item in cast(list[object], nested)
        ):
            return True
    for key in ("properties", "patternProperties", "$defs"):
        nested = schema.get(key)
        if isinstance(nested, dict) and any(
            isinstance(item, dict) and _has_unconstrained_json_schema(cast(dict[str, object], item))
            for item in cast(dict[str, object], nested).values()
        ):
            return True
    return False


def _validate_defaults(parameter_model: type[BaseModel]) -> None:
    try:
        parameter_model.model_validate({})
    except ValidationError as error:
        for detail in error.errors():
            location = detail["loc"]
            if not location or not isinstance(location[0], str):
                continue
            field = parameter_model.model_fields.get(location[0])
            if field is not None and not field.is_required():
                raise _contract(
                    f"operation parameter '{location[0]}' default does not match its annotation"
                ) from error


def _validate_requiredness(
    parameter_model: type[BaseModel],
    parameters: dict[str, inspect.Parameter],
) -> None:
    for name, parameter in parameters.items():
        python_required = parameter.default is inspect.Parameter.empty
        model_required = parameter_model.model_fields[name].is_required()
        if python_required != model_required:
            raise _contract(
                f"operation parameter '{name}' requiredness does not match Python signature"
            )


def _validate_model_fields(
    parameter_model: type[BaseModel], fields: dict[str, tuple[object, object]]
) -> None:
    declared_names = set(fields)
    model_names = set(parameter_model.model_fields)
    if declared_names != model_names:
        parameter_name = next(iter(declared_names.symmetric_difference(model_names)))
        raise _contract(
            f"operation parameter '{parameter_name}' must be represented in the parameter model"
        )
    for name, field in parameter_model.model_fields.items():
        _validate_generated_field(name, field)

    schema_properties = parameter_model.model_json_schema().get("properties")
    if (
        not isinstance(schema_properties, dict)
        or set(cast(dict[str, object], schema_properties)) != model_names
    ):
        raise _contract("operation JSON Schema property names must match Python parameters")

    constructed = parameter_model.model_construct(**dict.fromkeys(model_names))
    if set(constructed.model_dump(warnings=False)) != model_names:
        raise _contract("operation parameter model_dump must include every declared field")


def _validate_annotated_field_metadata(
    annotation: object,
    parameter_name: str,
    active_aliases: set[int] | None = None,
) -> None:
    active: set[int] = set() if active_aliases is None else active_aliases
    if isinstance(annotation, TypeAliasType):
        marker = id(annotation)
        if marker in active:
            raise _contract(
                f"operation parameter '{parameter_name}' contains a recursive type alias"
            )
        active.add(marker)
        try:
            _validate_annotated_field_metadata(annotation.__value__, parameter_name, active)
        finally:
            active.remove(marker)
        return
    if get_origin(annotation) is Annotated:
        base_annotation, *metadata = get_args(annotation)
        for item in metadata:
            if type(item) is FieldInfo:
                _validate_field_info(
                    parameter_name,
                    item,
                    base_annotation=base_annotation,
                    reject_default=True,
                )
            elif type(item) not in _NUMERIC_CONSTRAINT_METADATA_TYPES:
                raise _contract(
                    f"operation parameter '{parameter_name}' has unsupported Annotated metadata"
                )
            else:
                _validate_numeric_constraint(parameter_name, base_annotation, item)
        _validate_annotated_field_metadata(base_annotation, parameter_name, active)
        return
    for argument in get_args(annotation):
        _validate_annotated_field_metadata(argument, parameter_name, active)


def _validate_generated_field(name: str, field: FieldInfo) -> None:
    _validate_field_info(
        name,
        field,
        base_annotation=field.annotation,
        reject_default=False,
    )


def _validate_field_info(
    name: str,
    field: FieldInfo,
    *,
    base_annotation: object,
    reject_default: bool,
) -> None:
    forbidden_values = (
        ("alias", field.alias),
        ("validation_alias", field.validation_alias),
        ("serialization_alias", field.serialization_alias),
        ("exclude", field.exclude),
        ("exclude_if", field.exclude_if),
        ("discriminator", field.discriminator),
        ("json_schema_extra", field.json_schema_extra),
        ("validate_default", field.validate_default),
        ("alias_priority", field.alias_priority),
        ("field_title_generator", field.field_title_generator),
        ("deprecated", field.deprecated),
        ("frozen", field.frozen),
        ("init", field.init),
        ("init_var", field.init_var),
        ("kw_only", field.kw_only),
    )
    for attribute, value in forbidden_values:
        if value is not None:
            raise _contract(
                f"operation parameter '{name}' cannot set Pydantic {attribute} metadata"
            )
    if reject_default and (
        field.default is not PydanticUndefined or field.default_factory is not None
    ):
        raise _contract(
            f"operation parameter '{name}' defaults must come from the Python signature"
        )
    for item in field.metadata:
        if type(item) not in _NUMERIC_CONSTRAINT_METADATA_TYPES:
            raise _contract(f"operation parameter '{name}' has unsupported Pydantic metadata")
        _validate_numeric_constraint(name, base_annotation, item)
    if field.repr is not True:
        raise _contract(f"operation parameter '{name}' cannot set Pydantic repr metadata")
    unsupported_attributes = set(field.asdict()["attributes"]) - _FIELD_INFO_ATTRIBUTES
    if unsupported_attributes:
        attribute = sorted(unsupported_attributes)[0]
        raise _contract(f"operation parameter '{name}' cannot set Pydantic {attribute} metadata")


def _validate_numeric_constraint(
    name: str,
    base_annotation: object,
    metadata: object,
) -> None:
    numeric_base = _numeric_base_type(base_annotation)
    if not any(numeric_base is supported for supported in (int, float)):
        raise _contract(
            f"operation parameter '{name}' numeric constraints require an int or float annotation"
        )
    value_attribute = _NUMERIC_CONSTRAINT_VALUE_ATTRIBUTES[type(metadata)]
    value = cast(object, getattr(metadata, value_attribute))
    if type(value) is int or (type(value) is float and math.isfinite(value)):
        numeric_value = value
    else:
        raise _contract(
            f"operation parameter '{name}' numeric constraint operand must be a finite int or float"
        )
    if type(metadata) is MultipleOf and numeric_value <= 0:
        raise _contract(
            f"operation parameter '{name}' MultipleOf numeric constraint must be greater than zero"
        )


def _numeric_base_type(
    annotation: object,
    active_aliases: set[int] | None = None,
) -> object:
    active: set[int] = set() if active_aliases is None else active_aliases
    if isinstance(annotation, TypeAliasType):
        marker = id(annotation)
        if marker in active:
            return None
        active.add(marker)
        try:
            return _numeric_base_type(annotation.__value__, active)
        finally:
            active.remove(marker)
    if get_origin(annotation) is Annotated:
        arguments = get_args(annotation)
        return _numeric_base_type(arguments[0], active) if arguments else None
    return annotation


def _is_supported_json_annotation(
    annotation: object,
    active_aliases: set[int] | None = None,
    *,
    allow_path: bool = True,
) -> bool:
    # ``allow_path`` is the wall split. ``derive_operation_signature``'s
    # registry admission for the 20 released operations keeps the default
    # ``True`` (several released ops annotate bare ``Path`` parameters, e.g.
    # ``io.read_long_reads.reads`` - that admission is deliberately
    # unchanged). The bundle codec wall (``python_binding``'s JSON codec)
    # passes ``False``, so no Path-carrying annotation can ever take the
    # containment-free JSON codec.
    if any(annotation is primitive for primitive in _JSON_PRIMITIVE_TYPES) or (
        allow_path and annotation is Path
    ):
        return True
    if any(annotation is model_type for model_type in _PARAMETER_MODEL_TYPES):
        return True
    if annotation is Any or annotation is object:
        return False

    if isinstance(annotation, TypeAliasType):
        active: set[int] = set() if active_aliases is None else active_aliases
        marker = id(annotation)
        if marker in active:
            return False
        active.add(marker)
        try:
            return _is_supported_json_annotation(
                annotation.__value__, active, allow_path=allow_path
            )
        finally:
            active.remove(marker)

    origin = get_origin(annotation)
    arguments = get_args(annotation)

    if origin is Annotated:
        return bool(arguments) and _is_supported_json_annotation(
            arguments[0], active_aliases, allow_path=allow_path
        )
    if origin in {Union, UnionType}:
        return bool(arguments) and all(
            _is_supported_json_annotation(argument, active_aliases, allow_path=allow_path)
            for argument in arguments
        )
    if origin is Literal:
        return bool(arguments) and all(_is_supported_literal_value(value) for value in arguments)
    if origin is tuple:
        if not arguments:
            return False
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return _is_supported_json_annotation(
                arguments[0], active_aliases, allow_path=allow_path
            )
        return all(
            _is_supported_json_annotation(argument, active_aliases, allow_path=allow_path)
            for argument in arguments
        )
    if origin in _JSON_SEQUENCE_ORIGINS:
        return len(arguments) == 1 and _is_supported_json_annotation(
            arguments[0], active_aliases, allow_path=allow_path
        )
    if origin in _JSON_MAPPING_ORIGINS:
        return (
            len(arguments) == 2
            and arguments[0] is str
            and _is_supported_json_annotation(arguments[1], active_aliases, allow_path=allow_path)
        )
    if inspect.isclass(annotation) and issubclass(annotation, OperationParameterModel):
        # The base class itself is abstract; a plain BaseModel subclass is not an
        # OperationParameterModel and falls through to ``return False`` below. A concrete
        # subclass is admitted only after a fresh, side-effect-free audit of its fully
        # built state (config, mixins, hooks, decorators, fields). This re-audits on
        # every derivation/registration, so a model mutated and rebuilt after its
        # initial definition cannot regain admissibility by trusting a stale marker.
        if annotation is OperationParameterModel:
            return False
        assert_admissible_parameter_model(annotation)
        return True
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        return _is_supported_enum(annotation)
    return False


def _is_supported_enum(annotation: type[Enum]) -> bool:
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


def _is_supported_literal_value(value: object) -> bool:
    return type(value) is str


def _add_metadata_object_id_schema(
    parameter_model: type[BaseModel],
    schema: dict[str, Any],
    ref_template: str,
) -> None:
    properties = schema.get("properties")
    definitions = schema.get("$defs", {})
    if not isinstance(properties, dict) or not isinstance(definitions, dict):
        return
    property_schemas = cast(dict[str, object], properties)
    definition_schemas = cast(dict[str, object], definitions)
    definition_references = {
        ref_template.format(model=name): definition
        for name, definition in definition_schemas.items()
    }
    visited_nodes: set[int] = set()
    for name, field in parameter_model.model_fields.items():
        if not _annotation_contains_metadata(field.annotation):
            continue
        _augment_reachable_metadata_schema(
            property_schemas.get(name),
            definition_references,
            visited_nodes,
        )


def _annotation_contains_metadata(
    annotation: object,
    active_aliases: set[int] | None = None,
) -> bool:
    if annotation is OrganelleMetadata:
        return True
    active: set[int] = set() if active_aliases is None else active_aliases
    if isinstance(annotation, TypeAliasType):
        marker = id(annotation)
        if marker in active:
            return False
        active.add(marker)
        try:
            return _annotation_contains_metadata(annotation.__value__, active)
        finally:
            active.remove(marker)
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Annotated:
        return bool(arguments) and _annotation_contains_metadata(arguments[0], active)
    if origin is Literal:
        return False
    return any(_annotation_contains_metadata(argument, active) for argument in arguments)


def _augment_reachable_metadata_schema(
    value: object,
    definition_references: dict[str, object],
    visited_nodes: set[int],
) -> None:
    if not isinstance(value, dict):
        return
    node = cast(dict[str, object], value)
    marker = id(node)
    if marker in visited_nodes:
        return
    visited_nodes.add(marker)

    properties = node.get("properties")
    if _is_metadata_contract_schema(node, properties):
        cast(dict[str, object], properties)["object_id"] = {
            "pattern": _METADATA_OBJECT_ID_PATTERN,
            "readOnly": True,
            "title": "Object Id",
            "type": "string",
        }
        return

    reference = node.get("$ref")
    if isinstance(reference, str):
        _augment_reachable_metadata_schema(
            definition_references.get(reference),
            definition_references,
            visited_nodes,
        )
    for key in ("items", "contains", "additionalProperties"):
        _augment_reachable_metadata_schema(
            node.get(key),
            definition_references,
            visited_nodes,
        )
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        branches = node.get(key)
        if isinstance(branches, list):
            for branch in cast(list[object], branches):
                _augment_reachable_metadata_schema(
                    branch,
                    definition_references,
                    visited_nodes,
                )


def _is_metadata_contract_schema(node: dict[str, object], properties: object) -> bool:
    if not isinstance(properties, dict):
        return False
    property_map = cast(dict[str, object], properties)
    property_names = set(property_map)
    kind_schema = property_map.get("kind")
    # This marker is evaluated only while traversing a generated field whose exact
    # annotation recursively contains OrganelleMetadata. It distinguishes the model
    # definition from sibling union/container branches without scanning other fields.
    return (
        node.get("type") == "object"
        and node.get("additionalProperties") is False
        and frozenset(property_names - {"object_id"}) == _METADATA_SCHEMA_FIELD_NAMES
        and frozenset(property_names) <= _METADATA_SCHEMA_FIELD_NAMES | frozenset({"object_id"})
        and isinstance(kind_schema, dict)
        and cast(dict[str, object], kind_schema).get("const") == "metadata"
    )


def _contract(message: str) -> OrganelleContractError:
    return OrganelleContractError(code="contract.invalid_operation_signature", message=message)
