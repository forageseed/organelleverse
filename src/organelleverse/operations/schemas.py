"""Pure JSON Schema composition for operation parameters and invocations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import LineageRecord, OrganelleData
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import (
    ErrorDetail,
    Finding,
    OperationSuggestion,
    OrganelleResult,
)

from .spec import CoreKind, OperationSpec

if TYPE_CHECKING:
    from .data_contracts import DataContract
    from .registry import BoundOperation

_CORE_MODELS: dict[CoreKind, type[BaseModel]] = {
    CoreKind.GENOME: OrganelleGenome,
    CoreKind.DATA: OrganelleData,
    CoreKind.RESULT: OrganelleResult,
}

_OBJECT_ID_PATTERNS = {
    "ArtifactRef": r"^[a-z][a-z0-9_]*:sha256:[0-9a-f]{64}$",
    "ErrorDetail": r"^error:sha256:[0-9a-f]{64}$",
    "Finding": r"^finding:sha256:[0-9a-f]{64}$",
    "LineageRecord": r"^lineage:sha256:[0-9a-f]{64}$",
    "OperationSuggestion": r"^operation_suggestion:sha256:[0-9a-f]{64}$",
    "OrganelleData": r"^data:sha256:[0-9a-f]{64}$",
    "OrganelleGenome": r"^genome:sha256:[0-9a-f]{64}$",
    "OrganelleMetadata": r"^metadata:sha256:[0-9a-f]{64}$",
    "OrganelleResult": r"^result:sha256:[0-9a-f]{64}$",
    "ResultProvenance": r"^provenance:sha256:[0-9a-f]{64}$",
}


def make_computed_object_ids_optional(schema: Mapping[str, object]) -> dict[str, object]:
    """Retain computed object IDs while recursively removing their requiredness."""
    rewritten = deepcopy(dict(schema))

    def visit(value: object) -> None:
        if isinstance(value, dict):
            mapping = cast(dict[str, object], value)
            required = mapping.get("required")
            if isinstance(required, list):
                items = cast(list[object], required)
                mapping["required"] = [item for item in items if item != "object_id"]
            for nested in mapping.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in cast(list[object], value):
                visit(nested)

    visit(rewritten)
    return rewritten


def parameter_schema(binding: BoundOperation) -> dict[str, object]:
    """Return the strict serialization Schema for one binding's parameters."""
    return parameter_schema_for_model(binding.signature.parameter_model)


def parameter_schema_for_model(parameter_model: type[BaseModel]) -> dict[str, object]:
    """Produce the one canonical public/frozen schema for a parameter model."""
    generated = parameter_model.model_json_schema(mode="serialization")
    return make_computed_object_ids_optional(cast(dict[str, object], generated))


def parameter_schema_from_frozen(schema: Mapping[str, object]) -> dict[str, object]:
    """Return a defensive copy of a verification-pinned parameter schema."""
    return deepcopy(dict(schema))


def specialize_data_schema(
    *,
    modality: str,
    payload_model: type[BaseModel],
) -> dict[str, object]:
    """Specialize the generic data contract for one strict payload model."""
    payload_generated = payload_model.model_json_schema(mode="serialization")
    data_schema, data_definitions = _namespace_schema(
        _serialized_core_schema(CoreKind.DATA),
        "data",
    )
    payload_schema, payload_definitions = _namespace_schema(
        make_computed_object_ids_optional(cast(dict[str, object], payload_generated)),
        "payload",
    )
    properties = cast(dict[str, object], data_schema["properties"])
    properties["modality"] = {"type": "string", "const": modality}
    properties["payload"] = payload_schema
    definitions = {**data_definitions, **payload_definitions}
    if definitions:
        data_schema["$defs"] = definitions
    return data_schema


def invocation_schema(binding: BoundOperation) -> dict[str, object]:
    """Return the complete strict Agent invocation envelope for one binding."""
    input_schema, input_definitions = _input_schema_parts(binding)
    parameters, parameter_definitions = _namespace_schema(
        parameter_schema(binding),
        "parameters",
    )
    definitions = {**input_definitions, **parameter_definitions}
    operation_id = binding.spec.operation_id
    schema: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"{operation_id} invocation",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "operation_id": {"type": "string", "const": operation_id},
            "input": input_schema,
            "parameters": parameters,
        },
        "required": ["operation_id", "input", "parameters"],
    }
    if definitions:
        schema["$defs"] = definitions
    return schema


def invocation_schema_from_frozen(
    spec: OperationSpec,
    frozen_parameter_schema: Mapping[str, object],
    *,
    data_contracts: tuple[DataContract, ...] = (),
) -> dict[str, object]:
    """Compose an invocation schema without importing a capability implementation."""
    input_schema, input_definitions = _input_schema_parts_from_spec(spec, data_contracts)
    parameters, parameter_definitions = _namespace_schema(
        parameter_schema_from_frozen(frozen_parameter_schema),
        "parameters",
    )
    definitions = {**input_definitions, **parameter_definitions}
    schema: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"{spec.operation_id} invocation",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "operation_id": {"type": "string", "const": spec.operation_id},
            "input": input_schema,
            "parameters": parameters,
        },
        "required": ["operation_id", "input", "parameters"],
    }
    if definitions:
        schema["$defs"] = definitions
    return schema


def _input_schema_parts(binding: BoundOperation) -> tuple[dict[str, object], dict[str, object]]:
    if binding.spec.input_kind is CoreKind.NONE:
        return {"type": "null"}, {}
    if binding.spec.input_kind is CoreKind.DATA and binding.data_contracts:
        specialized: list[dict[str, object]] = []
        definitions: dict[str, object] = {}
        for index, contract in enumerate(binding.data_contracts):
            schema, contract_definitions = _namespace_schema(
                make_computed_object_ids_optional(contract.json_schema()),
                f"input_{index}_{contract.modality}",
            )
            specialized.append(schema)
            definitions.update(contract_definitions)
        if len(specialized) == 1:
            return specialized[0], definitions
        return {"oneOf": specialized}, definitions
    return _namespace_schema(
        _serialized_core_schema(binding.spec.input_kind),
        "input",
    )


def _input_schema_parts_from_spec(
    spec: OperationSpec,
    data_contracts: tuple[DataContract, ...] = (),
) -> tuple[dict[str, object], dict[str, object]]:
    if spec.input_kind is CoreKind.NONE:
        return {"type": "null"}, {}
    if spec.input_kind is CoreKind.DATA and data_contracts:
        specialized: list[dict[str, object]] = []
        definitions: dict[str, object] = {}
        for index, contract in enumerate(data_contracts):
            schema, contract_definitions = _namespace_schema(
                make_computed_object_ids_optional(contract.json_schema()),
                f"input_{index}_{contract.modality}",
            )
            specialized.append(schema)
            definitions.update(contract_definitions)
        if len(specialized) == 1:
            return specialized[0], definitions
        return {"oneOf": specialized}, definitions
    return _namespace_schema(_serialized_core_schema(spec.input_kind), "input")


def _serialized_core_schema(kind: CoreKind) -> dict[str, object]:
    if kind is CoreKind.DATA:
        schema = _serialized_data_schema()
    elif kind is CoreKind.RESULT:
        schema = _serialized_result_schema()
    else:
        generated = _CORE_MODELS[kind].model_json_schema(mode="serialization")
        schema = make_computed_object_ids_optional(cast(dict[str, object], generated))
    _tighten_object_id_schemas(schema)
    return schema


def _serialized_data_schema() -> dict[str, object]:
    generated = OrganelleData.model_json_schema(mode="serialization")
    schema = make_computed_object_ids_optional(cast(dict[str, object], generated))
    definitions = _frozen_json_definitions()
    _add_model_definition(definitions, ArtifactRef)
    _add_model_definition(definitions, LineageRecord)
    properties = cast(dict[str, object], schema["properties"])
    properties.update(
        {
            "artifacts": {
                "type": "object",
                "additionalProperties": {"$ref": "#/$defs/ArtifactRef"},
            },
            "payload": _frozen_json_map_schema(),
            "dimensions": {
                "type": "object",
                "additionalProperties": {"type": "integer", "minimum": 0},
            },
            "metadata": _frozen_json_map_schema(),
            "lineage": {
                "type": "array",
                "items": {"$ref": "#/$defs/LineageRecord"},
            },
        }
    )
    schema["$defs"] = definitions
    return schema


def _serialized_result_schema() -> dict[str, object]:
    generated = OrganelleResult.model_json_schema(mode="serialization")
    schema = make_computed_object_ids_optional(cast(dict[str, object], generated))
    definitions = _frozen_json_definitions()
    for model in (
        ArtifactRef,
        ErrorDetail,
        Finding,
        OperationSuggestion,
        ResultProvenance,
    ):
        _add_model_definition(definitions, model)

    _replace_mapping_fields(definitions["ErrorDetail"], "details", "suggested_action")
    _replace_mapping_fields(definitions["OperationSuggestion"], "parameter_changes")
    _replace_mapping_fields(
        definitions["ResultProvenance"],
        "software_versions",
        "model_hashes",
        "database_hashes",
    )
    properties = cast(dict[str, object], schema["properties"])
    properties.update(
        {
            "metrics": _frozen_json_map_schema(),
            "findings": {
                "type": "array",
                "items": {"$ref": "#/$defs/Finding"},
            },
            "artifacts": {
                "type": "array",
                "items": {"$ref": "#/$defs/ArtifactRef"},
            },
            "provenance": {
                "anyOf": [
                    {"$ref": "#/$defs/ResultProvenance"},
                    {"type": "null"},
                ]
            },
            "errors": {
                "type": "array",
                "items": {"$ref": "#/$defs/ErrorDetail"},
            },
            "suggested_operations": {
                "type": "array",
                "items": {"$ref": "#/$defs/OperationSuggestion"},
            },
        }
    )
    schema["$defs"] = definitions
    return schema


def _frozen_json_definitions() -> dict[str, object]:
    return {
        "FrozenJson": {
            "anyOf": [
                {"type": "null"},
                {"type": "boolean"},
                {"type": "integer"},
                {"type": "number"},
                {"type": "string"},
                {"type": "array", "items": {"$ref": "#/$defs/FrozenJson"}},
                {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/$defs/FrozenJson"},
                },
            ]
        }
    }


def _frozen_json_map_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": {"$ref": "#/$defs/FrozenJson"},
    }


def _add_model_definition(
    definitions: dict[str, object],
    model: type[BaseModel],
) -> None:
    generated = make_computed_object_ids_optional(
        cast(dict[str, object], model.model_json_schema(mode="serialization"))
    )
    nested = generated.pop("$defs", {})
    if isinstance(nested, Mapping):
        definitions.update(cast(Mapping[str, object], nested))
    definitions[model.__name__] = generated


def _replace_mapping_fields(schema: object, *field_names: str) -> None:
    if not isinstance(schema, dict):
        return
    typed_schema = cast(dict[str, object], schema)
    properties = typed_schema.get("properties")
    if not isinstance(properties, dict):
        return
    typed_properties = cast(dict[str, object], properties)
    for field_name in field_names:
        typed_properties[field_name] = _frozen_json_map_schema()


def _tighten_object_id_schemas(value: object) -> None:
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        title = mapping.get("title")
        properties = mapping.get("properties")
        if isinstance(title, str) and isinstance(properties, dict) and "object_id" in properties:
            pattern = _OBJECT_ID_PATTERNS.get(title)
            if pattern is not None:
                cast(dict[str, object], properties)["object_id"] = {
                    "type": "string",
                    "pattern": pattern,
                    "readOnly": True,
                }
        for nested in mapping.values():
            _tighten_object_id_schemas(nested)
    elif isinstance(value, list):
        for nested in cast(list[object], value):
            _tighten_object_id_schemas(nested)


def _namespace_schema(
    schema: Mapping[str, object],
    namespace: str,
) -> tuple[dict[str, object], dict[str, object]]:
    isolated = deepcopy(dict(schema))
    raw = isolated.pop("$defs", {})
    if not isinstance(raw, Mapping):
        return isolated, {}
    raw_definitions = cast(Mapping[str, object], raw)
    safe_namespace = re.sub(r"[^A-Za-z0-9_]", "_", namespace)
    names = {name: f"{safe_namespace}__{name}" for name in raw_definitions}
    _rewrite_definition_refs(isolated, names)
    definitions: dict[str, object] = {}
    for name, definition in raw_definitions.items():
        copied = deepcopy(definition)
        _rewrite_definition_refs(copied, names)
        definitions[names[name]] = copied
    return isolated, definitions


def _rewrite_definition_refs(value: object, names: Mapping[str, str]) -> None:
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        reference = mapping.get("$ref")
        if isinstance(reference, str):
            for old_name, new_name in names.items():
                old_prefix = f"#/$defs/{old_name}"
                if reference == old_prefix or reference.startswith(f"{old_prefix}/"):
                    mapping["$ref"] = f"#/$defs/{new_name}{reference[len(old_prefix) :]}"
                    break
        for nested in mapping.values():
            _rewrite_definition_refs(nested, names)
    elif isinstance(value, list):
        for nested in cast(list[object], value):
            _rewrite_definition_refs(nested, names)
