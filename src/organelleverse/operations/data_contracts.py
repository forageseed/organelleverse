"""Runtime validation and Schema discovery contracts for data modalities."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field

from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleContractError, OrganelleInputError

from .schemas import specialize_data_schema

DataValidator: TypeAlias = Callable[[OrganelleData], object]
SchemaProvider: TypeAlias = Callable[[], dict[str, object]]

_MODALITY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_TYPES = frozenset({"array", "boolean", "integer", "null", "number", "object", "string"})
_SCHEMA_VALUE_KEYS = frozenset(
    {
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_SCHEMA_MAPPING_KEYS = frozenset({"$defs", "dependentSchemas", "patternProperties", "properties"})
_SCHEMA_ARRAY_KEYS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_NONNEGATIVE_INTEGER_KEYS = frozenset(
    {
        "maxContains",
        "maxItems",
        "maxLength",
        "maxProperties",
        "minContains",
        "minItems",
        "minLength",
        "minProperties",
    }
)
_NUMBER_KEYS = frozenset(
    {"exclusiveMaximum", "exclusiveMinimum", "maximum", "minimum", "multipleOf"}
)
_STRING_KEYS = frozenset(
    {
        "$anchor",
        "$comment",
        "$dynamicAnchor",
        "$dynamicRef",
        "$id",
        "$ref",
        "contentEncoding",
        "contentMediaType",
        "description",
        "format",
        "title",
    }
)
_BOOLEAN_KEYS = frozenset({"deprecated", "readOnly", "uniqueItems", "writeOnly"})


@dataclass(frozen=True)
class DataContract:
    """One modality's runtime validator and discoverable serialization Schema."""

    modality: str
    schema_version: str
    validator: DataValidator
    schema_provider: SchemaProvider

    def __post_init__(self) -> None:
        if _MODALITY_PATTERN.fullmatch(self.modality) is None:
            raise OrganelleContractError(
                code="contract.invalid_data_modality",
                message=f"invalid data modality: {self.modality!r}",
            )
        if not self.schema_version.strip():
            raise OrganelleContractError(
                code="contract.invalid_data_schema_version",
                message="data contract schema version must not be blank",
            )

    def validate(self, data: OrganelleData) -> None:
        """Apply runtime scientific validation for this modality."""
        self.validator(data)

    def json_schema(self) -> dict[str, object]:
        """Return an isolated Schema without running the runtime validator."""
        provided = cast(object, self.schema_provider())
        if not isinstance(provided, Mapping):
            raise OrganelleContractError(
                code="contract.invalid_data_schema",
                message=f"{self.modality} data Schema provider must return an object",
            )
        schema = deepcopy(dict(_string_object_mapping(cast(object, provided), "data Schema")))
        try:
            _validate_draft_2020_12_schema(schema)
        except (TypeError, ValueError) as error:
            raise OrganelleContractError(
                code="contract.invalid_data_schema",
                message=f"{self.modality} data Schema is not valid Draft 2020-12",
                details={"modality": self.modality, "schema_version": self.schema_version},
            ) from error
        properties = schema.get("properties")
        modality_schema = (
            cast(Mapping[str, object], properties).get("modality")
            if isinstance(properties, Mapping)
            else None
        )
        modality_const = (
            cast(Mapping[str, object], modality_schema).get("const")
            if isinstance(modality_schema, Mapping)
            else None
        )
        if modality_const != self.modality:
            raise OrganelleContractError(
                code="contract.data_schema_modality_mismatch",
                message=(
                    f"{self.modality} data Schema must fix properties.modality.const "
                    f"to {self.modality!r}"
                ),
                details={
                    "modality": self.modality,
                    "schema_version": self.schema_version,
                    "schema_modality_const": modality_const,
                },
            )
        return schema


def _validate_draft_2020_12_schema(schema: Mapping[str, object]) -> None:
    json.dumps(schema, allow_nan=False)
    _validate_schema_node(schema)


def _validate_schema_node(value: object) -> None:
    if isinstance(value, bool):
        return
    schema = _string_object_mapping(value, "JSON Schema node")
    dialect = schema.get("$schema")
    if dialect is not None and dialect != _DRAFT_2020_12:
        raise ValueError("data Schema must use Draft 2020-12")

    schema_type = schema.get("type")
    if schema_type is not None:
        if isinstance(schema_type, str):
            schema_types = (schema_type,)
        elif isinstance(schema_type, list):
            type_items = cast(list[object], schema_type)
            if not all(isinstance(item, str) for item in type_items):
                raise ValueError("Schema type arrays must contain strings")
            schema_types = tuple(cast(list[str], type_items))
        else:
            raise ValueError("Schema type must be a string or string array")
        if not schema_types or len(set(schema_types)) != len(schema_types):
            raise ValueError("Schema type arrays must be non-empty and unique")
        if any(item not in _SCHEMA_TYPES for item in schema_types):
            raise ValueError("Schema contains an unknown JSON type")

    required = schema.get("required")
    if required is not None:
        _validate_unique_string_array(required, "required")
    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, list) or not enum:
            raise ValueError("Schema enum must be a non-empty array")
        enum_items = cast(list[object], enum)
        encoded = [json.dumps(item, allow_nan=False, sort_keys=True) for item in enum_items]
        if len(set(encoded)) != len(encoded):
            raise ValueError("Schema enum values must be unique")
    dependent_required = schema.get("dependentRequired")
    if dependent_required is not None:
        dependent_mapping = _string_object_mapping(dependent_required, "dependentRequired")
        for names in dependent_mapping.values():
            _validate_unique_string_array(names, "dependentRequired")

    for key, item in schema.items():
        if key in _SCHEMA_VALUE_KEYS:
            _validate_schema_node(item)
        elif key in _SCHEMA_MAPPING_KEYS:
            nested_mapping = _string_object_mapping(item, key)
            for nested_key, nested in nested_mapping.items():
                if key == "patternProperties":
                    _validate_pattern_lexically(nested_key)
                _validate_schema_node(nested)
        elif key in _SCHEMA_ARRAY_KEYS:
            if not isinstance(item, list) or not item:
                raise ValueError(f"{key} must be a non-empty Schema array")
            for nested in cast(list[object], item):
                _validate_schema_node(nested)
        elif key in _NONNEGATIVE_INTEGER_KEYS and (
            not isinstance(item, int) or isinstance(item, bool) or item < 0
        ):
            raise ValueError(f"{key} must be a non-negative integer")
        elif key in _NUMBER_KEYS:
            if not isinstance(item, (int, float)) or isinstance(item, bool):
                raise ValueError(f"{key} must be a number")
            if key == "multipleOf" and item <= 0:
                raise ValueError("multipleOf must be positive")
        elif key in _STRING_KEYS and not isinstance(item, str):
            raise ValueError(f"{key} must be a string")
        elif key in _BOOLEAN_KEYS and not isinstance(item, bool):
            raise ValueError(f"{key} must be a boolean")
        elif key == "pattern":
            if not isinstance(item, str):
                raise ValueError("pattern must be a string")
            _validate_pattern_lexically(item)
        elif key == "examples" and not isinstance(item, list):
            raise ValueError("examples must be an array")


def _validate_pattern_lexically(pattern: str) -> None:
    group_depth = 0
    in_character_class = False
    escaped = False
    for character in pattern:
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif in_character_class:
            if character == "]":
                in_character_class = False
        elif character == "[":
            in_character_class = True
        elif character == "(":
            group_depth += 1
        elif character == ")":
            if group_depth == 0:
                raise ValueError("pattern contains an unmatched closing group")
            group_depth -= 1
    if escaped:
        raise ValueError("pattern contains a dangling escape")
    if in_character_class:
        raise ValueError("pattern contains an unclosed character class")
    if group_depth:
        raise ValueError("pattern contains an unclosed group")


def _validate_unique_string_array(value: object, keyword: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{keyword} must be a string array")
    items = cast(list[object], value)
    if not all(isinstance(item, str) for item in items):
        raise ValueError(f"{keyword} must be a string array")
    string_items = cast(list[str], items)
    if len(set(string_items)) != len(string_items):
        raise ValueError(f"{keyword} entries must be unique")


def _string_object_mapping(value: object, keyword: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{keyword} must be a string-keyed object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{keyword} must be a string-keyed object")
    return cast(Mapping[str, object], mapping)


class DataContractRegistry:
    """Immutable, deterministic modality-to-contract lookup."""

    def __init__(self, contracts: tuple[DataContract, ...] = ()) -> None:
        indexed: dict[str, DataContract] = {}
        for contract in contracts:
            if contract.modality in indexed:
                raise OrganelleContractError(
                    code="contract.duplicate_data_modality",
                    message=f"duplicate data modality: {contract.modality}",
                )
            indexed[contract.modality] = contract
        self._contracts: Mapping[str, DataContract] = MappingProxyType(
            dict(sorted(indexed.items()))
        )

    def list(self) -> tuple[DataContract, ...]:
        """Return contracts in deterministic modality order."""
        return tuple(self._contracts.values())

    def require(self, modality: str) -> DataContract:
        """Resolve a modality without performing validation or Schema discovery."""
        try:
            return self._contracts[modality]
        except KeyError as error:
            raise OrganelleContractError(
                code="contract.unknown_data_modality",
                message=f"unknown data modality: {modality}",
            ) from error

    def find(self, modality: str) -> DataContract | None:
        """Resolve a modality, returning None when no contract is registered.

        Unlike :meth:`require`, absence is permitted: an ``output_modalities``
        entry without a resolvable contract is a class D audit finding (design
        I3), not a registration error, so output-contract resolution must not
        raise.
        """
        return self._contracts.get(modality)


class _OrganelleRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    accession: str = Field(min_length=1, pattern=r".*\S.*")


class _OrganelleRecordsPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    records: tuple[_OrganelleRecord, ...]


def _validate_organelle_records(data: OrganelleData) -> None:
    if type(data) is not OrganelleData or data.modality != "organelle_records":
        raise OrganelleInputError(
            code="input.invalid_organelle_records",
            message="organelle_records requires exact OrganelleData with matching modality",
        )
    records = data.payload.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        raise OrganelleInputError(
            code="input.invalid_organelle_records",
            message="organelle_records payload requires a records sequence",
        )
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise OrganelleInputError(
                code="input.invalid_organelle_records",
                message="organelle_records entries must be objects",
                details={"record_index": index},
            )
        accession = record.get("accession")
        if not isinstance(accession, str) or not accession.strip():
            raise OrganelleInputError(
                code="input.invalid_organelle_records",
                message="organelle_records entries require non-empty accessions",
                details={"record_index": index},
            )


def _organelle_records_json_schema() -> dict[str, object]:
    return specialize_data_schema(
        modality="organelle_records",
        payload_model=_OrganelleRecordsPayload,
    )


ORGANELLE_RECORDS_DATA_CONTRACT = DataContract(
    modality="organelle_records",
    schema_version="organelleverse.organelle-records.v1",
    validator=_validate_organelle_records,
    schema_provider=_organelle_records_json_schema,
)


class _NuclearAssemblyRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    accession: str = Field(min_length=1, pattern=r".*\S.*")


class _NuclearAssembliesPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    records: tuple[_NuclearAssemblyRecord, ...]


def _validate_nuclear_assemblies(data: OrganelleData) -> None:
    if type(data) is not OrganelleData or data.modality != "nuclear_assemblies":
        raise OrganelleInputError(
            code="input.invalid_nuclear_assemblies",
            message="nuclear_assemblies requires exact OrganelleData with matching modality",
        )
    records = data.payload.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        raise OrganelleInputError(
            code="input.invalid_nuclear_assemblies",
            message="nuclear_assemblies payload requires a records sequence",
        )
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise OrganelleInputError(
                code="input.invalid_nuclear_assemblies",
                message="nuclear_assemblies entries must be objects",
                details={"record_index": index},
            )
        accession = record.get("accession")
        if not isinstance(accession, str) or not accession.strip():
            raise OrganelleInputError(
                code="input.invalid_nuclear_assemblies",
                message="nuclear_assemblies entries require non-empty accessions",
                details={"record_index": index},
            )


def _nuclear_assemblies_json_schema() -> dict[str, object]:
    return specialize_data_schema(
        modality="nuclear_assemblies",
        payload_model=_NuclearAssembliesPayload,
    )


NUCLEAR_ASSEMBLIES_DATA_CONTRACT = DataContract(
    modality="nuclear_assemblies",
    schema_version="organelleverse.nuclear-assemblies.v1",
    validator=_validate_nuclear_assemblies,
    schema_provider=_nuclear_assemblies_json_schema,
)

BUILTIN_DATA_CONTRACTS = (ORGANELLE_RECORDS_DATA_CONTRACT, NUCLEAR_ASSEMBLIES_DATA_CONTRACT)
