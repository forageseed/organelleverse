import json
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, Flag, IntFlag, StrEnum, auto
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Protocol, TypedDict, cast

import pytest
from annotated_types import Ge, Gt, Le, Lt, MultipleOf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    PlainSerializer,
    PlainValidator,
    PrivateAttr,
    ValidationError,
    computed_field,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, PydanticUndefined, core_schema
from typing_extensions import TypeAliasType

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import LineageRecord, OrganelleData
from organelleverse.core.errors import OrganelleContractError
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import (
    ErrorDetail,
    Finding,
    OperationSuggestion,
    OrganelleResult,
)
from organelleverse.operations import OperationRegistry, OperationSpec
from organelleverse.operations.parameters import (
    OperationParameterModel,
    assert_admissible_parameter_model,
)
from organelleverse.operations.signature import (
    derive_operation_signature,
    operation_parameter_schema,
)


class OutputFormat(StrEnum):
    GFF = "gff"
    JSON = "json"


class Format(str, Enum):  # noqa: UP042
    JSON = "json"


class AliasFormat(StrEnum):
    GFF = "gff"
    JSON = "json"

    @classmethod
    def _missing_(cls, value: object) -> "AliasFormat | None":
        if value == "GFF3":
            return cls.GFF
        return None


class ConstructedFormat(StrEnum):
    GFF = "gff"

    def __new__(cls, value: str) -> "ConstructedFormat":
        member = str.__new__(cls, value)
        member._value_ = value
        return member


class InheritedAliasFormat(StrEnum):
    @classmethod
    def _missing_(cls, value: object) -> "InheritedAliasFormat | None":
        if value == "GFF3":
            return cls.__members__.get("GFF")
        return None


class AliasedChildFormat(InheritedAliasFormat):
    GFF = "gff"


class InheritedConstructedFormat(StrEnum):
    def __new__(cls, value: str) -> "InheritedConstructedFormat":
        member = str.__new__(cls, value)
        member._value_ = value
        return member


class ConstructedChildFormat(InheritedConstructedFormat):
    GFF = "gff"


class EmptyFormat(StrEnum):
    pass


class OutputBits(Flag):
    GFF = auto()
    JSON = auto()


class NumericBits(IntFlag):
    GFF = 1
    JSON = 2


class MetadataParameters(Protocol):
    metadata: OrganelleMetadata | None


class FormatParameters(Protocol):
    format: OutputFormat
    backend: Literal["native", "external"]


class StandardFormatParameters(Protocol):
    format: Format


class MetadataCollectionParameters(Protocol):
    metadata: list[OrganelleMetadata]


class MetadataAliasFamilyParameters(Protocol):
    direct: OrganelleMetadata
    renamed: OrganelleMetadata
    nested: OrganelleMetadata
    optional: OrganelleMetadata | None
    listed: list[OrganelleMetadata]
    mapped: dict[str, OrganelleMetadata]
    tupled: tuple[OrganelleMetadata, OrganelleMetadata | None]


class SchemaHookGt(Gt):
    def __get_pydantic_core_schema__(
        self,
        source_type: object,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.str_schema()


class EmptySchemaObject:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: object,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.any_schema()

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        return {}


class LyingStringSchema:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: object,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.any_schema()

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        return {"type": "string"}


class LyingClosedObjectSchema:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: object,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.any_schema()

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        return {"type": "object", "additionalProperties": False}


class LyingOpenObjectSchema:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: object,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.any_schema()

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        return {"type": "object", "additionalProperties": True}


class LyingEnumSchema(StrEnum):
    VALUE = "value"

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: object,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.any_schema()

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        return {"type": "string", "enum": ["value"]}


@dataclass
class DataclassPayload:
    label: str


class TypedDictPayload(TypedDict):
    label: str


class ExternalModelPayload(BaseModel):
    label: str


class ExternalMetadataPayload(OrganelleMetadata):
    pass


class OatkBackendParameters(OperationParameterModel):
    """Representative structured parameter model for an assembly backend."""

    backend: Literal["oatk"] = "oatk"
    minimum_kmer_coverage: Annotated[int, Field(default=30, ge=1)]


class NestedBackendParameters(OperationParameterModel):
    """A structured parameter model that itself contains another one."""

    backend: Literal["oatk"] = "oatk"
    coverage: OatkBackendParameters | None = None


LabelList = TypeAliasType("LabelList", list[str])
PositiveIntAlias = TypeAliasType("PositiveIntAlias", Annotated[int, Gt(0)])
NestedPositiveIntAlias = TypeAliasType("NestedPositiveIntAlias", PositiveIntAlias)
FieldPositiveFloatAlias = TypeAliasType(
    "FieldPositiveFloatAlias",
    Annotated[float, Field(gt=0, le=1)],
)
UnsafeValidatedIntAlias = TypeAliasType(
    "UnsafeValidatedIntAlias",
    Annotated[
        int,
        PlainValidator(lambda value: object(), json_schema_input_type=int),
    ],
)
NestedUnsafeValidatedIntAlias = TypeAliasType(
    "NestedUnsafeValidatedIntAlias",
    UnsafeValidatedIntAlias,
)
RecursivePayload = TypeAliasType("RecursivePayload", list["RecursivePayload"])
NestedRecursivePayload = TypeAliasType("NestedRecursivePayload", RecursivePayload)
MetadataAlias = TypeAliasType("MetadataAlias", OrganelleMetadata)
ReaderEnvelope = TypeAliasType("ReaderEnvelope", OrganelleMetadata)
NestedMetadataAlias = TypeAliasType("NestedMetadataAlias", ReaderEnvelope)
OptionalMetadataAlias = TypeAliasType(
    "OptionalMetadataAlias",
    ReaderEnvelope | None,
)
MetadataCollectionAlias = TypeAliasType(
    "MetadataCollectionAlias",
    list[NestedMetadataAlias],
)
MetadataMappingAlias = TypeAliasType(
    "MetadataMappingAlias",
    dict[str, NestedMetadataAlias],
)
MetadataTupleAlias = TypeAliasType(
    "MetadataTupleAlias",
    tuple[ReaderEnvelope, OptionalMetadataAlias],
)


def annotate(
    genome: OrganelleGenome,
    *,
    backend: Literal["native", "external"] = "native",
    output: Path | None = None,
) -> OrganelleResult:
    raise AssertionError("not executed")


def read_genome(
    path: str | Path,
    *,
    organelle: Literal["mitochondrion", "plastid"],
) -> OrganelleGenome:
    raise AssertionError("not executed")


def analyze_with_positional_parameter(
    genome: OrganelleGenome,
    backend: str,
) -> OrganelleResult:
    raise AssertionError("not executed")


def read_with_keyword_source(*, path: Path) -> OrganelleGenome:
    raise AssertionError("not executed")


def any_parameter(genome: OrganelleGenome, *, value: Any) -> OrganelleResult:
    raise AssertionError("not executed")


def positional_variadic(genome: OrganelleGenome, *values: str) -> OrganelleResult:
    raise AssertionError("not executed")


def core_positional_variadic(*genomes: OrganelleGenome) -> OrganelleResult:
    raise AssertionError("not executed")


def keyword_only_core_input(*, genome: OrganelleGenome) -> OrganelleResult:
    raise AssertionError("not executed")


def core_input_with_default(  # pyright: ignore[reportArgumentType]
    genome: OrganelleGenome = "bad",  # pyright: ignore[reportArgumentType]
) -> OrganelleResult:
    raise AssertionError("not executed")


def classvar_parameter(
    genome: OrganelleGenome,
    *,
    label: ClassVar[str] = "annotation",  # pyright: ignore[reportInvalidTypeForm]
) -> OrganelleResult:
    raise AssertionError("not executed")


def ellipsis_default(
    genome: OrganelleGenome,
    *,
    retries: int = ...,  # pyright: ignore[reportArgumentType]
) -> OrganelleResult:
    raise AssertionError("not executed")


def pydantic_undefined_default(
    genome: OrganelleGenome,
    *,
    retries: int = PydanticUndefined,  # pyright: ignore[reportArgumentType]
) -> OrganelleResult:
    raise AssertionError("not executed")


def keyword_variadic(genome: OrganelleGenome, **values: str) -> OrganelleResult:
    raise AssertionError("not executed")


def untyped_return(genome: OrganelleGenome):
    return genome


def _assert_invalid(function: Callable[..., object], spec: OperationSpec, message: str) -> None:
    with pytest.raises(OrganelleContractError, match=message) as raised:
        derive_operation_signature(function, spec)
    assert raised.value.as_dict()["error_code"] == "contract.invalid_operation_signature"


def _schemas_defining_property(value: object, property_name: str) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        properties = mapping.get("properties")
        if isinstance(properties, dict) and property_name in properties:
            matches.append(mapping)
        for nested in mapping.values():
            matches.extend(_schemas_defining_property(nested, property_name))
    elif isinstance(value, list):
        for nested in cast(list[object], value):
            matches.extend(_schemas_defining_property(nested, property_name))
    return matches


def _reachable_schemas_defining_property(
    schema: dict[str, object],
    field_name: str,
    property_name: str,
) -> list[dict[str, object]]:
    root_properties = cast(dict[str, object], schema["properties"])
    definitions = cast(dict[str, object], schema.get("$defs", {}))
    matches: list[dict[str, object]] = []
    visited_nodes: set[int] = set()

    def visit(value: object) -> None:
        if not isinstance(value, dict):
            return
        node = cast(dict[str, object], value)
        marker = id(node)
        if marker in visited_nodes:
            return
        visited_nodes.add(marker)

        properties = node.get("properties")
        if isinstance(properties, dict) and property_name in properties:
            matches.append(node)

        reference = node.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            target = definitions.get(reference.removeprefix("#/$defs/"))
            visit(target)

        for key in ("items", "contains", "additionalProperties"):
            visit(node.get(key))
        for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
            branches = node.get(key)
            if isinstance(branches, list):
                for branch in cast(list[object], branches):
                    visit(branch)

    visit(root_properties[field_name])
    return matches


def _contains_path_schema(value: object) -> bool:
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        if mapping.get("type") == "string" and mapping.get("format") == "path":
            return True
        return any(_contains_path_schema(item) for item in mapping.values())
    if isinstance(value, list):
        return any(_contains_path_schema(item) for item in cast(list[object], value))
    return False


def _resolve_nested_object_schema(
    root_schema: dict[str, object],
    field_schema: dict[str, object],
) -> dict[str, object]:
    """Follow anyOf/$ref to the nested structured parameter object schema."""
    definitions = cast(dict[str, object], root_schema.get("$defs", {}))
    candidates: list[dict[str, object]] = [field_schema]
    while candidates:
        node = candidates.pop()
        if node.get("type") == "object":
            return node
        reference = node.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            target = definitions.get(reference.removeprefix("#/$defs/"))
            if isinstance(target, dict):
                candidates.append(cast(dict[str, object], target))
            continue
        branches = node.get("anyOf")
        if isinstance(branches, list):
            branch_items = cast(list[object], branches)
            candidates.extend(cast(dict[str, object], branch) for branch in branch_items)
    raise AssertionError("nested structured parameter object schema was not emitted")


def test_signature_excludes_core_input_and_emits_strict_schema(analyze_spec: OperationSpec) -> None:
    derived = derive_operation_signature(annotate, analyze_spec)
    schema = derived.parameter_model.model_json_schema()

    assert derived.input_name == "genome"
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"backend", "output"}
    assert schema["properties"]["backend"]["enum"] == ["native", "external"]
    assert _contains_path_schema(schema["properties"]["output"])
    assert operation_parameter_schema(annotate, analyze_spec) == schema

    assert (
        derived.parameter_model.model_validate({"backend": "native"}).model_dump()["backend"]
        == "native"
    )
    with pytest.raises(ValidationError):
        derived.parameter_model.model_validate({"backend": 1})
    with pytest.raises(ValidationError):
        derived.parameter_model.model_validate({"backend": "native", "extra": True})


def test_read_signature_includes_source_as_a_parameter(read_spec: OperationSpec) -> None:
    derived = derive_operation_signature(read_genome, read_spec)
    schema = derived.parameter_model.model_json_schema()

    assert derived.input_name is None
    assert set(schema["properties"]) == {"path", "organelle"}
    assert schema["required"] == ["path", "organelle"]
    assert _contains_path_schema(schema["properties"]["path"])


def test_enum_optional_tuple_and_constrained_parameters_remain_strict(
    analyze_spec: OperationSpec,
) -> None:
    def parameterized(
        genome: OrganelleGenome,
        *,
        format: OutputFormat = OutputFormat.GFF,
        threshold: Annotated[float, Field(ge=0, le=1)] | None = None,
        coordinates: tuple[int, int] = (1, 2),
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    derived = derive_operation_signature(parameterized, analyze_spec)
    schema = derived.parameter_model.model_json_schema()
    properties = schema["properties"]

    assert schema["$defs"]["OutputFormat"]["enum"] == ["gff", "json"]
    assert properties["threshold"]["anyOf"][0]["minimum"] == 0
    assert properties["threshold"]["anyOf"][0]["maximum"] == 1
    assert properties["coordinates"]["prefixItems"] == [{"type": "integer"}] * 2
    assert derived.parameter_model.model_validate(
        {"format": OutputFormat.JSON, "threshold": 0.5, "coordinates": (3, 4)}
    ).model_dump()["coordinates"] == (3, 4)
    with pytest.raises(ValidationError):
        derived.parameter_model.model_validate({"coordinates": [3, 4]})
    with pytest.raises(ValidationError):
        derived.parameter_model.model_validate({"threshold": 1.1})


def test_valid_path_enum_literal_and_tuple_defaults_have_strict_json_schema(
    analyze_spec: OperationSpec,
) -> None:
    def parameterized(
        genome: OrganelleGenome,
        *,
        output: Path = Path("result.json"),
        format: OutputFormat = OutputFormat.JSON,
        backend: Literal["native", "external"] = "native",
        coordinates: tuple[int, int] = (1, 2),
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    schema = derive_operation_signature(
        parameterized, analyze_spec
    ).parameter_model.model_json_schema()

    json.dumps(schema, allow_nan=False)
    assert set(schema["properties"]) == {"output", "format", "backend", "coordinates"}


@pytest.mark.parametrize(
    "function, fixture_name, message",
    [
        (
            analyze_with_positional_parameter,
            "analyze_spec",
            "keyword-only",
        ),
        (
            read_with_keyword_source,
            "read_spec",
            "read source must be positional-or-keyword",
        ),
    ],
)
def test_rejects_wrong_parameter_positionality(
    request: pytest.FixtureRequest,
    function: Callable[..., object],
    fixture_name: str,
    message: str,
) -> None:
    _assert_invalid(function, cast(OperationSpec, request.getfixturevalue(fixture_name)), message)


def test_rejects_incorrect_core_input_annotation(analyze_spec: OperationSpec) -> None:
    def wrong_input(genome: str) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(wrong_input, analyze_spec, "first parameter does not match input_kind")


def test_rejects_incorrect_core_output_annotation(analyze_spec: OperationSpec) -> None:
    def wrong_output(genome: OrganelleGenome) -> OrganelleGenome:
        raise AssertionError("not executed")

    _assert_invalid(wrong_output, analyze_spec, "return annotation does not match output_kind")


def test_rejects_invalid_default(analyze_spec: OperationSpec) -> None:
    def invalid_default(  # pyright: ignore[reportArgumentType]
        genome: OrganelleGenome,
        *,
        retries: int = "3",  # pyright: ignore[reportArgumentType]
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(invalid_default, analyze_spec, "default does not match its annotation")


def test_rejects_validate_default_metadata_that_hides_an_invalid_default(
    analyze_spec: OperationSpec,
) -> None:
    def invalid_default(  # pyright: ignore[reportArgumentType]
        genome: OrganelleGenome,
        *,
        retries: Annotated[int, Field(validate_default=False)] = "3",  # pyright: ignore[reportArgumentType]
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(invalid_default, analyze_spec, "validate_default")


def test_rejects_pydantic_requiredness_mismatch(
    analyze_spec: OperationSpec,
) -> None:
    _assert_invalid(
        pydantic_undefined_default,
        analyze_spec,
        "requiredness does not match Python signature",
    )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (Field(alias="retryCount"), "alias"),
        (Field(validation_alias="retryCount"), "validation_alias"),
        (Field(serialization_alias="retryCount"), "serialization_alias"),
        (Field(exclude=True), "exclude"),
        (Field(exclude_if=lambda value: value == 0), "exclude_if"),
        (Field(discriminator="kind"), "discriminator"),
    ],
)
def test_rejects_annotated_metadata_that_changes_field_semantics(
    analyze_spec: OperationSpec,
    metadata: object,
    message: str,
) -> None:
    def invalid_metadata(
        genome: OrganelleGenome,
        *,
        retries: Annotated[int, metadata] = 1,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(invalid_metadata, analyze_spec, message)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (Field(strict=False), "metadata"),
        (Field(json_schema_extra={"type": "string"}), "json_schema_extra"),
        (PlainSerializer(str, return_type=str), "metadata"),
    ],
)
def test_rejects_annotation_metadata_that_changes_validation_schema_or_dump(
    analyze_spec: OperationSpec,
    metadata: object,
    message: str,
) -> None:
    def invalid_metadata(
        genome: OrganelleGenome,
        *,
        retries: Annotated[int, metadata] = 1,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(invalid_metadata, analyze_spec, message)


def test_direct_annotated_types_numeric_constraints_remain_supported(
    analyze_spec: OperationSpec,
) -> None:
    def constrained(
        genome: OrganelleGenome,
        *,
        retries: Annotated[int, Gt(0), Ge(1), Lt(10), Le(9), MultipleOf(2)] = 2,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    derived = derive_operation_signature(constrained, analyze_spec)
    schema = derived.parameter_model.model_json_schema()["properties"]["retries"]

    assert schema["exclusiveMinimum"] == 0
    assert schema["minimum"] == 1
    assert schema["exclusiveMaximum"] == 10
    assert schema["maximum"] == 9
    assert schema["multipleOf"] == 2


def test_safe_numeric_constraints_hidden_by_nested_aliases_remain_supported(
    analyze_spec: OperationSpec,
) -> None:
    def constrained(
        genome: OrganelleGenome,
        *,
        retries: NestedPositiveIntAlias = 1,
        threshold: FieldPositiveFloatAlias = 0.5,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    derived = derive_operation_signature(constrained, analyze_spec)
    derived.parameter_model.model_validate({"retries": 1, "threshold": 0.5})
    with pytest.raises(ValidationError):
        derived.parameter_model.model_validate({"retries": 0, "threshold": 0.5})
    with pytest.raises(ValidationError):
        derived.parameter_model.model_validate({"retries": 1, "threshold": 1.5})


@pytest.mark.parametrize(
    "annotation",
    [
        UnsafeValidatedIntAlias,
        NestedUnsafeValidatedIntAlias,
        UnsafeValidatedIntAlias | None,
        list[UnsafeValidatedIntAlias],
    ],
)
def test_rejects_unsafe_validator_metadata_hidden_by_aliases(
    analyze_spec: OperationSpec,
    annotation: object,
) -> None:
    def escaped(
        genome: OrganelleGenome,
        *,
        value: annotation,  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType]
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(
        cast(Callable[..., object], escaped),
        analyze_spec,
        "unsupported Annotated metadata",
    )


@pytest.mark.parametrize(
    ("base", "metadata"),
    [
        (str, Gt(0)),
        (str, Field(gt=0)),
        (bool, Ge(0)),
        (int, Gt(True)),
        (int, Le(cast(Any, "ten"))),
        (float, Lt(float("inf"))),
        (float, Field(ge=float("nan"))),
        (int, Gt(cast(Any, object()))),
        (int, MultipleOf(0)),
        (float, Field(multiple_of=-1)),
        (float, MultipleOf(cast(Any, object()))),
    ],
)
def test_rejects_semantically_invalid_numeric_constraints_before_pydantic(
    analyze_spec: OperationSpec,
    base: object,
    metadata: object,
) -> None:
    def invalid_constraint(
        genome: OrganelleGenome,
        *,
        value: Annotated[base, metadata],  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType]
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(
        cast(Callable[..., object], invalid_constraint),
        analyze_spec,
        "numeric constraint",
    )


def test_rejects_subclass_of_allowed_metadata_with_schema_hook(
    analyze_spec: OperationSpec,
) -> None:
    def escaped(
        genome: OrganelleGenome,
        *,
        retries: Annotated[int, SchemaHookGt(0)],
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(escaped, analyze_spec, "unsupported Annotated metadata")


@pytest.mark.parametrize("metadata", [Field(default=3), Field(default_factory=lambda: 3)])
def test_rejects_defaults_introduced_by_annotated_field_metadata(
    analyze_spec: OperationSpec,
    metadata: object,
) -> None:
    def invalid_metadata(
        genome: OrganelleGenome,
        *,
        retries: Annotated[int, metadata],
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(invalid_metadata, analyze_spec, "default")


@pytest.mark.parametrize("default", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_defaults(
    analyze_spec: OperationSpec,
    default: float,
) -> None:
    def invalid_default(
        genome: OrganelleGenome,
        *,
        threshold: float = default,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    with pytest.raises(OrganelleContractError, match="strict JSON") as raised:
        OperationRegistry().register(analyze_spec, invalid_default)
    assert raised.value.code == "contract.invalid_operation_signature"


def test_rejects_constructed_invalid_metadata_default(
    analyze_spec: OperationSpec,
) -> None:
    invalid_metadata = OrganelleMetadata.model_construct(genetic_code=99)

    def invalid_default(
        genome: OrganelleGenome,
        *,
        metadata: OrganelleMetadata = invalid_metadata,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(
        invalid_default,
        analyze_spec,
        "default does not match its annotation",
    )


@pytest.mark.parametrize("field_default", [Field(...), Field(default=3)])
def test_rejects_field_info_parameter_defaults(
    analyze_spec: OperationSpec,
    field_default: object,
) -> None:
    def invalid_default(
        genome: OrganelleGenome,
        *,
        retries: int = field_default,  # pyright: ignore[reportArgumentType]
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(
        invalid_default,
        analyze_spec,
        "parameter defaults cannot use Pydantic FieldInfo",
    )


@pytest.mark.parametrize(
    "function, message",
    [
        (core_input_with_default, "core input must be required"),
        (
            classvar_parameter,
            "JSON-safe annotation",
        ),
        (
            ellipsis_default,
            "parameter 'retries' cannot use Ellipsis as a default",
        ),
    ],
)
def test_rejects_signature_forms_that_change_callable_semantics(
    analyze_spec: OperationSpec,
    function: Callable[..., object],
    message: str,
) -> None:
    _assert_invalid(function, analyze_spec, message)


@pytest.mark.parametrize(
    "function, message",
    [
        (
            any_parameter,
            "must have a JSON-safe annotation",
        ),
        (
            positional_variadic,
            "variadic positional parameters are forbidden",
        ),
        (
            core_positional_variadic,
            "variadic positional parameters are forbidden",
        ),
        (
            keyword_only_core_input,
            "core input must be positional",
        ),
        (
            keyword_variadic,
            "variadic keyword parameters are forbidden",
        ),
        (
            untyped_return,
            "return annotation does not match output_kind",
        ),
    ],
)
def test_rejects_any_variadics_and_untyped_return(
    analyze_spec: OperationSpec,
    function: Callable[..., object],
    message: str,
) -> None:
    _assert_invalid(function, analyze_spec, message)


def test_rejects_callable_annotation_outside_supported_families(
    analyze_spec: OperationSpec,
) -> None:
    def callback_parameter(
        genome: OrganelleGenome,
        *,
        callback: Callable[[str], str] = lambda value: value,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(callback_parameter, analyze_spec, "JSON-safe annotation")


@pytest.mark.parametrize(
    "annotation",
    [
        object,
        object | None,
        list[object],
        dict[str, object],
        dict[int, str],
        Mapping[int, str],
        list,
        dict,
        tuple,
        Mapping,
        Sequence,
    ],
)
def test_rejects_arbitrary_object_and_untyped_container_annotations(
    analyze_spec: OperationSpec,
    annotation: object,
) -> None:
    def unsafe_parameter(
        genome: OrganelleGenome,
        *,
        value: annotation,  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType]
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(
        cast(Callable[..., object], unsafe_parameter),
        analyze_spec,
        "JSON-safe annotation",
    )


def test_typed_json_containers_and_primitive_unions_remain_supported(
    analyze_spec: OperationSpec,
) -> None:
    def safe_parameters(
        genome: OrganelleGenome,
        *,
        labels: list[str],
        scores: dict[str, int],
        selector: str | int | None = None,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    properties = derive_operation_signature(
        safe_parameters,
        analyze_spec,
    ).parameter_model.model_json_schema()["properties"]

    assert properties["labels"]["items"] == {"type": "string"}
    assert properties["scores"]["additionalProperties"] == {"type": "integer"}
    assert properties["selector"]["anyOf"] == [
        {"type": "string"},
        {"type": "integer"},
        {"type": "null"},
    ]


@pytest.mark.parametrize(
    "annotation",
    [
        AliasFormat,
        ConstructedFormat,
        AliasedChildFormat,
        ConstructedChildFormat,
        EmptyFormat,
        OutputBits,
        NumericBits,
        Literal[1],
        Literal[True],
        Literal["native", 1],
        set[str],
        frozenset[str],
    ],
)
def test_rejects_parameter_types_whose_json_schema_and_runtime_can_diverge(
    analyze_spec: OperationSpec,
    annotation: object,
) -> None:
    def unsafe_parameter(
        genome: OrganelleGenome,
        *,
        value: annotation,  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType]
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(
        cast(Callable[..., object], unsafe_parameter),
        analyze_spec,
        "JSON-safe annotation",
    )


def test_simple_string_enum_and_string_literal_remain_supported(
    analyze_spec: OperationSpec,
) -> None:
    def safe_parameter(
        genome: OrganelleGenome,
        *,
        format: OutputFormat,
        backend: Literal["native", "external"],
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    derived = derive_operation_signature(safe_parameter, analyze_spec)
    validated = derived.parameter_model.model_validate_json(
        json.dumps({"format": "json", "backend": "native"})
    )

    assert cast(FormatParameters, validated).format is OutputFormat.JSON
    assert cast(FormatParameters, validated).backend == "native"


def test_standard_string_enum_without_member_constructor_is_supported(
    analyze_spec: OperationSpec,
) -> None:
    def safe_parameter(
        genome: OrganelleGenome,
        *,
        format: Format,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    derived = derive_operation_signature(safe_parameter, analyze_spec)
    validated = derived.parameter_model.model_validate_json(json.dumps({"format": "json"}))

    assert cast(StandardFormatParameters, validated).format is Format.JSON


def test_all_intended_parameter_annotation_families_remain_agent_json_compatible(
    analyze_spec: OperationSpec,
) -> None:
    def supported_parameters(
        genome: OrganelleGenome,
        *,
        flag: bool,
        count: int,
        ratio: float,
        label: str,
        nothing: None,
        output: Path,
        format: OutputFormat,
        backend: Literal["native", "external"],
        labels: list[str],
        coordinates: tuple[int, str],
        series: tuple[int, ...],
        scores: dict[str, float],
        mapping: Mapping[str, int],
        sequence: Sequence[str],
        selector: str | int | None,
        threshold: Annotated[float, Ge(0), Le(1)],
        metadata: OrganelleMetadata,
        aliased_labels: LabelList,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    derived = derive_operation_signature(supported_parameters, analyze_spec)
    payload = {
        "flag": True,
        "count": 2,
        "ratio": 0.5,
        "label": "annotation",
        "nothing": None,
        "output": "result.json",
        "format": "json",
        "backend": "native",
        "labels": ["a", "b"],
        "coordinates": [1, "two"],
        "series": [1, 2],
        "scores": {"confidence": 0.9},
        "mapping": {"start": 1},
        "sequence": ["A", "T"],
        "selector": 3,
        "threshold": 0.75,
        "metadata": {"kind": "metadata", "species": "Arabidopsis thaliana"},
        "aliased_labels": ["trusted", "alias"],
    }

    validated = derived.parameter_model.model_validate_json(json.dumps(payload))
    dumped = validated.model_dump(mode="json")

    assert set(derived.parameter_model.model_json_schema()["properties"]) == set(payload)
    assert dumped["output"] == "result.json"
    assert dumped["metadata"]["species"] == "Arabidopsis thaliana"


@pytest.mark.parametrize(
    "annotation",
    [
        ArtifactRef,
        OrganelleGenome,
        OrganelleData,
        OrganelleResult,
        LineageRecord,
        ResultProvenance,
        Finding,
        ErrorDetail,
        OperationSuggestion,
    ],
)
def test_rejects_core_transport_and_internal_models_as_keyword_parameters(
    analyze_spec: OperationSpec,
    annotation: object,
) -> None:
    def invalid_model_parameter(
        genome: OrganelleGenome,
        *,
        value: annotation,  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType]
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(
        cast(Callable[..., object], invalid_model_parameter),
        analyze_spec,
        "JSON-safe annotation",
    )


def test_optional_organelle_metadata_schema_accepts_and_verifies_computed_object_id(
    read_spec: OperationSpec,
) -> None:
    def read_with_metadata(
        path: str,
        *,
        metadata: OrganelleMetadata | None = None,
    ) -> OrganelleGenome:
        raise AssertionError("not executed")

    derived = derive_operation_signature(read_with_metadata, read_spec)
    schema = derived.parameter_model.model_json_schema()
    object_id_schema = schema["$defs"]["OrganelleMetadata"]["properties"]["object_id"]
    metadata = OrganelleMetadata(species="Arabidopsis thaliana")
    serialized = metadata.model_dump(mode="json")

    assert object_id_schema["type"] == "string"
    assert object_id_schema["pattern"] == r"^metadata:sha256:[0-9a-f]{64}$"
    validated = derived.parameter_model.model_validate_json(
        json.dumps({"path": "genome.fa", "metadata": serialized})
    )
    assert cast(MetadataParameters, validated).metadata == metadata
    validated_none = derived.parameter_model.model_validate_json(
        json.dumps({"path": "genome.fa", "metadata": None})
    )
    assert cast(MetadataParameters, validated_none).metadata is None

    with pytest.raises(ValidationError, match="serialized object_id must be a string"):
        derived.parameter_model.model_validate_json(
            json.dumps(
                {
                    "path": "genome.fa",
                    "metadata": {**serialized, "object_id": None},
                }
            )
        )

    forged = {**serialized, "object_id": "metadata:sha256:" + "0" * 64}
    with pytest.raises(ValidationError, match="object_id"):
        derived.parameter_model.model_validate_json(
            json.dumps({"path": "genome.fa", "metadata": forged})
        )


@pytest.mark.parametrize(
    "annotation",
    [
        MetadataAlias,
        ReaderEnvelope,
        NestedMetadataAlias,
        OptionalMetadataAlias,
    ],
)
def test_metadata_object_id_schema_augmentation_is_alias_name_independent(
    read_spec: OperationSpec,
    annotation: object,
) -> None:
    def read_with_aliased_metadata(
        path: str,
        *,
        metadata: annotation,  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType]
    ) -> OrganelleGenome:
        raise AssertionError("not executed")

    derived = derive_operation_signature(
        cast(Callable[..., object], read_with_aliased_metadata),
        read_spec,
    )
    schema = derived.parameter_model.model_json_schema()
    metadata_schemas = _schemas_defining_property(schema, "object_id")
    metadata = OrganelleMetadata(species="Arabidopsis thaliana")
    validated = derived.parameter_model.model_validate_json(
        json.dumps(
            {
                "path": "genome.fa",
                "metadata": metadata.model_dump(mode="json"),
            }
        )
    )

    assert len(metadata_schemas) == 1
    properties = cast(dict[str, object], metadata_schemas[0]["properties"])
    assert cast(dict[str, object], properties["kind"])["const"] == "metadata"
    assert properties["object_id"] == {
        "pattern": r"^metadata:sha256:[0-9a-f]{64}$",
        "readOnly": True,
        "title": "Object Id",
        "type": "string",
    }
    assert cast(MetadataParameters, validated).metadata == metadata

    if annotation is OptionalMetadataAlias:
        validated_none = derived.parameter_model.model_validate_json(
            json.dumps({"path": "genome.fa", "metadata": None})
        )
        assert cast(MetadataParameters, validated_none).metadata is None


def test_metadata_object_id_schema_augmentation_reaches_nested_alias_containers(
    read_spec: OperationSpec,
) -> None:
    def read_with_metadata_collection(
        path: str,
        *,
        metadata: MetadataCollectionAlias,
        labels: dict[str, str],
    ) -> OrganelleGenome:
        raise AssertionError("not executed")

    derived = derive_operation_signature(read_with_metadata_collection, read_spec)
    schema = derived.parameter_model.model_json_schema()
    metadata = OrganelleMetadata(species="Arabidopsis thaliana")
    validated = derived.parameter_model.model_validate_json(
        json.dumps(
            {
                "path": "genome.fa",
                "metadata": [metadata.model_dump(mode="json")],
                "labels": {"source": "review"},
            }
        )
    )

    assert len(_schemas_defining_property(schema, "object_id")) == 1
    assert cast(MetadataCollectionParameters, validated).metadata == [metadata]
    assert not _schemas_defining_property(
        cast(dict[str, object], schema["properties"])["labels"],
        "object_id",
    )


def test_metadata_shaped_primitive_parameters_do_not_gain_object_id(
    analyze_spec: OperationSpec,
) -> None:
    def metadata_named_primitives(
        genome: OrganelleGenome,
        *,
        kind: Literal["metadata"],
        species: str,
        accession: str,
        genetic_code: int,
        assembly_type: str,
        plastid_type: str,
        source: str,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    schema = operation_parameter_schema(metadata_named_primitives, analyze_spec)
    properties = cast(dict[str, object], schema["properties"])

    assert set(properties) == {
        "kind",
        "species",
        "accession",
        "genetic_code",
        "assembly_type",
        "plastid_type",
        "source",
    }
    assert "object_id" not in properties


def test_metadata_schema_augmentation_follows_only_annotated_field_reachability(
    read_spec: OperationSpec,
) -> None:
    def read_with_metadata_families(
        path: str,
        *,
        direct: OrganelleMetadata,
        renamed: ReaderEnvelope,
        nested: NestedMetadataAlias,
        optional: OptionalMetadataAlias,
        listed: MetadataCollectionAlias,
        mapped: MetadataMappingAlias,
        tupled: MetadataTupleAlias,
        unrelated: dict[str, str],
    ) -> OrganelleGenome:
        raise AssertionError("not executed")

    derived = derive_operation_signature(read_with_metadata_families, read_spec)
    schema = derived.parameter_model.model_json_schema()
    metadata = OrganelleMetadata(species="Arabidopsis thaliana")
    serialized = metadata.model_dump(mode="json")
    validated = derived.parameter_model.model_validate_json(
        json.dumps(
            {
                "path": "genome.fa",
                "direct": serialized,
                "renamed": serialized,
                "nested": serialized,
                "optional": serialized,
                "listed": [serialized],
                "mapped": {"primary": serialized},
                "tupled": [serialized, None],
                "unrelated": {"kind": "metadata", "species": "not a contract"},
            }
        )
    )

    for field_name in (
        "direct",
        "renamed",
        "nested",
        "optional",
        "listed",
        "mapped",
        "tupled",
    ):
        reachable = _reachable_schemas_defining_property(schema, field_name, "object_id")
        assert reachable
        for metadata_schema in reachable:
            properties = cast(dict[str, object], metadata_schema["properties"])
            assert properties["object_id"] == {
                "pattern": r"^metadata:sha256:[0-9a-f]{64}$",
                "readOnly": True,
                "title": "Object Id",
                "type": "string",
            }

    parameters = cast(MetadataAliasFamilyParameters, validated)
    assert type(parameters.direct) is OrganelleMetadata
    assert type(parameters.renamed) is OrganelleMetadata
    assert type(parameters.nested) is OrganelleMetadata
    assert type(parameters.optional) is OrganelleMetadata
    assert all(type(item) is OrganelleMetadata for item in parameters.listed)
    assert all(type(item) is OrganelleMetadata for item in parameters.mapped.values())
    assert type(parameters.tupled[0]) is OrganelleMetadata
    assert parameters.tupled[1] is None
    assert not _reachable_schemas_defining_property(schema, "unrelated", "object_id")


@pytest.mark.parametrize(
    "annotation",
    [
        LyingStringSchema,
        LyingClosedObjectSchema,
        LyingOpenObjectSchema,
        LyingEnumSchema,
        DataclassPayload,
        TypedDictPayload,
        ExternalModelPayload,
        ExternalMetadataPayload,
        RecursivePayload,
        NestedRecursivePayload,
        RecursivePayload | None,
        list[RecursivePayload],
    ],
)
def test_rejects_untrusted_annotation_families_even_with_safe_looking_schema(
    analyze_spec: OperationSpec,
    annotation: object,
) -> None:
    def unsafe_parameter(
        genome: OrganelleGenome,
        *,
        value: annotation,  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType]
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(
        cast(Callable[..., object], unsafe_parameter),
        analyze_spec,
        "JSON-safe annotation",
    )


def test_rejects_custom_annotation_with_empty_json_schema(
    analyze_spec: OperationSpec,
) -> None:
    def unsafe_parameter(
        genome: OrganelleGenome,
        *,
        value: EmptySchemaObject,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(unsafe_parameter, analyze_spec, "JSON-safe annotation")


def test_rejects_unresolvable_forward_reference_with_stable_contract_error(
    analyze_spec: OperationSpec,
) -> None:
    def unresolved_parameter(
        genome: OrganelleGenome,
        *,
        value: str,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    unresolved_parameter.__annotations__["value"] = "MissingOperationParameter"
    _assert_invalid(unresolved_parameter, analyze_spec, "resolvable annotations")


def test_structured_parameter_model_is_an_optional_keyword_parameter(
    analyze_spec: OperationSpec,
) -> None:
    def with_backend_parameters(
        genome: OrganelleGenome,
        *,
        backend_parameters: OatkBackendParameters | None = None,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    derived = derive_operation_signature(with_backend_parameters, analyze_spec)

    assert set(derived.parameter_model.model_fields) == {"backend_parameters"}
    assert derived.parameter_model.model_fields["backend_parameters"].is_required() is False


def test_structured_parameter_model_emits_a_strict_closed_schema(
    analyze_spec: OperationSpec,
) -> None:
    def with_backend_parameters(
        genome: OrganelleGenome,
        *,
        backend_parameters: OatkBackendParameters | None = None,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    schema = derive_operation_signature(
        with_backend_parameters, analyze_spec
    ).parameter_model.model_json_schema()
    json.dumps(schema, allow_nan=False)
    properties = cast(dict[str, object], schema["properties"])
    assert schema["additionalProperties"] is False

    nested_object = _resolve_nested_object_schema(
        schema, cast(dict[str, object], properties["backend_parameters"])
    )
    assert nested_object["additionalProperties"] is False
    nested_properties = cast(dict[str, dict[str, object]], nested_object["properties"])
    assert nested_properties["backend"]["const"] == "oatk"
    assert nested_properties["minimum_kmer_coverage"]["minimum"] == 1
    assert nested_properties["minimum_kmer_coverage"]["default"] == 30


def test_structured_parameter_model_decodes_to_the_exact_nested_instance(
    analyze_spec: OperationSpec,
) -> None:
    def with_backend_parameters(
        genome: OrganelleGenome,
        *,
        backend_parameters: OatkBackendParameters | None = None,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    derived = derive_operation_signature(with_backend_parameters, analyze_spec)
    validated = derived.parameter_model.model_validate_json(
        json.dumps({"backend_parameters": {"backend": "oatk", "minimum_kmer_coverage": 41}})
    )

    parameters = cast("BackendParameterHolder", validated)
    assert parameters.backend_parameters is not None
    assert type(parameters.backend_parameters) is OatkBackendParameters
    assert parameters.backend_parameters.minimum_kmer_coverage == 41


def test_structured_parameter_model_keeps_strict_int_and_extra_rejection(
    analyze_spec: OperationSpec,
) -> None:
    def with_backend_parameters(
        genome: OrganelleGenome,
        *,
        backend_parameters: OatkBackendParameters | None = None,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    model = derive_operation_signature(with_backend_parameters, analyze_spec).parameter_model

    with pytest.raises(ValidationError) as coerced:
        model.model_validate_json(
            json.dumps({"backend_parameters": {"backend": "oatk", "minimum_kmer_coverage": "41"}})
        )
    assert coerced.value.errors()[0]["type"] == "int_type"
    with pytest.raises(ValidationError) as extra:
        model.model_validate_json(
            json.dumps(
                {
                    "backend_parameters": {
                        "backend": "oatk",
                        "minimum_kmer_coverage": 30,
                        "raw": True,
                    }
                }
            )
        )
    assert extra.value.errors()[0]["type"] == "extra_forbidden"


def test_structured_parameter_model_revalidates_constructed_invalid_state(
    analyze_spec: OperationSpec,
) -> None:
    def with_backend_parameters(
        genome: OrganelleGenome,
        *,
        backend_parameters: OatkBackendParameters | None = None,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    model = derive_operation_signature(with_backend_parameters, analyze_spec).parameter_model
    constructed = OatkBackendParameters.model_construct(minimum_kmer_coverage=0)

    with pytest.raises(ValidationError) as raised:
        model.model_validate({"backend_parameters": constructed})
    assert raised.value.errors()[0]["type"] == "greater_than_equal"


def test_structured_parameter_model_rejects_an_invalid_default(
    analyze_spec: OperationSpec,
) -> None:
    invalid_default = OatkBackendParameters.model_construct(minimum_kmer_coverage=0)

    def with_invalid_default(  # pyright: ignore[reportArgumentType]
        genome: OrganelleGenome,
        *,
        backend_parameters: OatkBackendParameters = invalid_default,  # pyright: ignore[reportArgumentType]
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(with_invalid_default, analyze_spec, "default does not match its annotation")


def test_nested_structured_parameter_model_remains_strict(analyze_spec: OperationSpec) -> None:
    def with_nested_parameters(
        genome: OrganelleGenome,
        *,
        backend_parameters: NestedBackendParameters | None = None,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    schema = derive_operation_signature(
        with_nested_parameters, analyze_spec
    ).parameter_model.model_json_schema()
    json.dumps(schema, allow_nan=False)
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    "annotation",
    [
        BaseModel,
        ExternalModelPayload,
        ExternalMetadataPayload,
        TypedDictPayload,
        DataclassPayload,
        EmptySchemaObject,
        LyingClosedObjectSchema,
        RecursivePayload,
        NestedRecursivePayload,
        RecursivePayload | None,
        list[RecursivePayload],
    ],
)
def test_structured_support_rejects_untrusted_model_families(
    analyze_spec: OperationSpec,
    annotation: object,
) -> None:
    def unsafe_parameter(
        genome: OrganelleGenome,
        *,
        value: annotation,  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType]
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    _assert_invalid(
        cast(Callable[..., object], unsafe_parameter),
        analyze_spec,
        "JSON-safe annotation",
    )


def test_structured_parameter_model_with_unsafe_field_is_rejected_at_definition() -> None:
    with pytest.raises(TypeError):

        class UnstructuredParameterModel(OperationParameterModel):  # pyright: ignore[reportUnusedClass]
            payload: dict[str, object]


class BackendParameterHolder(Protocol):
    backend_parameters: OatkBackendParameters | None


# ---------------------------------------------------------------------------
# Security boundary tests: every bypass below must be rejected at definition
# time or when an operation signature is derived/registered. These exercise the
# trust boundary repaired after the structured-parameter review.
# ---------------------------------------------------------------------------


def _define(cls_body: str, namespace: dict[str, Any] | None = None) -> None:
    """Compile a class body against OperationParameterModel (raises on rejection)."""
    active: dict[str, Any] = {
        "OperationParameterModel": OperationParameterModel,
        "Field": Field,
        "Mapping": Mapping,
        "field_validator": field_validator,
        "field_serializer": field_serializer,
        "model_validator": model_validator,
        "model_serializer": model_serializer,
        "computed_field": computed_field,
        "PrivateAttr": PrivateAttr,
        "cached_property": cached_property,
        "StrEnum": StrEnum,
    }
    if namespace:
        active.update(namespace)
    exec(f"class Probe(OperationParameterModel):\n{cls_body}", active)


def test_field_alias_is_rejected_on_a_structured_parameter_model() -> None:
    with pytest.raises(TypeError, match="alias"):
        _define("    value: int = Field(alias='renamed')")


def test_field_validation_and_serialization_aliases_are_rejected() -> None:
    with pytest.raises(TypeError, match="alias"):
        _define("    value: int = Field(validation_alias='renamed')")
    with pytest.raises(TypeError, match="alias"):
        _define("    value: int = Field(serialization_alias='renamed')")


def test_field_default_factory_is_rejected() -> None:
    with pytest.raises(TypeError, match="default_factory"):
        _define("    value: int = Field(default_factory=lambda: 7)")


def test_field_json_schema_extra_is_rejected() -> None:
    with pytest.raises(TypeError, match=r"json_schema_extra|schema"):
        _define("    value: int = Field(json_schema_extra={'x': 1})")


def test_model_config_alias_generator_is_rejected() -> None:
    def upper_alias(name: str) -> str:
        return name.upper()

    namespace: dict[str, Any] = {
        "bad_config": {
            **dict(OperationParameterModel.model_config),
            "alias_generator": upper_alias,
        }
    }
    with pytest.raises(TypeError, match="model_config"):
        _define("    model_config = bad_config\n    value: int = 1", namespace)


def test_model_config_json_schema_extra_is_rejected() -> None:
    namespace: dict[str, Any] = {
        "bad_config": {
            **dict(OperationParameterModel.model_config),
            "json_schema_extra": {"x": 1},
        }
    }
    with pytest.raises(TypeError, match="model_config"):
        _define("    model_config = bad_config\n    value: int = 1", namespace)


def test_subclass_init_subclass_cannot_relax_strictness_after_validation() -> None:
    # An intermediate class cannot override a Pydantic/Python lifecycle hook to relax
    # descendant configuration. The behaviour hook itself is rejected at definition, so
    # no descendant can be built that mutates config post-validation.
    src = textwrap.dedent(
        """
        class Evasive(OperationParameterModel):
            def __init_subclass__(cls, **kwargs):
                super().__init_subclass__(**kwargs)
                cls.model_config = {**dict(cls.model_config), 'strict': False}
                cls.model_rebuild(force=True)
        """
    )
    namespace: dict[str, object] = {"OperationParameterModel": OperationParameterModel}
    with pytest.raises(TypeError, match=r"behaviour hook|__init_subclass__"):
        exec(src, namespace)


def test_post_definition_config_mutation_and_rebuild_is_rejected_at_derivation(
    analyze_spec: OperationSpec,
) -> None:
    class Mutable(OperationParameterModel):
        value: int = 1

    # Mutate strictness only (keep extra="forbid") so the schema stays closed; the
    # mutation must still be caught by a fresh audit when an operation is derived.
    Mutable.model_config = cast("ConfigDict", {**dict(Mutable.model_config), "strict": False})
    Mutable.model_rebuild(force=True)
    assert Mutable.model_config.get("strict") is False

    def with_mutable(genome: OrganelleGenome, *, value: Mutable | None = None) -> OrganelleResult:
        raise AssertionError("not executed")

    with pytest.raises(OrganelleContractError) as raised:
        derive_operation_signature(with_mutable, analyze_spec)
    assert raised.value.code == "contract.invalid_operation_signature"
    assert isinstance(raised.value.__cause__, TypeError)


def test_inherited_model_post_init_is_rejected() -> None:
    src = textwrap.dedent(
        """
        class MutatingMixin(OperationParameterModel):
            def model_post_init(self, __context):
                object.__setattr__(self, 'value', 999)

        class Child(MutatingMixin):
            value: int = 1
        """
    )
    namespace: dict[str, object] = {"OperationParameterModel": OperationParameterModel}
    with pytest.raises(TypeError, match=r"model_post_init|behavior"):
        exec(src, namespace)


def test_non_parameter_model_mixin_is_rejected() -> None:
    src = textwrap.dedent(
        """
        class ForeignMixin:
            def model_post_init(self, __context):
                object.__setattr__(self, 'value', 999)

        class Child(ForeignMixin, OperationParameterModel):
            value: int = 1
        """
    )
    namespace: dict[str, object] = {"OperationParameterModel": OperationParameterModel}
    with pytest.raises(TypeError, match=r"mixin|model_post_init|behavior"):
        exec(src, namespace)


def test_field_validator_is_rejected_on_structured_model() -> None:
    with pytest.raises(TypeError, match="validator"):
        _define(
            "    value: int = 1\n"
            "    @field_validator('value')\n"
            "    @classmethod\n"
            "    def v(cls, val):\n"
            "        return val"
        )


def test_field_serializer_is_rejected_on_structured_model() -> None:
    with pytest.raises(TypeError, match="serializer"):
        _define(
            "    value: int = 1\n"
            "    @field_serializer('value')\n"
            "    def s(self, val):\n"
            "        return val"
        )


def test_computed_field_is_rejected_on_structured_model() -> None:
    with pytest.raises(TypeError, match="computed"):
        _define(
            "    value: int = 1\n"
            "    @computed_field\n"
            "    @property\n"
            "    def c(self) -> int:\n"
            "        return 1"
        )


def test_custom_missing_enum_is_rejected_on_a_structured_field() -> None:
    src = textwrap.dedent(
        """
        class CoercingEnum(StrEnum):
            ONE = 'one'
            @classmethod
            def _missing_(cls, value):
                return cls.ONE if value == 'uno' else None

        class WithEnum(OperationParameterModel):
            level: CoercingEnum = CoercingEnum.ONE
        """
    )
    namespace: dict[str, object] = {
        "OperationParameterModel": OperationParameterModel,
        "StrEnum": StrEnum,
    }
    with pytest.raises(TypeError, match="enum"):
        exec(src, namespace)


def test_invalid_field_default_is_rejected_at_definition() -> None:
    with pytest.raises(TypeError, match="default"):
        _define("    value: int = 'not-an-int'  # type: ignore[assignment]")


def test_nested_mapping_field_is_rejected() -> None:
    with pytest.raises(TypeError, match=r"dict|mapping|free-form|non-safe"):
        _define("    opts: dict[str, int] = {}")
    with pytest.raises(TypeError, match=r"dict|mapping|free-form|non-safe"):
        _define("    opts: Mapping[str, int] = {}")


def test_recursive_structured_model_is_rejected() -> None:
    src = textwrap.dedent(
        """
        class Recur(OperationParameterModel):
            self_ref: 'Recur | None' = None
        """
    )
    namespace: dict[str, object] = {"OperationParameterModel": OperationParameterModel}
    # Self-reference is rejected whether it surfaces as recursion or as an
    # unresolved forward reference; both are fail-closed.
    with pytest.raises(TypeError, match=r"recursive|non-safe|forward reference"):
        exec(src, namespace)


def test_assignment_field_strict_override_is_rejected() -> None:
    with pytest.raises(TypeError, match=r"metadata|strict|Field"):
        _define("    value: int = Field(strict=False)")


def test_custom_init_is_rejected_on_a_structured_parameter_model() -> None:
    with pytest.raises(TypeError, match=r"__init__|custom initialization|non-field"):
        _define(
            "    value: int\n"
            "    def __init__(self, **data):\n"
            "        super().__init__(**data)\n"
            "        object.__setattr__(self, 'value', 999)"
        )


def test_custom_attribute_access_is_rejected_on_a_structured_parameter_model() -> None:
    with pytest.raises(TypeError, match=r"__getattribute__|non-field|behaviour"):
        _define(
            "    value: int\n"
            "    def __getattribute__(self, name):\n"
            "        if name == 'value':\n"
            "            return 'schema-lied'\n"
            "        return super().__getattribute__(name)"
        )


def test_classvar_is_rejected_on_a_structured_parameter_model() -> None:
    with pytest.raises(TypeError, match=r"ClassVar|non-field"):
        _define(
            "    state: ClassVar[list[int]] = []\n    value: int = 1",
            {"ClassVar": ClassVar},
        )


def test_helper_method_is_rejected_on_a_structured_parameter_model() -> None:
    with pytest.raises(TypeError, match=r"helper|non-field|behaviour"):
        _define("    value: int = 1\n    def helper(self):\n        return self.value")


def test_private_attribute_is_rejected_on_a_structured_parameter_model() -> None:
    with pytest.raises(TypeError, match=r"PrivateAttr|private|non-field|model_post_init|behaviour"):
        _define("    _state: int = PrivateAttr(default=0)\n    value: int = 1")


def test_cached_property_is_rejected_on_a_structured_parameter_model() -> None:
    with pytest.raises(TypeError, match=r"cached|non-field|behaviour"):
        _define(
            "    value: int = 1\n"
            "    @cached_property\n"
            "    def cached(self):\n"
            "        return self.value"
        )


def test_post_definition_data_descriptor_is_rejected_at_readmission() -> None:
    class Mutable(OperationParameterModel):
        value: int

    class LyingDescriptor:
        def __get__(self, instance: object, owner: type[object]) -> object:
            if instance is None:
                return self
            return "schema-lied"

        def __set__(self, instance: object, value: object) -> None:
            vars(instance)["value"] = value

    type.__setattr__(Mutable, "value", LyingDescriptor())
    assert Mutable(value=7).value == "schema-lied"

    with pytest.raises(TypeError, match=r"value|namespace|non-field"):
        assert_admissible_parameter_model(Mutable)


def test_post_definition_hidden_state_is_rejected_at_readmission() -> None:
    class Mutable(OperationParameterModel):
        value: int = 1

    type.__setattr__(Mutable, "hidden_state", {})

    with pytest.raises(TypeError, match=r"hidden_state|namespace|non-field"):
        assert_admissible_parameter_model(Mutable)


def test_slots_are_rejected_on_a_structured_parameter_model() -> None:
    with pytest.raises(TypeError, match=r"__slots__|hidden|namespace|non-field"):

        class Slotted(OperationParameterModel):  # pyright: ignore[reportUnusedClass]
            __slots__ = ("hidden",)
            value: int = 1


def test_field_validate_default_override_is_rejected() -> None:
    with pytest.raises(TypeError, match=r"validate_default|Field"):
        _define("    value: int = Field(default=1, validate_default=False)")


def test_required_field_does_not_invalidate_an_unrelated_valid_default() -> None:
    class RequiredAndDefault(OperationParameterModel):
        required: int
        optional: int = 1

    value = RequiredAndDefault(required=7)

    assert value.required == 7
    assert value.optional == 1


def test_model_validator_is_rejected_on_structured_model() -> None:
    with pytest.raises(TypeError, match="validator"):
        _define(
            "    value: int = 1\n"
            "    @model_validator(mode='after')\n"
            "    def validate_model(self):\n"
            "        return self"
        )


def test_model_serializer_is_rejected_on_structured_model() -> None:
    with pytest.raises(TypeError, match="serializer"):
        _define(
            "    value: int = 1\n"
            "    @model_serializer\n"
            "    def serialize_model(self):\n"
            "        return {'value': self.value}"
        )


def test_concrete_parameter_models_use_flat_inheritance() -> None:
    class Parent(OperationParameterModel):
        common: int = 1

    with pytest.raises(TypeError, match=r"direct|base|inherit"):

        class Child(Parent):  # pyright: ignore[reportUnusedClass]
            value: int = 2


def test_parameter_model_base_cannot_be_used_as_a_nested_field() -> None:
    with pytest.raises(TypeError, match=r"OperationParameterModel|base|non-safe"):
        _define("    payload: OperationParameterModel")
