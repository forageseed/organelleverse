import math
import re
from collections.abc import Mapping, Sequence, Set
from copy import deepcopy
from enum import StrEnum
from typing import Any, Literal, Self, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.main import IncEx

_KEYWORD_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

AbstractSetIntStr: TypeAlias = Set[int] | Set[str]
MappingIntStrAny: TypeAlias = Mapping[int, Any] | Mapping[str, Any]


class OperationStage(StrEnum):
    READ = "read"
    TRANSFORM = "transform"
    ANALYZE = "analyze"
    CONSUME = "consume"


class CoreKind(StrEnum):
    NONE = "none"
    GENOME = "genome"
    DATA = "data"
    RESULT = "result"


class SideEffect(StrEnum):
    READ_FILES = "read_files"
    WRITE_FILES = "write_files"
    SUBPROCESS = "subprocess"
    NETWORK = "network"
    GPU = "gpu"


class DependencyKind(StrEnum):
    PYTHON = "python"
    EXECUTABLE = "executable"
    MODEL = "model"
    DATABASE = "database"


class ExecutionMode(StrEnum):
    """Whether a call may stay inline or always produces a durable run handle.

    A runtime contract, decided per operation from actual code facts (its own
    side effects and role) — never derived from a restoration-inventory
    ``execution_class`` (``pure``/``tool``/``model``/``network``), which is a
    separate, non-admission-participating migration hint.
    """

    INLINE = "inline"
    DURABLE = "durable"


class ArgumentMode(StrEnum):
    """How a Python binding maps invocation arguments onto the implementation."""

    CANONICAL_CORE = "canonical_core"
    NAMED_PARAMETERS = "named_parameters"
    PLUGIN_PROTOCOL = "plugin_protocol"


class ParameterCodec(StrEnum):
    """How one named parameter's Agent-facing JSON value decodes to Python."""

    JSON = "json"
    PATH = "path"
    DIRECTORY = "directory"
    LEGACY_GENOME = "legacy_genome"
    LEGACY_DATA = "legacy_data"
    LEGACY_RESULT = "legacy_result"
    NUMPY_ARTIFACT = "numpy_artifact"
    DATACLASS = "dataclass"


class ParameterSource(StrEnum):
    """Where a named parameter's value comes from."""

    AGENT = "agent"
    ENVIRONMENT = "environment"


class ExecutionContext(StrEnum):
    """Where a path value is anchored."""

    HOST = "host"
    WSL = "wsl"
    CONTAINER = "container"
    REMOTE = "remote"


class ProgressSource(StrEnum):
    """How the UI should present execution progress."""

    STAGES = "stages"
    TOOL = "tool"
    NONE = "none"


class AgentSurface(StrEnum):
    """How much of an interactive plugin is exposed to the agent."""

    GRANULAR = "granular"
    INTERACTIVE = "interactive"


class ResultCodec(StrEnum):
    """How an implementation's direct Python return value becomes an OrganelleResult."""

    CANONICAL = "canonical"
    CANONICAL_JSON = "canonical_json"
    LEGACY_RESULT = "legacy_result"
    JSON_METRIC = "json_metric"
    ARTIFACT = "artifact"


class StrictSpecModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        values = self.model_dump(mode="python", round_trip=True)
        if deep:
            values = deepcopy(values)
        if update is not None:
            values.update(update)
        return type(self).model_validate(values)

    def copy(
        self,
        *,
        include: AbstractSetIntStr | MappingIntStrAny | None = None,
        exclude: AbstractSetIntStr | MappingIntStrAny | None = None,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        values = self.model_dump(
            mode="python",
            include=cast(IncEx | None, include),
            exclude=cast(IncEx | None, exclude),
            round_trip=True,
        )
        if deep:
            values = deepcopy(values)
        if update is not None:
            values.update(update)
        return type(self).model_validate(values)


class DependencySpec(StrictSpecModel):
    kind: DependencyKind
    name: str = Field(min_length=1)
    version_spec: str = ""
    locator: str = ""
    optional: bool = False


class RetryPolicy(StrictSpecModel):
    max_attempts: int = Field(default=1, ge=1, le=10)
    retryable_error_codes: tuple[str, ...] = ()


class FallbackPolicy(StrictSpecModel):
    allowed: bool = False
    allowed_backends: tuple[str, ...] = ()


# Codecs whose Python-side type is not already one of signature.py's natively
# JSON-safe annotations (primitives, tuples/sequences/mappings thereof,
# Literal, supported Enum/OperationParameterModel types). An Agent-supplied
# parameter using one of these must supply its own closed schema; JSON and
# PATH parameters already have a schema signature.py can derive on its own.
# DIRECTORY, like PATH, derives its Agent schema as a plain JSON string and
# never declares a json_schema, so it is not a member here either; the
# ``elif self.json_schema`` branch in ParameterBindingSpec enforces that.
_SCHEMA_REQUIRED_CODECS = frozenset(
    {
        ParameterCodec.LEGACY_GENOME,
        ParameterCodec.LEGACY_DATA,
        ParameterCodec.NUMPY_ARTIFACT,
        ParameterCodec.DATACLASS,
    }
)
#: ``LEGACY_RESULT`` is absent by design (Ruling 2, Decision 004 approval
#: 2026-08-14): the codec's decoder validates the payload against the L1
#: ``OrganelleResult`` model itself at decode time - a frozen Pydantic model
#: is a strictly stronger contract than any hand-declared json_schema, and
#: the result envelope's genuinely-open ``metrics``/``findings`` maps cannot
#: be honestly described by a deeply-closed schema anyway.


_JSON_SCHEMA_CONTRACT_KEYS = frozenset({"type", "$ref", "enum", "const", "anyOf", "oneOf", "allOf"})


def _is_unconstrained_json_schema(schema: Mapping[str, object]) -> bool:
    """Whether *schema* fails to be a deeply, explicitly closed JSON Schema.

    A closed schema needs at least one real constraint key, and every
    object-shaped subschema — at any nesting depth reachable through
    ``items``/``contains``/``additionalProperties``, ``anyOf``/``oneOf``/
    ``allOf``/``prefixItems``, or ``properties``/``patternProperties``/
    ``$defs`` — must set ``additionalProperties`` to exactly ``False``.
    Omitting the key (JSON Schema's own default is "open") and setting it to
    ``True`` are both rejected identically.
    """
    if not schema or not (_JSON_SCHEMA_CONTRACT_KEYS & schema.keys()):
        return True
    if _looks_like_object_schema(schema) and schema.get("additionalProperties") is not False:
        return True
    for key in ("items", "contains", "additionalProperties"):
        nested = schema.get(key)
        if isinstance(nested, Mapping) and _is_unconstrained_json_schema(
            cast(Mapping[str, object], nested)
        ):
            return True
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        nested = schema.get(key)
        if isinstance(nested, Sequence) and any(
            isinstance(item, Mapping)
            and _is_unconstrained_json_schema(cast(Mapping[str, object], item))
            for item in cast(Sequence[object], nested)
        ):
            return True
    for key in ("properties", "patternProperties", "$defs"):
        nested = schema.get(key)
        if isinstance(nested, Mapping) and any(
            isinstance(item, Mapping)
            and _is_unconstrained_json_schema(cast(Mapping[str, object], item))
            for item in cast(Mapping[str, object], nested).values()
        ):
            return True
    return False


def _looks_like_object_schema(schema: Mapping[str, object]) -> bool:
    return schema.get("type") == "object" or "properties" in schema


def _reject_any_json_schema_ref(node: object) -> None:
    """Reject ``$ref`` anywhere in a declared parameter schema - local or remote.

    Runs before any jsonschema machinery ever sees the schema. Two independent
    reasons, not one:

    * A remote ``$ref`` (anything not starting with ``#``) could trigger a
      network fetch when the schema is later used to validate a value.
    * A *local* ``$ref`` is not safe either: python_binding.py embeds this
      schema via ``json_schema_extra`` into one field of a larger, dynamically
      built Agent parameter model. Pydantic's own schema generator tracks
      ``$defs`` for that whole model, not for this fragment in isolation, so
      "#/$defs/x" here does not resolve against this schema's own sibling
      ``$defs`` - it resolves (or fails to) against the outer model's. v1 does
      not implement the rebasing that would make a local ``$ref`` safe, so
      every ``$ref`` is rejected outright rather than only the remote ones.
    """
    if isinstance(node, Mapping):
        mapping_node = cast(Mapping[str, object], node)
        reference = mapping_node.get("$ref")
        if isinstance(reference, str):
            raise ValueError(
                f"parameter json_schema must not use $ref (v1 does not implement safe "
                f"embedding/rebasing of references): found {reference!r}"
            )
        for value in mapping_node.values():
            _reject_any_json_schema_ref(value)
    elif isinstance(node, Sequence) and not isinstance(node, str):
        for item in cast(Sequence[object], node):
            _reject_any_json_schema_ref(item)


def _validate_parameter_json_schema(
    name: str, codec: "ParameterCodec", schema: Mapping[str, object]
) -> None:
    """Validate *schema* is a valid, ref-free, deeply closed Draft 2020-12 schema.

    ``jsonschema`` is imported lazily here, not at module level: its optional
    RFC 3987 format-checking support does real file I/O (loading a bundled
    grammar) as a side effect of import, which would make importing
    ``organelleverse.operations`` no longer discovery-safe.
    """
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

    _reject_any_json_schema_ref(schema)
    try:
        Draft202012Validator.check_schema(dict(schema))
    except SchemaError as error:
        raise ValueError(
            f"parameter {name!r} json_schema is not a valid Draft 2020-12 schema: {error.message}"
        ) from error
    if _is_unconstrained_json_schema(schema):
        raise ValueError(
            f"parameter {name!r} uses codec {codec.value!r}, which is not "
            "already JSON-safe, and must declare a nonempty, deeply closed json_schema "
            "(every object level needs an explicit additionalProperties: false)"
        )


class ParameterBindingSpec(StrictSpecModel):
    """One named parameter's codec, source, and (for unsupported types) schema."""

    name: str = Field(min_length=1)
    codec: ParameterCodec = ParameterCodec.JSON
    source: ParameterSource = ParameterSource.AGENT
    json_schema: dict[str, JsonValue] = Field(default_factory=dict)
    target_type_locator: str = ""
    path_role: Literal["input", "output"] | None = None

    @model_validator(mode="after")
    def validate_binding(self) -> "ParameterBindingSpec":
        if self.codec is ParameterCodec.DATACLASS and not self.target_type_locator.strip():
            raise ValueError("dataclass parameter codec requires a target_type_locator")
        if self.codec is not ParameterCodec.DATACLASS and self.target_type_locator:
            raise ValueError("target_type_locator is only meaningful for the dataclass codec")
        if self.codec is ParameterCodec.DIRECTORY and self.path_role is None:
            raise ValueError(
                "directory parameter codec requires a path_role "
                "('input' for a pre-existing directory tree, 'output' for a "
                "write destination that may not exist yet)"
            )
        if (
            self.codec not in (ParameterCodec.DIRECTORY, ParameterCodec.PATH)
            and self.path_role is not None
        ):
            raise ValueError("path_role is only meaningful for the directory and path codecs")
        if self.source is ParameterSource.AGENT and self.codec in _SCHEMA_REQUIRED_CODECS:
            _validate_parameter_json_schema(self.name, self.codec, self.json_schema)
        elif self.json_schema:
            self._validate_presentation_json_schema()
        return self

    def _validate_presentation_json_schema(self) -> None:
        """Reject a declared ``json_schema`` on an already-JSON-safe codec.

        v1 rule, kept exactly: JSON, PATH and DIRECTORY parameters derive
        their Agent schema from the codec alone, so a declared schema would
        be silently ignored. The v2 plugin binding model overrides this hook
        to *consume* such a schema as presentation metadata instead.
        """
        raise ValueError(
            f"parameter {self.name!r} uses codec {self.codec.value!r}, which never "
            "consumes a declared json_schema; a non-empty one would be silently ignored"
        )


class PythonBindingSpec(StrictSpecModel):
    """How an OperationSpec's original Python callable is invoked and its result encoded."""

    argument_mode: ArgumentMode = ArgumentMode.CANONICAL_CORE
    core_input_parameter: str | None = None
    parameters: tuple[ParameterBindingSpec, ...] = ()
    result_codec: ResultCodec = ResultCodec.CANONICAL
    result_key: str | None = None

    @model_validator(mode="after")
    def validate_binding(self) -> "PythonBindingSpec":
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"duplicate parameter binding names: {duplicates}")
        if self.result_codec is ResultCodec.JSON_METRIC:
            if self.result_key is None or not self.result_key.strip():
                raise ValueError("json_metric result codec requires a nonblank result_key")
        elif self.result_key is not None:
            raise ValueError(
                f"result_key is only meaningful for the json_metric codec, not "
                f"{self.result_codec.value!r}"
            )
        if self.argument_mode is ArgumentMode.CANONICAL_CORE:
            if self.parameters:
                raise ValueError("canonical_core binding does not declare named parameters")
            if self.core_input_parameter is not None:
                raise ValueError("canonical_core binding does not use core_input_parameter")
        return self


class RewriteSpec(StrictSpecModel):
    """Declaration that a capability rewrites an original implementation.

    The verification runner invokes the original callable and compares its
    output to the current implementation's output, using the declared method
    and tolerance.  Known differences must be listed explicitly so reviewers
    can audit intentional deviations.
    """

    original: str = Field(pattern=r"^[a-zA-Z_][\w.]*:[a-zA-Z_]\w*$")
    method: Literal["exact", "numeric_tolerance"]
    tolerance: float | None = None
    known_differences: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_rewrite(self) -> "RewriteSpec":
        if self.method == "numeric_tolerance":
            if self.tolerance is None or not math.isfinite(self.tolerance) or self.tolerance <= 0:
                raise ValueError("numeric_tolerance rewrite requires a finite, positive tolerance")
        elif self.tolerance is not None:
            raise ValueError(
                f"tolerance is only meaningful for numeric_tolerance rewrite, not {self.method!r}"
            )
        return self


class OperationSpec(StrictSpecModel):
    operation_id: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    contract_version: str = Field(pattern=r"^[0-9]+\.[0-9]+$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=20)
    keywords: tuple[str, ...]
    execution_mode: ExecutionMode
    stage: OperationStage
    input_kind: CoreKind
    output_kind: CoreKind
    #: The core input is a SEQUENCE of the input_kind's objects (Ruling 3,
    #: Decision 004 approval 2026-08-14): the first positional parameter is
    #: list[T]/Sequence[T]/tuple[T, ...] for T the declared core type, and
    #: the Agent JSON input is an array of core-object payloads, each
    #: validated against T's L1 schema before the callable runs.
    input_sequence: bool = False
    organelle_types: tuple[Literal["mitochondrion", "plastid"], ...] = ()
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    callable_locator: str | None = Field(default=None, pattern=r"^[a-zA-Z_][\w.]*:[a-zA-Z_]\w*$")
    binding: PythonBindingSpec = PythonBindingSpec()
    dependencies: tuple[DependencySpec, ...] = ()
    side_effects: tuple[SideEffect, ...] = ()
    deterministic: bool = True
    deterministic_reason: str = ""
    rewrite: RewriteSpec | None = None
    idempotent: bool = True
    cacheable: bool = False
    retry: RetryPolicy = RetryPolicy()
    fallback: FallbackPolicy = FallbackPolicy()
    references: tuple[str, ...] = ()

    @field_validator("references")
    @classmethod
    def validate_references(cls, references: tuple[str, ...]) -> tuple[str, ...]:
        if any(not reference.strip() for reference in references):
            raise ValueError("reference entries must not be blank")
        return references

    @field_validator("title", "description")
    @classmethod
    def validate_no_surrounding_whitespace(cls, value: str, info: ValidationInfo) -> str:
        if value.strip() != value:
            raise ValueError(f"{info.field_name} must not have leading or trailing whitespace")
        return value

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, keywords: tuple[str, ...]) -> tuple[str, ...]:
        if not (3 <= len(keywords) <= 8):
            raise ValueError("keywords must declare between 3 and 8 entries")
        if len(keywords) != len(set(keywords)):
            duplicates = sorted({keyword for keyword in keywords if keywords.count(keyword) > 1})
            raise ValueError(f"keywords must not repeat: {duplicates}")
        if tuple(sorted(keywords)) != keywords:
            raise ValueError("keywords must already be sorted")
        invalid = [keyword for keyword in keywords if not _KEYWORD_PATTERN.match(keyword)]
        if invalid:
            raise ValueError(f"keywords must match {_KEYWORD_PATTERN.pattern!r}: {invalid}")
        return keywords

    @model_validator(mode="after")
    def validate_input_sequence(self) -> "OperationSpec":
        if self.input_sequence and self.input_kind is CoreKind.NONE:
            raise ValueError("input_sequence requires a core input kind")
        return self

    @model_validator(mode="after")
    def validate_contract(self) -> "OperationSpec":
        if self.stage is OperationStage.READ:
            if self.input_kind is not CoreKind.NONE or self.output_kind not in {
                CoreKind.GENOME,
                CoreKind.DATA,
            }:
                raise ValueError("read operation requires no core input and Genome/Data output")
        elif self.stage is OperationStage.TRANSFORM:
            if self.input_kind not in {CoreKind.GENOME, CoreKind.DATA} or self.output_kind not in {
                CoreKind.GENOME,
                CoreKind.DATA,
            }:
                raise ValueError("transform operation requires Genome/Data input and output")
        elif self.stage is OperationStage.ANALYZE:
            if (
                self.input_kind not in {CoreKind.NONE, CoreKind.GENOME, CoreKind.DATA}
                or self.output_kind is not CoreKind.RESULT
            ):
                raise ValueError(
                    "analyze operation requires None/Genome/Data input and Result output"
                )
        elif self.input_kind is not CoreKind.RESULT or self.output_kind is not CoreKind.RESULT:
            raise ValueError("consume operation requires Result input and Result output")
        if self.fallback.allowed != bool(self.fallback.allowed_backends):
            raise ValueError("fallback allowed flag and backend list must agree")
        if self.cacheable and not self.deterministic:
            raise ValueError("cacheable operation must be deterministic")
        if self.cacheable and any(
            effect in {SideEffect.NETWORK, SideEffect.WRITE_FILES} for effect in self.side_effects
        ):
            raise ValueError("cacheable operation cannot declare network or write side effects")
        if self.output_kind is CoreKind.DATA:
            if len(self.output_modalities) != 1:
                raise ValueError(
                    "output_modalities must declare exactly one modality for DATA output"
                )
        elif self.output_modalities:
            raise ValueError("output_modalities requires DATA output")
        if self.input_modalities and self.input_kind is not CoreKind.DATA:
            raise ValueError("input_modalities requires DATA input")
        if self.binding.core_input_parameter is not None and self.input_kind is CoreKind.NONE:
            raise ValueError("core_input_parameter requires a non-None input_kind to convert")
        return self

    @model_validator(mode="after")
    def validate_determinism(self) -> "OperationSpec":
        if not self.deterministic:
            if not self.deterministic_reason or not self.deterministic_reason.strip():
                raise ValueError("non-deterministic operation must declare deterministic_reason")
        elif self.deterministic_reason:
            raise ValueError("deterministic_reason is only meaningful when deterministic is false")
        return self


class PluginParameterBindingSpec(ParameterBindingSpec):
    """A v2 plugin binding parameter.

    Identical to the v1 parameter model except that the already-JSON-safe
    codecs (JSON, PATH, DIRECTORY) may declare a closed ``json_schema`` that
    form generators and Agent tools consume as presentation metadata (help
    text, ranges, defaults). The schema is validated with the same
    ref-free, deeply-closed rules v1 applies to the schema-required codecs.

    A parameter may declare that it accepts inline content in addition to, or
    instead of, a path. This keeps one parameter for both "paste" and "browse"
    inputs instead of forcing authors to maintain two fields.
    """

    accepts: tuple[Literal["path", "inline"], ...] = ()
    inline_max_bytes: int = Field(default=1_048_576, gt=0)

    @model_validator(mode="after")
    def validate_accepts(self) -> "PluginParameterBindingSpec":
        explicitly_declared = "accepts" in self.model_fields_set
        if explicitly_declared and not self.accepts and self.codec is not ParameterCodec.JSON:
            raise ValueError(
                f"parameter {self.name!r} must declare at least one accepted input form"
            )
        effective = self.accepts
        if not explicitly_declared and self.codec in {
            ParameterCodec.PATH,
            ParameterCodec.DIRECTORY,
        }:
            effective = ("path",)
        seen = set(effective)
        if not seen.issubset({"path", "inline"}):
            raise ValueError(
                f"parameter {self.name!r} accepts only 'path' or 'inline', got {sorted(seen)}"
            )
        allowed_by_codec = {
            ParameterCodec.JSON: {"inline"},
            ParameterCodec.PATH: {"path", "inline"},
            ParameterCodec.DIRECTORY: {"path"},
        }.get(self.codec, set())
        if not seen.issubset(allowed_by_codec):
            raise ValueError(
                f"parameter {self.name!r} declares input forms {sorted(seen)} that codec "
                f"{self.codec.value!r} cannot carry"
            )
        canonical = tuple(item for item in ("path", "inline") if item in seen)
        object.__setattr__(self, "accepts", canonical)
        if "inline" in seen:
            if self.codec not in (ParameterCodec.JSON, ParameterCodec.PATH):
                raise ValueError(
                    f"parameter {self.name!r} declares inline acceptance but codec "
                    f"{self.codec.value!r} cannot carry inline content"
                )
            if self.source is ParameterSource.ENVIRONMENT:
                raise ValueError(
                    f"parameter {self.name!r} cannot accept inline content when sourced from "
                    "the environment"
                )
            if self.inline_max_bytes <= 0:
                raise ValueError(f"parameter {self.name!r} inline_max_bytes must be positive")
        if self.codec in {ParameterCodec.PATH, ParameterCodec.DIRECTORY} and (
            "default" in self.json_schema
        ):
            raise ValueError(
                f"parameter {self.name!r} path/directory inputs may not declare defaults"
            )
        return self

    def _validate_presentation_json_schema(self) -> None:
        try:
            _validate_parameter_json_schema(self.name, self.codec, self.json_schema)
        except ValueError as error:
            raise ValueError(
                f"v2 parameter {self.name!r} declares a presentation json_schema; it must "
                f"be valid, ref-free, and deeply closed: {error}"
            ) from error


class PathParameterValue(StrictSpecModel):
    """A path value anchored to a specific execution context."""

    kind: Literal["path"] = "path"
    value: str = Field(min_length=1)
    context: ExecutionContext = ExecutionContext.HOST


class InlineParameterValue(StrictSpecModel):
    """Inline user-supplied content with a content hash for provenance."""

    kind: Literal["inline"] = "inline"
    value: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: str = Field(min_length=1)
    encoding: Literal["text", "base64"] = "text"


ParameterValue = PathParameterValue | InlineParameterValue


class PluginBindingSpec(PythonBindingSpec):
    """The v2 plugin binding: v1 rules plus presentation-schema parameters."""

    # Pydantic revalidates subclass field types; narrowing to the plugin
    # parameter model is what permits presentation schemas in v2 only.
    argument_mode: ArgumentMode = ArgumentMode.PLUGIN_PROTOCOL
    parameters: tuple[PluginParameterBindingSpec, ...] = ()  # pyright: ignore[reportIncompatibleVariableOverride]

    @model_validator(mode="after")
    def validate_plugin_protocol(self) -> "PluginBindingSpec":
        if self.argument_mode is not ArgumentMode.PLUGIN_PROTOCOL:
            raise ValueError("v2 plugin binding requires argument_mode 'plugin_protocol'")
        return self


class PluginOutput(StrictSpecModel):
    """One named plugin output port bound to a destination parameter."""

    name: str = Field(min_length=2, pattern=r"^[a-z][a-z0-9_]*$")
    kind: Literal["file", "directory"]
    parameter: str = Field(min_length=1)
    description: str = Field(min_length=1)


class PluginOptimization(StrictSpecModel):
    """An explicit, declared optimization experiment for a plugin."""

    score_locator: str = Field(pattern=r"^[a-zA-Z_][\w.]*:[a-zA-Z_]\w*$")
    parameters: tuple[str, ...] = Field(min_length=1)
    max_trials: int = Field(ge=1, le=256)
    parallelism: int = Field(ge=1, le=32)
    direction: Literal["maximize", "minimize"] = "maximize"

    @field_validator("parameters")
    @classmethod
    def validate_parameter_names(cls, parameters: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name.strip() for name in parameters):
            raise ValueError("optimization parameter names must not be blank")
        if len(parameters) != len(set(parameters)):
            raise ValueError("optimization parameter names must be unique")
        return parameters

    @model_validator(mode="after")
    def validate_parallelism(self) -> "PluginOptimization":
        if self.parallelism > self.max_trials:
            raise ValueError("optimization parallelism must not exceed max_trials")
        return self


class PluginOperationSpec(OperationSpec):
    """The v2 plugin contract: the v1 OperationSpec surface plus plugin ports."""

    binding: PluginBindingSpec = PluginBindingSpec()  # pyright: ignore[reportIncompatibleVariableOverride] - v2 narrows the binding to the plugin binding model
    outputs: tuple[PluginOutput, ...] = ()
    optimization: PluginOptimization | None = None
    execution_context: ExecutionContext = ExecutionContext.HOST
    progress_source: ProgressSource = ProgressSource.NONE

    @model_validator(mode="after")
    def validate_plugin_contract(self) -> "PluginOperationSpec":
        names = [output.name for output in self.outputs]
        if len(names) != len(set(names)):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"plugin output names must be unique: {duplicates}")
        references = [output.parameter for output in self.outputs]
        if len(references) != len(set(references)):
            duplicates = sorted({name for name in references if references.count(name) > 1})
            raise ValueError(f"plugin output parameter references must be unique: {duplicates}")

        bindings_by_name = {parameter.name: parameter for parameter in self.binding.parameters}
        for output in self.outputs:
            binding = bindings_by_name.get(output.parameter)
            if (
                binding is None
                or binding.source is not ParameterSource.AGENT
                or binding.codec not in {ParameterCodec.PATH, ParameterCodec.DIRECTORY}
            ):
                raise ValueError(
                    f"output parameter {output.parameter!r} must reference one "
                    "Agent-supplied path or directory binding parameter"
                )
            required_codec = (
                ParameterCodec.DIRECTORY if output.kind == "directory" else ParameterCodec.PATH
            )
            if binding.codec is not required_codec or binding.path_role != "output":
                raise ValueError(
                    f"{output.kind} output {output.name!r} requires a "
                    f"{required_codec.value} binding parameter with path_role 'output'"
                )
            if binding.accepts != ("path",):
                raise ValueError(
                    f"output parameter {output.parameter!r} must accept path transport only"
                )

        output_role_directories = {
            parameter.name
            for parameter in self.binding.parameters
            if parameter.codec is ParameterCodec.DIRECTORY and parameter.path_role == "output"
        }
        unreferenced = sorted(output_role_directories - set(references))
        if unreferenced:
            raise ValueError(
                "every output-role directory parameter must be referenced by exactly "
                f"one declared output: {unreferenced}"
            )

        if self.optimization is not None:
            agent_parameters = {
                parameter.name
                for parameter in self.binding.parameters
                if parameter.source is ParameterSource.AGENT
            }
            unknown = sorted(set(self.optimization.parameters) - agent_parameters)
            if unknown:
                raise ValueError(
                    f"optimization parameters must name Agent-supplied binding "
                    f"parameters: {unknown}"
                )
        return self
