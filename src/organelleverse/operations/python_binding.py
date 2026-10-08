"""Bind a capability bundle's declared binding to its real Python callable.

Python and Agent paths share one ``BoundOperation``: ``bind_python_capability``
never creates a second implementation. For ``canonical_core`` binding it is a
thin wrapper over the existing, heavily-tested signature-derivation and
registry-invocation mechanism used by all 20 released operations. For
``named_parameters`` binding it builds an Agent-facing parameter model from
the declared ``ParameterBindingSpec`` list and calls the *original* callable
through ``inspect.Signature.bind`` — reordering or renaming nothing the
implementation itself does not already accept — then encodes its direct
return value through :func:`organelleverse.capabilities.codecs.encode_capability_result`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
import math
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    Never,
    Protocol,
    TypeAlias,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import uuid4

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    JsonValue,
    WithJsonSchema,
    create_model,
)

from organelleverse.capabilities.codecs import CapabilityExecutionContext, encode_capability_result
from organelleverse.capabilities.models import CapabilityBundle, ImplementationKind
from organelleverse.capabilities.worker_contracts import WorkerParameter
from organelleverse.core.base import StrictFrozenModel
from organelleverse.core.errors import OrganelleContractError, OrganelleInputError
from organelleverse.core.result import OrganelleResult

from .data_contracts import DataContract
from .registry import BoundOperation, CoreObject, InvocationStrategy
from .signature import (
    CORE_TYPES,
    OperationSignature,
    StrictParameters,
    _is_supported_json_annotation,  # pyright: ignore[reportPrivateUsage]
    derive_operation_signature,
)
from .spec import (
    ArgumentMode,
    CoreKind,
    InlineParameterValue,
    OperationSpec,
    ParameterBindingSpec,
    ParameterCodec,
    ParameterSource,
    PathParameterValue,
    PythonBindingSpec,
    ResultCodec,
)


def _binding_error(message: str, **details: object) -> OrganelleContractError:
    return OrganelleContractError(code="capability.binding_invalid", message=message, details=details)


_UNIMPLEMENTED_PARAMETER_CODECS = frozenset(
    {ParameterCodec.LEGACY_GENOME, ParameterCodec.LEGACY_DATA}
)
"""Codecs v1 declares in the schema but never implements a decoder for.

A ``ParameterBindingSpec`` may still declare one of these (a parsed bundle
stays inspectable), but ``bind_python_capability`` must reject it immediately
- never defer the failure to the first invocation. ``LEGACY_RESULT`` left
this set with its decoder (Ruling 2, Decision 004 approval 2026-08-14): the
agent-facing field IS the L1 ``OrganelleResult`` model, so Pydantic's own
strict validation reconstructs the frozen result before the callable runs.
"""


EnvironmentParameterProvider: TypeAlias = Callable[
    [OperationSpec, "tuple[ParameterBindingSpec, ...]"], Mapping[str, object]
]
"""Resolves an operation's environment-sourced parameters at invocation time.

Called with the operation's spec and exactly its ENVIRONMENT-sourced
``ParameterBindingSpec`` entries; must return a mapping covering every one of
those names (extra or missing names are both rejected before the original
callable ever runs). A provider must only resolve parameter *values* — e.g.
looking up a configured executor object — never invoke L6 runtime machinery
itself; that remains the execution environment's job, not the provider's.
"""


def bind_python_capability(
    bundle: CapabilityBundle,
    implementation: Callable[..., object],
    frozen_schema: Mapping[str, object] | None,
    *,
    environment_provider: EnvironmentParameterProvider | None = None,
    data_contracts: tuple[DataContract, ...] = (),
) -> BoundOperation:
    """Bind *implementation* per ``bundle.contract.binding`` and return a BoundOperation.

    ``frozen_schema`` is a verified bundle's frozen parameter schema (recorded
    once, with import allowed, during Verification — a future phase this
    round does not implement). When supplied, it is compared against a live
    schema derived from *implementation* right now; any difference raises
    ``capability.schema_drift`` and *implementation* is never called. Passing
    ``None`` skips the comparison entirely and is permitted only for
    explicitly unverified test/verification-stage use — a real admission
    pathway must never invoke this with ``None``.

    ``environment_provider`` supplies values for this operation's
    ENVIRONMENT-sourced parameters at invocation time. Those parameters are
    never part of the Agent-facing schema — an Agent cannot submit them under
    any name — so a value can only ever come from the provider.

    A composite bundle (``bundle.capability.implementation ==
    ImplementationKind.COMPOSITE``) has no single Python callable to bind —
    per admission design §5.3, v1 only parses and statically validates its
    declarative step graph and never executes it — so this raises
    ``capability.implementation_not_supported`` immediately, before
    *implementation* is ever touched.
    """
    if bundle.capability.implementation is ImplementationKind.COMPOSITE:
        raise OrganelleContractError(
            code="capability.implementation_not_supported",
            message=(
                f"{bundle.contract.operation_id} is a composite capability; v1 parses and "
                "statically validates its step graph but never executes it"
            ),
            details={"operation_id": bundle.contract.operation_id},
        )
    spec = bundle.contract
    if spec.binding.argument_mode is ArgumentMode.CANONICAL_CORE:
        bound = _bind_canonical_core(spec, implementation)
    else:
        bound = _bind_named_parameters(spec, implementation, environment_provider)
    if frozen_schema is not None:
        _check_schema_drift(spec, bound.signature.parameter_model, frozen_schema)
    return dataclasses.replace(bound, data_contracts=data_contracts)


def _worker_provider_required(spec: OperationSpec, message: str, **details: object) -> Never:
    raise OrganelleContractError(
        code="capability.execution_provider_required",
        message=message,
        details={"operation_id": spec.operation_id, **details},
    )


def _validate_worker_contract_spec(
    spec: OperationSpec,
    frozen_schema: Mapping[str, object] | None = None,
) -> None:
    from .output_boundary import capability_final_write_parameters, is_writer_spec

    binding = spec.binding
    if frozen_schema is None:
        properties: dict[str, object] = {item.name: {} for item in binding.parameters}
        schema: Mapping[str, object] = {
            "type": "object",
            "properties": properties,
        }
    else:
        schema = frozen_schema
    forbidden = capability_final_write_parameters(spec, schema)
    if forbidden and binding.argument_mode is not ArgumentMode.PLUGIN_PROTOCOL:
        _worker_provider_required(
            spec,
            "controlled-worker capability schema declares final-write parameters",
            parameters=sorted(forbidden),
        )
    if is_writer_spec(spec):
        _worker_provider_required(
            spec,
            "bundle-local writer capabilities require an execution provider",
        )
    if binding.argument_mode not in {
        ArgumentMode.NAMED_PARAMETERS,
        ArgumentMode.PLUGIN_PROTOCOL,
    }:
        _worker_provider_required(
            spec,
            "bundle-local controlled workers support named_parameters or plugin_protocol bindings only",
        )
    if binding.argument_mode is not ArgumentMode.PLUGIN_PROTOCOL:
        if binding.result_codec in {ResultCodec.CANONICAL, ResultCodec.ARTIFACT}:
            _worker_provider_required(
                spec,
                f"result codec {binding.result_codec.value!r} requires an execution provider",
                result_codec=binding.result_codec.value,
            )
        if (
            binding.result_codec in {ResultCodec.JSON_METRIC, ResultCodec.LEGACY_RESULT}
            and spec.output_kind is not CoreKind.RESULT
        ):
            _worker_provider_required(
                spec,
                f"result codec {binding.result_codec.value!r} requires Result output",
                result_codec=binding.result_codec.value,
                output_kind=spec.output_kind.value,
            )
        if binding.result_codec is ResultCodec.CANONICAL_JSON and spec.output_kind not in {
            CoreKind.GENOME,
            CoreKind.DATA,
            CoreKind.RESULT,
        }:
            _worker_provider_required(
                spec,
                "canonical_json requires a declared Genome, Data, or Result output",
                result_codec=binding.result_codec.value,
                output_kind=spec.output_kind.value,
            )
    environment_parameters = tuple(
        item.name
        for item in binding.parameters
        if item.source is ParameterSource.ENVIRONMENT
    )
    if environment_parameters:
        _worker_provider_required(
            spec,
            "environment-sourced parameters require an execution provider",
            parameters=list(environment_parameters),
        )
    supported_codecs = (
        {ParameterCodec.JSON, ParameterCodec.PATH, ParameterCodec.DIRECTORY}
        if binding.argument_mode is ArgumentMode.PLUGIN_PROTOCOL
        else {ParameterCodec.JSON, ParameterCodec.PATH, ParameterCodec.LEGACY_RESULT}
    )
    unsupported_codecs = tuple(
        item.name
        for item in binding.parameters
        if item.codec not in supported_codecs
    )
    if unsupported_codecs:
        _worker_provider_required(
            spec,
            (
                "plugin-protocol controlled workers support JSON, path, and directory parameters only"
                if binding.argument_mode is ArgumentMode.PLUGIN_PROTOCOL
                else "bundle-local controlled workers support JSON and path parameters only"
            ),
            parameters=list(unsupported_codecs),
        )


def validate_worker_contract(
    bundle: CapabilityBundle,
    frozen_schema: Mapping[str, object] | None = None,
) -> None:
    """Reject statically unsupported worker contracts before any worker starts."""
    _validate_worker_contract_spec(bundle.contract, frozen_schema)


def _validate_worker_binding_mode(
    spec: OperationSpec,
    parameters: tuple[WorkerParameter, ...],
) -> None:
    _validate_worker_contract_spec(spec)
    names = [item.name for item in parameters]
    if len(names) != len(set(names)):
        raise _binding_error(
            "worker signature contains duplicate parameter names",
            operation_id=spec.operation_id,
        )
    if spec.binding.argument_mode is ArgumentMode.PLUGIN_PROTOCOL:
        observed = tuple((item.name, item.annotation) for item in parameters)
        expected = (
            ("inputs", "dict"),
            ("outputs", "dict"),
            ("parameters", "dict"),
            ("context", "plugin_context"),
        )
        if observed != expected:
            raise _binding_error(
                "plugin worker signature does not match the verified protocol",
                operation_id=spec.operation_id,
            )


def _worker_agent_annotation(
    spec: OperationSpec,
    declared: ParameterBindingSpec,
    parameter: WorkerParameter,
) -> object:
    token = parameter.annotation
    if token == "unsupported":
        _worker_provider_required(
            spec,
            f"parameter {parameter.name!r} annotation is outside the closed worker token language",
            parameter=parameter.name,
        )
    if declared.codec is ParameterCodec.PATH:
        if token not in {"str", "str_or_none", "path", "path_or_none"}:
            raise _binding_error(
                f"parameter {parameter.name!r} uses path codec with incompatible "
                f"worker annotation {token!r}",
                operation_id=spec.operation_id,
            )
        return str | None if token in {"str_or_none", "path_or_none"} else str
    if token in {"path", "path_or_none"}:
        raise _binding_error(
            f"parameter {parameter.name!r} needs the path codec for its worker annotation",
            operation_id=spec.operation_id,
        )
    annotations: dict[str, object] = {
        "str": str,
        "str_or_none": str | None,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "json": JsonValue,
    }
    return annotations[token]


def _worker_parameter_model(
    bundle: CapabilityBundle,
    parameters: tuple[WorkerParameter, ...],
) -> type[BaseModel]:
    spec = bundle.contract
    _validate_worker_binding_mode(spec, parameters)
    if spec.binding.argument_mode is ArgumentMode.PLUGIN_PROTOCOL:
        return _plugin_worker_parameter_model(spec)
    by_name = {item.name: item for item in parameters}
    declared_names = {item.name for item in spec.binding.parameters}
    core_name = spec.binding.core_input_parameter
    if core_name is not None:
        if core_name in declared_names or core_name not in by_name:
            raise _binding_error(
                "worker core_input_parameter is absent or multiply bound",
                operation_id=spec.operation_id,
                parameter=core_name,
            )
        core_token = by_name[core_name]
        if core_token.annotation not in {"dict", "json"}:
            _worker_provider_required(
                spec,
                "worker core input must use a dict or json signature token",
                parameter=core_name,
            )
    covered = set(declared_names)
    if core_name is not None:
        covered.add(core_name)
    targeted_positional = tuple(
        name
        for name in sorted(covered)
        if (parameter := by_name.get(name)) is not None
        and parameter.kind == "positional_only"
    )
    if targeted_positional:
        _worker_provider_required(
            spec,
            "named_parameters worker binding cannot safely target positional-only parameters",
            parameters=list(targeted_positional),
        )
    for parameter in parameters:
        if parameter.name not in covered and parameter.required:
            raise _binding_error(
                f"required worker parameter {parameter.name!r} is not covered by the binding",
                operation_id=spec.operation_id,
            )

    fields: dict[str, Any] = {}
    for declared in spec.binding.parameters:
        parameter = by_name.get(declared.name)
        if parameter is None:
            raise _binding_error(
                f"declared parameter {declared.name!r} is absent from the worker signature",
                operation_id=spec.operation_id,
            )
        annotation = _worker_agent_annotation(spec, declared, parameter)
        default: object = ... if parameter.required else parameter.default
        fields[declared.name] = (annotation, default)
    return cast(
        "type[BaseModel]",
        create_model(
            f"{spec.operation_id.replace('.', '_')}_WorkerParameters",
            __base__=StrictParameters,
            **fields,
        ),
    )


def _plugin_worker_parameter_model(spec: OperationSpec) -> type[BaseModel]:
    """Build the Agent schema from the v2 declaration, not ``run``'s ports."""

    output_parameters = {output.parameter for output in getattr(spec, "outputs", ())}
    fields: dict[str, Any] = {}
    for declared in spec.binding.parameters:
        if declared.source is not ParameterSource.AGENT:
            _worker_provider_required(
                spec,
                "plugin protocol parameters must be Agent-supplied",
                parameter=declared.name,
            )
        if declared.name in output_parameters:
            continue
        if declared.codec not in {
            ParameterCodec.JSON,
            ParameterCodec.PATH,
            ParameterCodec.DIRECTORY,
        }:
            _worker_provider_required(
                spec,
                f"plugin parameter codec {declared.codec.value!r} requires an execution provider",
                parameter=declared.name,
            )
        schema = declared.json_schema
        default: object = schema.get("default", ...)
        accepts: set[str] = set(
            cast(tuple[str, ...], getattr(declared, "accepts", ("path",)))
        )
        if declared.codec in {ParameterCodec.PATH, ParameterCodec.DIRECTORY}:
            if "inline" in accepts and "path" in accepts:
                annotation: object = str | PathParameterValue | InlineParameterValue
            elif "inline" in accepts:
                annotation = InlineParameterValue
            else:
                annotation = str | PathParameterValue
        elif "inline" in accepts:
            annotation = Any | InlineParameterValue
        else:
            annotation = Any
        fields[declared.name] = (
            Annotated[
                annotation,
                BeforeValidator(_plugin_transport_decoder(declared)),
                _plugin_schema_validator(schema),
                WithJsonSchema(_plugin_transport_schema(declared)),
            ],
            default,
        )
    return cast(
        "type[BaseModel]",
        create_model(
            f"{spec.operation_id.replace('.', '_')}_PluginParameters",
            __base__=StrictParameters,
            **fields,
        ),
    )


def _plugin_schema_validator(schema: Mapping[str, JsonValue]) -> AfterValidator:
    """Enforce the same closed JSON Schema that the plugin form exposes."""

    def validate(value: object) -> object:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

        # Tagged path/inline envelopes describe transport, not the scientific
        # value constrained by the field schema. Paths are checked and inline
        # JSON is decoded + schema-validated at the worker boundary.
        if isinstance(value, (PathParameterValue, InlineParameterValue)):
            return value
        validator = cast(_JsonSchemaValidator, Draft202012Validator(dict(schema)))
        try:
            validator.validate(value)
        except JsonSchemaValidationError as error:
            raise ValueError(error.message) from error
        return value

    return AfterValidator(validate)


def _plugin_transport_decoder(
    parameter: ParameterBindingSpec,
) -> Callable[[object], object]:
    """Parse only complete transport envelopes legal for this parameter."""

    accepts: set[str] = set(
        cast(tuple[str, ...], getattr(parameter, "accepts", ()))
    )
    path_keys = frozenset({"kind", "value", "context"})
    inline_keys = frozenset({"kind", "value", "sha256", "format", "encoding"})

    def decode(value: object) -> object:
        original: object = value
        if isinstance(value, Mapping):
            mapping = cast(Mapping[object, object], value)
            kind = mapping.get("kind")
            keys = set(mapping)
            if (
                kind == "path"
                and "path" in accepts
                and parameter.codec in {ParameterCodec.PATH, ParameterCodec.DIRECTORY}
                and {"kind", "value"}.issubset(keys)
                and keys.issubset(path_keys)
            ):
                return PathParameterValue.model_validate(dict(mapping))
            if (
                kind == "inline"
                and "inline" in accepts
                and {"kind", "value", "sha256", "format"}.issubset(keys)
                and keys.issubset(inline_keys)
            ):
                return InlineParameterValue.model_validate(dict(mapping))
        return original

    return decode


def _plugin_transport_schema(parameter: ParameterBindingSpec) -> dict[str, object]:
    """Describe raw scientific values and tagged transport envelopes without conflicts."""

    accepts: set[str] = set(
        cast(tuple[str, ...], getattr(parameter, "accepts", ("path",)))
    )
    branches: list[dict[str, object]] = []
    if parameter.codec is ParameterCodec.JSON or "path" in accepts:
        branches.append(dict(parameter.json_schema))
    if parameter.codec in {ParameterCodec.PATH, ParameterCodec.DIRECTORY} and "path" in accepts:
        branches.append(
            {
                "type": "object",
                "properties": {
                    "kind": {"const": "path"},
                    "value": {"type": "string", "minLength": 1},
                    "context": {
                        "type": "string",
                        "enum": ["host", "wsl", "container", "remote"],
                        "default": "host",
                    },
                },
                "required": ["kind", "value"],
                "additionalProperties": False,
            }
        )
    if "inline" in accepts:
        branches.append(
            {
                "type": "object",
                "properties": {
                    "kind": {"const": "inline"},
                    "value": {"type": "string"},
                    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "format": {"type": "string", "minLength": 1},
                    "encoding": {
                        "type": "string",
                        "enum": ["text", "base64"],
                        "default": "text",
                    },
                },
                "required": ["kind", "value", "sha256", "format"],
                "additionalProperties": False,
            }
        )
    if len(branches) == 1:
        return branches[0]
    projected: dict[str, object] = {"anyOf": branches}
    for key in ("title", "description", "ui_placeholder", "ui:placeholder", "ui_file_ext", "ui:fileExt"):
        if key in parameter.json_schema:
            projected[key] = parameter.json_schema[key]
    return projected


class _JsonSchemaValidator(Protocol):
    def validate(self, value: object) -> None: ...


def worker_parameter_schema(
    bundle: CapabilityBundle,
    parameters: tuple[WorkerParameter, ...],
) -> dict[str, object]:
    """Regenerate the exact Agent schema from closed worker signature tokens."""
    from .schemas import parameter_schema_for_model

    model = _worker_parameter_model(bundle, parameters)
    return parameter_schema_for_model(model)


def bind_worker_capability(
    bundle: CapabilityBundle,
    parameters: tuple[WorkerParameter, ...],
    frozen_schema: Mapping[str, object],
    *,
    invocation_strategy: InvocationStrategy,
) -> BoundOperation:
    """Bind a verified worker signature without importing its implementation."""
    spec = bundle.contract
    validate_worker_contract(bundle, frozen_schema)
    parameter_model = _worker_parameter_model(bundle, parameters)
    _check_schema_drift(spec, parameter_model, frozen_schema)
    core_input_type = (
        CORE_TYPES.get(spec.input_kind)
        if spec.binding.core_input_parameter is not None
        else None
    )

    def unavailable_worker_implementation(*args: object, **kwargs: object) -> CoreObject:
        raise AssertionError("worker implementation must be reached only through its strategy")

    signature = OperationSignature(
        python_signature=inspect.Signature(),
        input_name=spec.binding.core_input_parameter,
        input_type=core_input_type,
        output_type=CORE_TYPES[spec.output_kind],
        parameter_model=parameter_model,
    )
    placeholder = cast("Callable[..., CoreObject]", unavailable_worker_implementation)
    return BoundOperation(
        spec=spec,
        implementation=placeholder,
        function=placeholder,
        signature=signature,
        data_contracts=(),
        invocation_strategy=invocation_strategy,
    )


def _check_schema_drift(
    spec: OperationSpec,
    parameter_model: type[BaseModel],
    frozen_schema: Mapping[str, object],
) -> None:
    """Raise ``capability.schema_drift`` if the live schema no longer matches *frozen_schema*.

    Both schemas are compared as canonical JSON (recursively sorted object
    keys), so re-ordering a schema's properties — semantically identical —
    never false-positives; a real content change (added/removed field,
    changed ``required``, changed ``enum``) always does. This runs after
    binding derives ``parameter_model`` but implementation was never called
    to get there, so drift is caught before the original callable ever runs.
    """
    from .schemas import parameter_schema_for_model

    live_canonical = _canonical_json(parameter_schema_for_model(parameter_model))
    frozen_canonical = _canonical_json(dict(frozen_schema))
    if live_canonical == frozen_canonical:
        return
    raise OrganelleContractError(
        code="capability.schema_drift",
        message=(
            f"{spec.operation_id} live parameter schema no longer matches its frozen, "
            "verified schema"
        ),
        details={
            "operation_id": spec.operation_id,
            "frozen_schema_hash": _sha256_text(frozen_canonical),
            "live_schema_hash": _sha256_text(live_canonical),
        },
    )


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bind_canonical_core(spec: OperationSpec, implementation: Callable[..., object]) -> BoundOperation:
    """Delegate to the existing, already-tested canonical signature mechanism.

    This is exactly what ``OperationRegistry.register`` does internally,
    without inserting the binding into any registry: the same
    ``derive_operation_signature`` call, the same revalidation and output
    checks run by ``BoundOperation.invoke`` — no new logic, no re-implementation.
    """
    signature = derive_operation_signature(implementation, spec)
    # derive_operation_signature has just verified implementation's live return
    # annotation equals CORE_TYPES[spec.output_kind]; the public parameter type
    # stays the deliberately loose Callable[..., object] (a bundle's real
    # callable is arbitrary until bound), so the narrowing is explicit here.
    core_returning = cast("Callable[..., CoreObject]", implementation)
    return BoundOperation(
        spec=spec,
        implementation=core_returning,
        function=core_returning,
        signature=signature,
        data_contracts=(),
    )


@dataclasses.dataclass(frozen=True)
class _PathParameterPlan:
    """How one PATH-codec parameter's Agent-facing field and value must be handled.

    ``target_type`` is what a *submitted* value decodes to before the
    implementation is called; ``optional`` is whether the implementation's
    own annotation names ``None`` as acceptable (e.g. ``Path | None``,
    ``str | Path | None``) - which in turn decides whether the Agent field
    itself must be nullable (``str | None`` instead of plain ``str``) and
    whether a Python-level ``None`` default is even legal. ``multiple`` is
    whether the annotation is a one-argument ``list[...]``/``Sequence[...]``
    of path-likes: the Agent field is then ``list[str]`` and every element
    gets the same existence check and content hash a scalar PATH value gets.
    ``path_kind`` is ``"file"`` for every PATH-codec plan; DIRECTORY-codec
    plans carry ``"directory_input"`` (a pre-existing tree to read - PATH
    parity plus a whole-tree manifest hash) or ``"directory_output"`` (an
    Agent-chosen destination under the designed containment rules), decided
    by the declared ``path_role`` - never by the annotation.
    """

    target_type: type[str] | type[Path]
    optional: bool
    multiple: bool = False
    path_kind: Literal["file", "file_output", "directory_input", "directory_output"] = "file"


def _one_path_as_list(value: object) -> object:
    """A single path string for a list-of-paths parameter means a list of one.

    The Agent schema stays an array; this keeps callers that passed one file to a
    parameter that later became a list working.
    """
    return [value] if isinstance(value, str) else value


def _resolve_path_parameter_plan(
    name: str,
    annotation: object,
    operation_id: str,
    path_kind: Literal["file", "file_output"] = "file",
) -> _PathParameterPlan:
    """Decide whether a PATH-codec parameter's value should decode to str or Path.

    Based on the *implementation's own* annotation, never the Agent-facing
    schema (which is always a JSON string - or ``string | null`` when
    Optional - regardless of this): an exact ``str`` or ``Path`` annotation
    maps directly; a union naming ``str`` needs no conversion at all (the
    implementation already accepts the raw string); a union naming ``Path``
    but not ``str`` converts to ``Path``. A ``None`` arm anywhere in the
    union marks the parameter Optional.

    A one-argument ``list[...]``/``Sequence[...]`` whose element is ``str``,
    ``Path``, or a union of those yields a ``multiple`` plan instead (the
    Agent field is ``list[str]`` and every element is checked and hashed
    individually). A list element may never be ``None`` - an element cannot
    be "no artifact" the way a whole Optional value can.
    """
    if annotation is str:
        return _PathParameterPlan(target_type=str, optional=False, path_kind=path_kind)
    if annotation is Path:
        return _PathParameterPlan(target_type=Path, optional=False, path_kind=path_kind)
    origin = get_origin(annotation)
    if origin in (list, Sequence):
        element_args = get_args(annotation)
        if len(element_args) != 1:
            raise _binding_error(
                f"parameter {name!r} uses codec 'path' but its implementation annotation is "
                "not str, Path, or a union including one of them",
                operation_id=operation_id,
            )
        element = element_args[0]
        element_union_args = get_args(element) or (element,)
        if type(None) in element_union_args:
            raise _binding_error(
                f"parameter {name!r} uses codec 'path' but its list element may be None",
                operation_id=operation_id,
            )
        if str in element_union_args:
            return _PathParameterPlan(target_type=str, optional=False, multiple=True, path_kind=path_kind)
        if Path in element_union_args:
            return _PathParameterPlan(target_type=Path, optional=False, multiple=True, path_kind=path_kind)
        raise _binding_error(
            f"parameter {name!r} uses codec 'path' but its implementation annotation is "
            "not str, Path, or a union including one of them",
            operation_id=operation_id,
        )
    union_args = get_args(annotation)
    optional = type(None) in union_args
    if str in union_args:
        return _PathParameterPlan(target_type=str, optional=optional, path_kind=path_kind)
    if Path in union_args:
        return _PathParameterPlan(target_type=Path, optional=optional, path_kind=path_kind)
    raise _binding_error(
        f"parameter {name!r} uses codec 'path' but its implementation annotation is "
        "not str, Path, or a union including one of them",
        operation_id=operation_id,
    )


def _resolve_directory_parameter_plan(
    name: str,
    annotation: object,
    path_kind: Literal["directory_input", "directory_output"],
    operation_id: str,
) -> _PathParameterPlan:
    """Decide whether a DIRECTORY-codec parameter's value should decode to str or Path.

    Same annotation shapes as a scalar PATH parameter - an exact ``str`` or
    ``Path``, or a union naming one of them, with a ``None`` arm marking the
    parameter Optional - but never a ``list[...]``/``Sequence[...]``: a
    directory parameter names exactly one tree or destination, so the Agent
    field is always a plain JSON string (``string | null`` when Optional).
    The ``path_kind`` comes from the declared ``path_role`` on the
    ``ParameterBindingSpec``, not from anything in the annotation.
    """
    if annotation is str:
        return _PathParameterPlan(target_type=str, optional=False, path_kind=path_kind)
    if annotation is Path:
        return _PathParameterPlan(target_type=Path, optional=False, path_kind=path_kind)
    if get_origin(annotation) in (Union, UnionType):
        union_args = get_args(annotation)
        optional = type(None) in union_args
        if str in union_args:
            return _PathParameterPlan(target_type=str, optional=optional, path_kind=path_kind)
        if Path in union_args:
            return _PathParameterPlan(target_type=Path, optional=optional, path_kind=path_kind)
    raise _binding_error(
        f"parameter {name!r} uses codec 'directory' but its implementation annotation is "
        "not str, Path, or a union including one of them",
        operation_id=operation_id,
    )


def _normalize_path_default(
    default: object, *, plan: _PathParameterPlan, name: str, operation_id: str
) -> object:
    """Normalize a PATH- or DIRECTORY-codec parameter's Python default into its Agent-facing form.

    The Agent-facing field is always ``str`` (or ``str | None`` when
    Optional) - never ``Path`` - so a real ``Path`` default must become a
    string before it ever reaches ``create_model``: left as a raw ``Path``
    object, Pydantic's own ``validate_default=True`` + ``strict=True``
    rejects it the moment this parameter is omitted (i.e. every call that
    relies on the default). ``...`` (no default) and an already-``None``
    default on an Optional parameter both pass through unchanged; anything
    else with no defined normalization is rejected here, at bind time. A
    ``multiple`` (list) plan takes no default at all in v1: the Agent field
    is always a required ``list[str]``.
    """
    codec_name = "path" if plan.path_kind == "file" else "directory"
    if default is ...:
        return default
    if plan.multiple:
        raise _binding_error(
            f"parameter {name!r} uses codec {codec_name!r} with a list annotation; list-valued "
            "PATH parameters take no default in v1 (the Agent field is always a "
            "required list[str])",
            operation_id=operation_id,
        )
    if default is None:
        if not plan.optional:
            raise _binding_error(
                f"parameter {name!r} uses codec {codec_name!r} with a None default but its "
                "implementation annotation does not accept None",
                operation_id=operation_id,
            )
        return None
    if isinstance(default, (str, Path)):
        return str(default)
    raise _binding_error(
        f"parameter {name!r} uses codec {codec_name!r} with an unsupported default value type "
        f"{type(default).__name__}; only str, Path, or (when the annotation accepts it) "
        "None are supported",
        operation_id=operation_id,
    )


_UNADDRESSABLE_BY_KEYWORD = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.VAR_POSITIONAL,
    inspect.Parameter.VAR_KEYWORD,
)


def _validate_binding_completeness(
    spec: OperationSpec,
    python_signature: inspect.Signature,
    binding: PythonBindingSpec,
) -> None:
    """Reject an incomplete or unsupported named_parameters binding at bind time.

    named_parameters binding always calls ``python_signature.bind(**call_kwargs)``:
    every value the implementation receives arrives by keyword. That
    mechanism has no way to *address* a ``POSITIONAL_ONLY`` parameter, a
    ``*args`` (``VAR_POSITIONAL``), or a ``**kwargs`` (``VAR_KEYWORD``)
    parameter by name - so targeting one of those with an agent binding, an
    environment binding, or ``core_input_parameter`` is rejected here,
    rather than left to fail with a bare ``TypeError`` out of
    ``Signature.bind`` the first time someone actually invokes the binding.

    An *untargeted* ``**kwargs``/``*args`` is harmless (it always receives
    zero items through this keyword-only call) and is not rejected merely
    for existing - a real released capability (``plot_ogdraw_map``) uses
    exactly this shape as an optional escape hatch. An untargeted
    ``POSITIONAL_ONLY`` parameter is likewise fine when it has a default;
    without one, it is simply an uncovered required parameter, same as any
    other kind.

    Every remaining parameter without a default must be covered by exactly
    one of: an agent-sourced binding, an environment-sourced binding, or
    ``core_input_parameter`` - never zero (silently unfillable) and never
    more than one (ambiguous which source provides it).
    """
    declared_names = {parameter_binding.name for parameter_binding in binding.parameters}
    core_input_parameter = binding.core_input_parameter
    if core_input_parameter is not None:
        if core_input_parameter not in python_signature.parameters:
            raise _binding_error(
                f"core_input_parameter {core_input_parameter!r} is not a parameter of "
                "the implementation",
                operation_id=spec.operation_id,
            )
        if core_input_parameter in declared_names:
            raise _binding_error(
                f"parameter {core_input_parameter!r} is both core_input_parameter and a "
                "declared named parameter binding",
                operation_id=spec.operation_id,
            )

    covered_names: set[str] = set(declared_names)
    if core_input_parameter is not None:
        covered_names.add(core_input_parameter)

    for name, parameter in python_signature.parameters.items():
        if name in covered_names:
            if parameter.kind in _UNADDRESSABLE_BY_KEYWORD:
                raise _binding_error(
                    f"parameter {name!r} is {parameter.kind.description}, which "
                    "named_parameters binding cannot address (every argument is passed "
                    "to the implementation by keyword)",
                    operation_id=spec.operation_id,
                )
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue  # untargeted *args/**kwargs always receives zero items - harmless
        if parameter.default is inspect.Parameter.empty:
            raise _binding_error(
                f"implementation parameter {name!r} has no default and is covered by "
                "none of: an agent binding, an environment binding, or "
                "core_input_parameter",
                operation_id=spec.operation_id,
            )


def _bind_named_parameters(
    spec: OperationSpec,
    implementation: Callable[..., object],
    environment_provider: EnvironmentParameterProvider | None,
) -> BoundOperation:
    binding = spec.binding
    python_signature = inspect.signature(implementation)
    try:
        hints = get_type_hints(implementation, include_extras=True)
    except (TypeError, ValueError, NameError) as error:
        raise _binding_error(
            "implementation must be inspectable with resolvable annotations",
            operation_id=spec.operation_id,
        ) from error
    _validate_binding_completeness(spec, python_signature, binding)

    agent_bindings: list[ParameterBindingSpec] = []
    environment_bindings: list[ParameterBindingSpec] = []
    path_parameter_plans: dict[str, _PathParameterPlan] = {}
    # create_model's overloads cannot express generated fields statically
    # (mirrors signature.py's own _create_parameter_model); Any lets the
    # ** unpacking below reach create_model's dynamic-fields overload.
    fields: dict[str, Any] = {}
    for parameter_binding in binding.parameters:
        name = parameter_binding.name
        if name not in python_signature.parameters:
            raise _binding_error(
                f"declared parameter {name!r} is not a parameter of the implementation",
                operation_id=spec.operation_id,
            )
        if parameter_binding.source is ParameterSource.ENVIRONMENT:
            environment_bindings.append(parameter_binding)
            continue  # excluded from the Agent-facing model and schema entirely
        agent_bindings.append(parameter_binding)
        python_parameter = python_signature.parameters[name]
        default = ... if python_parameter.default is inspect.Parameter.empty else python_parameter.default
        if parameter_binding.codec is ParameterCodec.PATH:
            # The Agent-facing field is always a JSON string (or, when the
            # implementation's annotation is Optional, string | null) - or a
            # list of JSON strings for a ``multiple`` (list/Sequence)
            # annotation - regardless of what the implementation itself is
            # annotated with (Path, str, a union, possibly | None) - JSON has
            # no path type. The implementation's own annotation only decides
            # what encoding_wrapper converts back to before calling it.
            file_path_kind: Literal["file", "file_output"] = (
                "file_output" if parameter_binding.path_role == "output" else "file"
            )
            plan = _resolve_path_parameter_plan(
                name, hints.get(name), spec.operation_id, file_path_kind
            )
            if file_path_kind == "file_output" and plan.multiple:
                raise _binding_error(
                    f"parameter {name!r} is an output-file destination and must be a "
                    "single path, never a list",
                    operation_id=spec.operation_id,
                )
            path_parameter_plans[name] = plan
            agent_annotation = (
                Annotated[list[str], BeforeValidator(_one_path_as_list)]
                if plan.multiple
                else (str | None)
                if plan.optional
                else str
            )
            fields[name] = (
                agent_annotation,
                _normalize_path_default(default, plan=plan, name=name, operation_id=spec.operation_id),
            )
        elif parameter_binding.codec is ParameterCodec.DIRECTORY:
            # Exactly like the PATH branch for the Agent-facing field - always
            # a JSON string (or string | null when Optional), never a list and
            # never ``Path``. The declared path_role picks the plan kind:
            # "input" (a pre-existing tree, PATH parity plus a tree-manifest
            # hash) vs "output" (an Agent-chosen destination under the
            # designed containment rules). ParameterBindingSpec already
            # guarantees path_role is declared for this codec.
            path_kind = (
                "directory_input" if parameter_binding.path_role == "input" else "directory_output"
            )
            plan = _resolve_directory_parameter_plan(
                name, hints.get(name), path_kind, spec.operation_id
            )
            path_parameter_plans[name] = plan
            fields[name] = (
                (str | None) if plan.optional else str,
                _normalize_path_default(
                    default, plan=plan, name=name, operation_id=spec.operation_id
                ),
            )
        elif parameter_binding.codec is ParameterCodec.JSON:
            annotation = hints.get(name)
            # The codec wall is strict: no Path-carrying annotation may take
            # the containment-free JSON codec (allow_path=False). Registry
            # admission of the released ops keeps the lenient default in
            # signature.py's derive_operation_signature - a separate wall.
            if annotation is None or not _is_supported_json_annotation(
                annotation, allow_path=False
            ):
                raise _binding_error(
                    f"parameter {name!r} has no JSON-safe annotation for codec "
                    f"{parameter_binding.codec.value!r}; declare an explicit json_schema "
                    "with an unsupported-type codec instead",
                    operation_id=spec.operation_id,
                )
            fields[name] = (annotation, default)
        elif parameter_binding.codec is ParameterCodec.LEGACY_RESULT:
            # The agent-facing field IS the L1 result model (Ruling 2): Pydantic
            # validates the submitted JSON against the frozen OrganelleResult
            # contract during model construction, so an invalid payload is
            # rejected before the scientific callable ever runs, and the
            # implementation receives a real frozen result - never raw JSON.
            from organelleverse.core.result import OrganelleResult as _Result

            fields[name] = (_Result, default)
        elif parameter_binding.codec in _UNIMPLEMENTED_PARAMETER_CODECS:
            # Rejected here, at bind time, not deferred to the first invoke:
            # v1 has no decoder for these codecs at all (see
            # _decode_agent_parameter's final, unreachable branch).
            raise _binding_error(
                f"parameter codec {parameter_binding.codec.value!r} requires importing or "
                "restoring the pre-v1 legacy shim to decode, which is forbidden",
                operation_id=spec.operation_id,
                parameter=name,
            )
        else:
            fields[name] = (
                Annotated[Any, Field(json_schema_extra=dict(parameter_binding.json_schema))],
                default,
            )

    parameter_model = cast(
        "type[BaseModel]",
        create_model(
            f"{spec.operation_id.replace('.', '_')}_AgentParameters",
            __base__=StrictParameters,
            **fields,
        ),
    )
    agent_bindings_by_name = {parameter_binding.name: parameter_binding for parameter_binding in agent_bindings}
    environment_bindings_tuple = tuple(environment_bindings)
    core_input_type = CORE_TYPES.get(spec.input_kind) if binding.core_input_parameter is not None else None

    assert spec.callable_locator is not None, "named_parameters binding requires a callable_locator"
    callable_locator = spec.callable_locator

    def encoding_wrapper(*args: object, **agent_kwargs: object) -> OrganelleResult:
        call_kwargs: dict[str, object] = {
            name: _decode_agent_parameter(agent_bindings_by_name[name], value, operation_id=spec.operation_id)
            for name, value in agent_kwargs.items()
            if name not in path_parameter_plans
        }
        # Canonicalization, existence, content hashing, and destination
        # containment for every PATH- and DIRECTORY-codec parameter complete
        # here, before anything else - in particular before the scientific
        # callable is ever invoked.
        input_artifact_hashes = _prepare_path_parameters(
            path_parameter_plans, agent_kwargs, call_kwargs, operation_id=spec.operation_id
        )
        input_object_ids: tuple[str, ...] = ()
        if binding.core_input_parameter is not None:
            if len(args) != 1:
                raise _binding_error(  # pragma: no cover - guarded by OperationSpec/registry
                    "core_input_parameter binding requires exactly one core input value"
                )
            core_value = args[0]
            call_kwargs[binding.core_input_parameter] = core_value
            if isinstance(core_value, StrictFrozenModel):
                input_object_ids = (core_value.object_id,)
        elif args:  # pragma: no cover - guarded by OperationSpec/registry
            raise _binding_error("named_parameters binding without a core input received one")

        if environment_bindings_tuple:
            call_kwargs.update(
                _resolve_environment_parameters(
                    spec, environment_bindings_tuple, environment_provider
                )
            )

        # Every provenance preflight step - path hashing above, parameter
        # hashing here - must complete before the scientific callable is
        # ever invoked. A canonicalization failure (a non-finite float, an
        # uncanonicalizable value) must never surface only after the
        # callable already ran.
        parameters_hash = _hash_agent_parameters(agent_kwargs, operation_id=spec.operation_id)

        bound_arguments = python_signature.bind(**call_kwargs)
        bound_arguments.apply_defaults()

        # run_id generation and CapabilityExecutionContext's own validation
        # are themselves preflight, not detail folded into result encoding
        # after the fact: both must complete - successfully - before the
        # scientific callable ever runs, exactly like the hashing above.
        context = CapabilityExecutionContext(
            operation_id=spec.operation_id,
            operation_version=spec.contract_version,
            callable_locator=callable_locator,
            run_id=_new_run_id(),
            input_object_ids=input_object_ids,
            input_artifact_hashes=input_artifact_hashes,
            parameters_hash=parameters_hash,
        )

        raw_value = implementation(*bound_arguments.args, **bound_arguments.kwargs)
        # Result codec validation runs only now, after the callable has
        # returned - encoding what it produced is not itself a preflight
        # step and must not be confused for one.
        return encode_capability_result(raw_value, binding, context)

    signature = OperationSignature(
        python_signature=python_signature,
        input_name=binding.core_input_parameter,
        input_type=core_input_type,
        output_type=OrganelleResult,
        parameter_model=parameter_model,
    )
    return BoundOperation(
        spec=spec,
        implementation=encoding_wrapper,
        # .function intentionally holds the raw original callable, not
        # encoding_wrapper: its purpose here is identity/provenance ("this
        # capability really calls this exact function", never a second
        # implementation), not direct invocation. Direct calls must go
        # through .invoke(), which always calls encoding_wrapper via
        # .implementation. The real return type (e.g. a bare dict) need not
        # be CoreObject for that purpose, hence the explicit narrowing.
        function=cast("Callable[..., CoreObject]", implementation),
        signature=signature,
        data_contracts=(),
    )


def _resolve_environment_parameters(
    spec: OperationSpec,
    environment_bindings: tuple[ParameterBindingSpec, ...],
    environment_provider: EnvironmentParameterProvider | None,
) -> Mapping[str, object]:
    """Resolve every ENVIRONMENT-sourced parameter before the callable ever runs.

    Values are used exactly as the provider returns them — no codec decoding
    — since they are real Python objects the execution environment injects
    (an executor, a resource handle), not Agent-submitted JSON.
    """
    declared_names = {parameter_binding.name for parameter_binding in environment_bindings}
    provided = {} if environment_provider is None else dict(environment_provider(spec, environment_bindings))
    undeclared = set(provided) - declared_names
    if undeclared:
        raise _binding_error(
            "environment_provider returned parameters this operation does not declare "
            "as environment-sourced",
            operation_id=spec.operation_id,
            undeclared_parameters=sorted(undeclared),
        )
    missing = declared_names - set(provided)
    if missing:
        raise OrganelleContractError(
            code="capability.environment_parameter_missing",
            message=f"{spec.operation_id} is missing required environment-sourced parameters",
            details={
                "operation_id": spec.operation_id,
                "missing_parameters": sorted(missing),
            },
        )
    return provided


def _canonicalize_parameter_value(value: object) -> object:
    """Recursively project one Agent parameter value into JSON-safe form.

    ``agent_kwargs`` (see ``_hash_agent_parameters``) holds the Agent
    parameter model's *validated and default-filled* field values, exactly
    as Pydantic constructed them - not raw, undecoded JSON. For most codecs
    that is already a JSON scalar/list/dict. But a JSON-codec field
    annotated with an ``OperationParameterModel`` (e.g. ``OrganelleMetadata``)
    or a supported ``Enum`` makes Pydantic construct a real model or enum
    *instance*, which plain ``json.dumps`` cannot serialize. Such a value is
    projected through its own canonical form - ``model_dump(mode="json")``
    for a model, ``.value`` for an enum member - so the hash reflects
    declared content, never Python object identity or repr. A non-finite
    float fails closed here, before any hash is computed, catching values a
    schema-required codec's ``Any``-typed Agent field lets through (jsonschema's
    own ``{"type": "number"}`` does not itself reject NaN/Infinity).
    """
    if isinstance(value, BaseModel):
        return _canonicalize_parameter_value(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return _canonicalize_parameter_value(cast("object", value.value))
    if isinstance(value, Mapping):
        mapping_value = cast("Mapping[object, object]", value)
        return {
            str(key): _canonicalize_parameter_value(item) for key, item in mapping_value.items()
        }
    if isinstance(value, (list, tuple)):
        sequence_value = cast("Sequence[object]", value)
        return [_canonicalize_parameter_value(item) for item in sequence_value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float cannot be hashed for provenance: {value!r}")
        return value
    raise TypeError(f"{type(value).__name__} cannot be canonicalized to JSON for provenance hashing")


def _hash_agent_parameters(agent_kwargs: Mapping[str, object], *, operation_id: str) -> str:
    """Hash the Agent parameters, validated and default-filled: same -> same hash.

    ``agent_kwargs`` is exactly what Pydantic produced from the Agent
    parameter model - every declared field, submitted or defaulted, with no
    codec decoding applied yet - so an omitted-but-defaulted field enters
    the hash at its resolved value, never silently absent. Each value is
    canonicalized first (see ``_canonicalize_parameter_value``), then
    serialized deterministically (sorted keys, ``allow_nan=False`` as a
    second, structural guard after the canonicalizer's own finite-float
    check). Any failure becomes a structured
    ``capability.parameter_provenance_invalid`` - this is a preflight step,
    called before the scientific callable ever runs, so a raw ``TypeError``
    must never leak out of it.
    """
    try:
        canonical_kwargs = {
            name: _canonicalize_parameter_value(value) for name, value in agent_kwargs.items()
        }
        canonical = json.dumps(
            canonical_kwargs,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise OrganelleContractError(
            code="capability.parameter_provenance_invalid",
            message=f"{operation_id} parameters could not be canonicalized for provenance hashing",
            details={"operation_id": operation_id, "reason": str(error)},
        ) from error
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _prepare_path_parameters(
    path_parameter_plans: Mapping[str, _PathParameterPlan],
    agent_kwargs: Mapping[str, object],
    call_kwargs: dict[str, object],
    *,
    operation_id: str,
) -> tuple[str, ...]:
    """Canonicalize, verify, hash, and convert every PATH- and DIRECTORY-codec value.

    Runs to completion — including hashing real file or tree content — before
    the scientific callable is ever invoked. A missing or unreadable artifact,
    or a refused directory destination, raises here, so no partial or
    inconsistent provenance can ever be attached to a call that never
    happened. Converted values are written into *call_kwargs* in place; the
    returned tuple becomes ``input_artifact_hashes``.

    ``None`` (only possible for an Optional plan - Pydantic's own strict,
    non-nullable ``str`` field already rejects ``None`` before this ever
    runs otherwise) means "no artifact": no existence check, no hash, and
    the implementation receives a real ``None``, never a stringified
    sentinel or a check against a path that was never submitted.

    A ``multiple`` plan's raw value is a ``list[str]`` (the Pydantic field
    guarantees it); every element gets exactly the scalar preflight -
    ``is_file()`` else ``input.missing_artifact``, one content hash appended
    per element - and the implementation receives a ``list[Path]`` or
    ``list[str]`` per ``plan.target_type``.

    A ``directory_input`` plan resolves the submitted directory strictly
    (missing or dangling → ``input.missing_artifact``), requires it to be a
    real directory (a regular file is the wrong kind, still
    ``input.missing_artifact``), and appends one deterministic whole-tree
    manifest hash (``_sha256_directory_manifest``). A ``directory_output``
    plan runs the designed destination containment
    (``_validate_directory_output_target``) and hashes nothing - there is no
    content yet; the destination string itself is pinned by
    ``parameters_hash``.
    """
    hashes: list[str] = []
    for name, plan in path_parameter_plans.items():
        if name not in agent_kwargs:
            continue  # not supplied this call; Signature.bind resolves defaults
        raw_value = agent_kwargs[name]
        if raw_value is None:
            call_kwargs[name] = None
            continue
        if plan.path_kind == "file_output":
            if not isinstance(raw_value, str):
                raise _binding_error(  # pragma: no cover - the Agent field type is always str or str|None
                    f"parameter {name!r} uses codec 'path' (output) but did not receive a JSON string",
                    operation_id=operation_id,
                    actual_type=type(raw_value).__name__,
                )
            call_kwargs[name] = _validate_file_output_target(
                raw_value, plan=plan, name=name, operation_id=operation_id
            )
            continue  # a destination has no content to hash; parameters_hash pins the string
        if plan.path_kind == "directory_output":
            if not isinstance(raw_value, str):
                raise _binding_error(  # pragma: no cover - the Agent field type is always str or str|None
                    f"parameter {name!r} uses codec 'directory' but did not receive a JSON string",
                    operation_id=operation_id,
                    actual_type=type(raw_value).__name__,
                )
            call_kwargs[name] = _validate_directory_output_target(
                raw_value, plan=plan, name=name, operation_id=operation_id
            )
            continue  # a destination has no content to hash; parameters_hash pins the string
        if plan.path_kind == "directory_input":
            if not isinstance(raw_value, str):
                raise _binding_error(  # pragma: no cover - the Agent field type is always str or str|None
                    f"parameter {name!r} uses codec 'directory' but did not receive a JSON string",
                    operation_id=operation_id,
                    actual_type=type(raw_value).__name__,
                )
            candidate = Path(raw_value).expanduser()
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as error:
                raise OrganelleInputError(
                    code="input.missing_artifact",
                    message=f"directory does not exist: {candidate}",
                    details={
                        "operation_id": operation_id,
                        "parameter": name,
                        "path": str(candidate),
                    },
                ) from error
            if not resolved.is_dir():
                raise OrganelleInputError(
                    code="input.missing_artifact",
                    message=f"expected a directory but found a non-directory: {candidate}",
                    details={
                        "operation_id": operation_id,
                        "parameter": name,
                        "path": str(candidate),
                    },
                )
            try:
                hashes.append(_sha256_directory_manifest(resolved))
            except OSError as error:
                raise OrganelleInputError(
                    code="input.unreadable_artifact",
                    message=f"directory could not be read: {candidate}",
                    details={
                        "operation_id": operation_id,
                        "parameter": name,
                        "path": str(candidate),
                        "reason": str(error),
                    },
                ) from error
            call_kwargs[name] = resolved if plan.target_type is Path else raw_value
            continue
        if plan.multiple:
            if not isinstance(raw_value, list):
                raise _binding_error(  # pragma: no cover - the Agent field type is always list[str]
                    f"parameter {name!r} uses codec 'path' but did not receive a JSON "
                    "list of strings",
                    operation_id=operation_id,
                    actual_type=type(raw_value).__name__,
                )
            elements = cast(list[object], raw_value)
        else:
            elements = [raw_value]
        converted: list[str | Path] = []
        for element in elements:
            if not isinstance(element, str):
                raise _binding_error(  # pragma: no cover - the Agent field type is always str or str|None
                    f"parameter {name!r} uses codec 'path' but did not receive a JSON string",
                    operation_id=operation_id,
                    actual_type=type(element).__name__,
                )
            candidate = Path(element)
            if not candidate.is_file():
                raise OrganelleInputError(
                    code="input.missing_artifact",
                    message=f"artifact does not exist: {candidate}",
                    details={
                        "operation_id": operation_id,
                        "parameter": name,
                        "path": str(candidate),
                    },
                )
            try:
                hashes.append(_sha256_file(candidate))
            except OSError as error:
                raise OrganelleInputError(
                    code="input.unreadable_artifact",
                    message=f"artifact could not be read: {candidate}",
                    details={
                        "operation_id": operation_id,
                        "parameter": name,
                        "path": str(candidate),
                        "reason": str(error),
                    },
                ) from error
            converted.append(candidate if plan.target_type is Path else element)
        call_kwargs[name] = converted if plan.multiple else converted[0]
    return tuple(hashes)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory_manifest(root: Path) -> str:
    """Hash one directory tree as a deterministic manifest of its regular files.

    ``Path.rglob`` does not follow symlinked directories (py3.13 default), so
    the walked tree is exactly the submitted tree; a symlinked *file* hashes
    the content it points to, exactly as the PATH codec hashes a symlinked
    file today - recorded, never refused.
    """
    manifest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        manifest.update(path.relative_to(root).as_posix().encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(_sha256_file(path).encode("ascii"))
        manifest.update(b"\n")
    return manifest.hexdigest()


def _validate_directory_output_target(
    raw_value: str, *, plan: _PathParameterPlan, name: str, operation_id: str
) -> str | Path:
    """Validate an Agent-chosen directory destination under the containment rules.

    The destination must be *exactly where the path says it is*: no empty or
    parent-traversing (``..``) value, no symlink component anywhere from the
    nearest existing ancestor down to the leaf, the leaf itself - if it
    exists - a real directory, and the resolved destination never equal to,
    inside, or containing the managed run store (whose integrity is the
    publish protocol's job). Nothing is created and nothing is hashed:
    creation is the implementation's own act after validation, and the
    destination string is already pinned by ``parameters_hash``.
    """
    candidate = Path(raw_value).expanduser()
    if not raw_value.strip() or ".." in candidate.parts:
        raise OrganelleInputError(
            code="input.directory_destination_traversal",
            message=f"directory destination refuses empty or parent-traversing paths: {raw_value!r}",
            details={"operation_id": operation_id, "parameter": name, "path": raw_value},
        )
    if candidate.exists():
        if candidate.is_symlink() or not candidate.is_dir():
            raise OrganelleInputError(
                code="input.directory_destination_not_directory",
                message=f"directory destination exists and is not a real directory: {candidate}",
                details={"operation_id": operation_id, "parameter": name, "path": str(candidate)},
            )
        resolved = candidate.resolve()
    else:
        ancestor = candidate
        while not ancestor.exists():
            ancestor = ancestor.parent
        if ancestor.is_symlink() or not ancestor.is_dir():
            raise OrganelleInputError(
                code="input.directory_destination_not_directory",
                message=f"directory destination has no real directory ancestor: {candidate}",
                details={"operation_id": operation_id, "parameter": name, "path": str(candidate)},
            )
        current = ancestor
        for part in candidate.relative_to(ancestor).parts:
            current /= part
            if current.is_symlink():
                raise OrganelleInputError(
                    code="input.directory_destination_symlink",
                    message=f"directory destination traverses a symbolic link: {current}",
                    details={
                        "operation_id": operation_id,
                        "parameter": name,
                        "path": str(candidate),
                        "symlink": str(current),
                    },
                )
        resolved = ancestor.resolve() / candidate.relative_to(ancestor)
    from organelleverse.runtime import managed_runs_root

    runs_root = managed_runs_root().absolute()
    if resolved == runs_root or resolved in runs_root.parents or runs_root in resolved.parents:
        raise OrganelleInputError(
            code="input.directory_destination_run_store",
            message="directory destination must not be, contain, or sit inside the managed run store",
            details={"operation_id": operation_id, "parameter": name, "path": str(candidate)},
        )
    return resolved if plan.target_type is Path else raw_value


def _validate_file_output_target(
    raw_value: str, *, plan: _PathParameterPlan, name: str, operation_id: str
) -> str | Path:
    """Validate an Agent-chosen single-file destination under the containment rules.

    The file-output mirror of ``_validate_directory_output_target``: the
    destination must be exactly where the path says it is - no empty or
    parent-traversing value, no symlink component anywhere from the nearest
    existing ancestor to the leaf, the leaf (if it exists) a regular FILE
    never a directory or anything else, and the resolved destination never
    equal to, inside, or containing the managed run store. Nothing is
    created and nothing is hashed: creation is the implementation's own act
    after validation, and the destination string is pinned by
    ``parameters_hash`` exactly as directory destinations are.
    """
    candidate = Path(raw_value).expanduser()
    if not raw_value.strip() or ".." in candidate.parts:
        raise OrganelleInputError(
            code="input.file_destination_traversal",
            message=f"file destination refuses empty or parent-traversing paths: {raw_value!r}",
            details={"operation_id": operation_id, "parameter": name, "path": raw_value},
        )
    if candidate.exists():
        if candidate.is_symlink() or not candidate.is_file():
            raise OrganelleInputError(
                code="input.file_destination_not_file",
                message=f"file destination exists and is not a regular file: {candidate}",
                details={"operation_id": operation_id, "parameter": name, "path": str(candidate)},
            )
        resolved = candidate.resolve()
    else:
        ancestor = candidate
        while not ancestor.exists():
            ancestor = ancestor.parent
        if ancestor.is_symlink() or not ancestor.is_dir():
            raise OrganelleInputError(
                code="input.file_destination_not_file",
                message=f"file destination has no real directory ancestor: {candidate}",
                details={"operation_id": operation_id, "parameter": name, "path": str(candidate)},
            )
        current = ancestor
        for part in candidate.relative_to(ancestor).parts:
            current /= part
            if current.is_symlink():
                raise OrganelleInputError(
                    code="input.file_destination_symlink",
                    message=f"file destination traverses a symbolic link: {current}",
                    details={
                        "operation_id": operation_id,
                        "parameter": name,
                        "path": str(candidate),
                        "symlink": str(current),
                    },
                )
        resolved = ancestor.resolve() / candidate.relative_to(ancestor)
    from organelleverse.runtime import managed_runs_root

    runs_root = managed_runs_root().absolute()
    if resolved == runs_root or resolved in runs_root.parents or runs_root in resolved.parents:
        raise OrganelleInputError(
            code="input.file_destination_run_store",
            message="file destination must not be, contain, or sit inside the managed run store",
            details={"operation_id": operation_id, "parameter": name, "path": str(candidate)},
        )
    return resolved if plan.target_type is Path else raw_value


def _decode_agent_parameter(
    parameter_binding: ParameterBindingSpec, raw_value: object, *, operation_id: str
) -> object:
    if parameter_binding.codec in (
        ParameterCodec.JSON,
        ParameterCodec.PATH,
        ParameterCodec.DIRECTORY,
        ParameterCodec.LEGACY_RESULT,
    ):
        # Already validated by the generated Pydantic field's own JSON-safe
        # annotation (signature.py); these codecs never carry a declared
        # json_schema (OperationSpec forbids it), so there is nothing new to
        # check here — no duplicated or relaxed validation. DIRECTORY's real
        # validation is the _prepare_path_parameters preflight. LEGACY_RESULT
        # (Ruling 2) is validated by its own generated field being the L1
        # OrganelleResult model: the value here is already the reconstructed
        # frozen result, exactly what the implementation expects.
        return raw_value
    _validate_agent_value_against_schema(parameter_binding, raw_value, operation_id=operation_id)
    if parameter_binding.codec is ParameterCodec.DATACLASS:
        return _decode_dataclass_parameter(parameter_binding, raw_value)
    if parameter_binding.codec is ParameterCodec.NUMPY_ARTIFACT:
        return _decode_numpy_parameter(raw_value)
    raise _binding_error(  # pragma: no cover - _UNIMPLEMENTED_PARAMETER_CODECS rejects these at bind time
        f"parameter codec {parameter_binding.codec.value!r} requires importing or "
        "restoring the pre-v1 legacy shim to decode, which is forbidden",
        parameter=parameter_binding.name,
    )


def _schema_error(message: str, **details: object) -> OrganelleContractError:
    return OrganelleContractError(
        code="capability.parameter_schema_invalid", message=message, details=details
    )


def _validate_agent_value_against_schema(
    parameter_binding: ParameterBindingSpec, raw_value: object, *, operation_id: str
) -> None:
    """Validate *raw_value* against the parameter's declared json_schema.

    This is the runtime half of Task 2: ``ParameterBindingSpec`` already
    guarantees (at construction time, in operations/spec.py) that
    ``json_schema`` is a valid, deeply closed Draft 2020-12 document. Here
    that same schema — the one ``json_schema_extra`` already advertises in
    the documentation-facing Agent schema — is used to actually validate the
    value the Agent submitted, so the documented and enforced shapes can
    never drift apart.
    """
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(parameter_binding.json_schema)
    # raw_value is an arbitrary Agent-submitted value by design (that is
    # exactly what is being validated); jsonschema's own stub narrows its
    # instance parameter to a private recursive JSON type alias this module
    # has no way to name, so the boundary is cast explicitly, once, here.
    untyped_value = cast("Any", raw_value)
    errors = sorted(
        validator.iter_errors(untyped_value),  # pyright: ignore[reportUnknownMemberType]
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        raise _schema_error(
            f"parameter {parameter_binding.name!r} does not satisfy its declared json_schema",
            operation_id=operation_id,
            parameter=parameter_binding.name,
            errors=[
                {"path": list(error.absolute_path), "message": error.message} for error in errors
            ],
        )


def _decode_dataclass_parameter(parameter_binding: ParameterBindingSpec, raw_value: object) -> object:
    module_name, _, class_name = parameter_binding.target_type_locator.rpartition(":")
    if not module_name or not class_name:
        raise _binding_error(
            f"parameter {parameter_binding.name!r} target_type_locator must be 'module:ClassName'",
            target_type_locator=parameter_binding.target_type_locator,
        )
    module = importlib.import_module(module_name)
    target_type = getattr(module, class_name, None)
    if target_type is None or not (isinstance(target_type, type) and dataclasses.is_dataclass(target_type)):
        raise _binding_error(
            f"parameter {parameter_binding.name!r} target_type_locator does not resolve to a dataclass",
            target_type_locator=parameter_binding.target_type_locator,
        )
    if not isinstance(raw_value, Mapping):
        raise _binding_error(
            f"parameter {parameter_binding.name!r} dataclass codec requires a JSON object",
            actual_type=type(raw_value).__name__,
        )
    return target_type(**raw_value)


def _decode_numpy_parameter(raw_value: object) -> object:
    import numpy as np

    return np.asarray(raw_value)


def _new_run_id() -> str:
    return uuid4().hex


__all__ = [
    "EnvironmentParameterProvider",
    "bind_python_capability",
    "bind_worker_capability",
    "validate_worker_contract",
    "worker_parameter_schema",
]
