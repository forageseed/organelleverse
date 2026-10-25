"""Ordinary Python calls use the registry's existing checked invocation path.

Rich Python arguments keep their native types and return values. JSON codecs
remain the responsibility of the registry adapter; injected objects never
enter a JSON parameter model or lose their identity.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from organelleverse.core.errors import OrganelleContractError
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    create_model,
)

from .registry import CoreObject, _parameter_error, registry
from .signature import OperationSignature, StrictParameters


@cache
def python_signature(
    function: Callable[..., object],
    python_only: tuple[str, ...],
    input_name: str | None,
    path_parameters: tuple[str, ...] = (),
) -> tuple[inspect.Signature, dict[str, Any], type[BaseModel]]:
    """Build the native parameter model from the same scientific annotations."""
    signature = inspect.signature(function)
    hints = get_type_hints(function, include_extras=True)
    fields: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if (
            name.startswith("_")
            or name == input_name
            or parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ):
            continue
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        annotation = hints.get(name, Any)
        if name in python_only and any(
            isinstance(candidate, type)
            and getattr(candidate, "_is_protocol", False)
            and not getattr(candidate, "_is_runtime_protocol", False)
            for candidate in (annotation, *get_args(annotation))
        ):
            # Opaque duck-typed clients cannot be subjected to isinstance.
            # Callable injections and native filter dataclasses are otherwise
            # validated normally; Pydantic preserves their instance identity.
            continue
        if name in path_parameters and get_origin(annotation) in (list, Sequence):
            # The path codec already accepts one path as a one-element list.
            # Keep that documented Python convenience before strict sequence
            # validation, without coercing unrelated sequence arguments.
            from .python_binding import _one_path_as_list

            annotation = Annotated[annotation, BeforeValidator(_one_path_as_list)]
        fields[name] = (annotation, default)
    model = create_model(
        function.__name__ + "_PythonParameters",
        __base__=StrictParameters,
        __config__=ConfigDict(arbitrary_types_allowed=True),
        **fields,
    )
    return signature, hints, model


def invoke_python(
    operation_id: str,
    function: Callable[..., object],
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    python_only: tuple[str, ...] = (),
) -> object:
    """Check one native call without JSON conversion or an alternate algorithm."""
    from .registry import _invoke_checked_python

    binding = registry.require(operation_id)
    signature = inspect.signature(function)
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError as error:
        raise _parameter_error(error) from error
    bound.apply_defaults()
    core_name = binding.signature.input_name
    # An adapter's flattened signature may differ from its rich public function.
    if core_name not in signature.parameters:
        core_name = None
    path_parameters = tuple(
        parameter.name
        for parameter in binding.spec.binding.parameters
        if parameter.codec.value == "path"
    )
    _, hints, model = python_signature(function, tuple(python_only), core_name, path_parameters)
    native = OperationSignature(
        python_signature=signature,
        input_name=core_name,
        input_type=binding.signature.input_type if core_name else None,
        output_type=binding.signature.output_type,
        parameter_model=model,
        input_sequence=binding.signature.input_sequence,
    )
    values = {
        name: value
        for name, value in bound.arguments.items()
        if name in model.model_fields or name == core_name
    }

    def run(*core: CoreObject, **parameters: object) -> Any:
        if core_name:
            bound.arguments[core_name] = core[0]
        bound.arguments.update(parameters)
        return function(*bound.args, **bound.kwargs)

    result_hint = hints.get("return", Any)
    result_validator: Callable[[object], object] | None = None
    if result_hint is not binding.signature.output_type:
        owns_config = (
            isinstance(result_hint, type) and issubclass(result_hint, BaseModel)
        ) or dataclasses.is_dataclass(result_hint)
        adapter: TypeAdapter[Any] = TypeAdapter(
            result_hint, config=None if owns_config else ConfigDict(arbitrary_types_allowed=True)
        )

        def validate_result(result: object) -> object:
            try:
                return adapter.validate_python(result, strict=True)
            except (TypeError, ValueError, ValidationError) as error:
                raise OrganelleContractError(
                    code="contract.operation_output_type_mismatch",
                    message=f"{operation_id} returned a value outside its Python return annotation",
                ) from error

        result_validator = validate_result

    return _invoke_checked_python(
        binding.spec,
        native,
        run,
        values,
        binding.data_contracts,
        output_data_contracts=binding.output_data_contracts,
        suggestion_finalizer=binding.suggestion_finalizer,
        python_result_validator=result_validator,
    )
