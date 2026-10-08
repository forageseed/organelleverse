"""Permission-aware JSON transport for registered operations."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Set
from typing import cast

from pydantic import BaseModel, ConfigDict, ValidationError

from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import (
    OrganelleDependencyError,
    OrganelleError,
    OrganelleInputError,
    OrganelleNeedsInput,
    OrganelleParameterError,
    OrganellePermissionError,
)
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.registry import CoreObject, OperationRegistry
from organelleverse.operations.spec import CoreKind, SideEffect

from .._validation import validation_summaries, validation_summary


class AgentInvocationRequest(BaseModel):
    """The strict JSON-compatible request accepted by an Agent invocation."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    input: dict[str, object] | list[dict[str, object]] | None
    parameters: dict[str, object]


CORE_LOADERS: dict[CoreKind, type[CoreObject]] = {
    CoreKind.GENOME: OrganelleGenome,
    CoreKind.DATA: OrganelleData,
    CoreKind.RESULT: OrganelleResult,
}


def invoke_json(
    request: Mapping[str, object],
    *,
    registry: OperationRegistry,
    granted_side_effects: Collection[str | SideEffect],
) -> dict[str, object]:
    """Invoke an operation through the Agent JSON transport."""
    operation_id = str(request.get("operation_id", ""))
    try:
        decoded = _decode_request(request)
        spec = registry.describe(decoded.operation_id)
        granted = _decode_grants(granted_side_effects)
        missing = set(spec.side_effects) - granted
        if missing:
            raise OrganellePermissionError(
                code="permission.denied",
                message="operation side effects were not granted",
                details={"missing_side_effects": sorted(item.value for item in missing)},
            )
        dependency_report = registry.check_dependencies(decoded.operation_id)
        if not dependency_report.ready:
            raise OrganelleDependencyError(
                code="dependency.missing",
                message="operation dependencies are not ready",
                details=dependency_report.model_dump(mode="json"),
            )
        core_input = _decode_input(decoded.input, spec.input_kind, input_sequence=spec.input_sequence)
        binding = registry.require(decoded.operation_id)
        result = binding.invoke_with_parameter_decoder(
            core_input,
            decoded.parameters,
            parameter_decoder=_decode_parameters,
        )
        return {
            "ok": True,
            "operation_id": decoded.operation_id,
            "result": result.model_dump(mode="json"),
        }
    except OrganelleNeedsInput as error:
        return {
            "ok": False,
            "operation_id": operation_id,
            "error": error.as_dict(),
            "needs_input": error.needs_input.model_dump(mode="json"),
        }
    except OrganelleError as error:
        return {"ok": False, "operation_id": operation_id, "error": error.as_dict()}


def _decode_request(request: Mapping[str, object]) -> AgentInvocationRequest:
    try:
        return AgentInvocationRequest.model_validate(request)
    except ValidationError as error:
        raise OrganelleParameterError(
            code="parameter.invalid_request",
            message="agent invocation request is invalid",
            details={"validation_errors": validation_summaries(error)},
        ) from error


def _decode_grants(granted_side_effects: Collection[str | SideEffect]) -> set[SideEffect]:
    grants: set[SideEffect] = set()
    invalid: list[str] = []
    for grant in granted_side_effects:
        grant_value = cast(object, grant)
        if isinstance(grant_value, SideEffect):
            grants.add(grant_value)
            continue
        if not isinstance(grant_value, str):
            invalid.append(type(grant_value).__name__)
            continue
        try:
            grants.add(SideEffect(grant_value))
        except ValueError:
            invalid.append(grant_value)
    if invalid:
        if isinstance(granted_side_effects, Set):
            invalid.sort()
        raise OrganelleParameterError(
            code="parameter.invalid_side_effect_grant",
            message="side effect grants are invalid",
            details={"invalid_side_effect_grants": invalid},
        )
    return grants


def _decode_input(
    value: dict[str, object] | list[dict[str, object]] | None,
    expected_kind: CoreKind,
    *,
    input_sequence: bool = False,
) -> CoreObject | list[CoreObject] | None:
    if expected_kind is CoreKind.NONE:
        if value is None:
            return None
        raise OrganelleInputError(
            code="input.invalid_core_object",
            message="operation does not accept a core input",
            details={
                "expected_core_kind": expected_kind.value,
                "validation_errors": [
                    validation_summary((), "core input must be null", "unexpected")
                ],
            },
        )
    if value is None:
        raise OrganelleInputError(
            code="input.invalid_core_object",
            message="operation requires a core input",
            details={
                "expected_core_kind": expected_kind.value,
                "validation_errors": [validation_summary((), "core input is required", "missing")],
            },
        )
    loader = CORE_LOADERS[expected_kind]
    if input_sequence:
        # Ruling 3: the Agent JSON input is an array of core-object payloads;
        # every element is validated against the same L1 schema a single
        # input uses, before the callable runs.
        if not isinstance(value, list):
            raise OrganelleInputError(
                code="input.invalid_core_object",
                message="input_sequence operations require an array of core inputs",
                details={"expected_core_kind": expected_kind.value},
            )
        decoded: list[CoreObject] = []
        for index, element in enumerate(value):
            if not isinstance(element, dict):
                raise OrganelleInputError(
                    code="input.invalid_core_object",
                    message="input_sequence element is not a core-object payload",
                    details={"expected_core_kind": expected_kind.value, "index": index},
                )
            try:
                decoded.append(loader.model_validate(element))
            except ValidationError as error:
                raise OrganelleInputError(
                    code="input.invalid_core_object",
                    message="core input is invalid",
                    details={
                        "expected_core_kind": expected_kind.value,
                        "index": index,
                        "validation_errors": validation_summaries(error),
                    },
                ) from error
        return decoded
    try:
        return loader.model_validate(value)
    except ValidationError as error:
        raise OrganelleInputError(
            code="input.invalid_core_object",
            message="core input is invalid",
            details={
                "expected_core_kind": expected_kind.value,
                "validation_errors": validation_summaries(error),
            },
        ) from error


def _decode_parameters(
    parameter_model: type[BaseModel],
    parameters: Mapping[str, object],
) -> BaseModel:
    try:
        # Let the generated model reject IEEE specials at their typed field location;
        # Pydantic's JSON parser accepts these tokens so allow_inf_nan=False can report
        # the same finite_number error used by direct and Registry validation.
        parameters_json = json.dumps(dict(parameters), allow_nan=True)
        return parameter_model.model_validate_json(parameters_json)
    except (TypeError, ValueError, ValidationError) as error:
        summaries = (
            validation_summaries(error)
            if isinstance(error, ValidationError)
            else [validation_summary((), str(error), "json_serialization")]
        )
        raise OrganelleParameterError(
            code="parameter.invalid_operation_parameters",
            message="operation parameters are invalid",
            details={"validation_errors": summaries},
        ) from error
