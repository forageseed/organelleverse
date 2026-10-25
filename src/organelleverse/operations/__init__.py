from collections.abc import Mapping

from organelleverse.core.result import OrganelleResult

from .advice import AgentAdvice, NextOperationAdvice
from .data_contracts import DataContract, DataContractRegistry
from .decorator import operation
from .dependencies import DependencyCheck, DependencyReport, DependencyState
from .operation_search import OperationHit, search
from .public_api import PUBLIC_FUNCTIONS, public_function, resolve_public_function
from .registry import BoundOperation, CoreObject, OperationRegistry, registry
from .spec import (
    ArgumentMode,
    CoreKind,
    DependencyKind,
    DependencySpec,
    ExecutionMode,
    FallbackPolicy,
    OperationSpec,
    OperationStage,
    ParameterBindingSpec,
    ParameterCodec,
    ParameterSource,
    PythonBindingSpec,
    ResultCodec,
    RetryPolicy,
    SideEffect,
)

__all__ = [
    "AgentAdvice",
    "ArgumentMode",
    "BoundOperation",
    "CoreKind",
    "CoreObject",
    "DataContract",
    "DataContractRegistry",
    "DependencyCheck",
    "DependencyKind",
    "DependencyReport",
    "DependencySpec",
    "DependencyState",
    "ExecutionMode",
    "FallbackPolicy",
    "NextOperationAdvice",
    "OperationHit",
    "OperationRegistry",
    "OperationSpec",
    "OperationStage",
    "PUBLIC_FUNCTIONS",
    "ParameterBindingSpec",
    "ParameterCodec",
    "ParameterSource",
    "PythonBindingSpec",
    "ResultCodec",
    "RetryPolicy",
    "SideEffect",
    "agent_advice",
    "check_dependencies",
    "describe",
    "invocation_schema",
    "invoke",
    "list",
    "operation",
    "parameter_schema",
    "public_function",
    "registry",
    "resolve_public_function",
    "search",
]


def agent_advice(
    result: OrganelleResult,
    *,
    metadata: Mapping[str, object] | None = None,
    parameters: Mapping[str, Mapping[str, object]] | None = None,
    granted_side_effects: tuple[str, ...] = (),
) -> AgentAdvice:
    """Inspect live next-operation conditions without modifying a scientific result."""
    return registry.agent_advice(
        result,
        metadata=metadata,
        parameters=parameters,
        granted_side_effects=granted_side_effects,
    )


def list(
    *,
    stage: OperationStage | None = None,
    input_kind: CoreKind | None = None,
    organelle: str | None = None,
    input_modality: str | None = None,
    output_modality: str | None = None,
) -> tuple[OperationSpec, ...]:
    """List default-registry operations in deterministic ID order."""
    return registry.list(
        stage=stage,
        input_kind=input_kind,
        organelle=organelle,
        input_modality=input_modality,
        output_modality=output_modality,
    )


def describe(operation_id: str) -> OperationSpec:
    """Describe one default-registry operation."""
    return registry.describe(operation_id)


def parameter_schema(operation_id: str) -> dict[str, object]:
    """Return one default-registry operation's parameter JSON Schema."""
    return registry.parameter_schema(operation_id)


def invocation_schema(operation_id: str) -> dict[str, object]:
    """Return one default-registry operation's complete invocation JSON Schema."""
    return registry.invocation_schema(operation_id)


def check_dependencies(operation_id: str) -> DependencyReport:
    """Inspect one default-registry operation's declared dependencies."""
    return registry.check_dependencies(operation_id)


def invoke(
    operation_id: str,
    *,
    input: CoreObject | None,
    parameters: Mapping[str, object],
) -> CoreObject:
    """Invoke one default-registry operation with checked mapping parameters."""
    return registry.invoke(operation_id, input=input, parameters=parameters)
