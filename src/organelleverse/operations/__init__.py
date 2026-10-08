from collections.abc import Mapping

from .data_contracts import DataContract, DataContractRegistry
from .decorator import operation
from .dependencies import DependencyCheck, DependencyReport, DependencyState
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
    "OperationRegistry",
    "OperationSpec",
    "OperationStage",
    "ParameterBindingSpec",
    "ParameterCodec",
    "ParameterSource",
    "PythonBindingSpec",
    "ResultCodec",
    "RetryPolicy",
    "SideEffect",
    "check_dependencies",
    "describe",
    "invocation_schema",
    "invoke",
    "list",
    "operation",
    "parameter_schema",
    "registry",
]


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
