from dataclasses import dataclass
from typing import Literal

import pytest

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import ErrorDetail, OrganelleResult
from organelleverse.operations import (
    CoreKind,
    ExecutionMode,
    OperationRegistry,
    OperationSpec,
    OperationStage,
    SideEffect,
    operation,
)


@dataclass
class ExecutionState:
    value: bool = False


@pytest.fixture
def genome() -> OrganelleGenome:
    return OrganelleGenome(
        organelle="mitochondrion",
        sequence=ArtifactRef(
            kind="sequence",
            uri="genome.fa",
            format="fasta",
            sha256="a" * 64,
            size_bytes=8,
        ),
    )


@pytest.fixture
def genome_json(genome: OrganelleGenome) -> dict[str, object]:
    return genome.model_dump(mode="json")


@pytest.fixture
def executed() -> ExecutionState:
    return ExecutionState()


@pytest.fixture
def analyze_spec() -> OperationSpec:
    return OperationSpec(
        operation_id="annotation.annotate",
        contract_version="1.0",
        title="Annotate an organelle genome",
        description="Test fixture spec standing in for a real ANALYZE operation.",
        keywords=("analyze", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.ANALYZE,
        input_kind=CoreKind.GENOME,
        output_kind=CoreKind.RESULT,
        callable_locator="organelleverse.annotation.api:annotate",
    )


@pytest.fixture
def read_spec() -> OperationSpec:
    return OperationSpec(
        operation_id="io.read_genome",
        contract_version="1.0",
        title="Read an organelle genome",
        description="Test fixture spec standing in for a real READ operation.",
        keywords=("fixture", "read", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.READ,
        input_kind=CoreKind.NONE,
        output_kind=CoreKind.GENOME,
        callable_locator="organelleverse.io.api:read_genome",
    )


@pytest.fixture
def local_registry(
    analyze_spec: OperationSpec,
    executed: ExecutionState,
) -> OperationRegistry:
    registry = OperationRegistry()
    spec = analyze_spec.model_copy(update={"side_effects": (SideEffect.READ_FILES,)})

    @operation(spec=spec, registry=registry)
    def annotate(
        genome: OrganelleGenome,
        *,
        backend: Literal["native", "external"] = "native",
    ) -> OrganelleResult:
        executed.value = True
        return OrganelleResult(
            operation_id="annotation.annotate",
            scope=genome.organelle,
            status="ok",
            metrics=FrozenMap(
                {
                    "backend": backend,
                    "input_object_id": genome.object_id,
                    "object_id": "business-metric-object-id",
                }
            ),
        )

    assert registry.require("annotation.annotate").function is annotate
    return registry


@pytest.fixture
def failed_registry(analyze_spec: OperationSpec) -> OperationRegistry:
    registry = OperationRegistry()

    @operation(spec=analyze_spec, registry=registry)
    def annotate(genome: OrganelleGenome) -> OrganelleResult:
        return OrganelleResult(
            operation_id="annotation.annotate",
            scope=genome.organelle,
            status="failed",
            errors=(ErrorDetail(code="backend.failed", message="backend failed"),),
        )

    assert registry.require("annotation.annotate").function is annotate
    return registry
