from collections.abc import Callable, Iterator, Mapping
from typing import Literal, cast

import pytest
from pydantic import BaseModel

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import (
    OrganelleContractError,
    OrganelleInputError,
    OrganelleParameterError,
)
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OrganelleResult
from organelleverse.operations import (
    CoreKind,
    ExecutionMode,
    OperationRegistry,
    OperationSpec,
    OperationStage,
    operation,
)
from organelleverse.operations.adapters import invoke_json
from organelleverse.operations.registry import CoreObject


class MalformedParameters(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise ValueError("malformed parameter mapping")

    def __iter__(self) -> Iterator[str]:
        return iter(("backend",))

    def __len__(self) -> int:
        return 1


class WrongParameters(BaseModel):
    pass


def _spec(
    operation_id: str,
    stage: OperationStage,
    input_kind: CoreKind,
    output_kind: CoreKind,
) -> OperationSpec:
    return OperationSpec(
        operation_id=operation_id,
        contract_version="1.0",
        title="Demo core-boundary probe",
        description="Test fixture spec used only to probe the core boundary matrix.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=stage,
        input_kind=input_kind,
        output_kind=output_kind,
        # I1a/I1d: a DATA output must declare exactly one output modality. Derived
        # here so every existing call site keeps its current arguments.
        output_modalities=("boundary_records",) if output_kind is CoreKind.DATA else (),
        callable_locator="tests.operations.boundaries:run",
    )


def _invalid_artifact() -> ArtifactRef:
    return ArtifactRef.model_construct(
        kind="x",
        uri="",
        format="",
        sha256="bad",
        size_bytes=-1,
    )


def _invalid_data(artifact: ArtifactRef) -> OrganelleData:
    return OrganelleData.model_construct(
        modality="erc",
        artifacts=FrozenMap.from_items({"alignment": artifact}),
    )


def _error(response: dict[str, object]) -> dict[str, object]:
    error = response.get("error")
    assert isinstance(error, dict)
    return cast(dict[str, object], error)


def test_invalid_exact_core_input_matrix_never_executes(
    genome: OrganelleGenome,
) -> None:
    registry = OperationRegistry()
    calls: list[str] = []
    invalid_artifact = _invalid_artifact()
    invalid_data = _invalid_data(invalid_artifact)
    invalid_genome = OrganelleGenome.model_construct(
        organelle="mitochondrion",
        sequence=None,
        annotation=None,
    )
    invalid_result = OrganelleResult.model_construct(
        operation_id="boundary.consume_result",
        scope="mitochondrion",
        status="failed",
        errors=(),
    )

    @operation(
        spec=_spec(
            "boundary.validate_genome",
            OperationStage.ANALYZE,
            CoreKind.GENOME,
            CoreKind.RESULT,
        ),
        registry=registry,
    )
    def validate_genome(value: OrganelleGenome) -> OrganelleResult:
        calls.append("genome")
        raise AssertionError("invalid Genome reached implementation")

    @operation(
        spec=_spec(
            "boundary.validate_data",
            OperationStage.TRANSFORM,
            CoreKind.DATA,
            CoreKind.DATA,
        ),
        registry=registry,
    )
    def validate_data(value: OrganelleData) -> OrganelleData:
        calls.append("data")
        raise AssertionError("invalid Data reached implementation")

    @operation(
        spec=_spec(
            "boundary.validate_result",
            OperationStage.CONSUME,
            CoreKind.RESULT,
            CoreKind.RESULT,
        ),
        registry=registry,
    )
    def validate_result(value: OrganelleResult) -> OrganelleResult:
        calls.append("result")
        raise AssertionError("invalid Result reached implementation")

    assert invalid_data.artifacts["alignment"] is invalid_artifact
    cases: tuple[tuple[str, Callable[[object], object], CoreObject], ...] = (
        (
            "boundary.validate_genome",
            cast(Callable[[object], object], validate_genome),
            invalid_genome,
        ),
        (
            "boundary.validate_data",
            cast(Callable[[object], object], validate_data),
            invalid_data,
        ),
        (
            "boundary.validate_result",
            cast(Callable[[object], object], validate_result),
            invalid_result,
        ),
    )

    for operation_id, direct, invalid in cases:
        with pytest.raises(OrganelleInputError) as direct_error:
            direct(invalid)
        assert direct_error.value.code == "input.invalid_core_object"
        assert direct_error.value.as_dict()["details"]["validation_errors"]

        with pytest.raises(OrganelleInputError) as registry_error:
            registry.invoke(operation_id, input=invalid, parameters={})
        assert registry_error.value.code == "input.invalid_core_object"
        assert registry_error.value.as_dict()["details"]["validation_errors"]

        response = invoke_json(
            {
                "operation_id": operation_id,
                "input": invalid.model_dump(mode="json"),
                "parameters": {},
            },
            registry=registry,
            granted_side_effects=set(),
        )
        assert response["ok"] is False
        assert "result" not in response
        assert _error(response)["error_code"] == "input.invalid_core_object"

    assert calls == []
    assert genome.kind == "genome"


def test_invalid_exact_core_output_matrix_is_never_exposed_or_serialized(
    genome: OrganelleGenome,
) -> None:
    registry = OperationRegistry()
    calls: list[str] = []
    invalid_artifact = _invalid_artifact()
    invalid_data = _invalid_data(invalid_artifact)
    invalid_genome = OrganelleGenome.model_construct(
        organelle="mitochondrion",
        sequence=None,
        annotation=None,
    )
    invalid_result = OrganelleResult.model_construct(
        operation_id="boundary.output_result",
        scope="mitochondrion",
        status="failed",
        errors=(),
    )

    @operation(
        spec=_spec(
            "boundary.output_genome",
            OperationStage.READ,
            CoreKind.NONE,
            CoreKind.GENOME,
        ),
        registry=registry,
    )
    def output_genome(source: str) -> OrganelleGenome:
        calls.append("genome")
        return invalid_genome

    @operation(
        spec=_spec(
            "boundary.output_data",
            OperationStage.READ,
            CoreKind.NONE,
            CoreKind.DATA,
        ),
        registry=registry,
    )
    def output_data(source: str) -> OrganelleData:
        calls.append("data")
        return invalid_data

    @operation(
        spec=_spec(
            "boundary.output_result",
            OperationStage.ANALYZE,
            CoreKind.GENOME,
            CoreKind.RESULT,
        ),
        registry=registry,
    )
    def output_result(value: OrganelleGenome) -> OrganelleResult:
        calls.append("result")
        return invalid_result

    assert invalid_data.artifacts["alignment"] is invalid_artifact
    cases: tuple[tuple[str, Callable[[], object], Callable[[], object], dict[str, object]], ...] = (
        (
            "genome",
            lambda: output_genome("source"),
            lambda: registry.invoke(
                "boundary.output_genome", input=None, parameters={"source": "source"}
            ),
            {
                "operation_id": "boundary.output_genome",
                "input": None,
                "parameters": {"source": "source"},
            },
        ),
        (
            "data",
            lambda: output_data("source"),
            lambda: registry.invoke(
                "boundary.output_data", input=None, parameters={"source": "source"}
            ),
            {
                "operation_id": "boundary.output_data",
                "input": None,
                "parameters": {"source": "source"},
            },
        ),
        (
            "result",
            lambda: output_result(genome),
            lambda: registry.invoke("boundary.output_result", input=genome, parameters={}),
            {
                "operation_id": "boundary.output_result",
                "input": genome.model_dump(mode="json"),
                "parameters": {},
            },
        ),
    )

    for output_kind, direct, registered, request in cases:
        for invoke in (direct, registered):
            with pytest.raises(OrganelleContractError) as raised:
                invoke()
            assert raised.value.code == "contract.invalid_operation_output"
            assert raised.value.as_dict()["details"]["validation_errors"]

        response = invoke_json(
            request,
            registry=registry,
            granted_side_effects=set(),
        )
        assert response["ok"] is False
        assert "result" not in response
        assert _error(response)["error_code"] == "contract.invalid_operation_output"
        assert calls.count(output_kind) == 3


def test_invalid_core_input_precedes_invalid_parameters_across_all_paths() -> None:
    registry = OperationRegistry()
    calls: list[str] = []
    spec = _spec(
        "boundary.validate_precedence",
        OperationStage.ANALYZE,
        CoreKind.GENOME,
        CoreKind.RESULT,
    )

    @operation(spec=spec, registry=registry)
    def annotate(
        genome: OrganelleGenome,
        *,
        backend: Literal["native", "external"] = "native",
    ) -> OrganelleResult:
        calls.append(backend)
        return OrganelleResult(
            operation_id=spec.operation_id,
            scope=genome.organelle,
            status="ok",
        )

    invalid = OrganelleGenome.model_construct(
        organelle="mitochondrion",
        sequence=None,
        annotation=None,
    )
    direct = cast(Callable[..., object], annotate)

    with pytest.raises(OrganelleInputError) as direct_error:
        direct(invalid, backend="invalid")
    assert direct_error.value.code == "input.invalid_core_object"

    response = invoke_json(
        {
            "operation_id": spec.operation_id,
            "input": invalid.model_dump(mode="json"),
            "parameters": {"backend": "invalid"},
        },
        registry=registry,
        granted_side_effects=set(),
    )
    assert response["ok"] is False
    assert _error(response)["error_code"] == "input.invalid_core_object"

    with pytest.raises(OrganelleInputError) as registry_error:
        registry.invoke(
            spec.operation_id,
            input=invalid,
            parameters={"backend": "invalid"},
        )
    assert registry_error.value.code == "input.invalid_core_object"
    assert calls == []


def test_registry_mapping_and_validated_model_errors_remain_structured(
    genome: OrganelleGenome,
) -> None:
    registry = OperationRegistry()
    calls: list[str] = []
    spec = _spec(
        "boundary.validate_parameters",
        OperationStage.ANALYZE,
        CoreKind.GENOME,
        CoreKind.RESULT,
    )

    @operation(spec=spec, registry=registry)
    def annotate(
        genome: OrganelleGenome,
        *,
        backend: Literal["native", "external"] = "native",
    ) -> OrganelleResult:
        calls.append(backend)
        return OrganelleResult(
            operation_id=spec.operation_id,
            scope=genome.organelle,
            status="ok",
        )

    binding = registry.require(spec.operation_id)
    assert binding.function is annotate
    with pytest.raises(OrganelleParameterError) as malformed:
        registry.invoke(
            spec.operation_id,
            input=genome,
            parameters=MalformedParameters(),
        )
    assert malformed.value.code == "parameter.invalid_operation_parameters"
    assert malformed.value.as_dict()["details"]["validation_errors"]
    assert calls == []

    with pytest.raises(OrganelleParameterError) as non_string_key:
        registry.invoke(
            spec.operation_id,
            input=genome,
            parameters=cast(Mapping[str, object], {1: "native"}),
        )
    assert non_string_key.value.code == "parameter.invalid_operation_parameters"
    assert non_string_key.value.as_dict()["details"]["validation_errors"]
    assert calls == []

    validated = binding.signature.parameter_model.model_validate({"backend": "native"})
    assert binding.invoke_validated(genome, validated).kind == "result"
    assert calls == ["native"]
    with pytest.raises(OrganelleContractError) as mismatched:
        binding.invoke_validated(genome, WrongParameters())
    assert mismatched.value.code == "contract.parameter_model_mismatch"
