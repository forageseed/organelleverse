import asyncio
import json
from collections.abc import Callable, Iterator, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Protocol, cast

import pytest
from annotated_types import Ge

from organelleverse.core.errors import (
    OrganelleContractError,
    OrganelleInternalError,
    OrganelleNeedsInput,
    OrganelleParameterError,
)
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.core.input_requests import make_needs_input_request
from organelleverse.core.result import OrganelleResult
from organelleverse.operations import (
    CoreKind,
    DependencyKind,
    DependencySpec,
    OperationRegistry,
    OperationSpec,
    OperationStage,
    SideEffect,
    operation,
)
from organelleverse.operations.adapters import AgentInvocationRequest, invoke_json
from organelleverse.operations.parameters import OperationParameterModel


class ExecutionState(Protocol):
    value: bool


class ReverseGrantSet(set[str]):
    def __iter__(self) -> Iterator[str]:
        return iter(("zeta", "alpha"))


class ParameterFormat(StrEnum):
    JSON = "json"
    TSV = "tsv"


class ExtendedMetadata(OrganelleMetadata):
    review_note: str = "must not cross the boundary"


class OatkBackendParameters(OperationParameterModel):
    backend: Literal["oatk"] = "oatk"
    minimum_kmer_coverage: Annotated[int, Ge(1)] = 30


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _error(response: dict[str, object]) -> dict[str, object]:
    return _mapping(response["error"])


def _result(response: dict[str, object]) -> dict[str, object]:
    return _mapping(response["result"])


def _validation_errors(response: dict[str, object]) -> list[dict[str, object]]:
    value = _mapping(_error(response)["details"])["validation_errors"]
    return _validation_error_list(value)


def _validation_error_list(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    return [_mapping(item) for item in cast(list[object], value)]


def _raised_parameter_validation(invoke: Callable[[], object]) -> list[dict[str, object]]:
    with pytest.raises(OrganelleParameterError) as raised:
        invoke()
    value = _mapping(raised.value.as_dict()["details"])["validation_errors"]
    return _validation_error_list(value)


def _all_invocation_parameter_validation(
    registry: OperationRegistry,
    genome: OrganelleGenome,
    genome_json: dict[str, object],
    parameters: dict[str, object],
) -> list[list[dict[str, object]]]:
    binding = registry.require("annotation.annotate")
    direct = cast(Callable[..., object], binding.function)
    summaries = [
        _raised_parameter_validation(lambda: direct(genome, **parameters)),
        _raised_parameter_validation(
            lambda: registry.invoke(
                "annotation.annotate",
                input=genome,
                parameters=parameters,
            )
        ),
    ]
    response = invoke_json(
        {
            "operation_id": "annotation.annotate",
            "input": genome_json,
            "parameters": parameters,
        },
        registry=registry,
        granted_side_effects=set(),
    )
    summaries.append(_validation_errors(response))
    return summaries


def test_json_call_matches_direct_and_registry_calls_and_preserves_object_ids(
    local_registry: OperationRegistry,
    genome: OrganelleGenome,
    genome_json: dict[str, object],
) -> None:
    binding = local_registry.require("annotation.annotate")
    direct = binding.function(genome, backend="native")
    registered = local_registry.invoke(
        "annotation.annotate",
        input=genome,
        parameters={"backend": "native"},
    )

    response = invoke_json(
        {
            "operation_id": "annotation.annotate",
            "input": genome_json,
            "parameters": {"backend": "native"},
        },
        registry=local_registry,
        granted_side_effects={SideEffect.READ_FILES},
    )

    assert response["ok"] is True
    assert response["operation_id"] == "annotation.annotate"
    assert response["result"] == direct.model_dump(mode="json")
    assert registered == direct
    result = _result(response)
    assert result["object_id"] == direct.object_id
    metrics = _mapping(result["metrics"])
    assert metrics["input_object_id"] == genome.object_id
    assert metrics["object_id"] == "business-metric-object-id"


@pytest.mark.parametrize("invocation_path", ["direct", "registry", "agent"])
def test_all_invocation_paths_preserve_validated_python_parameter_types(
    analyze_spec: OperationSpec,
    genome: OrganelleGenome,
    genome_json: dict[str, object],
    invocation_path: str,
) -> None:
    registry = OperationRegistry()
    metadata = OrganelleMetadata(species="Arabidopsis thaliana")
    received: list[
        tuple[
            OrganelleMetadata,
            Path,
            ParameterFormat,
            Literal["native", "external"],
            bool,
            str,
            list[str],
            tuple[int, str],
            dict[str, int],
            Sequence[str],
            str | int,
            str | None,
            float,
        ]
    ] = []

    @operation(spec=analyze_spec, registry=registry)
    def inspect_parameters(
        genome: OrganelleGenome,
        *,
        metadata: OrganelleMetadata,
        output: Path,
        format: ParameterFormat,
        backend: Literal["native", "external"],
        enabled: bool,
        label: str,
        labels: list[str],
        coordinates: tuple[int, str],
        scores: dict[str, int],
        sequence: Sequence[str],
        selector: str | int,
        optional_label: str | None,
        threshold: Annotated[float, Ge(0)],
    ) -> OrganelleResult:
        assert type(metadata) is OrganelleMetadata
        assert metadata.species == "Arabidopsis thaliana"
        assert type(output) is type(Path())
        assert type(format) is ParameterFormat
        assert backend == "native"
        assert enabled is True
        assert label == "primary"
        assert type(labels) is list
        assert type(coordinates) is tuple
        assert type(scores) is dict
        assert type(sequence) is list
        assert type(selector) is int
        assert optional_label is None
        assert type(threshold) is float
        received.append(
            (
                metadata,
                output,
                format,
                backend,
                enabled,
                label,
                labels,
                coordinates,
                scores,
                cast(list[str], sequence),
                selector,
                optional_label,
                threshold,
            )
        )
        return OrganelleResult(
            operation_id="annotation.annotate",
            scope=genome.organelle,
            status="ok",
        )

    if invocation_path == "direct":
        inspect_parameters(
            genome,
            metadata=metadata,
            output=Path("result.json"),
            format=ParameterFormat.JSON,
            backend="native",
            enabled=True,
            label="primary",
            labels=["a", "b"],
            coordinates=(1, "two"),
            scores={"confidence": 1},
            sequence=["A", "T"],
            selector=3,
            optional_label=None,
            threshold=0.75,
        )
    elif invocation_path == "registry":
        registry.invoke(
            "annotation.annotate",
            input=genome,
            parameters={
                "metadata": metadata,
                "output": Path("result.json"),
                "format": ParameterFormat.JSON,
                "backend": "native",
                "enabled": True,
                "label": "primary",
                "labels": ["a", "b"],
                "coordinates": (1, "two"),
                "scores": {"confidence": 1},
                "sequence": ["A", "T"],
                "selector": 3,
                "optional_label": None,
                "threshold": 0.75,
            },
        )
    else:
        response = invoke_json(
            {
                "operation_id": "annotation.annotate",
                "input": genome_json,
                "parameters": {
                    "metadata": metadata.model_dump(mode="json"),
                    "output": "result.json",
                    "format": "json",
                    "backend": "native",
                    "enabled": True,
                    "label": "primary",
                    "labels": ["a", "b"],
                    "coordinates": [1, "two"],
                    "scores": {"confidence": 1},
                    "sequence": ["A", "T"],
                    "selector": 3,
                    "optional_label": None,
                    "threshold": 0.75,
                },
            },
            registry=registry,
            granted_side_effects=set(),
        )
        assert response["ok"] is True

    assert received == [
        (
            metadata,
            Path("result.json"),
            ParameterFormat.JSON,
            "native",
            True,
            "primary",
            ["a", "b"],
            (1, "two"),
            {"confidence": 1},
            ["A", "T"],
            3,
            None,
            0.75,
        )
    ]


@pytest.mark.parametrize("invocation_path", ["direct", "registry"])
def test_metadata_subclass_instances_do_not_reach_canonical_implementations(
    analyze_spec: OperationSpec,
    genome: OrganelleGenome,
    invocation_path: str,
) -> None:
    registry = OperationRegistry()
    executed: list[OrganelleMetadata] = []

    @operation(spec=analyze_spec, registry=registry)
    def inspect_metadata(
        genome: OrganelleGenome,
        *,
        metadata: OrganelleMetadata,
    ) -> OrganelleResult:
        executed.append(metadata)
        return OrganelleResult(
            operation_id="annotation.annotate",
            scope=genome.organelle,
            status="ok",
        )

    metadata = ExtendedMetadata(species="Arabidopsis thaliana")

    def invoke() -> object:
        if invocation_path == "direct":
            return inspect_metadata(genome, metadata=metadata)
        return registry.invoke(
            "annotation.annotate",
            input=genome,
            parameters={"metadata": metadata},
        )

    with pytest.raises(OrganelleParameterError) as raised:
        invoke()

    assert raised.value.code == "parameter.invalid_operation_parameters"
    assert executed == []


@pytest.mark.parametrize("invocation_path", ["direct", "registry"])
def test_constructed_invalid_metadata_is_revalidated_in_python_invocation_paths(
    analyze_spec: OperationSpec,
    genome: OrganelleGenome,
    invocation_path: str,
) -> None:
    registry = OperationRegistry()
    executed: list[OrganelleMetadata] = []

    @operation(spec=analyze_spec, registry=registry)
    def inspect_metadata(
        genome: OrganelleGenome,
        *,
        metadata: OrganelleMetadata,
    ) -> OrganelleResult:
        executed.append(metadata)
        return OrganelleResult(
            operation_id="annotation.annotate",
            scope=genome.organelle,
            status="ok",
        )

    metadata = OrganelleMetadata.model_construct(genetic_code=99)

    def invoke() -> object:
        if invocation_path == "direct":
            return inspect_metadata(genome, metadata=metadata)
        return registry.invoke(
            "annotation.annotate",
            input=genome,
            parameters={"metadata": metadata},
        )

    with pytest.raises(OrganelleParameterError) as raised:
        invoke()

    assert raised.value.code == "parameter.invalid_operation_parameters"
    assert _mapping(raised.value.as_dict()["details"])["validation_errors"]
    assert executed == []


def test_agent_result_json_chains_into_dedicated_consume_input_and_verifies_object_id(
    analyze_spec: OperationSpec,
    genome_json: dict[str, object],
) -> None:
    registry = OperationRegistry()
    consume_spec = analyze_spec.model_copy(
        update={
            "operation_id": "report.summarize",
            "stage": OperationStage.CONSUME,
            "input_kind": CoreKind.RESULT,
            "callable_locator": "organelleverse.reporting.api:summarize",
        }
    )
    consumed_ids: list[str] = []

    @operation(spec=analyze_spec, registry=registry)
    def annotate(genome: OrganelleGenome) -> OrganelleResult:
        return OrganelleResult(
            operation_id="annotation.annotate",
            scope=genome.organelle,
            status="ok",
        )

    @operation(spec=consume_spec, registry=registry)
    def summarize(result: OrganelleResult) -> OrganelleResult:
        consumed_ids.append(result.object_id)
        return OrganelleResult(
            operation_id="report.summarize",
            scope=result.scope,
            status="ok",
            metrics=FrozenMap({"source_object_id": result.object_id}),
        )

    assert registry.require("annotation.annotate").function is annotate
    assert registry.require("report.summarize").function is summarize
    first = invoke_json(
        {"operation_id": "annotation.annotate", "input": genome_json, "parameters": {}},
        registry=registry,
        granted_side_effects=set(),
    )
    first_result = _result(first)
    second = invoke_json(
        {"operation_id": "report.summarize", "input": first_result, "parameters": {}},
        registry=registry,
        granted_side_effects=set(),
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert consumed_ids == [first_result["object_id"]]
    assert _mapping(_result(second)["metrics"])["source_object_id"] == first_result["object_id"]

    forged_result = {**first_result, "object_id": "result:sha256:" + "0" * 64}
    forged = invoke_json(
        {"operation_id": "report.summarize", "input": forged_result, "parameters": {}},
        registry=registry,
        granted_side_effects=set(),
    )
    assert forged["ok"] is False
    assert _error(forged)["error_code"] == "input.invalid_core_object"
    assert "object_id" in str(_validation_errors(forged)[0]["message"])
    assert consumed_ids == [first_result["object_id"]]


def test_agent_rejects_invalid_exact_class_input_without_execution(
    analyze_spec: OperationSpec,
) -> None:
    registry = OperationRegistry()
    calls: list[str] = []

    @operation(spec=analyze_spec, registry=registry)
    def annotate(genome: OrganelleGenome) -> OrganelleResult:
        calls.append(genome.object_id)
        return OrganelleResult(
            operation_id="annotation.annotate",
            scope=genome.organelle,
            status="ok",
        )

    invalid = OrganelleGenome.model_construct(
        organelle="mitochondrion",
        sequence=None,
        annotation=None,
    )
    assert registry.require("annotation.annotate").function is annotate
    response = invoke_json(
        {
            "operation_id": "annotation.annotate",
            "input": invalid.model_dump(mode="json"),
            "parameters": {},
        },
        registry=registry,
        granted_side_effects=set(),
    )

    assert response["ok"] is False
    assert _error(response)["error_code"] == "input.invalid_core_object"
    assert calls == []


def test_agent_never_serializes_an_invalid_exact_class_output(
    analyze_spec: OperationSpec,
    genome_json: dict[str, object],
) -> None:
    registry = OperationRegistry()

    @operation(spec=analyze_spec, registry=registry)
    def annotate(genome: OrganelleGenome) -> OrganelleResult:
        return OrganelleResult.model_construct(
            operation_id="annotation.annotate",
            scope=genome.organelle,
            status="failed",
            errors=(),
        )

    assert registry.require("annotation.annotate").function is annotate
    response = invoke_json(
        {"operation_id": "annotation.annotate", "input": genome_json, "parameters": {}},
        registry=registry,
        granted_side_effects=set(),
    )

    assert response["ok"] is False
    assert "result" not in response
    assert _error(response)["error_code"] == "contract.invalid_operation_output"


def test_request_model_forbids_extra_fields() -> None:
    with pytest.raises(ValueError):
        AgentInvocationRequest.model_validate(
            {"operation_id": "annotation.annotate", "unexpected": True}
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"operation_id": "annotation.annotate", "parameters": {}},
        {"operation_id": "annotation.annotate", "input": None},
    ],
)
def test_request_requires_explicit_input_and_parameters(
    local_registry: OperationRegistry,
    payload: dict[str, object],
) -> None:
    response = invoke_json(
        payload,
        registry=local_registry,
        granted_side_effects={"read_files"},
    )

    assert response["ok"] is False
    assert _error(response)["error_code"] == "parameter.invalid_request"


def test_request_accepts_explicit_null_input_and_empty_parameters() -> None:
    request = AgentInvocationRequest.model_validate(
        {"operation_id": "io.read", "input": None, "parameters": {}}
    )

    assert request.input is None
    assert request.parameters == {}


def test_read_accepts_null_input_and_rejects_a_core_object(
    read_spec: OperationSpec,
    genome: OrganelleGenome,
    genome_json: dict[str, object],
) -> None:
    registry = OperationRegistry()
    calls: list[str] = []

    @operation(spec=read_spec, registry=registry)
    def load(path: str, *, organelle: Literal["mitochondrion", "plastid"]) -> OrganelleGenome:
        calls.append(path)
        assert organelle == "mitochondrion"
        return genome

    assert registry.require("io.read_genome").function is load
    valid = invoke_json(
        {
            "operation_id": "io.read_genome",
            "input": None,
            "parameters": {"path": "genome.fa", "organelle": "mitochondrion"},
        },
        registry=registry,
        granted_side_effects=set(),
    )
    invalid = invoke_json(
        {
            "operation_id": "io.read_genome",
            "input": genome_json,
            "parameters": {"path": "genome.fa", "organelle": "mitochondrion"},
        },
        registry=registry,
        granted_side_effects=set(),
    )

    assert valid["ok"] is True
    assert invalid["ok"] is False
    assert _error(invalid)["error_code"] == "input.invalid_core_object"
    assert calls == ["genome.fa"]


def test_analyze_requires_a_core_object_without_executing(
    local_registry: OperationRegistry,
    executed: ExecutionState,
) -> None:
    response = invoke_json(
        {"operation_id": "annotation.annotate", "input": None, "parameters": {}},
        registry=local_registry,
        granted_side_effects={"read_files"},
    )

    assert response["ok"] is False
    assert _error(response)["error_code"] == "input.invalid_core_object"
    assert executed.value is False


@pytest.mark.parametrize(
    "payload",
    [
        {"operation_id": "annotation.annotate", "unexpected": True},
        {"operation_id": "annotation.annotate", "input": []},
        {"operation_id": "annotation.annotate", "parameters": []},
    ],
)
def test_malformed_requests_return_structured_request_errors(
    local_registry: OperationRegistry,
    payload: dict[str, object],
) -> None:
    response = invoke_json(
        payload,
        registry=local_registry,
        granted_side_effects={"read_files"},
    )

    assert response["ok"] is False
    assert _error(response)["error_code"] == "parameter.invalid_request"
    validation = _validation_errors(response)
    assert set(validation[0]) == {"loc", "message", "type"}


def test_malformed_core_input_and_wrong_core_kind_do_not_execute(
    local_registry: OperationRegistry,
    genome_json: dict[str, object],
    executed: ExecutionState,
) -> None:
    malformed = invoke_json(
        {
            "operation_id": "annotation.annotate",
            "input": {"kind": "genome"},
            "parameters": {},
        },
        registry=local_registry,
        granted_side_effects={"read_files"},
    )
    wrong_kind = invoke_json(
        {
            "operation_id": "annotation.annotate",
            "input": {
                "kind": "result",
                "operation_id": "annotation.annotate",
                "scope": "mitochondrion",
                "status": "ok",
            },
            "parameters": {},
        },
        registry=local_registry,
        granted_side_effects={"read_files"},
    )

    assert _error(malformed)["error_code"] == "input.invalid_core_object"
    assert _error(wrong_kind)["error_code"] == "input.invalid_core_object"
    assert executed.value is False
    assert genome_json["kind"] == "genome"


def test_forged_core_object_id_is_rejected_without_execution(
    local_registry: OperationRegistry,
    genome_json: dict[str, object],
    executed: ExecutionState,
) -> None:
    forged = {**genome_json, "object_id": "genome:sha256:" + "0" * 64}

    response = invoke_json(
        {"operation_id": "annotation.annotate", "input": forged, "parameters": {}},
        registry=local_registry,
        granted_side_effects={"read_files"},
    )

    assert response["ok"] is False
    assert _error(response)["error_code"] == "input.invalid_core_object"
    message = _validation_errors(response)[0]["message"]
    assert isinstance(message, str)
    assert "object_id" in message
    assert executed.value is False


def test_invalid_operation_parameters_do_not_execute(
    local_registry: OperationRegistry,
    genome_json: dict[str, object],
    executed: ExecutionState,
) -> None:
    response = invoke_json(
        {
            "operation_id": "annotation.annotate",
            "input": genome_json,
            "parameters": {"backend": "unsupported"},
        },
        registry=local_registry,
        granted_side_effects={"read_files"},
    )

    assert response["ok"] is False
    assert _error(response)["error_code"] == "parameter.invalid_operation_parameters"
    assert _validation_errors(response)[0]["loc"] == ["backend"]
    assert executed.value is False


def test_validation_summaries_match_direct_registry_and_json_paths(
    local_registry: OperationRegistry,
    genome: OrganelleGenome,
    genome_json: dict[str, object],
) -> None:
    binding = local_registry.require("annotation.annotate")
    summaries: list[list[dict[str, object]]] = []

    for invoke in (
        lambda: binding.function(genome, backend="unsupported"),
        lambda: local_registry.invoke(
            "annotation.annotate", input=genome, parameters={"backend": "unsupported"}
        ),
    ):
        with pytest.raises(OrganelleParameterError) as raised:
            invoke()
        value = _mapping(raised.value.as_dict()["details"])["validation_errors"]
        summaries.append(_validation_error_list(value))

    response = invoke_json(
        {
            "operation_id": "annotation.annotate",
            "input": genome_json,
            "parameters": {"backend": "unsupported"},
        },
        registry=local_registry,
        granted_side_effects={"read_files"},
    )
    value = _mapping(_error(response)["details"])["validation_errors"]
    summaries.append(_validation_error_list(value))

    assert summaries[0] == summaries[1] == summaries[2]
    assert all(set(summary) == {"loc", "message", "type"} for summary in summaries[0])


def test_missing_required_keyword_validation_matches_all_invocation_paths(
    analyze_spec: OperationSpec,
    genome: OrganelleGenome,
    genome_json: dict[str, object],
) -> None:
    registry = OperationRegistry()

    @operation(spec=analyze_spec, registry=registry)
    def annotate(genome: OrganelleGenome, *, threshold: int) -> OrganelleResult:
        raise AssertionError("not executed")

    direct = cast(Callable[..., object], annotate)
    summaries = [
        _raised_parameter_validation(lambda: direct(genome)),
        _raised_parameter_validation(
            lambda: registry.invoke("annotation.annotate", input=genome, parameters={})
        ),
    ]
    response = invoke_json(
        {"operation_id": "annotation.annotate", "input": genome_json, "parameters": {}},
        registry=registry,
        granted_side_effects=set(),
    )
    summaries.append(_validation_errors(response))

    assert summaries[0] == summaries[1] == summaries[2]
    assert summaries[0] == [{"loc": ["threshold"], "message": "Field required", "type": "missing"}]


def test_missing_read_source_validation_matches_all_invocation_paths(
    read_spec: OperationSpec,
    genome: OrganelleGenome,
) -> None:
    registry = OperationRegistry()

    @operation(spec=read_spec, registry=registry)
    def load(
        path: str,
        *,
        organelle: Literal["mitochondrion", "plastid"] = "mitochondrion",
    ) -> OrganelleGenome:
        raise AssertionError("not executed")

    direct = cast(Callable[..., object], load)
    summaries = [
        _raised_parameter_validation(direct),
        _raised_parameter_validation(
            lambda: registry.invoke("io.read_genome", input=None, parameters={})
        ),
    ]
    response = invoke_json(
        {"operation_id": "io.read_genome", "input": None, "parameters": {}},
        registry=registry,
        granted_side_effects=set(),
    )
    summaries.append(_validation_errors(response))

    assert summaries[0] == summaries[1] == summaries[2]
    assert summaries[0] == [{"loc": ["path"], "message": "Field required", "type": "missing"}]


def test_extra_keyword_validation_matches_all_invocation_paths(
    local_registry: OperationRegistry,
    genome: OrganelleGenome,
    genome_json: dict[str, object],
) -> None:
    binding = local_registry.require("annotation.annotate")
    summaries = [
        _raised_parameter_validation(lambda: binding.function(genome, unexpected=True)),
        _raised_parameter_validation(
            lambda: local_registry.invoke(
                "annotation.annotate", input=genome, parameters={"unexpected": True}
            )
        ),
    ]
    response = invoke_json(
        {
            "operation_id": "annotation.annotate",
            "input": genome_json,
            "parameters": {"unexpected": True},
        },
        registry=local_registry,
        granted_side_effects={"read_files"},
    )
    summaries.append(_validation_errors(response))

    assert summaries[0] == summaries[1] == summaries[2]
    assert summaries[0] == [
        {
            "loc": ["unexpected"],
            "message": "Extra inputs are not permitted",
            "type": "extra_forbidden",
        }
    ]


@pytest.mark.parametrize("parameters", [{"backend": float("nan")}, {"backend": object()}])
def test_non_json_or_nan_parameters_do_not_execute(
    local_registry: OperationRegistry,
    genome_json: dict[str, object],
    executed: ExecutionState,
    parameters: dict[str, object],
) -> None:
    response = invoke_json(
        {
            "operation_id": "annotation.annotate",
            "input": genome_json,
            "parameters": parameters,
        },
        registry=local_registry,
        granted_side_effects={"read_files"},
    )

    assert response["ok"] is False
    assert _error(response)["error_code"] == "parameter.invalid_operation_parameters"
    assert executed.value is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_float_validation_matches_all_invocation_paths(
    analyze_spec: OperationSpec,
    genome: OrganelleGenome,
    genome_json: dict[str, object],
    value: float,
) -> None:
    registry = OperationRegistry()

    @operation(spec=analyze_spec, registry=registry)
    def annotate(genome: OrganelleGenome, *, threshold: float) -> OrganelleResult:
        return OrganelleResult(
            operation_id="annotation.annotate",
            scope=genome.organelle,
            status="ok",
        )

    assert registry.require("annotation.annotate").function is annotate
    summaries = _all_invocation_parameter_validation(
        registry,
        genome,
        genome_json,
        {"threshold": value},
    )

    assert summaries[0] == summaries[1] == summaries[2]
    assert summaries[0] == [
        {
            "loc": ["threshold"],
            "message": "Input should be a finite number",
            "type": "finite_number",
        }
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nested_nonfinite_float_validation_matches_all_invocation_paths(
    analyze_spec: OperationSpec,
    genome: OrganelleGenome,
    genome_json: dict[str, object],
    value: float,
) -> None:
    registry = OperationRegistry()

    @operation(spec=analyze_spec, registry=registry)
    def annotate(
        genome: OrganelleGenome,
        *,
        scores: dict[str, list[float]],
    ) -> OrganelleResult:
        return OrganelleResult(
            operation_id="annotation.annotate",
            scope=genome.organelle,
            status="ok",
        )

    assert registry.require("annotation.annotate").function is annotate
    summaries = _all_invocation_parameter_validation(
        registry,
        genome,
        genome_json,
        {"scores": {"review": [value]}},
    )

    assert summaries[0] == summaries[1] == summaries[2]
    assert summaries[0] == [
        {
            "loc": ["scores", "review", "0"],
            "message": "Input should be a finite number",
            "type": "finite_number",
        }
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_constrained_nonfinite_float_validation_matches_all_invocation_paths(
    analyze_spec: OperationSpec,
    genome: OrganelleGenome,
    genome_json: dict[str, object],
    value: float,
) -> None:
    registry = OperationRegistry()

    @operation(spec=analyze_spec, registry=registry)
    def annotate(
        genome: OrganelleGenome,
        *,
        threshold: Annotated[float, Ge(0)],
    ) -> OrganelleResult:
        return OrganelleResult(
            operation_id="annotation.annotate",
            scope=genome.organelle,
            status="ok",
        )

    assert registry.require("annotation.annotate").function is annotate
    summaries = _all_invocation_parameter_validation(
        registry,
        genome,
        genome_json,
        {"threshold": value},
    )

    assert summaries[0] == summaries[1] == summaries[2]
    assert summaries[0] == [
        {
            "loc": ["threshold"],
            "message": "Input should be a finite number",
            "type": "finite_number",
        }
    ]


def test_json_parameter_validation_decodes_path_values(
    read_spec: OperationSpec, genome: OrganelleGenome
) -> None:
    registry = OperationRegistry()
    seen: list[Path] = []

    @operation(spec=read_spec, registry=registry)
    def load(path: Path, *, organelle: Literal["mitochondrion", "plastid"]) -> OrganelleGenome:
        seen.append(path)
        assert organelle == "mitochondrion"
        return genome

    assert registry.require("io.read_genome").function is load
    response = invoke_json(
        {
            "operation_id": "io.read_genome",
            "input": None,
            "parameters": {"path": "inputs/genome.fa", "organelle": "mitochondrion"},
        },
        registry=registry,
        granted_side_effects=set(),
    )

    assert response["ok"] is True
    assert seen == [Path("inputs/genome.fa")]


def test_unknown_operation_and_unknown_grant_are_structured_errors(
    local_registry: OperationRegistry,
    genome_json: dict[str, object],
    executed: ExecutionState,
) -> None:
    unknown_operation = invoke_json(
        {"operation_id": "missing.operation", "input": None, "parameters": {}},
        registry=local_registry,
        granted_side_effects=set(),
    )
    unknown_grant = invoke_json(
        {"operation_id": "annotation.annotate", "input": genome_json, "parameters": {}},
        registry=local_registry,
        granted_side_effects={"unknown_grant"},
    )

    assert _error(unknown_operation)["error_code"] == "input.unknown_operation"
    assert _error(unknown_grant)["error_code"] == "parameter.invalid_side_effect_grant"
    assert executed.value is False


def test_non_string_side_effect_grant_is_json_safe_and_does_not_execute(
    local_registry: OperationRegistry,
    genome_json: dict[str, object],
    executed: ExecutionState,
) -> None:
    response = invoke_json(
        {"operation_id": "annotation.annotate", "input": genome_json, "parameters": {}},
        registry=local_registry,
        granted_side_effects=cast(list[str | SideEffect], [object()]),
    )

    assert response["ok"] is False
    error = _error(response)
    assert error["error_code"] == "parameter.invalid_side_effect_grant"
    assert _mapping(error["details"])["invalid_side_effect_grants"] == ["object"]
    json.dumps(response, allow_nan=False)
    assert executed.value is False


@pytest.mark.parametrize(
    ("grants", "expected"),
    [
        (ReverseGrantSet({"alpha", "zeta"}), ["alpha", "zeta"]),
        (["zeta", "alpha"], ["zeta", "alpha"]),
    ],
)
def test_invalid_grant_details_sort_unordered_inputs_and_preserve_ordered_inputs(
    local_registry: OperationRegistry,
    genome_json: dict[str, object],
    grants: list[str] | set[str],
    expected: list[str],
) -> None:
    response = invoke_json(
        {"operation_id": "annotation.annotate", "input": genome_json, "parameters": {}},
        registry=local_registry,
        granted_side_effects=grants,
    )

    assert _error(response)["error_code"] == "parameter.invalid_side_effect_grant"
    assert _mapping(_error(response)["details"])["invalid_side_effect_grants"] == expected


def test_ungranted_side_effect_does_not_execute(
    local_registry: OperationRegistry,
    genome_json: dict[str, object],
    executed: ExecutionState,
) -> None:
    response = invoke_json(
        {"operation_id": "annotation.annotate", "input": genome_json, "parameters": {}},
        registry=local_registry,
        granted_side_effects=set(),
    )

    assert response["ok"] is False
    assert _error(response)["error_code"] == "permission.denied"
    assert _mapping(_error(response)["details"])["missing_side_effects"] == ["read_files"]
    assert executed.value is False


def test_missing_dependencies_do_not_execute(
    analyze_spec: OperationSpec, genome_json: dict[str, object]
) -> None:
    registry = OperationRegistry()
    calls: list[bool] = []
    spec = analyze_spec.model_copy(
        update={
            "dependencies": (
                DependencySpec(kind=DependencyKind.PYTHON, name="adapter_dependency_missing"),
            )
        }
    )

    @operation(spec=spec, registry=registry)
    def annotate(genome: OrganelleGenome) -> OrganelleResult:
        calls.append(True)
        return OrganelleResult(
            operation_id="annotation.annotate",
            scope=genome.organelle,
            status="ok",
        )

    assert registry.require("annotation.annotate").function is annotate
    response = invoke_json(
        {"operation_id": "annotation.annotate", "input": genome_json, "parameters": {}},
        registry=registry,
        granted_side_effects=set(),
    )

    assert response["ok"] is False
    assert _error(response)["error_code"] == "dependency.missing"
    assert calls == []


def test_failed_result_is_transport_success(
    failed_registry: OperationRegistry,
    genome_json: dict[str, object],
) -> None:
    response = invoke_json(
        {"operation_id": "annotation.annotate", "input": genome_json, "parameters": {}},
        registry=failed_registry,
        granted_side_effects=set(),
    )

    assert response["ok"] is True
    assert _result(response)["status"] == "failed"


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (
            OrganelleContractError(code="contract.bad_output", message="bad output"),
            "contract.bad_output",
        ),
        (OrganelleInternalError(code="internal.broken", message="broken"), "internal.broken"),
    ],
)
def test_structured_operation_errors_are_transport_errors(
    analyze_spec: OperationSpec,
    genome_json: dict[str, object],
    error: Exception,
    code: str,
) -> None:
    registry = OperationRegistry()

    @operation(spec=analyze_spec, registry=registry)
    def annotate(genome: OrganelleGenome) -> OrganelleResult:
        raise error

    assert registry.require("annotation.annotate").function is annotate
    response = invoke_json(
        {"operation_id": "annotation.annotate", "input": genome_json, "parameters": {}},
        registry=registry,
        granted_side_effects=set(),
    )

    assert response["ok"] is False
    assert _error(response)["error_code"] == code


@pytest.mark.parametrize(
    "error",
    [RuntimeError("unexpected"), KeyboardInterrupt(), SystemExit(3), asyncio.CancelledError()],
)
def test_non_organelle_exceptions_propagate(
    analyze_spec: OperationSpec,
    genome_json: dict[str, object],
    error: BaseException,
) -> None:
    registry = OperationRegistry()

    @operation(spec=analyze_spec, registry=registry)
    def annotate(genome: OrganelleGenome) -> OrganelleResult:
        raise error

    assert registry.require("annotation.annotate").function is annotate
    with pytest.raises(type(error)):
        invoke_json(
            {"operation_id": "annotation.annotate", "input": genome_json, "parameters": {}},
            registry=registry,
            granted_side_effects=set(),
        )


@pytest.mark.parametrize("invocation_path", ["direct", "registry", "agent"])
def test_structured_parameter_model_reaches_canonical_callable_through_every_path(
    analyze_spec: OperationSpec,
    genome: OrganelleGenome,
    genome_json: dict[str, object],
    invocation_path: str,
) -> None:
    registry = OperationRegistry()
    received: list[OatkBackendParameters] = []

    @operation(spec=analyze_spec, registry=registry)
    def annotate(
        genome: OrganelleGenome,
        *,
        backend_parameters: OatkBackendParameters | None = None,
    ) -> OrganelleResult:
        received.append(cast(OatkBackendParameters, backend_parameters))
        return OrganelleResult(
            operation_id="annotation.annotate",
            scope=genome.organelle,
            status="ok",
        )

    assert registry.require("annotation.annotate").function is annotate
    raw_parameters: dict[str, object] = {
        "backend": "oatk",
        "minimum_kmer_coverage": 41,
    }
    expected = OatkBackendParameters.model_validate(raw_parameters)

    if invocation_path == "direct":
        annotate(
            genome,
            backend_parameters=cast(OatkBackendParameters, raw_parameters),
        )
    elif invocation_path == "registry":
        registry.invoke(
            "annotation.annotate",
            input=genome,
            parameters={"backend_parameters": raw_parameters},
        )
    else:
        response = invoke_json(
            {
                "operation_id": "annotation.annotate",
                "input": genome_json,
                "parameters": {"backend_parameters": raw_parameters},
            },
            registry=registry,
            granted_side_effects=set(),
        )
        assert response["ok"] is True

    assert len(received) == 1
    assert type(received[0]) is OatkBackendParameters
    assert received[0] == expected


def test_agent_structured_parameter_strictly_rejects_coerced_int_and_unknown_field(
    analyze_spec: OperationSpec,
    genome_json: dict[str, object],
) -> None:
    registry = OperationRegistry()

    @operation(spec=analyze_spec, registry=registry)
    def annotate(
        genome: OrganelleGenome,
        *,
        backend_parameters: OatkBackendParameters | None = None,
    ) -> OrganelleResult:
        raise AssertionError("not executed")

    assert registry.require("annotation.annotate").function is annotate
    coerced = invoke_json(
        {
            "operation_id": "annotation.annotate",
            "input": genome_json,
            "parameters": {
                "backend_parameters": {"backend": "oatk", "minimum_kmer_coverage": "41"}
            },
        },
        registry=registry,
        granted_side_effects=set(),
    )
    unknown = invoke_json(
        {
            "operation_id": "annotation.annotate",
            "input": genome_json,
            "parameters": {
                "backend_parameters": {"backend": "oatk", "minimum_kmer_coverage": 30, "raw": True}
            },
        },
        registry=registry,
        granted_side_effects=set(),
    )

    assert coerced["ok"] is False
    assert _error(coerced)["error_code"] == "parameter.invalid_operation_parameters"
    assert _validation_errors(coerced)[0]["type"] == "int_type"
    assert unknown["ok"] is False
    assert _error(unknown)["error_code"] == "parameter.invalid_operation_parameters"
    assert _validation_errors(unknown)[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize("invocation_path", ["direct", "registry"])
def test_constructed_invalid_structured_parameter_is_revalidated(
    analyze_spec: OperationSpec,
    genome: OrganelleGenome,
    invocation_path: str,
) -> None:
    registry = OperationRegistry()
    executed: list[bool] = []

    @operation(spec=analyze_spec, registry=registry)
    def annotate(
        genome: OrganelleGenome,
        *,
        backend_parameters: OatkBackendParameters | None = None,
    ) -> OrganelleResult:
        executed.append(True)
        return OrganelleResult(
            operation_id="annotation.annotate",
            scope=genome.organelle,
            status="ok",
        )

    assert registry.require("annotation.annotate").function is annotate
    constructed = OatkBackendParameters.model_construct(minimum_kmer_coverage=0)

    def invoke() -> object:
        if invocation_path == "direct":
            return annotate(genome, backend_parameters=constructed)
        return registry.invoke(
            "annotation.annotate",
            input=genome,
            parameters={"backend_parameters": constructed},
        )

    with pytest.raises(OrganelleParameterError) as raised:
        invoke()

    assert raised.value.code == "parameter.invalid_operation_parameters"
    assert executed == []


def test_needs_input_raises_directly_and_returns_agent_envelope(
    analyze_spec: OperationSpec,
    genome: OrganelleGenome,
    genome_json: dict[str, object],
) -> None:
    registry = OperationRegistry()
    needs_input_spec = analyze_spec.model_copy(update={"operation_id": "test.needs_input"})

    request = make_needs_input_request(
        semantic_payload={"organelle": "mitochondrion", "requested_method": "auto"},
        field="auxiliary.genome_size",
        choices=("use_default_oatk", "provide_genome_size", "lookup_by_species"),
    )

    @operation(spec=needs_input_spec, registry=registry)
    def needs_input(genome: OrganelleGenome) -> OrganelleResult:
        raise OrganelleNeedsInput(
            code="assembly.genome_size_required",
            message="nuclear genome size is required for coverage routing",
            needs_input=request,
        )

    assert registry.require("test.needs_input").function is needs_input

    # Direct and Registry invocations surface the structured exception, not a
    # fake result envelope, because missing human input is not a transport error.
    with pytest.raises(OrganelleNeedsInput) as direct:
        needs_input(genome)
    assert direct.value.needs_input is request
    with pytest.raises(OrganelleNeedsInput):
        registry.invoke("test.needs_input", input=genome, parameters={})

    # The Agent JSON transport converts the same exception into a machine-readable
    # needs_input envelope that precedes the generic error payload.
    response = invoke_json(
        {"operation_id": "test.needs_input", "input": genome_json, "parameters": {}},
        registry=registry,
        granted_side_effects=set(),
    )

    assert response == {
        "ok": False,
        "operation_id": "test.needs_input",
        "error": {
            "error_code": "assembly.genome_size_required",
            "message": "nuclear genome size is required for coverage routing",
            "details": {},
            "retryable": False,
            "suggested_action": {},
        },
        "needs_input": request.model_dump(mode="json"),
    }
    json.dumps(response, allow_nan=False)


def test_needs_input_envelope_precedes_ordinary_error_handling(
    analyze_spec: OperationSpec,
    genome_json: dict[str, object],
) -> None:
    """The specialized catch must run before the generic OrganelleError handler."""
    registry = OperationRegistry()
    spec = analyze_spec.model_copy(update={"operation_id": "test.needs_input_first"})

    @operation(spec=spec, registry=registry)
    def needs_input(genome: OrganelleGenome) -> OrganelleResult:
        raise OrganelleNeedsInput(
            code="assembly.genome_size_required",
            message="nuclear genome size is required for coverage routing",
            needs_input=make_needs_input_request(
                semantic_payload={"requested_method": "auto"},
                field="auxiliary.genome_size",
                choices=("use_default_oatk", "provide_genome_size"),
            ),
        )

    assert registry.require("test.needs_input_first").function is needs_input

    response = invoke_json(
        {"operation_id": "test.needs_input_first", "input": genome_json, "parameters": {}},
        registry=registry,
        granted_side_effects=set(),
    )
    assert response["ok"] is False
    assert "needs_input" in response
