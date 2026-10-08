import inspect
from collections.abc import Callable
from typing import Literal, cast

import pytest
from pydantic import ValidationError

from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import (
    OrganelleContractError,
    OrganelleInputError,
    OrganelleParameterError,
)
from organelleverse.core.frozen import FrozenJson, FrozenMap
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OrganelleResult
from organelleverse.operations import (
    CoreKind,
    ExecutionMode,
    OperationRegistry,
    OperationSpec,
    OperationStage,
    RetryPolicy,
    operation,
    registry,
)


def _forge_spec(spec: OperationSpec, **updates: object) -> OperationSpec:
    values = {name: getattr(spec, name) for name in OperationSpec.model_fields}
    values.update(updates)
    return OperationSpec.model_construct(**values)


def _result(genome: OrganelleGenome, backend: str = "native") -> OrganelleResult:
    return OrganelleResult(
        operation_id="annotation.annotate",
        scope=genome.organelle,
        status="ok",
        metrics=FrozenMap({"backend": backend}),
    )


def test_direct_and_registry_calls_are_equivalent(
    analyze_spec: OperationSpec, genome: OrganelleGenome
) -> None:
    local = OperationRegistry()

    @operation(spec=analyze_spec, registry=local)
    def annotate(
        genome: OrganelleGenome,
        *,
        backend: Literal["native", "external"] = "native",
    ) -> OrganelleResult:
        return _result(genome, backend)

    direct = annotate(genome, backend="native")
    registered = local.invoke(
        "annotation.annotate",
        input=genome,
        parameters={"backend": "native"},
    )

    assert registered == direct
    assert local.describe("annotation.annotate") == analyze_spec
    assert local.require("annotation.annotate").function is annotate


def test_registry_rejects_duplicate_operation_ids_with_valid_typed_functions(
    analyze_spec: OperationSpec,
) -> None:
    local = OperationRegistry()

    def first(genome: OrganelleGenome) -> OrganelleResult:
        return _result(genome)

    def second(genome: OrganelleGenome) -> OrganelleResult:
        return _result(genome)

    local.register(analyze_spec, first)

    with pytest.raises(OrganelleContractError) as raised:
        local.register(analyze_spec, second)

    assert raised.value.code == "contract.duplicate_operation_id"


@pytest.mark.parametrize("invalid_field", ["operation_id", "retry"])
def test_registry_revalidates_forged_operation_specs(
    analyze_spec: OperationSpec,
    invalid_field: str,
) -> None:
    def annotate(genome: OrganelleGenome) -> OrganelleResult:
        return _result(genome)

    update = (
        {"operation_id": "INVALID"}
        if invalid_field == "operation_id"
        else {"retry": RetryPolicy.model_construct(max_attempts=0)}
    )
    forged = _forge_spec(analyze_spec, **update)

    with pytest.raises(ValidationError):
        OperationRegistry().register(forged, annotate)


def test_registry_stores_a_newly_validated_operation_spec(
    analyze_spec: OperationSpec,
) -> None:
    local = OperationRegistry()

    def annotate(genome: OrganelleGenome) -> OrganelleResult:
        return _result(genome)

    binding = local.register(analyze_spec, annotate)

    assert binding.spec == analyze_spec
    assert binding.spec is not analyze_spec
    assert local.describe(analyze_spec.operation_id) is binding.spec


def test_operation_decorator_revalidates_its_spec_boundary(
    analyze_spec: OperationSpec,
) -> None:
    forged = _forge_spec(
        analyze_spec,
        retry=RetryPolicy.model_construct(max_attempts=0),
    )

    def annotate(genome: OrganelleGenome) -> OrganelleResult:
        return _result(genome)

    with pytest.raises(ValidationError):
        operation(spec=forged, registry=OperationRegistry())(annotate)


def test_registry_lists_sorted_operations_with_exact_filters(
    analyze_spec: OperationSpec,
    read_spec: OperationSpec,
    genome: OrganelleGenome,
) -> None:
    local = OperationRegistry()
    read = read_spec.model_copy(update={"organelle_types": ("plastid",)})
    analyze = analyze_spec.model_copy(update={"organelle_types": ("mitochondrion",)})
    consume = OperationSpec(
        operation_id="report.summarize",
        contract_version="1.0",
        title="Summarize a report",
        description="Test fixture spec used only to probe registry listing filters.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.CONSUME,
        input_kind=CoreKind.RESULT,
        output_kind=CoreKind.RESULT,
        callable_locator="organelleverse.reporting.api:summarize",
        organelle_types=("mitochondrion",),
    )

    def load(path: str, *, organelle: Literal["mitochondrion", "plastid"]) -> OrganelleGenome:
        return genome

    def annotate(genome: OrganelleGenome) -> OrganelleResult:
        return _result(genome)

    def summarize(result: OrganelleResult) -> OrganelleResult:
        return result

    local.register(read, load)
    local.register(consume, summarize)
    local.register(analyze, annotate)

    assert [spec.operation_id for spec in local.list()] == [
        "annotation.annotate",
        "io.read_genome",
        "report.summarize",
    ]
    assert local.list(stage=OperationStage.READ) == (read,)
    assert local.list(input_kind=CoreKind.RESULT) == (consume,)
    assert local.list(organelle="plastid") == (read,)


def test_list_filters_input_and_output_modalities_independently() -> None:
    """The split list() filters are independent: input_modality matches only
    spec.input_modalities and output_modality matches only spec.output_modalities.

    The read/analyze/consume fixtures above cannot carry modalities (their kinds
    forbid it under I1a/I2a), so a DATA-output READ operation exercises the
    filters that the old single ``modality=`` parameter covered.
    """
    local = OperationRegistry()
    read_data = OperationSpec(
        operation_id="io.read_records",
        contract_version="1.0",
        title="Read demo records",
        description="Test fixture spec used only to probe registry listing filters.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.READ,
        input_kind=CoreKind.NONE,
        output_kind=CoreKind.DATA,
        output_modalities=("fasta",),
        callable_locator="demo.api:read_records",
    )

    def read_records(path: str) -> OrganelleData:
        return OrganelleData(modality="fasta", payload=cast("FrozenMap[FrozenJson]", {}))

    local.register(read_data, read_records)
    assert local.list(output_modality="fasta") == (read_data,)
    assert local.list(input_modality="fasta") == ()
    assert local.list(output_modality="absent") == ()


def test_read_operations_invoke_from_mapping_parameters(
    read_spec: OperationSpec, genome: OrganelleGenome
) -> None:
    local = OperationRegistry()

    @operation(spec=read_spec, registry=local)
    def load(path: str, *, organelle: Literal["mitochondrion", "plastid"]) -> OrganelleGenome:
        assert path == "genome.fa"
        assert organelle == "mitochondrion"
        return genome

    assert local.require("io.read_genome").function is load
    result = local.invoke(
        "io.read_genome",
        input=None,
        parameters={"path": "genome.fa", "organelle": "mitochondrion"},
    )
    assert type(result) is OrganelleGenome
    assert result == genome


def test_read_registry_invoke_rejects_core_input_without_executing(
    read_spec: OperationSpec, genome: OrganelleGenome
) -> None:
    local = OperationRegistry()
    calls: list[str] = []

    @operation(spec=read_spec, registry=local)
    def load(path: str, *, organelle: Literal["mitochondrion", "plastid"]) -> OrganelleGenome:
        calls.append(path)
        return genome

    assert local.require("io.read_genome").function is load
    with pytest.raises(OrganelleInputError) as raised:
        local.invoke(
            "io.read_genome",
            input=genome,
            parameters={"path": "genome.fa", "organelle": "mitochondrion"},
        )

    assert raised.value.code == "input.unexpected_operation_core_input"
    assert raised.value.as_dict()["details"] == {
        "operation_id": "io.read_genome",
        "provided_input_type": "OrganelleGenome",
    }
    assert calls == []


def test_calls_convert_parameter_failures_to_stable_errors(
    analyze_spec: OperationSpec, genome: OrganelleGenome
) -> None:
    local = OperationRegistry()

    @operation(spec=analyze_spec, registry=local)
    def annotate(
        genome: OrganelleGenome,
        *,
        backend: Literal["native", "external"] = "native",
    ) -> OrganelleResult:
        return _result(genome, backend)

    assert local.require("annotation.annotate").function is annotate
    for invoke in (
        lambda: annotate(genome, backend="invalid"),  # pyright: ignore[reportArgumentType]
        lambda: local.invoke(
            "annotation.annotate",
            input=genome,
            parameters={"backend": "invalid"},
        ),
    ):
        with pytest.raises(OrganelleParameterError) as raised:
            invoke()
        assert raised.value.code == "parameter.invalid_operation_parameters"
        assert raised.value.as_dict()["details"]["validation_errors"]


def test_calls_enforce_exact_core_input_type(
    analyze_spec: OperationSpec, genome: OrganelleGenome
) -> None:
    local = OperationRegistry()

    @operation(spec=analyze_spec, registry=local)
    def annotate(genome: OrganelleGenome) -> OrganelleResult:
        return _result(genome)

    assert local.require("annotation.annotate").function is annotate
    wrong_input = OrganelleResult(
        operation_id="annotation.annotate",
        scope="mitochondrion",
        status="ok",
    )
    for invoke in (
        lambda: annotate(wrong_input),  # pyright: ignore[reportArgumentType]
        lambda: local.invoke("annotation.annotate", input=wrong_input, parameters={}),
    ):
        with pytest.raises(OrganelleInputError) as raised:
            invoke()
        assert raised.value.code == "input.operation_core_type_mismatch"


def test_calls_revalidate_exact_class_core_input_before_execution(
    analyze_spec: OperationSpec,
) -> None:
    local = OperationRegistry()
    calls: list[str] = []

    @operation(spec=analyze_spec, registry=local)
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
    for invoke in (
        lambda: annotate(invalid),
        lambda: local.invoke("annotation.annotate", input=invalid, parameters={}),
    ):
        with pytest.raises(OrganelleInputError) as raised:
            invoke()
        assert raised.value.code == "input.invalid_core_object"
        assert raised.value.as_dict()["details"]["validation_errors"]
    assert calls == []


def test_calls_enforce_exact_return_type(
    analyze_spec: OperationSpec, genome: OrganelleGenome
) -> None:
    local = OperationRegistry()

    @operation(spec=analyze_spec, registry=local)
    def incorrect_return(genome: OrganelleGenome) -> OrganelleResult:
        return genome  # type: ignore[return-value]

    assert local.require("annotation.annotate").function is incorrect_return
    for invoke in (
        lambda: incorrect_return(genome),
        lambda: local.invoke("annotation.annotate", input=genome, parameters={}),
    ):
        with pytest.raises(OrganelleContractError) as raised:
            invoke()
        assert raised.value.code == "contract.operation_output_type_mismatch"


def test_calls_revalidate_exact_class_output_before_exposure(
    analyze_spec: OperationSpec,
    genome: OrganelleGenome,
) -> None:
    local = OperationRegistry()

    @operation(spec=analyze_spec, registry=local)
    def invalid_output(genome: OrganelleGenome) -> OrganelleResult:
        return OrganelleResult.model_construct(
            operation_id="annotation.annotate",
            scope=genome.organelle,
            status="failed",
            errors=(),
        )

    for invoke in (
        lambda: invalid_output(genome),
        lambda: local.invoke("annotation.annotate", input=genome, parameters={}),
    ):
        with pytest.raises(OrganelleContractError) as raised:
            invoke()
        assert raised.value.code == "contract.invalid_operation_output"
        assert raised.value.as_dict()["details"]["validation_errors"]


def test_unknown_operation_id_is_a_structured_input_error(genome: OrganelleGenome) -> None:
    with pytest.raises(OrganelleInputError) as raised:
        OperationRegistry().invoke("unknown.operation", input=genome, parameters={})

    assert raised.value.code == "input.unknown_operation"


def test_direct_missing_core_input_uses_stable_core_input_error(
    local_registry: OperationRegistry,
) -> None:
    function = cast(Callable[..., object], local_registry.require("annotation.annotate").function)

    with pytest.raises(OrganelleInputError) as raised:
        function()

    assert raised.value.code == "input.operation_core_type_mismatch"


def test_decorator_preserves_public_signature_docstring_and_annotations(
    analyze_spec: OperationSpec,
) -> None:
    local = OperationRegistry()

    def annotate(
        genome: OrganelleGenome,
        *,
        backend: Literal["native", "external"] = "native",
    ) -> OrganelleResult:
        """Annotate a genome with the selected backend."""
        return _result(genome, backend)

    expected_signature = inspect.signature(annotate)
    decorated = operation(spec=analyze_spec, registry=local)(annotate)

    assert inspect.signature(decorated) == expected_signature
    assert decorated.__doc__ == annotate.__doc__
    assert decorated.__annotations__ == annotate.__annotations__


def test_package_exports_a_default_registry() -> None:
    assert isinstance(registry, OperationRegistry)


def test_all_release_bindings_remain_in_process_with_exact_function_identity() -> None:
    bindings = tuple(registry.require(spec.operation_id) for spec in registry.list())
    function_ids = {binding.spec.operation_id: binding.function for binding in bindings}

    assert len(bindings) == 20
    assert all(binding.invocation_strategy is None for binding in bindings)
    assert all(
        registry.require(operation_id).function is function
        for operation_id, function in function_ids.items()
    )
