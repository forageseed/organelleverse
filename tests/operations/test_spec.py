import pytest
from pydantic import ValidationError

from organelleverse.operations.spec import (
    CoreKind,
    DependencyKind,
    DependencySpec,
    ExecutionMode,
    FallbackPolicy,
    OperationSpec,
    OperationStage,
    RetryPolicy,
    SideEffect,
)


def valid_analyze_spec(**changes: object) -> OperationSpec:
    values: dict[str, object] = {
        "operation_id": "annotation.annotate",
        "contract_version": "1.0",
        "title": "Annotate an organelle genome",
        "description": "Test fixture spec standing in for a real ANALYZE operation.",
        "keywords": ("analyze", "fixture", "test"),
        "execution_mode": ExecutionMode.INLINE,
        "stage": OperationStage.ANALYZE,
        "input_kind": CoreKind.GENOME,
        "output_kind": CoreKind.RESULT,
        "callable_locator": "organelleverse.annotation.api:annotate",
        "side_effects": (SideEffect.READ_FILES,),
    }
    values.update(changes)
    return OperationSpec.model_validate(values)


def test_operation_spec_has_no_maturity_contract() -> None:
    spec = valid_analyze_spec()

    assert "maturity" not in OperationSpec.model_fields
    assert "maturity" not in OperationSpec.model_json_schema()["properties"]
    assert "maturity" not in spec.model_dump(mode="json")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        valid_analyze_spec(maturity="validated")


def test_references_are_optional_scientific_citations() -> None:
    assert valid_analyze_spec().references == ()
    assert valid_analyze_spec(references=("doi:10.0000/example",)).references == (
        "doi:10.0000/example",
    )


@pytest.mark.parametrize("references", [("",), ("   ",), ("paper", "\t")])
def test_reference_entries_must_not_be_blank(references: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError, match="reference"):
        valid_analyze_spec(references=references)


def test_external_dependency_does_not_require_a_status_label() -> None:
    dependency = DependencySpec(
        kind=DependencyKind.EXECUTABLE,
        name="blastn",
        version_spec=">=2.15",
    )

    assert valid_analyze_spec(dependencies=(dependency,)).dependencies == (dependency,)


def test_read_and_transform_state_constraints() -> None:
    with pytest.raises(ValidationError, match="read operation"):
        valid_analyze_spec(stage="read", input_kind="genome", output_kind="result")
    with pytest.raises(ValidationError, match="transform operation"):
        valid_analyze_spec(stage="transform", input_kind="data", output_kind="result")


def test_analyze_with_no_core_input_is_valid_for_named_parameter_binding() -> None:
    """analyze+none+result is legal: a pure-parameter analysis with no L1 object.

    This is what makes a named-parameter binding like composition.compute_gc_content
    (Capability Plan 02) representable as an OperationSpec at all.
    """
    spec = valid_analyze_spec(input_kind="none")
    assert spec.input_kind.value == "none"


@pytest.mark.parametrize(
    ("stage", "input_kind", "output_kind", "message"),
    [
        ("read", "genome", "genome", "read operation"),
        ("read", "none", "result", "read operation"),
        ("transform", "none", "genome", "transform operation"),
        ("transform", "genome", "result", "transform operation"),
        # analyze+none+result is deliberately NOT here: Capability Plan 02 Task 1
        # made it valid (named-parameter binding with no L1 core object, e.g.
        # composition.compute_gc_content). analyze+result+* stays rejected —
        # RESULT was never an admitted ANALYZE input kind.
        ("analyze", "result", "result", "analyze operation"),
        ("analyze", "genome", "data", "analyze operation"),
        ("consume", "genome", "result", "consume operation"),
        ("consume", "result", "data", "consume operation"),
    ],
)
def test_every_invalid_stage_input_and_output_branch_is_rejected(
    stage: str,
    input_kind: str,
    output_kind: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        valid_analyze_spec(stage=stage, input_kind=input_kind, output_kind=output_kind)


def test_fallback_requires_warning_semantics() -> None:
    with pytest.raises(ValidationError, match="fallback"):
        valid_analyze_spec(
            fallback=FallbackPolicy(allowed=True, allowed_backends=()),
        )


def test_cacheable_rejects_external_mutating_side_effects() -> None:
    with pytest.raises(ValidationError, match="cacheable"):
        valid_analyze_spec(cacheable=True, side_effects=(SideEffect.NETWORK,))


def test_cacheable_rejects_nondeterministic_operation() -> None:
    with pytest.raises(ValidationError, match="deterministic"):
        valid_analyze_spec(cacheable=True, deterministic=False)


def test_spec_models_are_frozen_and_forbid_extra_fields() -> None:
    spec = valid_analyze_spec()

    with pytest.raises(ValidationError, match="frozen"):
        spec.deterministic = False  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        valid_analyze_spec(unexpected=True)


@pytest.mark.parametrize("copy_method", ["model_copy", "copy"])
def test_spec_copy_updates_revalidate_stage_kind_and_extra_fields(copy_method: str) -> None:
    spec = valid_analyze_spec()
    copy = getattr(spec, copy_method)

    with pytest.raises(ValidationError, match="read operation"):
        copy(update={"stage": OperationStage.READ})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        copy(update={"unexpected": True})


@pytest.mark.parametrize("copy_method", ["model_copy", "copy"])
def test_spec_copy_updates_revalidate_references(copy_method: str) -> None:
    spec = valid_analyze_spec(references=("benchmark",))

    with pytest.raises(ValidationError, match="reference"):
        getattr(spec, copy_method)(update={"references": ("   ",)})


@pytest.mark.parametrize("copy_method", ["model_copy", "copy"])
def test_nested_spec_copy_updates_revalidate_constraints_and_extras(copy_method: str) -> None:
    retry = valid_analyze_spec().retry

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        getattr(retry, copy_method)(update={"max_attempts": 0})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        getattr(retry, copy_method)(update={"unexpected": True})


def test_revalidated_model_copy_preserves_deep_argument_and_frozen_behavior() -> None:
    copied = valid_analyze_spec().model_copy(
        deep=True,
        update={"references": ("doi:10.0000/example",)},
    )

    assert copied.references == ("doi:10.0000/example",)
    with pytest.raises(ValidationError, match="frozen"):
        copied.references = ()  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("copy_method", ["model_copy", "copy"])
def test_spec_copy_rejects_invalid_constructed_nested_model_instances(
    copy_method: str,
) -> None:
    spec = valid_analyze_spec()
    invalid_retry = RetryPolicy.model_construct(max_attempts=0)
    invalid_dependency = DependencySpec.model_construct(
        kind=DependencyKind.PYTHON,
        name="",
    )

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        getattr(spec, copy_method)(update={"retry": invalid_retry})
    with pytest.raises(ValidationError, match="at least 1 character"):
        getattr(spec, copy_method)(update={"dependencies": (invalid_dependency,)})


def test_model_validation_revalidates_constructed_top_level_and_nested_instances() -> None:
    spec = valid_analyze_spec()
    values = {name: getattr(spec, name) for name in OperationSpec.model_fields}
    invalid = OperationSpec.model_construct(
        **{name: value for name, value in values.items() if name != "retry"},
        retry=RetryPolicy.model_construct(max_attempts=0),
    )

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        OperationSpec.model_validate(invalid)


@pytest.mark.parametrize("copy_method", ["model_copy", "copy"])
def test_spec_copy_revalidates_nested_mappings_and_preserves_valid_immutable_copy(
    copy_method: str,
) -> None:
    spec = valid_analyze_spec()

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        getattr(spec, copy_method)(update={"retry": {"max_attempts": 0}})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        getattr(spec, copy_method)(
            update={
                "dependencies": (
                    {"kind": DependencyKind.PYTHON, "name": "backend", "unexpected": True},
                )
            }
        )

    copied = getattr(spec, copy_method)(
        update={"retry": RetryPolicy(max_attempts=2)},
        deep=True,
    )
    assert copied.retry.max_attempts == 2
    with pytest.raises(ValidationError, match="frozen"):
        copied.retry.max_attempts = 3  # pyright: ignore[reportAttributeAccessIssue]


def _spec(**overrides: object) -> OperationSpec:
    base: dict[str, object] = {
        "operation_id": "demo.op",
        "contract_version": "1.0",
        "title": "Demo probe operation",
        "description": "Test fixture spec used only to probe spec validation rules.",
        "keywords": ("demo", "fixture", "test"),
        "execution_mode": ExecutionMode.INLINE,
        "stage": OperationStage.READ,
        "input_kind": CoreKind.NONE,
        "output_kind": CoreKind.DATA,
        "output_modalities": ("demo_records",),
        "callable_locator": "demo.api:op",
    }
    base.update(overrides)
    return OperationSpec.model_validate(base)


def test_data_output_requires_exactly_one_output_modality() -> None:
    assert _spec().output_modalities == ("demo_records",)
    with pytest.raises(ValidationError, match="output_modalities"):
        _spec(output_modalities=())
    with pytest.raises(ValidationError, match="output_modalities"):
        _spec(output_modalities=("a", "b"))


def test_non_data_output_forbids_output_modalities() -> None:
    with pytest.raises(ValidationError, match="output_modalities"):
        _spec(
            stage=OperationStage.ANALYZE,
            input_kind=CoreKind.GENOME,
            output_kind=CoreKind.RESULT,
            output_modalities=("demo_records",),
        )


def test_input_modalities_require_data_input() -> None:
    with pytest.raises(ValidationError, match="input_modalities"):
        _spec(input_modalities=("demo_records",))


def test_legacy_modalities_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _spec(modalities=("demo_records",))
