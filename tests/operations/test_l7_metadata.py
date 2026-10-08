"""L7 discovery metadata and the bundle-binding schema fields on OperationSpec.

Every admitted operation must carry normalized title/description/keywords/
execution_mode (L7.1 Shared Foundation Task 1's contribution) and a
PythonBindingSpec-typed binding with a now-optional callable_locator
(Capability Plan 02 Task 1's contribution) — one schema revision, not two.

Specs are built via ``OperationSpec.model_validate(dict)`` rather than
``OperationSpec(**dict)`` throughout: the kwargs dict is deliberately
untyped (``dict[str, object]``, since individual tests corrupt one field at
a time to provoke a specific validation error), and ``model_validate``
accepts ``Any`` where keyword-unpacking a loosely-typed dict against
strictly-typed constructor parameters would not.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from organelleverse.operations import ExecutionMode, OperationSpec
from organelleverse.operations.spec import (
    ArgumentMode,
    ParameterBindingSpec,
    ParameterCodec,
    ParameterSource,
    PythonBindingSpec,
    ResultCodec,
)


def test_every_admitted_operation_has_normalized_l7_metadata() -> None:
    import organelleverse.operations as operations

    specs = operations.list()
    assert specs
    for spec in specs:
        assert spec.title.strip() == spec.title
        assert spec.title != ""
        assert spec.description.strip() == spec.description
        assert len(spec.description) >= 20
        assert spec.keywords == tuple(sorted(set(spec.keywords)))
        assert 3 <= len(spec.keywords) <= 8
        assert spec.execution_mode in {ExecutionMode.INLINE, ExecutionMode.DURABLE}
        assert OperationSpec.model_validate(spec.model_dump()) == spec


def _minimal_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "operation_id": "demo.probe",
        "contract_version": "1.0",
        "title": "Probe demo operation",
        "description": "A minimal analyze operation used only to test spec validation rules.",
        "keywords": ("demo", "probe", "spec"),
        "execution_mode": ExecutionMode.INLINE,
        "stage": "analyze",
        "input_kind": "genome",
        "output_kind": "result",
        "callable_locator": "demo.api:probe",
    }
    base.update(overrides)
    return base


def test_title_rejects_leading_or_trailing_whitespace() -> None:
    with pytest.raises(ValidationError):
        OperationSpec.model_validate(_minimal_kwargs(title=" Probe demo operation"))


def test_description_rejects_leading_or_trailing_whitespace() -> None:
    with pytest.raises(ValidationError):
        OperationSpec.model_validate(
            _minimal_kwargs(description=" a description with trailing space ")
        )


def test_description_must_be_at_least_twenty_characters() -> None:
    with pytest.raises(ValidationError):
        OperationSpec.model_validate(_minimal_kwargs(description="too short"))


def test_keywords_reject_blank_entries() -> None:
    with pytest.raises(ValidationError):
        OperationSpec.model_validate(_minimal_kwargs(keywords=("demo", "")))


def test_keywords_reject_duplicates() -> None:
    with pytest.raises(ValidationError):
        OperationSpec.model_validate(_minimal_kwargs(keywords=("demo", "demo", "probe")))


def test_keywords_reject_unsorted_order() -> None:
    with pytest.raises(ValidationError):
        OperationSpec.model_validate(_minimal_kwargs(keywords=("probe", "demo")))


def test_keywords_reject_fewer_than_three() -> None:
    with pytest.raises(ValidationError):
        OperationSpec.model_validate(_minimal_kwargs(keywords=("demo", "probe")))


def test_keywords_reject_more_than_eight() -> None:
    with pytest.raises(ValidationError):
        OperationSpec.model_validate(_minimal_kwargs(keywords=tuple(f"k{i}" for i in range(9))))


def test_keywords_reject_uppercase() -> None:
    with pytest.raises(ValidationError):
        OperationSpec.model_validate(_minimal_kwargs(keywords=("Demo", "probe", "x")))


def test_keywords_reject_a_leading_symbol() -> None:
    with pytest.raises(ValidationError):
        OperationSpec.model_validate(_minimal_kwargs(keywords=("-demo", "probe", "x")))


def test_execution_mode_rejects_an_unknown_value() -> None:
    with pytest.raises(ValidationError):
        OperationSpec.model_validate(_minimal_kwargs(execution_mode="sometimes"))


def test_execution_mode_is_required_with_no_default() -> None:
    kwargs = _minimal_kwargs()
    del kwargs["execution_mode"]
    with pytest.raises(ValidationError):
        OperationSpec.model_validate(kwargs)


def test_title_is_required_with_no_default() -> None:
    kwargs = _minimal_kwargs()
    del kwargs["title"]
    with pytest.raises(ValidationError):
        OperationSpec.model_validate(kwargs)


def test_callable_locator_is_now_optional() -> None:
    spec = OperationSpec.model_validate(_minimal_kwargs(callable_locator=None))
    assert spec.callable_locator is None


def test_callable_locator_pattern_still_applies_when_present() -> None:
    with pytest.raises(ValidationError):
        OperationSpec.model_validate(_minimal_kwargs(callable_locator="not a locator"))


def test_binding_defaults_to_canonical_core() -> None:
    spec = OperationSpec.model_validate(_minimal_kwargs())
    assert spec.binding == PythonBindingSpec()
    assert spec.binding.argument_mode is ArgumentMode.CANONICAL_CORE
    assert spec.binding.result_codec is ResultCodec.CANONICAL


def test_analyze_stage_permits_none_input_kind_for_named_parameters() -> None:
    spec = OperationSpec.model_validate(
        _minimal_kwargs(
            input_kind="none",
            binding=PythonBindingSpec(
                argument_mode=ArgumentMode.NAMED_PARAMETERS,
                parameters=(
                    ParameterBindingSpec(
                        name="sequence",
                        codec=ParameterCodec.JSON,
                        source=ParameterSource.AGENT,
                    ),
                ),
                result_codec=ResultCodec.JSON_METRIC,
                result_key="gc_content",
            ),
        )
    )
    assert spec.input_kind.value == "none"
    assert spec.binding.argument_mode is ArgumentMode.NAMED_PARAMETERS
