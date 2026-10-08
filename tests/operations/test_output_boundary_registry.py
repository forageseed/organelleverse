"""Registry output-boundary ratchet over released OperationSpecs.

Two complementary scanners derive their facts from each released spec's
registered JSON Schema (what an Agent sees), not from Python introspection:

- scientific specs must not expose a final-write field;
- writer specs (consume/write contract) must expose a closed destination.

A writer is classified by the consume/write contract: a CONSUME Result->Result
stage whose operation id declares writer intent (``.write``/``.save``/
``.export``/``.materialize``). The contract is required, not merely the suffix:
``qc.assembly`` shares the consume Result->Result stage but declares no writer
intent and no destination, so it stays scientific — and now that its compute
function publishes through the managed run store, it exposes no final-write
field either.
"""

from __future__ import annotations

import pytest

import organelleverse.operations.output_boundary as output_boundary
from organelleverse.core.data import OrganelleData
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.output_boundary import (
    is_writer_spec,
    released_scientific_specs_with_final_write,
    released_writer_specs_missing_destination,
)
from organelleverse.operations.registry import OperationRegistry
from organelleverse.operations.registry import registry as default_registry
from organelleverse.operations.spec import CoreKind, ExecutionMode, OperationSpec, OperationStage

# Every released scientific operation now computes through the managed run
# store and exposes no final-write field. The set must stay empty: a new
# released scientific spec that smuggles in a destination is a regression.
FROZEN_RELEASED_SCIENTIFIC_PENDING: frozenset[str] = frozenset()


@pytest.mark.parametrize("parameter", sorted(output_boundary.FORBIDDEN_FINAL_WRITE_PARAMS))
def test_capability_output_boundary_uses_every_forbidden_final_write_name(
    parameter: str,
) -> None:
    spec = OperationSpec(
        operation_id="suite.compute",
        contract_version="1.0",
        title="Capability compute fixture",
        description="Fixture proving capability-scoped compute-first enforcement.",
        keywords=("capability", "compute", "fixture"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.ANALYZE,
        input_kind=CoreKind.NONE,
        output_kind=CoreKind.RESULT,
        callable_locator="private_pkg.impl:run",
    )
    schema = {"type": "object", "properties": {parameter: {"type": "string"}}}

    assert output_boundary.capability_final_write_parameters(spec, schema) == frozenset(
        {parameter}
    )


def test_capability_output_boundary_has_no_read_or_writer_exemption() -> None:
    base = OperationSpec(
        operation_id="suite.fetch",
        contract_version="1.0",
        title="Capability read fixture",
        description="Fixture proving controlled workers never accept final destinations.",
        keywords=("capability", "fixture", "read"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.READ,
        input_kind=CoreKind.NONE,
        output_kind=CoreKind.DATA,
        output_modalities=("snapshot",),
        callable_locator="private_pkg.impl:run",
    )
    schema = {"type": "object", "properties": {"output_dir": {"type": "string"}}}
    writer = base.model_copy(
        update={
            "operation_id": "suite.write",
            "stage": OperationStage.CONSUME,
            "input_kind": CoreKind.RESULT,
            "output_kind": CoreKind.RESULT,
            "output_modalities": (),
        }
    )

    assert output_boundary.capability_final_write_parameters(base, schema) == frozenset(
        {"output_dir"}
    )
    assert output_boundary.capability_final_write_parameters(writer, schema) == frozenset(
        {"output_dir"}
    )


def test_released_scientific_specs_with_final_write_match_frozen_pending() -> None:
    actual = released_scientific_specs_with_final_write(default_registry)
    assert actual == FROZEN_RELEASED_SCIENTIFIC_PENDING, (
        f"released scientific pending inventory drifted: actual={sorted(actual)}"
    )


def test_released_writers_expose_a_destination() -> None:
    """``annotation.write`` and ``qc.write`` are the released writers today."""
    actual = released_writer_specs_missing_destination(default_registry)
    assert actual == frozenset(), f"released writer specs missing a destination: {sorted(actual)}"


def test_writer_classification_needs_both_contract_and_intent() -> None:
    """The consume/write stage contract and writer intent must both hold."""
    consume_write = OperationSpec(
        operation_id="suite.write",
        contract_version="1.0",
        title="Demo write operation",
        description="Test fixture spec used only to probe writer classification.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.CONSUME,
        input_kind=CoreKind.RESULT,
        output_kind=CoreKind.RESULT,
        callable_locator="organelleverse.suite.api:write",
    )
    analyze_write = consume_write.model_copy(
        update={
            "operation_id": "suite.annotate",
            "stage": OperationStage.ANALYZE,
            "input_kind": CoreKind.GENOME,
        }
    )
    consume_consumer = consume_write.model_copy(update={"operation_id": "qc.assembly"})

    assert is_writer_spec(consume_write) is True
    # A bare ``.write`` id on the wrong stage/kind is scientific, not a writer.
    assert is_writer_spec(analyze_write) is False
    # The consume Result->Result stage alone is not enough without writer intent.
    assert is_writer_spec(consume_consumer) is False


def test_released_writer_auditor_reports_missing_destination() -> None:
    """The auditor itself must fail closed for a malformed writer schema."""
    registry = OperationRegistry()

    def write(result: OrganelleResult) -> OrganelleResult:
        return result

    spec = OperationSpec(
        operation_id="suite.write",
        contract_version="1.0",
        title="Demo write operation",
        description="Test fixture spec used only to probe writer classification.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.CONSUME,
        input_kind=CoreKind.RESULT,
        output_kind=CoreKind.RESULT,
        callable_locator="organelleverse.suite.api:write",
    )
    registry.register(spec, write)

    assert released_writer_specs_missing_destination(registry) == frozenset({"suite.write"})


def test_writer_output_dir_is_a_valid_destination() -> None:
    registry = OperationRegistry()

    def write(result: OrganelleResult, *, output_dir: str) -> OrganelleResult:
        del output_dir
        return result

    spec = OperationSpec(
        operation_id="suite.write",
        contract_version="1.0",
        title="Demo write operation",
        description="Test fixture spec used only to probe writer classification.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.CONSUME,
        input_kind=CoreKind.RESULT,
        output_kind=CoreKind.RESULT,
        callable_locator="organelleverse.suite.api:write",
    )
    registry.register(spec, write)

    assert released_writer_specs_missing_destination(registry) == frozenset()


def test_read_primitive_destination_is_not_scientific_output() -> None:
    registry = OperationRegistry()

    def fetch(output_dir: str) -> OrganelleData:
        raise AssertionError(output_dir)

    spec = OperationSpec(
        operation_id="suite.fetch",
        contract_version="1.0",
        title="Demo fetch operation",
        description="Test fixture spec used only to probe reader-primitive classification.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.DURABLE,
        stage=OperationStage.READ,
        input_kind=CoreKind.NONE,
        output_kind=CoreKind.DATA,
        output_modalities=("snapshot",),
        callable_locator="organelleverse.suite.api:fetch",
    )
    registry.register(spec, fetch)

    assert released_scientific_specs_with_final_write(registry) == frozenset()
