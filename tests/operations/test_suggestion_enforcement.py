"""Task 4: a returned Result's suggestions must be followable by an agent.

Three rules are enforced inside the registry-supplied validator closure
(design §6.3), each raising a distinct ``contract.*`` code:

1. EXISTENCE — every ``suggested_operations[].operation_id`` exists in the
   invoking registry (``contract.unknown_suggested_operation``).
2. KIND COMPATIBILITY — the target's ``input_kind`` is a member of
   ``{NONE, emitting output_kind}`` (``contract.incompatible_suggested_operation``).
   Admitting NONE is deliberate: a suggestion may point at a root, which the
   agent calls independently rather than by chaining the Result into it. Without
   the kind test this runtime check would be weaker than the static AST scan and
   would pass the qc.assembly -> annotation.annotate shape.
3. SATISFIABILITY — ``parameter_changes`` validate against the target's
   parameter model (``contract.unsatisfiable_suggested_operation``). An EMPTY
   mapping must still validate, since an ordinary suggestion carries no
   overrides and any all-default target accepts it.

The validator resolves against the invoking registry at CALL time, so a target
registered after its suggester still resolves (pinned below). It fires only on
the RESULT output path.
"""

from __future__ import annotations

from typing import cast

import pytest

from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleContractError
from organelleverse.core.frozen import FrozenJson, FrozenMap
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OperationSuggestion, OrganelleResult
from organelleverse.operations import (
    CoreKind,
    ExecutionMode,
    OperationRegistry,
    OperationSpec,
    OperationStage,
)


def _consume_spec() -> OperationSpec:
    return OperationSpec(
        operation_id="demo.consume",
        contract_version="1.0",
        title="Demo consume operation",
        description="Test fixture spec used only to probe suggestion enforcement.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.CONSUME,
        input_kind=CoreKind.RESULT,
        output_kind=CoreKind.RESULT,
        callable_locator="demo.api:consume",
    )


def _target_spec() -> OperationSpec:
    return OperationSpec(
        operation_id="demo.target",
        contract_version="1.0",
        title="Demo target operation",
        description="Test fixture spec used only to probe suggestion enforcement.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.CONSUME,
        input_kind=CoreKind.RESULT,
        output_kind=CoreKind.RESULT,
        callable_locator="demo.api:target",
    )


def _result(*suggestions: OperationSuggestion) -> OrganelleResult:
    return OrganelleResult(
        operation_id="demo.consume",
        scope="none",
        status="ok",
        suggested_operations=suggestions,
    )


def _seed() -> OrganelleResult:
    return OrganelleResult(operation_id="demo.seed", scope="none", status="ok")


def test_unknown_suggestion_is_rejected() -> None:
    registry = OperationRegistry()

    def consume(result: OrganelleResult) -> OrganelleResult:
        return _result(OperationSuggestion(operation_id="io.nope", reason_code="demo"))

    registry.register(_consume_spec(), consume)
    with pytest.raises(OrganelleContractError) as excinfo:
        registry.invoke("demo.consume", input=_seed(), parameters={})
    assert excinfo.value.code == "contract.unknown_suggested_operation"


def test_kind_incompatible_suggestion_is_rejected() -> None:
    """A RESULT-producing operation cannot chain into a GENOME-consuming one.
    This is the qc.assembly -> annotation.annotate shape."""
    registry = OperationRegistry()

    analyze = OperationSpec(
        operation_id="demo.analyze",
        contract_version="1.0",
        title="Demo analyze operation",
        description="Test fixture spec used only to probe suggestion enforcement.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.ANALYZE,
        input_kind=CoreKind.GENOME,
        output_kind=CoreKind.RESULT,
        callable_locator="demo.api:analyze",
    )

    def analyze_fn(genome: OrganelleGenome) -> OrganelleResult:  # pragma: no cover - never invoked
        raise AssertionError

    def consume(result: OrganelleResult) -> OrganelleResult:
        return _result(OperationSuggestion(operation_id="demo.analyze", reason_code="demo"))

    registry.register(analyze, analyze_fn)
    registry.register(_consume_spec(), consume)
    with pytest.raises(OrganelleContractError) as excinfo:
        registry.invoke("demo.consume", input=_seed(), parameters={})
    assert excinfo.value.code == "contract.incompatible_suggested_operation"


def test_root_suggestion_is_accepted() -> None:
    """input_kind NONE is admitted: the agent calls a root independently.

    The read source carries a default so that an empty parameter_changes
    suggestion also satisfies the satisfiability rule, isolating this test on
    the kind-compatibility rule (NONE admission)."""
    registry = OperationRegistry()

    root = OperationSpec(
        operation_id="demo.root",
        contract_version="1.0",
        title="Demo root operation",
        description="Test fixture spec used only to probe suggestion enforcement.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.READ,
        input_kind=CoreKind.NONE,
        output_kind=CoreKind.DATA,
        output_modalities=("demo_records",),
        callable_locator="demo.api:root",
    )

    def root_fn(path: str = "demo") -> OrganelleData:  # pragma: no cover - never invoked
        raise AssertionError

    def consume(result: OrganelleResult) -> OrganelleResult:
        return _result(OperationSuggestion(operation_id="demo.root", reason_code="demo"))

    registry.register(root, root_fn)
    registry.register(_consume_spec(), consume)
    returned = registry.invoke("demo.consume", input=_seed(), parameters={})
    assert isinstance(returned, OrganelleResult)
    assert returned.status == "ok"


def test_unsatisfiable_parameter_changes_are_rejected() -> None:
    """parameter_changes must validate against the target's parameter model:
    a wrong-typed value is rejected by strict mode."""
    registry = OperationRegistry()

    def target_fn(
        result: OrganelleResult, *, limit: int = 5
    ) -> OrganelleResult:  # pragma: no cover
        raise AssertionError

    def consume(result: OrganelleResult) -> OrganelleResult:
        return _result(
            OperationSuggestion(
                operation_id="demo.target",
                reason_code="demo",
                parameter_changes=cast("FrozenMap[FrozenJson]", {"limit": "not-an-int"}),
            )
        )

    registry.register(_target_spec(), target_fn)
    registry.register(_consume_spec(), consume)
    with pytest.raises(OrganelleContractError) as excinfo:
        registry.invoke("demo.consume", input=_seed(), parameters={})
    assert excinfo.value.code == "contract.unsatisfiable_suggested_operation"


def test_unknown_parameter_name_in_changes_is_rejected() -> None:
    """A parameter the target does not declare is rejected by extra='forbid'."""
    registry = OperationRegistry()

    def target_fn(
        result: OrganelleResult, *, limit: int = 5
    ) -> OrganelleResult:  # pragma: no cover
        raise AssertionError

    def consume(result: OrganelleResult) -> OrganelleResult:
        return _result(
            OperationSuggestion(
                operation_id="demo.target",
                reason_code="demo",
                parameter_changes=cast("FrozenMap[FrozenJson]", {"bogus": 1}),
            )
        )

    registry.register(_target_spec(), target_fn)
    registry.register(_consume_spec(), consume)
    with pytest.raises(OrganelleContractError) as excinfo:
        registry.invoke("demo.consume", input=_seed(), parameters={})
    assert excinfo.value.code == "contract.unsatisfiable_suggested_operation"


def test_empty_parameter_changes_validate() -> None:
    """An empty parameter_changes mapping must still validate: an ordinary
    suggestion carries no overrides, and any all-default target accepts {}.
    Pinning this keeps rule 3 from regressing into rejecting every suggestion
    that omits parameter_changes."""
    registry = OperationRegistry()

    def target_fn(
        result: OrganelleResult, *, limit: int = 5
    ) -> OrganelleResult:  # pragma: no cover
        raise AssertionError

    def consume(result: OrganelleResult) -> OrganelleResult:
        return _result(OperationSuggestion(operation_id="demo.target", reason_code="demo"))

    registry.register(_target_spec(), target_fn)
    registry.register(_consume_spec(), consume)
    returned = registry.invoke("demo.consume", input=_seed(), parameters={})
    assert isinstance(returned, OrganelleResult)
    assert returned.status == "ok"


def test_target_registered_after_suggester_still_resolves() -> None:
    """The validator resolves against self._bindings at CALL time, so a target
    registered after its suggester is still accepted."""
    registry = OperationRegistry()

    def consume(result: OrganelleResult) -> OrganelleResult:
        return _result(OperationSuggestion(operation_id="demo.target", reason_code="demo"))

    registry.register(_consume_spec(), consume)  # suggester registered first

    def target_fn(
        result: OrganelleResult, *, limit: int = 5
    ) -> OrganelleResult:  # pragma: no cover
        raise AssertionError

    registry.register(_target_spec(), target_fn)  # target registered afterwards
    returned = registry.invoke("demo.consume", input=_seed(), parameters={})
    assert isinstance(returned, OrganelleResult)
    assert returned.status == "ok"
