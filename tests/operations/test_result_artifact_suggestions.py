"""Live advice names the operations whose declared input a Result carries.

``result_artifact_inputs`` on a target spec names the artifact kinds its GENOME or
DATA input is read from (assemble's ``primary_genome_manifest`` -> annotate). The
registry exposes candidates outside the Result's identity. An explicit suggestion
is still admitted only when the target declares the kind and the result carries it.
"""

from __future__ import annotations

import pytest

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleContractError
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OperationSuggestion, OrganelleResult
from organelleverse.operations import (
    CoreKind,
    ExecutionMode,
    OperationRegistry,
    OperationSpec,
    OperationStage,
)

_KIND = "genome_manifest"


def _artifact(kind: str) -> ArtifactRef:
    return ArtifactRef(kind=kind, uri="demo.json", format="json", sha256="0" * 64, size_bytes=0)


def _spec(
    operation_id: str,
    *,
    stage: OperationStage,
    input_kind: CoreKind,
    **extra: object,
) -> OperationSpec:
    return OperationSpec(
        operation_id=operation_id,
        contract_version="1.0",
        title="Demo operation",
        description="Test fixture spec used only to probe result artifact suggestions.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=stage,
        input_kind=input_kind,
        output_kind=CoreKind.RESULT,
        callable_locator="demo.api:op",
        **extra,  # type: ignore[arg-type]
    )


def _producer_spec() -> OperationSpec:
    return _spec("demo.produce", stage=OperationStage.ANALYZE, input_kind=CoreKind.NONE)


def _consumer_spec(**extra: object) -> OperationSpec:
    extra.setdefault("result_artifact_inputs", (_KIND,))
    extra.setdefault("organelle_types", ("plastid",))
    return _spec("demo.annotate", stage=OperationStage.ANALYZE, input_kind=CoreKind.GENOME, **extra)


def _consumer_fn(
    genome: OrganelleGenome, *, threads: int = 4
) -> OrganelleResult:  # pragma: no cover
    raise AssertionError


def _registry(
    *,
    scope: str = "plastid",
    kinds: tuple[str, ...] = (_KIND,),
    suggestions: tuple[OperationSuggestion, ...] = (),
    consumer: OperationSpec | None = None,
) -> OperationRegistry:
    registry = OperationRegistry()

    def produce() -> OrganelleResult:
        return OrganelleResult(
            operation_id="demo.produce",
            scope=scope,  # type: ignore[arg-type]
            status="ok",
            artifacts=tuple(_artifact(kind) for kind in kinds),
            suggested_operations=suggestions,
        )

    registry.register(_producer_spec(), produce)
    registry.register(consumer or _consumer_spec(), _consumer_fn)
    return registry


def _produced(registry: OperationRegistry) -> OrganelleResult:
    result = registry.invoke("demo.produce", input=None, parameters={})
    assert isinstance(result, OrganelleResult)
    return result


def test_declared_artifact_input_is_exposed_as_live_advice() -> None:
    registry = _registry()
    result = _produced(registry)
    assert result.suggested_operations == ()
    candidates = registry.agent_advice(result).next_operations
    assert len(candidates) == 1
    assert candidates[0].operation_id == "demo.annotate"
    assert candidates[0].reason_code == "result_artifact_feeds_input"
    assert candidates[0].input_artifact_kind == _KIND
    assert not candidates[0].ready


def test_nothing_is_derived_when_the_result_lacks_the_artifact() -> None:
    registry = _registry(kinds=("other_artifact",))
    assert registry.agent_advice(_produced(registry)).next_operations == ()


def test_nothing_is_derived_for_an_organelle_the_target_does_not_serve() -> None:
    registry = _registry(scope="mitochondrion")
    assert registry.agent_advice(_produced(registry)).next_operations == ()


def test_a_target_without_the_declaration_is_never_derived() -> None:
    registry = _registry(consumer=_consumer_spec(result_artifact_inputs=()))
    assert registry.agent_advice(_produced(registry)).next_operations == ()


def test_operation_supplied_suggestions_are_kept_not_derived() -> None:
    mine = OperationSuggestion(
        operation_id="demo.annotate",
        reason_code="own_reason",
        input_artifact_kind=_KIND,
    )
    assert _produced(_registry(suggestions=(mine,))).suggested_operations == (mine,)


def test_explicit_bridge_is_rejected_when_the_target_does_not_declare_the_kind() -> None:
    claim = OperationSuggestion(
        operation_id="demo.annotate", reason_code="own", input_artifact_kind=_KIND
    )
    registry = _registry(suggestions=(claim,), consumer=_consumer_spec(result_artifact_inputs=()))
    with pytest.raises(OrganelleContractError) as excinfo:
        _produced(registry)
    assert excinfo.value.code == "contract.incompatible_suggested_operation"


def test_explicit_bridge_is_rejected_when_the_result_does_not_carry_the_kind() -> None:
    claim = OperationSuggestion(
        operation_id="demo.annotate", reason_code="own", input_artifact_kind=_KIND
    )
    with pytest.raises(OrganelleContractError) as excinfo:
        _produced(_registry(suggestions=(claim,), kinds=("other_artifact",)))
    assert excinfo.value.code == "contract.incompatible_suggested_operation"


def test_artifact_inputs_require_a_genome_or_data_input() -> None:
    with pytest.raises(ValueError, match="GENOME or DATA"):
        _spec(
            "demo.bad",
            stage=OperationStage.CONSUME,
            input_kind=CoreKind.RESULT,
            result_artifact_inputs=(_KIND,),
        )


@pytest.mark.parametrize("kinds", [("b_kind", "a_kind"), ("a_kind", "a_kind"), ("Bad-Kind",)])
def test_artifact_inputs_must_be_sorted_unique_kinds(kinds: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        _consumer_spec(result_artifact_inputs=kinds)


def test_the_declaration_is_serialized_only_when_present() -> None:
    declared = _consumer_spec().model_dump(mode="json")
    undeclared = _consumer_spec(result_artifact_inputs=()).model_dump(mode="json")
    assert declared["result_artifact_inputs"] == [_KIND]
    assert "result_artifact_inputs" not in undeclared
    assert OperationSpec.model_validate(declared) == _consumer_spec()
