"""I1b/I1c: a declared output modality is bound to returned data.

An operation whose ``output_kind`` is DATA must return an :class:`OrganelleData`
whose modality is one it declares (I1b), and when a :class:`DataContract` is
resolvable for that modality the returned payload is additionally validated
(I1c). The output contract is resolved independently of the input contract,
which is the subject of
:func:`test_output_contract_resolves_independently_of_the_input_contract`.
"""

from __future__ import annotations

from typing import cast

import pytest

from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleContractError
from organelleverse.core.frozen import FrozenJson, FrozenMap
from organelleverse.operations import (
    CoreKind,
    ExecutionMode,
    OperationRegistry,
    OperationSpec,
    OperationStage,
)


def _read_spec(operation_id: str, modality: str) -> OperationSpec:
    return OperationSpec(
        operation_id=operation_id,
        contract_version="1.0",
        title="Demo read operation",
        description="Test fixture spec used only to probe output-binding rules.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.READ,
        input_kind=CoreKind.NONE,
        output_kind=CoreKind.DATA,
        output_modalities=(modality,),
        callable_locator="demo.api:op",
    )


def test_returned_modality_must_be_declared() -> None:
    registry = OperationRegistry()

    def op(path: str) -> OrganelleData:
        return OrganelleData(modality="banana")

    registry.register(_read_spec("demo.mismatch", "organelle_records"), op)
    with pytest.raises(OrganelleContractError) as excinfo:
        registry.invoke("demo.mismatch", input=None, parameters={"path": "x"})
    assert excinfo.value.code == "contract.operation_output_modality_mismatch"


def test_output_payload_is_validated_when_a_contract_exists() -> None:
    registry = OperationRegistry()

    def op(path: str) -> OrganelleData:
        return OrganelleData(
            modality="organelle_records",
            payload=cast("FrozenMap[FrozenJson]", {"records": "not-a-sequence"}),
        )

    registry.register(_read_spec("demo.badpayload", "organelle_records"), op)
    with pytest.raises(OrganelleContractError) as excinfo:
        registry.invoke("demo.badpayload", input=None, parameters={"path": "x"})
    assert "organelle_records" in str(excinfo.value)


def test_output_contract_resolves_independently_of_the_input_contract() -> None:
    """Regression: wiring I1c to BoundOperation.data_contracts (input contracts)
    makes this check silently never fire for a transform whose input and output
    modalities differ."""
    registry = OperationRegistry()
    spec = OperationSpec(
        operation_id="demo.crossmodality",
        contract_version="1.0",
        title="Demo cross-modality transform",
        description="Test fixture spec used only to probe output-binding rules.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.TRANSFORM,
        input_kind=CoreKind.DATA,
        output_kind=CoreKind.DATA,
        input_modalities=("organelle_records",),
        output_modalities=("nuclear_assemblies",),
        callable_locator="demo.api:op",
    )

    def op(data: OrganelleData) -> OrganelleData:
        return OrganelleData(
            modality="nuclear_assemblies",
            payload=cast("FrozenMap[FrozenJson]", {"records": [{}]}),
        )

    registry.register(spec, op)
    source = OrganelleData(
        modality="organelle_records",
        payload=cast("FrozenMap[FrozenJson]", {"records": [{"accession": "NC_000932.1"}]}),
    )
    with pytest.raises(OrganelleContractError) as excinfo:
        registry.invoke("demo.crossmodality", input=source, parameters={})
    assert "nuclear_assemblies" in str(excinfo.value)


def test_conforming_output_passes_unchanged() -> None:
    registry = OperationRegistry()

    def op(path: str) -> OrganelleData:
        return OrganelleData(
            modality="organelle_records",
            payload=cast("FrozenMap[FrozenJson]", {"records": [{"accession": "NC_000932.1"}]}),
        )

    registry.register(_read_spec("demo.ok", "organelle_records"), op)
    result = registry.invoke("demo.ok", input=None, parameters={"path": "x"})
    assert isinstance(result, OrganelleData)
    assert result.modality == "organelle_records"
