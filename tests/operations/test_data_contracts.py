from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel, ConfigDict

from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleContractError, OrganelleInputError
from organelleverse.core.frozen import FrozenJson, FrozenMap
from organelleverse.core.result import OrganelleResult
from organelleverse.operations import (
    CoreKind,
    ExecutionMode,
    OperationRegistry,
    OperationSpec,
    OperationStage,
)
from organelleverse.operations.data_contracts import DataContract, DataContractRegistry
from organelleverse.operations.schemas import make_computed_object_ids_optional


class _StrictMatrixPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: tuple[tuple[float, ...], ...]


def _strict_matrix_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"const": "data"},
            "modality": {"const": "matrix"},
            "payload": _StrictMatrixPayload.model_json_schema(mode="serialization"),
        },
        "required": ["kind", "modality", "payload"],
    }


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _contract(modality: str) -> DataContract:
    return DataContract(
        modality=modality,
        schema_version=f"{modality}.v1",
        validator=lambda data: data,
        schema_provider=_strict_matrix_schema,
    )


def _data_spec(*, input_modalities: tuple[str, ...]) -> OperationSpec:
    return OperationSpec(
        operation_id="analysis.inspect_matrix",
        contract_version="1.0",
        title="Inspect a demo data matrix",
        description="Test fixture spec used only to probe data contract resolution.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.ANALYZE,
        input_kind=CoreKind.DATA,
        output_kind=CoreKind.RESULT,
        input_modalities=input_modalities,
        callable_locator="organelleverse.analysis.api:inspect_matrix",
    )


def _analyze_data(data: OrganelleData, *, output_dir: Path) -> OrganelleResult:
    return OrganelleResult(
        operation_id="analysis.inspect_matrix",
        scope="none",
        status="ok",
    )


def test_data_contract_registry_rejects_duplicate_and_unknown_modalities() -> None:
    contract = _contract("matrix")
    with pytest.raises(OrganelleContractError) as duplicate:
        DataContractRegistry((contract, contract))
    assert duplicate.value.code == "contract.duplicate_data_modality"

    registry = OperationRegistry()
    with pytest.raises(OrganelleContractError) as unknown:
        registry.register(_data_spec(input_modalities=("unknown_matrix",)), _analyze_data)
    assert unknown.value.code == "contract.unknown_data_modality"


def test_operation_registration_rejects_duplicate_declared_modalities() -> None:
    registry = OperationRegistry(data_contracts=(_contract("matrix"),))

    with pytest.raises(OrganelleContractError) as duplicate:
        registry.register(_data_spec(input_modalities=("matrix", "matrix")), _analyze_data)

    assert duplicate.value.code == "contract.duplicate_operation_modality"
    assert registry.list() == ()


def test_operation_can_bind_a_private_data_contract_without_registry_mutation() -> None:
    registry = OperationRegistry()

    binding = registry.register(
        _data_spec(input_modalities=("matrix",)),
        _analyze_data,
        data_contracts=(_contract("matrix"),),
    )

    assert tuple(contract.modality for contract in binding.data_contracts) == ("matrix",)
    input_schema = _mapping(
        _mapping(registry.invocation_schema("analysis.inspect_matrix")["properties"])["input"]
    )
    assert _mapping(_mapping(input_schema["properties"])["modality"])["const"] == "matrix"

    isolated = OperationRegistry()
    with pytest.raises(OrganelleContractError) as unknown:
        isolated.register(_data_spec(input_modalities=("matrix",)), _analyze_data)
    assert unknown.value.code == "contract.unknown_data_modality"


def test_data_contract_registry_is_sorted_and_rejects_malformed_contracts() -> None:
    registry = DataContractRegistry((_contract("zeta"), _contract("alpha")))
    assert tuple(contract.modality for contract in registry.list()) == ("alpha", "zeta")
    assert registry.require("zeta").schema_version == "zeta.v1"

    with pytest.raises(OrganelleContractError):
        _contract("Not-valid")
    with pytest.raises(OrganelleContractError):
        DataContract(
            modality="matrix",
            schema_version="   ",
            validator=lambda data: data,
            schema_provider=_strict_matrix_schema,
        )


def test_schema_discovery_calls_schema_provider_but_never_validator() -> None:
    validated: list[str] = []

    def validator(data: OrganelleData) -> object:
        validated.append(data.modality)
        raise AssertionError("discovery ran a validator")

    contract = DataContract(
        modality="matrix",
        schema_version="matrix.v1",
        validator=validator,
        schema_provider=_strict_matrix_schema,
    )
    registry = OperationRegistry(data_contracts=(contract,))
    registry.register(_data_spec(input_modalities=("matrix",)), _analyze_data)

    schema = registry.invocation_schema("analysis.inspect_matrix")
    properties = _mapping(schema["properties"])
    assert _mapping(properties["operation_id"])["const"] == "analysis.inspect_matrix"
    assert validated == []


def test_contract_schema_results_are_defensive_copies() -> None:
    source = _strict_matrix_schema()
    contract = DataContract(
        modality="matrix",
        schema_version="matrix.v1",
        validator=lambda data: data,
        schema_provider=lambda: source,
    )

    first = contract.json_schema()
    first["title"] = "mutated"

    assert contract.json_schema() == source
    assert "title" not in source


def test_provider_schema_must_be_valid_draft_2020_12_without_mutating_source() -> None:
    source = _strict_matrix_schema()
    source["required"] = "modality"
    snapshot = deepcopy(source)
    contract = DataContract(
        modality="matrix",
        schema_version="matrix.v1",
        validator=lambda data: data,
        schema_provider=lambda: source,
    )
    registry = OperationRegistry(data_contracts=(contract,))
    registry.register(_data_spec(input_modalities=("matrix",)), _analyze_data)

    with pytest.raises(OrganelleContractError) as invalid:
        registry.invocation_schema("analysis.inspect_matrix")

    assert invalid.value.code == "contract.invalid_data_schema"
    assert source == snapshot


def test_provider_schema_modality_const_must_match_contract_without_mutation() -> None:
    source = _strict_matrix_schema()
    properties = _mapping(source["properties"])
    properties["modality"] = {"const": "alignment"}
    snapshot = deepcopy(source)
    contract = DataContract(
        modality="matrix",
        schema_version="matrix.v1",
        validator=lambda data: data,
        schema_provider=lambda: source,
    )
    registry = OperationRegistry(data_contracts=(contract,))
    registry.register(_data_spec(input_modalities=("matrix",)), _analyze_data)

    with pytest.raises(OrganelleContractError) as mismatch:
        registry.invocation_schema("analysis.inspect_matrix")

    assert mismatch.value.code == "contract.data_schema_modality_mismatch"
    assert source == snapshot


@pytest.mark.parametrize("pattern", ["[", "(", ")", "dangling\\"])
def test_provider_schema_rejects_lexically_malformed_patterns_without_mutation(
    pattern: str,
) -> None:
    source = _strict_matrix_schema()
    _mapping(source["properties"])["sample"] = {
        "type": "string",
        "pattern": pattern,
    }
    snapshot = deepcopy(source)
    contract = DataContract(
        modality="matrix",
        schema_version="matrix.v1",
        validator=lambda data: data,
        schema_provider=lambda: source,
    )

    with pytest.raises(OrganelleContractError) as invalid:
        contract.json_schema()

    assert invalid.value.code == "contract.invalid_data_schema"
    assert source == snapshot


def test_provider_schema_rejects_malformed_pattern_property_without_mutation() -> None:
    source = _strict_matrix_schema()
    source["patternProperties"] = {"[": {"type": "string"}}
    snapshot = deepcopy(source)
    contract = DataContract(
        modality="matrix",
        schema_version="matrix.v1",
        validator=lambda data: data,
        schema_provider=lambda: source,
    )

    with pytest.raises(OrganelleContractError) as invalid:
        contract.json_schema()

    assert invalid.value.code == "contract.invalid_data_schema"
    assert source == snapshot


@pytest.mark.parametrize("pattern", [r"(?<sample>[A-Z]+)", r"\p{Letter}+"])
def test_provider_schema_accepts_ecmascript_patterns_python_rejects(pattern: str) -> None:
    source = _strict_matrix_schema()
    _mapping(source["properties"])["sample"] = {
        "type": "string",
        "pattern": pattern,
    }
    snapshot = deepcopy(source)
    contract = DataContract(
        modality="matrix",
        schema_version="matrix.v1",
        validator=lambda data: data,
        schema_provider=lambda: source,
    )

    discovered = contract.json_schema()

    assert discovered == source
    assert source == snapshot


def test_declared_modality_mismatch_is_rejected_before_execution() -> None:
    calls: list[str] = []
    registry = OperationRegistry(data_contracts=(_contract("matrix"),))

    def inspect(data: OrganelleData, *, output_dir: Path) -> OrganelleResult:
        calls.append(data.modality)
        return OrganelleResult(
            operation_id="analysis.inspect_matrix",
            scope="none",
            status="ok",
        )

    checked = registry.register(_data_spec(input_modalities=("matrix",)), inspect).function

    with pytest.raises(OrganelleInputError) as raised:
        checked(OrganelleData(modality="alignment"), output_dir=Path("out"))

    assert raised.value.code == "input.unsupported_operation_modality"
    assert calls == []


def test_data_operation_without_modalities_keeps_generic_data_schema() -> None:
    registry = OperationRegistry()
    registry.register(_data_spec(input_modalities=()), _analyze_data)

    schema = registry.invocation_schema("analysis.inspect_matrix")
    input_schema = _mapping(_mapping(schema["properties"])["input"])

    modality = _mapping(_mapping(input_schema["properties"])["modality"])
    assert isinstance(modality, Mapping)
    assert "const" not in modality


def test_multiple_data_modalities_use_one_of() -> None:
    matrix = _contract("matrix")
    alignment_schema = _strict_matrix_schema()
    alignment_properties = _mapping(alignment_schema["properties"])
    alignment_properties["modality"] = {"const": "alignment"}
    alignment = DataContract(
        modality="alignment",
        schema_version="alignment.v1",
        validator=lambda data: data,
        schema_provider=lambda: alignment_schema,
    )
    registry = OperationRegistry(data_contracts=(matrix, alignment))
    registry.register(_data_spec(input_modalities=("matrix", "alignment")), _analyze_data)

    schema = registry.invocation_schema("analysis.inspect_matrix")
    input_schema = _mapping(_mapping(schema["properties"])["input"])
    one_of = input_schema["oneOf"]
    assert isinstance(one_of, list)
    assert len(cast(list[object], one_of)) == 2


def test_provider_computed_object_ids_become_optional_before_embedding() -> None:
    provider_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "modality": {"const": "matrix"},
            "object_id": {"type": "string"},
            "nested": {
                "type": "object",
                "properties": {"object_id": {"type": "string"}},
                "required": ["object_id"],
            },
        },
        "required": ["modality", "object_id"],
    }
    contract = DataContract(
        modality="matrix",
        schema_version="matrix.v1",
        validator=lambda data: data,
        schema_provider=lambda: provider_schema,
    )
    registry = OperationRegistry(data_contracts=(contract,))
    registry.register(_data_spec(input_modalities=("matrix",)), _analyze_data)

    schema = registry.invocation_schema("analysis.inspect_matrix")
    input_schema = _mapping(_mapping(schema["properties"])["input"])
    properties = _mapping(input_schema["properties"])

    assert "object_id" in properties
    assert input_schema["required"] == ["modality"]
    nested = _mapping(properties["nested"])
    assert nested["required"] == []
    assert provider_schema["required"] == ["modality", "object_id"]


def test_computed_object_id_rewrite_is_recursive_and_non_mutating() -> None:
    original: dict[str, object] = {
        "type": "object",
        "properties": {
            "object_id": {"type": "string"},
            "nested": {
                "type": "object",
                "properties": {"object_id": {"type": "string"}},
                "required": ["object_id"],
            },
        },
        "required": ["object_id", "nested"],
    }
    snapshot = json.loads(json.dumps(original))

    rewritten = make_computed_object_ids_optional(original)

    assert original == snapshot
    assert rewritten["required"] == ["nested"]
    nested = _mapping(_mapping(rewritten["properties"])["nested"])
    assert isinstance(nested, Mapping)
    assert nested["required"] == []


def test_nuclear_assemblies_data_contract_is_registered_and_validates() -> None:
    from organelleverse.core.data import OrganelleData
    from organelleverse.operations.data_contracts import BUILTIN_DATA_CONTRACTS

    contract = next(c for c in BUILTIN_DATA_CONTRACTS if c.modality == "nuclear_assemblies")
    contract.validate(
        OrganelleData(
            modality="nuclear_assemblies",
            payload=cast(
                "FrozenMap[FrozenJson]",
                {"records": [{"accession": "gir:Rosaceae/v1.2_Malus_domestica"}]},
            ),
        )
    )

    with pytest.raises(OrganelleInputError):
        contract.validate(
            OrganelleData(
                modality="nuclear_assemblies",
                payload=cast("FrozenMap[FrozenJson]", {"records": [{"accession": ""}]}),
            )
        )
