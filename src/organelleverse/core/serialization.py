"""Exact-type JSON persistence for immutable v0.1 core contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import UnionType
from typing import TypeVar, Union, cast, get_args, get_origin

from pydantic import BaseModel, ValidationError

from .base import StrictFrozenModel
from .data import OrganelleData
from .errors import OrganelleContractError, OrganelleInputError
from .frozen import FrozenMap
from .genome import OrganelleGenome
from .result import OrganelleResult

PersistedContract = OrganelleGenome | OrganelleData | OrganelleResult
ContractT = TypeVar("ContractT", OrganelleGenome, OrganelleData, OrganelleResult)

_CONTRACT_KINDS: dict[type[PersistedContract], str] = {
    OrganelleGenome: "genome",
    OrganelleData: "data",
    OrganelleResult: "result",
}


def save_contract(contract: PersistedContract, path: str | Path) -> Path:
    """Save one exact v0.1 core contract as a JSON manifest."""
    contract_type = type(contract)
    if contract_type not in _CONTRACT_KINDS:
        raise OrganelleContractError(
            code="contract.unsupported_persistence_type",
            message="Only Genome, Data, and Result contracts can be persisted",
            details={"actual_type": contract_type.__name__},
        )

    destination = Path(path)
    try:
        encoded = json.dumps(
            _manifest_value(contract, contract.model_dump(mode="json")),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        destination.write_text(f"{encoded}\n", encoding="utf-8")
    except (OSError, TypeError, ValueError) as error:
        raise OrganelleInputError(
            code="input.contract_manifest_write_failed",
            message=f"Could not write contract manifest: {destination}",
            details={"path": str(destination)},
        ) from error
    return destination


def load_genome(path: str | Path) -> OrganelleGenome:
    """Load a genome manifest, rejecting every other contract kind."""
    return _load_typed_contract(path, OrganelleGenome, "genome")


def load_data(path: str | Path) -> OrganelleData:
    """Load a data manifest, rejecting every other contract kind."""
    return _load_typed_contract(path, OrganelleData, "data")


def load_result(path: str | Path) -> OrganelleResult:
    """Load a result manifest, rejecting every other contract kind."""
    return _load_typed_contract(path, OrganelleResult, "result")


def _load_typed_contract(
    path: str | Path, contract_type: type[ContractT], expected_kind: str
) -> ContractT:
    payload = _read_manifest(path)
    actual_kind = payload.get("kind")
    if actual_kind != expected_kind:
        raise OrganelleContractError(
            code="contract.persistence_kind_mismatch",
            message=f"Expected {expected_kind} manifest, got {actual_kind!r}",
            details={"expected_kind": expected_kind, "actual_kind": actual_kind},
        )
    _reject_computed_object_ids(payload, contract_type)
    try:
        return contract_type.model_validate(payload)
    except ValidationError as error:
        raise OrganelleContractError(
            code="contract.invalid_persistence_manifest",
            message=f"invalid {expected_kind} contract manifest schema or fields",
            details={
                "expected_kind": expected_kind,
                "validation_errors": _validation_error_summary(error),
            },
        ) from error


def _read_manifest(path: str | Path) -> dict[str, object]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OrganelleInputError(
            code="input.contract_manifest_read_failed",
            message=f"Could not read contract manifest: {source}",
            details={"path": str(source)},
        ) from error
    if not isinstance(payload, Mapping):
        raise OrganelleInputError(
            code="input.contract_manifest_not_object",
            message=f"Contract manifest must contain a JSON object: {source}",
            details={"path": str(source)},
        )
    return dict(cast(Mapping[str, object], payload))


def _manifest_value(value: object, serialized: object) -> object:
    """Keep user JSON intact while omitting computed fields from contract models."""
    if isinstance(value, BaseModel):
        assert isinstance(serialized, Mapping)
        serialized_fields = cast(Mapping[str, object], serialized)
        return {
            field_name: _manifest_value(getattr(value, field_name), serialized_fields[field_name])
            for field_name in type(value).model_fields
        }
    if isinstance(value, Mapping):
        assert isinstance(serialized, Mapping)
        items = cast(Mapping[object, object], value)
        serialized_items = cast(Mapping[str, object], serialized)
        return {
            str(key): _manifest_value(item, serialized_items[str(key)])
            for key, item in items.items()
        }
    if isinstance(value, tuple):
        assert isinstance(serialized, list)
        items = cast(tuple[object, ...], value)
        serialized_items = cast(list[object], serialized)
        return [
            _manifest_value(item, item_serialized)
            for item, item_serialized in zip(items, serialized_items, strict=True)
        ]
    return serialized


def _reject_computed_object_ids(
    payload: Mapping[str, object], contract_type: type[ContractT]
) -> None:
    errors: list[dict[str, object]] = []
    _collect_computed_object_id_errors(payload, contract_type, (), errors)
    if errors:
        raise OrganelleContractError(
            code="contract.invalid_persistence_manifest",
            message="contract manifest must not contain computed object_id fields",
            details={"validation_errors": errors},
        )


def _collect_computed_object_id_errors(
    value: object,
    model_type: type[BaseModel],
    location: tuple[str | int, ...],
    errors: list[dict[str, object]],
) -> None:
    if not isinstance(value, Mapping):
        return
    payload = cast(Mapping[str, object], value)
    if "object_id" in payload:
        errors.append(
            {
                "loc": [*location, "object_id"],
                "message": "Computed object_id fields are not allowed in contract manifests",
                "type": "extra_forbidden",
            }
        )
    for name, field in model_type.model_fields.items():
        if name in payload:
            _collect_nested_model_errors(payload[name], field.annotation, (*location, name), errors)


def _collect_nested_model_errors(
    value: object,
    annotation: object,
    location: tuple[str | int, ...],
    errors: list[dict[str, object]],
) -> None:
    if isinstance(annotation, type) and issubclass(annotation, StrictFrozenModel):
        _collect_computed_object_id_errors(
            value, cast(type[BaseModel], annotation), location, errors
        )
        return

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, UnionType):
        for argument in arguments:
            _collect_nested_model_errors(value, argument, location, errors)
        return
    if origin is FrozenMap and isinstance(value, Mapping) and arguments:
        items = cast(Mapping[object, object], value)
        for key, item in items.items():
            _collect_nested_model_errors(item, arguments[0], (*location, str(key)), errors)
        return
    if origin is tuple and isinstance(value, list) and arguments:
        item_annotation = arguments[0]
        items = cast(list[object], value)
        for index, item in enumerate(items):
            _collect_nested_model_errors(item, item_annotation, (*location, index), errors)


def _validation_error_summary(error: ValidationError) -> list[dict[str, object]]:
    return [
        {
            "loc": list(item["loc"]),
            "message": str(item["msg"]),
            "type": str(item["type"]),
        }
        for item in error.errors(include_context=False, include_input=False, include_url=False)
    ]
