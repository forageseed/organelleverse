"""Trusted resolution of core Bundle DataContract factories.

Discovery and admission keep capability metadata inert. This module is used
only when a verified core Bundle is bound or verified, where importing code
from the installed OrganelleVerse distribution is already the intended trust
boundary.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable

from organelleverse.core.errors import OrganelleContractError
from organelleverse.operations.data_contracts import DataContract

from .index import CapabilityEntry


def resolve_core_data_contracts(entry: CapabilityEntry) -> tuple[DataContract, ...]:
    """Resolve and validate the declared factories for one all-core Bundle.

    A Python locator is a code-execution surface. It is therefore forbidden
    for local, project, and package capabilities until those channels have a
    provider protocol that can execute it under their worker identity.
    """
    if not entry.bundle.data_contracts:
        return ()
    if not all(origin.channel == "core" for origin in entry.origins):
        raise OrganelleContractError(
            code="capability.data_contract_provider_required",
            message=(
                "a non-core capability cannot resolve Python data-contract factories; "
                f"{entry.capability_id} needs an execution provider"
            ),
            details={"capability_id": entry.capability_id},
        )
    return tuple(
        _resolve_reference(entry, reference.modality, reference.factory_locator)
        for reference in entry.bundle.data_contracts
    )


def _resolve_reference(
    entry: CapabilityEntry, modality: str, locator: str
) -> DataContract:
    module_name, attribute_name = locator.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise OrganelleContractError(
            code="capability.data_contract_unresolvable",
            message=f"cannot import data-contract factory for {entry.capability_id}: {locator}",
            details={"capability_id": entry.capability_id, "factory_locator": locator},
        ) from error
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise OrganelleContractError(
            code="capability.data_contract_unresolvable",
            message=f"data-contract factory is not callable for {entry.capability_id}: {locator}",
            details={"capability_id": entry.capability_id, "factory_locator": locator},
        )
    _require_zero_argument_factory(entry, locator, factory)
    try:
        contract = factory()
    except Exception as error:
        raise OrganelleContractError(
            code="capability.data_contract_unresolvable",
            message=f"data-contract factory failed for {entry.capability_id}: {locator}",
            details={"capability_id": entry.capability_id, "factory_locator": locator},
        ) from error
    if not isinstance(contract, DataContract):
        raise OrganelleContractError(
            code="capability.data_contract_unresolvable",
            message=f"data-contract factory must return DataContract: {locator}",
            details={"capability_id": entry.capability_id, "factory_locator": locator},
        )
    if contract.modality != modality:
        raise OrganelleContractError(
            code="capability.data_contract_mismatch",
            message=(
                f"data-contract factory for {entry.capability_id} returned {contract.modality!r}; "
                f"Bundle declares {modality!r}"
            ),
            details={
                "capability_id": entry.capability_id,
                "factory_locator": locator,
                "declared_modality": modality,
                "actual_modality": contract.modality,
            },
        )
    return contract


def _require_zero_argument_factory(
    entry: CapabilityEntry, locator: str, factory: Callable[..., object]
) -> None:
    try:
        inspect.signature(factory).bind()
    except (TypeError, ValueError) as error:
        raise OrganelleContractError(
            code="capability.data_contract_unresolvable",
            message=f"data-contract factory must accept no arguments: {locator}",
            details={"capability_id": entry.capability_id, "factory_locator": locator},
        ) from error


__all__ = ["resolve_core_data_contracts"]
