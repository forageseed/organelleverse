"""Core Bundle DataContract references resolve only at the trusted binding boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from organelleverse.capabilities.data_contracts import resolve_core_data_contracts
from organelleverse.capabilities.index import CapabilityEntry, CapabilityOrigin
from organelleverse.capabilities.models import CapabilityBundle
from organelleverse.core.errors import OrganelleContractError


def _bundle(*, modality: str, factory_locator: str) -> CapabilityBundle:
    return CapabilityBundle.model_validate(
        {
            "schema": "organelleverse.capability.v1",
            "capability": {
                "id": "assembly.assemble",
                "bundle_version": "1.0.0",
                "implementation": "native",
            },
            "contract": {
                "operation_id": "assembly.assemble",
                "contract_version": "1.0",
                "title": "Assemble",
                "description": "Test DataContract reference resolution.",
                "keywords": ["assembly", "contract", "data"],
                "execution_mode": "inline",
                "stage": "analyze",
                "input_kind": "data",
                "input_modalities": [modality],
                "output_kind": "result",
                "callable_locator": "tests.capabilities.fixtures_core_capability:run",
            },
            "data_contract": [{"modality": modality, "factory_locator": factory_locator}],
        }
    )


def _entry(*, origin: str, modality: str, factory_locator: str) -> CapabilityEntry:
    bundle = _bundle(modality=modality, factory_locator=factory_locator)
    origins = (CapabilityOrigin(channel=origin, source_path="test"),)
    if origin != "core":
        # The resolver's own boundary must reject this before CapabilityEntry's
        # independent non-core execution-identity invariant becomes relevant.
        return cast(
            CapabilityEntry,
            SimpleNamespace(
                capability_id="assembly.assemble",
                bundle=bundle,
                origins=origins,
            ),
        )
    return CapabilityEntry(
        capability_id="assembly.assemble",
        content_hash="sha256:" + "0" * 64,
        bundle_root=Path("/tmp/core-data-contract-bundle"),
        bundle=bundle,
        origins=origins,
    )


def test_core_bundle_resolves_its_declared_data_contract() -> None:
    contracts = resolve_core_data_contracts(
        _entry(
            origin="core",
            modality="sequencing_reads",
            factory_locator=(
                "organelleverse.assembly.data_contract:"
                "released_assembly_sequencing_reads_data_contract"
            ),
        )
    )

    assert [contract.modality for contract in contracts] == ["sequencing_reads"]


def test_non_core_bundle_cannot_import_a_python_data_contract_factory() -> None:
    with pytest.raises(OrganelleContractError) as captured:
        resolve_core_data_contracts(
            _entry(
                origin="local",
                modality="sequencing_reads",
                factory_locator=(
                    "organelleverse.assembly.data_contract:"
                    "released_assembly_sequencing_reads_data_contract"
                ),
            )
        )

    assert captured.value.code == "capability.data_contract_provider_required"


def test_factory_result_must_match_the_declared_modality() -> None:
    with pytest.raises(OrganelleContractError) as captured:
        resolve_core_data_contracts(
            _entry(
                origin="core",
                modality="sequencing_reads",
                factory_locator="organelleverse.assembly.data_contract:pmat_graph_input_data_contract",
            )
        )

    assert captured.value.code == "capability.data_contract_mismatch"
