"""Legacy admitted plugin profiles for the v2 optimization contract."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from organelleverse.capabilities.index import (
    CapabilityEntry,
    CapabilityOrigin,
    CapabilityStatus,
)
from organelleverse.capabilities.models import CapabilityBundle, PluginCapabilityBundle
from organelleverse.capabilities.plugin_descriptor import PluginDescriptor, describe_plugin
from organelleverse.core.errors import OrganelleContractError
from organelleverse.optimization import contract_from_capability_entry
from organelleverse.optimization import profiles as optimization_profiles
from tests.capabilities.test_plugin_v2_models import v2_payload
from tests.plugin_experiments.conftest import admit, build_plugin_entry

_BUNDLE_HASH = "sha256:" + "c" * 64


def _native_bundle() -> CapabilityBundle:
    return CapabilityBundle.model_validate(
        {
            "schema": "organelleverse.capability.v1",
            "capability": {
                "id": "demo.native",
                "bundle_version": "1.0.0",
                "implementation": "native",
            },
            "contract": {
                "operation_id": "demo.native",
                "contract_version": "1.0",
                "title": "Native fixture",
                "description": "A non-plugin capability used to test profile rejection.",
                "keywords": ["fixture", "native", "optimization"],
                "execution_mode": "inline",
                "stage": "analyze",
                "input_kind": "none",
                "output_kind": "result",
                "callable_locator": "demo.native:run",
                "deterministic": True,
                "idempotent": True,
                "cacheable": True,
                "binding": {
                    "argument_mode": "named_parameters",
                    "result_codec": "canonical",
                },
            },
        }
    )


def test_admitted_plugin_entry_projects_exact_identity_and_limits(tmp_path: Path) -> None:
    entry = admit(
        build_plugin_entry(
            tmp_path / "bundle",
            max_trials=7,
            parallelism=3,
        )
    )

    contract = contract_from_capability_entry(entry, seed=41)

    assert contract.schema_version == "organelleverse.optimization.contract.v2"
    assert contract.target_capability_id == entry.capability_id
    assert contract.target_bundle_version == entry.bundle.capability.bundle_version
    assert contract.target_contract_version == entry.bundle.contract.contract_version
    assert contract.bundle_content_hash == entry.content_hash
    assert entry.execution_identity is not None
    assert contract.execution_identity == entry.execution_identity.digest
    assert [domain.name for domain in contract.parameters] == ["threshold"]
    assert contract.parameters[0].model_dump(mode="json") == {
        "name": "threshold",
        "kind": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "step": None,
        "logarithmic": False,
        "values": [],
    }
    assert contract.objective.name == "plugin_optimization_score"
    assert contract.objective.direction == "maximize"
    assert contract.strategies == ("explicit", "random", "agent")
    assert contract.seed == 41
    assert contract.budget.max_trials == 7
    assert contract.budget.parallelism == 3

    assert contract.digest.startswith("sha256:")
    assert contract_from_capability_entry(entry, seed=41).digest == contract.digest


def _entry_with_optimization_schemas(
    tmp_path: Path,
    schemas: dict[str, dict[str, object]],
) -> CapabilityEntry:
    payload = v2_payload()
    contract = cast(dict[str, object], payload["contract"])
    optimization = cast(dict[str, object], contract["optimization"])
    optimization["parameters"] = list(schemas)
    binding = cast(dict[str, object], contract["binding"])
    parameters = cast(list[dict[str, object]], binding["parameters"])
    parameters[:] = [item for item in parameters if item["name"] != "confidence_threshold"]
    parameters.extend(
        {"name": name, "codec": "json", "json_schema": schema} for name, schema in schemas.items()
    )
    bundle = PluginCapabilityBundle.model_validate(payload)
    return CapabilityEntry(
        capability_id=bundle.capability.id,
        content_hash=_BUNDLE_HASH,
        bundle_root=tmp_path,
        bundle=bundle,
        origins=(CapabilityOrigin(channel="core", source_path="fixture"),),
        status=CapabilityStatus.ADMITTED,
    )


def test_profile_converts_integer_enum_and_boolean_schemas(tmp_path: Path) -> None:
    entry = _entry_with_optimization_schemas(
        tmp_path,
        {
            "iterations": {
                "type": "integer",
                "minimum": 2,
                "maximum": 10,
                "multipleOf": 2,
            },
            "backend": {"type": "string", "enum": ["fast", "exact"]},
            "mask": {"type": "boolean"},
        },
    )

    contract = contract_from_capability_entry(entry)

    assert [domain.model_dump(mode="json") for domain in contract.parameters] == [
        {
            "name": "iterations",
            "kind": "integer",
            "minimum": 2,
            "maximum": 10,
            "step": 2,
            "logarithmic": False,
            "values": [],
        },
        {
            "name": "backend",
            "kind": "categorical",
            "minimum": None,
            "maximum": None,
            "step": None,
            "logarithmic": False,
            "values": ["fast", "exact"],
        },
        {
            "name": "mask",
            "kind": "boolean",
            "minimum": None,
            "maximum": None,
            "step": None,
            "logarithmic": False,
            "values": [],
        },
    ]
    assert "grid" in contract.strategies


def test_profile_normalizes_integral_float_integer_keywords(tmp_path: Path) -> None:
    entry = _entry_with_optimization_schemas(
        tmp_path,
        {
            "iterations": {
                "type": "integer",
                "minimum": 1.0,
                "maximum": 5.0,
                "multipleOf": 1.0,
            }
        },
    )

    domain = contract_from_capability_entry(entry).parameters[0]

    assert domain.minimum == 1
    assert type(domain.minimum) is int
    assert domain.maximum == 5
    assert type(domain.maximum) is int
    assert domain.step == 1
    assert type(domain.step) is int


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "number", "minimum": 0.0},
        {"type": "integer", "minimum": 1, "maximum": 4},
        {"type": "integer", "minimum": 1.5, "maximum": 4, "multipleOf": 1},
        {"type": "string"},
    ],
)
def test_unbounded_plugin_parameter_is_rejected_with_stable_details(
    tmp_path: Path,
    schema: dict[str, object],
) -> None:
    entry = _entry_with_optimization_schemas(tmp_path, {"open_parameter": schema})

    with pytest.raises(OrganelleContractError) as raised:
        contract_from_capability_entry(entry)

    assert raised.value.code == "optimization.domain_not_closed"
    assert raised.value.as_dict()["details"] == {
        "capability_id": entry.capability_id,
        "parameter": "open_parameter",
    }


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "enum": ["fast", 1]},
        {"type": "number", "minimum": 0.0, "maximum": 1.0, "multipleOf": 0.25},
        {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0},
        {"type": "number", "minimum": 0.0, "exclusiveMaximum": 1.0},
        {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "multipleOf": 2,
        },
    ],
)
def test_profile_refuses_to_widen_source_json_schema(
    tmp_path: Path,
    schema: dict[str, object],
) -> None:
    entry = _entry_with_optimization_schemas(tmp_path, {"constrained": schema})

    with pytest.raises(OrganelleContractError) as raised:
        contract_from_capability_entry(entry)

    assert raised.value.code == "optimization.domain_not_closed"
    assert raised.value.as_dict()["details"]["parameter"] == "constrained"


def test_profile_rejects_non_admitted_entry(tmp_path: Path) -> None:
    entry = _entry_with_optimization_schemas(
        tmp_path,
        {"threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
    ).model_copy(update={"status": CapabilityStatus.DISCOVERED})

    with pytest.raises(OrganelleContractError) as raised:
        contract_from_capability_entry(entry)

    assert raised.value.code == "capability.not_admitted"


def test_profile_rejects_non_plugin_entry(tmp_path: Path) -> None:
    bundle = _native_bundle()
    entry = CapabilityEntry(
        capability_id=bundle.capability.id,
        content_hash=_BUNDLE_HASH,
        bundle_root=tmp_path,
        bundle=bundle,
        origins=(CapabilityOrigin(channel="core", source_path="fixture"),),
        status=CapabilityStatus.ADMITTED,
    )

    with pytest.raises(OrganelleContractError) as raised:
        contract_from_capability_entry(entry)

    assert raised.value.code == "capability.plugin_required"


def test_profile_rejects_plugin_without_optimization(tmp_path: Path) -> None:
    entry = _entry_with_optimization_schemas(
        tmp_path,
        {"threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
    )
    bundle = cast(PluginCapabilityBundle, entry.bundle)
    contract = bundle.contract.model_copy(update={"optimization": None})
    entry = entry.model_copy(update={"bundle": bundle.model_copy(update={"contract": contract})})

    with pytest.raises(OrganelleContractError) as raised:
        contract_from_capability_entry(entry)

    assert raised.value.code == "capability.optimization_not_declared"


def test_profile_rejects_missing_descriptor_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry_with_optimization_schemas(
        tmp_path,
        {"threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
    )
    descriptor: PluginDescriptor = describe_plugin(cast(PluginCapabilityBundle, entry.bundle))

    def descriptor_without_parameters(_bundle: PluginCapabilityBundle) -> PluginDescriptor:
        return descriptor.model_copy(update={"parameters": ()})

    monkeypatch.setattr(
        optimization_profiles,
        "describe_plugin",
        descriptor_without_parameters,
    )

    with pytest.raises(OrganelleContractError) as raised:
        contract_from_capability_entry(entry)

    assert raised.value.code == "optimization.domain_not_closed"
    assert raised.value.as_dict()["details"]["parameter"] == "threshold"
