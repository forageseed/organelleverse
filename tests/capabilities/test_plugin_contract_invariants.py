"""Regression tests for Plugin-01 v2 contract invariants."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from tests.capabilities.test_plugin_v2_models import v2_payload


def test_v2_binding_requires_plugin_protocol_argument_mode() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    contract = cast(dict[str, object], payload["contract"])
    binding = cast(dict[str, object], contract["binding"])
    binding["argument_mode"] = "named_parameters"

    with pytest.raises(ValidationError, match="plugin_protocol"):
        PluginCapabilityBundle.model_validate(payload)


def test_v2_optimization_parameter_names_are_unique() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    contract = cast(dict[str, object], payload["contract"])
    optimization = cast(dict[str, object], contract["optimization"])
    optimization["parameters"] = [
        "confidence_threshold",
        "confidence_threshold",
    ]

    with pytest.raises(ValidationError, match="unique"):
        PluginCapabilityBundle.model_validate(payload)


def test_descriptor_treats_non_output_path_bindings_as_file_inputs() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle
    from organelleverse.capabilities.plugin_descriptor import describe_plugin

    payload = v2_payload()
    contract = cast(dict[str, object], payload["contract"])
    binding = cast(dict[str, object], contract["binding"])
    parameters = cast(list[object], binding["parameters"])
    parameters.insert(
        2,
        {
            "name": "metadata_file",
            "codec": "path",
            "json_schema": {"type": "string"},
        },
    )

    descriptor = describe_plugin(PluginCapabilityBundle.model_validate(payload))

    assert [field.name for field in descriptor.inputs] == ["images", "labels", "metadata_file", "source_file"]
    assert [field.name for field in descriptor.parameters] == ["confidence_threshold"]


def test_v2_output_name_is_a_safe_artifact_identifier() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    contract = cast(dict[str, object], payload["contract"])
    outputs = cast(list[dict[str, object]], contract["outputs"])
    outputs[0]["name"] = "../outside"

    with pytest.raises(ValidationError, match="pattern"):
        PluginCapabilityBundle.model_validate(payload)
