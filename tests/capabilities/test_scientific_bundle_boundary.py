"""Historical plugin manifests remain scientific inputs without application ownership."""
from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from organelleverse.capabilities.models import PluginCapabilityBundle
from organelleverse.optimization.evidence_models import RepeatEvidence
from organelleverse.plugin_protocol import PluginContext, PluginResult
from tests.capabilities.test_plugin_v2_models import v2_payload


def test_legacy_display_sections_do_not_enter_the_scientific_model() -> None:
    payload = v2_payload()
    before = deepcopy(payload)
    bundle = PluginCapabilityBundle.model_validate(payload)
    assert payload == before
    dumped = bundle.model_dump(mode="json", by_alias=True)
    assert not {"plugin", "agent", "gui"}.intersection(dumped)
    assert bundle.contract.outputs
    assert bundle.contract.optimization is not None
    assert not hasattr(bundle, "gui")


def test_scientific_v2_does_not_require_an_application() -> None:
    payload = v2_payload()
    for name in ("plugin", "agent", "gui"):
        payload.pop(name)
    bundle = PluginCapabilityBundle.model_validate(payload)
    assert bundle.capability.id == payload["capability"]["id"]
    assert PluginContext.__module__ == "organelleverse.plugin_protocol"
    assert PluginResult.__module__ == "organelleverse.plugin_protocol"
    assert RepeatEvidence.__module__ == "organelleverse.optimization.evidence_models"


def test_legacy_input_does_not_weaken_scientific_validation() -> None:
    payload = v2_payload()
    payload["contract"]["optimization"]["parameters"] = ["not_declared"]
    with pytest.raises(ValidationError, match="optimization"):
        PluginCapabilityBundle.model_validate(payload)
