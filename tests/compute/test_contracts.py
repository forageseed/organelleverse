from __future__ import annotations

import pytest
from pydantic import ValidationError

from organelleverse.compute.contracts import (
    ComputeTargetHint,
    ResolvedComputeTarget,
    TransportKind,
)


def test_transport_kind_values_are_the_stable_public_contract():
    # Spec §7.1: these exact strings are part of the public contract.
    assert {k.value for k in TransportKind} == {"wsl_mcp", "ssh_mcp", "slurm", "custom"}
    assert TransportKind("wsl_mcp") is TransportKind.WSL_MCP


def test_hint_defaults_to_auto_with_no_target():
    hint = ComputeTargetHint()
    assert hint.policy == "auto"
    assert hint.target_id is None


def test_resolved_compute_target_is_frozen_and_closed():
    target = ResolvedComputeTarget(
        target_id="wsl",
        provider_id="wsl",
        transport_kind=TransportKind.WSL_MCP,
        platform="linux-64",
        architecture="x86_64",
        artifact_transport="wsl-cache",
    )
    with pytest.raises((ValidationError, TypeError)):
        target.target_id = "mutated"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ResolvedComputeTarget(
            target_id="wsl",
            provider_id="wsl",
            transport_kind=TransportKind.WSL_MCP,
            platform="linux-64",
            architecture="x86_64",
            artifact_transport="wsl-cache",
            unexpected="field",
        )


def test_resolved_compute_target_round_trips_through_json():
    target = ResolvedComputeTarget(
        target_id="wsl",
        provider_id="wsl",
        transport_kind=TransportKind.WSL_MCP,
        platform="linux-64",
        architecture="x86_64",
        artifact_transport="wsl-cache",
    )
    rebuilt = ResolvedComputeTarget.model_validate_json(target.model_dump_json(by_alias=True))
    assert rebuilt == target
