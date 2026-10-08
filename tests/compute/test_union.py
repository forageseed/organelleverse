from __future__ import annotations

import pytest
from pydantic import ValidationError

from organelleverse.assembly.environment_contracts import (
    EnvironmentResolution,
    ManagedProviderPlan,
)
from organelleverse.compute.contracts import ResolvedComputeTarget, TransportKind


def _compute_target() -> ResolvedComputeTarget:
    return ResolvedComputeTarget(
        target_id="wsl",
        provider_id="wsl",
        transport_kind=TransportKind.WSL_MCP,
        platform="linux-64",
        architecture="x86_64",
        artifact_transport="wsl-cache",
    )


def test_union_accepts_compute_target_only():
    res = EnvironmentResolution(compute_target=_compute_target())
    assert res.compute_target is not None
    assert res.selected_provider is None
    assert res.managed_plan is None


def test_union_rejects_zero_branches():
    with pytest.raises(ValidationError):
        EnvironmentResolution()


def test_union_rejects_two_branches():
    plan = ManagedProviderPlan(backend_id="x", carrier="conda", platform="linux-64")
    with pytest.raises(ValidationError):
        EnvironmentResolution(compute_target=_compute_target(), managed_plan=plan)
