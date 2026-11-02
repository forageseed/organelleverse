"""The scientific resolver describes needs and delegates only through an injected hook."""

from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.assembly.environment_contracts import (
    BackendCapabilityContract, EnvironmentResolution, ResolvedBackendVersion,
)
from organelleverse.assembly.environment_resolver import EnvironmentResolver, require_local_resolution
from organelleverse.core.errors import OrganelleDependencyError
from organelleverse.operations.execution_hooks import ExecutionDelegated, execution_scope

from .test_environment_resolver import _request

CONTRACT = BackendCapabilityContract(
    schema_version="organelleverse.backend-capabilities.v1", backend_id="oatk", items=(),
)
VERSION = ResolvedBackendVersion(backend_id="oatk", version="1.0.0")
# An opaque caller-owned selection; the science package must not inspect it.
TARGET = {"caller_selection": "gpu1"}


class _Hooks:
    def __init__(self, target=TARGET):
        self.target = target
        self.requirements = []

    def select(self, requirement):
        self.requirements.append(requirement)
        return self.target


def _resolve(resolver: EnvironmentResolver, tmp_path: Path, source: str = "auto") -> EnvironmentResolution:
    return resolver.resolve(
        request=_request(tmp_path, environment_source=source),
        capability_contract=CONTRACT, resolved_version=VERSION, effective_parameters={},
    )


def _linux_resolver():
    return EnvironmentResolver(
        _host_os="posix", _find_in_path=lambda _name: [], _find_conda=lambda: [],
        _registry_lookup=lambda _b: None, _managed_spec=lambda _b: None,
    )


def test_windows_delegates_the_requirement_to_the_caller(tmp_path: Path) -> None:
    hooks = _Hooks()
    with execution_scope(hooks=hooks), pytest.raises(ExecutionDelegated) as raised:
        _resolve(EnvironmentResolver(_host_os="nt"), tmp_path)
    assert raised.value.selection is TARGET
    assert len(hooks.requirements) == 1
    assert hooks.requirements[0].reason == "unsupported_platform"
    assert hooks.requirements[0].backend_id == "oatk"


def test_windows_without_a_hook_reports_a_scientific_dependency(tmp_path: Path) -> None:
    with execution_scope(hooks=None), pytest.raises(OrganelleDependencyError) as raised:
        _resolve(EnvironmentResolver(_host_os="nt"), tmp_path)
    assert raised.value.code == "assembly.environment_unavailable"
    assert raised.value.details["needs"] == "linux_environment"
    assert raised.value.details["backend_id"] == "oatk"
    assert not raised.value.suggested_action


def test_a_declining_caller_keeps_the_windows_dependency_error(tmp_path: Path) -> None:
    hooks = _Hooks(target=None)
    with execution_scope(hooks=hooks), pytest.raises(OrganelleDependencyError):
        _resolve(EnvironmentResolver(_host_os="nt"), tmp_path, source="existing")
    assert len(hooks.requirements) == 1


def test_windows_local_only_never_calls_the_caller_hook(tmp_path: Path) -> None:
    hooks = _Hooks()
    with execution_scope(hooks=hooks, local_only=True), pytest.raises(OrganelleDependencyError):
        _resolve(EnvironmentResolver(_host_os="nt"), tmp_path, source="managed")
    assert hooks.requirements == []


def test_linux_without_a_caller_keeps_the_managed_plan(tmp_path: Path) -> None:
    with execution_scope(hooks=None):
        resolution = _resolve(_linux_resolver(), tmp_path)
    assert resolution.managed_plan is not None and resolution.compute_target is None


def test_linux_delegates_before_materializing_a_managed_plan(tmp_path: Path) -> None:
    hooks = _Hooks()
    with execution_scope(hooks=hooks), pytest.raises(ExecutionDelegated) as raised:
        _resolve(_linux_resolver(), tmp_path)
    assert raised.value.selection is TARGET
    assert hooks.requirements[0].reason == "environment_unavailable"


def test_constructing_the_resolver_does_not_need_posix() -> None:
    EnvironmentResolver(_host_os="nt")


def test_local_only_scope_prevents_nested_delegation(tmp_path: Path) -> None:
    hooks = _Hooks()
    with execution_scope(hooks=hooks, local_only=True):
        resolution = _resolve(_linux_resolver(), tmp_path)
    assert resolution.managed_plan is not None and resolution.compute_target is None
    assert hooks.requirements == []


def test_historical_external_selection_is_preserved_without_transport_models() -> None:
    resolution = EnvironmentResolution(compute_target=TARGET)
    with pytest.raises(ExecutionDelegated) as raised:
        require_local_resolution(resolution, backend_id="oatk")
    assert raised.value.selection == TARGET
