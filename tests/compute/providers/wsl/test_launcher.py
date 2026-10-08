from __future__ import annotations

from collections.abc import Mapping
from typing import BinaryIO

import pytest
from pydantic import ValidationError

from organelleverse.compute.contracts import ComputeProviderSpec, TransportKind
from organelleverse.compute.providers.wsl import PROVIDER_ID, provider_factory
from organelleverse.compute.providers.wsl.config import WslTargetConfig
from organelleverse.compute.providers.wsl.launcher import (
    ProcessOutcome,
    WslProviderLauncher,
    discover_wsl_targets,
)
from organelleverse.core.errors import OrganelleContractError, OrganelleDependencyError

# --- fake process runner ----------------------------------------------------


class _FakeRunner:
    def __init__(self, outcome: ProcessOutcome):
        self._outcome = outcome
        self.calls: list[tuple[str, ...]] = []

    def run(
        self,
        argv: tuple[str, ...],
        *,
        stdin: BinaryIO | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ProcessOutcome:
        self.calls.append(argv)
        return self._outcome


def _config(distro="Ubuntu-24.04"):
    return WslTargetConfig(target_id=f"wsl:{distro}", distribution=distro)


# === factory declaration (spec §7.1/§11) ====================================


def test_factory_declaration_is_closed_and_wsl_mcp():
    decl = provider_factory()
    assert isinstance(decl, ComputeProviderSpec)
    assert decl.provider_id == PROVIDER_ID == "wsl"
    assert decl.transport_kind is TransportKind.WSL_MCP
    assert decl.declaration_version == "1.0"
    assert decl.artifact_transports == ("wsl_content_cache",)
    assert decl.supports_prepare is True
    assert decl.supports_cancel is True


def test_factory_declaration_is_frozen():
    with pytest.raises((ValidationError, TypeError)):
        provider_factory().provider_id = "x"  # type: ignore[misc]


# === launcher argv (spec §11.1, §14 — no shell) =============================


def test_launcher_uses_exact_exec_argv_without_shell():
    argv = WslProviderLauncher().mcp_argv(_config("Ubuntu-24.04"))
    assert argv == (
        "wsl.exe",
        "--distribution",
        "Ubuntu-24.04",
        "--exec",
        "organelleverse-linux-provider",
        "mcp",
        "stdio",
    )
    # exactly seven tokens; the distro with a dot stays one token (no shell split)
    assert len(argv) == 7


def test_launcher_distro_with_spaces_stays_one_token():
    argv = WslProviderLauncher().mcp_argv(_config("My Distro"))
    assert argv[2] == "My Distro"
    assert len(argv) == 7


# === discovery (spec §5.1 — only explicit path runs wsl.exe) ================


def test_discovery_uses_exact_list_argv():
    runner = _FakeRunner(ProcessOutcome(returncode=0, stdout=b"Ubuntu-24.04\nDebian\n"))
    targets = discover_wsl_targets(runner)
    assert [t.distribution for t in targets] == ["Ubuntu-24.04", "Debian"]
    assert [t.target_id for t in targets] == ["wsl:Ubuntu-24.04", "wsl:Debian"]
    # the only argv ever run is the exact discover argv — never an install/probe
    assert runner.calls == [("wsl.exe", "--list", "--quiet")]


def test_discovery_decodes_utf16le_output():
    # WSL on Windows emits UTF-16LE; must still yield correct distro names.
    stdout = "Ubuntu-24.04\nDebian\n".encode("utf-16-le")
    runner = _FakeRunner(ProcessOutcome(returncode=0, stdout=stdout))
    targets = discover_wsl_targets(runner)
    assert [t.distribution for t in targets] == ["Ubuntu-24.04", "Debian"]


def test_discovery_rejects_option_shaped_names():
    # wsl.exe can print option-like header/footer lines; they must not become targets
    runner = _FakeRunner(
        ProcessOutcome(returncode=0, stdout=b"Ubuntu-24.04\n-something\nDebian\n")
    )
    targets = discover_wsl_targets(runner)
    assert [t.distribution for t in targets] == ["Ubuntu-24.04", "Debian"]


def test_discovery_deduplicates():
    runner = _FakeRunner(
        ProcessOutcome(returncode=0, stdout=b"Ubuntu-24.04\nUbuntu-24.04\n")
    )
    targets = discover_wsl_targets(runner)
    assert len(targets) == 1


def test_discovery_nonzero_returncode_raises_dependency_error():
    runner = _FakeRunner(ProcessOutcome(returncode=1, stderr=b"no wsl"))
    with pytest.raises(OrganelleDependencyError) as exc:
        discover_wsl_targets(runner)
    assert exc.value.code == "compute.wsl_list_failed"


def test_discovery_invalid_name_raises_contract_error():
    runner = _FakeRunner(ProcessOutcome(returncode=0, stdout=b"bad/name\n"))
    with pytest.raises(OrganelleContractError) as exc:
        discover_wsl_targets(runner)
    assert exc.value.code == "compute.wsl_invalid_distro_name"


# === config closure =========================================================


def test_target_id_must_derive_from_distribution():
    with pytest.raises(ValidationError):
        WslTargetConfig(target_id="wsl:Other", distribution="Ubuntu-24.04")


def test_distribution_pattern_rejects_blank_and_control():
    with pytest.raises(ValidationError):
        WslTargetConfig(target_id="wsl:", distribution="")
    with pytest.raises(ValidationError):
        WslTargetConfig(target_id="wsl:\x01bad", distribution="\x01bad")
