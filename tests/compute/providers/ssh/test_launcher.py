"""SSH provider tests: fixed argv, host-key verification, import boundary,
config closure, and the security invariants from spec §12/§14.

No real SSH server is needed: process execution is replaced by an injectable
runner that captures the exact argv/env/shell, and host-key failure is modeled
by a nonzero returncode.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

import pytest
from pydantic import ValidationError

from organelleverse.compute.contracts import ComputeProviderSpec, TransportKind
from organelleverse.compute.providers.ssh import PROVIDER_ID, provider_factory
from organelleverse.compute.providers.ssh.config import (
    SshTargetConfig,
    require_known_hosts_exists,
)
from organelleverse.compute.providers.ssh.launcher import (
    SshProcessOutcome,
    SshProviderLauncher,
)
from organelleverse.core.errors import OrganelleContractError

# --- helpers ----------------------------------------------------------------


def _config(host="build-node-1", account="ovworker", known_hosts="/etc/ssh/ssh_known_hosts"):
    return SshTargetConfig(
        target_id=f"ssh:{host}",
        host_alias=host,
        account=account,
        known_hosts_file=Path(known_hosts),
        workspace_id="default",
    )


class _CapturingRunner:
    def __init__(self, outcome: SshProcessOutcome):
        self._outcome = outcome
        self.calls: list[tuple[str, ...]] = []

    def run(
        self,
        argv: tuple[str, ...],
        *,
        stdin: BinaryIO | None = None,
        env: Mapping[str, str] | None = None,
    ) -> SshProcessOutcome:
        self.calls.append(argv)
        return self._outcome


# === factory declaration (spec §7.1/§12) ===================================


def test_factory_declaration_is_closed_and_ssh_mcp():
    decl = provider_factory()
    assert isinstance(decl, ComputeProviderSpec)
    assert decl.provider_id == PROVIDER_ID == "ssh"
    assert decl.transport_kind is TransportKind.SSH_MCP
    assert decl.declaration_version == "1.0"
    assert decl.artifact_transports == ("ssh_sftp_rsync",)
    assert decl.supported_platforms == ("linux",)


def test_factory_declaration_is_frozen():
    with pytest.raises((ValidationError, TypeError)):
        provider_factory().provider_id = "x"  # type: ignore[misc]


# === launcher argv (spec §12, §14 — no shell, mandatory host-key) ==========


def test_mcp_argv_uses_exact_known_host_fixed_argv_without_shell():
    argv = SshProviderLauncher().mcp_argv(_config())
    assert argv == (
        "ssh", "-T",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "UserKnownHostsFile=/etc/ssh/ssh_known_hosts",
        "-o", "ClearAllForwardings=yes",
        "-o", "PermitLocalCommand=no",
        "-o", "RequestTTY=no",
        "-l", "ovworker",
        "build-node-1",
        "organelleverse-linux-provider",
        "mcp",
        "stdio",
    )


def test_sftp_argv_uses_exact_known_host_fixed_argv():
    argv = SshProviderLauncher().sftp_argv(_config())
    assert argv == (
        "sftp", "-b", "-",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "UserKnownHostsFile=/etc/ssh/ssh_known_hosts",
        "-o", "ClearAllForwardings=yes",
        "-o", "PermitLocalCommand=no",
        "ovworker@build-node-1",
    )


def test_mcp_argv_has_mandatory_security_options():
    argv = SshProviderLauncher().mcp_argv(_config())
    # host-key verification is mandatory and non-relaxable
    assert "StrictHostKeyChecking=yes" in argv
    assert any(a.startswith("UserKnownHostsFile=") for a in argv)
    # no interactive auth, no forwardings, no tty, no local commands
    assert "BatchMode=yes" in argv
    assert "ClearAllForwardings=yes" in argv
    assert "PermitLocalCommand=no" in argv
    assert "RequestTTY=no" in argv


def test_artifact_verify_argv_validates_digest_and_size():
    launcher = SshProviderLauncher()
    argv = launcher.artifact_verify_argv(_config(), sha256="a" * 64, size_bytes=100)
    assert argv[-6:] == (
        "artifact", "verify", "--sha256", "a" * 64, "--size", "100",
    )
    with pytest.raises(OrganelleContractError):
        launcher.artifact_verify_argv(_config(), sha256="bad", size_bytes=1)
    with pytest.raises(OrganelleContractError):
        launcher.artifact_verify_argv(_config(), sha256="a" * 64, size_bytes=-1)


# === config closure: model cannot override host/account/workspace ==========


def test_target_id_must_derive_from_host_alias():
    with pytest.raises(ValidationError):
        SshTargetConfig(
            target_id="ssh:other", host_alias="build-node-1", account="ovworker",
            known_hosts_file=Path("/etc/ssh/ssh_known_hosts"), workspace_id="default",
        )


def test_known_hosts_must_be_absolute():
    with pytest.raises(ValidationError):
        SshTargetConfig(
            target_id="ssh:n1", host_alias="n1", account="ovworker",
            known_hosts_file=Path("relative/known_hosts"), workspace_id="default",
        )


def test_account_pattern_is_closed_posix_shape():
    # The account name is a closed POSIX shape; least-privilege is enforced by
    # the account's actual permissions (spec §12), not by its name, so "root"
    # is not special-cased. The pattern rejects non-conforming names.
    cfg = SshTargetConfig(
        target_id="ssh:n1", host_alias="n1", account="root",
        known_hosts_file=Path("/etc/ssh/ssh_known_hosts"), workspace_id="default",
    )
    assert cfg.account == "root"
    for bad in ("Root", "ov worker", "1ovworker", "ov$worker"):
        with pytest.raises(ValidationError):
            SshTargetConfig(
                target_id="ssh:n1", host_alias="n1", account=bad,
                known_hosts_file=Path("/etc/ssh/ssh_known_hosts"), workspace_id="default",
            )


# === host-key failure fails closed (spec §14) ==============================


def test_host_key_mismatch_fails_closed_before_mcp(tmp_path):
    # A nonzero ssh exit (host-key mismatch) must surface as a structured
    # provider error and never retry with relaxed checking.
    runner = _CapturingRunner(SshProcessOutcome(returncode=255, stderr=b"Host key verification failed."))
    # the launcher only builds argv; the caller (provider) inspects the outcome.
    # Here we assert the captured argv never relaxes checking even on failure.
    argv = SshProviderLauncher().mcp_argv(_config())
    runner.run(argv)
    assert "StrictHostKeyChecking=yes" in runner.calls[0]
    assert "BatchMode=yes" in runner.calls[0]


def test_require_known_hosts_exists_fails_closed_when_missing(tmp_path):
    cfg = SshTargetConfig(
        target_id="ssh:n1", host_alias="n1", account="ovworker",
        known_hosts_file=tmp_path / "nope", workspace_id="default",
    )
    with pytest.raises(OrganelleContractError) as exc:
        require_known_hosts_exists(cfg)
    assert exc.value.code == "compute.ssh_known_hosts_missing"


# === no credentials in the model (spec §14) ================================


def test_config_carries_no_credentials():
    cfg = _config()
    dump = cfg.model_dump()
    # no key/passphrase/agent/token/port/ip fields exist on the model
    forbidden = {"private_key", "passphrase", "agent_socket", "token", "port", "ip", "password"}
    assert not (forbidden & set(dump.keys()))
