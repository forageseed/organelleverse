"""SSH fixed-argv launchers (spec §12, §14).

Every SSH invocation is a fixed token list passed with ``shell=False`` — never
a shell string, never a model-supplied command. Host-key verification is
mandatory and non-relaxable (``StrictHostKeyChecking=yes`` + a pinned
``UserKnownHostsFile``); ``BatchMode=yes`` forbids interactive auth; all
forwardings, local commands, and TTY are disabled.

Three fixed argv shapes:
* ``mcp_argv`` — runs ``organelleverse-linux-provider mcp stdio`` over one SSH
  connection (the MCP control plane).
* ``sftp_argv`` — the SFTP batch data plane (spec §10; bulk bytes never in MCP).
* ``artifact_verify_argv`` — the fixed worker verifier over SSH.

Process execution is delegated to an injectable :class:`SshProcessRunner` so
tests capture the exact argv/env/shell without a real SSH server.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import BinaryIO, Protocol

from organelleverse.core.errors import OrganelleContractError
from organelleverse.operations.spec import StrictSpecModel

from .config import SshTargetConfig

__all__ = [
    "SshProcessOutcome",
    "SshProcessRunner",
    "SshProviderLauncher",
]

_HEX64 = r"^[0-9a-f]{64}$"


class SshProcessOutcome(StrictSpecModel):
    """Captured result of a fixed-argv SSH/SFTP child process (no shell)."""

    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


class SshProcessRunner(Protocol):
    """Runs a fixed argv list with no shell. Injected by callers/tests."""

    def run(
        self,
        argv: tuple[str, ...],
        *,
        stdin: BinaryIO | None = None,
        env: Mapping[str, str] | None = None,
    ) -> SshProcessOutcome: ...


def _ssh_option_prefix(config: SshTargetConfig) -> tuple[str, ...]:
    """The mandatory, non-relaxable SSH option/account/host prefix.

    These options are the security boundary (spec §12, §14): mandatory
    host-key verification, no interactive auth, no forwardings/TTY/local
    commands. Nothing in this prefix is model-supplied.
    """
    return (
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={config.known_hosts_file}",
        "-o", "ClearAllForwardings=yes",
        "-o", "PermitLocalCommand=no",
        "-o", "RequestTTY=no",
        "-l", config.account,
        config.host_alias,
    )


class SshProviderLauncher:
    """Builds the fixed, shell-free argv for MCP, SFTP, and artifact verify.

    No method returns a shell string, expands anything, or accepts a
    model-supplied executable/host/account/command. Callers pass these tuples
    to a process runner with ``shell=False``.
    """

    def mcp_argv(self, config: SshTargetConfig) -> tuple[str, ...]:
        return (
            "ssh",
            "-T",
            *_ssh_option_prefix(config),
            "organelleverse-linux-provider",
            "mcp",
            "stdio",
        )

    def sftp_argv(self, config: SshTargetConfig) -> tuple[str, ...]:
        # sftp uses account@host form (no separate -l); same host-key options.
        return (
            "sftp",
            "-b", "-",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={config.known_hosts_file}",
            "-o", "ClearAllForwardings=yes",
            "-o", "PermitLocalCommand=no",
            f"{config.account}@{config.host_alias}",
        )

    def artifact_verify_argv(
        self, config: SshTargetConfig, *, sha256: str, size_bytes: int
    ) -> tuple[str, ...]:
        # Validate digest/size before argv construction; neither a path nor a
        # user/model string enters this command.
        import re

        if re.fullmatch(_HEX64, sha256) is None:
            raise OrganelleContractError(
                code="compute.ssh_invalid_digest",
                message="artifact verify requires a valid 64-hex sha256",
                details={"sha256": sha256},
            )
        if size_bytes < 0:
            raise OrganelleContractError(
                code="compute.ssh_invalid_size",
                message="artifact verify requires a non-negative size",
                details={"size_bytes": size_bytes},
            )
        return (
            "ssh",
            "-T",
            *_ssh_option_prefix(config),
            "organelleverse-linux-provider",
            "artifact",
            "verify",
            "--sha256", sha256,
            "--size", str(size_bytes),
        )
