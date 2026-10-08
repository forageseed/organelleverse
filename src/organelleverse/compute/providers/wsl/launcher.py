"""WSL process discovery and the fixed MCP launch argv (spec §11.1).

Everything here is injectable and shell-free. The launcher produces the exact
``wsl.exe --distribution <distro> --exec organelleverse-linux-provider mcp stdio``
argv as separate tokens — never a shell string (spec §14). Discovery reads
``wsl.exe --list --quiet`` only on the explicit discovery path, never on import.

Process execution is delegated to a :class:`ProcessRunner` so tests inject a
fake; no real ``wsl.exe`` is required (and none exists on Linux/CI).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import BinaryIO, Protocol

from organelleverse.core.errors import OrganelleContractError, OrganelleDependencyError
from organelleverse.operations.spec import StrictSpecModel

from .config import WslDiscoveredTarget, WslTargetConfig

__all__ = [
    "ProcessOutcome",
    "ProcessRunner",
    "WslProviderLauncher",
    "discover_wsl_targets",
]


class ProcessOutcome(StrictSpecModel):
    """Captured result of a fixed-argv child process (no shell)."""

    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


class ProcessRunner(Protocol):
    """Runs a fixed argv list with no shell. Injected by callers/tests."""

    def run(
        self,
        argv: tuple[str, ...],
        *,
        stdin: BinaryIO | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ProcessOutcome: ...


# The only argv discovery is ever allowed to run. Never an install, probe, or
# arbitrary command; only this exact list (spec §5.1/§14).
_DISCOVER_ARGV: tuple[str, ...] = ("wsl.exe", "--list", "--quiet")

# Option-shaped or otherwise non-distro strings that ``wsl.exe --list`` can emit
# and that must never become targets.
_REJECTED_NAME_PREFIXES = ("-",)


def _decode_list_output(stdout: bytes) -> list[str]:
    """Decode ``wsl.exe --list --quiet`` output to distro names.

    WSL emits UTF-16LE on Windows but may emit UTF-8 elsewhere; accept both.
    Strip only the documented presentation bytes (BOM, NUL padding, trailing
    newlines); preserve valid Unicode distro names (spec §11.1, §16.2).
    """
    text: str
    # UTF-16LE detection: many NUL bytes in the high byte of ASCII chars.
    if len(stdout) >= 2 and stdout.count(b"\x00") >= max(1, len(stdout) // 4):
        try:
            text = stdout.decode("utf-16-le").lstrip("\ufeff")
        except UnicodeDecodeError:
            text = stdout.decode("utf-8", errors="replace")
    else:
        text = stdout.decode("utf-8", errors="replace")
    names: list[str] = []
    for raw in text.splitlines():
        name = raw.strip().rstrip("\x00")
        if not name:
            continue
        if any(name.startswith(p) for p in _REJECTED_NAME_PREFIXES):
            continue
        # Reject any control characters inside the name.
        if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in name):
            continue
        names.append(name)
    return names


def discover_wsl_targets(runner: ProcessRunner) -> tuple[WslDiscoveredTarget, ...]:
    """Return WSL distros visible to ``wsl.exe --list --quiet``.

    This is the *only* path that runs ``wsl.exe`` for discovery. It changes no
    state, installs nothing, and probes no target. The subsequent protocol
    probe (Task 2) verifies WSL 2 and Linux identity.
    """
    outcome = runner.run(_DISCOVER_ARGV)
    if outcome.returncode != 0:
        raise OrganelleDependencyError(
            code="compute.wsl_list_failed",
            message="wsl.exe --list --quiet did not succeed; WSL may be absent or unconfigured",
            details={"returncode": outcome.returncode},
        )
    names = _decode_list_output(outcome.stdout)
    targets: list[WslDiscoveredTarget] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        try:
            targets.append(
                WslDiscoveredTarget(target_id=f"wsl:{name}", distribution=name)
            )
        except Exception as exc:  # a name that fails the closed pattern
            raise OrganelleContractError(
                code="compute.wsl_invalid_distro_name",
                message="wsl.exe --list returned a name that is not a valid WSL distro identifier",
                details={"name": name, "reason": str(exc)},
            ) from exc
    return tuple(targets)


class WslProviderLauncher:
    """Builds the fixed, shell-free argv that starts the Linux Provider MCP.

    The argv is exactly ``wsl.exe --distribution <distro> --exec
    organelleverse-linux-provider mcp stdio`` as separate tokens (spec §11.1).
    No method here returns a shell string, forms a command line, or expands
    anything; the caller passes this tuple to a process runner with
    ``shell=False``.
    """

    def mcp_argv(self, config: WslTargetConfig) -> tuple[str, ...]:
        return (
            "wsl.exe",
            "--distribution",
            config.distribution,
            "--exec",
            "organelleverse-linux-provider",
            "mcp",
            "stdio",
        )
