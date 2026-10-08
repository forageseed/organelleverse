"""The first-party WSL compute provider.

``provider_factory`` is the zero-argument entry point registered in the
``organelleverse.compute_providers`` group. It is loaded exactly once, only
after explicit owner trust (spec §6), and returns the closed
:class:`~organelleverse.compute.contracts.ComputeProviderSpec` declaration.

This module imports only declaration/config code. The MCP client and worker
machinery are imported lazily inside the execution path (Task 2), so a bare
``import organelleverse`` never imports ``mcp`` or the provider's runtime code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from organelleverse.compute.contracts import ComputeProviderSpec, TransportKind

if TYPE_CHECKING:
    from .provider import WslComputeProvider

__all__ = ["PROVIDER_ID", "WSL_DECLARATION", "WslComputeProvider", "provider_factory"]


PROVIDER_ID = "wsl"


#: The closed declaration returned by :func:`provider_factory`. Pinned by spec
#: §7.1/§11; changing any field invalidates trust (compute.provider_trust_stale).
WSL_DECLARATION = ComputeProviderSpec(
    provider_id=PROVIDER_ID,
    display_name="Windows Subsystem for Linux",
    declaration_version="1.0",
    transport_kind=TransportKind.WSL_MCP,
    supported_platforms=("windows",),
    supports_prepare=True,
    supports_cancel=True,
    artifact_transports=("wsl_content_cache",),
)


def provider_factory() -> ComputeProviderSpec:
    """Return the closed WSL provider declaration.

    Loaded only after explicit trust. Returns the declaration object; the
    concrete provider (MCP client, worker manager) is constructed separately
    in Task 2 from this declaration.
    """
    return WSL_DECLARATION


def __getattr__(name: str) -> object:
    """Load the MCP client only when execution explicitly requests it."""

    if name == "WslComputeProvider":
        from .provider import WslComputeProvider

        return WslComputeProvider
    raise AttributeError(name)
