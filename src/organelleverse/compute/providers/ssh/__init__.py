"""The first-party SSH compute provider.

``provider_factory`` is the zero-argument entry point registered in the
``organelleverse.compute_providers`` group. It is loaded exactly once, only
after explicit owner trust (spec §6), and returns the closed
:class:`~organelleverse.compute.contracts.ComputeProviderSpec` declaration.

This module imports only declaration/config code. The MCP client, SFTP
transport, and process machinery are imported lazily inside the execution path
(spec §15), so a bare ``import organelleverse`` never loads ``mcp``, opens an
SSH connection, or reads credentials.
"""

from __future__ import annotations

from organelleverse.compute.contracts import ComputeProviderSpec, TransportKind

__all__ = ["PROVIDER_ID", "SSH_DECLARATION", "provider_factory"]


PROVIDER_ID = "ssh"


#: The closed declaration returned by :func:`provider_factory`. Pinned by spec
#: §7.1/§12; changing any field invalidates trust (compute.provider_trust_stale).
SSH_DECLARATION = ComputeProviderSpec(
    provider_id=PROVIDER_ID,
    display_name="SSH Linux Provider",
    declaration_version="1.0",
    transport_kind=TransportKind.SSH_MCP,
    supported_platforms=("linux",),
    supports_prepare=True,
    supports_cancel=True,
    artifact_transports=("ssh_sftp_rsync",),
)


def provider_factory() -> ComputeProviderSpec:
    """Return the closed SSH provider declaration.

    Loaded only after explicit trust. Returns the declaration object; the
    concrete provider (MCP client over SSH stdio, SFTP transport) is constructed
    separately on the execution path.
    """
    return SSH_DECLARATION
