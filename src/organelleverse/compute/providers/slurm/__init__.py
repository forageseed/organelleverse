"""The first-party Slurm compute provider.

``provider_factory`` is the zero-argument entry point registered in the
``organelleverse.compute_providers`` group, loaded exactly once after explicit
owner trust (spec §6). It returns the closed declaration. The job renderer,
state mapping, scheduler commands, and shared-filesystem transport live in
sibling modules and are exercised on the execution path; a bare
``import organelleverse`` loads none of them (spec §15).
"""

from __future__ import annotations

from organelleverse.compute.contracts import ComputeProviderSpec, TransportKind

__all__ = ["PROVIDER_ID", "SLURM_DECLARATION", "provider_factory"]


PROVIDER_ID = "slurm"


#: Closed declaration (spec §7.1/§13). Changing any field invalidates trust.
SLURM_DECLARATION = ComputeProviderSpec(
    provider_id=PROVIDER_ID,
    display_name="Slurm Provider",
    declaration_version="1.0",
    transport_kind=TransportKind.SLURM,
    supported_platforms=("linux",),
    supports_prepare=True,
    supports_cancel=True,
    artifact_transports=("shared_filesystem",),
)


def provider_factory() -> ComputeProviderSpec:
    """Return the closed Slurm provider declaration."""
    return SLURM_DECLARATION
