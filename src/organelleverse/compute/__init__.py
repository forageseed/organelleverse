"""OrganelleVerse L5 compute-target subsystem.

Exposes the routing-hook types (``TransportKind``, ``ComputeTargetHint``,
``ResolvedComputeTarget``) plus the provider-agnostic declaration/discovery/
trust layer (``ComputeProviderSpec``, ``ProviderCandidate``, ``EnabledProvider``,
``discover_compute_providers``, ``ComputeProviderTrustStore``) added by L5-02
Task 0. Concrete providers (WSL/SSH/Slurm) are shipped under
``compute.providers`` and register through the ``organelleverse.compute_providers``
entry-point group. Importing this package performs no provider import, subprocess,
socket, or host probe.
"""

from __future__ import annotations

from .contracts import (
    ComputeProviderFactory,
    ComputeProviderSpec,
    ComputeTargetHint,
    ComputeTargetPolicy,
    EnabledProvider,
    ProviderCandidate,
    ResolvedComputeTarget,
    TransportKind,
)
from .discovery import (
    DistributionLike,
    EntryPointLike,
    compute_provider_record_digest,
    discover_compute_providers,
)
from .trust import (
    ComputeProviderTrustStore,
    ProviderFactoryLoader,
    canonical_declaration_digest,
    default_factory_loader,
)

__all__ = [
    "ComputeProviderFactory",
    "ComputeProviderSpec",
    "ComputeProviderTrustStore",
    "ComputeTargetHint",
    "ComputeTargetPolicy",
    "DistributionLike",
    "EnabledProvider",
    "EntryPointLike",
    "ProviderCandidate",
    "ProviderFactoryLoader",
    "ResolvedComputeTarget",
    "TransportKind",
    "canonical_declaration_digest",
    "compute_provider_record_digest",
    "default_factory_loader",
    "discover_compute_providers",
]
