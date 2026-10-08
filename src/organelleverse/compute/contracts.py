"""L5 compute-target contracts.

Two layers live here:

* The **routing hook** types referenced by
  :class:`~organelleverse.assembly.environment_contracts.EnvironmentResolution`
  (the ``compute_target`` branch) and by a caller's hint — ``TransportKind``,
  ``ComputeTargetHint``, ``ResolvedComputeTarget``. These came from L5-01.
* The **provider declaration** types the first real provider (L5-02 WSL) needs —
  ``ComputeProviderSpec`` and ``ProviderCandidate``. L5-02 Task 0 adds them
  here because a trust/declaration apparatus is only justified once a provider
  exists; it lives in the provider-agnostic ``compute/`` layer so L5-03
  SSH/Slurm reuse it unchanged.

The content-based identity fields a real ``probe()`` populates
(``endpoint_fingerprint``, ``worker_identity``, ``component_identities``,
``target_digest``) are still absent from ``ResolvedComputeTarget``: nothing can
produce real fingerprints yet. They return with L5-02 Task 2 (the protocol
layer), where ``probe()`` is defined.

See ``docs/superpowers/specs/2026-07-29-l5-compute-provider-plugins-design.md``
§7.1/§7.2/§5.3 for the authoritative shapes.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Literal

from organelleverse.operations.spec import StrictSpecModel

__all__ = [
    "ComputeProviderFactory",
    "ComputeProviderSpec",
    "ComputeTargetHint",
    "ComputeTargetPolicy",
    "EnabledProvider",
    "ProviderCandidate",
    "ResolvedComputeTarget",
    "TransportKind",
]


class TransportKind(StrEnum):
    """How a provider reaches its worker (spec §7.1). Part of the public
    contract; do not rename the values."""

    WSL_MCP = "wsl_mcp"
    SSH_MCP = "ssh_mcp"
    SLURM = "slurm"
    CUSTOM = "custom"


#: Routing policy a caller may request. ``local_only`` forbids remote/WSL
#: routing; ``external_allowed`` permits it; ``auto`` is the default order.
ComputeTargetPolicy = Literal["auto", "local_only", "external_allowed"]


class ComputeTargetHint(StrictSpecModel):
    """Caller-supplied routing hint (spec §5.3).

    Never carries credentials, endpoints, distro names, paths, commands, or
    grants. It may name an enabled target by id and state a policy; the resolver
    remains decisive.
    """

    target_id: str | None = None
    policy: ComputeTargetPolicy = "auto"


class ResolvedComputeTarget(StrictSpecModel):
    """A verified external execution target (spec §7.2).

    Sibling to ``ResolvedProvider``, never a widening of it: a remote/WSL target
    has no valid host-local prefix, so pretending its paths were local would
    break identity and security.

    This is the minimal identity a routing decision needs today. The spec's
    content-based identity fields (``endpoint_fingerprint``,
    ``provider_declaration_digest``, ``worker_identity``,
    ``worker_catalog_digest``, ``component_identities``, ``target_digest``) are
    added by L5-02, where a real ``probe()`` can populate them with actual
    fingerprints rather than placeholder zeros.
    """

    target_id: str
    provider_id: str
    transport_kind: TransportKind
    platform: str
    architecture: str
    artifact_transport: str


# ---------------------------------------------------------------------------
# Provider declaration (spec §7.1, §6) — added by L5-02 Task 0
# ---------------------------------------------------------------------------


class ComputeProviderSpec(StrictSpecModel):
    """Closed, static declaration a loaded provider factory returns (spec §7.1).

    A provider is loaded only *after* its installed distribution identity has
    been trusted (spec §6). The declaration is then validated against this model
    and its canonical digest is bound to the trust record; a later declaration
    whose digest differs disables the provider until reviewed again.
    """

    provider_id: str
    display_name: str
    declaration_version: Literal["1.0"]
    transport_kind: TransportKind
    supported_platforms: tuple[str, ...]
    supports_prepare: bool
    supports_cancel: bool
    artifact_transports: tuple[str, ...]


class ProviderCandidate(StrictSpecModel):
    """A discovered-but-untrusted provider, described purely from metadata.

    Discovery (spec §6) reads only the entry-point name/value, distribution
    name/version, and a canonical digest of the installed ``RECORD`` entries via
    :mod:`importlib.metadata`. It never calls
    :meth:`importlib.metadata.EntryPoint.load`, never imports the factory, and
    never probes the host.
    """

    provider_id: str
    distribution_name: str
    distribution_version: str
    entry_point_locator: str
    distribution_record_digest: str


#: A zero-argument factory an entry point resolves to. Loaded exactly once,
#: only on the explicit ``enable()`` path, after the static installed identity
#: has been trusted. Returns the provider's closed declaration.
ComputeProviderFactory = Callable[[], ComputeProviderSpec]


class EnabledProvider(StrictSpecModel):
    """A candidate that has been explicitly trusted and whose closed declaration
    has been loaded and validated.

    ``declaration_digest`` is the canonical digest of the loaded
    :class:`ComputeProviderSpec`; trust is invalidated if the distribution
    version, locator, installed RECORD digest, or this declaration digest
    changes (spec §6).
    """

    candidate: ProviderCandidate
    declaration: ComputeProviderSpec
    declaration_digest: str
