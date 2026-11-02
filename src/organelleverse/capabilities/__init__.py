"""Capability bundles: the discovery/binding envelope around OperationSpec.

Bare ``import organelleverse`` must not import this package eagerly: it pulls in
the bundle models, discovery, admission, trust and the worker, none of which
belongs in the cost of importing the core scientific surface.

That constraint is satisfied by laziness, not by unreachability. ``ov.capabilities``
resolves through ``organelleverse.__getattr__``, which imports on first attribute
access and never at import time, so the name is available to an Agent driving the
authoring loop while the bare import stays cold. ``tests/test_capabilities_reachability.py``
holds both properties together.
"""

from __future__ import annotations

from .assets import AssetIdentity, register_asset_resolver, resolve_asset_identity
from .code_identity import ExecutionIdentity, inspect_bundle_code
from .discovery import discover_capabilities, discover_capability_candidates
from .index import (
    CapabilityConflict,
    CapabilityDiagnostic,
    CapabilityEntry,
    CapabilityIndex,
    CapabilityOrigin,
    CapabilityStatus,
)
from .models import (
    BundleDocument,
    CapabilityBundle,
    CapabilityDeclaration,
    EquivalenceMethod,
    FixtureSpec,
    ImplementationKind,
    PluginCapabilityBundle,
    ProbeSpec,
)
from .parser import parse_capability_bundle
from .scaffold import ScaffoldParameter, scaffold, scaffold_plugin
from .snapshot import (
    CapabilityAdmissionSnapshot,
    CapabilitySnapshotCounts,
    CapabilitySnapshotEntry,
    build_admission_snapshot,
)
from .trust import TrustStore, trust
from .verification import (
    LocalVerificationEnvironment,
    VerificationRecord,
    VerificationStore,
    verify_capability,
)
from .worker import OneShotBundleWorkerExecutor

__all__ = [
    "AssetIdentity",
    "BundleDocument",
    "CapabilityAdmissionSnapshot",
    "CapabilityBundle",
    "CapabilityConflict",
    "CapabilityDeclaration",
    "CapabilityDiagnostic",
    "CapabilityEntry",
    "CapabilityIndex",
    "CapabilityOrigin",
    "CapabilitySnapshotCounts",
    "CapabilitySnapshotEntry",
    "CapabilityStatus",
    "EquivalenceMethod",
    "ExecutionIdentity",
    "FixtureSpec",
    "ImplementationKind",
    "LocalVerificationEnvironment",
    "OneShotBundleWorkerExecutor",
    "PluginCapabilityBundle",
    "ProbeSpec",
    "ScaffoldParameter",
    "TrustStore",
    "VerificationRecord",
    "VerificationStore",
    "build_admission_snapshot",
    "build_registry",
    "discover_capabilities",
    "discover_capability_candidates",
    "inspect_bundle_code",
    "parse_capability_bundle",
    "register_asset_resolver",
    "resolve_asset_identity",
    "scaffold",
    "scaffold_plugin",
    "trust",
    "verify_capability",
]


def __getattr__(name: str) -> object:
    if name == "build_registry":
        from .capability_registry import build_registry

        return build_registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
