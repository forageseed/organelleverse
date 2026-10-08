"""Build the same checked operation registry for Python and Agent callers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .index import CapabilityIndex
from .trust import TrustStore

if TYPE_CHECKING:
    from organelleverse.operations.registry import OperationRegistry


def build_registry(
    index: CapabilityIndex,
    *,
    trust_store: TrustStore | None = None,
) -> OperationRegistry:
    """Combine residual released operations with one admitted capability snapshot.

    Both Python callers and the Agent catalog should use this constructor so
    operation lookup, parameter validation and result contracts agree.  The
    index retains admission state; a discovered or rejected bundle is never
    made callable merely by constructing a registry.
    """
    from organelleverse.operations.catalog import load_release_catalog
    from organelleverse.operations.registry import OperationRegistry

    registry = OperationRegistry()
    load_release_catalog(registry)
    registry.attach_capability_source(index.binding_source(trust_store=trust_store))
    return registry
