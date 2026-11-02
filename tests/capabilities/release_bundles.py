"""Shared helper: verify the 16 bundle-migrated release operations into a store.

The release surface is 20 operations: 16 native core bundles (Plan 03 Task 1)
plus four residual operations still served by the Python catalog (see
``operations/catalog.py``'s docstring for the two structural reasons).

Verification of first-party core bundles is advisory (af9b1c73): every pure-core
native bundle is admitted, with a missing record kept as a diagnostic, so the
default registry lists these 20 together with the other first-party
capabilities and the ``tools.adapter`` plugin. Tests therefore assert that the
20 release operations are present, not that they are the only ones. Plugin,
bundle-local and external bundles still need a verification record; test
sessions generate records into an isolated ``ORGANELLEVERSE_HOME``.
"""

from __future__ import annotations

import json
from pathlib import Path

ORACLE = Path(__file__).parent / "fixtures" / "release_catalog_v0.json"

#: Mirrors RESIDUAL_CATALOG in scripts/capabilities/migrate_release_catalog.py.
RESIDUAL_CATALOG_IDS: frozenset[str] = frozenset(
    {
        "annotation.annotate",
        "assembly.assemble",
        "assembly.pmat_graph_build",
        "io.read_long_reads",
    }
)


def release_bundle_ids() -> tuple[str, ...]:
    """The 16 release operations served from core bundles."""

    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    return tuple(sorted(set(oracle) - RESIDUAL_CATALOG_IDS))


def release_operation_ids() -> frozenset[str]:
    """The 20 release operations: 16 bundle-served plus the four residual ones."""

    return frozenset(release_bundle_ids()) | RESIDUAL_CATALOG_IDS


def verify_release_bundles(home: Path) -> None:
    """Verify all 16 bundle-served release operations into ``home``."""

    from organelleverse.capabilities.discovery import discover_capabilities
    from organelleverse.capabilities.verification import (
        LocalVerificationEnvironment,
        VerificationStore,
        verify_capability,
    )

    index = discover_capabilities()
    environment = LocalVerificationEnvironment(index)
    store = VerificationStore(home / "verifications")
    for operation_id in release_bundle_ids():
        verify_capability(operation_id, store=store, environment=environment)