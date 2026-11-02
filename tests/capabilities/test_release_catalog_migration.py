"""Release-catalog migration guards (Plan 03, Task 1).

The release surface is 20 operations: 16 migrated to core capability bundles
discovered through the same channel as every third-party bundle, plus four
residual operations still served by the Python catalog (see
``operations/catalog.py``'s docstring for the two structural reasons). These
tests pin the before/after contract equivalence against the frozen
``release_catalog_v0.json`` oracle, the residual catalog's scope, and the
import-safety of the discovery path.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.capabilities.release_bundles import (
    RESIDUAL_CATALOG_IDS,
    release_bundle_ids,
    release_operation_ids,
    verify_release_bundles,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fresh_release_registry():
    from organelleverse.operations.registry import _ReleaseOperationRegistry

    return _ReleaseOperationRegistry()


def test_existing_release_contracts_are_preserved_by_bundles() -> None:
    expected = json.loads((FIXTURES / "release_catalog_v0.json").read_text(encoding="utf-8"))
    bundle_ids = set(release_bundle_ids())
    assert set(expected) == bundle_ids | RESIDUAL_CATALOG_IDS

    from organelleverse.capabilities.discovery import discover_capabilities

    discovered = discover_capabilities()
    actual = {
        entry.bundle.contract.operation_id: entry.bundle.contract.model_dump(mode="json")
        for entry in discovered.entries
        if entry.bundle.contract.operation_id in bundle_ids
    }
    assert actual == {key: expected[key] for key in bundle_ids}
    assert len(actual) == 16
    assert not (RESIDUAL_CATALOG_IDS & set(actual))


def _release_bundle_diagnostics() -> dict[str, str | None]:
    from organelleverse.capabilities.discovery import discover_capabilities

    bundle_ids = set(release_bundle_ids())
    return {
        entry.capability_id: entry.diagnostic.code if entry.diagnostic else None
        for entry in discover_capabilities().entries
        if entry.capability_id in bundle_ids
    }


def test_bare_environment_serves_the_release_surface_with_an_advisory_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First-party verification is advisory: no record still serves all 20.

    The missing record is kept as a diagnostic on each admitted bundle, not
    silently dropped.
    """

    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "empty-home"))

    ids = {spec.operation_id for spec in _fresh_release_registry().list()}
    assert release_operation_ids() <= ids
    assert set(_release_bundle_diagnostics().values()) == {"capability.verification_missing"}


def test_verified_release_bundles_complete_the_release_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once verified, all 20 are served and the advisory diagnostic is gone."""

    home = tmp_path / "home"
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    verify_release_bundles(home)

    ids = {spec.operation_id for spec in _fresh_release_registry().list()}
    assert release_operation_ids() <= ids
    diagnostics = _release_bundle_diagnostics()
    assert len(diagnostics) == 16
    assert "capability.verification_missing" not in diagnostics.values()


def test_bundle_discovery_does_not_import_implementations() -> None:
    script = (
        "import sys\n"
        "from organelleverse.capabilities.discovery import discover_capabilities\n"
        "discover_capabilities()\n"
        "surfaces = (\n"
        "    'organelleverse.annotation.api',\n"
        "    'organelleverse.assembly.api',\n"
        "    'organelleverse.assembly.service',\n"
        "    'organelleverse.quality_control.service',\n"
        "    'organelleverse.fetch.operations',\n"
        "    'organelleverse.io_genome',\n"
        "    'organelleverse.io_reads',\n"
        ")\n"
        "loaded = sorted(m for m in sys.modules if m in surfaces)\n"
        "print(loaded)\n"
        "sys.exit(1 if loaded else 0)\n"
    )
    completed = _cold_interpreter(script)
    assert completed.returncode == 0, (
        f"discovery imported implementation surfaces: {completed.stdout}\n{completed.stderr}"
    )


def _cold_interpreter(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_bare_environment_serves_the_release_surface_in_a_cold_interpreter() -> None:
    script = (
        "import os, sys, tempfile\n"
        "os.environ['ORGANELLEVERSE_HOME'] = tempfile.mkdtemp()\n"
        "from organelleverse.operations.registry import registry\n"
        "ids = sorted(spec.operation_id for spec in registry.list())\n"
        "print(ids)\n"
        f"expected = {sorted(release_operation_ids())!r}\n"
        "sys.exit(0 if set(expected) <= set(ids) else 1)\n"
    )
    completed = _cold_interpreter(script)
    assert completed.returncode == 0, (
        f"unexpected bare-environment release surface: {completed.stdout}\n{completed.stderr}"
    )
