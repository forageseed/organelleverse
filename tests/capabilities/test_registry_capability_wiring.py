"""Capability Plan 03, Task 2: wire the seam into the default registry.

Before this task, `attach_capability_source` (`operations/registry.py`) had no
production caller: the only non-test reference was the `__init__` self-call
at `registry.py:170`, and the module-level default registry
(`registry = _ReleaseOperationRegistry()`, `registry.py:841`) was constructed
with no capability source at all. Everything Plan 02 built - discovery,
verification, trust, admission, the worker - and everything Task 1 fixed for
the `core` channel was reachable only if some caller assembled it by hand.
Nothing shipped it.

Every test below drives the real `discover_capabilities` / `verify_capability`
/ `admit_capabilities` pipeline against a genuinely importable core-channel
bundle, exactly like `tests/capabilities/test_core_admission.py` does for
Task 1. None of them construct a `CapabilityEntry` with `status="admitted"`,
a `VerificationRecord`, or an `execution_identity` by hand.

Each test builds its own fresh `_ReleaseOperationRegistry()` instance rather
than importing the shared `organelleverse.operations.registry.registry`
singleton. The singleton's catalog (and any capability source it attaches)
loads exactly once per process and is cached forever after
(`_catalog_loaded`), so whichever test in the suite happens to touch it first
would poison every other test's view of it. A fresh instance gives every test
its own `_ensure_catalog()` run against its own isolated filesystem roots.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import logging
from pathlib import Path

import pytest

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.registry import _ReleaseOperationRegistry

_CAPABILITY_ID = "capability03.wiring_probe"

_BUNDLE_TOML = """\
schema = "organelleverse.capability.v1"

[capability]
id = "capability03.wiring_probe"
bundle_version = "1.0.0"
implementation = "native"

[contract]
contract_version = "1.0"
title = "Default registry wiring probe"
description = "A minimal genuinely-importable core-channel capability proving the default registry reaches real admitted capabilities without a hand-assembled caller."
keywords = ["admission", "capability", "wiring"]
execution_mode = "inline"
stage = "analyze"
input_kind = "none"
output_kind = "result"
callable_locator = "tests.capabilities.fixtures_core_capability:run"

[contract.binding]
argument_mode = "named_parameters"
result_codec = "canonical"

[[contract.binding.parameters]]
name = "value"
codec = "json"
"""

# An id still served, in-process, by the residual release catalog
# (`operations/catalog.py`'s registration of `ASSEMBLE_SPEC` - one of the four
# operations the bundle contract cannot yet express). Used to pin the
# silent-shadowing decision with real admitted content under a colliding id,
# not a hand-built collision.
_SHADOWED_CAPABILITY_ID = "assembly.assemble"

_SHADOW_BUNDLE_TOML = """\
schema = "organelleverse.capability.v1"

[capability]
id = "assembly.assemble"
bundle_version = "1.0.0"
implementation = "native"

[contract]
contract_version = "1.0"
title = "Shadowed capability probe"
description = "A real admitted core capability that happens to share an id with a released operation, proving the release binding keeps winning even once a real capability source is attached."
keywords = ["admission", "capability", "shadow"]
execution_mode = "inline"
stage = "analyze"
input_kind = "none"
output_kind = "result"
callable_locator = "tests.capabilities.fixtures_core_capability:run"

[contract.binding]
argument_mode = "named_parameters"
result_codec = "canonical"

[[contract.binding.parameters]]
name = "value"
codec = "json"
"""


def _isolate_standard_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    bundle_toml: str = _BUNDLE_TOML,
    bundle_dirname: str = "wiring-probe",
) -> Path:
    """Route every standard discovery root to isolated tmp directories.

    Same shape as `test_core_admission.py`'s `_isolate_standard_roots`: the
    ``core`` channel (the only channel `discovery.py` labels ``channel="core"``,
    `discovery.py:93`) is routed to a bundle this test controls. Returns the
    isolated ``ORGANELLEVERSE_HOME`` so a test can build the matching
    `VerificationStore`.
    """
    package = tmp_path / "installed-package"
    home = tmp_path / "home"
    project = tmp_path / "project"
    project_capabilities = project / ".organelleverse" / "capabilities"
    project_capabilities.mkdir(parents=True)
    home.mkdir()
    bundle_root = package / "capabilities" / bundle_dirname
    bundle_root.mkdir(parents=True)
    (bundle_root / "capability.toml").write_text(bundle_toml, encoding="utf-8")

    def packaged_files(package_name: str) -> Path:
        return package

    def no_entry_points(**kwargs: object) -> tuple[()]:
        return ()

    monkeypatch.setattr(importlib.resources, "files", packaged_files)
    monkeypatch.setattr(importlib.metadata, "entry_points", no_entry_points)
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(project)
    return home


def _seed_release_bundles(tmp_path: Path) -> None:
    """Copy the 16 real release bundle manifests into the isolated core root.

    `_isolate_standard_roots` reroutes the core channel to
    ``tmp_path / "installed-package" / "capabilities"``; copying the checked-in
    manifests verbatim keeps their content (and therefore their
    content-addressed verification keys) identical to production.
    """
    import shutil

    from tests.capabilities.release_bundles import release_bundle_ids

    slug = lambda operation_id: operation_id.replace("_", "-").replace(".", "-")  # noqa: E731
    source_root = Path(__file__).resolve().parents[2] / "src" / "organelleverse" / "capabilities"
    target_root = tmp_path / "installed-package" / "capabilities"
    for operation_id in release_bundle_ids():
        shutil.copytree(
            source_root / slug(operation_id),
            target_root / slug(operation_id),
        )


def _admit(capability_id: str, home: Path) -> None:
    """Drive discover -> verify -> re-discover so *capability_id* reaches ADMITTED."""
    discovered = discover_capabilities()
    assert discovered.describe(capability_id).status is CapabilityStatus.REJECTED

    store = VerificationStore(home / "verifications")
    verify_capability(
        capability_id,
        store=store,
        environment=LocalVerificationEnvironment(discovered),
    )

    admitted = discover_capabilities()
    assert admitted.describe(capability_id).status is CapabilityStatus.ADMITTED


def test_a_fresh_default_registry_reaches_a_real_admitted_capability_with_no_manual_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real fix, proved end to end: nobody calls `attach_capability_source`.

    Before this task, a brand-new `_ReleaseOperationRegistry()` - exactly the
    class the shared default `registry` is - never attaches any capability
    source of its own accord, so `_CAPABILITY_ID` stays unreachable through it
    no matter how real its `ADMITTED` status is.
    """
    home = _isolate_standard_roots(tmp_path, monkeypatch)
    _admit(_CAPABILITY_ID, home)

    fresh_registry = _ReleaseOperationRegistry()

    assert _CAPABILITY_ID in {spec.operation_id for spec in fresh_registry.list()}

    binding = fresh_registry.require(_CAPABILITY_ID)
    result = binding.invoke(None, {"value": 11})

    assert isinstance(result, OrganelleResult)
    assert result.metrics["value"] == 11


def test_reaching_the_capability_does_not_break_the_twenty_released_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Binding invariant 1: attaching a real source changes nothing about the 20.

    The release surface is 20 operations: 16 migrated to core bundles (Plan 03
    Task 1) plus 4 residual catalog operations. The core channel is rerouted
    to an isolated package root seeded with byte-copies of the real release
    bundles, and the 16 are verified into the isolated home so the default
    registry serves them exactly as production would.
    """
    from tests.capabilities.release_bundles import verify_release_bundles

    home = _isolate_standard_roots(tmp_path, monkeypatch)
    _seed_release_bundles(tmp_path)
    verify_release_bundles(home)
    _admit(_CAPABILITY_ID, home)

    fresh_registry = _ReleaseOperationRegistry()
    fresh_registry.list()  # trigger catalog + capability source attachment

    release_specs = tuple(
        spec for spec in fresh_registry.list() if spec.operation_id != _CAPABILITY_ID
    )
    bindings = tuple(fresh_registry.require(spec.operation_id) for spec in release_specs)

    assert len(bindings) == 20
    assert all(binding.invocation_strategy is None for binding in bindings)


def test_a_capability_sharing_a_released_id_stays_shadowed_but_is_diagnosable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Pins the silent-shadowing decision with a real admitted collision.

    `io.read_fasta_genome` is already served in-process by the release
    catalog. A real, ADMITTED capability bundle for the same id must not
    replace it (`registry.py:482-484`, `393-403`) - but it must no longer be
    an invisible debugging trap: `shadowed_capability_ids()` reports it, and
    attaching the source logs a warning naming it.
    """
    home = _isolate_standard_roots(
        tmp_path,
        monkeypatch,
        bundle_toml=_SHADOW_BUNDLE_TOML,
        bundle_dirname="shadow-probe",
    )
    _admit(_SHADOWED_CAPABILITY_ID, home)

    fresh_registry = _ReleaseOperationRegistry()
    with caplog.at_level(logging.WARNING, logger="organelleverse.operations.registry"):
        binding = fresh_registry.require(_SHADOWED_CAPABILITY_ID)

    # The release binding won: it resolves the real assembly service through
    # the residual catalog, not the capability's `fixtures_core_capability:run`
    # stand-in.
    assert binding.spec.callable_locator == "organelleverse.assembly.api:assemble"
    assert binding.invocation_strategy is None

    assert fresh_registry.shadowed_capability_ids() == (_SHADOWED_CAPABILITY_ID,)
    assert any(_SHADOWED_CAPABILITY_ID in record.message for record in caplog.records)
