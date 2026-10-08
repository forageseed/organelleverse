"""Capability Plan 03, Task 1: a real core-origin bundle through the real pipeline.

Before this task, a `core`-origin bundle could never reach `ADMITTED`:
`discovery.py` never computes an `ExecutionIdentity` for `origin.channel ==
"core"` (there is no bundle-local `code/` tree to hash - the implementation
is the installed `organelleverse` distribution itself), and
`verify_capability` unconditionally demanded one, so no core capability
could ever acquire the `VerificationRecord` admission requires.

Every test below drives the real `discover_capabilities` / `verify_capability`
/ `admit_capabilities` pipeline against a genuinely importable core capability.
None of them construct a `CapabilityEntry` with `status="admitted"`, a
`VerificationRecord`, or an `execution_identity` by hand.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
from collections.abc import Callable
from pathlib import Path

import pytest

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.errors import OrganelleContractError

_CAPABILITY_ID = "demo.core_capability"

_BUNDLE_TOML = """\
schema = "organelleverse.capability.v1"

[capability]
id = "demo.core_capability"
bundle_version = "1.0.0"
implementation = "native"

[contract]
contract_version = "1.0"
title = "Core admission fixture"
description = "A minimal genuinely-importable core-channel native capability proving the real discover, verify, admit pipeline."
keywords = ["admission", "capability", "core"]
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


def _write_core_bundle(package_capabilities_root: Path) -> None:
    bundle_root = package_capabilities_root / "demo-core-capability"
    bundle_root.mkdir(parents=True)
    (bundle_root / "capability.toml").write_text(_BUNDLE_TOML, encoding="utf-8")


_EXTERNAL_CAPABILITY_ID = "demo.core_external_capability"

_EXTERNAL_BUNDLE_TOML = """\
schema = "organelleverse.capability.v1"

[capability]
id = "demo.core_external_capability"
bundle_version = "1.0.0"
implementation = "external"

[contract]
contract_version = "1.0"
title = "Core external admission fixture"
description = "An external-implementation bundle whose origins are all core, pinning that being all-core alone must not exempt it from needing an execution provider."
keywords = ["admission", "capability", "external"]
execution_mode = "inline"
stage = "analyze"
input_kind = "none"
output_kind = "result"
callable_locator = "tests.capabilities.fixtures_core_capability:run"

[contract.binding]
argument_mode = "named_parameters"
result_codec = "canonical_json"

[[contract.binding.parameters]]
name = "value"
codec = "json"

[[contract.dependencies]]
kind = "executable"
name = "demo_tool"

[[probe]]
dependency = "demo_tool"
help_argv = ["--help"]
"""


def _write_core_external_bundle(package_capabilities_root: Path) -> None:
    bundle_root = package_capabilities_root / "demo-core-external-capability"
    bundle_root.mkdir(parents=True)
    (bundle_root / "capability.toml").write_text(_EXTERNAL_BUNDLE_TOML, encoding="utf-8")


def _isolate_standard_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    write_bundle: Callable[[Path], None] = _write_core_bundle,
) -> Path:
    """Route every standard discovery root to isolated tmp directories.

    In particular, route the *only* root ``discovery.py`` labels
    ``channel="core"`` (``discovery.py:93``,
    ``SearchRoot("core", package_root / "capabilities")``) to a directory
    this test controls, exactly as
    ``test_standard_directory_channels_are_all_discovered_with_fixed_origins``
    in ``test_discovery.py`` already does for the same reason. Returns the
    isolated ``ORGANELLEVERSE_HOME`` so a test can find the verification
    store discovery itself will read from. ``write_bundle`` selects which
    bundle populates the "core" root - defaults to the native fixture used
    throughout this file; pass ``_write_core_external_bundle`` to place the
    external one instead.
    """
    package = tmp_path / "installed-package"
    home = tmp_path / "home"
    project = tmp_path / "project"
    project_capabilities = project / ".organelleverse" / "capabilities"
    project_capabilities.mkdir(parents=True)
    home.mkdir()
    package.mkdir()
    write_bundle(package / "capabilities")

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


def test_a_core_origin_bundle_is_rejected_before_any_verification_record_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Locks in the pre-verification state.

    Discovery correctly labels this bundle's origin ``"core"``, but with no
    verification record on file yet, admission rejects it - exactly like
    every other channel before its first ``verify_capability`` call.
    """
    _isolate_standard_roots(tmp_path, monkeypatch)

    index = discover_capabilities()

    entry = index.describe(_CAPABILITY_ID)
    assert entry.origins[0].channel == "core"
    assert entry.execution_identity is None
    assert entry.status is CapabilityStatus.REJECTED
    assert entry.diagnostic is not None
    assert entry.diagnostic.code == "capability.verification_missing"


def test_a_core_origin_bundle_reaches_admitted_through_the_real_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real fix, proved end to end: discover -> verify -> re-admit -> invoke.

    Before Task 1's fix, the ``verify_capability`` call below raised
    ``OrganelleContractError`` (code ``capability.execution_provider_required``,
    "capability has no content-addressed execution provider: demo.core_capability")
    because ``entry.execution_identity`` is unconditionally ``None`` for a
    pure-``core`` bundle. This is the exact failure Task 1's charter asked to
    be proven and then closed.
    """
    home = _isolate_standard_roots(tmp_path, monkeypatch)

    discovered = discover_capabilities()
    assert discovered.describe(_CAPABILITY_ID).status is CapabilityStatus.REJECTED

    store = VerificationStore(home / "verifications")
    record = verify_capability(
        _CAPABILITY_ID,
        store=store,
        environment=LocalVerificationEnvironment(discovered),
    )

    # No content-addressed code identity: a core capability's integrity
    # comes from the installed distribution itself, not a bundle-local
    # code/ digest - see verification.py's _is_pure_core_native.
    assert record.execution_identity is None
    assert record.worker_parameters == ()
    schema_properties = record.parameter_schema["properties"]
    assert isinstance(schema_properties, dict)
    assert set(schema_properties) == {"value"}

    admitted = discover_capabilities()
    entry = admitted.describe(_CAPABILITY_ID)
    assert entry.status is CapabilityStatus.ADMITTED
    assert tuple(item.capability_id for item in admitted.list()) == (_CAPABILITY_ID,)

    binding = admitted.binding_source().resolve(_CAPABILITY_ID)
    assert binding is not None
    result = binding.invoke(None, {"value": 7})
    assert result.metrics["value"] == 7


def test_being_all_core_origin_alone_does_not_exempt_an_external_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the NATIVE half of ``verification._is_pure_core_native``.

    Only a NATIVE capability binds in-process by a plain import
    (``IndexBindingSource.resolve()``'s pure-core branch) - that is the
    entire security argument for letting a pure-``core`` bundle skip the
    content-addressed ``ExecutionIdentity``. An EXTERNAL capability always
    wraps a probed executable and always needs the worker-verification
    machinery this whole task is careful not to weaken. Being all-``core``-
    origin must not, by itself, buy an EXTERNAL bundle the exemption
    ``_is_pure_core_native`` grants NATIVE ones - it must still be refused
    exactly like every other channel, with no verification record ever
    written.
    """
    _isolate_standard_roots(tmp_path, monkeypatch, write_bundle=_write_core_external_bundle)

    discovered = discover_capabilities()
    entry = discovered.describe(_EXTERNAL_CAPABILITY_ID)
    assert entry.origins[0].channel == "core"
    assert entry.execution_identity is None

    store = VerificationStore(tmp_path / "external-records")

    with pytest.raises(OrganelleContractError) as captured:
        verify_capability(
            _EXTERNAL_CAPABILITY_ID,
            store=store,
            environment=LocalVerificationEnvironment(discovered),
        )

    assert captured.value.code == "capability.execution_provider_required"
    assert store.records_for(_EXTERNAL_CAPABILITY_ID) == ()
