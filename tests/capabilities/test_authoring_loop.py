"""The one end-to-end authoring-loop test: discover -> verify -> trust -> admit -> invoke.

Capability Plan 02 Task 3's charter (``.superpowers/sdd/charters/
capability-02-task-3-authoring.md``) measured that no existing test drives a
capability bundle through the *whole* seam without forging some part of it:
the worker tests (``test_worker.py``) hand-fabricate ``status="admitted"``
via ``entry.model_copy`` and skip verification and admission outright; the
admission tests (``test_admission.py``) hand-forge a ``VerificationRecord``
and write it straight to the store rather than calling ``verify_capability``.

This test forges nothing. It writes a real ``capability.toml`` and a real
Python implementation to a temporary directory the test itself creates, then
drives that bundle through every real public step of the authoring loop:

    discover_capabilities(paths=[tmp_root])
      -> verify_capability(id, store=VerificationStore(tmp), environment=...)
      -> trust(id, entry.execution_identity, store=TrustStore(tmp))
      -> admit_capabilities(index, store=...)
      -> index.binding_source(trust_store=..., executor=...)
      -> OperationRegistry.invoke(...)

and asserts on the real value a real subprocess worker actually returned.
The bundle declares zero fixtures, so ``evaluate_fixture`` is never called -
this proves the loop the charter calls closed, not the fixture-equivalence
path the charter explicitly defers to individual authors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.capabilities.admission import admit_capabilities
from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.trust import TrustStore, trust
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.capabilities.worker import OneShotBundleWorkerExecutor
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.registry import OperationRegistry

_CAPABILITY_ID = "demo.authoring"

_CAPABILITY_TOML = """\
schema = "organelleverse.capability.v1"

[capability]
id = "demo.authoring"
bundle_version = "1.0.0"
implementation = "native"

[contract]
contract_version = "1.0"
title = "Authoring loop demo"
description = "A disk-based capability bundle proving the authoring loop end-to-end."
keywords = ["authoring", "capability", "demo"]
execution_mode = "inline"
stage = "analyze"
input_kind = "none"
output_kind = "result"
callable_locator = "authoring_pkg.impl:run"

[contract.binding]
argument_mode = "named_parameters"
result_codec = "canonical_json"

[[contract.binding.parameters]]
name = "value"
codec = "json"
"""

_IMPLEMENTATION_SOURCE = """\
def run(*, value: int = 1):
    return {
        "schema_version": "organelleverse.result.v1",
        "kind": "result",
        "operation_id": "demo.authoring",
        "scope": "none",
        "status": "ok",
        "metrics": {"doubled": value * 2},
    }
"""


def _write_bundle(root: Path) -> Path:
    """Write a real, disk-based, zero-fixture capability bundle under *root*."""
    bundle_root = root / "demo-authoring"
    bundle_root.mkdir(parents=True)
    (bundle_root / "capability.toml").write_text(_CAPABILITY_TOML, encoding="utf-8")
    package = bundle_root / "code" / "authoring_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "impl.py").write_text(_IMPLEMENTATION_SOURCE, encoding="utf-8")
    return bundle_root


def test_authoring_loop_discovers_verifies_trusts_admits_and_invokes_for_real(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Hermetic isolation: nothing here should read or write a real developer's
    # ~/.organelleverse, and the worker's managed staging/run directories must
    # land under this test's own tmp_path.
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))

    search_root = tmp_path / "capabilities"
    _write_bundle(search_root)

    # 1. discover_capabilities(paths=[tmp_root])
    discovered = discover_capabilities(paths=[search_root])
    entry = discovered.describe(_CAPABILITY_ID)
    assert entry.execution_identity is not None
    # Nothing has verified this bundle yet: discovery's own internal admission
    # pass (against the default, untouched store) cannot have admitted it.
    assert entry.status is not CapabilityStatus.ADMITTED

    # 2. verify_capability(id, store=VerificationStore(tmp), environment=<adapter>)
    verification_store = VerificationStore(tmp_path / "verifications")
    environment = LocalVerificationEnvironment(discovered)
    record = verify_capability(
        _CAPABILITY_ID,
        store=verification_store,
        environment=environment,
    )
    assert record.capability_id == _CAPABILITY_ID
    assert record.bundle_content_hash == entry.content_hash
    assert record.execution_identity == entry.execution_identity
    assert record.equivalence == ()  # zero-fixture bundle: evaluate_fixture never ran

    # 3. trust(id, entry.execution_identity, store=TrustStore(tmp))
    trust_store = TrustStore(tmp_path / "trust.json")
    trust(_CAPABILITY_ID, entry.execution_identity, store=trust_store)

    # 4. admit_capabilities(index, store=...)
    admitted_index = admit_capabilities(discovered, store=verification_store)
    admitted_entry = admitted_index.describe(_CAPABILITY_ID)
    assert admitted_entry.status is CapabilityStatus.ADMITTED, admitted_entry.diagnostic
    assert tuple(item.capability_id for item in admitted_index.list()) == (_CAPABILITY_ID,)

    # 5. index.binding_source(trust_store=..., executor=...)
    executor = OneShotBundleWorkerExecutor()
    binding_source = admitted_index.binding_source(trust_store=trust_store, executor=executor)

    # 6. OperationRegistry.invoke(...) -- and assert the real returned value.
    registry = OperationRegistry(capability_source=binding_source)
    result = registry.invoke(_CAPABILITY_ID, input=None, parameters={"value": 21})

    assert isinstance(result, OrganelleResult)
    assert result.metrics["doubled"] == 42
