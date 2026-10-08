"""``scaffold(...)`` must generate a bundle that passes its own gates.

Capability Plan 02 Task 3's charter (``.superpowers/sdd/charters/
capability-02-task-3-authoring.md``, deliverable 4) requires a scaffold whose
output passes the authoring loop proven by ``test_authoring_loop.py``
(``discover_capabilities -> verify_capability(LocalVerificationEnvironment) ->
trust -> admit_capabilities -> invoke``) with **no manual edit** - forging
nothing, hand-fixing nothing.

Each test below targets exactly one hard requirement from the charter's
binding invariants:

1. ``test_scaffold_places_generated_code_under_bundle_local_code_single_root_package``
   - generated code lives in bundle-local ``code/<single_root_package>/`` and
   is covered by the bundle content hash (the one-root rule enforced at
   ``code_identity.py:356-367``).
2. ``test_scaffold_output_passes_full_authoring_loop_and_returns_computed_value``
   - the scaffold's own output survives the real loop unmodified and returns
   a real computed value from a real subprocess worker.
3. ``test_scaffold_variant_with_code_outside_code_directory_fails_discovery``
   - a bundle whose importable code has escaped ``code/`` must fail
   discovery/validation, not silently succeed.
4. ``test_mutating_scaffolded_code_byte_changes_bundle_hash_and_fails_admission``
   - one mutated byte anywhere under ``code/`` must change the bundle content
   hash and make a previously-verified bundle fail admission.
5. ``test_scaffold_with_forbidden_final_write_parameter_is_provider_required_before_inspection``
   - a scaffolded capability that declares a forbidden final-write parameter
   name must fail with ``capability.execution_provider_required`` *before*
   the worker's real ``inspect()`` (the controlled-subprocess import) ever
   runs.
6. ``test_scaffold_readme_warns_verification_executes_untrusted_code``
   - the generated README must say plainly that ``verify_capability``
   executes the bundle's code before any trust record exists; the charter
   calls a scaffold that hides this a trap.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.capabilities.admission import admit_capabilities
from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.hashing import hash_bundle
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.scaffold import ScaffoldParameter, scaffold
from organelleverse.capabilities.trust import TrustStore, trust
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.capabilities.worker import OneShotBundleWorkerExecutor
from organelleverse.core.errors import OrganelleContractError, OrganelleInputError
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.registry import OperationRegistry


def test_scaffold_places_generated_code_under_bundle_local_code_single_root_package(
    tmp_path: Path,
) -> None:
    from organelleverse.capabilities.code_identity import inspect_bundle_code

    capability_id = "demo.layout"
    bundle_root = scaffold(
        tmp_path / "demo-layout",
        capability_id=capability_id,
        title="Layout demo",
        description="Proves generated code lives under code/<single_root_package>.",
    )

    code_root = bundle_root / "code"
    assert code_root.is_dir()
    top_level = list(code_root.iterdir())
    assert len(top_level) == 1, "code/ must contain exactly one root package"
    package_dir = top_level[0]
    assert package_dir.is_dir()
    assert package_dir.name.isidentifier()
    assert (package_dir / "__init__.py").is_file()
    assert (package_dir / "impl.py").is_file()

    content_hash = hash_bundle(bundle_root)
    identity = inspect_bundle_code(
        bundle_root,
        capability_id=capability_id,
        bundle_content_hash=content_hash,
        callable_locator=f"{package_dir.name}.impl:run",
    )
    assert identity.bundle_content_hash == content_hash

    # The bundle content hash must actually cover the code tree: a byte
    # change under code/ has to move it, or "covered by the content hash"
    # would be a documentation claim rather than a fact.
    impl_path = package_dir / "impl.py"
    impl_path.write_bytes(impl_path.read_bytes() + b"\n# touch\n")
    assert hash_bundle(bundle_root) != content_hash


def test_scaffold_output_passes_full_authoring_loop_and_returns_computed_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))

    capability_id = "demo.scaffolded"
    search_root = tmp_path / "capabilities"
    bundle_root = scaffold(
        search_root / "demo-scaffolded",
        capability_id=capability_id,
        title="Scaffolded authoring demo",
        description="A bundle generated end-to-end by scaffold(), never hand-edited.",
        parameters=(ScaffoldParameter(name="value", annotation="int", default=1),),
    )
    assert bundle_root == search_root / "demo-scaffolded"

    # 1. discover_capabilities(paths=[search_root])
    discovered = discover_capabilities(paths=[search_root])
    entry = discovered.describe(capability_id)
    assert entry.execution_identity is not None
    assert entry.status is not CapabilityStatus.ADMITTED

    # 2. verify_capability(id, store=VerificationStore(tmp), environment=<adapter>)
    verification_store = VerificationStore(tmp_path / "verifications")
    environment = LocalVerificationEnvironment(discovered)
    record = verify_capability(capability_id, store=verification_store, environment=environment)
    assert record.capability_id == capability_id
    assert record.bundle_content_hash == entry.content_hash
    assert record.execution_identity == entry.execution_identity
    assert record.equivalence == ()  # zero-fixture bundle

    # 3. trust(id, entry.execution_identity, store=TrustStore(tmp))
    trust_store = TrustStore(tmp_path / "trust.json")
    trust(capability_id, entry.execution_identity, store=trust_store)

    # 4. admit_capabilities(index, store=...)
    admitted_index = admit_capabilities(discovered, store=verification_store)
    admitted_entry = admitted_index.describe(capability_id)
    assert admitted_entry.status is CapabilityStatus.ADMITTED, admitted_entry.diagnostic

    # 5. index.binding_source(trust_store=..., executor=...)
    executor = OneShotBundleWorkerExecutor()
    binding_source = admitted_index.binding_source(trust_store=trust_store, executor=executor)

    # 6. OperationRegistry.invoke(...) -- assert the real returned value.
    registry = OperationRegistry(capability_source=binding_source)
    result = registry.invoke(capability_id, input=None, parameters={"value": 21})

    assert isinstance(result, OrganelleResult)
    assert result.metrics["doubled"] == 42
    assert result.metrics["value"] == 21


def test_scaffold_variant_with_code_outside_code_directory_fails_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))

    capability_id = "demo.misplaced"
    search_root = tmp_path / "capabilities"
    bundle_root = scaffold(
        search_root / "demo-misplaced",
        capability_id=capability_id,
        title="Misplaced code demo",
        description="A tampered scaffold output whose code has escaped code/.",
    )

    code_root = bundle_root / "code"
    package_dirs = [item for item in code_root.iterdir() if item.is_dir()]
    assert len(package_dirs) == 1
    package_dir = package_dirs[0]

    # Simulate a broken scaffold variant: importable code sitting directly
    # under the bundle root, a sibling of (now-empty) code/, rather than
    # inside it.
    escaped = bundle_root / package_dir.name
    package_dir.rename(escaped)

    discovered = discover_capabilities(paths=[search_root])
    with pytest.raises(OrganelleInputError):
        discovered.describe(capability_id)

    codes = {
        diagnostic.code
        for diagnostic in discovered.diagnostics()
        if diagnostic.capability_id == capability_id
    }
    assert "capability.code_layout_invalid" in codes


def test_mutating_scaffolded_code_byte_changes_bundle_hash_and_fails_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))

    capability_id = "demo.mutate"
    search_root = tmp_path / "capabilities"
    bundle_root = scaffold(
        search_root / "demo-mutate",
        capability_id=capability_id,
        title="Mutation demo",
        description="Proves one mutated byte under code/ breaks admission.",
    )

    discovered = discover_capabilities(paths=[search_root])
    entry_before = discovered.describe(capability_id)
    hash_before = entry_before.content_hash

    store = VerificationStore(tmp_path / "verifications")
    environment = LocalVerificationEnvironment(discovered)
    verify_capability(capability_id, store=store, environment=environment)

    code_root = bundle_root / "code"
    package_dir = next(item for item in code_root.iterdir() if item.is_dir())
    impl_path = package_dir / "impl.py"
    original = bytearray(impl_path.read_bytes())
    original[0] ^= 0xFF  # flip exactly one byte
    impl_path.write_bytes(bytes(original))

    rediscovered = discover_capabilities(paths=[search_root])
    entry_after = rediscovered.describe(capability_id)
    assert entry_after.content_hash != hash_before

    admitted_index = admit_capabilities(rediscovered, store=store)
    admitted_entry = admitted_index.describe(capability_id)
    assert admitted_entry.status is not CapabilityStatus.ADMITTED
    assert admitted_entry.diagnostic is not None
    assert admitted_entry.diagnostic.code == "capability.verification_stale"


def test_scaffold_with_forbidden_final_write_parameter_is_provider_required_before_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))

    capability_id = "demo.forbidden_param"
    search_root = tmp_path / "capabilities"
    scaffold(
        search_root / "demo-forbidden",
        capability_id=capability_id,
        title="Forbidden parameter demo",
        description="Declares a forbidden final-write parameter name on purpose.",
        parameters=(ScaffoldParameter(name="output_dir", annotation="int", default=1),),
    )

    discovered = discover_capabilities(paths=[search_root])
    entry = discovered.describe(capability_id)
    # Discovery only ever hashes and statically inspects source bytes; it
    # must not have executed anything to reach this point.
    assert entry.execution_identity is not None

    executor = OneShotBundleWorkerExecutor()
    inspected_calls: list[object] = []
    original_inspect = executor.inspect

    def _spy_inspect(entry_argument: object) -> object:
        inspected_calls.append(entry_argument)
        return original_inspect(entry_argument)  # type: ignore[arg-type]

    monkeypatch.setattr(executor, "inspect", _spy_inspect)

    environment = LocalVerificationEnvironment(discovered, executor=executor)
    store = VerificationStore(tmp_path / "verifications")

    with pytest.raises(OrganelleContractError) as excinfo:
        verify_capability(capability_id, store=store, environment=environment)

    assert excinfo.value.code == "capability.execution_provider_required"
    # The failure must come from static schema inspection, never from the
    # worker subprocess actually importing the bundle's code.
    assert inspected_calls == []


def test_scaffold_readme_warns_verification_executes_untrusted_code(tmp_path: Path) -> None:
    bundle_root = scaffold(
        tmp_path / "demo-readme",
        capability_id="demo.readme",
        title="README demo",
        description="Proves the generated README states the verification security property.",
    )

    readme_path = bundle_root / "README.md"
    assert readme_path.is_file()
    text = readme_path.read_text(encoding="utf-8").lower()
    assert "verify_capability" in text
    assert "executes" in text
    assert "before any trust record exists" in text
