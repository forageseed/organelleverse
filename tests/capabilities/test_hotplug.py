"""Focused tests for the governed hot-plug lifecycle (TASK-B backend).

Every test runs against explicit, per-test temporary search roots and stores,
so assertions are incremental (counts relative to the test's own baseline),
never against environment-dependent registry totals.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from organelleverse.capabilities.admission import LocalAdmissionEnvironment
from organelleverse.capabilities.hashing import hash_fixture_dataset
from organelleverse.capabilities.hotplug import (
    CapabilityChangeKind,
    HotPlugReport,
    PluginHotPlugController,
    WatchState,
)
from organelleverse.capabilities.index import CapabilityEntry, CapabilityStatus
from organelleverse.capabilities.trust import TrustStore, trust
from organelleverse.capabilities.verification import (
    VerificationRecord,
    VerificationStore,
    compute_environment_key,
)
from organelleverse.core.errors import OrganelleError, OrganelleInputError
from organelleverse.operations.python_binding import worker_parameter_schema


def _bundle_text(capability_id: str, *, version: str = "1.0.0") -> str:
    return f'''\
schema = "organelleverse.capability.v1"

[capability]
id = "{capability_id}"
bundle_version = "{version}"
implementation = "native"

[contract]
contract_version = "1.0"
title = "Demo"
description = "A deterministic hot-plug fixture for capability tests."
keywords = ["capability", "demo", "hotplug"]
execution_mode = "inline"
stage = "analyze"
input_kind = "none"
output_kind = "result"
organelle_types = ["mitochondrion"]
callable_locator = "thirdparty.impl:run"
side_effects = []
deterministic = true
idempotent = true
cacheable = true
references = []

[contract.binding]
argument_mode = "named_parameters"
result_codec = "canonical_json"

[contract.fallback]
allowed = false
'''


def _write_bundle(root: Path, capability_id: str, *, marker: str = "one") -> Path:
    bundle_root = root / capability_id.replace(".", "-")
    bundle_root.mkdir(parents=True)
    (bundle_root / "capability.toml").write_text(
        _bundle_text(capability_id), encoding="utf-8"
    )
    package = bundle_root / "code" / "thirdparty"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "impl.py").write_text(
        f'MARKER = "{marker}"\n\n\ndef run():\n    return None\n', encoding="utf-8"
    )
    return bundle_root


def _admission_record(entry: CapabilityEntry, store: VerificationStore) -> VerificationRecord:
    """Write the verification record that admits *entry* in its current environment."""
    environment = LocalAdmissionEnvironment().observe_environment(entry)
    record = VerificationRecord(
        protocol_version="1.0",
        capability_id=entry.capability_id,
        bundle_content_hash=entry.content_hash,
        execution_identity=entry.execution_identity,
        fixture_dataset_hash=hash_fixture_dataset(entry.bundle_root),
        adapter_requirements={},
        observed_environment=environment.observed_environment,
        environment_key=compute_environment_key(environment),
        equivalence=(),
        platform=environment.platform,
        worker_parameters=(),
        parameter_schema=worker_parameter_schema(entry.bundle, ()),
        verified_at="2026-08-25T00:00:00Z",
        organelleverse_version="0.0.1",
    )
    store.write(record)
    return record


class _ManualTimer:
    def __init__(self, callback: Callable[[], None]) -> None:
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class _ManualTimers:
    def __init__(self) -> None:
        self.created: list[_ManualTimer] = []

    def factory(self, _delay: float, callback: Callable[[], None]) -> _ManualTimer:
        timer = _ManualTimer(callback)
        self.created.append(timer)
        return timer

    def fire(self) -> None:
        live = [timer for timer in self.created if not timer.cancelled]
        assert len(live) == 1, f"expected exactly one live debounce timer, found {len(live)}"
        timer = live[0]
        timer.cancelled = True
        timer.callback()


class _FakeWatcher:
    def __init__(self, *, failure: OrganelleError | None = None) -> None:
        self.failure = failure
        self.started_roots: tuple[Path, ...] | None = None
        self.on_event: Callable[[Path], None] | None = None
        self.syncs = 0
        self.stopped = False

    def start(self, roots: tuple[Path, ...], on_event: Callable[[Path], None]) -> None:
        if self.failure is not None:
            raise self.failure
        self.started_roots = roots
        self.on_event = on_event

    def sync(self, roots: tuple[Path, ...]) -> None:
        self.syncs += 1

    def stop(self) -> None:
        self.stopped = True

    def emit(self, path: Path) -> None:
        assert self.on_event is not None, "watcher was not started"
        self.on_event(path)


def _controller(
    *,
    paths: tuple[Path, ...],
    timers: _ManualTimers,
    watcher: _FakeWatcher,
    store: VerificationStore,
) -> PluginHotPlugController:
    return PluginHotPlugController(
        paths=paths,
        entry_points=(),
        debounce_seconds=30.0,
        timer_factory=timers.factory,
        watcher=watcher,
        verification_store=store,
    )


@pytest.fixture
def rig(tmp_path: Path):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    timers = _ManualTimers()
    watcher = _FakeWatcher()
    store = VerificationStore(tmp_path / "verifications")
    trust_store = TrustStore(tmp_path / "trust.json")
    controller = _controller(
        paths=(plugins,), timers=timers, watcher=watcher, store=store
    )
    return controller, plugins, timers, watcher, store, trust_store


def test_start_scans_declared_roots_and_starts_watcher(rig) -> None:
    controller, plugins, _timers, watcher, _store, _trust = rig

    report = controller.start()

    assert report.revision == 1
    assert report.changes == ()
    assert report.watch_state is WatchState.WATCHING
    assert watcher.started_roots == (plugins.resolve(),)
    snapshot = controller.snapshot()
    assert snapshot.revision == 1
    assert snapshot.index.entries == ()
    controller.stop()
    assert watcher.stopped


def test_new_bundle_is_discovered_but_not_admitted_or_callable(rig) -> None:
    controller, plugins, timers, watcher, _store, _trust = rig
    controller.start()
    bundle = _write_bundle(plugins, "thirdparty.demo")

    watcher.emit(bundle / "capability.toml")
    timers.fire()

    snapshot = controller.snapshot()
    assert snapshot.revision == 2
    change = next(
        item for item in snapshot_report_changes(controller) if item.capability_id == "thirdparty.demo"
    )
    assert change.kind is CapabilityChangeKind.ADDED
    entry = snapshot.index.describe("thirdparty.demo")
    assert entry.status is CapabilityStatus.REJECTED
    assert entry.diagnostic is not None
    assert entry.diagnostic.code == "capability.verification_missing"
    # Not callable without verification and trust: binding resolution fails closed.
    assert snapshot.binding_source().resolve("thirdparty.demo") is None


def snapshot_report_changes(controller: PluginHotPlugController):
    return controller.last_report.changes


def test_repeated_editor_events_debounce_into_one_rescan(rig) -> None:
    controller, plugins, timers, watcher, _store, _trust = rig
    controller.start()
    reports: list[HotPlugReport] = []
    controller.add_listener(reports.append)
    bundle = _write_bundle(plugins, "thirdparty.demo")

    watcher.emit(bundle / "capability.toml")
    watcher.emit(bundle / "code" / "thirdparty" / "__init__.py")
    watcher.emit(bundle / "code" / "thirdparty" / "impl.py")

    assert len(timers.created) == 3
    assert timers.created[0].cancelled
    assert timers.created[1].cancelled
    assert not timers.created[2].cancelled
    assert reports == []

    timers.fire()

    assert len(reports) == 1
    assert reports[0].revision == 2
    assert [change.kind for change in reports[0].changes] == [CapabilityChangeKind.ADDED]


def test_modified_admitted_bundle_loses_verification_and_trust(rig) -> None:
    controller, plugins, timers, watcher, store, trust_store = rig
    bundle = _write_bundle(plugins, "thirdparty.demo")
    controller.start()
    entry = controller.snapshot().index.describe("thirdparty.demo")
    _admission_record(entry, store)
    trusted = trust("thirdparty.demo", entry.execution_identity, store=trust_store)
    controller.rescan()
    admitted = controller.snapshot().index.describe("thirdparty.demo")
    assert admitted.status is CapabilityStatus.ADMITTED
    previous_hash = admitted.content_hash

    (bundle / "code" / "thirdparty" / "impl.py").write_text(
        'MARKER = "two"\n\n\ndef run():\n    return None\n', encoding="utf-8"
    )
    watcher.emit(bundle / "code" / "thirdparty" / "impl.py")
    timers.fire()

    snapshot = controller.snapshot()
    entry_after = snapshot.index.describe("thirdparty.demo")
    assert entry_after.content_hash != previous_hash
    assert entry_after.status is CapabilityStatus.REJECTED
    assert entry_after.diagnostic is not None
    assert entry_after.diagnostic.code == "capability.verification_stale"
    change = next(
        item
        for item in controller.last_report.changes
        if item.capability_id == "thirdparty.demo"
    )
    assert change.kind is CapabilityChangeKind.MODIFIED
    assert change.previous_content_hash == previous_hash
    assert change.current_content_hash == entry_after.content_hash
    # Trust is of exact content: the old identity stays trusted, the new one
    # is not, and the old trust must not be carried over.
    assert trust_store.is_trusted(trusted.execution_identity)
    assert not trust_store.is_trusted(entry_after.execution_identity.digest)
    assert snapshot.binding_source().resolve("thirdparty.demo") is None


def test_deleted_bundle_goes_offline_while_old_registry_view_remains_readable(rig) -> None:
    controller, plugins, timers, watcher, store, trust_store = rig
    bundle = _write_bundle(plugins, "thirdparty.demo")
    controller.start()
    entry = controller.snapshot().index.describe("thirdparty.demo")
    _admission_record(entry, store)
    trust("thirdparty.demo", entry.execution_identity, store=trust_store)
    controller.rescan()
    pinned_snapshot = controller.snapshot()
    pinned_source = pinned_snapshot.binding_source(trust_store=trust_store)
    pinned_binding = pinned_source.resolve("thirdparty.demo")
    assert pinned_binding is not None

    shutil.rmtree(bundle)
    watcher.emit(bundle / "capability.toml")
    timers.fire()

    snapshot = controller.snapshot()
    change = next(
        item
        for item in controller.last_report.changes
        if item.capability_id == "thirdparty.demo"
    )
    assert change.kind is CapabilityChangeKind.REMOVED
    with pytest.raises(OrganelleInputError):
        snapshot.index.describe("thirdparty.demo")
    # No new run of that version can be created (unresolvable in the new snapshot)...
    assert snapshot.binding_source(trust_store=trust_store).resolve("thirdparty.demo") is None
    # The old registry view remains internally stable. This does not claim that
    # worker execution survives deletion of the live bundle directory.
    assert pinned_snapshot.index.describe("thirdparty.demo").status is CapabilityStatus.ADMITTED
    assert pinned_source.resolve("thirdparty.demo") is pinned_binding


def test_unparseable_bundle_fails_closed_with_diagnostic(rig) -> None:
    controller, plugins, timers, watcher, _store, _trust = rig
    controller.start()
    bundle = plugins / "broken"
    bundle.mkdir()
    (bundle / "capability.toml").write_text("this is not = valid toml =", encoding="utf-8")

    watcher.emit(bundle / "capability.toml")
    timers.fire()

    snapshot = controller.snapshot()
    with pytest.raises(OrganelleInputError):
        snapshot.index.describe("broken")
    diagnostics = snapshot.index.diagnostics()
    assert any(
        any(bundle.as_posix() in origin.source_path for origin in item.origins)
        for item in diagnostics
    ), f"no diagnostic recorded for the broken bundle: {diagnostics!r}"


def test_watch_failure_degrades_to_manual_refresh_with_explicit_diagnostic(
    tmp_path: Path,
) -> None:
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    timers = _ManualTimers()
    watcher = _FakeWatcher(
        failure=OrganelleError(
            code="capability.watch_unavailable",
            message="filesystem watching is unavailable: inotify handles exhausted",
            details={},
        )
    )
    store = VerificationStore(tmp_path / "verifications")
    controller = _controller(
        paths=(plugins,), timers=timers, watcher=watcher, store=store
    )

    report = controller.start()

    assert report.watch_state is WatchState.DEGRADED
    assert report.watch_diagnostic is not None
    assert report.watch_diagnostic.code == "capability.watch_unavailable"
    status = controller.status()
    assert status.watch_state is WatchState.DEGRADED
    assert status.watch_diagnostic is not None

    # Degradation is not silent failure: manual refresh still works.
    _write_bundle(plugins, "thirdparty.demo")
    controller.rescan()
    entry = controller.snapshot().index.describe("thirdparty.demo")
    assert entry.status is CapabilityStatus.REJECTED


def test_rescan_is_incremental_across_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import organelleverse.capabilities.hotplug as hotplug

    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    _write_bundle(root_a, "thirdparty.alpha")
    _write_bundle(root_b, "thirdparty.beta")
    timers = _ManualTimers()
    watcher = _FakeWatcher()
    store = VerificationStore(tmp_path / "verifications")
    controller = _controller(
        paths=(root_a, root_b), timers=timers, watcher=watcher, store=store
    )
    scanned: list[tuple[Path, ...]] = []
    real_scan = hotplug.scan_search_roots

    def spy(roots):
        scanned.append(tuple(root.root for root in roots))
        return real_scan(roots)

    monkeypatch.setattr(hotplug, "scan_search_roots", spy)

    controller.start()
    assert sorted(call[0] for call in scanned) == sorted([root_a, root_b])

    scanned.clear()
    (root_a / "thirdparty-alpha" / "capability.toml").write_text(
        _bundle_text("thirdparty.alpha", version="1.0.1"), encoding="utf-8"
    )
    watcher.emit(root_a / "thirdparty-alpha" / "capability.toml")
    timers.fire()

    assert scanned == [(root_a,)], f"rescan must be incremental, rescanned: {scanned!r}"
    snapshot = controller.snapshot()
    assert snapshot.index.describe("thirdparty.beta").capability_id == "thirdparty.beta"


def test_root_scan_error_fails_closed_without_partial_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import organelleverse.capabilities.hotplug as hotplug

    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    _write_bundle(root_a, "thirdparty.alpha")
    _write_bundle(root_b, "thirdparty.beta")
    timers = _ManualTimers()
    watcher = _FakeWatcher()
    store = VerificationStore(tmp_path / "verifications")
    controller = _controller(
        paths=(root_a, root_b), timers=timers, watcher=watcher, store=store
    )
    controller.start()
    assert controller.snapshot().index.describe("thirdparty.alpha").capability_id

    real_scan = hotplug.scan_search_roots

    def failing(roots):
        if any(root.root == root_a for root in roots):
            raise OSError("simulated watch-root enumeration failure")
        return real_scan(roots)

    monkeypatch.setattr(hotplug, "scan_search_roots", failing)
    watcher.emit(root_a / "thirdparty-alpha" / "capability.toml")
    timers.fire()

    snapshot = controller.snapshot()
    # Fail closed: a root that cannot be rescanned goes offline with a
    # diagnostic; the other root is unaffected.
    with pytest.raises(OrganelleInputError):
        snapshot.index.describe("thirdparty.alpha")
    assert snapshot.index.describe("thirdparty.beta").capability_id == "thirdparty.beta"
    assert any(
        item.code == "capability.rescan_failed" for item in snapshot.index.diagnostics()
    )
