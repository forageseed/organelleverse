"""Governed hot-plug lifecycle for capability bundles (TASK-B backend).

Hot-plug is never a bypass of the admission gate. Every newly appearing or
modified bundle goes through the complete ``discover → verify → trust → admit``
pipeline on its new content identity; this module only adds the trigger and
the lifecycle management on top of the existing, unchanged gates:

- watches exactly the declared search roots (:func:`.discovery.select_search_roots`),
  never anything wider;
- debounces editor save bursts into one logical rescan;
- rescans incrementally — only roots that saw filesystem events are re-parsed
  and re-hashed, every other root's cached scan is reused, and the global
  dedup/admission rules then run over the merged raw entries;
- serves the result as one atomic, immutable :class:`CapabilityIndex` snapshot
  swap, so readers never observe a half-rebuilt registry;
- fails closed: a root that cannot be rescanned drops its bundles from the
  served snapshot with a ``capability.rescan_failed`` diagnostic instead of
  serving stale admissions, and an unavailable watcher degrades the controller
  to manual refresh with an explicit ``capability.watch_unavailable``
  diagnostic rather than silently not working.

Trust and verification invalidation on modification need no machinery here by
design: both the trust store and the verification store are content-addressed,
so a changed bundle hash is automatically a new, untrusted, unverified
identity. What this module guarantees is that no status is ever carried over
between snapshots — each snapshot's admission is computed from the stores as
they are at rescan time.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import Field

from organelleverse.core.errors import OrganelleContractError, OrganelleError
from organelleverse.core.frozen import thaw_json
from organelleverse.operations.spec import StrictSpecModel

from .admission import admit_capabilities
from .discovery import (
    EntryPointLike,
    SearchRoot,
    deduplicate_entries,
    scan_search_roots,
    select_search_roots,
)
from .index import CapabilityDiagnostic, CapabilityEntry, CapabilityIndex, CapabilityStatus

if TYPE_CHECKING:
    from organelleverse.operations.python_binding import EnvironmentParameterProvider

    from .index import IndexBindingSource
    from .trust import TrustStore
    from .verification import VerificationStore
    from .worker import BundleWorkerExecutor

_HASH_PATTERN = r"^sha256:[0-9a-f]{64}$"

_RootKey = tuple[str, str]


def _error_diagnostic(error: OrganelleError) -> CapabilityDiagnostic:
    return CapabilityDiagnostic.model_validate(
        {
            "code": error.code,
            "message": error.message,
            "details": thaw_json(error.details),
        }
    )


class WatchState(StrEnum):
    """How the controller currently learns about filesystem changes."""

    DISABLED = "disabled"
    WATCHING = "watching"
    DEGRADED = "degraded"


class CapabilityChangeKind(StrEnum):
    """The lifecycle transition one capability ID went through in a rescan."""

    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"
    STATUS_CHANGED = "status_changed"


class CapabilityChange(StrictSpecModel):
    """One capability ID's transition between two consecutive snapshots."""

    capability_id: str = Field(min_length=1)
    kind: CapabilityChangeKind
    status: CapabilityStatus | None = None
    previous_content_hash: str | None = Field(default=None, pattern=_HASH_PATTERN)
    current_content_hash: str | None = Field(default=None, pattern=_HASH_PATTERN)
    diagnostic: CapabilityDiagnostic | None = None


class HotPlugReport(StrictSpecModel):
    """What one applied rescan changed, delivered to listeners and return values."""

    revision: int = Field(ge=1)
    changes: tuple[CapabilityChange, ...] = ()
    watch_state: WatchState
    watch_diagnostic: CapabilityDiagnostic | None = None


class HotPlugStatus(StrictSpecModel):
    """Point-in-time controller state for status surfaces (never a transient toast)."""

    revision: int = Field(ge=0)
    watch_state: WatchState
    watch_diagnostic: CapabilityDiagnostic | None = None


class HotPlugSnapshot:
    """One atomic, immutable registry view swapped in by each applied rescan.

    Holding on to a snapshot preserves the registry view that existed at that
    revision.  It does *not* preserve the bundle files themselves: true
    in-flight execution across modification or deletion still requires a
    content-addressed immutable bundle snapshot at the worker boundary.
    """

    __slots__ = ("index", "revision")

    def __init__(self, revision: int, index: CapabilityIndex) -> None:
        self.revision = revision
        self.index = index

    def binding_source(
        self,
        *,
        trust_store: TrustStore | None = None,
        environment_provider: EnvironmentParameterProvider | None = None,
        executor: BundleWorkerExecutor | None = None,
    ) -> IndexBindingSource:
        """Create the stateful lazy-binding adapter consumed by OperationRegistry."""
        return self.index.binding_source(
            trust_store=trust_store,
            environment_provider=environment_provider,
            executor=executor,
        )


class TimerHandle(Protocol):
    def cancel(self) -> None: ...


TimerFactory = Callable[[float, Callable[[], None]], TimerHandle]


class RootWatcher(Protocol):
    """Filesystem event source restricted to the declared search roots."""

    def start(self, roots: tuple[Path, ...], on_event: Callable[[Path], None]) -> None: ...

    def sync(self, roots: tuple[Path, ...]) -> None: ...

    def stop(self) -> None: ...


def _threading_timer(delay: float, callback: Callable[[], None]) -> TimerHandle:
    timer = threading.Timer(delay, callback)
    timer.start()
    return timer


class WatchdogWatcher:
    """``watchdog``-backed watcher restricted to the declared search roots.

    A root that does not exist yet is observed through its nearest existing
    ancestor (non-recursive) so that creating the root directory itself is
    still seen; the recursive root watch replaces the ancestor watch on the
    next :meth:`sync`. ``watchdog`` is imported lazily: its absence, or any
    watch-setup failure (e.g. inotify handle exhaustion), raises
    ``capability.watch_unavailable`` so the controller can degrade to manual
    refresh loudly instead of silently not working.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # ``watchdog`` is an optional (desktop) dependency imported lazily in
        # start(); the observer, handler, and watch handles stay ``Any`` here.
        self._observer: Any = None
        self._handler: Any = None
        self._on_event: Callable[[Path], None] | None = None
        self._watches: dict[Path, tuple[Any, bool]] = {}

    def start(self, roots: tuple[Path, ...], on_event: Callable[[Path], None]) -> None:
        try:
            from watchdog.events import (  # pyright: ignore[reportMissingImports]
                FileSystemEventHandler,  # pyright: ignore[reportUnknownVariableType]
            )
            from watchdog.observers import (  # pyright: ignore[reportMissingImports]
                Observer,  # pyright: ignore[reportUnknownVariableType]
            )
        except ImportError as error:
            raise OrganelleError(
                code="capability.watch_unavailable",
                message=(
                    "filesystem watching requires the 'watchdog' package, which is not "
                    "installed; hot-plug is degraded to manual refresh"
                ),
                details={"reason": str(error)},
            ) from error
        owner = self

        class _Handler(FileSystemEventHandler):  # pyright: ignore[reportUntypedBaseClass]
            def on_any_event(self, event: Any) -> None:
                callback = owner._on_event
                if callback is None:
                    return
                callback(Path(event.src_path))
                dest_path = getattr(event, "dest_path", None)
                if dest_path:
                    callback(Path(dest_path))

        observer: Any = Observer()  # pyright: ignore[reportUnknownVariableType]
        try:
            observer.start()  # pyright: ignore[reportUnknownMemberType]
        except OSError as error:
            raise OrganelleError(
                code="capability.watch_unavailable",
                message=f"filesystem watcher failed to start: {error}",
                details={"reason": str(error)},
            ) from error
        handler = _Handler()
        with self._lock:
            self._observer = observer
            self._handler = handler
            self._on_event = on_event
        try:
            self.sync(roots)
        except OrganelleError:
            with self._lock:
                self._observer = None
                self._handler = None
                self._on_event = None
                self._watches.clear()
            observer.stop()  # pyright: ignore[reportUnknownMemberType]
            observer.join(timeout=5)  # pyright: ignore[reportUnknownMemberType]
            raise

    def sync(self, roots: tuple[Path, ...]) -> None:
        with self._lock:
            observer = self._observer
            handler = self._handler
        if observer is None or handler is None:
            return
        desired: dict[Path, bool] = {}
        for root in roots:
            anchor = Path(root).resolve(strict=False)
            recursive = anchor.is_dir()
            while not anchor.is_dir() and anchor != anchor.parent:
                anchor = anchor.parent
            if not anchor.is_dir():
                continue
            desired[anchor] = desired.get(anchor, False) or recursive
        with self._lock:
            if self._observer is not observer:
                return
            for path, (watch, _recursive) in list(self._watches.items()):
                if path not in desired:
                    observer.unschedule(watch)
                    del self._watches[path]
            for path, recursive in desired.items():
                existing = self._watches.get(path)
                if existing is not None and existing[1] == recursive:
                    continue
                if existing is not None:
                    observer.unschedule(existing[0])
                    del self._watches[path]
                try:
                    watch = observer.schedule(handler, str(path), recursive=recursive)
                except OSError as error:
                    raise OrganelleError(
                        code="capability.watch_unavailable",
                        message=f"filesystem watch could not be installed: {error}",
                        details={"path": str(path), "reason": str(error)},
                    ) from error
                self._watches[path] = (watch, recursive)

    def stop(self) -> None:
        with self._lock:
            observer = self._observer
            self._observer = None
            self._handler = None
            self._on_event = None
            self._watches.clear()
        if observer is not None:
            observer.stop()
            observer.join(timeout=5)


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _diff(
    previous: CapabilityIndex | None, current: CapabilityIndex
) -> tuple[CapabilityChange, ...]:
    before = {entry.capability_id: entry for entry in previous.entries} if previous else {}
    after = {entry.capability_id: entry for entry in current.entries}
    changes: list[CapabilityChange] = []
    for capability_id in sorted(set(before) | set(after)):
        old = before.get(capability_id)
        new = after.get(capability_id)
        if old is None and new is not None:
            changes.append(
                CapabilityChange(
                    capability_id=capability_id,
                    kind=CapabilityChangeKind.ADDED,
                    status=new.status,
                    current_content_hash=new.content_hash,
                    diagnostic=new.diagnostic,
                )
            )
        elif new is None and old is not None:
            changes.append(
                CapabilityChange(
                    capability_id=capability_id,
                    kind=CapabilityChangeKind.REMOVED,
                    previous_content_hash=old.content_hash,
                )
            )
        elif old is not None and new is not None:
            if old.content_hash != new.content_hash:
                changes.append(
                    CapabilityChange(
                        capability_id=capability_id,
                        kind=CapabilityChangeKind.MODIFIED,
                        status=new.status,
                        previous_content_hash=old.content_hash,
                        current_content_hash=new.content_hash,
                        diagnostic=new.diagnostic,
                    )
                )
            elif old.status != new.status:
                changes.append(
                    CapabilityChange(
                        capability_id=capability_id,
                        kind=CapabilityChangeKind.STATUS_CHANGED,
                        status=new.status,
                        previous_content_hash=old.content_hash,
                        current_content_hash=new.content_hash,
                        diagnostic=new.diagnostic,
                    )
                )
    return tuple(changes)


class PluginHotPlugController:
    """Owns the watch → debounce → incremental rescan → atomic snapshot lifecycle.

    The controller never verifies, trusts, or admits anything on its own: it
    re-runs the existing gates over changed roots and serves the outcome.
    Listeners receive one :class:`HotPlugReport` per applied rescan; they run
    on the calling thread (the debounce timer thread for filesystem-driven
    rescans) and must not raise.
    """

    def __init__(
        self,
        *,
        paths: Iterable[Path] | None = None,
        extra_paths: Iterable[Path] | None = None,
        entry_points: Iterable[EntryPointLike] | None = None,
        debounce_seconds: float = 0.3,
        timer_factory: TimerFactory | None = None,
        watcher: RootWatcher | None = None,
        watch: bool = True,
        verification_store: VerificationStore | None = None,
    ) -> None:
        if debounce_seconds <= 0:
            raise ValueError(f"debounce_seconds must be positive: {debounce_seconds!r}")
        roots, selection_diagnostics = select_search_roots(
            paths=paths, extra_paths=extra_paths, entry_points=entry_points
        )
        ordered: list[SearchRoot] = []
        keys: list[_RootKey] = []
        seen: set[_RootKey] = set()
        for root in roots:
            key = (root.channel, str(root.root.resolve(strict=False)))
            if key in seen:
                continue
            seen.add(key)
            ordered.append(root)
            keys.append(key)
        self._roots = tuple(ordered)
        self._root_keys = tuple(keys)
        self._root_by_key = dict(zip(self._root_keys, self._roots, strict=True))
        self._selection_diagnostics = selection_diagnostics
        self._debounce_seconds = float(debounce_seconds)
        self._timer_factory: TimerFactory = timer_factory or _threading_timer
        self._watcher = watcher
        self._watch_requested = watch
        self._verification_store = verification_store
        self._lock = threading.RLock()
        self._scan_cache: dict[
            _RootKey, tuple[tuple[CapabilityEntry, ...], tuple[CapabilityDiagnostic, ...]]
        ] = {}
        self._snapshot: HotPlugSnapshot | None = None
        self._last_report: HotPlugReport | None = None
        self._pending: set[_RootKey] = set()
        self._timer: TimerHandle | None = None
        self._listeners: list[Callable[[HotPlugReport], None]] = []
        self._missing: set[_RootKey] = set()
        self._watch_state = WatchState.DISABLED
        self._watch_diagnostic: CapabilityDiagnostic | None = None

    def start(self) -> HotPlugReport:
        """Take the initial full snapshot, then start watching the declared roots."""
        with self._lock:
            if self._snapshot is not None:
                raise OrganelleContractError(
                    code="capability.hotplug_already_started",
                    message="this hot-plug controller has already been started",
                )
        watch_state = WatchState.DISABLED
        watch_diagnostic: CapabilityDiagnostic | None = None
        if self._watch_requested:
            watcher = self._watcher if self._watcher is not None else WatchdogWatcher()
            try:
                watcher.start(self._watch_paths(), self._on_fs_event)
                watch_state = WatchState.WATCHING
            except OrganelleError as error:
                # Documented degradation (TASK-B T-B1): manual refresh keeps
                # working and the state is explicit, never a silent no-op.
                watch_state = WatchState.DEGRADED
                watch_diagnostic = _error_diagnostic(error)
            self._watcher = watcher
        with self._lock:
            self._watch_state = watch_state
            self._watch_diagnostic = watch_diagnostic
            report = self._apply_rescan(self._root_keys)
        self._notify(report)
        return report

    def stop(self) -> None:
        """Cancel any pending debounce and stop the watcher; snapshots stay readable."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._pending.clear()
            watcher = self._watcher
        if watcher is not None:
            watcher.stop()

    def rescan(self) -> HotPlugReport:
        """Manual full refresh — the supported path when watching is degraded."""
        with self._lock:
            if self._snapshot is None:
                raise OrganelleContractError(
                    code="capability.hotplug_not_started",
                    message="hot-plug controller must be started before rescan",
                )
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._pending.clear()
            report = self._apply_rescan(self._root_keys)
        self._sync_watcher()
        self._notify(report)
        return report

    def snapshot(self) -> HotPlugSnapshot:
        """Return the current atomic registry view."""
        with self._lock:
            if self._snapshot is None:
                raise OrganelleContractError(
                    code="capability.hotplug_not_started",
                    message="hot-plug controller must be started before reading snapshots",
                )
            return self._snapshot

    @property
    def last_report(self) -> HotPlugReport:
        with self._lock:
            if self._last_report is None:
                raise OrganelleContractError(
                    code="capability.hotplug_not_started",
                    message="hot-plug controller must be started before reading reports",
                )
            return self._last_report

    def status(self) -> HotPlugStatus:
        with self._lock:
            return HotPlugStatus(
                revision=0 if self._snapshot is None else self._snapshot.revision,
                watch_state=self._watch_state,
                watch_diagnostic=self._watch_diagnostic,
            )

    def add_listener(self, listener: Callable[[HotPlugReport], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def _watch_paths(self) -> tuple[Path, ...]:
        return tuple(root.root.resolve(strict=False) for root in self._roots)

    def _sync_watcher(self) -> None:
        with self._lock:
            watcher = self._watcher
            watching = self._watch_state is WatchState.WATCHING
        if watcher is None or not watching:
            return
        try:
            watcher.sync(self._watch_paths())
        except OrganelleError as error:
            with self._lock:
                self._watch_state = WatchState.DEGRADED
                self._watch_diagnostic = _error_diagnostic(error)

    def _on_fs_event(self, path: Path) -> None:
        resolved = Path(path).resolve(strict=False)
        with self._lock:
            if self._snapshot is None:
                # Events between watcher start and the initial snapshot are
                # covered by that snapshot's full scan.
                return
            targets = {
                key for key in self._root_keys if _within(resolved, Path(key[1]))
            }
            if not targets:
                targets = {
                    key for key in self._missing if self._root_by_key[key].root.exists()
                }
            if not targets:
                return
            self._pending |= targets
            if self._timer is not None:
                self._timer.cancel()
            self._timer = self._timer_factory(self._debounce_seconds, self._debounce_elapsed)

    def _debounce_elapsed(self) -> None:
        with self._lock:
            if self._snapshot is None:
                return
            keys = tuple(key for key in self._root_keys if key in self._pending)
            self._pending.clear()
            self._timer = None
            if not keys:
                return
            report = self._apply_rescan(keys)
        self._sync_watcher()
        self._notify(report)

    def _apply_rescan(self, keys: tuple[_RootKey, ...]) -> HotPlugReport:
        """Rescan exactly *keys*, rebuild one index, and swap it in atomically.

        Callers must hold ``self._lock``. The new index is fully constructed
        before the snapshot reference is replaced, so readers never observe a
        half-built registry.
        """
        for key in keys:
            root = self._root_by_key[key]
            try:
                entries, diagnostics = scan_search_roots((root,))
                self._scan_cache[key] = (tuple(entries), tuple(diagnostics))
            except OSError as error:
                # Fail closed: a root whose current contents are unknown
                # contributes nothing to the served snapshot.
                self._scan_cache[key] = (
                    (),
                    (
                        CapabilityDiagnostic.model_validate(
                            {
                                "code": "capability.rescan_failed",
                                "message": (
                                    "capability search root could not be rescanned; its "
                                    f"bundles were taken offline: {root.root}"
                                ),
                                "details": {
                                    "search_root": str(root.root),
                                    "reason": str(error),
                                },
                            }
                        ),
                    ),
                )
        raw = [
            entry
            for key in self._root_keys
            for entry in self._scan_cache.get(key, ((), ()))[0]
        ]
        scan_diagnostics = [
            diagnostic
            for key in self._root_keys
            for diagnostic in self._scan_cache.get(key, ((), ()))[1]
        ]
        entries, conflicts, conflict_diagnostics = deduplicate_entries(raw)
        index = CapabilityIndex(
            entries=entries,
            diagnostic_items=tuple(
                (*self._selection_diagnostics, *scan_diagnostics, *conflict_diagnostics)
            ),
            conflict_items=conflicts,
        )
        index = admit_capabilities(index, store=self._verification_store)
        previous = self._snapshot.index if self._snapshot is not None else None
        revision = 1 if self._snapshot is None else self._snapshot.revision + 1
        self._snapshot = HotPlugSnapshot(revision, index)
        self._missing = {
            key for key in self._root_keys if not self._root_by_key[key].root.exists()
        }
        report = HotPlugReport(
            revision=revision,
            changes=_diff(previous, index),
            watch_state=self._watch_state,
            watch_diagnostic=self._watch_diagnostic,
        )
        self._last_report = report
        return report

    def _notify(self, report: HotPlugReport) -> None:
        with self._lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            listener(report)


__all__ = [
    "CapabilityChange",
    "CapabilityChangeKind",
    "HotPlugReport",
    "HotPlugSnapshot",
    "HotPlugStatus",
    "PluginHotPlugController",
    "RootWatcher",
    "TimerFactory",
    "TimerHandle",
    "WatchState",
    "WatchdogWatcher",
]
