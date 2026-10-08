"""Dependency-aware execution coordinator over isolated kernels."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleError
from organelleverse.notebooks.kernels import KernelSession
from organelleverse.notebooks.models import (
    CellExecutionRecord,
    CellRevision,
    InlineOutput,
    KernelEnvironmentRefs,
    NotebookRevision,
)
from organelleverse.notebooks.store import NotebookStore
from organelleverse.operations.spec import StrictSpecModel

__all__ = [
    "NO_DECLARATION",
    "ArtifactExchangeRecord",
    "CellDeclaration",
    "KernelCoordinator",
]

# pyright: reportUnknownMemberType=false

DECLARED_FORMATS = frozenset({"arrow", "parquet", "csv", "tsv", "json", "png", "svg"})
_INLINE_OUTPUT_LIMIT = 64 * 1024


class ArtifactExchangeRecord(StrictSpecModel):
    """One managed-artifact production or consumption event."""

    artifact_id: str
    producer_cell_revision_id: str  # "" on a consumption-only record
    consumer_cell_revision_id: str  # "" on a production-only record
    declared_format: str
    environment_refs: KernelEnvironmentRefs


@dataclass(frozen=True)
class CellDeclaration:
    """Declared managed-artifact reads and writes for one cell execution."""

    reads: tuple[str, ...] = ()
    writes: tuple[tuple[str, str], ...] = ()


NO_DECLARATION = CellDeclaration()

class _OutputSlot(NamedTuple):
    artifact_id: str
    declared_format: str
    path: Path
    producer_cell_revision_id: str = ""


def _error(code: str, message: str, **details: object) -> OrganelleError:
    from organelleverse.core.errors import OrganelleContractError

    return OrganelleContractError(code=code, message=message, details=details)


def _media_type(declared_format: str) -> str:
    return {
        "csv": "text/csv",
        "tsv": "text/tab-separated-values",
        "json": "application/json",
        "png": "image/png",
        "svg": "image/svg+xml",
        "arrow": "application/x-arrow",
        "parquet": "application/x-parquet",
    }.get(declared_format, "application/octet-stream")


class _ManagedArtifact(NamedTuple):
    artifact_id: str
    declared_format: str
    path: Path
    producer_cell_revision_id: str
    content_ref: ArtifactRef


class KernelCoordinator:
    """One execution path over one Python and one R kernel session.

    Cross-language exchange happens only through declared managed artifacts
    recorded in the exchange ledger. Kernel memory is never snapshotted;
    recovery of a selected revision is ordered re-execution in new kernels.
    """

    def __init__(
        self,
        store: NotebookStore,
        artifacts_root: Path,
        *,
        python_executable: str = "",
    ) -> None:
        self._store = store
        self._root = Path(artifacts_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._python_executable = python_executable
        self._sessions: dict[str, KernelSession] = {}
        self._artifacts: dict[str, _ManagedArtifact] = {}
        self._ledger: list[ArtifactExchangeRecord] = []
        self._lock = threading.RLock()
        self._cancel_requested = False
        self._running = False

    # -- sessions ----------------------------------------------------------

    def _session(self, language: str) -> KernelSession:
        with self._lock:
            session = self._sessions.get(language)
            if session is None:
                session = KernelSession(
                    "python" if language == "python" else "r",  # type: ignore[arg-type]
                    python_executable=self._python_executable,
                )
                self._sessions[language] = session
            return session

    def restart(self) -> None:
        """Drop both kernels; new ones start on demand."""
        with self._lock:
            for session in self._sessions.values():
                session.shutdown()
            self._sessions.clear()

    def _require_live_session(self, language: str) -> KernelSession:
        with self._lock:
            session = self._sessions.get(language)
            if session is not None and not session.alive():
                self._sessions.pop(language, None)
                session.shutdown()
                raise _error(
                    "coordinator.kernel_lost",
                    "the kernel process died before this execution; "
                    "the next execution starts a fresh kernel",
                    language=language,
                )
            return session if session is not None else self._session(language)

    # -- execution ---------------------------------------------------------

    def execute(
        self,
        cell: CellRevision,
        declaration: CellDeclaration = NO_DECLARATION,
    ) -> CellRevision:
        """Execute one submitted cell revision; return its derived revision.

        Outputs are recorded against exactly the submitted cell revision.
        Kernel failure, cancellation, and process loss produce durable
        terminal cell revisions with diagnostics — never a success.
        """
        started = datetime.now(UTC)
        self._ensure_registered(cell)
        with self._lock:
            self._cancel_requested = False
            self._running = True
        try:
            inputs = self._resolve_inputs(cell, declaration.reads)
            slots = self._create_outputs(cell, declaration.writes)
            session = self._require_live_session(cell.language)
            response = session.execute(
                cell.source,
                inputs={
                    artifact_id: str(artifact.path)
                    for artifact_id, artifact in inputs.items()
                },
                outputs={name: str(slot.path) for name, slot in slots.items()},
            )
        except OrganelleError as error:
            if self._cancel_requested:
                return self._record(
                    cell, started, status="cancelled", diagnostics=("cancelled by user",)
                )
            return self._record_failure(cell, started, (error.message,))
        finally:
            with self._lock:
                self._running = False
        if self._cancel_requested:
            return self._record(
                cell, started, status="cancelled", diagnostics=("cancelled by user",)
            )
        if not response.ok:
            return self._record_failure(cell, started, (response.error or "kernel error",))
        missing = [name for name, slot in slots.items() if not slot.path.is_file()]
        if missing:
            return self._record_failure(
                cell,
                started,
                tuple(f"declared output {name!r} was not written" for name in missing),
            )
        produced = {
            name: self._register_production(cell, slot)
            for name, slot in slots.items()
        }
        self._record_consumption(cell, inputs)
        return self._record(
            cell,
            started,
            status="succeeded",
            stdout=response.stdout,
            result=response.result,
            artifacts=tuple(
                artifact.artifact_id for artifact in produced.values()
            ),
        )

    # -- artifacts and ledger ----------------------------------------------

    def _resolve_inputs(
        self, cell: CellRevision, reads: tuple[str, ...]
    ) -> dict[str, _ManagedArtifact]:
        resolved: dict[str, _ManagedArtifact] = {}
        for artifact_id in reads:
            artifact = self._artifacts.get(artifact_id)
            if artifact is None:
                raise _error(
                    "coordinator.undeclared_artifact",
                    "this cell depends on an artifact id no prior cell produced",
                    artifact_id=artifact_id,
                    cell_revision_id=cell.cell_revision_id,
                )
            resolved[artifact_id] = artifact
        return resolved

    def _create_outputs(
        self, cell: CellRevision, writes: tuple[tuple[str, str], ...]
    ) -> dict[str, _OutputSlot]:
        slots: dict[str, _OutputSlot] = {}
        for name, declared_format in writes:
            if declared_format not in DECLARED_FORMATS:
                raise _error(
                    "coordinator.undeclared_format",
                    "the declared artifact format is not in the managed vocabulary",
                    name=name,
                    declared_format=declared_format,
                )
            artifact_id = f"art-{uuid4().hex}"
            slots[name] = _OutputSlot(
                artifact_id=artifact_id,
                declared_format=declared_format,
                path=self._root / artifact_id,
                producer_cell_revision_id=cell.cell_revision_id,
            )
        return slots

    def _register_production(
        self, cell: CellRevision, slot: _OutputSlot
    ) -> _ManagedArtifact:
        content_ref = ArtifactRef.from_path(
            slot.path,
            kind="notebook_artifact",
            format=slot.declared_format,
            media_type=_media_type(slot.declared_format),
        )
        artifact = _ManagedArtifact(
            artifact_id=slot.artifact_id,
            declared_format=slot.declared_format,
            path=slot.path,
            producer_cell_revision_id=cell.cell_revision_id,
            content_ref=content_ref,
        )
        with self._lock:
            self._artifacts[artifact.artifact_id] = artifact
            self._ledger.append(
                ArtifactExchangeRecord(
                    artifact_id=artifact.artifact_id,
                    producer_cell_revision_id=cell.cell_revision_id,
                    consumer_cell_revision_id="",
                    declared_format=artifact.declared_format,
                    environment_refs=KernelEnvironmentRefs(),
                )
            )
        return artifact

    def _record_consumption(
        self, cell: CellRevision, inputs: dict[str, _ManagedArtifact]
    ) -> None:
        with self._lock:
            for artifact in inputs.values():
                self._ledger.append(
                    ArtifactExchangeRecord(
                        artifact_id=artifact.artifact_id,
                        producer_cell_revision_id=artifact.producer_cell_revision_id,
                        consumer_cell_revision_id=cell.cell_revision_id,
                        declared_format=artifact.declared_format,
                        environment_refs=KernelEnvironmentRefs(),
                    )
                )

    def ledger(self) -> tuple[ArtifactExchangeRecord, ...]:
        with self._lock:
            return tuple(self._ledger)

    def artifact_ref(self, artifact_id: str) -> ArtifactRef:
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            raise _error(
                "coordinator.unknown_artifact",
                "no managed artifact with this id was produced",
                artifact_id=artifact_id,
            )
        return artifact.content_ref

    def _ensure_registered(self, cell: CellRevision) -> None:
        """A submitted cell must be durable before it can be executed."""
        try:
            self._store.get_cell(cell.cell_revision_id)
        except OrganelleError:
            self._store.register_cell(cell)

    # -- derived revision recording ----------------------------------------

    def _record_failure(
        self, cell: CellRevision, started: datetime, diagnostics: tuple[str, ...]
    ) -> CellRevision:
        return self._record(cell, started, status="failed", diagnostics=diagnostics)

    def _record(
        self,
        cell: CellRevision,
        started: datetime,
        *,
        status: str,
        diagnostics: tuple[str, ...] = (),
        stdout: str = "",
        result: str | None = None,
        artifacts: tuple[str, ...] = (),
    ) -> CellRevision:
        inline: list[InlineOutput] = []
        if stdout:
            inline.append(
                InlineOutput(media_type="text/plain", text=stdout[:_INLINE_OUTPUT_LIMIT])
            )
        if result:
            inline.append(
                InlineOutput(media_type="text/plain", text=result[:_INLINE_OUTPUT_LIMIT])
            )
        derived = CellRevision(
            cell_revision_id=f"cell-{uuid4().hex}",
            notebook_id=cell.notebook_id,
            parent_cell_revision_id=cell.cell_revision_id,
            language=cell.language,
            source=cell.source,
            status=status,  # type: ignore[arg-type]
            execution=CellExecutionRecord(
                kernel_session_id=f"session-{cell.language}",
                started_at=started,
                finished_at=datetime.now(UTC),
            ),
            inline_outputs=tuple(inline),
            artifact_output_ids=artifacts,
            evidence_links=(),
            diagnostics=diagnostics,
            created_at=datetime.now(UTC),
        )
        return self._store.register_cell(derived)

    # -- cancellation ------------------------------------------------------

    def cancel(self) -> None:
        """Request cancellation of the running execution and stop the kernels."""
        with self._lock:
            self._cancel_requested = True
            for session in self._sessions.values():
                session.shutdown()
            self._sessions.clear()

    @property
    def running(self) -> bool:
        return self._running

    # -- replay ------------------------------------------------------------

    def replay(
        self,
        revision: NotebookRevision,
        cells: Mapping[str, CellRevision],
        declarations: Mapping[str, CellDeclaration],
    ) -> dict[str, CellRevision]:
        """Ordered re-execution of a selected revision in fresh kernels."""
        with self._lock:
            self.restart()
        results: dict[str, CellRevision] = {}
        for cell_revision_id in revision.cell_revisions:
            cell = cells[cell_revision_id]
            derived = self.execute(
                cell, declarations.get(cell_revision_id, CellDeclaration())
            )
            results[cell_revision_id] = derived
            if derived.status != "succeeded":
                break
        return results
