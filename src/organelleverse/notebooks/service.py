"""One facade over the notebook store and the kernel coordinator."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from organelleverse.notebooks.coordinator import (
    NO_DECLARATION,
    CellDeclaration,
    KernelCoordinator,
)
from organelleverse.notebooks.models import CellRevision, NotebookRevision
from organelleverse.notebooks.store import NotebookStore

__all__ = ["NotebookService"]


class NotebookService:
    """The single entry point used by Desktop and the Agent."""

    def __init__(
        self,
        root: Path,
        *,
        python_executable: str = "",
    ) -> None:
        root = Path(root)
        self.store = NotebookStore(root / "revisions")
        self.coordinator = KernelCoordinator(
            self.store, root / "artifacts", python_executable=python_executable
        )

    def submit(
        self, cell: CellRevision, declaration: CellDeclaration = NO_DECLARATION
    ) -> CellRevision:
        """Register and execute one submitted cell revision."""
        self.store.register_cell(cell)
        return self.coordinator.execute(cell, declaration)

    def recover(
        self,
        revision: NotebookRevision,
        cells: Mapping[str, CellRevision],
        declarations: Mapping[str, CellDeclaration],
    ) -> dict[str, CellRevision]:
        """Reproduce a selected revision by ordered replay in new kernels."""
        return self.coordinator.replay(revision, cells, declarations)
