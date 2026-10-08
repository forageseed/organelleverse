# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false
"""Strictly one-way ``.ipynb`` exchange projection.

The internal two-level revision store is authoritative. Export projects a
selected notebook revision into an nbformat document; import creates a new
draft branch. Outputs, execution identity, and evidence links do not survive
the projection — that loss is declared, not hidden.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from uuid import uuid4

import nbformat
from nbformat import NotebookNode

from organelleverse.core.errors import OrganelleContractError
from organelleverse.notebooks.models import (
    CellRevision,
    KernelEnvironmentRefs,
    NotebookRevision,
)

__all__ = ["export_ipynb", "import_ipynb"]

_KERNELSPEC_BY_LANGUAGE = {
    "python": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "r": {"display_name": "R", "language": "R", "name": "ir"},
}
_LANGUAGE_BY_KERNELSPEC_NAME = {
    "python3": "python",
    "ir": "r",
}


def _source_text(source: object) -> str:
    """nbformat normalizes multi-line sources into line lists; join them."""
    if isinstance(source, list):
        return "".join(str(line) for line in source)
    return str(source)


def _error(code: str, message: str, **details: object) -> OrganelleContractError:
    return OrganelleContractError(code=code, message=message, details=details)


def export_ipynb(
    revision: NotebookRevision, cells: Mapping[str, CellRevision]
) -> NotebookNode:
    """Project one notebook revision into an nbformat v4 document.

    Inline outputs project as stream outputs; artifact references project as
    cell metadata without inventing content. Environment identities stay
    server-side.
    """
    first_language = "python"
    for cell_id in revision.cell_revisions:
        cell = cells[cell_id]
        if cell.language in _KERNELSPEC_BY_LANGUAGE:
            first_language = cell.language
            break
    document = nbformat.v4.new_notebook(
        metadata={"kernelspec": dict(_KERNELSPEC_BY_LANGUAGE[first_language])}
    )
    for cell_id in revision.cell_revisions:
        cell = cells[cell_id]
        projected = nbformat.v4.new_code_cell(cell.source)
        projected["metadata"]["organelleverse_language"] = cell.language
        if cell.artifact_output_ids:
            projected["metadata"]["organelleverse_artifacts"] = list(
                cell.artifact_output_ids
            )
        outputs = [
            nbformat.v4.new_output(
                "stream",
                text=output.text,
                name="stdout",
            )
            for output in cell.inline_outputs
        ]
        if outputs:
            projected["outputs"] = outputs
        document.cells.append(projected)
    return document


def import_ipynb(
    document: NotebookNode, *, notebook_id: str
) -> tuple[NotebookRevision, dict[str, CellRevision]]:
    """Create a fresh draft branch from an nbformat document.

    Sources round-trip exactly; outputs and execution identity are stripped
    (declared lossy). Only the ``python3`` and ``ir`` kernelspecs resolve;
    anything else fails loudly instead of being coerced.
    """
    kernelspec = document.get("metadata", {}).get("kernelspec", {})
    kernel_name = kernelspec.get("name", "") if isinstance(kernelspec, dict) else ""
    default_language = _LANGUAGE_BY_KERNELSPEC_NAME.get(kernel_name)
    if default_language is None:
        raise _error(
            "notebook.import_unsupported_language",
            "only the python3 and ir kernelspecs import; other languages are refused",
            kernelspec_name=kernel_name,
        )

    created = datetime.now(UTC)
    imported_cells: dict[str, CellRevision] = {}
    ordered_ids: list[str] = []
    for cell in document.cells:
        if cell.get("cell_type") != "code":
            continue
        language = cell.get("metadata", {}).get("organelleverse_language")
        if language not in _KERNELSPEC_BY_LANGUAGE:
            language = default_language
        cell_revision = CellRevision(
            cell_revision_id=f"cell-{uuid4().hex}",
            notebook_id=notebook_id,
            parent_cell_revision_id=None,
            language=language,  # type: ignore[arg-type]
            source=_source_text(cell.get("source", "")),
            status="draft",
            execution=None,
            inline_outputs=(),
            artifact_output_ids=(),
            evidence_links=(),
            diagnostics=(),
            created_at=created,
        )
        imported_cells[cell_revision.cell_revision_id] = cell_revision
        ordered_ids.append(cell_revision.cell_revision_id)

    revision = NotebookRevision(
        notebook_revision_id=f"notebook-{uuid4().hex}",
        notebook_id=notebook_id,
        parent_notebook_revision_ids=(),
        cell_revisions=tuple(ordered_ids),
        kernel_environment_refs=KernelEnvironmentRefs(),
        milestone_name=None,
        created_at=created,
    )
    return revision, imported_cells
