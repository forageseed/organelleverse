"""Strictly one-way .ipynb exchange projection."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import nbformat
import pytest

from organelleverse.core.errors import OrganelleContractError
from organelleverse.notebooks import (
    CellRevision,
    InlineOutput,
    KernelEnvironmentRefs,
    NotebookRevision,
)
from organelleverse.notebooks.ipynb import export_ipynb, import_ipynb

_CREATED = datetime(2026, 8, 14, tzinfo=UTC)


def _cell(
    cell_revision_id: str,
    *,
    language: str = "python",
    source: str = "print('hi')",
    status: str = "succeeded",
    artifacts: tuple[str, ...] = (),
) -> CellRevision:
    return CellRevision(
        cell_revision_id=cell_revision_id,
        notebook_id="nb-1",
        parent_cell_revision_id=None,
        language=language,  # type: ignore[arg-type]
        source=source,
        status=status,  # type: ignore[arg-type]
        execution=None,
        inline_outputs=(InlineOutput(media_type="text/plain", text="hi"),)
        if status == "succeeded"
        else (),
        artifact_output_ids=artifacts,
        evidence_links=(),
        created_at=_CREATED,
    )


def _notebook(cell_revisions: tuple[str, ...]) -> NotebookRevision:
    return NotebookRevision(
        notebook_revision_id="nr-1",
        notebook_id="nb-1",
        parent_notebook_revision_ids=(),
        cell_revisions=cell_revisions,
        kernel_environment_refs=KernelEnvironmentRefs(python="env-py-1", r="env-r-1"),
        milestone_name=None,
        created_at=_CREATED,
    )


def test_export_orders_cells_maps_kernels_and_projects_outputs() -> None:
    cells = {
        "cell-1": _cell("cell-1", source="print('one')"),
        "cell-2": _cell("cell-2", language="r", source="cat('two')", artifacts=("art-9",)),
    }

    document = export_ipynb(_notebook(("cell-1", "cell-2")), cells)

    assert [cell.source for cell in document.cells] == ["print('one')", "cat('two')"]
    assert document.cells[0]["metadata"]["organelleverse_language"] == "python"
    assert document.cells[1]["metadata"]["organelleverse_language"] == "r"
    assert document.cells[1]["metadata"]["organelleverse_artifacts"] == ["art-9"]
    outputs = document.cells[0].get("outputs", [])
    assert outputs and outputs[0].get("text") == "hi"
    dumped = nbformat.writes(document)
    assert "env-py-1" not in dumped


def test_import_round_trips_sources_as_draft_and_strips_outputs() -> None:
    exported = export_ipynb(
        _notebook(("cell-1",)),
        {"cell-1": _cell("cell-1", source="x = 41 + 1")},
    )

    revision, cells = import_ipynb(exported, notebook_id="nb-import")

    assert revision.notebook_id == "nb-import"
    assert revision.milestone_name is None
    assert len(revision.cell_revisions) == 1
    imported = cells[revision.cell_revisions[0]]
    assert imported.source == "x = 41 + 1"
    assert imported.language == "python"
    assert imported.status == "draft"
    assert imported.inline_outputs == ()
    assert imported.execution is None


def test_import_rejects_unsupported_kernelspec_languages() -> None:
    document = nbformat.v4.new_notebook(
        metadata={"kernelspec": {"display_name": "Julia", "language": "julia", "name": "julia-1.10"}},
        cells=[nbformat.v4.new_code_cell("sqrt(2.0)")],
    )

    with pytest.raises(OrganelleContractError) as info:
        import_ipynb(document, notebook_id="nb-import")
    assert info.value.code == "notebook.import_unsupported_language"


def test_import_maps_r_kernelspec(tmp_path: Path) -> None:
    document = nbformat.v4.new_notebook(
        metadata={"kernelspec": {"display_name": "R", "language": "R", "name": "ir"}},
        cells=[nbformat.v4.new_code_cell("cat('bonjour')")],
    )

    revision, cells = import_ipynb(document, notebook_id="nb-import")

    imported = cells[revision.cell_revisions[0]]
    assert imported.language == "r"
    assert imported.source == "cat('bonjour')"
