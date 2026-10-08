"""Closed vocabulary for versioned notebooks and live kernels."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from organelleverse.operations.spec import StrictSpecModel
from organelleverse.revisions import EvidenceLink

CellLanguage = Literal["python", "r"]
CellStatus = Literal["draft", "queued", "running", "succeeded", "failed", "cancelled"]


class InlineOutput(StrictSpecModel):
    """One small display output stored inline on the cell revision."""

    media_type: str = Field(min_length=1)
    text: str


class CellExecutionRecord(StrictSpecModel):
    """Execution identity for one submitted cell revision."""

    kernel_session_id: str = Field(min_length=1)
    started_at: datetime
    finished_at: datetime | None = None


class KernelEnvironmentRefs(StrictSpecModel):
    """Environment identities the notebook revision was produced against."""

    python: str = ""
    r: str = ""


class CellRevision(StrictSpecModel):
    """One immutable cell revision: source, outputs, execution, status."""

    cell_revision_id: str = Field(min_length=1)
    notebook_id: str = Field(min_length=1)
    parent_cell_revision_id: str | None = None
    language: CellLanguage
    source: str
    status: CellStatus = "draft"
    execution: CellExecutionRecord | None = None
    inline_outputs: tuple[InlineOutput, ...] = ()
    artifact_output_ids: tuple[str, ...] = ()
    evidence_links: tuple[EvidenceLink, ...] = ()
    diagnostics: tuple[str, ...] = ()
    created_at: datetime


class NotebookRevision(StrictSpecModel):
    """One immutable notebook revision: an ordered cell-revision selection."""

    notebook_revision_id: str = Field(min_length=1)
    notebook_id: str = Field(min_length=1)
    parent_notebook_revision_ids: tuple[str, ...] = ()
    cell_revisions: tuple[str, ...] = ()
    kernel_environment_refs: KernelEnvironmentRefs
    milestone_name: str | None = None
    created_at: datetime
