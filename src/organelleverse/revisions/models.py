"""Closed vocabulary for common immutable artifact revisions."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, JsonValue

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.operations.spec import StrictSpecModel

RevisionKind = Literal["text", "markdown", "image", "table"]
RevisionStatus = Literal["available", "broken"]


class EvidenceLink(StrictSpecModel):
    """One durable evidence reference cited by a revision."""

    evidence_id: str = Field(min_length=1)
    kind: Literal["run", "artifact", "comment"]
    target_id: str = Field(min_length=1)
    label: str = ""


class ImageDimensions(StrictSpecModel):
    """Declared pixel dimensions for an image revision.

    Comments anchor to these in normalized coordinates; dimensions are
    declared at registration time, never guessed from bytes.
    """

    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ArtifactRevision(StrictSpecModel):
    """One immutable revision of a managed artifact.

    The wrapped ``content_ref`` is the only content identity: the existing
    managed-artifact sha256. There is no second hash, and no field of this
    model is ever mutated — corrections are new revisions.
    """

    revision_id: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    parent_revision_id: str | None = None
    kind: RevisionKind
    author: Literal["agent"] = "agent"
    content_ref: ArtifactRef
    status: RevisionStatus = "available"
    diagnostics: tuple[str, ...] = ()
    evidence_links: tuple[EvidenceLink, ...] = ()
    dimensions: ImageDimensions | None = None
    operation: str = ""
    """Name of the operation that produced this revision (Galaxy-style
    tool identity); empty for the uploaded root revision."""
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    """Parameters the operation ran with — the lineage records not just
    what happened, but the exact invocation that must be reproducible."""
    created_at: datetime
