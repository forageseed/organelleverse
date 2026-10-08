"""Atomic, append-only persistence for common artifact revisions."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from organelleverse.core.base import ContractValidationContext
from organelleverse.core.errors import OrganelleContractError

from .models import ArtifactRevision

__all__ = ["ArtifactRevisionStore"]


def fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ArtifactRevisionStore:
    """One canonical JSON file per revision beneath ``root``.

    The store is append-only by construction: it exposes ``register`` and
    read accessors only. There is no update or delete path, and reopening
    after a restart restores the full lineage.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._revisions: dict[str, ArtifactRevision] = {}
        for path in sorted(self._root.glob("*.json")):
            self._restore(self._read_revision(path))

    @staticmethod
    def _history_invalid(path: Path, error: Exception) -> OrganelleContractError:
        return OrganelleContractError(
            code="revision.history_invalid",
            message="a persisted artifact revision is not valid canonical JSON",
            details={"path": str(path), "reason": str(error)},
        )

    @classmethod
    def _read_revision(cls, path: Path) -> ArtifactRevision:
        try:
            return ArtifactRevision.model_validate_json(
                path.read_text(encoding="utf-8"),
                context=ContractValidationContext(allow_computed_object_id=True),
            )
        except (OSError, ValidationError, ValueError) as error:
            raise cls._history_invalid(path, error) from error

    def _restore(self, revision: ArtifactRevision) -> None:
        if revision.revision_id in self._revisions:
            raise OrganelleContractError(
                code="revision.history_invalid",
                message="two persisted revisions share one revision id",
                details={"revision_id": revision.revision_id},
            )
        self._revisions[revision.revision_id] = revision

    def _path(self, revision_id: str) -> Path:
        return self._root / f"{revision_id}.json"

    def _write(self, revision: ArtifactRevision) -> None:
        destination = self._path(revision.revision_id)
        temporary = self._root / f".{revision.revision_id}.{uuid4().hex}.partial"
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(revision.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            fsync_directory(self._root)
        finally:
            if temporary.exists():
                temporary.unlink()

    def register(self, revision: ArtifactRevision) -> ArtifactRevision:
        """Append one immutable revision; the model is stored as given."""
        with self._lock:
            if revision.revision_id in self._revisions:
                raise OrganelleContractError(
                    code="revision.duplicate_id",
                    message="a revision with this id already exists",
                    details={"revision_id": revision.revision_id},
                )
            parent = None
            if revision.parent_revision_id is not None:
                parent = self._revisions.get(revision.parent_revision_id)
                if parent is None:
                    raise OrganelleContractError(
                        code="revision.unknown_parent",
                        message="the parent revision does not exist",
                        details={
                            "revision_id": revision.revision_id,
                            "parent_revision_id": revision.parent_revision_id,
                        },
                    )
                if parent.artifact_id != revision.artifact_id:
                    raise OrganelleContractError(
                        code="revision.parent_mismatch",
                        message="the parent revision belongs to another artifact",
                        details={
                            "revision_id": revision.revision_id,
                            "parent_artifact_id": parent.artifact_id,
                            "artifact_id": revision.artifact_id,
                        },
                    )
            self._write(revision)
            self._revisions[revision.revision_id] = revision
            return revision

    def list(self) -> tuple[ArtifactRevision, ...]:
        """Every revision in registration order (index position)."""
        with self._lock:
            return tuple(self._revisions.values())

    def get(self, revision_id: str) -> ArtifactRevision:
        try:
            return self._revisions[revision_id]
        except KeyError:
            raise OrganelleContractError(
                code="revision.unknown_revision",
                message="no artifact revision exists with this id",
                details={"revision_id": revision_id},
            ) from None

    def lineage(self, artifact_id: str) -> tuple[ArtifactRevision, ...]:
        """All revisions of one artifact, oldest first."""
        revisions = [r for r in self._revisions.values() if r.artifact_id == artifact_id]
        return tuple(sorted(revisions, key=lambda r: (r.created_at, r.revision_id)))

    def common_ancestor(
        self, revision_id_a: str, revision_id_b: str
    ) -> ArtifactRevision:
        """Nearest revision reachable from both inputs, inclusive of each."""
        chain_a = self._ancestor_chain(revision_id_a)
        reachable_from_b = {r.revision_id for r in self._ancestor_chain(revision_id_b)}
        for candidate in chain_a:
            if candidate.revision_id in reachable_from_b:
                return candidate
        raise OrganelleContractError(
            code="revision.no_common_ancestor",
            message="the two revisions share no common ancestor",
            details={
                "revision_id_a": revision_id_a,
                "revision_id_b": revision_id_b,
            },
        )

    def _ancestor_chain(self, revision_id: str) -> list[ArtifactRevision]:
        chain: list[ArtifactRevision] = []
        seen: set[str] = set()
        current: ArtifactRevision | None = self.get(revision_id)
        while current is not None:
            if current.revision_id in seen:
                raise OrganelleContractError(
                    code="revision.history_invalid",
                    message="the persisted parent chain contains a cycle",
                    details={"revision_id": current.revision_id},
                )
            seen.add(current.revision_id)
            chain.append(current)
            current = (
                self._revisions.get(current.parent_revision_id)
                if current.parent_revision_id is not None
                else None
            )
        return chain
