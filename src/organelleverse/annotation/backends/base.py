"""Released annotation backend contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from organelleverse.core.genome import OrganelleGenome, OrganelleType

from ..execution import CommandEvidence
from ..models import AnnotationDocument

AnnotationStage = Literal["pcg", "trna", "rrna"]
AnnotationBackendName = Literal["auto", "mitochondrion", "plastome"]


@dataclass(frozen=True)
class AnnotationRequest:
    """Validated execution request for the released annotation service."""

    backend: AnnotationBackendName
    workspace: Path
    threads: int
    stages: tuple[AnnotationStage, ...]
    call_orfs: bool = False
    orf_min_aa: int = 30
    orf_circular: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "mitochondrion", "plastome"}:
            raise ValueError("backend must be 'auto', 'mitochondrion', or 'plastome'")
        if self.threads < 1:
            raise ValueError("threads must be positive")
        if (
            isinstance(self.orf_min_aa, bool)
            or not isinstance(self.orf_min_aa, int)
            or self.orf_min_aa < 1
        ):
            raise ValueError("orf_min_aa must be a positive integer")
        if not self.stages:
            raise ValueError("at least one annotation stage is required")
        if len(set(self.stages)) != len(self.stages):
            raise ValueError("annotation stages must be unique")
        if any(stage not in {"pcg", "trna", "rrna"} for stage in self.stages):
            raise ValueError("unsupported annotation stage")
        object.__setattr__(self, "workspace", Path(self.workspace))


@dataclass(frozen=True)
class BackendRun:
    """Normalized backend output before service-level validation and persistence."""

    document: AnnotationDocument
    commands: tuple[CommandEvidence, ...]
    software_versions: Mapping[str, str]
    database_hashes: Mapping[str, str]
    logs: tuple[Path, ...]


class AnnotationBackend(Protocol):
    """Runtime contract implemented only by released annotation backends."""

    @property
    def name(self) -> str:
        """Stable released backend name."""

        ...

    @property
    def organelle_types(self) -> tuple[OrganelleType, ...]:
        """Canonical organelle scopes supported by this backend."""

        ...

    def run(
        self,
        genome: OrganelleGenome,
        request: AnnotationRequest,
        scratch: Path,
    ) -> BackendRun:
        """Execute and return canonical, not yet service-persisted output."""

        ...
