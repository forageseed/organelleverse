"""Canonical-contract helpers private to the pangenome suite.

The pangenome operations are not registered v1 operations yet, so they build
canonical :class:`~organelleverse.core.result.OrganelleResult` values directly
instead of going through the operations registry. These helpers keep the frozen
contract bookkeeping — parameters hash, package version, git commit, artifact
references — in one place instead of repeating it in every entry point.

Nothing here performs scientific computation; this module is purely the I/O
boundary between the suite and :mod:`organelleverse.core`.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from ..core.artifacts import ArtifactRef
from ..core.genome import OrganelleGenome
from ..core.provenance import ResultProvenance
from ..core.result import Finding, ResultScope

#: Contract version reported by every pangenome result. The suite is pre-release
#: (no entry in the released operations catalog), so it stays below 1.0.
OPERATION_VERSION = "0.1"

__all__ = [
    "OPERATION_VERSION",
    "annotation_path",
    "artifact_from_path",
    "finding",
    "input_artifact_hashes",
    "input_object_ids",
    "make_provenance",
    "package_version",
    "parameters_hash",
    "result_scope",
    "sequence_path",
    "utc_now",
]


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def parameters_hash(parameters: Mapping[str, Any]) -> str:
    """Return the SHA256 of a canonical JSON encoding of ``parameters``."""
    payload = json.dumps(
        dict(parameters),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def package_version() -> str:
    """Return the installed package version, or the pre-release default."""
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def make_provenance(
    *,
    operation_id: str,
    parameters: Mapping[str, Any],
    input_object_ids: tuple[str, ...] = (),
    input_artifact_hashes: tuple[str, ...] = (),
    requested_backend: str = "",
    actual_backend: str = "",
    attempted_backends: tuple[str, ...] = (),
    argv: tuple[str, ...] = (),
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> ResultProvenance:
    """Build the canonical provenance record for one pangenome operation."""
    duration = (
        max(0.0, (finished_at - started_at).total_seconds())
        if started_at is not None and finished_at is not None
        else None
    )
    return ResultProvenance(
        operation_id=operation_id,
        operation_version=OPERATION_VERSION,
        package_version=package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        input_object_ids=input_object_ids,
        input_artifact_hashes=input_artifact_hashes,
        parameters_hash=parameters_hash(parameters),
        requested_backend=requested_backend,
        actual_backend=actual_backend,
        attempted_backends=attempted_backends,
        argv=argv,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration,
    )


def finding(
    code: str,
    value: int | float | str | bool | None,
    *,
    metric: str = "",
    unit: str = "",
) -> Finding:
    """Build one canonical scalar finding.

    Canonical findings only carry scalar values, so list-valued observations
    (such as an argv vector) belong in provenance or metrics, not here.
    """
    return Finding(code=code, metric=metric or code, value=value, unit=unit)


def artifact_from_path(
    path: str | Path,
    *,
    kind: str,
    format: str,
    media_type: str = "application/octet-stream",
) -> ArtifactRef | None:
    """Reference an existing output file, or return ``None`` when absent.

    Canonical artifacts are content-addressed files; a directory (as produced by
    ``pggb``) cannot be referenced and yields ``None``.
    """
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return ArtifactRef.from_path(candidate, kind=kind, format=format, media_type=media_type)


def sequence_path(genome: OrganelleGenome) -> Path | None:
    """Resolve the FASTA path behind a canonical genome's sequence artifact."""
    return genome.sequence.resolve() if genome.sequence is not None else None


def annotation_path(genome: OrganelleGenome) -> Path | None:
    """Resolve the annotation path behind a canonical genome's artifact."""
    return genome.annotation.resolve() if genome.annotation is not None else None


def result_scope(genomes: list[OrganelleGenome]) -> ResultScope:
    """Derive the canonical result scope from a genome list.

    A single organelle type maps straight through; a mixed list is ``mixed`` and
    an empty list is ``none``.
    """
    organelles = {genome.organelle for genome in genomes}
    if not organelles:
        return "none"
    if len(organelles) == 1:
        return cast(ResultScope, next(iter(organelles)))
    return "mixed"


def input_object_ids(genomes: list[OrganelleGenome]) -> tuple[str, ...]:
    """Return the content identity of every input genome."""
    return tuple(genome.object_id for genome in genomes)


def input_artifact_hashes(genomes: list[OrganelleGenome]) -> tuple[str, ...]:
    """Return the content hashes of every sequence/annotation artifact consumed."""
    hashes: list[str] = []
    for genome in genomes:
        if genome.sequence is not None:
            hashes.append(genome.sequence.sha256)
        if genome.annotation is not None:
            hashes.append(genome.annotation.sha256)
    return tuple(hashes)
