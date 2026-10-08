"""Canonical-contract helpers private to the barcode suite.

The barcode operations are not registered v1 operations yet, so they build
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
from typing import Any

from ..core.artifacts import ArtifactRef
from ..core.provenance import ResultProvenance
from ..core.result import Finding

#: Contract version reported by every barcode result. The suite is pre-release
#: (no entry in the released operations catalog), so it stays below 1.0.
OPERATION_VERSION = "0.1"

#: Barcode inputs are plain FASTA files rather than typed genomes, so the suite
#: cannot derive an organelle from its inputs. The pre-port code hard-coded the
#: legacy ``chloro`` organelle; the canonical spelling of that scope is
#: ``plastid``.
RESULT_SCOPE = "plastid"

__all__ = [
    "OPERATION_VERSION",
    "RESULT_SCOPE",
    "fasta_artifact",
    "finding",
    "input_hashes",
    "make_provenance",
    "package_version",
    "parameters_hash",
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
    input_artifact_hashes: tuple[str, ...] = (),
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> ResultProvenance:
    """Build the canonical provenance record for one barcode operation."""
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
        input_artifact_hashes=input_artifact_hashes,
        parameters_hash=parameters_hash(parameters),
        actual_backend="organelleverse",
        attempted_backends=("organelleverse",),
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

    Canonical findings only carry scalar values, so composite observations (a
    barcode window, a best-match pair) encode their identity in ``metric`` and
    their measurement in ``value``.
    """
    return Finding(code=code, metric=metric or code, value=value, unit=unit)


def fasta_artifact(path: str | Path, *, kind: str) -> ArtifactRef | None:
    """Content-address an input FASTA, or return ``None`` when it is absent."""
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return ArtifactRef.from_path(
        candidate,
        kind=kind,
        format="fasta",
        media_type="text/x-fasta",
    )


def input_hashes(*artifacts: ArtifactRef | None) -> tuple[str, ...]:
    """Collect the content hashes of the artifacts that could be resolved."""
    return tuple(artifact.sha256 for artifact in artifacts if artifact is not None)
