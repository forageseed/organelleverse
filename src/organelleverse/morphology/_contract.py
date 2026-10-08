"""Canonical v1 contract construction helpers for the morphology suite.

The morphology bridges (:mod:`.orgseg`, :mod:`.train`) return the canonical
immutable :class:`organelleverse.core.OrganelleResult`. Both need the same small
amount of boilerplate (package version, git commit, parameters hash, artifact
refs), so it lives here rather than being duplicated.

Nothing in this module imports numpy / scikit-image / torch / gradio, so it stays
cheap to import.

Scope note
----------
OrgSegNet segments four organelle classes at once (Chloroplast, Mitochondria,
Vacuole, Nucleus) from a single electron-micrograph. The canonical
``ResultScope`` enum has no value meaning "one subcellular image containing
several organelle types", so morphology results declare :data:`MORPHOLOGY_SCOPE`
= ``"mixed"``. That is the closest honest member of the enum; the pre-v1 suite
declared ``"plastid"``, which claimed a chloroplast-only scope the operation
never had.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..core import ArtifactRef, ResultProvenance, ResultScope

__all__ = [
    "MORPHOLOGY_OPERATION_VERSION",
    "MORPHOLOGY_SCOPE",
    "collect_artifacts",
    "make_provenance",
    "parameters_hash",
    "utc_now",
]

#: Version of the morphology operation contracts declared by this module.
MORPHOLOGY_OPERATION_VERSION = "1.0"

#: See the module docstring: OrgSegNet is inherently multi-organelle.
MORPHOLOGY_SCOPE: ResultScope = "mixed"


def utc_now() -> datetime:
    """Timezone-aware wall clock used for provenance timestamps."""
    return datetime.now(UTC)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")


def parameters_hash(parameters: dict[str, Any]) -> str:
    """Return the SHA256 of the canonical JSON encoding of ``parameters``."""
    return hashlib.sha256(_canonical_json_bytes(parameters)).hexdigest()


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def make_provenance(
    operation_id: str,
    *,
    params_hash: str,
    argv: tuple[str, ...] = (),
    actual_backend: str = "",
    attempted_backends: tuple[str, ...] = (),
    software_versions: dict[str, Any] | None = None,
    model_hashes: dict[str, Any] | None = None,
    random_seed: int | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> ResultProvenance:
    """Build a canonical :class:`ResultProvenance` for a morphology operation."""
    duration = None
    if started_at is not None and finished_at is not None:
        duration = max(0.0, (finished_at - started_at).total_seconds())
    return ResultProvenance(
        operation_id=operation_id,
        operation_version=MORPHOLOGY_OPERATION_VERSION,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        parameters_hash=params_hash,
        actual_backend=actual_backend,
        attempted_backends=attempted_backends,
        software_versions=software_versions or {},
        model_hashes=model_hashes or {},
        argv=argv,
        random_seed=random_seed,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration,
    )


def collect_artifacts(
    paths: list[Path] | tuple[Path, ...],
    *,
    kind: str,
    format: str,
    media_type: str = "application/octet-stream",
) -> tuple[ArtifactRef, ...]:
    """Content-address every existing path, skipping the ones that vanished.

    ``ArtifactRef.from_path`` reads and hashes the file, so this is only called
    on paths an executor has just reported as written.
    """
    refs: list[ArtifactRef] = []
    for path in paths:
        candidate = Path(path)
        if not candidate.is_file():
            continue
        refs.append(
            ArtifactRef.from_path(
                candidate,
                kind=kind,
                format=format,
                media_type=media_type,
            )
        )
    return tuple(refs)
