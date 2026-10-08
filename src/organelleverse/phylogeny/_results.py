"""Canonical v1 result construction for the phylogeny suite.

This module is the whole type boundary of the suite: every phylogeny entry
point returns an immutable :class:`organelleverse.core.result.OrganelleResult`
built here, and no scientific code below this layer knows about contracts.

Mapping from the removed pre-v1 dataclass contract:

``suite``/``op``        -> ``operation_id`` (``"phylogeny.<op>"``)
``organelle="mito"``    -> ``scope="mitochondrion"``
``observed_metrics``    -> ``metrics`` (frozen JSON)
``key_findings``        -> ``findings`` (``Finding`` values must be scalars, so
                           list-valued findings such as ``argv`` and
                           ``shared_genes`` live in ``metrics`` instead)
``output_paths``        -> ``artifacts`` (content-addressed ``ArtifactRef``)
``anomalies``           -> ``errors[].code`` + ``errors[].details["anomalies"]``
``ResultProvenance``    -> canonical ``ResultProvenance``: ``method`` becomes
                           ``actual_backend``/``attempted_backends``, ``argv``
                           stays ``argv``, and the call parameters are hashed
                           into ``parameters_hash``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..core.artifacts import ArtifactRef
from ..core.provenance import ResultProvenance
from ..core.result import ErrorDetail, Finding, OrganelleResult, ResultScope

OPERATION_VERSION = "1.0"

# The pre-v1 suite tagged every phylogeny result ``organelle="mito"``. The
# canonical literal for that value is "mitochondrion"; the value is preserved
# verbatim so no observable claim changes with the port.
SCOPE: ResultScope = "mitochondrion"

__all__ = [
    "OPERATION_VERSION",
    "SCOPE",
    "artifact_for",
    "failed_result",
    "findings",
    "ok_result",
    "operation_id",
    "package_version",
    "parameters_hash",
    "provenance",
]


def operation_id(op: str) -> str:
    """Return the canonical operation id for a phylogeny operation name."""
    return f"phylogeny.{op}"


def package_version() -> str:
    """Return the installed package version, or the source default."""
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def parameters_hash(parameters: Mapping[str, object] | None) -> str:
    """Hash the call parameters into the 64-hex digest the contract requires."""
    payload = json.dumps(
        dict(parameters or {}),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def provenance(
    op: str,
    *,
    method: str = "",
    argv: Sequence[str] = (),
    parameters: Mapping[str, object] | None = None,
    random_seed: int | None = None,
) -> ResultProvenance:
    """Build canonical provenance for one phylogeny operation."""
    backend = method or ""
    return ResultProvenance(
        operation_id=operation_id(op),
        operation_version=OPERATION_VERSION,
        package_version=package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        parameters_hash=parameters_hash(parameters),
        actual_backend=backend,
        attempted_backends=(backend,) if backend else (),
        argv=tuple(argv),
        random_seed=random_seed,
    )


def findings(*pairs: tuple[str, int | float | str | bool | None]) -> tuple[Finding, ...]:
    """Build scalar findings from ``(metric, value)`` pairs."""
    return tuple(
        Finding(code=f"phylogeny.{metric}", metric=metric, value=value) for metric, value in pairs
    )


def ok_result(
    op: str,
    *,
    status: str = "ok",
    summary_text: str,
    metrics: Mapping[str, object],
    result_findings: tuple[Finding, ...] = (),
    flags: tuple[str, ...] = (),
    artifacts: tuple[ArtifactRef, ...] = (),
    result_provenance: ResultProvenance | None = None,
) -> OrganelleResult:
    """Build a successful canonical phylogeny result."""
    return OrganelleResult(
        operation_id=operation_id(op),
        operation_version=OPERATION_VERSION,
        scope=SCOPE,
        status="ok" if status == "ok" else "warning",
        summary_text=summary_text,
        metrics=dict(metrics),
        findings=result_findings,
        flags=flags,
        artifacts=artifacts,
        provenance=result_provenance,
    )


def failed_result(
    op: str,
    *,
    summary_text: str,
    code: str,
    anomalies: Sequence[str] = (),
    details: Mapping[str, object] | None = None,
    result_provenance: ResultProvenance | None = None,
) -> OrganelleResult:
    """Build a failed canonical phylogeny result.

    The pre-v1 ``anomalies`` tuple has no canonical counterpart; its tokens are
    preserved verbatim under ``errors[0].details["anomalies"]`` so no diagnostic
    string is lost.
    """
    payload: dict[str, object] = {"anomalies": list(anomalies)}
    payload.update(dict(details or {}))
    return OrganelleResult(
        operation_id=operation_id(op),
        operation_version=OPERATION_VERSION,
        scope=SCOPE,
        status="failed",
        summary_text=summary_text,
        provenance=result_provenance or provenance(op),
        errors=(
            ErrorDetail(
                code=code,
                message=summary_text,
                details=payload,
                retryable=False,
            ),
        ),
    )


def artifact_for(
    path: str | Path,
    *,
    kind: str,
    format: str,
    media_type: str = "text/plain",
) -> ArtifactRef | None:
    """Return a content-addressed reference, or None when the file is absent.

    Planning-only runs (and injected test executors) never create the output
    file; the pre-v1 contract still listed the path, but an ``ArtifactRef`` is
    content-addressed and cannot reference a file that does not exist. Callers
    keep the planned path in ``metrics`` instead.
    """
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return ArtifactRef.from_path(candidate, kind=kind, format=format, media_type=media_type)
