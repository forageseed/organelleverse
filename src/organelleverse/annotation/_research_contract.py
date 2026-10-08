"""Canonical v1 contract construction for the research annotation backends.

The scientific behaviour of :mod:`organelleverse.annotation.research` is not
affected by this module. It only builds the frozen
:mod:`organelleverse.core` objects that the research entry points return at
their I/O boundary, so the backend code stays focused on annotation instead of
on contract plumbing.

Mapping from the retired pre-v1 dataclass contract:

==============================  ==============================================
pre-v1 field                    canonical field
==============================  ==============================================
``suite`` + ``op``              ``operation_id`` (``"annotation.<op>"``)
``organelle="mito"``            ``scope="mitochondrion"``
``organelle="chloro"``          ``scope="plastid"``
``organelle="plastid"``         ``scope="plastid"``
``observed_metrics``            ``metrics`` (frozen JSON object)
``key_findings``                ``findings``; :class:`Finding` values are
                                scalar-only, so any list/mapping payload stays
                                in ``metrics``
``output_paths``                ``artifacts`` (content-addressed
                                :class:`ArtifactRef`). An ``ArtifactRef``
                                cannot reference a file that was never
                                written, so the full path list is preserved
                                verbatim under ``metrics["output_paths"]``
``anomalies``                   ``errors`` — one :class:`ErrorDetail` per
                                token, token kept under
                                ``errors[].details["anomaly"]`` (the canonical
                                contract requires at least one error whenever
                                ``status == "failed"``)
``provenance.suite``/``op``     ``provenance.operation_id``
``provenance.method``           ``provenance.actual_backend`` /
                                ``requested_backend`` /
                                ``attempted_backends``; this suite never falls
                                back, so requested and actual are always the
                                same resolved backend identity and the
                                caller's ``backend`` selector ("auto", ...) is
                                recorded through ``parameters_hash`` instead
``provenance.software_version``  ``provenance.software_versions[method]``
==============================  ==============================================
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..core.artifacts import ArtifactRef
from ..core.genome import OrganelleGenome
from ..core.provenance import ResultProvenance
from ..core.result import ErrorDetail, Finding, OrganelleResult, ResultScope

__all__ = [
    "OPERATION_VERSION",
    "SUITE",
    "artifact",
    "artifacts",
    "failed",
    "findings",
    "legacy_organelle",
    "ok",
    "operation_id",
    "package_version",
    "parameters_hash",
    "provenance",
    "scope_for",
]

SUITE = "annotation"
OPERATION_VERSION = "1.0"

#: Canonical result scope for each organelle token. The retired pre-v1 tokens
#: are kept so the table also documents the pre-v1 mapping.
_SCOPE_BY_ORGANELLE: dict[str, ResultScope] = {
    "mitochondrion": "mitochondrion",
    "mito": "mitochondrion",
    "plastid": "plastid",
    "chloro": "plastid",
}

#: Pre-v1 organelle token for each canonical organelle. ``_resolve_backend``
#: in :mod:`~organelleverse.annotation.research` dispatches on the pre-v1
#: vocabulary; normalizing here keeps that resolver byte-for-byte unchanged.
_LEGACY_ORGANELLE: dict[str, str] = {
    "mitochondrion": "mito",
    "plastid": "plastid",
}

# Artifact classification kept semantically identical to the retired pre-v1
# ``_artifact_kind`` suffix table.
_ARTIFACT_KINDS: dict[str, str] = {
    ".tsv": "table",
    ".csv": "table",
    ".nwk": "tree",
    ".tree": "tree",
    ".treefile": "tree",
    ".gfa": "graph",
    ".graph": "graph",
    ".svg": "figure",
    ".png": "figure",
    ".pdf": "figure",
    ".fa": "sequence",
    ".fasta": "sequence",
    ".fna": "sequence",
    ".faa": "sequence",
    ".gff": "annotation",
    ".gff3": "annotation",
    ".gb": "annotation",
    ".gbk": "annotation",
    ".json": "json",
}

_MEDIA_TYPES: dict[str, str] = {
    ".json": "application/json",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".pdf": "application/pdf",
}

_TEXT_SUFFIXES = frozenset(
    {
        ".tsv",
        ".csv",
        ".nwk",
        ".tree",
        ".treefile",
        ".gfa",
        ".fa",
        ".fasta",
        ".fna",
        ".faa",
        ".gff",
        ".gff3",
        ".gb",
        ".gbk",
        ".txt",
    }
)


def operation_id(op: str) -> str:
    """Return the canonical ``annotation.<op>`` operation identifier."""
    return f"{SUITE}.{op}"


def scope_for(genome: OrganelleGenome) -> ResultScope:
    """Return the canonical result scope for one genome's organelle."""
    return _SCOPE_BY_ORGANELLE[genome.organelle]


def legacy_organelle(genome: OrganelleGenome) -> str:
    """Return the pre-v1 organelle token the backend resolver dispatches on."""
    return _LEGACY_ORGANELLE[genome.organelle]


def package_version() -> str:
    """Return the installed package version, or the source default."""
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def parameters_hash(parameters: Mapping[str, Any] | None = None) -> str:
    """Return a deterministic SHA256 over the recorded call parameters."""
    payload = json.dumps(
        dict(parameters or {}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def provenance(
    op: str,
    *,
    method: str = "",
    software_version: str = "",
    parameters: Mapping[str, Any] | None = None,
    argv: Sequence[str] = (),
) -> ResultProvenance:
    """Build canonical provenance for one research annotation operation.

    The pre-v1 ``method`` string (``"orf_sixframe"``, ``"pyhmmer+orf"``,
    ``"mitochondrion"``, ``"plastome"``) becomes the resolved backend identity.
    """
    return ResultProvenance(
        operation_id=operation_id(op),
        operation_version=OPERATION_VERSION,
        package_version=package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        parameters_hash=parameters_hash(parameters),
        requested_backend=method,
        actual_backend=method,
        attempted_backends=(method,) if method else (),
        software_versions={method: software_version} if method and software_version else {},
        argv=tuple(argv),
    )


def artifact(path: str | Path) -> ArtifactRef:
    """Content-address one materialized annotation output file."""
    candidate = Path(path)
    suffix = candidate.suffix.lower()
    return ArtifactRef.from_path(
        candidate,
        kind=_ARTIFACT_KINDS.get(suffix, "file"),
        format=suffix.lstrip(".") or "text",
        media_type=_media_type(suffix),
    )


def artifacts(paths: Iterable[str | Path]) -> tuple[ArtifactRef, ...]:
    """Content-address every output file that was actually written.

    ``ArtifactRef`` is content-addressed and cannot describe a file that does
    not exist, so absent paths are skipped here; callers keep the complete path
    list in ``metrics["output_paths"]``.
    """
    return tuple(artifact(path) for path in paths if Path(path).is_file())


def _media_type(suffix: str) -> str:
    if suffix in _MEDIA_TYPES:
        return _MEDIA_TYPES[suffix]
    if suffix in _TEXT_SUFFIXES:
        return "text/plain"
    return "application/octet-stream"


def findings(
    op: str,
    rows: Iterable[tuple[str, int | float | str | bool | None]],
) -> tuple[Finding, ...]:
    """Build canonical scalar findings from pre-v1 ``(metric, value)`` rows."""
    return tuple(
        Finding(code=f"{SUITE}.{op}.{metric}", metric=metric, value=value) for metric, value in rows
    )


def _with_output_paths(
    metrics: Mapping[str, Any] | None,
    output_paths: Sequence[str | Path],
) -> dict[str, Any]:
    merged = dict(metrics or {})
    if output_paths:
        merged["output_paths"] = [str(Path(path)) for path in output_paths]
    return merged


def ok(
    op: str,
    *,
    scope: ResultScope,
    summary_text: str,
    metrics: Mapping[str, Any] | None = None,
    result_findings: tuple[Finding, ...] = (),
    flags: Sequence[str] = (),
    output_paths: Sequence[str | Path] = (),
    method: str = "",
    software_version: str = "",
    parameters: Mapping[str, Any] | None = None,
    argv: Sequence[str] = (),
) -> OrganelleResult:
    """Build a successful canonical research annotation result."""
    return OrganelleResult(
        operation_id=operation_id(op),
        operation_version=OPERATION_VERSION,
        scope=scope,
        status="ok",
        summary_text=summary_text,
        metrics=_with_output_paths(metrics, output_paths),
        findings=result_findings,
        flags=tuple(flags),
        artifacts=artifacts(output_paths),
        provenance=provenance(
            op,
            method=method,
            software_version=software_version,
            parameters=parameters,
            argv=argv,
        ),
    )


def failed(
    op: str,
    *,
    scope: ResultScope,
    summary_text: str,
    anomalies: Sequence[str] = (),
    metrics: Mapping[str, Any] | None = None,
    flags: Sequence[str] = (),
    output_paths: Sequence[str | Path] = (),
    method: str = "",
    parameters: Mapping[str, Any] | None = None,
    argv: Sequence[str] = (),
) -> OrganelleResult:
    """Build a failed canonical research annotation result.

    Every pre-v1 anomaly token becomes one structured error and is retained
    verbatim under ``details["anomaly"]`` so no diagnostic string is lost.
    """
    tokens = tuple(anomalies) or ("failed",)
    errors = tuple(
        ErrorDetail(
            code=f"{SUITE}.{op}.{token}",
            message=summary_text,
            details={"anomaly": token},
        )
        for token in tokens
    )
    return OrganelleResult(
        operation_id=operation_id(op),
        operation_version=OPERATION_VERSION,
        scope=scope,
        status="failed",
        summary_text=summary_text,
        metrics=_with_output_paths(metrics, output_paths),
        flags=tuple(flags),
        artifacts=artifacts(output_paths),
        provenance=provenance(op, method=method, parameters=parameters, argv=argv),
        errors=errors,
    )
