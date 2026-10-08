"""Canonical v1 contract construction for the selection suite.

The scientific algorithms in this suite are unchanged. This module only builds
the frozen :mod:`organelleverse.core` contract objects that the suite returns
at its I/O boundary, so the selection functions can stay focused on the
science instead of on contract plumbing.

Mapping from the retired pre-v1 dataclass contract:

======================  ==============================================
pre-v1 field            canonical field
======================  ==============================================
``suite`` + ``op``      ``operation_id`` (``"selection.<op>"``)
``organelle="mito"``    ``scope="mitochondrion"``
``observed_metrics``    ``metrics`` (frozen JSON object)
``key_findings``        ``findings`` (tuple of :class:`Finding`)
``output_paths``        ``artifacts`` (tuple of :class:`ArtifactRef`)
``anomalies``           ``errors`` (tuple of :class:`ErrorDetail`)
``provenance.method``   ``provenance.actual_backend``
======================  ==============================================
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
from ..core.provenance import ResultProvenance
from ..core.result import ErrorDetail, Finding, OrganelleResult

__all__ = [
    "OPERATION_VERSION",
    "SCOPE",
    "SUITE",
    "artifact",
    "artifacts",
    "failed",
    "findings",
    "ok",
    "operation_id",
    "provenance",
]

SUITE = "selection"
#: Every selection operation was declared for mitochondrial input pre-v1.
SCOPE = "mitochondrion"
OPERATION_VERSION = "1.0"

# Artifact classification kept semantically identical to the retired pre-v1
# suffix table, plus the two codeml-specific formats this suite materializes.
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
    ".ctl": "codeml_control",
    ".paml": "paml_alignment",
    ".pml": "paml_alignment",
    ".axt": "axt_alignment",
    ".out": "codeml_report",
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
        ".ctl",
        ".paml",
        ".pml",
        ".axt",
        ".out",
        ".txt",
    }
)


def operation_id(op: str) -> str:
    """Return the canonical ``selection.<op>`` operation identifier."""
    return f"{SUITE}.{op}"


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
    parameters: Mapping[str, Any] | None = None,
    software_versions: Mapping[str, Any] | None = None,
    argv: Sequence[str] = (),
) -> ResultProvenance:
    """Build canonical provenance for one selection operation.

    The pre-v1 ``method`` string (``"nei_gojobori_1986"``, ``"codeml_branch"``,
    ``"KaKs_Calculator_YN"``, ...) becomes the resolved backend identity.
    """
    return ResultProvenance(
        operation_id=operation_id(op),
        operation_version=OPERATION_VERSION,
        package_version=package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        parameters_hash=parameters_hash(parameters),
        actual_backend=method,
        attempted_backends=(method,) if method else (),
        software_versions=dict(software_versions or {}),
        argv=tuple(argv),
    )


def artifact(
    path: str | Path,
    *,
    kind: str | None = None,
    format: str | None = None,
    media_type: str | None = None,
) -> ArtifactRef:
    """Content-address one materialized selection output file."""
    candidate = Path(path)
    suffix = candidate.suffix.lower()
    return ArtifactRef.from_path(
        candidate,
        kind=kind or _ARTIFACT_KINDS.get(suffix, "file"),
        format=format or (suffix.lstrip(".") or "text"),
        media_type=media_type or _media_type(suffix),
    )


def artifacts(paths: Iterable[str | Path]) -> tuple[ArtifactRef, ...]:
    """Content-address a sequence of materialized output files."""
    return tuple(artifact(path) for path in paths)


def _media_type(suffix: str) -> str:
    if suffix in _MEDIA_TYPES:
        return _MEDIA_TYPES[suffix]
    if suffix in _TEXT_SUFFIXES:
        return "text/plain"
    return "application/octet-stream"


def findings(
    op: str,
    rows: Iterable[tuple[str, Any] | tuple[str, Any, str]],
) -> tuple[Finding, ...]:
    """Build canonical findings from ``(metric, value[, unit])`` rows."""
    built: list[Finding] = []
    for row in rows:
        metric = str(row[0])
        value = row[1]
        unit = str(row[2]) if len(row) > 2 else ""
        built.append(
            Finding(
                code=f"{SUITE}.{op}.{metric}",
                metric=metric,
                value=value,
                unit=unit,
            )
        )
    return tuple(built)


def ok(
    op: str,
    *,
    summary_text: str,
    metrics: Mapping[str, Any] | None = None,
    findings: tuple[Finding, ...] = (),
    flags: Sequence[str] = (),
    artifacts: tuple[ArtifactRef, ...] = (),
    method: str = "",
    parameters: Mapping[str, Any] | None = None,
    software_versions: Mapping[str, Any] | None = None,
    argv: Sequence[str] = (),
) -> OrganelleResult:
    """Build a successful canonical selection result.

    ``software_versions``/``argv`` record an external tool's identity and exact
    command line (e.g. HyPhy) in provenance; both default to empty.
    """
    return OrganelleResult(
        operation_id=operation_id(op),
        operation_version=OPERATION_VERSION,
        scope=SCOPE,
        status="ok",
        summary_text=summary_text,
        metrics=dict(metrics or {}),
        findings=findings,
        flags=tuple(flags),
        artifacts=artifacts,
        provenance=provenance(
            op,
            method=method,
            parameters=parameters,
            software_versions=software_versions,
            argv=argv,
        ),
    )


def failed(
    op: str,
    *,
    summary_text: str,
    anomalies: Sequence[str] = (),
    metrics: Mapping[str, Any] | None = None,
    flags: Sequence[str] = (),
    artifacts: tuple[ArtifactRef, ...] = (),
    method: str = "",
    parameters: Mapping[str, Any] | None = None,
) -> OrganelleResult:
    """Build a failed canonical selection result.

    Each pre-v1 anomaly token becomes one structured error; the exact token is
    retained in ``details`` so nothing observable is lost in the port.
    """
    tokens = tuple(anomalies) or ("failed",)
    errors = tuple(
        ErrorDetail(
            code=f"{SUITE}.{op}.{token.split(':', 1)[0]}",
            message=summary_text,
            details={"anomaly": token},
        )
        for token in tokens
    )
    return OrganelleResult(
        operation_id=operation_id(op),
        operation_version=OPERATION_VERSION,
        scope=SCOPE,
        status="failed",
        summary_text=summary_text,
        metrics=dict(metrics or {}),
        flags=tuple(flags),
        artifacts=artifacts,
        provenance=provenance(op, method=method, parameters=parameters),
        errors=errors,
    )
