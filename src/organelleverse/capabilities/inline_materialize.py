"""Materialize inline parameter values into temporary files for worker execution.

A plugin parameter may declare that it accepts inline content in addition to, or
instead of, a file path. The runtime writes that content into the run's owned
staging area, verifies its SHA-256, and passes the resulting path to the worker.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from organelleverse.core.errors import (
    OrganelleContractError,
    OrganelleExecutionError,
    OrganelleInputError,
)
from organelleverse.operations.spec import InlineParameterValue


def _decoded_bytes(value: InlineParameterValue) -> bytes:
    if value.encoding == "base64":
        try:
            return base64.b64decode(value.value, validate=True)
        except ValueError as error:
            raise OrganelleInputError(
                code="input.inline_decode_failed",
                message="inline base64 content could not be decoded",
                details={"format": value.format, "reason": str(error)},
            ) from error
    return value.value.encode("utf-8")


def _suffix_from_format(format_value: str) -> str:
    """Pick a reasonable file extension for an inline value."""
    mapping = {
        "fasta": ".fasta",
        "fa": ".fa",
        "fastq": ".fastq",
        "fq": ".fq",
        "genbank": ".gb",
        "gb": ".gb",
        "json": ".json",
        "txt": ".txt",
        "csv": ".csv",
        "tsv": ".tsv",
        "png": ".png",
        "jpg": ".jpg",
        "jpeg": ".jpeg",
        "tif": ".tif",
        "tiff": ".tiff",
    }
    lowered = format_value.lower()
    return mapping.get(lowered, f".{lowered}" if lowered.isalnum() else ".bin")


def decode_inline_content(
    value: InlineParameterValue,
    *,
    parameter_name: str,
    capability_id: str,
    inline_max_bytes: int,
) -> bytes:
    """Decode and verify an inline value, returning its raw bytes.

    Raises if the declared size limit is exceeded or the SHA-256 does not match.
    """
    content = _decoded_bytes(value)
    if len(content) > inline_max_bytes:
        raise OrganelleInputError(
            code="input.inline_too_large",
            message=f"inline content for parameter '{parameter_name}' exceeds inline_max_bytes",
            details={
                "capability_id": capability_id,
                "parameter": parameter_name,
                "observed_bytes": len(content),
                "max_bytes": inline_max_bytes,
            },
        )
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != value.sha256:
        raise OrganelleContractError(
            code="input.inline_digest_mismatch",
            message=f"inline content SHA-256 does not match the declared value for parameter '{parameter_name}'",
            details={
                "capability_id": capability_id,
                "parameter": parameter_name,
                "declared_sha256": value.sha256,
                "actual_sha256": actual_sha256,
            },
        )
    return content


def materialize_inline_parameter(
    value: InlineParameterValue,
    staging_dir: Path,
    *,
    parameter_name: str,
    capability_id: str,
    inline_max_bytes: int,
) -> Path:
    """Write inline content to a file in *staging_dir* and verify its hash.

    The written file is named after the parameter and format so that worker
    logs and provenance stay human-readable. Raises if the declared size limit
    is exceeded or the SHA-256 does not match.
    """
    content = decode_inline_content(
        value,
        parameter_name=parameter_name,
        capability_id=capability_id,
        inline_max_bytes=inline_max_bytes,
    )
    suffix = _suffix_from_format(value.format)
    destination = staging_dir / f"inline-{parameter_name}{suffix}"
    try:
        staging_dir.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    except OSError as error:
        raise OrganelleExecutionError(
            code="input.inline_materialize_failed",
            message=f"inline content could not be materialized for parameter '{parameter_name}'",
            details={
                "capability_id": capability_id,
                "parameter": parameter_name,
                "destination": str(destination),
                "reason": str(error),
            },
        ) from error

    return destination


__all__ = ["decode_inline_content", "materialize_inline_parameter"]
