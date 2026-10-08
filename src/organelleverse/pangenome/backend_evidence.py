"""Capture actual backend version output alongside a successful graph build."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..core.artifacts import ArtifactRef
from ..core.result import ErrorDetail, OrganelleResult


def record_backend_version(
    result: OrganelleResult, executable: str, directory: Path, runner
) -> OrganelleResult:
    if result.status == "failed":
        return result
    method = result.provenance.actual_backend
    version_file = directory / f"{method}-version.txt"
    evidence_file = directory / f"{method}-version-command.json"
    try:
        command = runner([executable, "--version"], cwd=directory, stdout_path=version_file)
        evidence_file.write_text(json.dumps(asdict(command), indent=2) + "\n")
        version = version_file.read_text().strip()
        if not command.ok or not version:
            raise ValueError(
                f"{method} version probe did not return successful, nonempty version output"
            )
    except (OSError, ValueError) as error:
        return result.model_copy(
            update={
                "status": "failed",
                "summary_text": str(error),
                "flags": tuple(flag for flag in result.flags if flag != "graph_built"),
                "artifacts": (),
                "errors": (
                    ErrorDetail(code="pangenome.backend_version_failed", message=str(error)),
                ),
            }
        )
    return result.model_copy(
        update={
            "artifacts": (
                *result.artifacts,
                ArtifactRef.from_path(
                    version_file, kind="software_version", format="txt", media_type="text/plain"
                ),
                ArtifactRef.from_path(
                    evidence_file,
                    kind="execution_evidence",
                    format="json",
                    media_type="application/json",
                ),
            ),
            "provenance": result.provenance.model_copy(
                update={"software_versions": {method: version}}
            ),
        }
    )
