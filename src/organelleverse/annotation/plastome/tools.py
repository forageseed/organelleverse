"""External tools used by the OrganelleVerse plastome backend."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from Bio.Blast import NCBIWWW

from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.core.external import run_external

from .models import BlastTools


def run_ncbi_qblast(program: str, database: str, sequence: str, **kwargs):
    return NCBIWWW.qblast(program, database, sequence, **kwargs)


def find_blast_tools(
    blastn_path: str | os.PathLike[str] | None = None,
    makeblastdb_path: str | os.PathLike[str] | None = None,
    tblastn_path: str | os.PathLike[str] | None = None,
    losat_path: str | os.PathLike[str] | None = None,
) -> BlastTools:
    return BlastTools(
        blastn=_resolve_executable(blastn_path, "blastn"),
        makeblastdb=_resolve_executable(makeblastdb_path, "makeblastdb"),
        tblastn=_resolve_executable(tblastn_path, "tblastn"),
        losat=_resolve_losat(losat_path),
    )


def _resolve_losat(losat_path: str | os.PathLike[str] | None) -> str | None:
    """Resolve the LOSAT binary, preferring explicit path then env then checkout."""
    if losat_path is not None:
        return _resolve_executable(losat_path, "LOSAT")
    env_value = os.environ.get("ORG_VERSE_LOSAT_BIN")
    if env_value:
        # Explicit opt-out (A/B runs against NCBI BLAST+): the checkout
        # otherwise always finds its vendored build.
        if env_value.strip().casefold() in {"none", "off", "disable", "ncbi", "0"}:
            return None
        resolved = _resolve_executable(env_value, "LOSAT")
        if resolved is not None:
            return resolved
    candidate = (
        Path(__file__).resolve().parents[4]
        / "external_tools"
        / "LOSAT"
        / "LOSAT"
        / "target"
        / "release"
        / "LOSAT"
    )
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which("losat") or shutil.which("LOSAT")


def run_command(cmd: list[str], *, timeout: int) -> None:
    run_external(
        cmd,
        timeout=timeout,
        code="external_command_failed",
        tool=Path(cmd[0]).name,
        error_class=OrganelleExecutionError,
    )


def _resolve_executable(path_or_name: str | os.PathLike[str] | None, default: str) -> str | None:
    if path_or_name is None:
        return shutil.which(default)
    value = os.fspath(path_or_name)
    candidate = Path(value)
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which(value)
