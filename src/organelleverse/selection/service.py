"""Managed public entry points for file-producing selection tools."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..core.result import OrganelleResult
from ..runtime import managed_run_path


def fasta_to_axt(fasta_path: str | Path) -> Path:
    """Convert aligned FASTA to an AXT artifact in managed storage."""
    from .kaks_calculator import fasta_to_axt as _convert

    output = managed_run_path("selection.fasta_to_axt", uuid4().hex) / "aligned.axt"
    output.parent.mkdir(parents=True, exist_ok=True)
    return _convert(fasta_path, output)


def run_kaks_calculator_workflow(
    cds_fasta: str | Path,
    *,
    method: str = "YN",
    genetic_code: int = 1,
    kaks_path: str | Path | None = None,
) -> OrganelleResult:
    """Run KaKs Calculator with intermediate files in managed storage."""
    from .kaks_calculator import run_kaks_calculator_workflow as _run

    return _run(
        cds_fasta,
        method=method,
        output_dir=managed_run_path("selection.run_kaks_calculator_workflow", uuid4().hex),
        genetic_code=genetic_code,
        kaks_path=kaks_path,
    )


def run_codeml(ctl_path: str | Path, *, timeout: float | None = 3600.0) -> dict[str, Any] | None:
    """Run CodeML using a managed copy of its control and input files."""
    from .models import run_codeml as _run

    source = Path(ctl_path).resolve()
    work = managed_run_path("selection.run_codeml", uuid4().hex)
    work.mkdir(parents=True, exist_ok=True)
    text = source.read_text()
    for key in ("seqfile", "treefile"):
        match = re.search(rf"(?m)^\s*{key}\s*=\s*(\S+)", text)
        if match is None:
            continue
        value = Path(match.group(1))
        if not value.is_absolute():
            target_relative = Path(value.name) if ".." in value.parts else value
            if target_relative != value:
                text = text[: match.start(1)] + str(target_relative) + text[match.end(1) :]
            target = work / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.parent / value, target)
    output = re.search(r"(?m)^\s*outfile\s*=\s*(\S+)", text)
    if output is not None:
        output_value = Path(output.group(1))
        if output_value.is_absolute() or ".." in output_value.parts:
            text = text[: output.start(1)] + "run.out" + text[output.end(1) :]
            output_value = Path("run.out")
        (work / output_value).parent.mkdir(parents=True, exist_ok=True)
    local_ctl = work / source.name
    local_ctl.write_text(text)
    return _run(local_ctl, work, timeout=timeout)
