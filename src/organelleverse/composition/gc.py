"""GC content analysis — self-contained, pure Python."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from .._bio import read_fasta
from ..core.frozen import FrozenMap, thaw_json
from ..core.provenance import ResultProvenance
from ..core.result import ErrorDetail, Finding, OrganelleResult

_OPERATION_ID = "composition.gc_content"
_OPERATION_VERSION = "1.0"
_METHOD = "organelleverse"


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def _parameters_hash(parameters: dict[str, Any]) -> str:
    payload = json.dumps(
        parameters,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> tuple[str, ...]:
    """Content identity of the input, so a Result is not tied to a file path."""
    if not path.is_file():
        return ()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return (digest.hexdigest(),)


def _provenance(parameters: dict[str, Any], source: Path) -> ResultProvenance:
    return ResultProvenance(
        operation_id=_OPERATION_ID,
        operation_version=_OPERATION_VERSION,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        input_artifact_hashes=_file_sha256(source),
        parameters_hash=_parameters_hash(parameters),
        actual_backend=_METHOD,
        attempted_backends=(_METHOD,),
    )


def gc_content(
    genome_fasta: str | Path,
    *,
    window_size: int = 500,
) -> OrganelleResult:
    """Compute global + windowed GC content (and GC skew) from a FASTA."""
    source = Path(genome_fasta)
    parameters = {"window_size": window_size}
    seqs = read_fasta(source)
    if not seqs:
        return OrganelleResult(
            operation_id=_OPERATION_ID,
            operation_version=_OPERATION_VERSION,
            scope="none",
            status="failed",
            summary_text="No sequences.",
            provenance=_provenance(parameters, source),
            errors=(
                ErrorDetail(
                    code="input.empty_fasta",
                    message="No sequences.",
                    details={"path": str(genome_fasta)},
                ),
            ),
        )
    all_seq = "".join(s.upper() for _, s in seqs)
    total = len(all_seq)
    gc = all_seq.count("G") + all_seq.count("C")
    gc_frac = gc / total if total else 0.0
    # GC skew: (G-C)/(G+C) per window
    windows = []
    for start in range(0, max(1, total - window_size + 1), window_size):
        end = min(start + window_size, total)
        w = all_seq[start:end]
        g = w.count("G")
        c = w.count("C")
        skew = (g - c) / (g + c) if (g + c) else 0.0
        windows.append({"start": start + 1, "end": end, "skew": round(skew, 4)})
    return OrganelleResult(
        operation_id=_OPERATION_ID,
        operation_version=_OPERATION_VERSION,
        scope="none",
        status="ok",
        summary_text=f"GC content = {gc_frac:.4f} over {total} bp.",
        metrics=FrozenMap.from_json(
            {
                "gc_content": round(gc_frac, 4),
                "total_length": total,
                "windows": len(windows),
                "windowed_gc_skew": windows,
            }
        ),
        findings=(
            Finding(code="composition.gc_content", metric="gc_content", value=round(gc_frac, 4)),
        ),
        flags=(),
        artifacts=(),
        provenance=_provenance(parameters, source),
    )


def write_content(
    result: OrganelleResult | Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write GC-content sidecar tables from a computed result."""
    metrics = (
        cast(dict[str, Any], thaw_json(result.metrics))
        if isinstance(result, OrganelleResult)
        else dict(result)
    )
    windows = list(metrics.get("windowed_gc_skew", ()))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    window_path = out / "gc.tsv"
    window_path.write_text(
        "start\tend\tGC_skew\n"
        + "\n".join(f"{w['start']}\t{w['end']}\t{w['skew']}" for w in windows)
        + ("\n" if windows else "")
    )
    return {"windows": window_path}
