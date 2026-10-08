"""SNP/substitution analysis — standard estimators.

- snp(): report SNPs/substitutions per sequence relative to a reference, with
  Ti/Tv ratio. Backend: scikit-allel when available, else pure Python over the
  aligned FASTA.
- snp_density(): sliding-window SNP density (SNPs/kb) — common organelle
  population-genetics summary.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from .._bio import read_fasta
from ..core.frozen import thaw_json
from ..core.provenance import ResultProvenance
from ..core.result import ErrorDetail, Finding, OrganelleResult

_PURINES = {"A", "G"}
_PYRIMIDINES = {"C", "T"}

_OPERATION_VERSION = "1.0"
_SCOPE = "mitochondrion"


# -- canonical contract helpers -------------------------------------------


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


def _provenance(
    operation: str,
    *,
    parameters: dict[str, Any],
    method: str = "",
) -> ResultProvenance:
    """Build canonical provenance for one ``variation.*`` operation."""
    return ResultProvenance(
        operation_id=f"variation.{operation}",
        operation_version=_OPERATION_VERSION,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        parameters_hash=_parameters_hash(parameters),
        requested_backend=method,
        actual_backend=method,
        attempted_backends=(method,) if method else (),
    )


def _failed(
    operation: str,
    *,
    code: str,
    message: str,
    parameters: dict[str, Any],
) -> OrganelleResult:
    """Canonical failed result (a failed result must carry at least one error)."""
    return OrganelleResult(
        operation_id=f"variation.{operation}",
        operation_version=_OPERATION_VERSION,
        scope=_SCOPE,
        status="failed",
        summary_text=message,
        provenance=_provenance(operation, parameters=parameters),
        errors=(ErrorDetail(code=code, message=message),),
    )


def _result_metrics(result: OrganelleResult | Mapping[str, Any]) -> dict[str, Any]:
    """Read the metric mapping from a canonical result or a plain mapping."""
    if isinstance(result, OrganelleResult):
        return cast(dict[str, Any], thaw_json(result.metrics))
    return dict(result)


def snp(
    alignment_fasta: str | Path,
    *,
    reference_index: int = 0,
) -> OrganelleResult:
    """Report SNPs/substitutions in each sequence relative to a reference.

    The reference is the sequence at ``reference_index`` in the alignment.
    Uses scikit-allel for variant encoding when available; otherwise a pure
    Python column scan. Both paths produce the same Ti/Tv ratio.
    """
    parameters = {
        "alignment_fasta": str(alignment_fasta),
        "reference_index": reference_index,
    }
    seqs = read_fasta(Path(alignment_fasta))
    if len(seqs) < 2:
        return _failed(
            "snp",
            code="variation.snp.too_few",
            message="snp requires ≥2 aligned sequences.",
            parameters=parameters,
        )
    ref = seqs[reference_index][1].upper()
    L = len(ref)
    try:
        import allel  # type: ignore  # noqa: F401

        method = "scikit-allel"
    except ImportError:
        method = "organelleverse_python"

    transition = transversion = total_snps = 0
    per_seq = []
    for name, seq in seqs:
        seq = seq.upper()[:L]
        if len(seq) < L:
            seq = seq + "N" * (L - len(seq))
        snps = []
        for i in range(L):
            if seq[i] != ref[i] and seq[i] in "ACGT" and ref[i] in "ACGT":
                snps.append({"pos": i + 1, "ref": ref[i], "alt": seq[i]})
                total_snps += 1
                if {seq[i], ref[i]} <= _PURINES or {seq[i], ref[i]} <= _PYRIMIDINES:
                    transition += 1
                else:
                    transversion += 1
        per_seq.append({"name": name, "snp_count": len(snps)})
    if transversion:
        ti_tv = round(transition / transversion, 4)
        ti_tv_state = "finite"
        flags = ()
        ti_tv_text = f"{ti_tv:.3f}"
    elif transition:
        ti_tv = None
        ti_tv_state = "infinite"
        flags = ("ti_tv_infinite",)
        ti_tv_text = "infinite (Tv=0)"
    else:
        ti_tv = None
        ti_tv_state = "undefined"
        flags = ("ti_tv_undefined",)
        ti_tv_text = "undefined (Ti=Tv=0)"
    return OrganelleResult(
        operation_id="variation.snp",
        operation_version=_OPERATION_VERSION,
        scope=_SCOPE,
        status="ok",
        summary_text=(
            f"{total_snps} SNPs (Ti={transition}, Tv={transversion}, Ti/Tv={ti_tv_text})."
        ),
        metrics={
            "total_snps": total_snps,
            "transitions": transition,
            "transversions": transversion,
            "ti_tv": ti_tv,
            "ti_tv_state": ti_tv_state,
            "per_sequence": per_seq,
        },
        findings=(
            Finding(code="variation.snp.total_snps", metric="total_snps", value=total_snps),
            Finding(code="variation.snp.ti_tv_ratio", metric="ti_tv_ratio", value=ti_tv),
        ),
        flags=flags,
        artifacts=(),
        provenance=_provenance("snp", parameters=parameters, method=method),
    )


def snp_density(
    alignment_fasta: str | Path,
    *,
    reference_index: int = 0,
    window_size: int = 1000,
    step: int = 500,
) -> OrganelleResult:
    """Sliding-window SNP density (SNPs/kb) relative to a reference.

    Standard organelle population-genomics summary for spotting mutational
    hotspots (e.g. around the repeat-mediated recombination junctions).

    Windows start at position 1 and advance by ``step``; the trailing partial
    windows (``L`` not being a multiple of ``step``) are emitted as well, so
    every reference column lies in at least one window (windows overlap when
    ``step`` < ``window_size``). ``columns_not_covered``
    reports the columns outside any window for a non-tiling ``step``
    (``step`` > ``window_size``), i.e. when the windows are deliberately
    gapped.
    """
    parameters = {
        "alignment_fasta": str(alignment_fasta),
        "reference_index": reference_index,
        "window_size": window_size,
        "step": step,
    }
    seqs = read_fasta(Path(alignment_fasta))
    if len(seqs) < 2:
        return _failed(
            "snp_density",
            code="variation.snp_density.too_few",
            message="snp_density requires ≥2 aligned sequences.",
            parameters=parameters,
        )
    ref = seqs[reference_index][1].upper()
    L = len(ref)
    # Count SNPs at each position across all non-ref sequences (relative to ref).
    snp_per_pos = [0] * L
    for _, seq in seqs:
        seq = seq.upper()[:L]
        if len(seq) < L:
            seq = seq + "N" * (L - len(seq))
        for i in range(L):
            if seq[i] != ref[i] and seq[i] in "ACGT" and ref[i] in "ACGT":
                snp_per_pos[i] += 1
    windows = []
    covered = [False] * L
    for start in range(0, max(1, L), step):
        end = min(start + window_size, L)
        if end <= start:
            break
        n_snps = sum(snp_per_pos[start:end])
        density = n_snps / max(1, (end - start)) * 1000  # SNPs/kb
        windows.append(
            {
                "start": start + 1,
                "end": end,
                "snp_count": n_snps,
                "density_per_kb": round(density, 6),
            }
        )
        for i in range(start, end):
            covered[i] = True
    total = sum(snp_per_pos)
    not_covered = L - sum(covered)
    return OrganelleResult(
        operation_id="variation.snp_density",
        operation_version=_OPERATION_VERSION,
        scope=_SCOPE,
        status="ok",
        summary_text=f"{total} SNPs over {len(windows)} windows ({window_size}bp, step {step}).",
        metrics={
            "total_snps": total,
            "windows": len(windows),
            "window_size": window_size,
            "step": step,
            "columns_covered": sum(covered),
            "columns_not_covered": not_covered,
            "windowed_density": windows,
        },
        findings=(
            Finding(code="variation.snp_density.total_snps", metric="total_snps", value=total),
            Finding(code="variation.snp_density.windows", metric="windows", value=len(windows)),
        ),
        flags=(),
        artifacts=(),
        provenance=_provenance("snp_density", parameters=parameters, method="organelleverse"),
    )


def write_snp(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write per-sequence SNP counts from ``snp()``."""
    metrics = _result_metrics(result)
    path = _resolve_output_path(output, "snp.tsv")
    rows = list(metrics.get("per_sequence", ()))
    path.write_text(
        "name\tsnp_count\n"
        + "\n".join(f"{row.get('name', '')}\t{row.get('snp_count', 0)}" for row in rows)
        + ("\n" if rows else "")
    )
    return path


def write_snp_density(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write the windowed SNP-density table from ``snp_density()``."""
    metrics = _result_metrics(result)
    windows = list(metrics.get("windowed_density", ()))
    path = _resolve_output_path(output, "snp_density.windowed.tsv")
    path.write_text(
        "start\tend\tsnp_count\tdensity_per_kb\n"
        + "\n".join(
            f"{w['start']}\t{w['end']}\t{w['snp_count']}\t{w['density_per_kb']}" for w in windows
        )
        + ("\n" if windows else "")
    )
    return path


def _resolve_output_path(output: str | Path, default_name: str) -> Path:
    path = Path(output)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / default_name
