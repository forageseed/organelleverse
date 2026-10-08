"""Codon usage (RSCU/ENC/GC3s) and amino-acid composition.

Codon tables come from Biopython's NCBI genetic-code registry.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from ..core.frozen import FrozenMap, thaw_json
from ..core.provenance import ResultProvenance
from ..core.result import Finding, OrganelleResult, ResultScope
from .codon_core import compute_amino_acid_composition, compute_codon_usage

_OPERATION_VERSION = "1.0"
_METHOD = "organelleverse-biopython-codon-table"

# Legacy suites accepted free-text organelle names; the canonical result scope
# is a closed vocabulary. Anything unrecognized is reported as "none" rather
# than silently claimed as an organelle.
_SCOPE_BY_ORGANELLE: dict[str, ResultScope] = {
    "mito": "mitochondrion",
    "mt": "mitochondrion",
    "mitochondria": "mitochondrion",
    "mitochondrion": "mitochondrion",
    "chloro": "plastid",
    "chloroplast": "plastid",
    "cp": "plastid",
    "pt": "plastid",
    "plastid": "plastid",
}


def _scope(organelle: str) -> ResultScope:
    return _SCOPE_BY_ORGANELLE.get(str(organelle).strip().lower(), "none")


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


def _provenance(operation_id: str, parameters: dict[str, Any], source: Path) -> ResultProvenance:
    return ResultProvenance(
        operation_id=operation_id,
        operation_version=_OPERATION_VERSION,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        input_artifact_hashes=_file_sha256(source),
        parameters_hash=_parameters_hash(parameters),
        actual_backend=_METHOD,
        attempted_backends=(_METHOD,),
    )


def codon_usage(
    cds_fasta: str | Path,
    *,
    organelle: str = "mito",
    genetic_code: int | str | None = None,
) -> OrganelleResult:
    """Compute RSCU, ENC, GC3s from CDS sequences (ENC per Wright 1990).

    ENC extremes are 20 (one codon per amino acid) to 61 (uniform); real
    organelle CDS lands ~45-55 (plastid panel: 47.2). An ENC pinned at
    exactly 61 signals uniform input, not absence of bias.
    """
    metrics = compute_codon_usage(cds_fasta, organelle=organelle, genetic_code=genetic_code)
    enc = metrics["ENC"]
    gc3s = metrics["GC3s"]
    total_codons = metrics["total_codons"]
    return OrganelleResult(
        operation_id="codon_composition.codon_usage",
        operation_version=_OPERATION_VERSION,
        scope=_scope(organelle),
        status="ok",
        summary_text=(
            f"ENC={enc:.2f}, GC3s={gc3s:.4f}, {total_codons} codons "
            f"(NCBI transl_table={metrics['genetic_code']})."
        ),
        metrics=FrozenMap.from_json(metrics),
        findings=(
            Finding(code="codon_composition.enc", metric="ENC", value=enc),
            Finding(code="codon_composition.gc3s", metric="GC3s", value=gc3s),
            Finding(
                code="codon_composition.genetic_code",
                metric="genetic_code",
                value=metrics["genetic_code"],
            ),
        ),
        flags=(),
        artifacts=(),
        provenance=_provenance(
            "codon_composition.codon_usage",
            {"organelle": organelle, "genetic_code": genetic_code},
            Path(cds_fasta),
        ),
    )


def amino_acid(
    cds_fasta: str | Path,
    *,
    organelle: str = "mito",
    genetic_code: int | str | None = None,
    include_stop: bool = False,
) -> OrganelleResult:
    """Compute amino-acid composition from CDS sequences."""
    metrics = compute_amino_acid_composition(
        cds_fasta,
        organelle=organelle,
        genetic_code=genetic_code,
        include_stop=include_stop,
    )
    aa_counts = metrics["counts"]
    total = metrics["total"]
    freq = {aa: round(c / total, 4) for aa, c in aa_counts.items()} if total else {}
    return OrganelleResult(
        operation_id="codon_composition.amino_acid",
        operation_version=_OPERATION_VERSION,
        scope=_scope(organelle),
        status="ok",
        summary_text=f"Amino-acid composition over {total} residues.",
        metrics=FrozenMap.from_json(
            {
                "aa_counts": aa_counts,
                "total": total,
                "frequencies": freq,
                "genetic_code": metrics["genetic_code"],
            }
        ),
        findings=tuple(
            Finding(code="codon_composition.aa_frequency", metric=a, value=f)
            for a, f in freq.items()
        )[:5],
        flags=(),
        artifacts=(),
        provenance=_provenance(
            "codon_composition.amino_acid",
            {
                "organelle": organelle,
                "genetic_code": genetic_code,
                "include_stop": include_stop,
            },
            Path(cds_fasta),
        ),
    )


def write_usage(
    usage: OrganelleResult | Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write codon-usage result tables without recomputing the analysis."""
    metrics = _usage_metrics(usage)
    rows = metrics.get("codon_table")
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("usage result must contain metrics['codon_table']")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    usage_path = out / "codon_usage.tsv"
    usage_path.write_text(_usage_tsv(rows))
    rscu_path = out / "rscu.tsv"
    rscu_path.write_text(
        "codon\tAA\tcount\tRSCU\n"
        + "\n".join(f"{row['codon']}\t{row['AA']}\t{row['count']}\t{row['RSCU']}" for row in rows)
        + "\n"
    )
    return {"codon_usage": usage_path, "rscu": rscu_path}


def write_amino_acid(
    composition: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write amino-acid composition rows without recomputing the analysis."""
    metrics = _result_metrics(composition)
    counts = metrics.get("aa_counts", metrics.get("counts", {}))
    frequencies = metrics.get("frequencies", {})
    if not isinstance(counts, Mapping):
        raise ValueError("amino-acid result must contain counts")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["AA\tcount\tfrequency"]
    for aa in sorted(counts):
        freq = frequencies.get(aa, 0.0) if isinstance(frequencies, Mapping) else 0.0
        lines.append(f"{aa}\t{counts[aa]}\t{freq}")
    out.write_text("\n".join(lines) + "\n")
    return out


def _usage_metrics(usage: OrganelleResult | Mapping[str, Any]) -> dict[str, Any]:
    metrics = _result_metrics(usage)
    if "codon_table" not in metrics:
        raise ValueError("usage result must contain metrics['codon_table']")
    return metrics


def _result_metrics(result: OrganelleResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, OrganelleResult):
        return cast(dict[str, Any], thaw_json(result.metrics))
    if isinstance(result, Mapping):
        payload = result.get("metrics")
        if isinstance(payload, Mapping):
            return dict(payload)
        return dict(result)
    raise TypeError("result must be an OrganelleResult or metrics mapping")


def _usage_tsv(rows: list[dict[str, Any]]) -> str:
    header = ["codon", "AA", "fraction", "frequency_per_thousand", "count", "RSCU"]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row[key]) for key in header))
    return "\n".join(lines) + "\n"
