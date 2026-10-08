"""Horizontal gene transfer (HGT) candidate detection."""

from __future__ import annotations
import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping
from ..core.frozen import FrozenMap
from ..core.provenance import ResultProvenance
from ..core.result import ErrorDetail, Finding, OrganelleResult, ResultScope
from .._bio import read_fasta
from .align import (
    HGTAlignment,
    HGTBackendError,
    HGTBlastHit,
    run_blastn_confirmation,
    run_hgt_alignments,
)

_SUITE = "hgt"
_OPERATION_VERSION = "1.0"
# HGT donor/recipient comparison is run on organelle genomes; the legacy suite
# reported every detect_hgt result under the mitochondrial scope.
_SCOPE: ResultScope = "mitochondrion"


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _provenance(op: str, parameters: Mapping[str, Any], *, method: str) -> ResultProvenance:
    """Build canonical provenance for one ``hgt`` operation."""
    package_version = _package_version()
    return ResultProvenance(
        operation_id=f"{_SUITE}.{op}",
        operation_version=_OPERATION_VERSION,
        package_version=package_version,
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        parameters_hash=_sha256_json(dict(parameters)),
        actual_backend=method,
        attempted_backends=(method,),
        software_versions=FrozenMap({"organelleverse": package_version}),
    )


def detect_hgt(
    donor_fasta: str | Path,
    recipient_fasta: str | Path,
    *,
    min_len: int = 200,
    gc_threshold: float = 0.05,
    min_identity: float = 90.0,
    backend: str = "auto",
    preset: str = "asm5",
    minimap2_path: str | Path | None = None,
    confirm_blast: bool = True,
    blastn_path: str | Path | None = None,
    makeblastdb_path: str | Path | None = None,
    phylogeny_evidence: Mapping[str, Any] | str | Path | None = None,
) -> OrganelleResult:
    """Detect horizontal gene transfer from donor to recipient.

    Finds recipient fragments aligning to donor sequence with minimap2/mappy
    semantics, then flags candidate-fragment GC content that diverges from the
    recipient mean.
    """
    parameters = {
        "min_len": min_len,
        "gc_threshold": gc_threshold,
        "min_identity": min_identity,
        "backend": backend,
        "preset": preset,
        "confirm_blast": confirm_blast,
    }
    try:
        selected_backend, alignments = run_hgt_alignments(
            donor_fasta,
            recipient_fasta,
            backend=backend,
            min_len=min_len,
            min_identity=min_identity,
            preset=preset,
            minimap2_path=minimap2_path,
        )
    except HGTBackendError as exc:
        return OrganelleResult(
            operation_id=f"{_SUITE}.detect_hgt",
            operation_version=_OPERATION_VERSION,
            scope=_SCOPE,
            status="failed",
            summary_text=str(exc),
            metrics=FrozenMap({"backend": backend, "error": str(exc)}),
            errors=(ErrorDetail(code=exc.code, message=str(exc)),),
            provenance=_provenance("detect_hgt", parameters, method="alignment_backend"),
        )

    metrics = _build_hgt_metrics(
        alignments,
        donor_fasta=donor_fasta,
        recipient_fasta=recipient_fasta,
        backend=selected_backend,
        min_len=min_len,
        min_identity=min_identity,
        gc_threshold=gc_threshold,
        confirm_blast=confirm_blast,
        blastn_path=blastn_path,
        makeblastdb_path=makeblastdb_path,
        phylogeny_evidence=phylogeny_evidence,
    )
    flags = [f"alignment_backend_{selected_backend}"]
    if metrics["fragment_count"]:
        flags.append("alignment_candidate")
        flags.append("transfer_detected")
    if metrics["blast"]["supported"]:
        flags.append("blast_supported")
    if metrics["composition"]["supported"]:
        flags.append("composition_supported")
        flags.append("gc_anomaly_hgt_signal")
    if metrics["phylogeny"]["supported"]:
        flags.append("phylogeny_supported")
    quality = metrics["quality"]
    if quality["low_complexity"]:
        flags.append("low_complexity_candidate")
    if quality["n_content_risk"]:
        flags.append("n_content_candidate")
    if quality["repeat_or_paralogy_risk"]:
        flags.append("ambiguous_repeat_or_paralogy")
    total_bp = metrics["total_transferred_bp"]
    return OrganelleResult(
        operation_id=f"{_SUITE}.detect_hgt",
        operation_version=_OPERATION_VERSION,
        scope=_SCOPE,
        status="ok",
        summary_text=(
            f"{metrics['fragment_count']} HGT candidate fragments, {total_bp} bp total "
            f"({selected_backend}); evidence={metrics['evidence_level']}; "
            f"candidate GC delta={metrics['gc_delta']:.3f}."
        ),
        metrics=FrozenMap(metrics),
        findings=(
            Finding(code="fragments", metric="fragments", value=metrics["fragment_count"]),
            Finding(code="total_bp", metric="total_bp", value=total_bp, unit="bp"),
            Finding(code="gc_delta", metric="gc_delta", value=round(metrics["gc_delta"], 4)),
        ),
        flags=tuple(flags),
        artifacts=(),
        provenance=_provenance(
            "detect_hgt",
            parameters,
            method="mappy_minimap2" if selected_backend == "mappy" else "minimap2_cli",
        ),
    )


def _build_hgt_metrics(
    alignments: list[HGTAlignment],
    *,
    donor_fasta: str | Path,
    recipient_fasta: str | Path,
    backend: str,
    min_len: int,
    min_identity: float,
    gc_threshold: float,
    confirm_blast: bool,
    blastn_path: str | Path | None = None,
    makeblastdb_path: str | Path | None = None,
    phylogeny_evidence: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, object]:
    metrics = _alignment_metrics(
        alignments,
        donor_fasta=donor_fasta,
        recipient_fasta=recipient_fasta,
        backend=backend,
    )
    metrics["blast"] = _blast_evidence(
        alignments,
        donor_fasta=donor_fasta,
        recipient_fasta=recipient_fasta,
        min_len=min_len,
        min_identity=min_identity,
        confirm_blast=confirm_blast,
        blastn_path=blastn_path,
        makeblastdb_path=makeblastdb_path,
    )
    metrics["composition"] = {
        "supported": bool(metrics["fragment_count"]) and metrics["gc_delta"] >= gc_threshold,
        "metric": "candidate_gc_delta",
        "threshold": gc_threshold,
        "value": metrics["gc_delta"],
    }
    quality = dict(metrics["quality"])
    quality["repeat_or_paralogy_risk"] = _repeat_or_paralogy_risk(metrics)
    quality["pass"] = not (
        quality["low_complexity"] or quality["n_content_risk"] or quality["repeat_or_paralogy_risk"]
    )
    metrics["quality"] = quality
    metrics["phylogeny"] = _normalize_phylogeny_evidence(phylogeny_evidence)
    metrics["evidence_level"] = _evidence_level(metrics)
    return metrics


def _alignment_metrics(
    alignments: list[HGTAlignment],
    *,
    donor_fasta: str | Path,
    recipient_fasta: str | Path,
    backend: str,
) -> dict[str, object]:
    recipient_records = {name: seq.upper() for name, seq in read_fasta(Path(recipient_fasta))}
    donor_seq = "".join(s.upper() for _, s in read_fasta(Path(donor_fasta)))
    recipient_seq = "".join(recipient_records.values())
    merged_intervals = _merge_recipient_intervals(alignments)
    candidate_seq = "".join(
        recipient_records.get(recipient_id, "")[start - 1 : end]
        for recipient_id, start, end in merged_intervals
    )
    total_bp = sum(end - start + 1 for _, start, end in merged_intervals)
    recipient_len = len(recipient_seq)
    recipient_gc = _gc_fraction(recipient_seq)
    candidate_gc = _gc_fraction(candidate_seq) if candidate_seq else 0.0
    donor_gc = _gc_fraction(donor_seq)
    gc_delta = abs(candidate_gc - recipient_gc) if candidate_seq else 0.0
    return {
        "backend": backend,
        "alignment_count": len(alignments),
        "fragment_count": len(merged_intervals),
        "total_transferred_bp": total_bp,
        "fraction": round(total_bp / recipient_len, 6) if recipient_len else 0,
        "best_identity": round(max((a.identity for a in alignments), default=0.0), 4),
        "recipient_gc": round(recipient_gc, 4),
        "candidate_gc": round(candidate_gc, 4),
        "donor_gc": round(donor_gc, 4),
        "gc_delta": round(gc_delta, 4),
        "quality": _candidate_quality(candidate_seq),
        "fragments": [a.as_dict() for a in alignments],
    }


def _blast_evidence(
    alignments: list[HGTAlignment],
    *,
    donor_fasta: str | Path,
    recipient_fasta: str | Path,
    min_len: int,
    min_identity: float,
    confirm_blast: bool,
    blastn_path: str | Path | None,
    makeblastdb_path: str | Path | None,
) -> dict[str, object]:
    if not confirm_blast:
        return {"status": "skipped", "supported": False, "hit_count": 0, "hits": []}
    try:
        hits = run_blastn_confirmation(
            donor_fasta,
            recipient_fasta,
            min_len=min_len,
            min_identity=min_identity,
            blastn_path=blastn_path,
            makeblastdb_path=makeblastdb_path,
        )
    except HGTBackendError as exc:
        return {
            "status": "failed",
            "supported": False,
            "hit_count": 0,
            "hits": [],
            "anomaly": exc.code,
            "error": str(exc),
        }

    supporting_hits = [hit for hit in hits if _blast_hit_overlaps_alignment(hit, alignments)]
    return {
        "status": "ok",
        "supported": bool(supporting_hits),
        "hit_count": len(supporting_hits),
        "raw_hit_count": len(hits),
        "best_identity": round(max((hit.identity for hit in supporting_hits), default=0.0), 4),
        "best_bitscore": round(max((hit.bitscore for hit in supporting_hits), default=0.0), 4),
        "hits": [hit.as_dict() for hit in supporting_hits],
    }


def _blast_hit_overlaps_alignment(hit: HGTBlastHit, alignments: list[HGTAlignment]) -> bool:
    for alignment in alignments:
        if hit.recipient_id != alignment.recipient_id:
            continue
        overlap = _interval_overlap(
            hit.recipient_start,
            hit.recipient_end,
            alignment.recipient_start,
            alignment.recipient_end,
        )
        if overlap <= 0:
            continue
        min_span = min(hit.length, alignment.length)
        if overlap / min_span >= 0.5:
            return True
    return False


def _candidate_quality(candidate_sequence: str) -> dict[str, object]:
    sequence = candidate_sequence.upper()
    length = len(sequence)
    if not sequence:
        return {
            "length": 0,
            "n_fraction": 0.0,
            "max_base_fraction": 0.0,
            "low_complexity": False,
            "n_content_risk": False,
            "repeat_or_paralogy_risk": False,
            "pass": True,
        }
    base_counts = {base: sequence.count(base) for base in "ACGT"}
    max_base_fraction = max(base_counts.values(), default=0) / length
    n_fraction = sequence.count("N") / length
    return {
        "length": length,
        "n_fraction": round(n_fraction, 4),
        "max_base_fraction": round(max_base_fraction, 4),
        "low_complexity": max_base_fraction >= 0.8,
        "n_content_risk": n_fraction > 0.05,
        "repeat_or_paralogy_risk": False,
        "pass": True,
    }


def _repeat_or_paralogy_risk(metrics: Mapping[str, Any]) -> bool:
    fragment_count = int(metrics["fragment_count"])
    if fragment_count == 0:
        return False
    blast = metrics["blast"]
    if blast.get("status") != "ok":
        return False
    hit_count = int(blast.get("hit_count", 0))
    return hit_count > max(3, fragment_count * 3)


def _normalize_phylogeny_evidence(
    evidence: Mapping[str, Any] | str | Path | None,
) -> dict[str, object]:
    if evidence is None:
        return {"status": "not_provided", "supported": False}
    if isinstance(evidence, (str, Path)):
        path = Path(evidence)
        try:
            loaded = json.loads(path.read_text())
        except OSError as exc:
            return {
                "status": "failed",
                "supported": False,
                "anomaly": "phylogeny_evidence_unreadable",
                "error": str(exc),
            }
        except json.JSONDecodeError as exc:
            return {
                "status": "failed",
                "supported": False,
                "anomaly": "phylogeny_evidence_invalid_json",
                "error": str(exc),
            }
        if not isinstance(loaded, dict):
            return {
                "status": "failed",
                "supported": False,
                "anomaly": "phylogeny_evidence_invalid",
                "error": "Phylogeny evidence JSON must contain an object.",
            }
        evidence = loaded

    normalized = dict(evidence)
    supported = bool(
        normalized.get("supported")
        or normalized.get("phylogeny_supported")
        or normalized.get("gene_tree_species_tree_conflict")
    )
    normalized["supported"] = supported
    normalized.setdefault("status", "ok" if supported else "not_supported")
    normalized.setdefault("method", "external_phylogeny_evidence")
    return normalized


def _evidence_level(metrics: Mapping[str, Any]) -> str:
    if metrics["phylogeny"]["supported"]:
        return "phylogeny_supported"
    if metrics["fragment_count"] and not metrics["quality"]["pass"]:
        return "ambiguous_candidate"
    if metrics["blast"]["supported"] and metrics["composition"]["supported"]:
        return "blast_composition_supported"
    if metrics["blast"]["supported"]:
        return "blast_supported"
    if metrics["composition"]["supported"]:
        return "composition_supported"
    if metrics["fragment_count"]:
        return "alignment_candidate"
    return "none"


def _interval_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b) + 1)


def _merge_recipient_intervals(alignments: list[HGTAlignment]) -> list[tuple[str, int, int]]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for alignment in alignments:
        grouped.setdefault(alignment.recipient_id, []).append(
            (alignment.recipient_start, alignment.recipient_end)
        )
    merged: list[tuple[str, int, int]] = []
    for recipient_id, intervals in grouped.items():
        for start, end in sorted(intervals):
            if not merged or merged[-1][0] != recipient_id or start > merged[-1][2] + 1:
                merged.append((recipient_id, start, end))
            else:
                prev_id, prev_start, prev_end = merged[-1]
                merged[-1] = (prev_id, prev_start, max(prev_end, end))
    return merged


def _gc_fraction(sequence: str) -> float:
    if not sequence:
        return 0.0
    sequence = sequence.upper()
    return (sequence.count("G") + sequence.count("C")) / len(sequence)
