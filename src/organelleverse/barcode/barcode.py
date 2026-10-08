"""DNA barcode design and species identification. Self-contained.

- design_barcode(): from a multiple-genome alignment, find highly variable
  regions suitable as barcodes (high inter-species variation, low intra).
- identify(): match a query sequence against a barcode reference DB via k-mer.
"""

from __future__ import annotations
from pathlib import Path
from ..core.result import ErrorDetail, OrganelleResult
from .._bio import read_fasta
from ._contract import (
    OPERATION_VERSION,
    RESULT_SCOPE,
    fasta_artifact,
    finding,
    input_hashes,
    make_provenance,
    utc_now,
)


def design_barcode(
    alignment_fasta: str | Path,
    *,
    min_window: int = 100,
    min_p_variability: float = 0.05,
) -> OrganelleResult:
    """Design barcode regions from a multi-species alignment.

    Scans sliding windows for regions with variability >= min_p_variability
    (fraction of polymorphic sites). Returns candidate barcode windows.

    Each candidate window becomes one canonical finding whose ``metric`` is the
    1-based ``start-end`` window and whose ``value`` is its variable-site
    fraction; the top five are reported, as before.
    """
    started_at = utc_now()
    parameters = {"min_window": min_window, "min_p_variability": min_p_variability}
    alignment_artifact = fasta_artifact(alignment_fasta, kind="alignment")
    seqs = [s.upper() for _, s in read_fasta(Path(alignment_fasta))]
    if len(seqs) < 2:
        return OrganelleResult(
            operation_id="barcode.design_barcode",
            operation_version=OPERATION_VERSION,
            scope=RESULT_SCOPE,
            status="failed",
            summary_text="design_barcode needs >=2 sequences.",
            errors=(
                ErrorDetail(
                    code="barcode.too_few_sequences",
                    message="design_barcode needs at least two aligned sequences.",
                    details={"n_sequences": len(seqs)},
                ),
            ),
            provenance=make_provenance(
                operation_id="barcode.design_barcode",
                parameters=parameters,
                input_artifact_hashes=input_hashes(alignment_artifact),
                started_at=started_at,
                finished_at=utc_now(),
            ),
        )
    if min_window < 2:
        min_window = 2
    L = min(len(s) for s in seqs)
    step = max(1, min_window // 2)
    candidates: list[dict] = []
    for start in range(0, L - min_window + 1, step):
        end = start + min_window
        var_sites = _count_variable_sites(seqs, start, end)
        p_var = var_sites / (end - start)
        if p_var >= min_p_variability:
            candidates.append(
                {
                    "start": start + 1,
                    "end": end,
                    "length": min_window,
                    "p_variable": round(p_var, 4),
                    "variable_sites": var_sites,
                }
            )
    candidates.sort(key=lambda c: c["p_variable"], reverse=True)
    return OrganelleResult(
        operation_id="barcode.design_barcode",
        operation_version=OPERATION_VERSION,
        scope=RESULT_SCOPE,
        status="ok",
        summary_text=(
            f"{len(candidates)} candidate barcode regions (>= {min_p_variability} variability)."
        ),
        metrics={
            "candidate_regions": len(candidates),
            "alignment_length": L,
            "n_sequences": len(seqs),
        },
        findings=tuple(
            finding(
                "barcode_region",
                c["p_variable"],
                metric=f"{c['start']}-{c['end']}",
                unit="fraction_variable_sites",
            )
            for c in candidates[:5]
        ),
        flags=("barcodes_found",) if candidates else (),
        provenance=make_provenance(
            operation_id="barcode.design_barcode",
            parameters=parameters,
            input_artifact_hashes=input_hashes(alignment_artifact),
            started_at=started_at,
            finished_at=utc_now(),
        ),
    )


def identify(
    query_fasta: str | Path,
    reference_fasta: str | Path,
    *,
    k: int = 31,
) -> OrganelleResult:
    """Identify a query sequence against a reference barcode DB via k-mer overlap.

    The reference FASTA should have sequence IDs matching species names.
    """
    started_at = utc_now()
    parameters = {"k": k}
    query_artifact = fasta_artifact(query_fasta, kind="query_sequence")
    reference_artifact = fasta_artifact(reference_fasta, kind="barcode_reference")
    artifact_hashes = input_hashes(query_artifact, reference_artifact)
    query = "".join(s.upper() for _, s in read_fasta(Path(query_fasta)))
    refs = read_fasta(Path(reference_fasta))
    if not query or not refs:
        return OrganelleResult(
            operation_id="barcode.identify",
            operation_version=OPERATION_VERSION,
            scope=RESULT_SCOPE,
            status="failed",
            summary_text="identify() needs non-empty query and reference.",
            errors=(
                ErrorDetail(
                    code="barcode.empty_input",
                    message="identify() needs a non-empty query and a non-empty reference set.",
                    details={"query_length": len(query), "n_references": len(refs)},
                ),
            ),
            provenance=make_provenance(
                operation_id="barcode.identify",
                parameters=parameters,
                input_artifact_hashes=artifact_hashes,
                started_at=started_at,
                finished_at=utc_now(),
            ),
        )
    query_kmers = {
        query[i : i + k] for i in range(len(query) - k + 1) if "N" not in query[i : i + k]
    }
    scores: list[tuple[str, float]] = []
    for ref_name, ref_seq in refs:
        ref_kmers = {
            ref_seq.upper()[i : i + k]
            for i in range(len(ref_seq) - k + 1)
            if "N" not in ref_seq.upper()[i : i + k]
        }
        if not query_kmers or not ref_kmers:
            continue
        overlap = len(query_kmers & ref_kmers)
        jaccard = overlap / len(query_kmers | ref_kmers)
        scores.append((ref_name, round(jaccard, 4)))
    scores.sort(key=lambda x: x[1], reverse=True)
    best_match = scores[0] if scores else ("none", 0.0)
    return OrganelleResult(
        operation_id="barcode.identify",
        operation_version=OPERATION_VERSION,
        scope=RESULT_SCOPE,
        status="ok",
        summary_text=f"Best match: {best_match[0]} (Jaccard={best_match[1]:.3f}).",
        metrics={
            "best_match": best_match[0],
            "jaccard": best_match[1],
            "n_references": len(refs),
        },
        findings=(finding("best_match", best_match[1], metric=best_match[0], unit="jaccard"),),
        flags=("identified",) if best_match[1] > 0.3 else (),
        provenance=make_provenance(
            operation_id="barcode.identify",
            parameters=parameters,
            input_artifact_hashes=artifact_hashes,
            started_at=started_at,
            finished_at=utc_now(),
        ),
    )


def _count_variable_sites(seqs: list[str], start: int, end: int) -> int:
    """Count columns with >1 base (ignoring gaps/N)."""
    var = 0
    for i in range(start, end):
        col = {s[i] for s in seqs if i < len(s) and s[i] in "ACGT"}
        if len(col) > 1:
            var += 1
    return var
