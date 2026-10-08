"""Strand-aware reference breadth; plastome quadrants are independent queries."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from organelleverse._bio import read_fasta
from organelleverse.comparative.identity import _require_mappy
from organelleverse.comparative.orientation import (
    _segment,
    normalize_plastome_orientation,
)
from organelleverse.core.errors import OrganelleInputError, OrganelleParameterError
from organelleverse.core.result import OrganelleResult


def _read(path: Path) -> list[tuple[str, str]]:
    try:
        records = [(name, seq.upper()) for name, seq in read_fasta(path)]
    except (OSError, ValueError) as exc:
        raise OrganelleInputError(code="qc.reference.invalid_fasta", message=str(exc)) from exc
    if any(not seq or set(seq) - set("ACGTN") for _, seq in records):
        raise OrganelleInputError(
            code="qc.reference.invalid_dna", message="Nonempty ungapped ACGTN DNA is required."
        )
    return records


def _union(intervals: list[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def _circular_interval(start: int, end: int, length: int) -> list[list[int]]:
    """Split a forward-coordinate interval that crosses the input origin."""
    size = end - start
    start %= length
    end = start + size
    return [[start, end]] if end <= length else [[start, length], [0, end - length]]


def _compare(aligner, sequence: str, reference_length: int, regions=None) -> dict:
    regions = regions or {"genome": {"start": 0, "length": len(sequence)}}
    blocks = []
    for region, coordinates in regions.items():
        for hit in aligner.map(_segment(sequence, coordinates)):
            # Keep both strands and primary/secondary hits. A repeat's reverse
            # secondary hit is evidence, not an asserted biological inversion.
            blocks.append(
                {
                    "region": region,
                    "query_intervals": _circular_interval(
                        coordinates["start"] + hit.q_st,
                        coordinates["start"] + hit.q_en,
                        len(sequence),
                    ),
                    "reference_start": hit.r_st,
                    "reference_end": hit.r_en,
                    "strand": hit.strand,
                    "is_primary": bool(hit.is_primary),
                    "matches": hit.mlen,
                    "alignment_length": hit.blen,
                }
            )
    covered = _union([[b["reference_start"], b["reference_end"]] for b in blocks])
    bases = sum(end - start for start, end in covered)
    matches = sum(b["matches"] for b in blocks)
    length = sum(b["alignment_length"] for b in blocks)
    return {
        "reference_length": reference_length,
        "reference_covered_bases": bases,
        "reference_coverage": bases / reference_length,
        "reference_intervals": covered,
        "matches": matches,
        "alignment_length": length,
        "alignment_identity": matches / length if length else None,
        "alignment_blocks": blocks,
        "inverted_blocks": [b for b in blocks if b["strand"] == -1],
    }


def compare_assembly_to_reference(
    input_fasta: str | Path,
    reference_fasta: str | Path,
    *,
    organelle: Literal["plastid", "generic"] = "generic",
) -> OrganelleResult:
    """Compare independent candidate sequences to one FASTA reference (asm5).

    Each input record is a separate candidate, NOT a contig to pool with other
    records. ``raw`` and ``normalized`` use all emitted forward/reverse blocks;
    coverage is their reference interval union. Identity is sum(mlen)/sum(blen),
    an alignment-column weighted statistic including secondary/repeat hits,
    not one-to-one genome identity. No hits means zero breadth and null identity.

    Plastids reuse circular IR detection and orientation normalization. Their
    four quadrants map independently to the entire reference, before and after
    normalization, so chaining across IRs cannot suppress an inverted SSC.
    Whole-sequence summaries are also retained to expose that suppression.
    Generic comparisons do not rotate or normalize: normalized is null.

    All coordinates are zero-based, half-open in the corresponding raw or
    normalized sequence. Wrapped query intervals are split. ``inverted_blocks``
    means reverse alignment evidence, including ambiguous IR cross-mappings;
    it does not assert a structural variant. Unresolved plastids retain raw
    whole-sequence evidence but have null raw/normalized quadrant comparisons
    and explicit warning/failed status. No output destination is accepted.
    """
    if organelle not in {"plastid", "generic"}:
        raise OrganelleParameterError(
            code="qc.reference.organelle", message="organelle must be plastid or generic."
        )
    mp = _require_mappy()
    candidates = _read(Path(input_fasta))
    reference_records = _read(Path(reference_fasta))
    if len(reference_records) != 1:
        raise OrganelleInputError(
            code="qc.reference.reference_count", message="Exactly one reference record is required."
        )
    reference_name, reference = reference_records[0]
    raw_aligner = mp.Aligner(seq=reference, preset="asm5")
    normalization = reference_normalization = None
    normalized_sequences, transformations = {}, {}
    normalized_reference = None
    normalized_aligner = None
    if organelle == "plastid":
        normalization = normalize_plastome_orientation(input_fasta, reference_fasta=reference_fasta)
        reference_normalization = normalize_plastome_orientation(
            reference_fasta, reference_fasta=reference_fasta
        )
        normalized_sequences = dict(normalization.metrics["sequences"])
        transformations = {s["sample"]: s for s in normalization.metrics["samples"]}
        if reference_normalization.metrics["sequences"]:
            normalized_reference = reference_normalization.metrics["sequences"][0][1]
            normalized_aligner = mp.Aligner(seq=normalized_reference, preset="asm5")
    comparisons = []
    for name, sequence in candidates:
        whole_raw = _compare(raw_aligner, sequence, len(reference))
        transform = transformations.get(name)
        raw, normalized, whole_normalized = whole_raw, None, None
        if organelle == "plastid":
            raw = None
            if transform is not None and normalized_aligner is not None:
                raw = _compare(raw_aligner, sequence, len(reference), transform["input_regions"])
                normalized = _compare(
                    normalized_aligner,
                    normalized_sequences[name],
                    len(normalized_reference),
                    transform["regions"],
                )
                whole_normalized = _compare(
                    normalized_aligner, normalized_sequences[name], len(normalized_reference)
                )
        comparisons.append(
            {
                "sample": name,
                "query_length": len(sequence),
                "raw": raw,
                "normalized": normalized,
                "whole_sequence_raw": whole_raw,
                "whole_sequence_normalized": whole_normalized,
                "normalization": transform,
            }
        )
    status = normalization.status if normalization is not None else "ok"
    unresolved = normalization.metrics["unresolved"] if normalization is not None else []
    errors = normalization.errors if normalization is not None else ()
    if reference_normalization is not None and reference_normalization.status == "failed":
        status = "failed"
        errors = reference_normalization.errors
    if status == "ok" and any(c["raw"]["alignment_identity"] is None for c in comparisons):
        status = "warning"
    return OrganelleResult(
        operation_id="qc.compare_assembly_to_reference",
        scope="plastid" if organelle == "plastid" else "none",
        status=status,
        summary_text=f"Compared {len(comparisons)} candidates to {reference_name} ({organelle}).",
        metrics={
            "reference": reference_name,
            "preset": "asm5",
            "comparison_unit": "quadrants" if organelle == "plastid" else "whole_sequence",
            "identity_definition": "sum(matches)/sum(alignment_length), including secondary hits",
            "comparisons": comparisons,
            "unresolved": unresolved,
            "reference_normalization": (
                reference_normalization.metrics["samples"][0]
                if reference_normalization is not None
                and reference_normalization.metrics["samples"]
                else None
            ),
        },
        errors=tuple(errors),
    )
