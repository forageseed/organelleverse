"""Circular plastome orientation using the existing IR detector and mappy anchors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Bio.Seq import reverse_complement

from .._bio import read_fasta
from ..core.errors import OrganelleDependencyError, OrganelleInputError
from ..core.result import ErrorDetail, OrganelleResult
from ..ir_boundary.ir_boundary import _detect_ir_from_sequence
from .compare import _provenance


def _quadripartite(sequence: str) -> dict[str, dict[str, int]] | None:
    """Use circular IR detection so an IR crossing the origin is complete.

    Coordinates are zero-based start plus length, modulo the input length.
    The longer of the two single-copy arcs is LSC. IRb follows LSC in the
    input strand. No new repeat detector or boundary refinement is used.
    """
    n = len(sequence)
    pair = _detect_ir_from_sequence(sequence, circular=True)
    if pair is None:
        return None
    arms = sorted((r["start"] % n, r["end"] - r["start"]) for r in pair)
    (a, la), (b, lb) = arms
    gap_ab, gap_ba = b - a - la, n + a - b - lb
    if min(gap_ab, gap_ba) <= 0 or gap_ab == gap_ba:
        return None
    if gap_ab > gap_ba:
        a, la, b, lb = b, lb, a, la
        gap_ab, gap_ba = gap_ba, gap_ab
    return {
        "LSC": {"start": (b + lb) % n, "length": gap_ba},
        "IRb": {"start": a, "length": la},
        "SSC": {"start": (a + la) % n, "length": gap_ab},
        "IRa": {"start": b, "length": lb},
    }


def _segment(sequence: str, region: dict[str, int]) -> str:
    start = region["start"]
    return (sequence * 2)[start : start + region["length"]]


def _strand(aligner: Any, sequence: str) -> int | None:
    """Primary anchors must agree on strand; absent/mixed evidence is unresolved."""
    strands = {hit.strand for hit in aligner.map(sequence) if hit.is_primary}
    return next(iter(strands)) if len(strands) == 1 else None


def _read_plastomes(path):
    try:
        records = [(name, seq.upper()) for name, seq in read_fasta(Path(path))]
    except ValueError as exc:
        raise OrganelleInputError(
            code="comparative.orientation.invalid_fasta", message=str(exc)
        ) from exc
    if any(not seq or set(seq) - set("ACGTN") for _, seq in records):
        raise OrganelleInputError(
            code="comparative.orientation.ungapped_dna_required",
            message="Orientation requires nonempty ungapped ACGTN sequences.",
        )
    return records


def normalize_plastome_orientation(
    input_fasta: str | Path,
    *,
    reference_fasta: str | Path | None = None,
) -> OrganelleResult:
    """Normalize ungapped circular plastomes to LSC/IRb/SSC/IRa.

    With a reference, both single-copy strands follow its input orientation.
    Otherwise each region follows the majority strand relative to the first
    identifiable sample (ties keep that sample's strand). Samples without a
    quadripartite structure or unambiguous primary LSC/SSC anchors are listed
    in ``unresolved`` and are NOT included in ``sequences``. A failed reference
    makes the operation fail; a partially resolved cohort returns ``warning``.

    ``operations`` are replayable in order: rotate left, optionally reverse
    complement the genome and rotate back to LSC, then optionally reverse
    complement SSC. Region coordinates use zero-based start and length;
    original regions may wrap the origin. Boundary precision is that of the
    existing seed/extension IR detector, not annotation-exact junctions.
    """
    try:
        import mappy
    except ImportError as exc:
        raise OrganelleDependencyError(
            code="dependency.mappy",
            message="Plastome orientation requires mappy; install 'organelleverse[align]' "
            "on Linux, macOS or WSL (mappy has no Windows wheel).",
        ) from exc
    records = _read_plastomes(input_fasta)
    structures = {name: _quadripartite(seq) for name, seq in records}
    unresolved = [
        {"sample": name, "reason": "quadripartite_not_identified"}
        for name, _ in records
        if structures[name] is None
    ]
    valid = [(name, seq) for name, seq in records if structures[name] is not None]
    parameters = {
        "input_fasta": str(input_fasta),
        "reference_fasta": str(reference_fasta) if reference_fasta else None,
    }
    reference_name = None
    reference_regions = None
    if reference_fasta is not None:
        reference_records = _read_plastomes(reference_fasta)
        if len(reference_records) != 1:
            raise OrganelleInputError(
                code="comparative.orientation.reference_count",
                message="reference_fasta must contain exactly one plastome.",
            )
        reference_name, reference = reference_records[0]
        reference_regions = _quadripartite(reference)
    elif valid:
        reference_name, reference = valid[0]
        reference_regions = structures[reference_name]
    if reference_regions is None:
        return OrganelleResult(
            operation_id="comparative.normalize_plastome_orientation",
            scope="plastid",
            status="failed",
            summary_text="No identifiable quadripartite reference.",
            metrics={"sequences": [], "samples": [], "unresolved": unresolved},
            errors=(
                ErrorDetail(
                    code="comparative.orientation.no_reference",
                    message="No identifiable quadripartite reference.",
                ),
            ),
            provenance=_provenance("normalize_plastome_orientation", parameters).evolve(
                actual_backend="mappy", attempted_backends=("mappy",)
            ),
        )
    aligners = {
        region: mappy.Aligner(seq=_segment(reference, reference_regions[region]), preset="asm5")
        for region in ("LSC", "SSC")
    }
    strands = {}
    for name, seq in valid:
        regions = structures[name]
        evidence = {
            region: _strand(aligners[region], _segment(seq, regions[region]))
            for region in ("LSC", "SSC")
        }
        if None in evidence.values():
            unresolved.append(
                {
                    "sample": name,
                    "reason": "orientation_anchors_unresolved",
                    "anchor_strands": evidence,
                }
            )
        else:
            strands[name] = evidence
    target = {
        region: (
            1
            if reference_fasta is not None or sum(s[region] for s in strands.values()) >= 0
            else -1
        )
        for region in ("LSC", "SSC")
    }
    samples, sequences = [], []
    for name, seq in valid:
        if name not in strands:
            continue
        regions = structures[name]
        offset = regions["LSC"]["start"]
        normalized = seq[offset:] + seq[:offset]
        operations = [{"operation": "rotate_left", "bases": offset}]
        lengths = {region: r["length"] for region, r in regions.items()}
        flip_lsc = strands[name]["LSC"] != target["LSC"]
        flip_ssc_net = strands[name]["SSC"] != target["SSC"]
        if flip_lsc:
            normalized = reverse_complement(normalized)
            rotation = len(seq) - lengths["LSC"]
            normalized = normalized[rotation:] + normalized[:rotation]
            lengths["IRa"], lengths["IRb"] = lengths["IRb"], lengths["IRa"]
            operations.extend(
                [
                    {
                        "operation": "reverse_complement",
                        "region": "genome",
                        "start": 0,
                        "end": len(seq),
                    },
                    {"operation": "rotate_left", "bases": rotation},
                ]
            )
        ssc_start = lengths["LSC"] + lengths["IRb"]
        ssc_end = ssc_start + lengths["SSC"]
        if flip_ssc_net != flip_lsc:
            normalized = (
                normalized[:ssc_start]
                + reverse_complement(normalized[ssc_start:ssc_end])
                + normalized[ssc_end:]
            )
            operations.append(
                {
                    "operation": "reverse_complement",
                    "region": "SSC",
                    "start": ssc_start,
                    "end": ssc_end,
                }
            )
        output_regions, start = {}, 0
        for region, length in lengths.items():
            output_regions[region] = {"start": start, "length": length}
            start += length
        sequences.append([name, normalized])
        samples.append(
            {
                "sample": name,
                "input_regions": regions,
                "regions": output_regions,
                "anchor_strands": strands[name],
                "rotation": offset,
                "lsc_reverse_complemented": flip_lsc,
                "ssc_reverse_complemented": flip_ssc_net,
                "operations": operations,
            }
        )
    status = "ok" if not unresolved else ("warning" if sequences else "failed")
    return OrganelleResult(
        operation_id="comparative.normalize_plastome_orientation",
        scope="plastid",
        status=status,
        summary_text=f"Normalized {len(sequences)} plastomes; {len(unresolved)} unresolved.",
        metrics={
            "sequences": sequences,
            "samples": samples,
            "unresolved": unresolved,
            "reference": reference_name,
            "orientation_policy": "reference" if reference_fasta else "majority",
            "target_strands": target,
        },
        flags=("unresolved_samples",) if unresolved else (),
        errors=(
            ErrorDetail(
                code="comparative.orientation.no_resolved_samples",
                message="No samples have unambiguous single-copy anchors.",
            ),
        )
        if status == "failed"
        else (),
        provenance=_provenance("normalize_plastome_orientation", parameters).evolve(
            actual_backend="mappy", attempted_backends=("mappy",)
        ),
    )
