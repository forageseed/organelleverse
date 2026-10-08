"""Molecular-marker primer design on hypervariable regions.

:func:`design_marker_primers` implements the second half of the classic
DnaSP → Primer3 workflow used by organelle phylogeography papers: take the
mutation hotspots (高变区) of a multi-sequence alignment, and design primer
pairs in the *conserved flanks* around each hotspot with the Primer3 C
library (``primer3-py`` bindings, not the external ``primer3_core``
program). Every returned pair is filtered for cross-sample conservation —
the binding site must anneal to every sample with at most
``max_primer_mismatches`` mismatches and zero mismatches in the 3' last five
bases — and is scored by the π and haplotype resolution of its amplicon.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._bio import read_fasta, reverse_complement
from ..core.errors import OrganelleDependencyError
from ..core.result import ErrorDetail, OrganelleResult
from ..diversity.diversity import (
    _call_hotspots,
    _columns_between,
    _sliding_window_stats,
    _window_pi_s,
)
from ._contract import (
    OPERATION_VERSION,
    RESULT_SCOPE,
    finding,
    make_provenance,
)

_THREE_PRIME_GUARD = 5
_PRIMER3_GLOBAL_ARGS: dict[str, Any] = {
    "PRIMER_MIN_SIZE": 18,
    "PRIMER_OPT_SIZE": 20,
    "PRIMER_MAX_SIZE": 25,
    "PRIMER_MIN_TM": 57.0,
    "PRIMER_OPT_TM": 60.0,
    "PRIMER_MAX_TM": 63.0,
    "PRIMER_MIN_GC": 20.0,
    "PRIMER_MAX_GC": 80.0,
}


def _load_primer3():
    """Import primer3-py, failing closed with an install hint when absent."""
    try:
        import primer3  # type: ignore
    except ImportError as error:
        raise OrganelleDependencyError(
            code="barcode.design_marker_primers.missing_primer3",
            message=(
                "design_marker_primers needs the primer3-py package (Primer3 C "
                "library bindings). Install it with: pip install primer3-py"
            ),
            details={"package": "primer3-py", "import_name": "primer3"},
        ) from error
    return primer3


def design_marker_primers(
    alignment_fasta: str | Path,
    *,
    regions: list[dict[str, int]] | None = None,
    reference_index: int = 0,
    window_size: int = 600,
    step: int = 200,
    gap_mode: str = "exclude",
    pi_threshold: float | None = 0.03,
    top_fraction: float = 0.05,
    product_size_min: int = 300,
    product_size_max: int = 800,
    primer_num_return: int = 10,
    max_primer_mismatches: int = 2,
    max_candidates: int = 5,
) -> OrganelleResult:
    """Design conservation-checked marker primers around 高变区 of an alignment.

    Hotspot regions come either from ``regions`` (a list of ``{"start",
    "end"}`` 1-based inclusive alignment coordinates) or, when ``regions``
    is None, from the same sliding-window π hotspot calling as
    :func:`organelleverse.diversity.diversity.sliding_window_diversity`
    (DnaSP defaults: 600 bp window, 200 bp step; a window is a hotspot when
    its π ranks in the top ``top_fraction`` or π ≥ ``pi_threshold``).

    For each region, primer3 designs pairs on the reference sequence
    (``reference_index``) whose amplicon covers the region, with product
    size in ``[product_size_min, product_size_max]`` (default 300-800 bp,
    the common organelle-marker range). Auto-discovered hotspots wider than
    ``product_size_max`` are targeted at their peak π window instead, so
    the most variable part still lands inside the amplicon; regions that
    cannot fit at all are reported in ``skipped_regions``. Pairs are then
    filtered for cross-sample conservation: a primer must match every
    sample over its binding site with at most ``max_primer_mismatches``
    mismatches (gaps/N count as mismatches) and with zero mismatches in the
    3' last five bases.
    The lowest-penalty surviving pair becomes the region's candidate; its
    amplicon is reported with π, segregating-site count, distinct haplotype
    count and the number of samples uniquely distinguished.
    """
    parameters = {
        "alignment_fasta": str(alignment_fasta),
        "regions": regions,
        "reference_index": reference_index,
        "window_size": window_size,
        "step": step,
        "gap_mode": gap_mode,
        "pi_threshold": pi_threshold,
        "top_fraction": top_fraction,
        "product_size_min": product_size_min,
        "product_size_max": product_size_max,
        "primer_num_return": primer_num_return,
        "max_primer_mismatches": max_primer_mismatches,
        "max_candidates": max_candidates,
    }
    seqs = [s.upper() for _, s in read_fasta(Path(alignment_fasta))]
    names = [name for name, _ in read_fasta(Path(alignment_fasta))]
    if len(seqs) < 2:
        return OrganelleResult(
            operation_id="barcode.design_marker_primers",
            operation_version=OPERATION_VERSION,
            scope=RESULT_SCOPE,
            status="failed",
            summary_text="design_marker_primers needs ≥2 aligned sequences.",
            errors=(
                ErrorDetail(
                    code="barcode.design_marker_primers.too_few_sequences",
                    message="design_marker_primers needs at least two aligned sequences.",
                    details={"n_sequences": len(seqs)},
                ),
            ),
            provenance=make_provenance(
                operation_id="barcode.design_marker_primers",
                parameters=parameters,
            ),
        )
    if not (0 <= reference_index < len(seqs)):
        return OrganelleResult(
            operation_id="barcode.design_marker_primers",
            operation_version=OPERATION_VERSION,
            scope=RESULT_SCOPE,
            status="failed",
            summary_text="reference_index out of range for this alignment.",
            errors=(
                ErrorDetail(
                    code="barcode.design_marker_primers.bad_reference",
                    message="reference_index out of range for this alignment.",
                    details={"reference_index": reference_index, "n_sequences": len(seqs)},
                ),
            ),
            provenance=make_provenance(
                operation_id="barcode.design_marker_primers",
                parameters=parameters,
            ),
        )
    primer3 = _load_primer3()

    L = min(len(s) for s in seqs)
    seqs = [s[:L] for s in seqs]
    ref = seqs[reference_index]

    if regions is None:
        windows = _sliding_window_stats(seqs, ref, window_size, step, gap_mode, [])
        hotspots = _call_hotspots(
            windows, seqs, ref, gap_mode, pi_threshold, top_fraction, []
        )
        targets = []
        for h in hotspots:
            # primer3's SEQUENCE_TARGET is the region's peak-π window: the
            # amplicon must cover the hypervariable core; the merged span's
            # moderate-variable flanks may fall inside or outside the product.
            peak = {
                "start": h["start"],
                "end": h["end"],
                "region_pi": h["pi"],
                "target_start": h["peak_window_start"],
                "target_end": h["peak_window_end"],
            }
            if h["end"] - h["start"] + 1 <= product_size_max:
                targets.append(peak)
                continue
            # Oversized hotspot: target only its peak window.
            peak["source_region"] = [h["start"], h["end"]]
            peak["start"], peak["end"] = h["peak_window_start"], h["peak_window_end"]
            if h["peak_window_end"] - h["peak_window_start"] + 1 <= product_size_max:
                targets.append(peak)
            else:
                targets.append({**peak, "oversized": True})
    else:
        targets = []
        for raw in regions:
            start, end = int(raw["start"]), int(raw["end"])
            if start < 1 or end > L or start > end:
                targets.append({"start": start, "end": end, "invalid": True})
            elif end - start + 1 > product_size_max:
                targets.append({"start": start, "end": end, "oversized": True})
            else:
                targets.append({"start": start, "end": end})

    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for target in targets:
        if target.get("invalid"):
            skipped.append(
                {
                    "start": target["start"],
                    "end": target["end"],
                    "reason": "region outside alignment coordinates",
                }
            )
            continue
        if target.get("oversized"):
            skipped.append(
                {
                    "start": target["start"],
                    "end": target["end"],
                    "reason": "region larger than max product size",
                }
            )
            continue
        if len(candidates) >= max_candidates:
            break
        candidate = _design_for_region(
            primer3,
            seqs,
            names,
            ref,
            target["start"],
            target["end"],
            target.get("target_start", target["start"]),
            target.get("target_end", target["end"]),
            product_size_min=product_size_min,
            product_size_max=product_size_max,
            primer_num_return=primer_num_return,
            max_primer_mismatches=max_primer_mismatches,
            gap_mode=gap_mode,
        )
        if candidate is None:
            skipped.append(
                {
                    "start": target["start"],
                    "end": target["end"],
                    "reason": "no primer pair passed conservation filters",
                }
            )
        else:
            if "region_pi" in target:
                candidate["region_pi"] = target["region_pi"]
            if "source_region" in target:
                candidate["source_region"] = target["source_region"]
            candidates.append(candidate)

    return OrganelleResult(
        operation_id="barcode.design_marker_primers",
        operation_version=OPERATION_VERSION,
        scope=RESULT_SCOPE,
        status="ok",
        summary_text=(
            f"{len(candidates)} marker primer candidate(s) from "
            f"{len(targets)} hotspot region(s); {len(skipped)} region(s) skipped."
        ),
        metrics={
            "n_regions": len(targets),
            "n_candidates": len(candidates),
            "n_skipped": len(skipped),
            "reference": names[reference_index] if names else str(reference_index),
            "product_size_range": [product_size_min, product_size_max],
            "max_primer_mismatches": max_primer_mismatches,
            "candidates": candidates,
            "skipped_regions": skipped,
        },
        findings=tuple(
            finding(
                "barcode.design_marker_primers.candidate",
                c["amplicon_pi"],
                metric=f"{c['amplicon_start']}-{c['amplicon_end']}",
                unit="amplicon_pi",
            )
            for c in candidates[:5]
        ),
        flags=("markers_found",) if candidates else (),
        provenance=make_provenance(
            operation_id="barcode.design_marker_primers",
            parameters=parameters,
        ),
    )


def _design_for_region(
    primer3: Any,
    seqs: list[str],
    names: list[str],
    ref: str,
    region_start: int,
    region_end: int,
    target_start: int,
    target_end: int,
    *,
    product_size_min: int,
    product_size_max: int,
    primer_num_return: int,
    max_primer_mismatches: int,
    gap_mode: str,
) -> dict[str, Any] | None:
    """Best conservation-passing primer pair for one 1-based inclusive region.

    ``region_start``/``region_end`` identify the hotspot for reporting;
    ``target_start``/``target_end`` (within it) is what the amplicon must
    cover — for auto-discovered hotspots this is the peak-π window.
    """
    L = len(ref)
    # Template: reference slice with product_size_max flank on each side, gaps stripped.
    a0 = max(0, region_start - 1 - product_size_max)
    a1 = min(L, region_end + product_size_max)
    aln_positions = [i for i in range(a0, a1) if ref[i] in "ACGT"]
    if not aln_positions:
        return None
    template = "".join(ref[i] for i in aln_positions)
    rank_of_aln = {aln: rank for rank, aln in enumerate(aln_positions)}

    target_cols = [i for i in range(target_start - 1, target_end) if i in rank_of_aln]
    if not target_cols:
        return None
    t_start = rank_of_aln[target_cols[0]]
    t_end = rank_of_aln[target_cols[-1]]
    target_span = t_end - t_start + 1
    size_lo = max(product_size_min, target_span)
    if size_lo > product_size_max:
        return None

    seq_args = {
        "SEQUENCE_ID": f"region_{region_start}_{region_end}",
        "SEQUENCE_TEMPLATE": template,
        "SEQUENCE_TARGET": [t_start, target_span],
    }
    global_args = dict(_PRIMER3_GLOBAL_ARGS)
    global_args["PRIMER_NUM_RETURN"] = max(1, primer_num_return)
    global_args["PRIMER_PRODUCT_SIZE_RANGE"] = [[size_lo, product_size_max]]
    result = primer3.bindings.design_primers(seq_args, global_args)
    n_pairs = int(result.get("PRIMER_PAIR_NUM_RETURNED", 0))
    for i in range(n_pairs):
        fwd_pos, fwd_len = (int(v) for v in result[f"PRIMER_LEFT_{i}"])
        rev_pos, rev_len = (int(v) for v in result[f"PRIMER_RIGHT_{i}"])
        fwd_seq = str(result[f"PRIMER_LEFT_{i}_SEQUENCE"]).upper()
        rev_seq = str(result[f"PRIMER_RIGHT_{i}_SEQUENCE"]).upper()
        fwd_cols = aln_positions[fwd_pos : fwd_pos + fwd_len]
        rev_cols = aln_positions[rev_pos - rev_len + 1 : rev_pos + 1]
        fwd_mm, fwd_prime = _sample_mismatches(seqs, names, fwd_cols, fwd_seq)
        rev_mm, rev_prime = _sample_mismatches(
            seqs, names, rev_cols, rev_seq, revcomp=True
        )
        if not _conserved(fwd_mm, fwd_prime, max_primer_mismatches) or not _conserved(
            rev_mm, rev_prime, max_primer_mismatches
        ):
            continue
        amplicon_start = fwd_cols[0] + 1
        amplicon_end = rev_cols[-1] + 1
        columns = _columns_between(seqs, amplicon_start - 1, amplicon_end, gap_mode)
        pi, seg, _comp = _window_pi_s(columns)
        haplotypes = [
            "".join(s[i] for i in range(amplicon_start - 1, amplicon_end)) for s in seqs
        ]
        distinct = len(set(haplotypes))
        uniquely = sum(haplotypes.count(h) == 1 for h in haplotypes)
        return {
            "region_start": region_start,
            "region_end": region_end,
            "product_size": rev_pos + 1 - fwd_pos,
            "amplicon_start": amplicon_start,
            "amplicon_end": amplicon_end,
            "amplicon_pi": round(pi, 6),
            "amplicon_segregating_sites": seg,
            "distinct_haplotypes": distinct,
            "samples_uniquely_distinguished": uniquely,
            "n_samples": len(seqs),
            "penalty": round(float(result[f"PRIMER_PAIR_{i}_PENALTY"]), 6),
            "forward": _primer_record(result, i, "LEFT", fwd_seq, fwd_cols, fwd_mm),
            "reverse": _primer_record(result, i, "RIGHT", rev_seq, rev_cols, rev_mm),
        }
    return None


def _sample_mismatches(
    seqs: list[str],
    names: list[str],
    aln_cols: list[int],
    primer_seq: str,
    *,
    revcomp: bool = False,
) -> tuple[dict[str, int], dict[str, int]]:
    """Per-sample (total, 3') mismatch counts of a primer against its span.

    Sample bases are read at ``aln_cols`` (alignment orientation); with
    ``revcomp=True`` the span is reverse-complemented first so both primer
    orientations end up compared 5'→3' with the 3' end at the last index.
    Gaps/N in a sample count as mismatches.
    """
    total_mm: dict[str, int] = {}
    prime_mm: dict[str, int] = {}
    for name, seq in zip(names, seqs, strict=True):
        span = "".join(
            base if base in "ACGT" else "N"
            for base in (seq[col] if col < len(seq) else "N" for col in aln_cols)
        )
        if revcomp:
            span = reverse_complement(span)
        total = 0
        prime = 0
        for offset, base in enumerate(span):
            if base not in "ACGT" or base != primer_seq[offset]:
                total += 1
                if offset >= len(span) - _THREE_PRIME_GUARD:
                    prime += 1
        total_mm[name] = total
        prime_mm[name] = prime
    return total_mm, prime_mm


def _conserved(
    total_mm: dict[str, int], prime_mm: dict[str, int], max_mismatches: int
) -> bool:
    """All samples within ``max_mismatches`` total and zero 3'-end mismatches."""
    return all(
        total_mm[name] <= max_mismatches and prime_mm[name] == 0
        for name in total_mm
    )


def _primer_record(
    result: dict[str, Any],
    index: int,
    side: str,
    sequence: str,
    aln_cols: list[int],
    mismatches: dict[str, int],
) -> dict[str, Any]:
    """One primer's report record (Tm/GC from primer3, conservation stats)."""
    return {
        "sequence": sequence,
        "tm": round(float(result[f"PRIMER_{side}_{index}_TM"]), 6),
        "gc_percent": round(float(result[f"PRIMER_{side}_{index}_GC_PERCENT"]), 6),
        "aln_start": aln_cols[0] + 1,
        "aln_end": aln_cols[-1] + 1,
        "max_mismatches": max(mismatches.values()),
        "mismatches_by_sample": mismatches,
    }


__all__ = ["design_marker_primers"]
