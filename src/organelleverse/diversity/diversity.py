"""Nucleotide diversity (π), Watterson's θ_W, Tajima's D — standard estimators.

Backend selection:
- **scikit-allel** (preferred, the community-standard population-genetics
  library; same estimators as VCFtools/dadi): ``allel.sequence_diversity``,
  ``allel.watterson_theta``, ``allel.tajima_d``.
- **pure-Python fallback** (when scikit-allel is not installed): implements
  the same estimators directly from the standard formulas (Nei 1987 for π,
  Watterson 1975 for θ_W, Tajima 1989 for D).

Inputs:
- a multi-sequence FASTA alignment (per-site column allele frequencies), OR
- a VCF (biallelic SNPs over a genome span) — reuses ``ov.io.read_vcf``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from .._bio import read_fasta, read_genbank
from ..core.errors import OrganelleDependencyError, OrganelleInputError
from ..core.frozen import thaw_json
from ..core.provenance import ResultProvenance
from ..core.result import ErrorDetail, Finding, OrganelleResult

_OPERATION_VERSION = "1.0"
_SCOPE = "mitochondrion"


# ---------------------------------------------------------------------------
# Canonical contract helpers
# ---------------------------------------------------------------------------


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
    """Build canonical provenance for one ``diversity.*`` operation."""
    return ResultProvenance(
        operation_id=f"diversity.{operation}",
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
        operation_id=f"diversity.{operation}",
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


# ---------------------------------------------------------------------------
# Internal: alignment → per-site allele counts
# ---------------------------------------------------------------------------


def _alignment_length(seqs: list[str]) -> int:
    """Require one alignment width rather than truncating records."""
    lengths = {len(sequence) for sequence in seqs}
    if len(lengths) > 1:
        raise ValueError(
            "Aligned FASTA records must have equal lengths; align the sequences first."
        )
    return next(iter(lengths), 0)


def _comparable_mask(seqs: list[str]) -> list[bool]:
    """Preserve alignment coordinates while marking columns with >=2 ACGT calls."""
    return [sum(base in "ACGT" for base in column) >= 2 for column in zip(*seqs, strict=True)]


def _alignment_stats(seqs: list[str]) -> tuple[list[int], list[tuple[int, int]], int]:
    """From a list of equal-length aligned sequences, return per-site info.

    Returns ``(segregating_positions, allele_counts, n_comparable)``:
    - ``segregating_positions``: 1-based positions that are polymorphic.
    - ``allele_counts``: per-site tuple of counts PER ALLELE (descending) -
      multiallelic-capable. (An earlier biallelic ref/alt encoding lumped
      3rd+ alleles into one "alt" and understated pi by ~12% on deep
      alignments where a third of segregating columns are triallelic+.)
    - ``n_comparable``: number of ACGT columns (sites with ≥2 called bases).
    """
    L = _alignment_length(seqs)
    seg_pos: list[int] = []
    alt_counts: list[tuple[int, int]] = []
    n_comparable = 0
    for i in range(L):
        col = [s[i] for s in seqs if s[i] in "ACGT"]
        if len(col) < 2:
            continue
        n_comparable += 1
        bases = set(col)
        if len(bases) > 1:
            seg_pos.append(i + 1)
            alt_counts.append(tuple(sorted((col.count(b) for b in bases), reverse=True)))
    return seg_pos, alt_counts, n_comparable


def _harmonic_a(n: int) -> float:
    """Watterson's a_n = Σ_{i=1}^{n-1} 1/i (the (n-1)th harmonic number)."""
    return sum(1.0 / i for i in range(1, n))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def nucleotide_diversity(
    alignment_fasta: str | Path,
    *,
    window_size: int = 1000,
    step: int = 500,
) -> OrganelleResult:
    """Compute nucleotide diversity π from a multi-sequence FASTA alignment.

    π is the average number of pairwise nucleotide differences per site
    (Nei 1987): per-site unbiased heterozygosity n/(n-1)·(1 - Σ p_i²) over
    ALL alleles, averaged over comparable ACGT columns (multiallelic-aware;
    the pure-Python windowed path counts differing pairs directly and agrees
    exactly on gap-free data).
    """
    parameters = {
        "alignment_fasta": str(alignment_fasta),
        "window_size": window_size,
        "step": step,
    }
    seqs = [s for _, s in read_fasta(Path(alignment_fasta))]
    if len(seqs) < 2:
        return _failed(
            "nucleotide_diversity",
            code="diversity.nucleotide_diversity.too_few_sequences",
            message="π requires ≥2 aligned sequences.",
            parameters=parameters,
        )
    try:
        L = _alignment_length(seqs)
    except ValueError as exc:
        return _failed(
            "nucleotide_diversity",
            code="diversity.nucleotide_diversity.unequal_alignment_lengths",
            message=str(exc),
            parameters=parameters,
        )
    seqs = [s.upper() for s in seqs]
    n = len(seqs)

    seg_pos, alt_counts, n_comp = _alignment_stats(seqs)
    method = "organelleverse_multiallelic"
    pi = _pi_skitallel(alt_counts, seg_pos, n_comp) if seg_pos else 0.0

    # windowed π (always pure-Python; small and clear)
    windows = []
    for start in range(0, max(1, L - window_size + 1), step):
        end = min(start + window_size, L)
        w_pi = _pi_pure([s[start:end] for s in seqs])
        windows.append({"start": start + 1, "end": end, "pi": round(w_pi, 6)})

    return OrganelleResult(
        operation_id="diversity.nucleotide_diversity",
        operation_version=_OPERATION_VERSION,
        scope=_SCOPE,
        status="ok",
        summary_text=f"π = {pi:.6f} ({n} sequences, {method}).",
        metrics={
            "pi": round(pi, 6),
            "n_sequences": n,
            "windows": len(windows),
            "windowed_pi": windows,
        },
        findings=(
            Finding(code="diversity.nucleotide_diversity.pi", metric="pi", value=round(pi, 6)),
            Finding(code="diversity.nucleotide_diversity.method", metric="method", value=method),
        ),
        flags=(),
        artifacts=(),
        provenance=_provenance("nucleotide_diversity", parameters=parameters, method=method),
    )


def sliding_window_diversity(
    alignment_fasta: str | Path,
    *,
    window_size: int = 600,
    step: int = 200,
    gap_mode: str = "exclude",
    pi_threshold: float | None = 0.03,
    top_fraction: float = 0.05,
    genbank: str | Path | None = None,
    reference_index: int = 0,
    normalize_orientation: bool = False,
    orientation_reference: str | Path | None = None,
    alignment_method: str = "auto",
    single_sample_drop: float = 0.5,
) -> OrganelleResult:
    """DnaSP-style sliding-window π plus mutation-hotspot (高变区) calling.

    Scans a multi-sequence FASTA alignment with a sliding window (DnaSP
    defaults: 600 bp window, 200 bp step) and reports, per window, the
    1-based alignment coordinates, the coordinates mapped back onto the
    reference sequence (``reference_index``), π, the segregating-site count
    S, and the number of comparable sites. When ``genbank`` is given, each
    window is labelled with the gene it overlaps or with the flanking-gene
    spacer name (e.g. ``trnH-psbA``).

    π per window is the average number of pairwise differences per site
    (Nei 1987, multiallelic-aware), matching the estimator used by
    :func:`nucleotide_diversity`. Gap handling (``gap_mode``):

    - ``"exclude"`` (default, DnaSP's default): complete deletion — a column
      enters the window only when every sequence has an unambiguous ACGT
      base, i.e. columns containing gaps/N are excluded from the window.
    - ``"pairwise"``: pairwise deletion — a column with ≥2 called bases
      contributes, with each sequence pair compared only where both members
      are called.

    Hotspot calling follows the conventions common in the organelle
    phylogeography / marker literature (DnaSP sliding-window analyses):
    a window is a 高变区 (mutation hotspot) when its π ranks in the top
    ``top_fraction`` fraction of all windows (default 5%) or when
    π ≥ ``pi_threshold`` (default 0.03, an absolute cutoff widely used for
    chloroplast hypervariable regions). Pass ``None`` to disable either
    rule; windows with π = 0 are never called. Adjacent/overlapping hotspot
    windows are merged into hotspot regions, whose π and S are recomputed
    over the merged span.

    ``normalize_orientation=True`` treats the input as UNALIGNED plastomes,
    normalizes their circular origin and LSC/SSC strands, then calls the
    existing phylogeny alignment backend. It defaults off because this API
    also accepts existing alignments and non-plastid sequences. Unresolved
    samples fail the pipeline explicitly; no sample is silently excluded.
    ``orientation_reference`` is a single-genome FASTA, otherwise majority
    orientation is used. Reference coordinates then refer to the normalized
    sequence. GenBank labels cannot be combined with this coordinate change.

    Every window and merged hotspot reports leave-one-out sensitivity:
    ``single_sample_driven`` means removing one sample reduces pi by at least
    ``single_sample_drop`` (default 0.5). Exclude mode holds the original
    complete-site mask fixed, avoiding new sites exposed by removing gaps.
    Pairwise mode recomputes callable counts; comparable-site counts are
    reported per omitted sample. This flag is a QC prompt, not proof of error.
    With fewer than three samples the diagnostic is unavailable (null).
    """
    parameters = {
        "alignment_fasta": str(alignment_fasta),
        "window_size": window_size,
        "step": step,
        "gap_mode": gap_mode,
        "pi_threshold": pi_threshold,
        "top_fraction": top_fraction,
        "genbank": str(genbank) if genbank is not None else None,
        "reference_index": reference_index,
    }
    parameters.update(
        {
            "normalize_orientation": normalize_orientation,
            "orientation_reference": str(orientation_reference) if orientation_reference else None,
            "alignment_method": alignment_method,
            "single_sample_drop": single_sample_drop,
        }
    )
    if not 0 < single_sample_drop <= 1:
        raise OrganelleInputError(
            code="diversity.bad_single_sample_drop", message="single_sample_drop must be in (0, 1]."
        )
    records = read_fasta(Path(alignment_fasta))
    orientation = None
    if normalize_orientation:
        if genbank is not None:
            raise OrganelleInputError(
                code="diversity.orientation_annotation_coordinates",
                message="Normalize and align separately before supplying GenBank annotations "
                "in the normalized reference coordinate system.",
            )
        records, orientation = _normalize_and_align(
            alignment_fasta, orientation_reference, alignment_method
        )
        if records is None:
            failed = _failed(
                "sliding_window_diversity",
                code="diversity.orientation_unresolved",
                message="Plastome normalization has unresolved samples; inspect orientation metrics.",
                parameters=parameters,
            )
            return failed.evolve(metrics={"orientation": orientation})
    names = [name for name, _ in records]
    seqs = [s.upper() for _, s in records]
    if len(seqs) < 2:
        return _failed(
            "sliding_window_diversity",
            code="diversity.sliding_window_diversity.too_few_sequences",
            message="sliding_window_diversity requires ≥2 aligned sequences.",
            parameters=parameters,
        )
    if gap_mode not in ("exclude", "pairwise"):
        return _failed(
            "sliding_window_diversity",
            code="diversity.sliding_window_diversity.unknown_gap_mode",
            message="gap_mode must be 'exclude' (DnaSP default) or 'pairwise'.",
            parameters=parameters,
        )
    if window_size < 2 or step < 1:
        return _failed(
            "sliding_window_diversity",
            code="diversity.sliding_window_diversity.bad_window",
            message="window_size must be ≥2 and step must be ≥1.",
            parameters=parameters,
        )
    if not (0 <= reference_index < len(seqs)):
        return _failed(
            "sliding_window_diversity",
            code="diversity.sliding_window_diversity.bad_reference",
            message="reference_index out of range for this alignment.",
            parameters=parameters,
        )
    if len({len(s) for s in seqs}) != 1:
        raise OrganelleInputError(
            code="diversity.unequal_alignment_lengths",
            message="Aligned sequences must have equal lengths; align the input first.",
        )
    L = len(seqs[0])
    ref = seqs[reference_index]

    features = _feature_intervals(genbank) if genbank is not None else []
    windows = _sliding_window_stats(seqs, ref, window_size, step, gap_mode, features)
    hotspots = _call_hotspots(windows, seqs, ref, gap_mode, pi_threshold, top_fraction, features)

    _single_sample_qc(windows + hotspots, seqs, names, gap_mode, single_sample_drop)
    mean_pi = round(sum(w["pi"] for w in windows) / len(windows), 6) if windows else 0.0
    n = len(seqs)
    rule = []
    if pi_threshold is not None:
        rule.append(f"π ≥ {pi_threshold}")
    if top_fraction > 0:
        rule.append(f"top {top_fraction:.0%}")
    return OrganelleResult(
        operation_id="diversity.sliding_window_diversity",
        operation_version=_OPERATION_VERSION,
        scope="plastid" if normalize_orientation else _SCOPE,
        status="ok",
        summary_text=(
            f"{len(windows)} windows ({window_size}bp, step {step}, gap_mode="
            f"{gap_mode}); {len(hotspots)} hotspot regions ({' or '.join(rule)})."
        ),
        metrics={
            "window_size": window_size,
            "step": step,
            "gap_mode": gap_mode,
            "n_sequences": n,
            "alignment_length": L,
            "n_windows": len(windows),
            "mean_pi": mean_pi,
            "n_hotspots": len(hotspots),
            "windows": windows,
            "hotspots": hotspots,
            "single_sample_drop": single_sample_drop,
            "orientation": orientation,
        },
        findings=tuple(
            Finding(
                code="diversity.sliding_window_diversity.hotspot",
                metric=f"{h['start']}-{h['end']}",
                value=h["pi"],
                unit="pi",
            )
            for h in hotspots[:5]
        ),
        flags=(("hotspots_found",) if hotspots else ())
        + (
            ("single_sample_driven_hotspots",)
            if any(h["single_sample_driven"] for h in hotspots)
            else ()
        ),
        artifacts=(),
        provenance=_provenance(
            "sliding_window_diversity", parameters=parameters, method="organelleverse"
        ),
    )


def write_sliding_window_diversity(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write the per-window π table from ``sliding_window_diversity()``."""
    metrics = _result_metrics(result)
    windows = list(metrics.get("windows", ()))
    path = _resolve_output_path(output, "sliding_window.tsv")
    labelled = any(w.get("label") for w in windows)
    header = "start\tend\tref_start\tref_end\tpi\tsegregating_sites\tcomparable_sites"
    header += "\tsingle_sample_driven\tdriving_sample\tmax_pi_drop_fraction"
    if labelled:
        header += "\tlabel"
    lines = [header]
    for w in windows:
        row = (
            f"{w['start']}\t{w['end']}\t{w['ref_start']}\t{w['ref_end']}"
            f"\t{w['pi']}\t{w['segregating_sites']}\t{w['comparable_sites']}"
        )
        row += (
            f"\t{w.get('single_sample_driven')}\t{w.get('driving_sample') or ''}"
            f"\t{w.get('max_pi_drop_fraction')}"
        )
        if labelled:
            row += f"\t{w.get('label') or ''}"
        lines.append(row)
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Sliding-window internals
# ---------------------------------------------------------------------------


def _normalize_and_align(input_fasta, reference_fasta, method):
    from tempfile import TemporaryDirectory

    from .._bio import write_fasta
    from ..comparative import normalize_plastome_orientation
    from ..phylogeny import align

    normalized = normalize_plastome_orientation(input_fasta, reference_fasta=reference_fasta)
    metrics = thaw_json(normalized.metrics)
    records = metrics.pop("sequences")
    metrics["status"] = normalized.status
    if normalized.status != "ok":
        return None, metrics
    with TemporaryDirectory(prefix="organelleverse-orientation-") as directory:
        path = Path(directory) / "normalized.fasta"
        write_fasta(path, records)
        aligned = align(path, method=method)
    if aligned.status != "ok" or "placeholder_alignment" in aligned.flags:
        raise OrganelleDependencyError(
            code="diversity.real_alignment_required",
            message="Orientation preprocessing requires a working MAFFT backend. "
            "Install MAFFT and select alignment_method='mafft'; "
            "the backend did not produce a real alignment.",
        )
    metrics["alignment_method"] = aligned.metrics["method"]
    return list(aligned.metrics["alignment"]), metrics


def _single_sample_qc(regions, seqs, names, gap_mode, threshold):
    """Exact rational leave-one-out pi from integer per-column pair counts."""
    from fractions import Fraction

    import numpy as np

    if len(seqs) < 3:
        for region in regions:
            region.update(
                single_sample_driven=None,
                driving_sample=None,
                max_pi_drop_fraction=None,
                leave_one_out=[],
            )
        return
    bases = np.array([list(seq) for seq in seqs])
    counts = np.array([(bases == base).sum(axis=0) for base in "ACGT"])
    called = counts.sum(axis=0)
    different = called * (called - 1) - (counts * (counts - 1)).sum(axis=0)
    complete = called == len(seqs)
    omitted_counts = []
    for sample in bases:
        sample_called = np.isin(sample, list("ACGT"))
        allele_count = sum(counts[i] * (sample == base) for i, base in enumerate("ACGT"))
        omitted_counts.append(
            (called - sample_called, different - 2 * (called - allele_count) * sample_called)
        )

    def estimate(calls, differences, original_complete):
        mask = original_complete if gap_mode == "exclude" else calls >= 2
        comparable = int(mask.sum())
        if not comparable:
            return None, 0
        # Group integer mismatch counts by their denominator k*(k-1).
        # Exact fractions make threshold equality and sample ties independent
        # of floating-point summation order and window position.
        totals = np.zeros(len(seqs) + 1, dtype=np.int64)
        np.add.at(totals, calls[mask], differences[mask])
        pi = (
            sum((Fraction(int(totals[k]), k * (k - 1)) for k in range(2, len(totals))), Fraction(0))
            / comparable
        )
        return pi, comparable

    cutoff = Fraction(str(threshold))
    for region in regions:
        start, end = region["start"] - 1, region["end"]
        mask = complete[start:end]
        baseline, _ = estimate(called[start:end], different[start:end], mask)
        rows, drops = [], []
        for name, (calls, differences) in zip(names, omitted_counts, strict=True):
            pi, comparable = estimate(calls[start:end], differences[start:end], mask)
            drop = max(Fraction(0), 1 - pi / baseline) if baseline and pi is not None else None
            drops.append(drop)
            rows.append(
                {
                    "sample": name,
                    "pi": round(float(pi), 6) if pi is not None else None,
                    "comparable_sites": comparable,
                    "pi_drop_fraction": round(float(drop), 6) if drop is not None else None,
                }
            )
        best = max(range(len(names)), key=lambda i: drops[i] if drops[i] is not None else -1)
        drop = drops[best]
        region.update(
            single_sample_driven=drop is not None and drop >= cutoff,
            driving_sample=names[best] if drop else None,
            max_pi_drop_fraction=round(float(drop), 6) if drop is not None else None,
            leave_one_out=rows,
        )


def _window_pi_s(columns: list[list[str]]) -> tuple[float, int, int]:
    """π, segregating sites and comparable-site count over pre-cut columns.

    ``columns`` holds, per alignment column, the list of called (ACGT) bases
    (possibly shorter than n under pairwise deletion). ``"exclude"`` callers
    pass only complete columns; ``"pairwise"`` callers pass every column
    with ≥2 calls. π is the mean over comparable columns of the pairwise
    difference fraction (differing pairs / C(k,2)); with complete columns
    this equals the unbiased heterozygosity n/(n-1)·(1-Σp²) used above.
    """
    comparable = segregating = 0
    diffs = 0.0
    for col in columns:
        if len(col) < 2:
            continue
        comparable += 1
        if len(set(col)) > 1:
            segregating += 1
            differing = sum(
                col[a] != col[b] for a in range(len(col)) for b in range(a + 1, len(col))
            )
            diffs += differing / (len(col) * (len(col) - 1) / 2)
    return (diffs / comparable if comparable else 0.0), segregating, comparable


def _ref_coords(ref: str, start: int, end: int) -> tuple[int | None, int | None]:
    """Reference (ungapped) 1-based coordinates of alignment span [start, end)."""
    seen = [i for i in range(start, end) if i < len(ref) and ref[i] in "ACGT"]
    if not seen:
        return None, None
    # ref coordinate (0-based) of alignment column i = called-base count before i.
    prefix = [0] * (len(ref) + 1)
    for i, base in enumerate(ref):
        prefix[i + 1] = prefix[i] + (1 if base in "ACGT" else 0)
    return prefix[seen[0]] + 1, prefix[seen[-1]] + 1


def _sliding_window_stats(
    seqs: list[str],
    ref: str,
    window_size: int,
    step: int,
    gap_mode: str,
    features: list[tuple[int, int, str]],
) -> list[dict[str, Any]]:
    """Sliding-window π/S/comparable-sites over ``seqs`` plus coordinate mapping."""
    L = min(len(s) for s in seqs)
    windows: list[dict[str, Any]] = []
    for start in range(0, max(1, L - window_size + 1), step):
        end = min(start + window_size, L)
        if gap_mode == "exclude":
            columns = [
                [s[i] for s in seqs] for i in range(start, end) if all(s[i] in "ACGT" for s in seqs)
            ]
        else:
            columns = [[s[i] for s in seqs if s[i] in "ACGT"] for i in range(start, end)]
        pi, seg, comp = _window_pi_s(columns)
        ref_start, ref_end = _ref_coords(ref, start, end)
        window: dict[str, Any] = {
            "start": start + 1,
            "end": end,
            "ref_start": ref_start,
            "ref_end": ref_end,
            "pi": round(pi, 6),
            "segregating_sites": seg,
            "comparable_sites": comp,
        }
        label = _label_region(ref_start, ref_end, features) if features else None
        if label is not None:
            window["label"] = label
        windows.append(window)
    return windows


def _call_hotspots(
    windows: list[dict[str, Any]],
    seqs: list[str],
    ref: str,
    gap_mode: str,
    pi_threshold: float | None,
    top_fraction: float,
    features: list[tuple[int, int, str]],
) -> list[dict[str, Any]]:
    """Merge qualifying windows into hotspot regions.

    A window qualifies when π > 0 and (π ≥ pi_threshold or π is among the
    top ``top_fraction`` of window π values). Merged regions recompute π and
    S over the full merged span with the same estimator.
    """
    if not windows:
        return []
    qualifying = set()
    if pi_threshold is not None:
        qualifying.update(i for i, w in enumerate(windows) if w["pi"] >= pi_threshold)
    if top_fraction > 0:
        ranked = sorted(windows, key=lambda w: w["pi"], reverse=True)
        k = max(1, math.ceil(len(ranked) * top_fraction))  # top ceil(f·n) windows
        cut = ranked[k - 1]["pi"]
        qualifying.update(i for i, w in enumerate(windows) if w["pi"] >= cut)
    qualifying = {i for i in qualifying if windows[i]["pi"] > 0}

    merged: list[list[int]] = []
    for i in sorted(qualifying):
        if merged and windows[i]["start"] <= windows[merged[-1][-1]]["end"] + 1:
            merged[-1].append(i)
        else:
            merged.append([i])

    hotspots = []
    for members in merged:
        span_start = windows[members[0]]["start"]
        span_end = windows[members[-1]]["end"]
        peak = max(members, key=lambda i: windows[i]["pi"])
        columns = _columns_between(seqs, span_start - 1, span_end, gap_mode)
        pi, seg, _comp = _window_pi_s(columns)
        ref_start, ref_end = _ref_coords(ref, span_start - 1, span_end)
        hotspot: dict[str, Any] = {
            "start": span_start,
            "end": span_end,
            "ref_start": ref_start,
            "ref_end": ref_end,
            "pi": round(pi, 6),
            "mean_window_pi": round(sum(windows[i]["pi"] for i in members) / len(members), 6),
            "max_window_pi": windows[peak]["pi"],
            "segregating_sites": seg,
            "n_windows": len(members),
            "peak_window_start": windows[peak]["start"],
            "peak_window_end": windows[peak]["end"],
        }
        label = _label_region(ref_start, ref_end, features) if features else None
        if label is not None:
            hotspot["label"] = label
        hotspots.append(hotspot)
    return hotspots


def _columns_between(seqs: list[str], start: int, end: int, gap_mode: str) -> list[list[str]]:
    """Called-base columns of alignment span [start, end) under ``gap_mode``."""
    columns = []
    for i in range(start, end):
        if gap_mode == "exclude":
            if all(i < len(s) and s[i] in "ACGT" for s in seqs):
                columns.append([s[i] for s in seqs])
        else:
            col = [s[i] for s in seqs if i < len(s) and s[i] in "ACGT"]
            if len(col) >= 2:
                columns.append(col)
    return columns


def _feature_intervals(genbank: str | Path) -> list[tuple[int, int, str]]:
    """Gene-feature (ref) coordinate intervals from a GenBank file.

    Returns 1-based inclusive ``(start, end, name)`` triples for features of
    type gene/CDS/tRNA/rRNA, named by the first available qualifier of
    gene, product, locus_tag, note. Coordinates are taken on the GenBank
    record's own sequence (the reference the alignment maps to).
    """
    intervals: list[tuple[int, int, str]] = []
    for _seqid, _seq, feats in read_genbank(Path(genbank)):
        for feat in feats:
            if feat.get("type") not in ("gene", "CDS", "tRNA", "rRNA"):
                continue
            quals = feat.get("qualifiers", {})
            name = None
            for key in ("gene", "product", "locus_tag", "note"):
                values = quals.get(key)
                if values:
                    name = str(values[0])
                    break
            if name is None:
                continue
            start = int(feat["start"]) + 1  # GenBank parser is 0-based half-open
            end = int(feat["end"])
            intervals.append((start, end, name))
    return intervals


def _label_region(
    ref_start: int | None,
    ref_end: int | None,
    features: list[tuple[int, int, str]],
) -> str | None:
    """Gene or flanking-spacer name for a reference-coordinate span.

    The span is labelled with the feature it overlaps most; if it overlaps
    none, with ``upstream-downstream`` flanking genes (the convention used
    for chloroplast spacers, e.g. ``trnH-psbA``).
    """
    if ref_start is None or ref_end is None or not features:
        return None
    best = None
    best_overlap = 0
    for start, end, name in features:
        overlap = max(0, min(end, ref_end) - max(start, ref_start) + 1)
        if overlap > best_overlap:
            best_overlap = overlap
            best = name
    if best is not None:
        return best
    upstream = [f for f in features if f[1] < ref_start]
    downstream = [f for f in features if f[0] > ref_end]
    left = max(upstream, key=lambda f: f[1])[2] if upstream else ""
    right = min(downstream, key=lambda f: f[0])[2] if downstream else ""
    if left and right:
        return f"{left}-{right}"
    return left or right or "intergenic"


def neutral_tests(
    alignment_fasta: str | Path,
) -> OrganelleResult:
    """Compute the standard neutral-equilibrium tests from an alignment.

    Returns π, θ_π, Watterson's θ_W, Tajima's D, segregating-site count S,
    and sample size n. Backend: scikit-allel when available, else the
    textbook formulas (Nei 1987; Watterson 1975; Tajima 1989).
    """
    parameters = {"alignment_fasta": str(alignment_fasta)}
    seqs = [s for _, s in read_fasta(Path(alignment_fasta))]
    if len(seqs) < 2:
        return _failed(
            "neutral_tests",
            code="diversity.neutral_tests.too_few_sequences",
            message="neutral_tests requires ≥2 sequences.",
            parameters=parameters,
        )
    try:
        L = _alignment_length(seqs)
    except ValueError as exc:
        return _failed(
            "neutral_tests",
            code="diversity.neutral_tests.unequal_alignment_lengths",
            message=str(exc),
            parameters=parameters,
        )
    seqs = [s.upper() for s in seqs]
    n = len(seqs)

    try:
        import allel  # type: ignore
        import numpy as np  # type: ignore

        seg_pos, alt_counts, n_comp = _alignment_stats(seqs)
        method = "scikit-allel"
        S = len(seg_pos)
        if S == 0:
            pi = theta_pi = theta_w = 0.0
            tajima = float("nan")
        else:
            width = max(len(c) for c in alt_counts)
            ac = allel.AlleleCountsArray(
                np.array([tuple(c) + (0,) * (width - len(c)) for c in alt_counts])
            )
            pos = np.array(seg_pos)
            pi = _pi_skitallel(alt_counts, seg_pos, n_comp)
            theta_pi = pi  # θ_π ≡ π (Nei & Tajima 1981)
            theta_w = (
                allel.watterson_theta(
                    pos, ac, start=1, stop=L, is_accessible=np.array(_comparable_mask(seqs))
                )
                if n_comp > 0
                else 0.0
            )
            tajima = float(allel.tajima_d(ac, pos=pos)) if S >= 3 else float("nan")
    except ImportError:
        method = "organelleverse_python"
        pi = _pi_pure(seqs)
        theta_pi = pi
        seg_pos, alt_counts, n_comp = _alignment_stats(seqs)
        S = len(seg_pos)
        theta_w, tajima = _theta_w_and_tajima_d(S, n, n_comp, pi)

    # The canonical contract carries finite JSON numbers only, so the "not
    # computable" sentinel is JSON null instead of float('nan'). The estimators
    # themselves are unchanged; ``theta_w != theta_w`` still means NaN upstream.
    metrics = {
        "pi": round(pi, 6),
        "theta_pi": round(theta_pi, 6),
        "theta_w": round(theta_w, 6) if theta_w == theta_w else None,
        "tajima_d": round(tajima, 6) if tajima == tajima else None,
        "segregating_sites": S,
        "n_sequences": n,
        "comparable_sites": n_comp,
    }

    return OrganelleResult(
        operation_id="diversity.neutral_tests",
        operation_version=_OPERATION_VERSION,
        scope=_SCOPE,
        status="ok",
        summary_text=(
            f"π={pi:.4f}, θ_W={theta_w:.4f}, Tajima's D={tajima:.4f}"
            if tajima == tajima
            else f"π={pi:.4f}, θ_W={theta_w:.4f}, Tajima's D=NA (<3 seg sites)"
        ),
        metrics=metrics,
        findings=(
            Finding(code="diversity.neutral_tests.pi", metric="pi", value=metrics["pi"]),
            Finding(
                code="diversity.neutral_tests.theta_w", metric="theta_w", value=metrics["theta_w"]
            ),
            Finding(
                code="diversity.neutral_tests.tajima_d",
                metric="tajima_d",
                value=metrics["tajima_d"],
            ),
        ),
        flags=("tajima_d_negative",) if tajima == tajima and tajima < 0 else (),
        artifacts=(),
        provenance=_provenance("neutral_tests", parameters=parameters, method=method),
    )


def write_nucleotide(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write the windowed nucleotide-diversity table from ``nucleotide_diversity()``."""
    metrics = _result_metrics(result)
    windows = list(metrics.get("windowed_pi", ()))
    path = _resolve_output_path(output, "pi.windowed.tsv")
    path.write_text(
        "start\tend\tpi\n"
        + "\n".join(f"{w['start']}\t{w['end']}\t{w['pi']}" for w in windows)
        + ("\n" if windows else "")
    )
    return path


def write_neutral_tests(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write neutral-test metrics as a two-column TSV."""
    metrics = _result_metrics(result)
    path = _resolve_output_path(output, "neutral_tests.tsv")
    path.write_text(
        "metric\tvalue\n"
        + "\n".join(f"{key}\t{value}" for key, value in metrics.items())
        + ("\n" if metrics else "")
    )
    return path


def _resolve_output_path(output: str | Path, default_name: str) -> Path:
    path = Path(output)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / default_name


# ---------------------------------------------------------------------------
# Estimator helpers
# ---------------------------------------------------------------------------


def _pi_pure(seqs: list[str]) -> float:
    """Pure-Python π: mean over comparable columns of (differing_pairs / C(n,2))."""
    L = _alignment_length(seqs)
    diffs = comparable = 0
    for i in range(L):
        col = [s[i] for s in seqs if s[i] in "ACGT"]
        if len(col) < 2:
            continue
        comparable += 1
        d = 0
        for a in range(len(col)):
            for b in range(a + 1, len(col)):
                if col[a] != col[b]:
                    d += 1
        diffs += d / (len(col) * (len(col) - 1) / 2)
    return diffs / comparable if comparable else 0.0


def _pi_skitallel(ac, pos, n_comparable: int) -> float:
    """π from per-site allele counts, normalised by comparable sites (Nei 1987).

    Per-site unbiased heterozygosity h = n/(n-1) · (1 - Σ p_i²) over ALL
    alleles at the site (multiallelic-aware), summed over segregating sites
    and divided by the number of comparable (ACGT) columns. The historical
    name is kept for API compatibility; scikit-allel is no longer needed.
    """
    if len(ac) == 0 or n_comparable == 0:
        return 0.0
    total = 0.0
    for counts in ac:
        n = sum(counts)
        if n < 2:
            continue
        h = 1.0 - sum((c / n) ** 2 for c in counts)
        total += (n / (n - 1.0)) * h
    return float(total / n_comparable)


def _theta_w_and_tajima_d(S: int, n: int, n_comparable: int, pi: float):
    """Watterson's θ_W per site and Tajima's D (1989), pure-Python.

    θ_W = S / (a_1 * L)  where a_1 = Σ_{i=1}^{n-1} 1/i, L = comparable sites.
    Tajima's D = (d) / √(e1*S + e2*S*(S-1)), with d = θ̂_π - θ̂_W (absolute,
    not per-site), and the variance coefficients from Tajima (1989):
        a_2 = Σ_{i=1}^{n-1} 1/i²
        b_1 = (n+1) / (3(n-1));  b_2 = 2(n²+n+3) / (9n(n-1))
        c_1 = b_1 - 1/a_1;      c_2 = b_2 - (n+2)/(a_1·n) + a_2/a_1²
        e_1 = c_1 / a_1;        e_2 = c_2 / (a_1² + a_2)
    This matches scikit-allel's implementation exactly.
    """
    import math

    if S == 0 or n_comparable == 0:
        return 0.0, float("nan")
    a1 = _harmonic_a(n)
    theta_w = S / (a1 * n_comparable)
    if S < 3:
        return theta_w, float("nan")
    a2 = sum(1.0 / (i * i) for i in range(1, n))
    b1 = (n + 1.0) / (3.0 * (n - 1.0))
    b2 = 2.0 * (n * n + n + 3.0) / (9.0 * n * (n - 1.0))
    c1 = b1 - 1.0 / a1
    c2 = b2 - (n + 2.0) / (a1 * n) + a2 / (a1 * a1)
    e1 = c1 / a1
    e2 = c2 / (a1 * a1 + a2)
    var_d = e1 * S + e2 * S * (S - 1)
    if var_d <= 0:
        return theta_w, float("nan")
    # Tajima's D uses the ABSOLUTE difference of unnormalised θ estimators.
    d = pi * n_comparable - S / a1  # θ̂_π (abs) - θ̂_W (abs)
    tajima = d / math.sqrt(var_d)
    return theta_w, float(tajima)
