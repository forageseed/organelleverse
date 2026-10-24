"""Multi-sample organelle stability summaries: window depth and SNV/InDel fractions.

A thin layer over the per-sample estimators for DNA-maintenance mutants,
populations, regenerants or CMS lines aligned to one organelle reference:

* :func:`summarize_depth_windows` - mean read depth in fixed windows per sample
  and contig, and its log2 ratio to that sample's own median window depth on the
  same contig (mitochondrion and plastid are normalized separately because their
  copy numbers differ).
* :func:`summarize_variant_fractions` - candidate SNVs and InDels found in any
  sample, then the alternate-read fraction of every candidate in every sample.
  SNV fractions come from :func:`organelleverse.heteroplasmy.quantify_variant_fractions`
  (quality-filtered pileup, Wilson interval, homology mask); InDel fractions are
  counted from CIGAR operations over reads spanning the anchor base.

Both return canonical results whose metrics feed the plots in
:mod:`organelleverse.visualization.stability_plots`. Fractions are read
fractions, not molecule fractions; neither function calls genotypes.
"""

from __future__ import annotations

import csv
import math
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from organelleverse._bam import aligned_depth
from organelleverse._bio import read_fasta
from organelleverse.core.result import ErrorDetail, Finding, OrganelleResult
from organelleverse.heteroplasmy import (
    _mask_source,
    _masked_intervals,
    _MaskError,
    _provenance,
    _wilson,
    quantify_variant_fractions,
)

Scope = Literal["mitochondrion", "plastid"]
CATEGORIES = ("low_confidence", "repeat_related", "other")


def _failed(op: str, code: str, message: str, scope: str = "mixed") -> OrganelleResult:
    return OrganelleResult(
        operation_id=f"heteroplasmy.{op}",
        operation_version="1.0",
        scope=scope,
        status="failed",
        summary_text=message,
        errors=(ErrorDetail(code=code, message=message),),
        provenance=_provenance(op, parameters={}, backend="rubam"),
    )


def _read_filter(min_mapping_quality: int, exclude_duplicates: bool):
    def keep(read) -> bool:
        return not (
            read.is_unmapped
            or read.is_secondary
            or read.is_supplementary
            or read.is_qcfail
            or (exclude_duplicates and read.is_duplicate)
            or read.mapping_quality < min_mapping_quality
        )

    return keep


def summarize_depth_windows(
    bam_paths: Mapping[str, str | Path],
    *,
    window_size: int = 1000,
    organelles: Mapping[str, Scope] | None = None,
    sequences: Sequence[str] | None = None,
    min_mapping_quality: int = 0,
    exclude_duplicates: bool = True,
) -> OrganelleResult:
    """Window depth per sample and its log2 ratio to the sample's median window.

    ``bam_paths`` maps a sample label to an indexed BAM/CRAM; every BAM must
    carry the same reference sequences (names and lengths). Depth is
    ``samtools depth`` over primary alignments, deletions not counted (``min_mapping_quality``
    defaults to 0 so both copies of a repeat keep their reads); a window's value
    is the mean over its bases, the last window being shorter. ``log2_ratio`` is
    ``log2((depth + 1) / (median + 1))`` where ``median`` is the sample's median
    window depth on that contig, so 0 = typical depth for that sample and
    organelle. ``organelles`` labels contigs (``mitochondrion``/``plastid``);
    ``sequences`` limits the summary to those reference sequences (e.g. the
    organelles of a BAM mapped competitively against the nuclear genome too).
    """
    import numpy as np

    op = "summarize_depth_windows"
    if not bam_paths:
        return _failed(op, "heteroplasmy.no_samples", "bam_paths is empty.")
    if window_size < 1 or min_mapping_quality < 0:
        return _failed(
            op, "heteroplasmy.invalid_threshold", "window_size >= 1 and min_mapping_quality >= 0."
        )
    try:
        import rubam
    except ImportError:
        return _failed(
            op, "heteroplasmy.rubam_missing", "Install organelleverse[qc] for BAM depth."
        )
    # Count aligned M/=/X blocks; deletions and reference skips contribute no depth.
    contigs: dict[str, int] | None = None
    profiles = []
    try:
        for sample, path in bam_paths.items():
            with rubam.AlignmentFile(str(path), "rb") as bam:
                lengths = {
                    seqid: length
                    for seqid, length in zip(bam.references, bam.lengths, strict=True)
                    if sequences is None or seqid in sequences
                }
            if sequences is not None and set(sequences) - set(lengths):
                raise ValueError(f"{sample}: BAM lacks {sorted(set(sequences) - set(lengths))}")
            if contigs is None:
                contigs = lengths
            elif lengths != contigs:
                raise ValueError(f"{sample}: reference sequences differ from the first BAM")
            for seqid, length in lengths.items():
                with rubam.AlignmentFile(str(path), "rb") as bam:
                    depth = aligned_depth(
                        bam,
                        seqid,
                        length,
                        min_mapping_quality=min_mapping_quality,
                        exclude_duplicates=exclude_duplicates,
                    )
                starts = np.arange(0, length, window_size)
                means = np.add.reduceat(depth, starts) / np.diff(np.append(starts, length))
                median = float(np.median(means))
                profiles.append(
                    {
                        "sample": sample,
                        "seqid": seqid,
                        "median_depth": round(median, 3),
                        "mean_depth": [round(float(v), 3) for v in means],
                        "log2_ratio": [
                            round(math.log2((float(v) + 1) / (median + 1)), 4) for v in means
                        ],
                    }
                )
    except (OSError, ValueError, KeyError) as exc:
        return _failed(op, "heteroplasmy.depth_failed", f"Could not read BAM depth: {exc}")
    labels = dict(organelles or {})
    contig_rows = [
        {"seqid": seqid, "length": length, "organelle": labels.get(seqid)}
        for seqid, length in (contigs or {}).items()
    ]
    scopes = {row["organelle"] for row in contig_rows}
    return OrganelleResult(
        operation_id=f"heteroplasmy.{op}",
        operation_version="1.0",
        scope=next(iter(scopes)) if len(scopes) == 1 and None not in scopes else "mixed",
        status="ok",
        summary_text=(
            f"Window depth for {len(bam_paths)} samples over {len(contig_rows)} sequences "
            f"({window_size} bp windows)."
        ),
        metrics={
            "window_size": window_size,
            "samples": list(bam_paths),
            "contigs": contig_rows,
            "profiles": profiles,
            "ratio_definition": "log2((window mean depth + 1) / (sample median window depth + 1)), per contig",
        },
        findings=(
            Finding(code="heteroplasmy.depth_samples", metric="samples", value=len(bam_paths)),
        ),
        flags=("read_depth_not_copy_number",),
        provenance=_provenance(
            op,
            parameters={
                "bam_paths": {k: str(v) for k, v in bam_paths.items()},
                "window_size": window_size,
                "organelles": labels,
                "sequences": list(sequences) if sequences is not None else None,
                "min_mapping_quality": min_mapping_quality,
                "exclude_duplicates": exclude_duplicates,
            },
            backend="rubam",
            software_versions={"rubam": rubam.__version__},
        ),
    )


_BASE_CODE = bytes.maketrans(b"ACGTacgt", b"\x00\x01\x02\x03\x00\x01\x02\x03")
_FLUSH_BASES = 20_000_000


def _discover(bam, seqid, sequence, keep, min_base_quality):
    """Per-position A/C/G/T counts, InDel allele counts and spanning depth for one contig.

    One pass over the reads: aligned blocks are collected as (reference start,
    bases, qualities) and counted in batches with ``numpy.bincount`` (rubam's
    ``count_coverage`` takes minutes per 20 kb at plastid depths).
    """
    import numpy as np

    length = len(sequence)
    acgt = np.zeros(4 * length, dtype=np.int64)
    starts: list[int] = []
    sizes: list[int] = []
    bases: list[str] = []
    quals: list[str] = []
    pending = 0

    def flush() -> None:
        if not starts:
            return
        size = np.array(sizes, dtype=np.int64)
        offset = np.repeat(np.array(starts, dtype=np.int64) - np.cumsum(size) + size, size)
        pos = offset + np.arange(int(size.sum()), dtype=np.int64)
        code = np.frombuffer("".join(bases).encode("ascii").translate(_BASE_CODE), dtype=np.uint8)
        qual = np.frombuffer("".join(quals).encode("ascii"), dtype=np.uint8) - 33
        ok = (code < 4) & (qual >= min_base_quality)
        acgt[:] += np.bincount(code[ok].astype(np.int64) * length + pos[ok], minlength=4 * length)
        starts.clear(), sizes.clear(), bases.clear(), quals.clear()

    indels: Counter[tuple[int, str, str]] = Counter()
    span = np.zeros(length + 1, dtype=np.int64)
    for read in bam.fetch(seqid, 0, bam.get_reference_length(seqid)):
        if not keep(read):
            continue
        span[read.reference_start] += 1
        span[read.reference_end] -= 1
        pos = read.reference_start  # 0-based reference coordinate of the next aligned base
        qpos = 0
        seq = read.query_sequence
        qstr = read.query_qualities_str
        for op, n in read.cigartuples:
            if op in (0, 7, 8):
                starts.append(pos)
                sizes.append(n)
                bases.append(seq[qpos : qpos + n])
                quals.append(qstr[qpos : qpos + n])
                pending += n
                pos += n
                qpos += n
            elif op == 1:
                if 0 < pos <= length:
                    anchor = sequence[pos - 1]
                    indels[(pos, anchor, anchor + seq[qpos : qpos + n])] += 1
                qpos += n
            elif op in (2, 3):
                if op == 2 and pos > 0:
                    anchor = sequence[pos - 1]
                    indels[(pos, anchor + sequence[pos : pos + n], anchor)] += 1
                pos += n
            elif op == 4:
                qpos += n
        if pending >= _FLUSH_BASES:
            flush()
            pending = 0
    flush()
    return acgt.reshape(4, length), indels, np.cumsum(span)[:length]


def _repeat_intervals(repeats) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in repeats or ():
        length = int(row["length"])
        for start in row["positions"]:
            out[str(row["sequence_id"])].append((int(start), int(start) + length - 1))
    return out


def _in_repeat(
    intervals: dict[str, list[tuple[int, int]]], seqid: str, start: int, end: int
) -> bool:
    return any(a <= end and start <= b for a, b in intervals.get(seqid, ()))


def summarize_variant_fractions(
    bam_paths: Mapping[str, str | Path],
    references: Mapping[Scope, str | Path],
    *,
    min_alt_fraction: float = 0.01,
    min_alt_reads: int = 3,
    min_depth: int = 20,
    min_mapping_quality: int = 20,
    min_base_quality: int = 20,
    exclude_duplicates: bool = True,
    homology_mask: bool = True,
    homology_min_identity: float = 80.0,
    homology_min_length: int = 100,
    repeats: Sequence[Mapping[str, object]] | None = None,
    control_samples: Sequence[str] = (),
) -> OrganelleResult:
    """SNV/InDel alternate-read fractions for every candidate in every sample.

    ``references`` maps ``mitochondrion``/``plastid`` to the FASTA each BAM was
    aligned to (one BAM may hold both). A candidate is any SNV or InDel that, in
    at least one sample, has >= ``min_alt_reads`` supporting reads, depth >=
    ``min_depth`` and fraction >= ``min_alt_fraction``. Every candidate is then
    measured in every sample: SNVs with ``quantify_variant_fractions`` at the
    union of SNV sites; InDels as reads carrying that exact InDel over reads
    spanning its anchor base (VCF-style anchored alleles, 1-based position).

    ``homology_mask`` (needs both references) masks regions of each organelle that
    match the other one (MTPT-like, ``transfer.detect_transfers_blast``), once per
    organelle. ``repeats`` takes ``structure.compute_multiconf(...)["repeats"]`` rows
    (``sequence_id``, ``positions``, ``length``).

    Each sample row gets ``detected`` (passes the candidate thresholds in that
    sample) and a ``category``: ``low_confidence`` (homology-masked, seen in
    any of ``control_samples`` at >= ``min_alt_fraction`` at any depth, depth below ``min_depth`` or fewer than
    ``min_alt_reads`` supporting reads), ``repeat_related`` (the variant overlaps
    a repeat copy) or ``other``, in that order of precedence. Sites record
    ``masked``, ``in_control`` and ``in_repeat``.
    """
    op = "summarize_variant_fractions"
    if not bam_paths:
        return _failed(op, "heteroplasmy.no_samples", "bam_paths is empty.")
    if not references or any(s not in ("mitochondrion", "plastid") for s in references):
        return _failed(
            op, "heteroplasmy.invalid_scope", "references keys must be mitochondrion or plastid."
        )
    if not 0 < min_alt_fraction < 1 or min_alt_reads < 1 or min_depth < 1:
        return _failed(
            op,
            "heteroplasmy.invalid_threshold",
            "min_alt_fraction in (0, 1), min_alt_reads >= 1, min_depth >= 1.",
        )
    if set(control_samples) - set(bam_paths):
        return _failed(
            op, "heteroplasmy.unknown_control", "control_samples must be keys of bam_paths."
        )
    if homology_mask and len(references) != 2:
        return _failed(
            op,
            "heteroplasmy.mask_needs_reference",
            "homology_mask compares the two organelles; pass both references or homology_mask=False.",
        )
    try:
        import rubam
    except ImportError:
        return _failed(op, "heteroplasmy.rubam_missing", "Install organelleverse[qc] for pileups.")

    keep = _read_filter(min_mapping_quality, exclude_duplicates)
    sequences = {scope: dict(read_fasta(Path(path))) for scope, path in references.items()}
    scope_of = {seqid: scope for scope, seqs in sequences.items() for seqid in seqs}
    masks: dict[str, dict[str, list[tuple[int, int, str]]]] = {}
    versions: dict[str, str] = {"rubam": rubam.__version__}
    if homology_mask:
        try:
            for scope, path in references.items():
                other = "plastid" if scope == "mitochondrion" else "mitochondrion"
                masks[scope], found = _masked_intervals(
                    None,
                    references[other],
                    path,
                    homology_min_identity,
                    homology_min_length,
                    other=other,
                )
                versions.update(found)
        except _MaskError as exc:
            return _failed(op, exc.code, str(exc))

    # Discovery: union of candidates over samples; InDel counts are kept for every sample.
    snv_sites: dict[tuple[str, int, str, str], None] = {}
    indel_sites: dict[tuple[str, int, str, str], None] = {}
    indel_counts: dict[str, dict[str, tuple[Counter, object]]] = {}
    try:
        import numpy as np

        for sample, path in bam_paths.items():
            indel_counts[sample] = {}
            with rubam.AlignmentFile(str(path), "rb") as bam:
                for seqid, scope in scope_of.items():
                    if seqid not in bam.references:
                        raise ValueError(f"{sample}: BAM has no reference sequence {seqid!r}")
                    sequence = sequences[scope][seqid].upper()
                    acgt, indels, span = _discover(bam, seqid, sequence, keep, min_base_quality)
                    indel_counts[sample][seqid] = (indels, span)
                    depth = acgt.sum(axis=0)
                    for b, base in enumerate("ACGT"):
                        alt = acgt[b]
                        hit = (alt >= min_alt_reads) & (depth >= min_depth)
                        hit &= alt >= min_alt_fraction * np.maximum(depth, 1)
                        for pos0 in np.flatnonzero(hit):
                            ref = sequence[pos0]
                            if ref in "ACGT" and ref != base:
                                snv_sites[(seqid, int(pos0) + 1, ref, base)] = None
                    for (pos, ref, alt), n in indels.items():
                        cover = int(span[pos - 1])
                        if (
                            n >= min_alt_reads
                            and cover >= min_depth
                            and n >= min_alt_fraction * cover
                        ):
                            indel_sites[(seqid, pos, ref, alt)] = None
    except (OSError, ValueError) as exc:
        return _failed(op, "heteroplasmy.pileup_failed", f"Could not scan BAM: {exc}")

    repeat_spans = _repeat_intervals(repeats)
    sites = []
    for kind, keys in (("SNV", snv_sites), ("InDel", indel_sites)):
        for seqid, pos, ref, alt in sorted(keys):
            scope = scope_of[seqid]
            end = pos + max(len(ref), len(alt)) - 1
            sites.append(
                {
                    "site_id": f"{seqid}:{pos}:{ref}>{alt}",
                    "scope": scope,
                    "seqid": seqid,
                    "position": pos,
                    "ref": ref,
                    "alt": alt,
                    "kind": kind,
                    "masked": _mask_source(masks.get(scope, {}), seqid, pos - 1) is not None,
                    "in_repeat": _in_repeat(repeat_spans, seqid, pos, end),
                }
            )

    def row(sample, site, alt_count, depth, fraction, lo, hi):
        enough = depth >= min_depth and alt_count >= min_alt_reads
        return {
            "sample": sample,
            "site_id": site["site_id"],
            "alt_count": alt_count,
            "depth": depth,
            "alt_fraction": fraction,
            "ci95_lower": lo,
            "ci95_upper": hi,
            "detected": enough and alt_count >= min_alt_fraction * depth,
        }

    # SNVs: the existing per-sample estimator at the union of SNV sites. The mask is
    # applied as a label here (not inside the estimator) so masked sites keep a fraction.
    rows = []
    by_id = {s["site_id"]: s for s in sites}
    with tempfile.TemporaryDirectory(prefix="ov_stability_") as tmp:
        site_files = {}
        for scope in references:
            snvs = [s for s in sites if s["kind"] == "SNV" and s["scope"] == scope]
            if not snvs:
                continue
            path = Path(tmp) / f"{scope}.sites.tsv"
            with path.open("w", encoding="utf-8", newline="\n") as out:
                writer = csv.writer(out, delimiter="\t", lineterminator="\n")
                writer.writerow(["site_id", "seqid", "position", "ref", "alt"])
                for s in snvs:
                    writer.writerow([s["site_id"], s["seqid"], s["position"], s["ref"], s["alt"]])
            site_files[scope] = path
        for sample, path in bam_paths.items():
            for scope, sites_tsv in site_files.items():
                result = quantify_variant_fractions(
                    path,
                    sites_tsv,
                    scope=scope,
                    min_mapping_quality=min_mapping_quality,
                    min_base_quality=min_base_quality,
                    min_depth=1,
                    exclude_duplicates=exclude_duplicates,
                    auto_homology_mask=False,
                )
                if result.status != "ok":
                    return _failed(op, result.errors[0].code, f"{sample}: {result.summary_text}")
                for r in result.metrics["sites"]:
                    rows.append(
                        row(
                            sample,
                            by_id[r["site_id"]],
                            int(r["alt_count"]),
                            int(r["depth"]),
                            r["alt_fraction"],
                            r["ci95_lower"],
                            r["ci95_upper"],
                        )
                    )
    for site in sites:
        if site["kind"] != "InDel":
            continue
        for sample in bam_paths:
            indels, span = indel_counts[sample][site["seqid"]]
            alt_count = int(indels[(site["position"], site["ref"], site["alt"])])
            depth = int(span[site["position"] - 1])
            if depth:
                lo, hi = _wilson(alt_count, depth)
                values = (round(alt_count / depth, 6), round(lo, 6), round(hi, 6))
            else:
                values = (None, None, None)
            rows.append(row(sample, site, alt_count, depth, *values))
    # A variant the control samples carry cannot be attributed to the treatment:
    # shared NUMT/NUPT reads, reference/line differences and systematic errors.
    controls = set(control_samples)
    # Seen at the candidate fraction in any control, whatever its depth: a control
    # with few uniquely mapped reads still shows the allele is in the background.
    in_control = {
        r["site_id"]
        for r in rows
        if r["sample"] in controls and (r["alt_fraction"] or 0) >= min_alt_fraction
    }
    for site in sites:
        site["in_control"] = site["site_id"] in in_control
    for r in rows:
        site = by_id[r["site_id"]]
        enough = r["depth"] >= min_depth and r["alt_count"] >= min_alt_reads
        if site["masked"] or site["in_control"] or not enough:
            r["category"] = "low_confidence"
        else:
            r["category"] = "repeat_related" if site["in_repeat"] else "other"
    counts = Counter((s["kind"], s["scope"]) for s in sites)
    return OrganelleResult(
        operation_id=f"heteroplasmy.{op}",
        operation_version="1.0",
        scope=next(iter(references)) if len(references) == 1 else "mixed",
        status="ok",
        summary_text=(
            f"{counts[('SNV', 'mitochondrion')] + counts[('SNV', 'plastid')]} SNV and "
            f"{counts[('InDel', 'mitochondrion')] + counts[('InDel', 'plastid')]} InDel "
            f"candidates measured in {len(bam_paths)} samples."
        ),
        metrics={
            "samples": list(bam_paths),
            "references": {scope: list(seqs) for scope, seqs in sequences.items()},
            "thresholds": {
                "min_alt_fraction": min_alt_fraction,
                "min_alt_reads": min_alt_reads,
                "min_depth": min_depth,
                "min_mapping_quality": min_mapping_quality,
                "min_base_quality": min_base_quality,
            },
            "homology_mask": {scope: sum(len(v) for v in m.values()) for scope, m in masks.items()},
            "repeat_pairs": len(repeats or ()),
            "control_samples": list(control_samples),
            "sites": sites,
            "fractions": rows,
            "categories": list(CATEGORIES),
        },
        findings=tuple(
            Finding(
                code=f"heteroplasmy.{kind.lower()}_candidates", metric=f"{kind}_{scope}", value=n
            )
            for (kind, scope), n in sorted(counts.items())
        ),
        flags=(
            "read_fraction_not_molecule_fraction",
            "homologous_regions_masked" if homology_mask else "numt_mapping_not_resolved",
        ),
        provenance=_provenance(
            op,
            parameters={
                "bam_paths": {k: str(v) for k, v in bam_paths.items()},
                "references": {k: str(v) for k, v in references.items()},
                "min_alt_fraction": min_alt_fraction,
                "min_alt_reads": min_alt_reads,
                "min_depth": min_depth,
                "min_mapping_quality": min_mapping_quality,
                "min_base_quality": min_base_quality,
                "exclude_duplicates": exclude_duplicates,
                "homology_mask": homology_mask,
                "homology_min_identity": homology_min_identity,
                "homology_min_length": homology_min_length,
                "repeat_pairs": len(repeats or ()),
                "control_samples": list(control_samples),
            },
            backend="rubam",
            software_versions=versions,
        ),
    )


__all__ = ["CATEGORIES", "summarize_depth_windows", "summarize_variant_fractions"]
