"""Quantify RNA editing at caller-supplied sites from BAM/CRAM pileups.

This operation measures known sites only; it does not discover editing sites.
Coordinates are 1-based reference coordinates. ``ref`` and ``edited`` are
bases on the reference plus strand (the orientation stored by BAM pileups),
regardless of transcript strand. ``strand`` is retained as site annotation.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Literal

from ..core.result import Finding, OrganelleResult
from .rna_edit import _failed, _provenance

_OPERATION = "quantify_efficiency"
_VERSION = "1.0"


def quantify_known_site_efficiency(
    bam_path: str | Path,
    sites_tsv: str | Path,
    *,
    scope: Literal["mitochondrion", "plastid"] = "mitochondrion",
    min_mapping_quality: int = 20,
    min_base_quality: int = 20,
    exclude_duplicates: bool = True,
    reference_fasta: str | Path | None = None,
) -> OrganelleResult:
    """Count reference, edited, other and usable reads at known editing sites.

    ``sites_tsv`` requires a header and columns ``site_id,seqid,position,ref,
    edited,strand``. Positions are 1-based. ``ref`` and ``edited`` must be
    distinct A/C/G/T alleles on the reference plus strand. ``strand`` is ``+``
    or ``-`` and is metadata only: for a reverse-strand gene, callers should
    still supply the alleles as they appear on the genomic reference.

    BAM/CRAM must be coordinate-sorted and indexed. Unmapped, secondary,
    supplementary, QC-failed reads and (by default) duplicate reads are
    excluded, along with bases below either quality threshold. The 95% interval
    is the Wilson score interval for a binomial proportion, conditional on the
    informative depth (ref + edited + other A/C/G/T).
    """
    params = {
        "bam_path": str(bam_path),
        "sites_tsv": str(sites_tsv),
        "scope": scope,
        "min_mapping_quality": min_mapping_quality,
        "min_base_quality": min_base_quality,
        "exclude_duplicates": exclude_duplicates,
        "reference_fasta": str(reference_fasta) if reference_fasta is not None else None,
    }
    if scope not in ("mitochondrion", "plastid"):
        return _failed(
            _OPERATION,
            scope="mitochondrion",
            code="rna_editing.invalid_scope",
            message="scope must be 'mitochondrion' or 'plastid'.",
            backend="pysam",
        )
    if min_mapping_quality < 0 or min_base_quality < 0:
        return _failed(
            _OPERATION,
            scope=scope,
            code="rna_editing.invalid_quality_threshold",
            message="Quality thresholds must be non-negative.",
            backend="pysam",
        )
    try:
        sites = _read_sites(sites_tsv)
    except (OSError, ValueError) as exc:
        return _failed(
            _OPERATION,
            scope=scope,
            code="rna_editing.invalid_sites",
            message=f"Could not read known-site TSV: {exc}",
            backend="pysam",
        )
    try:
        import pysam
    except ImportError:
        return _failed(
            _OPERATION,
            scope=scope,
            code="rna_editing.pysam_not_installed",
            message="BAM/CRAM quantification requires pysam; install with `pip install 'organelleverse[qc]'`.",
            backend="pysam",
        )

    try:
        open_kwargs = (
            {"reference_filename": str(reference_fasta)} if reference_fasta is not None else {}
        )
        counts_by_site: dict[str, Counter[str]] = {site["site_id"]: Counter() for site in sites}
        with pysam.AlignmentFile(str(bam_path), "r", **open_kwargs) as alignment:
            for site in sites:
                contig = site["seqid"]
                if contig not in alignment.references:
                    raise ValueError(f"BAM/CRAM has no reference sequence {contig!r}")
                pos = int(site["position"]) - 1
                for column in alignment.pileup(
                    contig,
                    pos,
                    pos + 1,
                    truncate=True,
                    stepper="all",
                    min_mapping_quality=0,
                    min_base_quality=0,
                    max_depth=1_000_000,
                    ignore_overlaps=False,
                    ignore_orphans=False,
                    compute_baq=False,
                ):
                    if column.reference_pos != pos:
                        continue
                    for pileup_read in column.pileups:
                        read = pileup_read.alignment
                        if pileup_read.is_del or pileup_read.is_refskip:
                            continue
                        if (
                            read.is_unmapped
                            or read.is_secondary
                            or read.is_supplementary
                            or read.is_qcfail
                        ):
                            continue
                        if exclude_duplicates and read.is_duplicate:
                            continue
                        if read.mapping_quality < min_mapping_quality:
                            continue
                        qpos = pileup_read.query_position
                        if qpos is None or read.query_qualities is None:
                            continue
                        if read.query_qualities[qpos] < min_base_quality:
                            continue
                        base = read.query_sequence[qpos].upper()
                        if base in "ACGT":
                            counts_by_site[site["site_id"]][base] += 1
    except (OSError, ValueError, KeyError) as exc:
        return _failed(
            _OPERATION,
            scope=scope,
            code="rna_editing.pileup_failed",
            message=f"Could not quantify BAM/CRAM pileup: {exc}",
            backend="pysam",
        )

    rows: list[dict[str, object]] = []
    for site in sites:
        counts = counts_by_site[site["site_id"]]
        ref_count = counts[site["ref"]]
        edited_count = counts[site["edited"]]
        depth = sum(counts.values())
        other_count = depth - ref_count - edited_count
        rows.append(
            {
                **site,
                "ref_count": ref_count,
                "edited_count": edited_count,
                "other_count": other_count,
                "depth": depth,
                "editing_fraction": round(edited_count / depth, 6) if depth else None,
                "ci95_lower": round(_wilson(edited_count, depth)[0], 6) if depth else None,
                "ci95_upper": round(_wilson(edited_count, depth)[1], 6) if depth else None,
            }
        )
    total_sites = len(rows)
    covered = sum(row["depth"] > 0 for row in rows)
    total_edited = sum(int(row["edited_count"]) for row in rows)
    return OrganelleResult(
        operation_id="rna_editing.quantify_efficiency",
        operation_version=_VERSION,
        scope=scope,
        status="ok",
        summary_text=f"Quantified {covered}/{total_sites} known RNA editing sites from RNA-seq pileups.",
        metrics={
            "site_count": total_sites,
            "covered_sites": covered,
            "total_edited_reads": total_edited,
            "sites": rows,
        },
        findings=(
            Finding(code="rna_editing.quantified_sites", metric="covered_sites", value=covered),
        ),
        flags=("known_sites_only",),
        provenance=_provenance(
            _OPERATION,
            parameters=params,
            backend="pysam",
            software_versions={"pysam": pysam.__version__},
        ),
    )


def _read_sites(path: str | Path) -> list[dict[str, str | int]]:
    import csv

    required = {"site_id", "seqid", "position", "ref", "edited", "strand"}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        sites: list[dict[str, str | int]] = []
        seen: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            site_id = (row.get("site_id") or "").strip()
            seqid = (row.get("seqid") or "").strip()
            ref = (row.get("ref") or "").strip().upper()
            edited = (row.get("edited") or "").strip().upper()
            strand = (row.get("strand") or "").strip()
            try:
                position = int(row.get("position", ""))
            except ValueError as exc:
                raise ValueError(f"line {line_number}: position must be an integer") from exc
            if not site_id or not seqid or position < 1:
                raise ValueError(
                    f"line {line_number}: site_id/seqid must be non-empty and position >= 1"
                )
            if site_id in seen:
                raise ValueError(f"line {line_number}: duplicate site_id {site_id!r}")
            if (
                ref not in "ACGT"
                or edited not in "ACGT"
                or len(ref) != 1
                or len(edited) != 1
                or ref == edited
            ):
                raise ValueError(
                    f"line {line_number}: ref and edited must be distinct single A/C/G/T bases"
                )
            if strand not in ("+", "-"):
                raise ValueError(f"line {line_number}: strand must be '+' or '-'")
            seen.add(site_id)
            sites.append(
                {
                    "site_id": site_id,
                    "seqid": seqid,
                    "position": position,
                    "ref": ref,
                    "edited": edited,
                    "strand": strand,
                }
            )
    if not sites:
        raise ValueError("site table contains no sites")
    return sites


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials and trials > 0")
    p = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)
