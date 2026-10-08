"""RNA-seq summaries for annotated organelle introns and exons."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Literal

from ..core.result import ErrorDetail, Finding, OrganelleResult

_SCOPE = Literal["mitochondrion", "plastid"]


def quantify_splicing_efficiency(
    bam_path: str | Path,
    introns_tsv: str | Path,
    *,
    scope: _SCOPE = "mitochondrion",
    min_mapping_quality: int = 20,
    min_anchor: int = 8,
    exclude_duplicates: bool = True,
    reference_fasta: str | Path | None = None,
) -> OrganelleResult:
    """Estimate annotated intron excision from exact CIGAR N junction support.

    TSV columns: intron_id, gene_id, seqid, start, end, strand. Coordinates are
    1-based inclusive on the forward genomic reference for *both* strands;
    negative-strand introns still require start <= end. A spliced read must
    have one CIGAR N exactly matching the annotated interval and at least
    ``min_anchor`` aligned reference bases on each side. An unspliced read must
    have a single contiguous aligned CIGAR block covering the entire intron and
    the same anchor on both sides. Other junctions are not assigned to either
    class. Efficiency is spliced/(spliced+unspliced), conditional on these
    informative alignments; it is not a transcript abundance estimate.

    The BAM/CRAM must be coordinate-sorted and indexed. Secondary, supplementary,
    unmapped, QC-failed, low-MAPQ and (by default) duplicate alignments are
    excluded. Intron strand is retained as annotation; coordinates and CIGAR
    comparisons are always in reference-forward orientation.
    """
    operation = "rna_seq.quantify_splicing_efficiency"
    if scope not in ("mitochondrion", "plastid"):
        return _failed(operation, "scope must be 'mitochondrion' or 'plastid'.")
    if min_mapping_quality < 0 or min_anchor < 1:
        return _failed(
            operation, "min_mapping_quality must be >= 0 and min_anchor >= 1.", scope=scope
        )
    try:
        introns = _read_features(introns_tsv, kind="intron")
    except (OSError, ValueError) as exc:
        return _failed(operation, f"Could not read intron TSV: {exc}", scope=scope)
    try:
        import pysam
    except ImportError:
        return _failed(
            operation, "BAM/CRAM analysis requires pysam; install organelleverse[qc].", scope=scope
        )

    rows: list[dict[str, object]] = []
    open_kwargs = {"reference_filename": str(reference_fasta)} if reference_fasta else {}
    try:
        with pysam.AlignmentFile(str(bam_path), "r", **open_kwargs) as bam:
            for intron in introns:
                if intron["seqid"] not in bam.references:
                    raise ValueError(f"BAM has no reference sequence {intron['seqid']!r}")
                start0, end0 = int(intron["start"]) - 1, int(intron["end"])
                spliced = unspliced = 0
                query_start = max(0, start0 - min_anchor)
                query_end = end0 + min_anchor
                for read in bam.fetch(str(intron["seqid"]), query_start, query_end):
                    if not _usable(read, min_mapping_quality, exclude_duplicates):
                        continue
                    evidence = _intron_evidence(read, start0, end0, min_anchor)
                    spliced += evidence == "spliced"
                    unspliced += evidence == "unspliced"
                informative = spliced + unspliced
                rows.append(
                    {
                        **intron,
                        "spliced_reads": spliced,
                        "unspliced_reads": unspliced,
                        "informative_reads": informative,
                        "excision_efficiency": round(spliced / informative, 6)
                        if informative
                        else None,
                    }
                )
    except (OSError, ValueError, KeyError) as exc:
        return _failed(operation, f"Could not analyze indexed BAM/CRAM: {exc}", scope=scope)

    measured = [r for r in rows if r["informative_reads"]]
    return OrganelleResult(
        operation_id=operation,
        operation_version="1.0",
        scope=scope,
        status="ok",
        summary_text=f"Estimated excision efficiency for {len(measured)}/{len(rows)} annotated introns.",
        metrics={
            "introns": rows,
            "annotated_introns": len(rows),
            "introns_with_evidence": len(measured),
            "efficiency_definition": "exact annotated CIGAR N / (exact N + continuous intron-spanning alignments)",
        },
        findings=(
            Finding(
                code="rna_seq.introns_quantified",
                metric="introns_with_evidence",
                value=len(measured),
            ),
        ),
        flags=("annotated_introns_only", "alignment_fraction_not_transcript_abundance"),
    )


def quantify_exon_expression(
    bam_path: str | Path,
    exons_tsv: str | Path,
    *,
    scope: _SCOPE = "mitochondrion",
    min_mapping_quality: int = 20,
    exclude_duplicates: bool = True,
    reference_fasta: str | Path | None = None,
) -> OrganelleResult:
    """Summarize annotated gene exon-overlapping reads and library-size CPM.

    TSV columns: feature_id, gene_id, seqid, start, end, strand. Coordinates
    are 1-based inclusive on the forward genomic reference on both strands.
    Counts are unique primary alignments per gene (paired mates count as two
    reads); a read overlapping multiple exons of one gene counts once. A read
    overlapping exons from different genes can count for each such gene.
    CPM uses all primary mapped reads passing the same MAPQ/duplicate filters
    across the BAM, not only organelle reads. This is a simple library-size
    normalized count, not TPM and not isoform-level expression.
    """
    operation = "rna_seq.quantify_exon_expression"
    if scope not in ("mitochondrion", "plastid"):
        return _failed(operation, "scope must be 'mitochondrion' or 'plastid'.")
    if min_mapping_quality < 0:
        return _failed(operation, "min_mapping_quality must be >= 0.", scope=scope)
    try:
        exons = _read_features(exons_tsv, kind="exon")
    except (OSError, ValueError) as exc:
        return _failed(operation, f"Could not read exon TSV: {exc}", scope=scope)
    try:
        import pysam
    except ImportError:
        return _failed(
            operation, "BAM/CRAM analysis requires pysam; install organelleverse[qc].", scope=scope
        )

    exon_bins = _build_exon_bins(exons)
    read_counts: defaultdict[str, int] = defaultdict(int)
    open_kwargs = {"reference_filename": str(reference_fasta)} if reference_fasta else {}
    try:
        with pysam.AlignmentFile(str(bam_path), "r", **open_kwargs) as bam:
            if not bam.has_index():
                raise ValueError("BAM/CRAM must be indexed")
            missing = sorted(set(exon_bins) - set(bam.references))
            if missing:
                raise ValueError(f"BAM has no reference sequence {missing[0]!r}")
            total_mapped = 0
            for read in bam.fetch(until_eof=True):
                if not _usable(read, min_mapping_quality, exclude_duplicates):
                    continue
                total_mapped += 1
                if read.reference_name not in exon_bins:
                    continue
                for gene in _overlapping_genes(read, exon_bins[read.reference_name]):
                    read_counts[gene] += 1
    except (OSError, ValueError, KeyError) as exc:
        return _failed(operation, f"Could not analyze indexed BAM/CRAM: {exc}", scope=scope)

    genes = sorted({str(exon["gene_id"]) for exon in exons})
    rows = [
        {
            "gene_id": gene,
            "read_count": read_counts[gene],
            "cpm": round(read_counts[gene] * 1_000_000 / total_mapped, 6)
            if total_mapped
            else None,
        }
        for gene in genes
    ]
    return OrganelleResult(
        operation_id=operation,
        operation_version="1.0",
        scope=scope,
        status="ok",
        summary_text=f"Summarized exon-overlapping reads for {len(rows)} genes; denominator {total_mapped} mapped reads.",
        metrics={
            "genes": rows,
            "gene_count": len(rows),
            "library_mapped_reads": total_mapped,
            "normalization": "CPM over all primary mapped reads passing filters",
        },
        findings=(Finding(code="rna_seq.genes_quantified", metric="gene_count", value=len(rows)),),
        flags=("annotated_exons_only", "read_counts_not_tpm"),
    )


def _failed(
    operation: str, message: str, *, scope: _SCOPE | Literal["none"] = "none"
) -> OrganelleResult:
    return OrganelleResult(
        operation_id=operation,
        operation_version="1.0",
        scope=scope,
        status="failed",
        summary_text=message,
        errors=(ErrorDetail(code="rna_seq.input_or_alignment_error", message=message),),
    )


def _usable(read: object, min_mapq: int, exclude_duplicates: bool) -> bool:
    return not (
        read.is_unmapped
        or read.is_secondary
        or read.is_supplementary
        or read.is_qcfail
        or (exclude_duplicates and read.is_duplicate)
        or read.mapping_quality < min_mapq
    )


def _intron_evidence(read: object, intron_start0: int, intron_end0: int, anchor: int) -> str | None:
    """Classify one read only when it proves a matching junction or full intron."""
    refpos = read.reference_start
    cigar = read.cigartuples or ()
    aligned_start = None
    for index, (op, length) in enumerate(cigar):
        if op == 3:  # CIGAR N, skipped reference interval
            aligned_start = None
            if refpos == intron_start0 and refpos + length == intron_end0:
                left = _aligned_anchor(cigar[:index], from_right=True)
                right = _aligned_anchor(cigar[index + 1 :], from_right=False)
                if left >= anchor and right >= anchor:
                    return "spliced"
            refpos += length
        elif op in (0, 2, 7, 8):  # M, D, =, X consume reference
            if op in (0, 7, 8):
                if aligned_start is None:
                    aligned_start = refpos
                if (
                    aligned_start <= intron_start0 - anchor
                    and refpos + length >= intron_end0 + anchor
                ):
                    return "unspliced"
            else:  # A deletion cannot establish coverage through the intron.
                aligned_start = None
            refpos += length
        elif op not in (1, 6):  # insertion/padding preserve reference adjacency
            aligned_start = None
    return None


def _aligned_anchor(ops: tuple[tuple[int, int], ...], *, from_right: bool) -> int:
    ordered = reversed(ops) if from_right else iter(ops)
    total = 0
    for op, length in ordered:
        if op in (0, 7, 8):
            total += length
        elif op in (
            1,
            4,
            5,
            6,
        ):  # query-only or clipping operations do not break reference adjacency
            continue
        else:
            break
    return total


_EXON_BIN_SIZE = 4096


def _build_exon_bins(
    exons: list[dict[str, str | int]],
) -> dict[str, dict[int, list[tuple[int, int, str]]]]:
    """Index exons in fixed genomic bins without retaining any read identifiers."""
    index: dict[str, dict[int, list[tuple[int, int, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for exon in exons:
        seqid = str(exon["seqid"])
        start0, end0 = int(exon["start"]) - 1, int(exon["end"])
        interval = (start0, end0, str(exon["gene_id"]))
        for bin_number in range(start0 // _EXON_BIN_SIZE, (end0 - 1) // _EXON_BIN_SIZE + 1):
            index[seqid][bin_number].append(interval)
    return {seqid: dict(bins) for seqid, bins in index.items()}


def _overlapping_genes(
    read: object, bins: dict[int, list[tuple[int, int, str]]]
) -> set[str]:
    genes: set[str] = set()
    for block_start, block_end in read.get_blocks():
        for bin_number in range(
            block_start // _EXON_BIN_SIZE, (block_end - 1) // _EXON_BIN_SIZE + 1
        ):
            for exon_start, exon_end, gene in bins.get(bin_number, ()):
                if exon_start < block_end and exon_end > block_start:
                    genes.add(gene)
    return genes


def _read_features(path: str | Path, *, kind: str) -> list[dict[str, str | int]]:
    required = (
        {"intron_id", "gene_id", "seqid", "start", "end", "strand"}
        if kind == "intron"
        else {"feature_id", "gene_id", "seqid", "start", "end", "strand"}
    )
    id_column = "intron_id" if kind == "intron" else "feature_id"
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        features: list[dict[str, str | int]] = []
        seen: set[str] = set()
        for line, row in enumerate(reader, 2):
            identifier, gene, seqid = (
                (row.get(id_column) or "").strip(),
                (row.get("gene_id") or "").strip(),
                (row.get("seqid") or "").strip(),
            )
            strand = (row.get("strand") or "").strip()
            try:
                start, end = int(row.get("start", "")), int(row.get("end", ""))
            except ValueError as exc:
                raise ValueError(f"line {line}: start and end must be integers") from exc
            if (
                not identifier
                or not gene
                or not seqid
                or start < 1
                or end < start
                or strand not in ("+", "-")
            ):
                raise ValueError(f"line {line}: invalid ID, coordinates, or strand")
            if identifier in seen:
                raise ValueError(f"line {line}: duplicate {id_column} {identifier!r}")
            seen.add(identifier)
            features.append(
                {
                    id_column: identifier,
                    "gene_id": gene,
                    "seqid": seqid,
                    "start": start,
                    "end": end,
                    "strand": strand,
                }
            )
        return features
