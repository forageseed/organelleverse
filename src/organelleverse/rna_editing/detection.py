"""Discover C-to-U candidates from quality-filtered RNA alignment evidence.

Coordinates and alleles always use the genomic reference, including G>A on
minus transcripts. No known-site list or sequence predictor enters calling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq

from ..core.errors import OrganelleDependencyError, OrganelleInputError, OrganelleParameterError
from ..core.result import Finding, OrganelleResult
from ..rna_seq.quantification import _usable

_BASES = "ACGT"
_COMPLEMENT = str.maketrans("ACGT", "TGCA")
_LIBRARY = Literal["unstranded", "fr-firststrand", "fr-secondstrand"]


def detect_editing_sites(
    bam_path: str | Path,
    reference_fasta: str | Path,
    *,
    annotation_genbank: str | Path | None = None,
    dna_bam_path: str | Path | None = None,
    scope: Literal["mitochondrion", "plastid"] = "mitochondrion",
    library_type: _LIBRARY = "unstranded",
    min_mapping_quality: int = 25,
    min_base_quality: int = 25,
    min_depth: int = 10,
    min_edited_reads: int = 3,
    min_editing_fraction: float = 0.1,
    trim_read_ends: int = 5,
    exclude_duplicates: bool = True,
    min_dna_depth: int = 10,
    max_dna_nonref_fraction: float = 0.1,
) -> OrganelleResult:
    """Scan indexed, coordinate-sorted BAM for candidate C-to-U editing sites.

    Depth is the number of passing A/C/G/T aligned observations; paired mates
    count separately, including overlaps. BAQ, overlap quality adjustment and
    depth downsampling are not applied. End trimming removes the first/last N
    sequenced bases (soft clips count toward N). CIGAR deletions and skips do
    not contribute. Missing qualities and nonprimary/QC-failed reads are excluded.

    First-strand libraries: R1/single reads are antisense, R2 sense; second-
    strand reverses this rule. Stranded depths are transcript-specific. For
    unstranded data C>T implies '+' and G>A implies '-' but this is only an
    assumed editing direction, not measured transcript orientation.

    REDItools defaults motivate Q25/MAPQ25, depth 10, 3 edited reads and fraction
    0.1. Symmetric 5-base end trimming is an explicit OrganelleVerse policy
    (REDItools defaults to 0). DNA depth 10 and nonreference fraction 0.1 follow
    REDItoolDnaRna. With DNA supplied, insufficient coverage excludes a candidate;
    ANY nonreference allele fraction >= the DNA threshold excludes it as well.
    See docs/operations/rna-editing-denovo.md for sources and limitations.
    """
    thresholds = dict(
        min_mapping_quality=min_mapping_quality,
        min_base_quality=min_base_quality,
        min_depth=min_depth,
        min_edited_reads=min_edited_reads,
        min_editing_fraction=min_editing_fraction,
        trim_read_ends=trim_read_ends,
        exclude_duplicates=exclude_duplicates,
        min_dna_depth=min_dna_depth,
        max_dna_nonref_fraction=max_dna_nonref_fraction,
    )
    if scope not in ("mitochondrion", "plastid") or library_type not in (
        "unstranded",
        "fr-firststrand",
        "fr-secondstrand",
    ):
        raise OrganelleParameterError(
            code="rna_editing.invalid_parameter", message="Invalid scope or library_type."
        )
    for name in (
        "min_mapping_quality",
        "min_base_quality",
        "trim_read_ends",
        "min_depth",
        "min_edited_reads",
        "min_dna_depth",
    ):
        value = thresholds[name]
        lower = 1 if name in ("min_depth", "min_edited_reads", "min_dna_depth") else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise OrganelleParameterError(
                code="rna_editing.invalid_parameter",
                message=f"{name} must be an integer >= {lower}.",
            )
    for name in ("min_editing_fraction", "max_dna_nonref_fraction"):
        if not 0 < thresholds[name] <= 1:
            raise OrganelleParameterError(
                code="rna_editing.invalid_parameter", message=f"{name} must be in (0, 1]."
            )
    try:
        import pysam
    except ImportError as exc:
        raise OrganelleDependencyError(
            code="rna_editing.pysam_not_installed",
            message="BAM detection requires pysam; install with `pip install 'organelleverse[qc]'`.",
        ) from exc

    sites = []
    background = []
    exclusions = {"insufficient_dna_depth": 0, "dna_nonreference": 0}
    callable_positions = 0
    try:
        references = SeqIO.to_dict(SeqIO.parse(str(reference_fasta), "fasta"))
        if not references or any(not len(r) for r in references.values()):
            raise ValueError("Reference FASTA must contain nonempty records with unique IDs.")
        annotations = _read_annotation(annotation_genbank, references)
        from contextlib import ExitStack

        with ExitStack() as stack:
            rna = stack.enter_context(pysam.AlignmentFile(str(bam_path), "rb"))
            dna = (
                stack.enter_context(pysam.AlignmentFile(str(dna_bam_path), "rb"))
                if dna_bam_path is not None
                else None
            )
            for bam in (rna, dna):
                if bam is None:
                    continue
                if not bam.is_bam or not bam.has_index():
                    raise ValueError("Input must be indexed, coordinate-sorted BAM.")
                for seqid, record in references.items():
                    if seqid not in bam.references or bam.get_reference_length(seqid) != len(
                        record
                    ):
                        raise ValueError(f"BAM reference name/length does not match FASTA: {seqid}")
            for seqid, record in references.items():
                sequence = str(record.seq).upper()
                rna_counts = _count_alignments(rna, seqid, len(record), library_type, thresholds)
                dna_counts = (
                    _count_alignments(dna, seqid, len(record), "unstranded", thresholds)[0]
                    if dna is not None
                    else None
                )
                reference_bases = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
                for group, counts in enumerate(rna_counts):
                    strand = "." if library_type == "unstranded" else ("+" if group == 0 else "-")
                    depths = counts.sum(axis=1)
                    covered = depths >= min_depth
                    callable_positions += int(
                        np.count_nonzero(covered & np.isin(reference_bases, list(b"ACGT")))
                    )
                    for ref_index, ref in enumerate(_BASES):
                        positions = np.flatnonzero(covered & (reference_bases == ord(ref)))
                        for alt_index, alt in enumerate(_BASES):
                            if alt == ref:
                                continue
                            alt_counts = counts[positions, alt_index]
                            passing = (alt_counts >= min_edited_reads) & (
                                alt_counts / depths[positions] >= min_editing_fraction
                            )
                            passing_positions = positions[passing]
                            transcript_change = (
                                f"{ref}>{alt}"
                                if strand != "-"
                                else f"{ref.translate(_COMPLEMENT)}>{alt.translate(_COMPLEMENT)}"
                            )
                            is_c_to_u = ((ref, alt) == ("C", "T") and strand in (".", "+")) or (
                                (ref, alt) == ("G", "A") and strand in (".", "-")
                            )
                            background.append(
                                {
                                    "seqid": seqid,
                                    "strand": strand,
                                    "genomic_change": f"{ref}>{alt}",
                                    "transcript_change": transcript_change
                                    if strand != "."
                                    else None,
                                    "is_c_to_u_compatible": is_c_to_u,
                                    "callable_positions": len(positions),
                                    "depth_sum": int(depths[positions].sum()),
                                    "mismatch_reads": int(alt_counts.sum()),
                                    "passing_sites": len(passing_positions),
                                }
                            )
                            if not is_c_to_u:
                                continue
                            for pos in passing_positions:
                                dna_depth = dna_fraction = None
                                if dna_counts is not None:
                                    dna_depth = int(dna_counts[pos].sum())
                                    if dna_depth < min_dna_depth:
                                        exclusions["insufficient_dna_depth"] += 1
                                        continue
                                    dna_fraction = (
                                        dna_depth - int(dna_counts[pos, ref_index])
                                    ) / dna_depth
                                    if dna_fraction >= max_dna_nonref_fraction:
                                        exclusions["dna_nonreference"] += 1
                                        continue
                                site_strand = "+" if ref == "C" else "-"
                                depth = int(depths[pos])
                                edited = int(counts[pos, alt_index])
                                sites.append(
                                    {
                                        "seqid": seqid,
                                        "position": int(pos) + 1,
                                        "strand": site_strand,
                                        "strand_evidence": "substitution_assumed"
                                        if strand == "."
                                        else "library",
                                        "ref": ref,
                                        "edited": alt,
                                        "transcript_ref": "C",
                                        "transcript_edited": "U",
                                        "depth": depth,
                                        "ref_count": int(counts[pos, ref_index]),
                                        "edited_count": edited,
                                        "other_count": depth - edited - int(counts[pos, ref_index]),
                                        "editing_fraction": edited / depth,
                                        "dna_depth": dna_depth,
                                        "dna_nonref_fraction": dna_fraction,
                                        "effects": _effects(
                                            annotations.get(seqid, []),
                                            record.seq,
                                            int(pos),
                                            site_strand,
                                            alt,
                                        ),
                                    }
                                )
    except (OSError, ValueError, KeyError) as exc:
        raise OrganelleInputError(
            code="rna_editing.invalid_alignment_input",
            message=f"Cannot detect editing sites: {exc}",
        ) from exc

    sites.sort(key=lambda row: (row["seqid"], row["position"], row["strand"]))
    flags = ["candidate_sites_not_validated_editing", "mates_counted_separately"]
    if library_type == "unstranded":
        flags.append("transcript_strand_assumed_from_substitution")
    if dna_bam_path is None:
        flags.append("no_dna_snp_exclusion")
    return OrganelleResult(
        operation_id="rna_editing.detect_editing_sites",
        operation_version="1.0",
        scope=scope,
        status="ok",
        summary_text=f"Detected {len(sites)} candidate C-to-U RNA editing sites.",
        metrics={
            "site_count": len(sites),
            "sites": sites,
            "callable_position_strands": callable_positions,
            "mismatch_statistics": background,
            "dna_exclusions": exclusions,
            "parameters": {**thresholds, "library_type": library_type},
        },
        findings=(
            Finding(code="rna_editing.candidate_sites", metric="site_count", value=len(sites)),
        ),
        flags=tuple(flags),
    )


def _count_alignments(bam, seqid, length, library_type, thresholds):
    """Accumulate aligned query bases without a pileup depth cap or BAQ edits."""
    counts = np.zeros((1 if library_type == "unstranded" else 2, length, 4), dtype=np.int64)
    lookup = np.full(256, -1, dtype=np.int8)
    lookup[list(b"ACGT")] = np.arange(4)
    trim = thresholds["trim_read_ends"]
    for read in bam.fetch(seqid):
        if not _usable(read, thresholds["min_mapping_quality"], thresholds["exclude_duplicates"]):
            continue
        if read.query_sequence is None or read.query_qualities is None:
            continue
        pairs = np.asarray(read.get_aligned_pairs(matches_only=True), dtype=np.int64)
        if not len(pairs):
            continue
        query, reference = pairs.T
        bases = lookup[np.frombuffer(read.query_sequence.upper().encode("ascii"), dtype=np.uint8)]
        qualities = np.asarray(read.query_qualities)
        keep = (
            (query >= trim)
            & (query < len(bases) - trim)
            & (bases[query] >= 0)
            & (qualities[query] >= thresholds["min_base_quality"])
        )
        group = 0
        if library_type != "unstranded":
            if read.is_paired and read.is_read1 == read.is_read2:
                raise ValueError("Stranded paired reads must identify exactly one of R1/R2.")
            antisense = (library_type == "fr-firststrand") != (read.is_paired and read.is_read2)
            group = int(read.is_reverse != antisense)
        np.add.at(counts[group], (reference[keep], bases[query[keep]]), 1)
    return counts


def _read_annotation(path, references):
    if path is None:
        return {}
    records = SeqIO.to_dict(SeqIO.parse(str(path), "genbank"))
    if not records:
        raise ValueError("GenBank contains no records.")
    annotations = {}
    for seqid, record in records.items():
        if seqid not in references or record.seq.upper() != references[seqid].seq.upper():
            raise ValueError(f"GenBank ID/sequence does not match FASTA: {seqid}")
        features = [f for f in record.features if f.type in ("gene", "CDS")]
        if any(
            f.location is None or any(p.ref is not None for p in f.location.parts) for f in features
        ):
            raise ValueError("Gene/CDS features require local GenBank locations.")
        annotations[seqid] = features
    return annotations


def _effects(features, sequence, pos, strand, alt):
    """Single-site consequences in transcript order, including joined CDSs."""
    effects = []
    sign = 1 if strand == "+" else -1
    cds_genes = set()
    for feature in features:
        if feature.type != "CDS" or feature.location.strand != sign or pos not in feature.location:
            continue
        gene = feature.qualifiers.get("gene", feature.qualifiers.get("locus_tag", [""]))[0]
        cds_genes.add(gene)
        positions = list(feature.location)
        offset = positions.index(pos) - (int(feature.qualifiers.get("codon_start", [1])[0]) - 1)
        row = {
            "gene": gene,
            "feature_type": "CDS",
            "codon_number": None,
            "codon_position": None,
            "ref_codon": None,
            "edited_codon": None,
            "amino_acid_change": None,
        }
        cds = str(feature.extract(sequence)).upper()[
            int(feature.qualifiers.get("codon_start", [1])[0]) - 1 :
        ]
        if offset >= 0:
            start = offset // 3 * 3
            codon = cds[start : start + 3]
            if len(codon) == 3:
                edited = list(codon)
                edited[offset % 3] = alt if sign == 1 else alt.translate(_COMPLEMENT)
                edited = "".join(edited)
                table = int(feature.qualifiers.get("transl_table", [11])[0])
                row.update(
                    codon_number=offset // 3 + 1,
                    codon_position=offset % 3 + 1,
                    ref_codon=codon,
                    edited_codon=edited,
                    amino_acid_change=f"{Seq(codon).translate(table=table)}>{Seq(edited).translate(table=table)}",
                )
        effects.append(row)
    for feature in features:
        if feature.type == "gene" and feature.location.strand == sign and pos in feature.location:
            gene = feature.qualifiers.get("gene", feature.qualifiers.get("locus_tag", [""]))[0]
            if gene not in cds_genes:
                effects.append(
                    {
                        "gene": gene,
                        "feature_type": "gene",
                        "codon_number": None,
                        "codon_position": None,
                        "ref_codon": None,
                        "edited_codon": None,
                        "amino_acid_change": None,
                    }
                )
    return effects
