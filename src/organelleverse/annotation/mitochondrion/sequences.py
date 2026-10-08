"""Sequence extraction from annotated genomes.

Extracts:
- CDS nucleotide sequences
- Protein sequences (translated with NCBI Table 1)
- tRNA sequences
- rRNA sequences
- Intron sequences (between exons)
- Gene sequences (full gene including introns)
"""

from __future__ import annotations

import logging
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .models.feature import rRNAAnnotation, tRNAAnnotation
from .models.gene import GeneAnnotation, Strand
from .models.genome import GenomeSequence
from .pcg import translate_sequence

logger = logging.getLogger(__name__)


def extract_all(
    annotations: list[GeneAnnotation],
    trna_annotations: list[tRNAAnnotation],
    rrna_annotations: list[rRNAAnnotation],
    genome: GenomeSequence,
    output_dir: Path,
    project_name: str,
) -> dict[str, Path]:
    """Extract all sequence types and write FASTA files.

    Returns:
        Dict mapping sequence type to output file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    # CDS
    cds_path = output_dir / f"{project_name}.CDS.fasta"
    _write_cds(annotations, genome, cds_path)
    files["CDS"] = cds_path

    # Protein
    prot_path = output_dir / f"{project_name}.Protein.fasta"
    _write_protein(annotations, genome, prot_path)
    files["Protein"] = prot_path

    # tRNA
    trna_path = output_dir / f"{project_name}.tRNA.fasta"
    _write_trna(trna_annotations, genome, trna_path)
    files["tRNA"] = trna_path

    # rRNA
    rrna_path = output_dir / f"{project_name}.rRNA.fasta"
    _write_rrna(rrna_annotations, genome, rrna_path)
    files["rRNA"] = rrna_path

    # Gene (full gene including introns)
    gene_path = output_dir / f"{project_name}.Gene.fasta"
    _write_gene(annotations, genome, gene_path)
    files["Gene"] = gene_path

    # Intron
    intron_path = output_dir / f"{project_name}.intron.fasta"
    _write_introns(annotations, genome, intron_path)
    files["intron"] = intron_path

    logger.info(f"Extracted sequences to {output_dir}")
    return files


def _get_gene_sequence(ann: GeneAnnotation, genome: GenomeSequence) -> str:
    """Get full gene sequence (genomic range, oriented)."""
    return genome.get_sequence_for_range(ann.genomic_start, ann.genomic_end, ann.strand)


def _get_cds_sequence(ann: GeneAnnotation, genome: GenomeSequence) -> str:
    """Get CDS sequence (exons concatenated, oriented).

    For multi-exon genes: concatenate exon sequences in order,
    skipping introns. For minus strand, reverse order of exons.
    """
    if len(ann.exons) == 1:
        return genome.get_sequence_for_range(ann.exons[0].start, ann.exons[0].end, ann.strand)

    # Multi-exon: collect exon sequences in transcription order
    exon_seqs = []
    exons = sorted(ann.exons, key=lambda e: e.start)

    for exon in exons:
        seq = genome.subsequence(exon.start, exon.end)
        if ann.strand == Strand.MINUS:
            comp = str.maketrans("ATGCatgcNn", "TACGtacgNn")
            seq = seq.translate(comp)[::-1]
        exon_seqs.append(seq)

    # For minus strand, reverse order
    if ann.strand == Strand.MINUS:
        exon_seqs = exon_seqs[::-1]

    return "".join(exon_seqs)


def _write_cds(annotations: list, genome: GenomeSequence, path: Path) -> None:
    """Write CDS nucleotide sequences."""
    records = []
    for ann in annotations:
        if ann.gene_type != "CDS":
            continue
        seq = _get_cds_sequence(ann, genome)
        if seq:
            records.append(
                SeqRecord(
                    Seq(seq),
                    id=ann.gene_name,
                    description=f"CDS {ann.product}; length={len(seq)}bp",
                )
            )
    SeqIO.write(records, str(path), "fasta")
    logger.info(f"CDS: {len(records)} sequences -> {path}")


def _write_protein(annotations: list, genome: GenomeSequence, path: Path) -> None:
    """Write protein sequences (translated with NCBI Table 1)."""
    records = []
    for ann in annotations:
        if ann.gene_type != "CDS":
            continue
        nt_seq = _get_cds_sequence(ann, genome)
        if not nt_seq:
            continue
        protein = translate_sequence(nt_seq)
        if protein and len(protein) > 10:
            records.append(
                SeqRecord(
                    Seq(protein),
                    id=ann.gene_name,
                    description=f"protein {ann.product}; length={len(protein)}aa",
                )
            )
    SeqIO.write(records, str(path), "fasta")
    logger.info(f"Protein: {len(records)} sequences -> {path}")


def _write_trna(trna_annotations: list, genome: GenomeSequence, path: Path) -> None:
    """Write tRNA sequences."""
    records = []
    for trna in trna_annotations:
        seq = _get_gene_sequence(trna, genome)
        if seq:
            records.append(
                SeqRecord(
                    Seq(seq),
                    id=trna.gene_name,
                    description=f"tRNA-{trna.amino_acid}-{trna.anticodon}; length={len(seq)}bp",
                )
            )
    SeqIO.write(records, str(path), "fasta")
    logger.info(f"tRNA: {len(records)} sequences -> {path}")


def _write_rrna(rrna_annotations: list, genome: GenomeSequence, path: Path) -> None:
    """Write rRNA sequences."""
    records = []
    for rrna in rrna_annotations:
        seq = _get_gene_sequence(rrna, genome)
        if seq:
            records.append(
                SeqRecord(
                    Seq(seq),
                    id=rrna.gene_name,
                    description=f"rRNA {rrna.rrna_type}; length={len(seq)}bp",
                )
            )
    SeqIO.write(records, str(path), "fasta")
    logger.info(f"rRNA: {len(records)} sequences -> {path}")


def _write_gene(annotations: list, genome: GenomeSequence, path: Path) -> None:
    """Write full gene sequences (including introns)."""
    records = []
    for ann in annotations:
        seq = _get_gene_sequence(ann, genome)
        if seq:
            records.append(
                SeqRecord(
                    Seq(seq),
                    id=ann.gene_name,
                    description=f"gene {ann.product}; length={len(seq)}bp",
                )
            )
    SeqIO.write(records, str(path), "fasta")
    logger.info(f"Gene: {len(records)} sequences -> {path}")


def _write_introns(annotations: list, genome: GenomeSequence, path: Path) -> None:
    """Write intron sequences (regions between exons)."""
    records = []
    intron_num = 0
    for ann in annotations:
        if len(ann.exons) < 2:
            continue

        exons = sorted(ann.exons, key=lambda e: e.start)
        for i in range(len(exons) - 1):
            intron_start = exons[i].end + 1
            intron_end = exons[i + 1].start - 1
            if intron_end <= intron_start:
                continue

            intron_num += 1
            seq = genome.get_sequence_for_range(intron_start, intron_end, ann.strand)
            if seq:
                records.append(
                    SeqRecord(
                        Seq(seq),
                        id=f"{ann.gene_name}_intron_{i + 1}",
                        description=f"intron {i + 1} of {ann.gene_name}; "
                        f"pos={intron_start}-{intron_end}; length={len(seq)}bp",
                    )
                )

    SeqIO.write(records, str(path), "fasta")
    logger.info(f"Intron: {len(records)} sequences -> {path}")
