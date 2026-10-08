"""GFF3 / GenBank format output.

Handles conversion from internal GeneAnnotation objects to standard file formats:
- GFF3: Standard feature format
- GenBank: NCBI submission format (via Biopython)
- TBL: NCBI table format (intermediate for GenBank)
"""

from __future__ import annotations

import logging
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from .models.feature import rRNAAnnotation, tRNAAnnotation
from .models.gene import GeneAnnotation, Strand
from .models.genome import GenomeSequence
from .models.gff import GFF3Record

logger = logging.getLogger(__name__)


def write_gff3(
    annotations: list[GeneAnnotation],
    trna_annotations: list[tRNAAnnotation],
    rrna_annotations: list[rRNAAnnotation],
    genome: GenomeSequence,
    output_path: Path,
) -> None:
    """Write all annotations to GFF3 format.

    GFF3 spec: https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Track used gene IDs to ensure uniqueness
    id_counter: dict[str, int] = {}

    with open(output_path, "w") as f:
        # Header
        f.write("##gff-version 3\n")
        f.write(f"##sequence-region {genome.seqid} 1 {genome.length}\n")

        # Source feature
        topo = "circular" if genome.is_circular else "linear"
        f.write(
            f"{genome.seqid}\tOrganelleVerse\tregion\t1\t{genome.length}\t.\t+\t.\t"
            f"ID={genome.seqid};Name={genome.seqid};{topo}=true\n"
        )

        # Write each annotation
        for ann in _sorted_annotations(annotations, trna_annotations, rrna_annotations):
            # Assign unique gene ID
            base_name = ann.gene_name
            if base_name not in id_counter:
                id_counter[base_name] = 1
                gene_id = base_name
            else:
                id_counter[base_name] += 1
                gene_id = f"{base_name}.{id_counter[base_name]}"

            records = _annotation_to_gff3_records(ann, genome.seqid, gene_id)
            for rec in records:
                f.write(rec.to_line() + "\n")

        f.write("###\n")

    logger.info(
        f"Wrote GFF3: {output_path} ({len(annotations) + len(trna_annotations) + len(rrna_annotations)} features)"
    )


def write_genbank(
    annotations: list[GeneAnnotation],
    trna_annotations: list[tRNAAnnotation],
    rrna_annotations: list[rRNAAnnotation],
    genome: GenomeSequence,
    output_path: Path,
    organism: str = "",
    topology: str = "circular",
) -> None:
    """Write all annotations to GenBank format via Biopython."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seq = Seq(genome.sequence)
    record = SeqRecord(
        seq,
        id=genome.seqid,
        name=genome.seqid[:16],  # GenBank name max 16 chars
        description=f"{organism} mitochondrion, complete genome"
        if organism
        else "mitochondrion, complete genome",
    )
    record.annotations["topology"] = topology
    record.annotations["molecule_type"] = "DNA"
    record.annotations["data_file_division"] = "PLN"
    if organism:
        record.annotations["organism"] = organism

    # Add features
    sorted_anns = _sorted_annotations(annotations, trna_annotations, rrna_annotations)
    for ann in sorted_anns:
        features = _annotation_to_seq_features(ann, genome)
        record.features.extend(features)

    with open(output_path, "w") as f:
        SeqIO.write(record, f, "genbank")

    logger.info(f"Wrote GenBank: {output_path}")


def _sorted_annotations(
    annotations: list[GeneAnnotation],
    trna_annotations: list[tRNAAnnotation],
    rrna_annotations: list[rRNAAnnotation],
) -> list[GeneAnnotation]:
    """Combine and sort all annotations by genomic position."""
    all_anns = list(annotations) + list(trna_annotations) + list(rrna_annotations)
    return sorted(all_anns, key=lambda a: a.genomic_start)


def _annotation_to_gff3_records(
    ann: GeneAnnotation, seqid: str, gene_id: str | None = None
) -> list[GFF3Record]:
    """Convert a GeneAnnotation to GFF3 records."""
    records = []
    if gene_id is None:
        gene_id = ann.gene_name
    mrna_id = f"{gene_id}-RA"
    strand_str = ann.strand.symbol

    # Gene feature
    attrs = {"ID": gene_id, "Name": ann.gene_name}
    if ann.product:
        attrs["product"] = ann.product
    records.append(
        GFF3Record(
            seqid=seqid,
            type="gene",
            start=ann.genomic_start,
            end=ann.genomic_end,
            strand=strand_str,
            attributes=attrs,
        )
    )

    # mRNA / exon / CDS features (for CDS genes)
    if ann.gene_type == "CDS":
        records.append(
            GFF3Record(
                seqid=seqid,
                type="mRNA",
                start=ann.genomic_start,
                end=ann.genomic_end,
                strand=strand_str,
                attributes={"ID": mrna_id, "Parent": gene_id, "Name": ann.gene_name},
            )
        )

        # CDS phase: phase indicates how many bases at the start of the feature
        # should be skipped to reach the first in-frame codon.
        # For the first exon: phase=0. For subsequent exons: phase = (exon_len % 3) ? (3 - exon_len % 3) : 0
        # Actually phase = (cumulative_exon_length % 3) for the start of each exon.
        # But since exons are in coding frame, phase for exon i = (sum of lengths of exons 0..i-1) % 3
        cumulative_len = 0
        for i, exon in enumerate(ann.exons, 1):
            exon_len = exon.end - exon.start + 1
            exon_strand = exon.strand.symbol
            records.append(
                GFF3Record(
                    seqid=seqid,
                    type="exon",
                    start=exon.start,
                    end=exon.end,
                    strand=exon_strand,
                    attributes={"ID": f"{mrna_id}:exon:{i}", "Parent": mrna_id, "number": str(i)},
                )
            )
            phase = cumulative_len % 3
            records.append(
                GFF3Record(
                    seqid=seqid,
                    type="CDS",
                    start=exon.start,
                    end=exon.end,
                    strand=exon_strand,
                    phase=phase,
                    attributes={"ID": f"{mrna_id}:cds:{i}", "Parent": mrna_id},
                )
            )
            cumulative_len += exon_len

    elif ann.gene_type == "tRNA" or ann.gene_type == "rRNA":
        records.append(
            GFF3Record(
                seqid=seqid,
                type="exon",
                start=ann.genomic_start,
                end=ann.genomic_end,
                strand=strand_str,
                attributes={"ID": f"{gene_id}:exon", "Parent": gene_id},
            )
        )

    # Add notes as attributes
    if ann.notes:
        records[0].attributes["Note"] = "; ".join(ann.notes)
    if ann.exceptions:
        records[0].attributes["exception"] = "; ".join(ann.exceptions)

    return records


def _annotation_to_seq_features(
    ann: GeneAnnotation, genome: GenomeSequence | None = None
) -> list[SeqFeature]:
    """Convert a GeneAnnotation to Biopython SeqFeature objects."""
    features = []
    strand_biopython = 1 if ann.strand == Strand.PLUS else -1

    # Gene feature
    gene_feature = SeqFeature(
        FeatureLocation(ann.genomic_start - 1, ann.genomic_end, strand=strand_biopython),
        type="gene",
        qualifiers={
            "gene": [ann.gene_name],
            "locus_tag": [ann.gene_name],
        },
    )
    if ann.product:
        gene_feature.qualifiers["product"] = [ann.product]
    features.append(gene_feature)

    if ann.gene_type == "CDS":
        # CDS feature (may be multi-exon with join)
        if len(ann.exons) > 1:
            # Multi-exon: use CompoundLocation
            from Bio.SeqFeature import CompoundLocation

            locations = []
            for exon in ann.exons:
                exon_strand = 1 if exon.strand == Strand.PLUS else -1
                locations.append(FeatureLocation(exon.start - 1, exon.end, strand=exon_strand))
            # GeneAnnotation.exons is already biological 5'->3' order. Sorting
            # here corrupts trans-spliced genes whose numbered exons are not in
            # genomic order, including uniform minus-strand compounds.
            cds_loc = CompoundLocation(locations)
        else:
            exon = ann.exons[0]
            exon_strand = 1 if exon.strand == Strand.PLUS else -1
            cds_loc = FeatureLocation(exon.start - 1, exon.end, strand=exon_strand)

        cds_feature = SeqFeature(
            cds_loc,
            type="CDS",
            qualifiers={
                "gene": [ann.gene_name],
                "locus_tag": [ann.gene_name],
                "transl_table": [str(ann.transl_table)],
            },
        )
        if ann.product:
            cds_feature.qualifiers["product"] = [ann.product]
        if ann.notes:
            cds_feature.qualifiers["note"] = ann.notes
        if ann.exceptions:
            cds_feature.qualifiers["exception"] = ann.exceptions
        if genome is not None:
            from .cds import _extract_cds_sequence
            from .editing_gains import edited_protein

            coding_sequence = _extract_cds_sequence(ann, genome)
            if coding_sequence and len(coding_sequence) % 3 == 0:
                edits = getattr(ann, "rna_edits", [])
                cds_feature.qualifiers["translation"] = [
                    edited_protein(coding_sequence, edits, table=ann.transl_table)
                ]
        if ann.is_pseudo:
            cds_feature.qualifiers["pseudo"] = [""]
        if ann.is_partial_5prime or ann.is_partial_3prime:
            cds_feature.qualifiers["partial"] = [""]
        features.append(cds_feature)

    elif ann.gene_type == "tRNA":
        from .trna import trna_product

        trna = ann  # type: ignore
        trna_feature = SeqFeature(
            FeatureLocation(ann.genomic_start - 1, ann.genomic_end, strand=strand_biopython),
            type="tRNA",
            qualifiers={
                "gene": [ann.gene_name],
                "product": [
                    getattr(trna, "product", "") or trna_product(getattr(trna, "amino_acid", ""))
                ],
            },
        )
        features.append(trna_feature)

    elif ann.gene_type == "rRNA":
        rrna = ann  # type: ignore
        rrna_feature = SeqFeature(
            FeatureLocation(ann.genomic_start - 1, ann.genomic_end, strand=strand_biopython),
            type="rRNA",
            qualifiers={
                "gene": [ann.gene_name],
                "product": [ann.product or f"{getattr(rrna, 'rrna_type', '')} ribosomal RNA"],
            },
        )
        features.append(rrna_feature)

    return features
