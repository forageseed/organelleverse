"""Plastome annotation data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord


@dataclass
class PlastomeAnnotationStats:
    sample_name: str
    input_fasta: Path
    total_genes_reference: int = 0
    total_genes_annotated: int = 0
    annotation_success: bool = False
    warnings: list[str] = field(default_factory=list)
    unannotated_genes: list[str] = field(default_factory=list)
    output_genbank: Path | None = None
    ssc_orientation: str = ""


@dataclass(frozen=True)
class ReferenceFeature:
    feature_id: str
    feature: SeqFeature
    sequence: str
    gene: str
    feature_type: str
    reference_name: str
    exon_index: int = 1
    exon_count: int = 1


@dataclass(frozen=True)
class ReferenceRecord:
    path: Path
    record: SeqRecord
    features: tuple[ReferenceFeature, ...]
    kmers: frozenset[str] | None = None


@dataclass(frozen=True)
class ReferenceQuery:
    query_id: str
    group: str
    sequence: str
    reference_feature: ReferenceFeature


@dataclass(frozen=True)
class BlastHit:
    query_id: str
    pident: float
    qcov: float
    start: int
    end: int
    strand: int
    bitscore: float
    evalue: float
    align_length: int
    # Protein (tblastn) query-alignment extent: which reference residues aligned
    # (qstart/qend, 1-based) and the reference protein length (qlen, in residues).
    # Zero when unknown (nucleotide blastn hits). Used to extend the aligned
    # subject span to the true gene termini before codon refinement.
    qstart: int = 0
    qend: int = 0
    qlen: int = 0


@dataclass(frozen=True)
class BlastTools:
    blastn: str | None
    makeblastdb: str | None
    tblastn: str | None
    # LOSAT (Rust BLAST-compatible aligner). When present it is preferred for
    # plastome search groups: `losat tblastn` serves the protein groups
    # (reference3/reference4) and `losat blastn` the nucleotide groups and IR
    # self-search, in -subject mode (no makeblastdb step). NCBI BLAST+ stays
    # the fallback when the binary is unavailable.
    losat: str | None = None
