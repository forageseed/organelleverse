"""Gene annotation data models."""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, computed_field


class Strand(IntEnum):
    """Gene strand orientation."""

    PLUS = 1
    MINUS = -1

    @property
    def symbol(self) -> str:
        return "+" if self is Strand.PLUS else "-"

    @classmethod
    def from_symbol(cls, s: str) -> Strand:
        return cls.PLUS if s == "+" else cls.MINUS


class ExonRecord(BaseModel):
    """A single exon."""

    start: int  # 1-based genome coordinate
    end: int  # inclusive
    strand: Strand
    number: int = 1  # 1-based exon number (auto-assigned if single)
    phase: int = 0  # Reading-frame phase at 5' end (0, 1, or 2)

    @computed_field
    @property
    def length(self) -> int:
        return self.end - self.start + 1


class GeneAnnotation(BaseModel):
    """A single annotated gene (protein-coding)."""

    gene_name: str  # Standard name: atp1, cox1, etc.
    gene_type: str = "CDS"  # CDS | tRNA | rRNA
    product: str = ""  # Product description
    exons: list[ExonRecord]
    strand: Strand
    notes: list[str] = []
    exceptions: list[str] = []  # e.g. "RNA editing"
    #: Codon indices (0-based, in the spliced CDS) whose C is edited to U:
    #: ACG start gains and CAA/CGA/CAG stop-gain or internal editing sites.
    #: Populated by editing_gains.annotate_rna_edits and consumed by the
    #: writers to emit the edited /translation= alongside the DNA coordinates.
    rna_edits: list[int] = []
    transl_table: int = 1  # NCBI Table 1 (standard genetic code)
    is_pseudo: bool = False
    is_partial_5prime: bool = False
    is_partial_3prime: bool = False
    source_method: str = ""  # "HMM" | "BLAST" | "tRNAscan-SE"
    confidence: float = 0.0
    score: float = 0.0  # Raw alignment score
    evalue: float = 1.0

    @computed_field
    @property
    def genomic_start(self) -> int:
        return min(e.start for e in self.exons)

    @computed_field
    @property
    def genomic_end(self) -> int:
        return max(e.end for e in self.exons)

    @computed_field
    @property
    def is_multiexon(self) -> bool:
        return len(self.exons) > 1

    @computed_field
    @property
    def total_exon_length(self) -> int:
        return sum(e.end - e.start + 1 for e in self.exons)
