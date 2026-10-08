"""GFF3 record model."""

from __future__ import annotations

from pydantic import BaseModel


class GFF3Record(BaseModel):
    """A single GFF3 line."""

    seqid: str
    source: str = "OrganelleVerse"
    type: str  # gene, CDS, exon, tRNA, rRNA, repeat_region, etc.
    start: int  # 1-based, inclusive
    end: int  # 1-based, inclusive
    score: float | None = None
    strand: str  # + | - | .
    phase: int | None = None  # Reading frame for CDS (0, 1, 2)
    attributes: dict[str, str] = {}

    def to_line(self) -> str:
        """Format as GFF3 tab-separated line."""
        score_str = "." if self.score is None else f"{self.score:.2f}"
        phase_str = "." if self.phase is None else str(self.phase)
        attrs_str = ";".join(f"{k}={v}" for k, v in self.attributes.items())
        return f"{self.seqid}\t{self.source}\t{self.type}\t{self.start}\t{self.end}\t{score_str}\t{self.strand}\t{phase_str}\t{attrs_str}"

    @classmethod
    def from_gene(cls, gene, seqid: str) -> GFF3Record:
        """Create GFF3 records from a GeneAnnotation."""
        records = []
        gene_id = gene.gene_name
        parent_id = f"{gene.gene_name}-RA"

        # Gene feature
        records.append(
            cls(
                seqid=seqid,
                type="gene",
                start=gene.genomic_start,
                end=gene.genomic_end,
                strand=gene.strand.symbol,
                attributes={"ID": gene_id, "Name": gene.gene_name},
            )
        )

        # mRNA feature
        records.append(
            cls(
                seqid=seqid,
                type="mRNA",
                start=gene.genomic_start,
                end=gene.genomic_end,
                strand=gene.strand.symbol,
                attributes={"ID": parent_id, "Parent": gene_id, "Name": gene.gene_name},
            )
        )

        # Exon + CDS features
        for i, exon in enumerate(gene.exons, 1):
            records.append(
                cls(
                    seqid=seqid,
                    type="exon",
                    start=exon.start,
                    end=exon.end,
                    strand=gene.strand.symbol,
                    attributes={"ID": f"{parent_id}:exon:{i}", "Parent": parent_id},
                )
            )
            if gene.gene_type == "CDS":
                records.append(
                    cls(
                        seqid=seqid,
                        type="CDS",
                        start=exon.start,
                        end=exon.end,
                        strand=gene.strand.symbol,
                        phase=0,  # Simplified; real phase depends on reading frame
                        attributes={"ID": f"{parent_id}:cds", "Parent": parent_id},
                    )
                )

        return records
