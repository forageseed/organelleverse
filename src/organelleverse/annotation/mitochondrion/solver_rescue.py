"""Opt-in reference-guided rescue of missing trans-spliced mitochondrial CDS.

Ported from the integration worktree to the current position-specific solver.
Both orientations are aligned independently; exon order follows the protein,
and every minus-strand coordinate is converted back to the original genome.
This helper does not replace the primary annotation pipeline automatically.
"""

from __future__ import annotations

from pathlib import Path

from Bio.Seq import Seq

from ..solver import trans
from ..solver.dp import Exon
from .models.gene import ExonRecord, GeneAnnotation, Strand
from .solver_models import _homologous, _profile, _reference_proteins

RESCUE_GENES = frozenset({"nad1", "nad2", "nad5", "rps12", "nad7", "rpl2"})


def rescue_trans_spliced(
    genome_sequence,
    db_manager,
    missing_genes: list[str],
    *,
    min_score: float = 30.0,
) -> list[GeneAnnotation]:
    """Return complete, reference-validated models for requested missing genes.

    Uses one representative reference (the longest, as in the original helper).
    Incomplete profile coverage, overlapping genomic pieces, internal stops and
    failed protein validation return no model. Scores are raw alignment scores,
    not probabilities. This is an opt-in rescue, not a benchmark certification.
    """
    ref_dir = getattr(db_manager, "blast_ref_dir", None)
    if ref_dir is None or not Path(ref_dir).is_dir():
        return []
    sequence = genome_sequence.sequence.upper()
    reverse = str(Seq(sequence).reverse_complement())
    length = len(sequence)
    rescued = []
    seen: set[str] = set()
    for requested in missing_genes:
        gene = requested.lower()
        if gene not in RESCUE_GENES or gene in seen:
            continue
        seen.add(gene)
        proteins = _reference_proteins(Path(ref_dir), gene)
        if not proteins:
            continue
        protein = max(proteins, key=len)
        profile = _profile(gene, protein)
        pieces = trans.align_pieces(profile, sequence, "+", min_piece=15, min_score=min_score)
        for piece in trans.align_pieces(profile, reverse, "-", min_piece=15, min_score=min_score):
            piece.exons = [Exon(length - e.end + 1, length - e.start + 1) for e in piece.exons]
            pieces.append(piece)
        assembly = trans.assemble(profile, pieces)
        if assembly is None or not assembly.is_complete or assembly.score < min_score:
            continue
        records = []
        bases = []
        covered: set[int] = set()
        cumulative = 0
        valid = True
        # assembly.exons is genome-sorted; assembly.pieces retains protein order.
        for piece in assembly.pieces:
            strand = Strand.PLUS if piece.strand == "+" else Strand.MINUS
            for exon in piece.exons:
                coordinates = set(range(exon.start, exon.end + 1))
                if exon.start < 1 or exon.end > length or not coordinates or covered & coordinates:
                    valid = False
                    break
                covered.update(coordinates)
                segment = sequence[exon.start - 1 : exon.end]
                bases.append(str(Seq(segment).reverse_complement()) if strand == Strand.MINUS else segment)
                records.append(ExonRecord(start=exon.start, end=exon.end, strand=strand,
                                          number=len(records) + 1, phase=cumulative % 3))
                cumulative += len(segment)
            if not valid:
                break
        if not valid or not records or cumulative % 3:
            continue
        peptide = str(Seq("".join(bases)).translate()).rstrip("*")
        if (not peptide or "*" in peptide or not 0.9 * len(protein) <= len(peptide) <= 1.1 * len(protein)
                or not _homologous(peptide, protein)):
            continue
        rescued.append(GeneAnnotation(
            gene_name=gene, exons=records, strand=records[0].strand,
            source_method="solver_rescue", score=assembly.score,
            notes=["Complete profile coverage and reference-protein validation; opt-in trans-splice rescue"],
        ))
    return rescued
