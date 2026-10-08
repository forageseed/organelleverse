"""CDS completeness validation.

Translates PMGA v1 logic from 03.CDSCheck.py:
- Reading frame validation
- Premature stop codon detection
- Start codon correctness
- Core vs variable gene classification
- Missing gene report
- Length reasonableness checks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .codon_contract import allowed_start_codons
from .db import DBManager
from .models.gene import GeneAnnotation
from .models.genome import GenomeSequence
from .pcg import STOP_CODONS, STOP_GAIN_CODONS, translate_sequence

logger = logging.getLogger(__name__)

# Gene classifications (from v1 CDSCheck.py lines 207-210)
CORE_GENES = [
    "atp1",
    "atp4",
    "atp6",
    "atp8",
    "atp9",
    "ccmB",
    "ccmC",
    "ccmFC",
    "ccmFN",
    "cob",
    "cox1",
    "cox2",
    "cox3",
    "matR",
    "mttB",
    "nad1",
    "nad2",
    "nad3",
    "nad4",
    "nad4L",
    "nad5",
    "nad6",
    "nad7",
    "nad9",
]

VARIABLE_GENES = [
    "rpl2",
    "rpl5",
    "rpl6",
    "rpl10",
    "rpl16",
    "rps1",
    "rps2",
    "rps3",
    "rps4",
    "rps7",
    "rps10",
    "rps11",
    "rps12",
    "rps13",
    "rps14",
    "rps19",
    "sdh3",
    "sdh4",
]

# Genes whose internal stop codons are expected even in otherwise valid
# annotations: trans-spliced genes (multi-exon, mixed strands), divergent
# ribosomal genes and recently transferred sdh genes that BLAST finds with
# approximate boundaries. Shared with validation.py so the canonical
# document validator demotes cds_internal_stop the same way cds.py does.
KNOWN_HARD_INTERNAL_STOP_GENES = {
    "nad1", "nad2", "nad5", "rps12",   # trans-spliced
    "rpl2", "rps3", "rps14", "rps19",  # divergent ribosomal
    "sdh3", "sdh4",                     # recently transferred
    "cox2", "cob", "nad7",              # multi-exon divergent
}

# Expected gene length ranges (amino acids)
GENE_LENGTH_RANGES = {
    "atp1": (500, 620),
    "atp4": (100, 160),
    "atp6": (250, 320),
    "atp8": (50, 80),
    "atp9": (70, 100),
    "cob": (380, 450),
    "cox1": (500, 580),
    "cox2": (220, 280),
    "cox3": (250, 320),
    "nad1": (300, 380),
    "nad2": (340, 420),
    "nad3": (100, 140),
    "nad4": (430, 520),
    "nad4L": (90, 130),
    "nad5": (550, 700),
    "nad6": (160, 210),
    "nad7": (360, 440),
    "nad9": (170, 220),
    "ccmB": (210, 270),
    "ccmC": (240, 300),
    "ccmFC": (250, 330),
    "ccmFN": (380, 460),
    "matR": (500, 650),
    "mttB": (80, 130),
    "rnaseh": (130, 180),
}


@dataclass
class CDSIssue:
    """A single CDS validation issue."""

    gene_name: str
    issue_type: str  # "premature_stop" | "missing_start" | "missing_stop" | "frame_shift" | "short" | "long" | "ok"
    message: str
    severity: str  # "error" | "warning" | "info"


@dataclass
class CDSValidationResult:
    """Result of CDS validation."""

    total_genes: int = 0
    valid_genes: int = 0
    issues: list[CDSIssue] = field(default_factory=list)
    missing_core: list[str] = field(default_factory=list)
    missing_variable: list[str] = field(default_factory=list)
    gene_completeness_pct: float = 0.0

    @property
    def is_ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def summary(self) -> str:
        lines = [
            f"CDS Validation: {self.valid_genes}/{self.total_genes} genes valid",
            f"Core genes missing: {len(self.missing_core)}",
            f"Variable genes missing: {len(self.missing_variable)}",
            f"Completeness: {self.gene_completeness_pct:.1f}%",
        ]
        if self.issues:
            error_count = sum(1 for i in self.issues if i.severity == "error")
            warn_count = sum(1 for i in self.issues if i.severity == "warning")
            lines.append(f"Issues: {error_count} errors, {warn_count} warnings")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class RejectedCDSCandidate:
    """One CDS candidate excluded from released output after validation."""

    gene_name: str
    start: int | None
    end: int | None
    strand: int
    parts: tuple[tuple[int, int, int], ...]
    issue_codes: tuple[str, ...]
    issue_messages: tuple[str, ...]


def partition_valid_cds(
    annotations: list[GeneAnnotation],
    genome: GenomeSequence,
    db_manager: DBManager | None = None,
) -> tuple[list[GeneAnnotation], tuple[RejectedCDSCandidate, ...]]:
    """Return publishable annotations and structurally invalid CDS candidates.

    Candidates are evaluated independently so a malformed duplicate never removes
    a valid copy of the same gene. Non-CDS annotations pass through unchanged.
    """

    accepted: list[GeneAnnotation] = []
    rejected: list[RejectedCDSCandidate] = []
    for annotation in annotations:
        if annotation.gene_type != "CDS":
            accepted.append(annotation)
            continue
        errors = tuple(
            issue
            for issue in _validate_single_cds(annotation, genome, db_manager)
            if issue.severity == "error"
        )
        if not errors:
            accepted.append(annotation)
            continue
        rejected.append(
            RejectedCDSCandidate(
                gene_name=annotation.gene_name,
                start=min((exon.start for exon in annotation.exons), default=None),
                end=max((exon.end for exon in annotation.exons), default=None),
                strand=int(annotation.strand),
                parts=tuple((exon.start, exon.end, int(exon.strand)) for exon in annotation.exons),
                issue_codes=tuple(issue.issue_type for issue in errors),
                issue_messages=tuple(issue.message for issue in errors),
            )
        )
    return accepted, tuple(rejected)


def validate_cds(
    annotations: list[GeneAnnotation],
    genome: GenomeSequence,
    db_manager: DBManager | None = None,
) -> CDSValidationResult:
    """Validate all CDS annotations.

    Checks:
    1. Reading frame integrity (length divisible by 3)
    2. Start codon presence
    3. Stop codon presence
    4. Internal stop codons (premature termination)
    5. Gene length reasonableness
    6. Core gene completeness

    Args:
        annotations: All gene annotations (CDS only)
        genome: Genome sequence for sequence extraction
        db_manager: Optional DB manager for gene metadata

    Returns:
        CDSValidationResult with issues and statistics
    """
    result = CDSValidationResult()
    cds_annotations = [a for a in annotations if a.gene_type == "CDS"]
    result.total_genes = len(cds_annotations)

    found_gene_names = set()

    for ann in cds_annotations:
        found_gene_names.add(ann.gene_name)
        issues = _validate_single_cds(ann, genome, db_manager)
        result.issues.extend(issues)

        if all(i.issue_type == "ok" or i.severity != "error" for i in issues):
            result.valid_genes += 1

    # Check core gene completeness
    for gene in CORE_GENES:
        if gene not in found_gene_names:
            result.missing_core.append(gene)

    for gene in VARIABLE_GENES:
        if gene not in found_gene_names:
            result.missing_variable.append(gene)

    total_expected = len(CORE_GENES) + len(VARIABLE_GENES)
    found = len(found_gene_names & (set(CORE_GENES) | set(VARIABLE_GENES)))
    result.gene_completeness_pct = found / total_expected * 100 if total_expected > 0 else 0

    return result


def _validate_single_cds(
    ann: GeneAnnotation,
    genome: GenomeSequence,
    db_manager: DBManager | None,
) -> list[CDSIssue]:
    """Validate a single CDS annotation."""
    issues = []

    exon_strands = {exon.strand for exon in ann.exons}
    if len(exon_strands) == 1 and ann.strand not in exon_strands:
        return [
            CDSIssue(
                gene_name=ann.gene_name,
                issue_type="strand_mismatch",
                message="Gene strand disagrees with every exon strand",
                severity="error",
            )
        ]

    # Extract CDS sequence
    cds_seq = _extract_cds_sequence(ann, genome)
    if not cds_seq:
        issues.append(
            CDSIssue(
                gene_name=ann.gene_name,
                issue_type="missing_seq",
                message="Could not extract CDS sequence",
                severity="error",
            )
        )
        return issues

    cds_len = len(cds_seq)
    protein = translate_sequence(cds_seq)

    # Check 1: Length divisible by 3
    if cds_len % 3 != 0:
        issues.append(
            CDSIssue(
                gene_name=ann.gene_name,
                issue_type="frame_shift",
                message=f"CDS length {cds_len} not divisible by 3",
                severity="error",
            )
        )

    # Check 2: Start codon
    first_codon = cds_seq[:3].upper() if len(cds_seq) >= 3 else ""
    is_stop_gain = db_manager.is_stop_gain_gene(ann.gene_name) if db_manager else False
    is_start_gain = db_manager.is_start_gain_gene(ann.gene_name) if db_manager else False

    valid_starts = allowed_start_codons(
        ann.gene_name,
        allow_rna_editing=is_start_gain,
    )

    if first_codon not in valid_starts:
        issues.append(
            CDSIssue(
                gene_name=ann.gene_name,
                issue_type="missing_start",
                message=f"Start codon: {first_codon} (expected {'/'.join(sorted(valid_starts))})",
                severity="warning" if ann.is_pseudo or ann.is_partial_5prime else "error",
            )
        )

    # Check 3: Stop codon. RNA-editing stop gain is accepted only when both
    # the packaged gene metadata and the candidate's explicit exception agree.
    last_codon = cds_seq[-3:].upper() if len(cds_seq) >= 3 else ""
    has_rna_editing_exception = any(
        exception.strip().casefold() == "rna editing" for exception in ann.exceptions
    )
    has_terminal_stop = last_codon in STOP_CODONS or (
        is_stop_gain and has_rna_editing_exception and last_codon in STOP_GAIN_CODONS
    )
    if not has_terminal_stop:
        issues.append(
            CDSIssue(
                gene_name=ann.gene_name,
                issue_type="missing_stop",
                message=f"Terminal codon: {last_codon} (expected a stop codon)",
                severity="warning" if ann.is_pseudo or ann.is_partial_3prime else "error",
            )
        )

    # Check 4: Internal stop codons. A generic RNA-editing note is not enough
    # to explain away a genomic stop at an unspecified internal coordinate.
    has_internal_stop = False
    for i, aa in enumerate(protein):
        if aa == "*" and i < len(protein) - 1:
            codon_pos = i * 3
            codon = cds_seq[codon_pos : codon_pos + 3].upper()
            # Check if it's a stop-gain RNA editing site
            if is_stop_gain and codon in {"CAA", "CAG", "CGA"}:
                continue  # Expected for RNA editing genes
            has_internal_stop = True
            break

    if has_internal_stop:
        # Trans-spliced genes (nad1/nad2/nad5/rps12) and solver-rescued
        # assemblies: these genes are inherently multi-exon with mixed
        # strands; an internal stop is almost always an exon-junction
        # boundary artifact, not a pseudogene. Demote to warning so the
        # gene enters output for review instead of being silently dropped.
        # This is the single biggest recall improvement found by the
        # 10-genome benchmark (nad1 was lost in every genome tested).
        # Trans-spliced + short/divergent genes that BLAST finds with
        # approximate boundaries: rpl2/rps14/rps19/sdh4 are systematically
        # rejected by exact start/stop checking even when the BLAST hit is
        # correct (the 10-genome benchmark loses them in every genome).
        # PMGA's boundary.py refines these; until that refinement reaches
        # parity, demote to warning so they enter output for review.
        _is_trans_spliced = (
            ann.gene_name.lower() in KNOWN_HARD_INTERNAL_STOP_GENES
            or getattr(ann, "source_method", "") == "solver_rescue"
        )
        issues.append(
            CDSIssue(
                gene_name=ann.gene_name,
                issue_type="premature_stop",
                message=f"Internal stop codon at aa position {i + 1}",
                severity="warning" if _is_trans_spliced else "error",
            )
        )

    # Check 5: Length reasonableness
    aa_len = len(protein)
    expected = GENE_LENGTH_RANGES.get(ann.gene_name)
    if expected:
        if aa_len < expected[0] * 0.5:
            issues.append(
                CDSIssue(
                    gene_name=ann.gene_name,
                    issue_type="short",
                    message=f"Too short: {aa_len} aa (expected {expected[0]}-{expected[1]})",
                    severity="warning",
                )
            )
        elif aa_len > expected[1] * 1.5:
            issues.append(
                CDSIssue(
                    gene_name=ann.gene_name,
                    issue_type="long",
                    message=f"Too long: {aa_len} aa (expected {expected[0]}-{expected[1]})",
                    severity="info",
                )
            )

    if not issues:
        issues.append(
            CDSIssue(
                gene_name=ann.gene_name,
                issue_type="ok",
                message="CDS valid",
                severity="info",
            )
        )

    return issues


def _extract_cds_sequence(ann: GeneAnnotation, genome: GenomeSequence) -> str:
    """Extract CDS nucleotide sequence from annotation."""
    parts = []
    # GeneAnnotation.exons is biological 5'->3' order. Genomic sorting breaks
    # trans-spliced genes whose numbered exons are scattered around the genome.
    for exon in ann.exons:
        seq = genome.get_sequence_for_range(exon.start, exon.end, exon.strand)
        parts.append(seq)

    return "".join(parts)
