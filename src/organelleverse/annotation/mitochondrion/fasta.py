"""FASTA input handling — load, validate, merge multi-contig."""

from __future__ import annotations

from pathlib import Path

from Bio import SeqIO

from .models.genome import ContigInfo, GenomeSequence


def load_fasta(path: Path | str) -> GenomeSequence:
    """Load a FASTA file into a GenomeSequence.

    Handles both single-sequence and multi-contig inputs:
    - Single sequence: use directly
    - Multi-contig: merge with N-stretches, track contig map
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {path}")

    records = list(SeqIO.parse(str(path), "fasta"))
    if not records:
        raise ValueError(f"No sequences found in {path}")

    if len(records) == 1:
        return GenomeSequence(
            seqid=records[0].id,
            sequence=str(records[0].seq).upper(),
        )

    return _merge_contigs(records)


def _merge_contigs(records) -> GenomeSequence:
    """Merge multiple contigs into one sequence with N-stretches."""
    GAP = "N" * 200
    parts = []
    contig_map = []
    pos = 1

    for i, rec in enumerate(records):
        seq = str(rec.seq).upper()
        slen = len(seq)
        contig_map.append(
            ContigInfo(
                original_id=rec.id,
                start=pos,
                end=pos + slen - 1,
                length=slen,
            )
        )
        parts.append(seq)
        pos += slen
        if i < len(records) - 1:
            parts.append(GAP)
            pos += len(GAP)

    return GenomeSequence(
        seqid=f"merged_{len(records)}_contigs",
        sequence="".join(parts),
        contig_map=contig_map,
        is_circular=False,
    )


def validate_fasta(genome: GenomeSequence) -> list[str]:
    """Validate FASTA for common issues. Returns list of warnings."""
    warnings = []
    seq = genome.sequence
    length = genome.length
    gc = genome.gc_content

    if length < 10_000:
        warnings.append(f"Sequence very short ({length:,} bp). Plant mito typically ≥200 kb.")
    elif length > 15_000_000:
        warnings.append(f"Sequence very long ({length:,} bp). Check for contamination.")

    valid_bases = set("ATGCN")
    bad = set(seq.upper()) - valid_bases
    if bad:
        warnings.append(f"Non-standard characters found: {bad}")

    if gc < 35 or gc > 55:
        warnings.append(f"Unusual GC content: {gc:.1f}% (plant mito typically 43-45%)")

    n_pct = seq.upper().count("N") / length * 100 if length > 0 else 0
    if n_pct > 5:
        warnings.append(f"High N content: {n_pct:.1f}% (>5%)")

    return warnings
