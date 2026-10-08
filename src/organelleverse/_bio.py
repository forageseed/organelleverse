"""Shared Biopython-backed functions — single import point for all suites.

This module wraps Bio.Seq, Bio.SeqIO, Bio.Phylo etc. so every suite imports
from here instead of reimplementing parsing/translation/IO.

Replaces the old _read_fasta, _translate, _CODONS, parse_newick etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# =========================================================================
# FASTA I/O (Bio.SeqIO)
# =========================================================================


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Read FASTA via Bio.SeqIO. Returns [(name, seq_str), ...].

    The path must exist, contain at least one record, and every record identifier
    must be unique.
    """
    from Bio import SeqIO  # type: ignore

    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    records = [(record.id, str(record.seq)) for record in SeqIO.parse(str(candidate), "fasta")]
    if not records:
        raise ValueError(f"FASTA contains no records: {candidate}")
    identifiers = tuple(identifier for identifier, _ in records)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"FASTA identifiers must be unique: {candidate}")
    return records


def write_fasta(path: str | Path, sequences: list[tuple[str, str]]) -> Path:
    """Write FASTA via Bio.SeqIO."""
    from Bio import SeqIO  # type: ignore
    from Bio.Seq import Seq  # type: ignore
    from Bio.SeqRecord import SeqRecord  # type: ignore

    records = [SeqRecord(Seq(seq), id=name, description="") for name, seq in sequences]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, str(p), "fasta")
    return p


# =========================================================================
# GenBank I/O (Bio.SeqIO)
# =========================================================================


def read_genbank(path: str | Path) -> list[tuple[str, str, list[dict]]]:
    """Read GenBank via Bio.SeqIO. Returns [(seqid, seq, [feature_dict, ...])]."""
    from Bio import SeqIO  # type: ignore

    out: list[tuple[str, str, list[dict]]] = []
    for rec in SeqIO.parse(str(path), "genbank"):
        feats: list[dict] = []
        for f in rec.features:
            feats.append(
                {
                    "type": f.type,
                    "start": int(f.location.start),
                    "end": int(f.location.end),
                    "strand": 1 if f.location.strand in (1, None) else -1,
                    "qualifiers": {k: v for k, v in f.qualifiers.items()},
                }
            )
        out.append((rec.id, str(rec.seq), feats))
    return out


def write_genbank(
    path: str | Path,
    records: list[tuple[str, str, list[dict]]],
    *,
    molecule_type: str = "DNA",
    topology: str = "circular",
) -> Path:
    """Write GenBank via Bio.SeqIO."""
    from Bio import SeqIO  # type: ignore
    from Bio.Seq import Seq  # type: ignore
    from Bio.SeqFeature import FeatureLocation, SeqFeature  # type: ignore
    from Bio.SeqRecord import SeqRecord  # type: ignore

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    bio_records = []
    for seqid, seq, feats in records:
        rec = SeqRecord(Seq(seq), id=seqid, name=seqid[:16], description="OrganelleVerse")
        rec.annotations["molecule_type"] = molecule_type
        rec.annotations["topology"] = topology
        for f in feats:
            loc = FeatureLocation(
                int(f.get("start", 0)),
                int(f.get("end", len(seq))),
                strand=f.get("strand", 1),
            )
            quals = {}
            for k, v in f.get("qualifiers", {}).items():
                quals[k] = v if isinstance(v, list) else [str(v)]
            rec.features.append(
                SeqFeature(loc, type=f.get("type", "misc_feature"), qualifiers=quals)
            )
        bio_records.append(rec)
    SeqIO.write(bio_records, str(p), "genbank")
    return p


# =========================================================================
# Translation (Bio.Seq)
# =========================================================================


def translate(dna: str, *, to_stop: bool = True, table: int = 1) -> str:
    """Translate DNA to protein via Bio.Seq."""
    from Bio.Seq import Seq  # type: ignore

    return str(Seq(dna).translate(to_stop=to_stop, table=table))


def reverse_complement(dna: str) -> str:
    """Reverse complement via Bio.Seq."""
    from Bio.Seq import Seq  # type: ignore

    return str(Seq(dna).reverse_complement())


# =========================================================================
# Newick tree I/O (Bio.Phylo)
# =========================================================================


def read_newick(path: str | Path) -> Any:
    """Read a Newick tree via Bio.Phylo. Returns a Bio.Phylo BaseTree.Tree."""
    from Bio import Phylo  # type: ignore

    return Phylo.read(str(path), "newick")


def write_newick(path: str | Path, tree_or_str: Any) -> Path:
    """Write a Newick tree via Bio.Phylo."""
    from Bio import Phylo  # type: ignore

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(tree_or_str, str):
        p.write_text(tree_or_str.rstrip(";") + ";\n")
    else:
        Phylo.write(tree_or_str, str(p), "newick")
    return p


# =========================================================================
# Codon table (Bio.Seq) — replaces hardcoded _CODONS
# =========================================================================


def get_codon_table(table: int = 1) -> dict[str, str]:
    """Get codon→amino acid mapping from Biopython's NCBI table."""
    from itertools import product

    from Bio.Data import CodonTable  # type: ignore

    bio_table = CodonTable.unambiguous_dna_by_id[int(table)]
    table_dict: dict[str, str] = {}
    for a, b, c in product("TCAG", repeat=3):
        codon = a + b + c
        if codon in bio_table.stop_codons:
            table_dict[codon] = "*"
        else:
            table_dict[codon] = bio_table.forward_table[codon]
    return table_dict


def get_codon_table_info(table: int = 1) -> dict[str, Any]:
    """Return NCBI codon-table metadata from Biopython."""
    from Bio.Data import CodonTable  # type: ignore

    bio_table = CodonTable.unambiguous_dna_by_id[int(table)]
    return {
        "id": int(table),
        "names": tuple(name for name in bio_table.names if name),
        "start_codons": tuple(bio_table.start_codons),
        "stop_codons": tuple(bio_table.stop_codons),
    }


def is_synonymous(codon1: str, codon2: str, table: int = 1) -> bool:
    """Check if two codons encode the same amino acid (via Bio.Seq)."""
    from Bio.Seq import Seq  # type: ignore

    aa1 = str(Seq(codon1).translate(table=table))
    aa2 = str(Seq(codon2).translate(table=table))
    return aa1 == aa2


# =========================================================================
# SSR / tandem repeat detection (pytrf)
# =========================================================================


def find_ssrs(seq: str, *, min_unit: int = 1, max_unit: int = 6, min_copy: int = 3) -> list[dict]:
    """Detect SSR/tandem repeats via pytrf (C extension, fast).

    Returns list of {"motif", "start", "end", "length", "copies"}.
    """
    import pytrf

    results: list[dict] = []
    for record in pytrf.STRFinder("seq", seq.upper()):
        # pytrf returns: name, start, end, motif, copies, length (varies by type)
        # STRFinder yields STRfinder objects with attributes
        results.append(
            {
                "motif": getattr(record, "motif", getattr(record, "motif_sequence", "")),
                "start": getattr(record, "start", 0),
                "end": getattr(record, "end", 0),
                "length": getattr(record, "length", getattr(record, "match_length", 0)),
                "copies": getattr(record, "copies", getattr(record, "repeat_number", 0)),
            }
        )
    return results
