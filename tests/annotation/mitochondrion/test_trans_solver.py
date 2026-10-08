"""Trans-spliced assembly with the reading-frame DP.

Monocot nad5 was left as one fragment: blastn hit ends a base or two off the
exon ends put every combination out of frame, and a near-equal copy of exon 2
in a repeat survived as a separate "nad5.2" gene.
"""

from __future__ import annotations

import random
from pathlib import Path

from Bio.Seq import Seq

from organelleverse.annotation.mitochondrion import trans_splicing
from organelleverse.annotation.mitochondrion.models.gene import ExonRecord, GeneAnnotation, Strand
from organelleverse.annotation.mitochondrion.models.genome import GenomeSequence
from organelleverse.annotation.mitochondrion.trans_assembly import (
    assemble_exons_by_solver,
    length_is_off,
)
from organelleverse.annotation.solver.dp import AMINO_ACIDS

CODON = {
    "A": "GCT",
    "C": "TGT",
    "D": "GAT",
    "E": "GAA",
    "F": "TTT",
    "G": "GGT",
    "H": "CAT",
    "I": "ATT",
    "K": "AAA",
    "L": "CTT",
    "M": "ATG",
    "N": "AAT",
    "P": "CCT",
    "Q": "CAA",
    "R": "CGT",
    "S": "TCT",
    "T": "ACT",
    "V": "GTT",
    "W": "TGG",
    "Y": "TAT",
}


def _scattered_gene(seed: int = 11):
    """Exon 1 (+) at 3 kb, a 21 bp exon 2 (-) at 12 kb, exon 3 (-) at 7 kb."""
    rng = random.Random(seed)
    protein = "M" + "".join(rng.choice(AMINO_ACIDS) for _ in range(139))
    cds = "".join(CODON[a] for a in protein) + "TAA"
    e1, e2, e3 = cds[:151], cds[151:172], cds[172:]
    genome = [rng.choice("ACGT") for _ in range(16000)]

    def put(seq: str, at: int, strand: int) -> tuple[int, int, int]:
        text = seq if strand == 1 else str(Seq(seq).reverse_complement())
        genome[at - 1 : at - 1 + len(text)] = list(text)
        return at, at + len(text) - 1, strand

    exons = [put(e1, 3001, 1), put(e2, 12001, -1), put(e3, 7001, -1)]
    return "".join(genome), protein, exons


def test_solver_places_exact_splice_sites_from_imprecise_hits(tmp_path: Path) -> None:
    seq, protein, exons = _scattered_gene()
    (tmp_path / "nad5.Protein.fasta").write_text(f">ref\n{protein}\n")
    # hits a base or two off at each end, as blastn reports them; the micro-exon
    # reference carries a flank, so its hit runs 60 bp past the exon
    hits = {
        1: [(exons[0][0] + 1, exons[0][1] - 2, Strand.PLUS, 99.0, 148, 151)],
        2: [(exons[1][0], exons[1][1] + 60, Strand.MINUS, 100.0, 81, 21)],
        3: [(exons[2][0] + 2, exons[2][1] - 1, Strand.MINUS, 98.0, 272, 275)],
    }
    genome = GenomeSequence(seqid="g", sequence=seq)

    ann = assemble_exons_by_solver("nad5", genome, hits, 3, tmp_path)

    assert ann is not None
    assert [(e.start, e.end, int(e.strand)) for e in ann.exons] == exons


def test_length_gate_keeps_plausible_annotations(tmp_path: Path) -> None:
    (tmp_path / "nad1.Protein.fasta").write_text(">r\n" + "A" * 325 + "\n")

    def ann(length: int) -> GeneAnnotation:
        return GeneAnnotation(
            gene_name="nad1",
            exons=[ExonRecord(start=1, end=length, strand=Strand.PLUS)],
            strand=Strand.PLUS,
        )

    assert length_is_off(None, tmp_path)
    assert length_is_off(ann(600), tmp_path)  # clearly incomplete
    assert not length_is_off(ann(1026), tmp_path)  # moss nad1: 9-14% long, plausible
    assert length_is_off(ann(1400), tmp_path)  # clearly overgrown


def test_assembled_gene_absorbs_its_duplicate_pieces(monkeypatch, tmp_path: Path) -> None:
    def piece(start: int, end: int) -> GeneAnnotation:
        return GeneAnnotation(
            gene_name="nad5",
            exons=[ExonRecord(start=start, end=end, strand=Strand.PLUS)],
            strand=Strand.PLUS,
        )

    assembled = GeneAnnotation(
        gene_name="nad5",
        exons=[
            ExonRecord(start=100, end=330, strand=Strand.PLUS),
            ExonRecord(start=1200, end=2400, strand=Strand.PLUS),
        ],
        strand=Strand.PLUS,
    )
    existing = {
        "nad5": piece(50000, 51200),  # the repeat copy the HMM path chose
        "nad5.2": piece(1220, 2400),  # a piece of the assembled gene
        "nad5.3": piece(90000, 91000),  # elsewhere: left alone
    }
    ref = tmp_path / "nad5.exons.fasta"
    ref.write_text(">x\nA\n")
    monkeypatch.setattr(
        trans_splicing, "find_exon_reference_file", lambda g, db: ref if g == "nad5" else None
    )
    monkeypatch.setattr(
        trans_splicing,
        "search_exons_blastn",
        lambda *a, **k: {1: [(100, 330, Strand.PLUS, 99.0, 231, 231)]},
    )
    from organelleverse.annotation.mitochondrion import trans_assembly

    monkeypatch.setattr(trans_assembly, "assemble_exons_by_protein", lambda *a, **k: assembled)
    db = type("DB", (), {"blast_ref_dir": tmp_path})()

    result = trans_splicing.annotate_trans_spliced_genes(
        GenomeSequence(seqid="g", sequence="A" * 100000),
        db,
        existing,
        tool_paths={"losat": "LOSAT"},
    )

    assert result["nad5"] is assembled
    assert "nad5.2" not in result
    assert "nad5.3" in result
