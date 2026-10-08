"""Protein-guided solver models for mitochondrial genes the primary path missed.

Moss cox1 (five exons), nad9 and atp9 were lost because the primary path
encodes angiosperm exon structures. The models are built by the reading-frame
DP from a reference protein and kept only when they look like the gene:
homologous over most of their length, with no protein-coding "intron".
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

from Bio.Seq import Seq

from organelleverse.annotation.mitochondrion import solver_models as sm
from organelleverse.annotation.mitochondrion.models.gene import ExonRecord, GeneAnnotation, Strand
from organelleverse.annotation.mitochondrion.models.genome import GenomeSequence
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


def _random_dna(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _gene(seed: int = 5):
    """A 2-exon gene (500 bp intron) in random DNA, and its protein."""
    rng = random.Random(seed)
    protein = "M" + "".join(rng.choice(AMINO_ACIDS) for _ in range(119))
    cds = "".join(CODON[a] for a in protein) + "TAA"
    cut = 181  # phase-1 intron
    intron = "GTGCG" + _random_dna(rng, 490) + "AC"
    left, right = _random_dna(rng, 2000), _random_dna(rng, 2000)
    seq = left + cds[:cut] + intron + cds[cut:] + right
    exon1 = (2001, 2000 + cut)
    exon2 = (exon1[1] + len(intron) + 1, 2000 + len(cds) + len(intron))
    return seq, protein, exon1, exon2


def _run(monkeypatch, tmp_path: Path, seq: str, protein: str, hits, existing=()):
    (tmp_path / "cox1.Protein.fasta").write_text(f">ref\n{protein}\n")
    monkeypatch.setattr(sm, "_search", lambda *a, **k: hits)
    db = SimpleNamespace(blast_ref_dir=tmp_path)
    genome = GenomeSequence(seqid="g", sequence=seq)
    return sm.model_missing_genes(genome, db, ["cox1"], list(existing))


def test_two_exon_gene_is_modelled_from_the_protein(monkeypatch, tmp_path: Path) -> None:
    seq, protein, exon1, exon2 = _gene()
    hits = [sm._Hit(0, exon1[0], exon1[1], 1, 200.0), sm._Hit(0, exon2[0], exon2[1], 1, 150.0)]

    models, displaced = _run(monkeypatch, tmp_path, seq, protein, hits)

    assert displaced == []
    assert len(models) == 1
    assert [(e.start, e.end) for e in models[0].exons] == [exon1, exon2]
    assert models[0].source_method == "solver_model"
    assert not models[0].is_partial_5prime  # ATG initiator


def test_remnant_of_another_gene_is_displaced_but_a_complete_gene_blocks(
    monkeypatch, tmp_path: Path
) -> None:
    seq, protein, exon1, exon2 = _gene()
    hits = [sm._Hit(0, exon1[0], exon1[1], 1, 200.0), sm._Hit(0, exon2[0], exon2[1], 1, 150.0)]
    (tmp_path / "nad6.Protein.fasta").write_text(">n\n" + "A" * 200 + "\n")

    def other(length: int) -> GeneAnnotation:
        start = exon2[0]
        return GeneAnnotation(
            gene_name="nad6",
            exons=[ExonRecord(start=start, end=start + length - 1, strand=Strand.PLUS)],
            strand=Strand.PLUS,
        )

    remnant = other(90)  # 30 aa against a 200 aa nad6 reference
    models, displaced = _run(monkeypatch, tmp_path, seq, protein, hits, [remnant])
    assert len(models) == 1 and displaced == [remnant]

    complete = other(330)
    (tmp_path / "nad6.Protein.fasta").write_text(">n\n" + "A" * 110 + "\n")
    models, _ = _run(monkeypatch, tmp_path, seq, protein, hits, [complete])
    assert models == []


def test_a_protein_coding_intron_is_rejected() -> None:
    rng = random.Random(1)
    protein = "".join(rng.choice(AMINO_ACIDS) for _ in range(200))
    coding_stretch = "".join(CODON[a] for a in protein[50:180])
    assert sm._intron_codes_protein(coding_stretch, protein)
    assert not sm._intron_codes_protein(_random_dna(rng, 600), protein)


def test_models_must_be_homologous_over_most_of_their_length() -> None:
    rng = random.Random(2)
    protein = "".join(rng.choice(AMINO_ACIDS) for _ in range(150))
    assert sm._homologous(protein[5:145], protein)
    stitched = protein[:40] + "".join(rng.choice(AMINO_ACIDS) for _ in range(110))
    assert not sm._homologous(stitched, protein)


def test_non_atg_initiator_is_marked_partial(monkeypatch, tmp_path: Path) -> None:
    seq, protein, exon1, exon2 = _gene()
    seq = seq[: exon1[0] - 1] + "GTG" + seq[exon1[0] + 2 :]  # moss nad9-style GTG start
    hits = [sm._Hit(0, exon1[0], exon1[1], 1, 200.0), sm._Hit(0, exon2[0], exon2[1], 1, 150.0)]

    models, _ = _run(monkeypatch, tmp_path, seq, protein, hits)

    assert len(models) == 1
    assert models[0].is_partial_5prime
    assert str(Seq(seq[exon1[0] - 1 : exon1[0] + 2])) == "GTG"
