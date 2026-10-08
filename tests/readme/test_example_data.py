"""The example genome shipped in examples/data/ reads correctly in both formats."""

from __future__ import annotations

from pathlib import Path

import organelleverse as ov

DATA = Path(__file__).resolve().parents[2] / "examples" / "data"


def test_example_genome_reads_as_fasta_and_genbank() -> None:
    fasta = ov.read(
        DATA / "arabidopsis_mitochondrion.fasta",
        organelle="mitochondrion",
        species="Arabidopsis thaliana",
    )
    genbank = ov.read(
        DATA / "arabidopsis_mitochondrion.gb",
        organelle="mitochondrion",
        species="Arabidopsis thaliana",
    )

    # FASTA carries the sequence only; GenBank carries the annotation.
    assert fasta.sequence is not None and fasta.sequence.validated
    assert fasta.annotation is None
    assert genbank.annotation is not None
