from pathlib import Path

import organelleverse as ov
import organelleverse.core as core
from organelleverse.core.data import OrganelleData
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import OrganelleResult


def test_root_and_core_export_the_canonical_v1_contracts() -> None:
    assert ov.OrganelleData is OrganelleData
    assert ov.OrganelleGenome is OrganelleGenome
    assert ov.OrganelleMetadata is OrganelleMetadata
    assert ov.OrganelleResult is OrganelleResult
    assert ov.ResultProvenance is ResultProvenance

    assert core.OrganelleData is OrganelleData
    assert core.OrganelleGenome is OrganelleGenome
    assert core.OrganelleMetadata is OrganelleMetadata
    assert core.OrganelleResult is OrganelleResult
    assert core.ResultProvenance is ResultProvenance


def test_annotation_readers_return_canonical_v1_genomes(tmp_path: Path) -> None:
    fasta = tmp_path / "mitochondrion.fasta"
    fasta.write_text(">mitochondrion\nACGT\n", encoding="utf-8")
    genbank = tmp_path / "plastid.gb"
    genbank.write_text(
        """\
LOCUS       PLASTID                    4 bp    DNA     circular PLN 01-JAN-2025
FEATURES             Location/Qualifiers
ORIGIN
        1 acgt
//
""",
        encoding="utf-8",
    )

    sequence_genome = ov.annotation.read_fasta(
        fasta,
        organelle="mito",
        species="Arabidopsis thaliana",
    )
    annotation_genome = ov.annotation.read_genbank(genbank, organelle="chloro")

    assert type(sequence_genome) is OrganelleGenome
    assert sequence_genome.organelle == "mitochondrion"
    assert sequence_genome.sequence is not None
    assert sequence_genome.sequence.resolve() == fasta
    assert sequence_genome.metadata.species == "Arabidopsis thaliana"

    assert type(annotation_genome) is OrganelleGenome
    assert annotation_genome.organelle == "plastid"
    assert annotation_genome.annotation is not None
    assert annotation_genome.annotation.resolve() == genbank


def test_unified_reader_uses_the_same_canonical_genome_reader(tmp_path: Path) -> None:
    fasta = tmp_path / "mitochondrion.fasta"
    fasta.write_text(">mitochondrion\nACGT\n", encoding="utf-8")

    genome = ov.io.read_fasta(
        fasta,
        organelle="mitochondrion",
        species="Arabidopsis thaliana",
    )

    assert type(genome) is OrganelleGenome
    assert genome.organelle == "mitochondrion"
    assert genome.metadata.species == "Arabidopsis thaliana"
