from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

import organelleverse as ov
from organelleverse.assembly import LongReadLibrary
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.genome import OrganelleGenome
from organelleverse.io_genome import read_fasta_genome, read_genbank_genome


def test_root_read_is_the_canonical_io_convenience_reader() -> None:
    assert ov.read is ov.io.read
    assert ov.io.read_fasta is read_fasta_genome
    assert ov.io.read_genbank is read_genbank_genome
    assert "read" in ov.__all__
    assert "read" in dir(ov)


def test_read_detects_and_validates_fasta_content(tmp_path: Path) -> None:
    fasta = tmp_path / "sequence_without_suffix"
    fasta.write_text(">mitochondrion\nACGT\n", encoding="utf-8")

    genome = ov.read(
        fasta,
        organelle="mitochondrion",
        species="Arabidopsis thaliana",
    )

    assert type(genome) is OrganelleGenome
    assert genome.sequence is not None
    assert genome.sequence.format == "fasta"
    assert genome.sequence.validated is True
    assert genome.annotation is None


def test_read_detects_and_validates_genbank_content(tmp_path: Path) -> None:
    genbank = tmp_path / "annotation_without_suffix"
    genbank.write_text(
        """\
LOCUS       MITOCHONDRION              4 bp    DNA     circular PLN 01-JAN-2025
FEATURES             Location/Qualifiers
ORIGIN
        1 acgt
//
""",
        encoding="utf-8",
    )

    genome = ov.read(
        genbank,
        organelle="mitochondrion",
        species="Arabidopsis thaliana",
    )

    assert type(genome) is OrganelleGenome
    assert genome.annotation is not None
    assert genome.annotation.format == "genbank"
    assert genome.annotation.validated is True
    assert genome.sequence is None


def test_read_accepts_the_complete_structured_reads_form(tmp_path: Path) -> None:
    reads = tmp_path / "sample.fastq"
    reads.write_text("@read-1\nACGT\n+\nIIII\n", encoding="utf-8")
    library = LongReadLibrary(
        technology="pacbio_hifi",
        quality_state="ccs",
        reads=reads,
    )

    actual = ov.read(long_libraries=(library,))
    canonical = ov.io.read_reads(long_libraries=(library,))

    assert actual == canonical
    assert actual.modality == "sequencing_reads"


def test_read_rejects_mixed_genome_and_reads_forms(tmp_path: Path) -> None:
    fasta = tmp_path / "mitochondrion.fasta"
    fasta.write_text(">mitochondrion\nACGT\n", encoding="utf-8")
    reads = tmp_path / "sample.fastq"
    reads.write_text("@read-1\nACGT\n+\nIIII\n", encoding="utf-8")

    with pytest.raises(OrganelleInputError) as captured:
        ov.read(
            fasta,
            organelle="mitochondrion",
            species="Arabidopsis thaliana",
            long_libraries=(
                LongReadLibrary(
                    technology="pacbio_hifi",
                    quality_state="ccs",
                    reads=reads,
                ),
            ),
        )

    assert captured.value.code == "input.ambiguous_read"


def test_read_rejects_content_with_no_supported_genome_format(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown"
    unknown.write_text("not a biological sequence file\n", encoding="utf-8")

    with pytest.raises(OrganelleInputError) as captured:
        ov.read(
            unknown,
            organelle="mitochondrion",
            species="Arabidopsis thaliana",
        )

    assert captured.value.code == "input.unknown_genome_format"


@pytest.mark.parametrize(
    ("parameters", "error_code"),
    [
        ({"organelle": "unknown"}, "input.invalid_organelle"),
        (
            {"organelle": "mitochondrion", "format": "bam"},
            "input.invalid_read_format",
        ),
    ],
)
def test_read_rejects_unknown_direct_call_literals(
    tmp_path: Path,
    parameters: dict[str, object],
    error_code: str,
) -> None:
    fasta = tmp_path / "mitochondrion.fasta"
    fasta.write_text(">mitochondrion\nACGT\n", encoding="utf-8")

    with pytest.raises(OrganelleInputError) as captured:
        ov.read(fasta, **cast(Any, parameters))

    assert captured.value.code == error_code
