from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.genome import OrganelleGenome
from organelleverse.phylogeny import Sequences
from organelleverse.read_facade import readable_modules


def _fasta(path: Path, *records: tuple[str, str]) -> Path:
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in records), encoding="utf-8")
    return path


def test_every_readable_module_publishes_read() -> None:
    assert {"annotation", "assembly", "phylogeny", "population", "morphology"} <= set(
        readable_modules()
    )
    for name in readable_modules():
        assert callable(importlib.import_module(f"organelleverse.{name}").read)


def test_read_with_a_module_is_that_modules_read(tmp_path: Path) -> None:
    genes = _fasta(tmp_path / "genes.fasta", ("g1", "ACGT"), ("g2", "ACGA"))

    via_top = ov.read(genes, module="phylogeny")
    via_module = ov.phylogeny.read(genes)

    assert type(via_top) is Sequences
    assert via_top == via_module
    assert via_top.names == ("g1", "g2")
    assert ov.read(genes, module="phylo") == via_module


def test_a_sequence_file_says_nothing_about_what_it_is(tmp_path: Path) -> None:
    unaligned = _fasta(tmp_path / "a.fasta", ("g1", "ACGT"), ("g2", "AC"))
    aligned = _fasta(tmp_path / "b.fasta", ("g1", "ACGT"), ("g2", "AC--"))

    assert ov.phylogeny.read(unaligned).is_aligned is False
    assert ov.phylogeny.read(aligned).is_aligned is True


def test_genome_and_reads_options_route_to_their_module(tmp_path: Path) -> None:
    fasta = _fasta(tmp_path / "mt.fasta", ("mt", "ACGT"))
    reads = tmp_path / "r.fastq"
    reads.write_text("@r1\nACGT\n+\nIIII\n", encoding="utf-8")

    assert type(ov.read(fasta, organelle="mito", species="Oryza sativa")) is OrganelleGenome
    assert type(ov.read(reads, technology="illumina")) is OrganelleData
    genome = ov.annotation.read(fasta, organelle="mito", species="Oryza sativa")
    assert type(genome) is OrganelleGenome
    assert type(ov.assembly.read(reads, technology="illumina")) is OrganelleData


def test_a_plain_sequence_file_asks_for_a_module(tmp_path: Path) -> None:
    genes = _fasta(tmp_path / "genes.fasta", ("g1", "ACGT"))

    with pytest.raises(OrganelleInputError) as caught:
        ov.read(genes)

    assert caught.value.code == "input.module_required"
    assert caught.value.details["readable_modules"] == readable_modules()


def test_an_unknown_module_lists_the_readable_ones(tmp_path: Path) -> None:
    genes = _fasta(tmp_path / "genes.fasta", ("g1", "ACGT"))

    with pytest.raises(OrganelleInputError) as caught:
        ov.read(genes, module="qc")

    assert caught.value.code == "input.unknown_read_module"


def test_genome_options_without_a_path_are_refused() -> None:
    with pytest.raises(OrganelleInputError) as caught:
        ov.read(organelle="mito")

    assert caught.value.code == "input.sequence_parameters_without_path"


def test_phylogeny_read_reports_a_missing_or_empty_file(tmp_path: Path) -> None:
    with pytest.raises(OrganelleInputError) as missing:
        ov.phylogeny.read(tmp_path / "absent.fasta")
    assert missing.value.code == "input.missing_artifact"

    empty = tmp_path / "empty.fasta"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(OrganelleInputError) as no_records:
        ov.phylogeny.read(empty)
    assert no_records.value.code == "input.unknown_file_format"


def test_the_io_module_is_deprecated_but_still_exports_read() -> None:
    import organelleverse.io as legacy

    with pytest.warns(DeprecationWarning, match="organelleverse.io is deprecated"):
        importlib.reload(legacy)

    assert legacy.read is ov.read
    assert legacy.read_reads is ov.io.read_reads
