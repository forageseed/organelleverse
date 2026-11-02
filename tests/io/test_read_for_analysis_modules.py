from __future__ import annotations

from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.sequences import Sequences

ALIGNED = (
    ("s1", "ACGTACGTACGTACGTACGT"),
    ("s2", "ACGTACGTACGAACGTACGT"),
    ("s3", "ACGTTCGTACGAACGTACGA"),
    ("s4", "ACGTTCGTACGTACGTACGA"),
)
CDS = (
    ("c1", "ATGGCTAAAGGTCCA"),
    ("c2", "ATGGCCAAAGGACCA"),
    ("c3", "ATGGCTAAGGGTCCT"),
)


def _fasta(path: Path, records: tuple[tuple[str, str], ...]) -> Path:
    path.write_text("".join(f">{n}\n{s}\n" for n, s in records), encoding="utf-8")
    return path


@pytest.mark.parametrize("module", ["phylogeny", "diversity", "variation", "barcode", "selection"])
def test_every_analysis_module_reads_the_same_sequences(tmp_path: Path, module: str) -> None:
    fasta = _fasta(tmp_path / "a.fasta", ALIGNED)

    module_read = getattr(ov, module).read
    via_module = module_read(fasta)

    assert type(via_module) is Sequences
    assert via_module == ov.read(fasta, module=module)
    assert via_module.is_aligned


def test_diversity_gives_the_same_answer_for_read_sequences(tmp_path: Path) -> None:
    fasta = _fasta(tmp_path / "a.fasta", ALIGNED)
    sequences = ov.diversity.read(fasta)

    for function in (ov.diversity.nucleotide_diversity, ov.diversity.neutral_tests):
        from_path = function(fasta)
        from_read = function(sequences)
        assert from_read.status == from_path.status == "ok"
        assert from_read.metrics == from_path.metrics


def test_variation_gives_the_same_answer_for_read_sequences(tmp_path: Path) -> None:
    fasta = _fasta(tmp_path / "a.fasta", ALIGNED)

    from_path = ov.variation.snp(fasta)
    from_read = ov.variation.snp(ov.variation.read(fasta))

    assert from_read.status == from_path.status == "ok"
    assert from_read.metrics == from_path.metrics


def test_selection_takes_read_cds(tmp_path: Path) -> None:
    fasta = _fasta(tmp_path / "cds.fasta", CDS)

    from_path = ov.selection.validate_cds(fasta)
    from_read = ov.selection.validate_cds(ov.selection.read(fasta))
    kaks_path = ov.selection.kaks(fasta)
    kaks_read = ov.selection.kaks(ov.selection.read(fasta))

    assert from_read.status == from_path.status == "ok"
    assert from_read.metrics == from_path.metrics
    assert kaks_read.status == kaks_path.status
    assert kaks_read.metrics == kaks_path.metrics


def test_barcode_takes_read_alignment(tmp_path: Path) -> None:
    fasta = _fasta(tmp_path / "a.fasta", ALIGNED)

    from_path = ov.barcode.design_barcode(fasta)
    from_read = ov.barcode.design_barcode(ov.barcode.read(fasta))

    assert from_read.status == from_path.status
    assert from_read.metrics == from_path.metrics
