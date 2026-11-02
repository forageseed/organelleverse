"""A genome that has only a FASTA is analysed from its sequence (it used to come back as 0 bp, no IR)."""

from __future__ import annotations

import random
from pathlib import Path

from organelleverse.io_genome import read_fasta_genome
from organelleverse.ir_boundary.ir_boundary import ir_boundary

_COMP = str.maketrans("ACGT", "TGCA")


def _plastome(lsc: int = 6000, ir: int = 1200, ssc: int = 2000) -> tuple[str, int]:
    rng = random.Random(20261013)
    draw = lambda n: "".join(rng.choice("ACGT") for _ in range(n))  # noqa: E731
    arm = draw(ir)
    return draw(lsc) + arm + draw(ssc) + arm.translate(_COMP)[::-1], ir


def test_fasta_only_genome_gets_ir_lsc_ssc_from_its_sequence(tmp_path: Path) -> None:
    sequence, ir = _plastome()
    fasta = tmp_path / "plastome.fasta"
    fasta.write_text(">plastome\n" + "\n".join(sequence[i : i + 70] for i in range(0, len(sequence), 70)) + "\n")
    genome = read_fasta_genome(fasta, organelle="plastid", species="Synthetic plastid")

    result = ir_boundary([genome])

    assert result.status in ("ok", "succeeded"), result.summary_text
    per_genome = result.metrics["per_genome"][0]
    assert per_genome["total_length"] == len(sequence)
    assert abs(per_genome["ir_length"] - ir) <= 20
    assert abs(per_genome["lsc_length"] - 6000) <= 40
    assert abs(per_genome["ssc_length"] - 2000) <= 40


def test_lsc_and_ssc_do_not_swap_when_the_sequence_starts_in_the_ssc(tmp_path: Path) -> None:
    sequence, ir = _plastome()
    start = 6000 + ir + 1000  # middle of the SSC
    rotated = sequence[start:] + sequence[:start]
    fasta = tmp_path / "rotated.fasta"
    fasta.write_text(">plastome\n" + "\n".join(rotated[i : i + 70] for i in range(0, len(rotated), 70)) + "\n")
    genome = read_fasta_genome(fasta, organelle="plastid", species="Synthetic plastid")

    per_genome = ir_boundary([genome]).metrics["per_genome"][0]

    assert abs(per_genome["ir_length"] - ir) <= 20
    assert abs(per_genome["lsc_length"] - 6000) <= 40
    assert abs(per_genome["ssc_length"] - 2000) <= 40


def test_a_long_gene_spanning_a_junction_is_reported_not_intergenic() -> None:
    from organelleverse.ir_boundary.ir_boundary import _gene_at

    genes = [{"name": "ycf1", "start": 1000, "end": 6500}, {"name": "trnN", "start": 6700, "end": 6772}]
    at = _gene_at(genes, 6000)
    assert at["gene"] == "ycf1" and at["distance"] == 0
    near = _gene_at(genes, 6600)
    assert near["gene"] in ("ycf1", "trnN") and near["distance"] == 100
    assert _gene_at(genes, 9000)["gene"] == "intergenic"
