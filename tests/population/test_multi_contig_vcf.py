"""Sites at the same position on different contigs must not overwrite each other."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("vcfpy")

from organelleverse.core.errors import OrganelleInputError  # noqa: E402
from organelleverse.io_core import read_vcf  # noqa: E402
from organelleverse.population import population_core  # noqa: E402
from organelleverse.population.population import fst_scan  # noqa: E402

_SAMPLES = ["p1a", "p1b", "p2a", "p2b"]
_ASSIGN = {"p1a": "pop1", "p1b": "pop1", "p2a": "pop2", "p2b": "pop2"}


def _vcf(path: Path) -> Path:
    header = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=chrM,length=5000>",
        "##contig=<ID=chrC,length=5000>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(_SAMPLES),
    ]
    rows = []
    # chrM: fixed differences between the populations (F_ST = 1)
    for pos in (100, 200, 300):
        rows.append(f"chrM\t{pos}\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/0\t1/1\t1/1")
    # chrC: the same positions, no differentiation
    for pos in (100, 200, 300):
        rows.append(f"chrC\t{pos}\t.\tC\tT\t.\tPASS\t.\tGT\t0/1\t0/1\t0/1\t0/1")
    path.write_text("\n".join(header + rows) + "\n")
    return path


def test_compute_fst_keeps_sites_of_every_contig(tmp_path: Path) -> None:
    result = population_core.compute_fst(str(_vcf(tmp_path / "x.vcf")), _ASSIGN)
    assert result["n_snps"] == 6
    by_contig = {(row["chrom"], row["pos"]): row["fst"] for row in result["windows"]}
    assert all(by_contig[("chrM", pos)] == 1.0 for pos in (100, 200, 300))
    assert all(by_contig[("chrC", pos)] == 0.0 for pos in (100, 200, 300))


def test_fst_scan_windows_are_per_contig(tmp_path: Path) -> None:
    result = fst_scan(_vcf(tmp_path / "x.vcf"), pop_assignments=_ASSIGN, window_size=1000, step=500)
    assert result.metrics["n_snps"] == 6
    assert result.metrics["max_fst"] == 1.0  # chrM's differentiated sites were overwritten before
    top = dict(result.metrics["top_windows"][0])
    assert top["chrom"] == "chrM" and top["n_snps"] == 3


def test_read_vcf_refuses_ambiguous_positions_and_reads_one_contig(tmp_path: Path) -> None:
    vcf = _vcf(tmp_path / "x.vcf")
    with pytest.raises(OrganelleInputError) as excinfo:
        read_vcf(vcf)
    assert excinfo.value.code == "input.vcf_multiple_contigs"
    chrm = read_vcf(vcf, contig="chrM")
    assert sorted(chrm) == [100, 200, 300]
    assert chrm[100]["p2a"] == "1/1"
