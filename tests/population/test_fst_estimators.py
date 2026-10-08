"""F_ST estimators: the default is unchanged, Weir-Cockerham and Hudson match independent code.

On the real Triticeae plastomes the package's heterozygosity-ratio estimator
averaged 0.139 while vcftools / scikit-allel Weir-Cockerham averaged 0.043
(87% of sites negative): different quantities that are easy to confuse. These
tests pin the three estimators against scikit-allel, which does not share code
with the package.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from organelleverse.population import population as pop
from organelleverse.population import population_core

allel = pytest.importorskip("allel")
np = pytest.importorskip("numpy")

_HEAD = """\
##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{cols}
"""


def _write_vcf(path: Path, genotypes: list[list[str]]) -> tuple[Path, dict[str, str]]:
    n = len(genotypes[0])
    samples = [f"s{i}" for i in range(n)]
    rows = [
        "\t".join(["chr1", str(100 + 10 * i), ".", "A", "T", ".", "PASS", ".", "GT", *gts])
        for i, gts in enumerate(genotypes)
    ]
    path.write_text(_HEAD.format(cols="\t".join(samples)) + "\n".join(rows) + "\n", encoding="utf-8")
    assign = {s: ("pop1" if i < n // 2 else "pop2") for i, s in enumerate(samples)}
    return path, assign


def _random_genotypes(
    n_sites: int, n_samples: int, *, diploid: bool, seed: int, missing: float = 0.05
) -> list[list[str]]:
    rng = random.Random(seed)
    sites = []
    for _ in range(n_sites):
        shift = rng.uniform(-0.3, 0.3)  # differentiation between the two halves
        row = []
        for i in range(n_samples):
            p = min(0.95, max(0.05, 0.5 + (shift if i < n_samples // 2 else -shift)))
            alleles = [str(int(rng.random() < p)) for _ in range(2 if diploid else 1)]
            if rng.random() < missing:
                alleles = ["."] * len(alleles)  # missing call
            row.append("/".join(alleles))
        sites.append(row)
    return sites


def _site_values(result: dict) -> np.ndarray:
    return np.array([w["fst"] for w in result["windows"]])


def test_hudson_matches_scikit_allel_per_site_on_haploid_calls(tmp_path: Path) -> None:
    genotypes = _random_genotypes(60, 20, diploid=False, seed=1)
    vcf, assign = _write_vcf(tmp_path / "h.vcf", genotypes)
    ours = _site_values(population_core.compute_fst(str(vcf), assign, estimator="hudson"))

    ac1, ac2 = [], []
    for row in genotypes:
        for target, calls in ((ac1, row[:10]), (ac2, row[10:])):
            alts = sum(1 for g in calls if g == "1")
            refs = sum(1 for g in calls if g == "0")
            target.append([refs, alts])
    num, den = allel.hudson_fst(np.array(ac1), np.array(ac2))
    expected = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
    assert np.max(np.abs(ours - np.round(expected, 4))) < 1.1e-4  # per-site values are rounded to 4 decimals


def test_weir_cockerham_matches_scikit_allel_per_site_on_diploid_calls(tmp_path: Path) -> None:
    genotypes = _random_genotypes(60, 24, diploid=True, seed=2, missing=0.0)
    vcf, assign = _write_vcf(tmp_path / "d.vcf", genotypes)
    ours = _site_values(population_core.compute_fst(str(vcf), assign, estimator="weir_cockerham"))

    calls = np.full((len(genotypes), 24, 2), -1, dtype="i1")
    for i, row in enumerate(genotypes):
        for j, cell in enumerate(row):
            if "." not in cell:
                calls[i, j] = [int(x) for x in cell.split("/")]
    g = allel.GenotypeArray(calls)
    a, b, c = allel.weir_cockerham_fst(g, [list(range(12)), list(range(12, 24))], max_allele=1)
    expected = a.sum(axis=1) / (a.sum(axis=1) + b.sum(axis=1) + c.sum(axis=1))
    assert np.max(np.abs(ours - np.round(expected, 4))) < 1.1e-4


def test_weir_cockerham_haploid_calls_count_as_homozygous_diploids(tmp_path: Path) -> None:
    """The vcftools convention for haploid organelle VCFs (written as 0/0, 1/1)."""
    haploid = _random_genotypes(40, 16, diploid=False, seed=3)
    doubled = [[("." if g == "." else f"{g}/{g}") for g in row] for row in haploid]
    v1, assign = _write_vcf(tmp_path / "hap.vcf", haploid)
    v2, _ = _write_vcf(tmp_path / "dip.vcf", doubled)
    one = population_core.compute_fst(str(v1), assign, estimator="weir_cockerham")
    two = population_core.compute_fst(str(v2), assign, estimator="weir_cockerham")
    assert one["windows"] == two["windows"]


def test_default_estimator_is_the_heterozygosity_ratio_and_unchanged(tmp_path: Path) -> None:
    # pop1 = 3 x ref + 1 x alt, pop2 = 1 x ref + 3 x alt  ->  p1 = .25, p2 = .75
    vcf, assign = _write_vcf(tmp_path / "x.vcf", [["0", "0", "0", "1", "0", "1", "1", "1"]])
    result = population_core.compute_fst(str(vcf), assign)
    h_total = 2 * 0.5 * 0.5
    h_within = 2 * 0.25 * 0.75
    assert result["estimator"] == "heterozygosity"
    assert result["windows"][0]["fst"] == round((h_total - h_within) / h_total, 4)


def test_estimators_are_not_interchangeable(tmp_path: Path) -> None:
    genotypes = _random_genotypes(200, 20, diploid=False, seed=4)
    vcf, assign = _write_vcf(tmp_path / "m.vcf", genotypes)
    means = {
        name: population_core.compute_fst(str(vcf), assign, estimator=name)["mean_fst"]
        for name in ("heterozygosity", "weir_cockerham", "hudson")
    }
    assert means["heterozygosity"] > means["weir_cockerham"]  # the clamp at 0 biases it upward
    assert len({round(v, 3) for v in means.values()}) == 3


def test_unknown_estimator_is_rejected_not_silently_defaulted(tmp_path: Path) -> None:
    vcf, assign = _write_vcf(tmp_path / "u.vcf", [["0", "1", "0", "1"]] * 5)
    with pytest.raises(ValueError, match="Unsupported F_ST estimator"):
        population_core.compute_fst(str(vcf), assign, estimator="nei")
    failed = pop.fst_scan(str(vcf), pop_assignments=assign, window_size=1000, step=500, estimator="nei")
    assert failed.status == "failed"
    assert failed.errors[0].code == "population.fst_scan.unsupported_estimator"


def test_fst_scan_reports_the_estimator_it_used(tmp_path: Path) -> None:
    genotypes = _random_genotypes(30, 12, diploid=False, seed=5)
    vcf, assign = _write_vcf(tmp_path / "s.vcf", genotypes)
    for name in ("heterozygosity", "weir_cockerham", "hudson"):
        result = pop.fst_scan(str(vcf), pop_assignments=assign, window_size=100, step=50, estimator=name)
        assert result.status == "ok"
        assert result.metrics["estimator"] == name
