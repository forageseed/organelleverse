"""Typed cores for population (OmicVerse-style data-in/data-out).

VCF parsing is delegated to ``population._parse_vcf_biallelic``, which keeps
concrete biallelic SNPs only (a symbolic ALT such as ``*`` is not an allele).
"""

from __future__ import annotations
from pathlib import Path
from .population import _parse_vcf_biallelic, _fst_at_site


def compute_fst(
    vcf_path: str | Path,
    pop_assignments: dict[str, str],
    estimator: str = "heterozygosity",
) -> dict:
    """Compute per-site F_ST from a VCF using the heterozygosity-ratio estimator.

F_ST = (H_T - H_S) / H_T with allele frequencies pooled per population;
sites where H_S >= H_T are clamped to 0 (so the mean is upward-biased
relative to Weir & Cockerham 1984, whose unbiased per-site values can be
negative - expect moderate correlation, not numeric agreement).

``estimator`` picks the per-site statistic: ``heterozygosity`` (above, the
default), ``weir_cockerham`` (Weir & Cockerham 1984, unclamped, as vcftools and
scikit-allel) or ``hudson`` (Hudson 1992 on allele counts); they are not
interchangeable, see ``docs/operations/population-fst-estimators.md``.

``samples_missing_from_vcf`` lists the ``pop_assignments`` keys absent from the
VCF: a typo there silently shrinks a population instead of raising. Returns dict.
"""
    from .population import FST_ESTIMATORS

    if estimator not in FST_ESTIMATORS:
        raise ValueError(
            f"Unsupported F_ST estimator {estimator!r}; choose one of "
            + ", ".join(sorted(FST_ESTIMATORS))
        )
    site_fst = FST_ESTIMATORS[estimator]
    snps = _parse_vcf_biallelic(vcf_path)
    vcf_samples: set[str] = set()
    for calls in snps.values():
        vcf_samples.update(calls)
    missing_samples = sorted(s for s in pop_assignments if s not in vcf_samples)
    pop1 = [s for s, p in pop_assignments.items() if p == "pop1"]
    pop2 = [s for s, p in pop_assignments.items() if p == "pop2"]
    if not pop1 or not pop2:
        return {
            "windows": [],
            "n_snps": len(snps),
            "samples_missing_from_vcf": missing_samples,
        }
    positions = sorted(snps)
    fst_values = []
    for pos in positions:
        fst = site_fst(snps[pos], pop1, pop2)
        fst_values.append({"pos": pos, "fst": round(fst, 4)})
    return {
        "windows": fst_values,
        "n_snps": len(snps),
        "samples_missing_from_vcf": missing_samples,
        "estimator": estimator,
        "population_sizes": {"pop1": len(pop1), "pop2": len(pop2)},
        "mean_fst": round(sum(f["fst"] for f in fst_values) / len(fst_values), 4)
        if fst_values
        else 0,
    }
