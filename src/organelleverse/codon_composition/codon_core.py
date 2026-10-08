"""Typed cores for codon composition.

Codon tables are sourced from Biopython's NCBI genetic-code registry. The
reported usage table follows the same core columns as EMBOSS ``cusp`` and adds
RSCU for downstream plotting.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any

from .._bio import get_codon_table, get_codon_table_info, read_fasta

_BASES = "TCAG"
_CODON_ORDER = tuple("".join(codon) for codon in product(_BASES, repeat=3))
_ORGANELLE_CODE_DEFAULTS = {
    "mito": 1,
    "mitochondrion": 1,
    "mitochondria": 1,
    "mt": 1,
    "plastid": 11,
    "chloroplast": 11,
    "chloro": 11,
    "cp": 11,
    "pt": 11,
}
_GENETIC_CODE_ALIASES = {
    "standard": 1,
    "plant_mitochondrial": 1,
    "plant_mito": 1,
    "plastid": 11,
    "plant_plastid": 11,
    "chloroplast": 11,
    "bacterial": 11,
    "chlorophycean_mitochondrial": 16,
    "scenedesmus_mitochondrial": 22,
    "balanophoraceae_plastid": 32,
}


def compute_codon_usage(
    cds_fasta: str | Path,
    *,
    organelle: str = "mito",
    genetic_code: int | str | None = None,
) -> dict[str, Any]:
    """Compute codon counts, RSCU, EMBOSS-like usage rows, ENC, and GC3s."""
    table_id = resolve_genetic_code(organelle=organelle, genetic_code=genetic_code)
    codon_to_aa = get_codon_table(table_id)
    info = get_codon_table_info(table_id)
    counts = _count_codons(cds_fasta, codon_to_aa)
    rows = _usage_rows(counts, codon_to_aa)
    rscu = {row["codon"]: row["RSCU"] for row in rows}
    gc3s = _gc_at_position(counts, 2)
    enc = compute_enc(counts, codon_to_aa)
    return {
        "ENC": round(enc, 2),
        "GC3s": round(gc3s, 4),
        "total_codons": sum(counts.values()),
        "RSCU": rscu,
        "codon_counts": {codon: counts.get(codon, 0) for codon in _CODON_ORDER},
        "codon_table": rows,
        "genetic_code": table_id,
        "genetic_code_name": ", ".join(info["names"]) if info["names"] else str(table_id),
        "start_codons": tuple(info["start_codons"]),
        "stop_codons": tuple(info["stop_codons"]),
    }


def compute_amino_acid_composition(
    cds_fasta: str | Path,
    *,
    organelle: str = "mito",
    genetic_code: int | str | None = None,
    include_stop: bool = False,
) -> dict[str, Any]:
    """Compute amino-acid composition from CDS using the selected code."""
    table_id = resolve_genetic_code(organelle=organelle, genetic_code=genetic_code)
    codon_to_aa = get_codon_table(table_id)
    counts = _count_codons(cds_fasta, codon_to_aa)
    aa_counts: dict[str, int] = {}
    for codon, count in counts.items():
        aa = codon_to_aa[codon]
        if aa == "*" and not include_stop:
            continue
        aa_counts[aa] = aa_counts.get(aa, 0) + count
    total = sum(aa_counts.values())
    freq = {aa: round(c / total, 4) for aa, c in aa_counts.items()} if total else {}
    return {
        "counts": aa_counts,
        "total": total,
        "frequencies": freq,
        "genetic_code": table_id,
    }


def resolve_genetic_code(
    *,
    organelle: str = "mito",
    genetic_code: int | str | None = None,
) -> int:
    """Resolve user input to an NCBI ``transl_table`` id."""
    if genetic_code is None or str(genetic_code).lower() == "auto":
        return _ORGANELLE_CODE_DEFAULTS.get(str(organelle).lower(), 1)
    if isinstance(genetic_code, int):
        return genetic_code
    text = str(genetic_code).strip().lower().replace("-", "_").replace(" ", "_")
    if text.isdigit():
        return int(text)
    if text in _GENETIC_CODE_ALIASES:
        return _GENETIC_CODE_ALIASES[text]
    raise ValueError(f"unknown genetic_code: {genetic_code!r}")


def compute_enc(counts: dict[str, int], codon_to_aa: dict[str, str]) -> float:
    """Compute Wright's effective number of codons (ENC).

    Wright (1990), Gene 87:23-29: ENC = 2 + 9/F2 + 1/F3 + 5/F4 + 3/F6, where
    each Fi is the mean over the i-fold synonymous families of the raw codon
    homozygosity sum(p_i^2) (range [1/n, 1]), and the constant 2 counts Met
    and Trp. The original implementation here divided the class size by a
    bias-corrected homozygosity in [0, 1], which is near 0 for the moderately
    even codon usage of real organelle genes — so every family hit its
    ceiling and ENC always collapsed to the 61 cap.
    """
    families = _synonymous_families(codon_to_aa, include_stop=False)
    class_homozygosities: dict[int, list[Fraction]] = {}
    for _aa, family in sorted(families.items()):
        n = len(family)
        if n == 1:
            continue  # Met/Trp contribute the constant 2
        total = sum(counts.get(codon, 0) for codon in family)
        if total == 0:
            continue  # amino acid absent from this gene set
        class_homozygosities.setdefault(n, []).append(
            Fraction(sum(counts.get(codon, 0) ** 2 for codon in family), total**2)
        )
    enc = Fraction(2)
    for _n, values in sorted(class_homozygosities.items()):
        mean_homozygosity = sum(values, Fraction(0)) / len(values)
        if mean_homozygosity <= 0:
            continue
        enc += len(values) / mean_homozygosity
    return float(min(Fraction(61), enc))


def _count_codons(cds_fasta: str | Path, codon_to_aa: dict[str, str]) -> dict[str, int]:
    seqs = read_fasta(Path(cds_fasta))
    counts = {codon: 0 for codon in _CODON_ORDER if codon in codon_to_aa}
    for _, seq in seqs:
        seq = seq.upper().replace("U", "T")
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i : i + 3]
            if codon in counts:
                counts[codon] += 1
    return counts


def _usage_rows(counts: dict[str, int], codon_to_aa: dict[str, str]) -> list[dict[str, Any]]:
    families = _synonymous_families(codon_to_aa, include_stop=True)
    family_totals = {
        aa: sum(counts.get(codon, 0) for codon in family) for aa, family in families.items()
    }
    total_codons = sum(counts.values())
    rows: list[dict[str, Any]] = []
    for codon in _CODON_ORDER:
        if codon not in codon_to_aa:
            continue
        aa = codon_to_aa[codon]
        count = counts.get(codon, 0)
        family = families[aa]
        family_total = family_totals[aa]
        fraction = count / family_total if family_total else 0.0
        frequency = (count / total_codons * 1000.0) if total_codons else 0.0
        rscu = fraction * len(family) if family_total else 0.0
        rows.append(
            {
                "codon": codon,
                "AA": aa,
                "fraction": round(fraction, 6),
                "frequency_per_thousand": round(frequency, 6),
                "count": count,
                "RSCU": round(rscu, 4),
            }
        )
    return rows


def _synonymous_families(
    codon_to_aa: dict[str, str],
    *,
    include_stop: bool,
) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for codon in _CODON_ORDER:
        aa = codon_to_aa.get(codon)
        if aa is None or (aa == "*" and not include_stop):
            continue
        families.setdefault(aa, []).append(codon)
    return families


def _gc_at_position(counts: dict[str, int], position: int) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    gc = sum(count for codon, count in counts.items() if codon[position] in "GC")
    return gc / total
