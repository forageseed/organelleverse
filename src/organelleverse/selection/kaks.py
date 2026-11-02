"""Ka/Ks (dN/dS) selection pressure — Nei-Gojobori (1986), PAML-faithful.

Implements the **NG86** method following PAML's ``difcodonNG`` +
``DistanceMatNG86`` (yn00's "(A) Nei-Gojobori" section), verified
position-exact against PAML 4.10.9 on synthetic minimal cases and 213 real
plastid gene pairs (docs/reports/selection-kaks-realdata):

1. **Synonymous/non-synonymous site counting**: per codon, the 9 single-step
   neighbours are classified as synonymous or stop-target; a pair
   contributes ``S = (syn1+syn2)*3/18`` and ``N = 3*(1-nstop/18) - S``,
   rescaled so S+N = 3 per compared codon (stop-neighbour mutations are
   degenerate sites — the naive ``N = 3 - S`` biases dN down).
2. **Pathway enumeration** for multi-nucleotide differences, averaged over
   pathways that avoid stop intermediates (fallback for all-stops paths:
   (0,2) for 2-diff, (1,2) for 3-diff — PAML convention).
3. **Jukes-Cantor correction**: optional (default off = raw p-distance).
   PAML's yn00 NG86 section and the original 1986 paper always apply the
   JC correction; pass ``jukes_cantor=True`` to match it numerically.

Pure Python, no external dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import permutations
from pathlib import Path
from typing import Any

from organelleverse._public_validation import validated_public

from .._bio import get_codon_table
from ..core.result import OrganelleResult
from ..core.errors import OrganelleInputError
from ..sequences import Sequences, as_records
from . import _contract

_CODONS: dict[str, str] = get_codon_table(1)  # NCBI table 1 (standard)

_STOP_CODONS = {"TAA", "TAG", "TGA"}


@validated_public('selection.kaks')
def kaks(
    cds_pairs_fasta: str | Path | Sequences,
    *,
    jukes_cantor: bool = False,
) -> OrganelleResult:
    """Compute Ka/Ks for all pairs of CDS sequences using NG86.

    Parameters
    ----------
    cds_pairs_fasta : path to a FASTA of CDS sequences (codon-aligned or same-length)
    jukes_cantor : if True, apply Jukes-Cantor multiple-hit correction
                   (d = -3/4 * ln(1 - 4p/3)). Default False (raw
                   p-distance). PAML yn00's NG86 output always applies JC
                   — use ``jukes_cantor=True`` to match it numerically.
    """
    parameters = {"jukes_cantor": jukes_cantor}
    seqs = as_records(cds_pairs_fasta)
    if len(seqs) < 2:
        return _contract.failed(
            "kaks",
            summary_text="kaks requires >=2 CDS sequences.",
            anomalies=["too_few_sequences"],
            parameters=parameters,
        )

    metrics = _pairwise_kaks(seqs, jukes_cantor=jukes_cantor)
    if metrics["pairs"] == 0:
        return _contract.failed("kaks", summary_text="No comparable codon pairs.",
                                anomalies=["no_comparable"], parameters=parameters)
    mean_ka, mean_ks = metrics["Ka"], metrics["Ks"]
    ratio, ratio_state = metrics["Ka_Ks"], metrics["Ka_Ks_state"]
    ratio_value = ratio
    if ratio_state == "saturated":
        flags = ("selection_undefined", "ng86_saturated")
        ratio_text = "not estimable (JC saturation)"
    elif ratio_state == "infinite":
        flags = ("ka_ks_infinite",)
        ratio_text = "infinite (Ks=0)"
    elif ratio_state == "undefined":
        flags = ("selection_undefined", "ka_ks_undefined")
        ratio_text = "undefined (Ka=Ks=0)"
    else:
        flags = ("purifying" if ratio < 1 else "positive_selection" if ratio > 1 else "neutral",)
        ratio_text = f"{ratio:.4f}"
    if metrics["saturated_pairs"] and "ng86_saturated" not in flags:
        flags += ("ng86_saturated_pairs_excluded",)

    return _contract.ok(
        "kaks",
        artifacts=(),
        metrics=metrics,
        # The Ka/Ks zero-denominator state stays machine-readable in
        # ``metrics["Ka_Ks_state"]``; a Finding carries one scalar value.
        findings=_contract.findings(
            "kaks",
            (
                ("Ka", mean_ka),
                ("Ks", mean_ks),
                ("Ka_Ks", ratio_value),
            ),
        ),
        flags=flags,
        summary_text=(
            f"Mean Ka={mean_ka}, Ks={mean_ks}, "
            f"Ka/Ks={ratio_text} over {metrics['estimable_pairs']} estimable pairs "
            f"(NG86{'+JC' if jukes_cantor else ''})."
        ),
        method="nei_gojobori_1986",
        parameters=parameters,
    )


def _pairwise_kaks(seqs: list[tuple[str, str]], *, jukes_cantor: bool = False) -> dict[str, Any]:
    """Shared native/result contract; average only jointly estimable pairs."""
    results = []
    for a in range(len(seqs)):
        for b in range(a + 1, len(seqs)):
            row = _nei_gojobori(seqs[a][1].upper().replace("U", "T"),
                                seqs[b][1].upper().replace("U", "T"),
                                jukes_cantor=jukes_cantor)
            if row is not None:
                row["sequence_a"], row["sequence_b"] = seqs[a][0], seqs[b][0]
                ratio, state = _ratio_with_zero_denominator_state(row["Ka"], row["Ks"])
                row.update(Ka_Ks=ratio, Ka_Ks_state=state)
                results.append(row)
    estimable = [r for r in results if r["Ka"] is not None and r["Ks"] is not None]
    ka = sum(r["Ka"] for r in estimable) / len(estimable) if estimable else None
    ks = sum(r["Ks"] for r in estimable) / len(estimable) if estimable else None
    ratio, state = _ratio_with_zero_denominator_state(ka, ks)
    return {"Ka": round(ka, 6) if ka is not None else None,
            "Ks": round(ks, 6) if ks is not None else None,
            "Ka_Ks": round(ratio, 6) if ratio is not None else None,
            "Ka_Ks_state": state if results else "no_comparable", "pairs": len(results),
            "estimable_pairs": len(estimable), "saturated_pairs": len(results)-len(estimable),
            "method": "NG86" + ("+JC" if jukes_cantor else ""), "pair_results": results}


def write_kaks(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write Ka/Ks pair rows from ``kaks()``."""
    metrics: Mapping[str, Any] = result.metrics if isinstance(result, OrganelleResult) else result
    rows = list(metrics.get("pair_results", ()))
    path = _resolve_output_path(output, "kaks.tsv")
    lines = ["pair\tKa\tKs\tKa_Ks"]
    if rows:
        for i, row in enumerate(rows, 1):
            if "Ka_Ks" in row:
                ratio = row["Ka_Ks"]
                if ratio is None:
                    ratio = row.get("Ka_Ks_state", "undefined")
            else:
                ks = row.get("Ks", 0)
                ratio, ratio_state = _ratio_with_zero_denominator_state(row.get("Ka", 0), ks)
                if ratio is None:
                    ratio = ratio_state
            lines.append(f"pair_{i}\t{row.get('Ka', '')}\t{row.get('Ks', '')}\t{ratio}")
    else:
        lines.append(
            f"mean\t{metrics.get('Ka', '')}\t{metrics.get('Ks', '')}\t{metrics.get('Ka_Ks', '')}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def _resolve_output_path(output: str | Path, default_name: str) -> Path:
    path = Path(output)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / default_name


def _ratio_with_zero_denominator_state(
    numerator: float | None, denominator: float | None
) -> tuple[float | None, str]:
    if numerator is None or denominator is None:
        return None, "saturated"
    if denominator != 0:
        return numerator / denominator, "finite"
    if numerator > 0:
        return None, "infinite"
    return None, "undefined"


# =========================================================================
# Standard NG86 implementation
# =========================================================================


def _nei_gojobori(s1: str, s2: str, *, jukes_cantor: bool = False) -> dict | None:
    """Nei-Gojobori (1986) counting method, PAML-faithful.

    1. S/N sites: per codon, count synonymous mutations and mutations that
       hit a stop codon over the 9 single-step neighbours; a pair contributes
       ``S = (syn_1 + syn_2) * 3/18`` and ``N = 3 * (1 - nstop/18) - S``;
       totals are rescaled so S + N = 3 per compared codon. This is PAML's
       ``difcodonNG`` convention (stop-neighbour mutations are degenerate
       sites, excluded from N) — the naive ``N = 3 - S`` overcounts N for
       any codon with stop neighbours and biases dN down.
    2. S_d, N_d: pathway enumeration for multi-nucleotide differences
       (averaged over pathways that avoid stop intermediates; the
       all-paths-through-stops fallback is (0, 2) for 2-diff and (1, 2) for
       3-diff, matching PAML).
    3. Optional Jukes-Cantor correction (PAML's yn00 NG86 section always
       applies it, as does the original 1986 paper; raw p-distance is the
       no-correction variant).

    Returns {"Ka", "Ks"} or None if too few comparable codons.
    """
    if len(s1) != len(s2) or len(s1) % 3:
        raise OrganelleInputError(code="selection.kaks_alignment",
                                  message="Ka/Ks requires equal-length, in-frame codon alignments.")
    L = len(s1) // 3
    if L < 3:
        return None

    S_sites = N_sites = 0.0
    n_compared = 0
    Sd = Nd = 0.0

    for i in range(L):
        c1 = s1[i * 3 : i * 3 + 3]
        c2 = s2[i * 3 : i * 3 + 3]
        if c1 not in _CODONS or c2 not in _CODONS or c1 in _STOP_CODONS or c2 in _STOP_CODONS:
            continue
        n_compared += 1
        syn1, stop1 = _count_syn_and_stop_neighbors(c1)
        syn2, stop2 = _count_syn_and_stop_neighbors(c2)
        pair_s = (syn1 + syn2) * 3.0 / 18.0
        pair_n = 3.0 * (1.0 - (stop1 + stop2) / 18.0) - pair_s
        S_sites += pair_s
        N_sites += pair_n

        # Difference counting
        if c1 == c2:
            continue

        ndiff = sum(1 for x, y in zip(c1, c2, strict=True) if x != y)
        if ndiff == 1:
            # Single nucleotide difference: directly classify
            if _CODONS[c1] == _CODONS[c2]:
                Sd += 1  # synonymous
            else:
                Nd += 1  # non-synonymous
        else:
            # Multi-nucleotide difference: enumerate all pathways
            sd, nd = _count_diffs_pathway(c1, c2)
            Sd += sd
            Nd += nd

    if n_compared == 0:
        return None
    # PAML DistanceMatNG86: rescale so S + N = 3 per compared codon
    if S_sites + N_sites > 0:
        y = n_compared * 3.0 / (S_sites + N_sites)
        S_sites *= y
        N_sites *= y

    # Compute p-distances
    pS = Sd / S_sites if S_sites > 0 else 0.0
    pN = Nd / N_sites if N_sites > 0 else 0.0

    # Optional Jukes-Cantor correction
    if jukes_cantor:
        Ks = _jukes_cantor(pS)
        Ka = _jukes_cantor(pN)
    else:
        Ks = pS
        Ka = pN

    return {"Ka": Ka, "Ks": Ks,
            "Ka_state": "saturated" if Ka is None else "finite",
            "Ks_state": "saturated" if Ks is None else "finite"}


def _count_syn_and_stop_neighbors(codon: str) -> tuple[int, int]:
    """(synonymous mutations, stop-target mutations) over the 9 single-step
    neighbours of a codon (PAML difcodonNG convention; stop codons excluded
    from the codon's own site budget entirely)."""
    if codon not in _CODONS or codon in _STOP_CODONS:
        return (0, 0)
    aa = _CODONS[codon]
    syn = stop = 0
    for pos in range(3):
        for base in "ACGT":
            if base == codon[pos]:
                continue
            mutant = codon[:pos] + base + codon[pos + 1 :]
            if mutant in _STOP_CODONS:
                stop += 1
            elif mutant in _CODONS and _CODONS[mutant] == aa:
                syn += 1
    return (syn, stop)


def _count_s_sites(codon: str) -> float:
    """Count synonymous sites in a codon (naive NG86, kept for callers).

    For each of the 3 positions, examine all 3 possible mutations.
    A site is "synonymous" if the mutation doesn't change the amino acid.
    Returns the total S count (0 to 3, fractional).
    """
    if codon not in _CODONS or codon in _STOP_CODONS:
        return 0.0  # stop codons: no valid S/N sites

    aa = _CODONS[codon]
    s_sites = 0.0
    for pos in range(3):
        for base in "ACGT":
            if base == codon[pos]:
                continue
            mutant = codon[:pos] + base + codon[pos + 1 :]
            if mutant in _CODONS and _CODONS[mutant] == aa:
                s_sites += 1.0 / 3.0  # each mutation = 1/3 of a site
    return s_sites


def _count_diffs_pathway(c1: str, c2: str) -> tuple[float, float]:
    """Enumerate all evolutionary pathways between two codons.

    For codons differing by 2-3 nucleotides, enumerate all possible orders
    of nucleotide changes (pathways). For each pathway, count the number of
    synonymous (Sd) and non-synonymous (Nd) steps. Exclude pathways passing
    through stop codons. Average across all valid pathways.

    This is the exact algorithm from Nei & Gojobori (1986), also used by
    PAML codeml and KaKs_Calculator.
    """
    diff_positions = [i for i in range(3) if c1[i] != c2[i]]
    n_diffs = len(diff_positions)

    # Enumerate all permutations of the differing positions (pathways)
    pathways = list(permutations(diff_positions))

    total_sd = 0.0
    total_nd = 0.0
    valid_pathways = 0

    for path in pathways:
        current = list(c1)
        sd = nd = 0
        valid = True
        for pos in path:
            current[pos] = c2[pos]
            intermediate = "".join(current)
            if intermediate in _STOP_CODONS:
                valid = False
                break
            # classify this step
            # Actually: the codon before this step is the previous `current`
            # Need to track properly
        # Re-do with proper tracking
        current = list(c1)
        sd = nd = 0
        valid = True
        for pos in path:
            prev_codon = "".join(current)
            current[pos] = c2[pos]
            curr_codon = "".join(current)
            if curr_codon in _STOP_CODONS:
                valid = False
                break
            if prev_codon in _CODONS and curr_codon in _CODONS:
                if _CODONS[prev_codon] == _CODONS[curr_codon]:
                    sd += 1
                else:
                    nd += 1

        if valid:
            total_sd += sd
            total_nd += nd
            valid_pathways += 1

    if valid_pathways == 0:
        # PAML's all-paths-through-stops fallback
        return (0.0, 2.0) if n_diffs == 2 else (1.0, 2.0)

    return (total_sd / valid_pathways, total_nd / valid_pathways)


def _jukes_cantor(p: float) -> float | None:
    """Jukes-Cantor (1969) correction: d = -3/4 * ln(1 - 4p/3).

    Saturated p >= 0.75 is not estimable; retain it as None, never zero.
    """
    if p >= 0.75:
        return None
    if p <= 0:
        return 0.0
    import math

    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _codon_s_fraction(codon: str) -> float:
    """Fraction of positions where a single mutation is synonymous.

    Kept for backward compatibility. New code should use ``_count_s_sites``.
    """
    s = _count_s_sites(codon)
    return s / 3.0 if s > 0 else 0.5
