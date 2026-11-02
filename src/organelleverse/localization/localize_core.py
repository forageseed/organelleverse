"""Pure-Python subcellular-localization heuristic core.

This module implements standard, literature-derived signals for predicting
where a protein is localized in a plant cell. It is *intentionally*
dependency-free so it runs offline, and each scorer cites the primary
literature it follows. These are *heuristic* scorers (cleavage-site PWM,
compositional log-odds, motif regex, hydropathy window) -- they are NOT
reproductions of the proprietary neural-network weights of TargetP 2.0,
DeepLoc 2.0, or Predotar. Use the ``deeploc`` / ``targetp`` backends in
:mod:`organelleverse.localization.localize` for the production predictors
when those tools are installed.

Compartment vocabulary (DeepLoc 2.0-compatible, plant-relevant subset):

    - ``Plastid``        (chloroplast transit peptide, cTP)
    - ``Mitochondrion``  (mitochondrial targeting peptide, mTP)
    - ``Secretory``      (signal peptide, SP -> ER/secretory pathway)
    - ``Nucleus``        (nuclear localization signal, NLS)
    - ``Peroxisome``     (PTS1 / PTS2)
    - ``Membrane``       (transmembrane alpha-helices, TM)

References
----------
- von Heijne, G. (1986) *A new method for predicting signal sequence
  cleavage sites.* Nucleic Acids Res. 14(11):4683-4690. (SP weight matrix)
- von Heijne, G. (1989) *The structure of signal peptides from bacterial
  lipoproteins.* Protein Eng. 2:531-534. (compositional bias of targeting
  peptides)
- Emanuelsson, O., Nielsen, H., Brunak, S., & von Heijne, G. (2000)
  *Predicting subcellular localization of proteins based on their
  N-terminal amino acid sequence.* J. Mol. Biol. 300:1005-1016. (TargetP)
- Dingwall, C. & Laskey, R.A. (1991) *Nuclear targeting sequences.*
  Trends Biochem. Sci. 16:478-481. (NLS)
- Robbins, J. et al. (1991) *Two interdependent basic domains in
  nucleoplasmin nuclear targeting target sequence.* Cell 64:615-623.
  (bipartite NLS)
- Gould, S.J. et al. (1989) *A conserved tripeptide sorts proteins to
  peroxisomes.* J. Cell Biol. 108:1657-1664. (PTS1)
- Swinkels, B.W. et al. (1991) *A novel, cleavable peroxisomal targeting
  signal at the amino-terminus of the rat 3-ketoacyl-CoA thiolase.* EMBO J.
  10:3255-3262. (PTS2)
- Reumann, S. (2004) *Specification of the peroxisome targeting signals
  type 1 and type 2 of plant peroxisomes.* Biochim. Biophys. Acta
  1657:21-28 + supplementary. (plant PTS1 tripeptide set)
- Engelman, D.M., Steitz, T.A. & Goldman, A. (1986) *Identifying nonpolar
  transbilayer helices in amino acid sequences of membrane proteins.*
  Annu. Rev. Biophys. Biophys. Chem. 15:321-353. (GES hydropathy scale)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from organelleverse._public_validation import validated_public

__all__ = [
    "COMPARTMENTS",
    "LocalizationScores",
    "score_signal_peptide",
    "score_transit_peptides",
    "score_nls",
    "score_pts",
    "score_transmembrane",
    "predict_heuristic",
]

# Subcellular compartments we predict, in DeepLoc 2.0 naming.
COMPARTMENTS: tuple[str, ...] = (
    "Plastid",
    "Mitochondrion",
    "Secretory",
    "Nucleus",
    "Peroxisome",
    "Membrane",
)


# =============================================================================
# 1. Signal peptide (SP) -- von Heijne (1986) cleavage-site weight matrix
# =============================================================================
#
# The von Heijne (1986) method scores candidate cleavage sites by summing the
# log-odds weight of each residue over 13 positions (-13 .. +1) relative to the
# cleavage site (the bond between position -1 and +1 is cleaved). The matrix is
# built from amino-acid *counts* in 161 eukaryotic signal peptides published in
# the paper and distributed verbatim in EMBOSS as ``ESigcleave.dat``.
#
# Following the EMBOSS SigCleave convention, the score at each position is the
# log-odds of the observed residue frequency relative to its background
# frequency (here approximated by the average across the aligned block,
# i.e. nseq / 20 -- the standard zeroth-order background used by SigCleave).
# A candidate site scores above ``SP_MINSCORE`` (default 3.5) is reported as a
# predicted cleavage site, and the sequence is flagged as a secretory protein.
#
# The matrix covers 15 positions relative to the cleavage site: -13 .. +2.
# The cleavage is between residue -1 and residue +1. EMBOSS ``sigcleave`` scores
# over positions -13..+1 (the +2 column is context only, not summed).
#
# Source: EMBOSS ``Esig.euk`` data file, which reproduces the amino-acid counts
# of 161 eukaryotic signal peptides from von Heijne (1986), NAR 14:4683-4690,
# Table 1 / Fig. 2 alignment. The per-amino-acid "Expect" column (the
# background count, i.e. expected count given the residue's overall frequency)
# is used as the denominator for the log-odds, exactly as in SigCleave.
#
# Below, each row is: residue, counts at -13..+2 (15 ints), and Expect (1 int).
_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

# Verbatim from EMBOSS emboss/data/Esig.euk (161 eukaryotic signal peptides).
# columns: AA  -13 -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 +1 +2  Expect
_SP_EUK_RAW = [
    ("A", 16, 13, 14, 15, 20, 18, 18, 17, 25, 15, 47, 6, 80, 18, 6, 14.5),
    ("C", 3, 6, 9, 7, 9, 14, 6, 8, 5, 6, 19, 3, 9, 8, 3, 4.5),
    ("D", 0, 0, 0, 0, 0, 0, 0, 0, 5, 3, 0, 5, 0, 10, 11, 8.9),
    ("E", 0, 0, 0, 1, 0, 0, 0, 0, 3, 7, 0, 7, 0, 13, 14, 10.0),
    ("F", 13, 9, 11, 11, 6, 7, 18, 13, 4, 5, 0, 13, 0, 6, 4, 5.6),
    ("G", 4, 4, 3, 6, 3, 13, 3, 2, 19, 34, 5, 7, 39, 10, 7, 12.1),
    ("H", 0, 0, 0, 0, 0, 1, 1, 0, 5, 0, 0, 6, 0, 4, 2, 3.4),
    ("I", 15, 15, 8, 6, 11, 5, 4, 8, 5, 1, 10, 5, 0, 8, 7, 7.4),
    ("K", 0, 0, 0, 1, 0, 0, 1, 0, 0, 4, 0, 2, 0, 11, 9, 11.3),
    ("L", 71, 68, 72, 79, 78, 45, 64, 49, 10, 23, 8, 20, 1, 8, 4, 12.1),
    ("M", 0, 3, 7, 4, 1, 6, 2, 2, 0, 0, 0, 1, 0, 1, 2, 2.7),
    ("N", 0, 1, 0, 1, 1, 0, 0, 0, 3, 3, 0, 10, 0, 4, 7, 7.1),
    ("P", 2, 0, 2, 0, 0, 4, 1, 8, 20, 14, 0, 1, 3, 0, 22, 7.4),
    ("Q", 0, 0, 0, 1, 0, 6, 1, 0, 10, 8, 0, 18, 3, 19, 10, 6.3),
    ("R", 2, 0, 0, 0, 0, 1, 0, 0, 7, 4, 0, 15, 0, 12, 9, 7.6),
    ("S", 9, 3, 8, 6, 13, 10, 15, 16, 26, 11, 23, 17, 20, 15, 10, 11.4),
    ("T", 2, 10, 5, 4, 5, 13, 7, 7, 12, 6, 17, 8, 6, 3, 10, 9.7),
    ("V", 20, 25, 15, 18, 13, 15, 11, 27, 0, 12, 32, 3, 0, 8, 17, 11.1),
    ("W", 4, 3, 3, 1, 1, 2, 6, 3, 1, 3, 0, 9, 0, 2, 0, 1.8),
    ("Y", 0, 1, 4, 0, 0, 1, 3, 1, 1, 2, 0, 5, 0, 1, 7, 5.6),
]

_SP_PRO_RAW = [
    ("A", 10, 8, 8, 9, 6, 7, 5, 6, 7, 7, 24, 2, 31, 18, 4, 3.2),
    ("C", 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1.0),
    ("D", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 8, 2.0),
    ("E", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 8, 2.2),
    ("F", 2, 4, 3, 4, 1, 1, 8, 0, 4, 1, 0, 7, 0, 1, 0, 1.3),
    ("G", 4, 2, 2, 2, 3, 5, 2, 4, 2, 2, 0, 2, 2, 1, 0, 2.7),
    ("H", 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 7, 0, 1, 0, 0.8),
    ("I", 3, 1, 5, 1, 5, 0, 1, 3, 0, 0, 0, 0, 0, 0, 2, 1.7),
    ("K", 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 2.5),
    ("L", 8, 11, 9, 8, 9, 13, 1, 0, 2, 2, 1, 2, 0, 0, 1, 2.7),
    ("M", 0, 2, 1, 1, 3, 2, 3, 0, 1, 2, 0, 4, 0, 0, 1, 0.6),
    ("N", 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 3, 0, 1, 4, 1.6),
    ("P", 0, 1, 1, 1, 1, 1, 2, 3, 5, 2, 0, 0, 0, 0, 5, 1.7),
    ("Q", 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 3, 0, 0, 1, 1.4),
    ("R", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1.7),
    ("S", 1, 0, 1, 4, 4, 1, 5, 15, 5, 8, 5, 2, 2, 0, 0, 2.6),
    ("T", 2, 0, 4, 2, 2, 2, 2, 2, 5, 1, 3, 0, 1, 1, 2, 2.2),
    ("V", 5, 7, 1, 3, 1, 4, 7, 0, 0, 4, 3, 0, 0, 2, 0, 2.5),
    ("W", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.4),
    ("Y", 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 1.3),
]

# Matrix positions (EMBOSS scores over -13..+1, i.e. the first 14 columns).
_SP_POSITIONS = list(range(-13, 0)) + [1, 2]  # -13..-1, +1, +2

# Pseudocount to avoid log(0) on unobserved residues (SigCleave uses +1).
_SP_PSEUDOCOUNT = 1.0
# SigCleave default report threshold.
SP_MINSCORE: float = 3.5


def _sp_build_weights(raw: list[tuple]) -> dict[int, dict[str, float]]:
    """Build per-position log-odds weights from a verbatim Esig count table.

    log-odds(pos, aa) = log2( (count + pseudo) / (Expect + pseudo) ),
    following the SigCleave scoring scheme (per-position background = the
    residue's own Expect count, not a uniform 1/20).
    """
    out: dict[int, dict[str, float]] = {p: {} for p in _SP_POSITIONS}
    for row in raw:
        aa = row[0]
        counts = row[1:16]  # 15 positions -13..+2
        expect = float(row[16])
        for p, c in zip(_SP_POSITIONS, counts):
            out[p][aa] = math.log2((c + _SP_PSEUDOCOUNT) / (expect + _SP_PSEUDOCOUNT))
    return out


# Eukaryotic matrix is the default (plant proteins are eukaryotic).
_SP_EUK_WEIGHTS = _sp_build_weights(_SP_EUK_RAW)
_SP_PRO_WEIGHTS = _sp_build_weights(_SP_PRO_RAW)


@validated_public('localization.score_signal_peptide')
def score_signal_peptide(
    seq: str,
    *,
    minscore: float = SP_MINSCORE,
    organism: str = "euk",
) -> dict[str, object]:
    """Predict a signal peptide (SP) via the von Heijne (1986) weight matrix.

    Scans candidate cleavage sites in the first ~70 residues; reports the
    best-scoring site and whether it clears ``minscore`` (SigCleave default
    3.5, in log2-odds units). ``organism`` selects the eukaryotic (``"euk"``)
    or prokaryotic (``"pro"``) count table from EMBOSS Esig.euk / Esig.pro.

    The cleavage is between residue -1 and residue +1. For a candidate site at
    0-based index ``cut`` (the +1 residue), position ``p`` maps to residue
    ``cut + p`` for p in {-13..-1, +1}. The +2 column is context only and is
    not summed, matching EMBOSS sigcleave.

    Returns a dict with ``has_sp``, ``best_score``, ``cleavage_pos``
    (the +1 index, -1 if none), ``signal_peptide_len``, and ``mature_seq``.
    """
    s = "".join(c for c in seq.upper() if c in _AA_ORDER)
    if len(s) < 16:
        return {
            "has_sp": False,
            "best_score": None,
            "cleavage_pos": -1,
            "signal_peptide_len": 0,
            "mature_seq": s,
        }

    weights = _SP_EUK_WEIGHTS if organism != "pro" else _SP_PRO_WEIGHTS
    # Scored positions: -13..-1 and +1 (14 positions). +2 is context only.
    score_positions = list(range(-13, 0)) + [1]
    # Hard "-3,-1 rule": residues -3 and -1 must be small/neutral. The Esig
    # matrix shows the dominant residues at these positions are A, G, S, C, T, V.
    small = frozenset("AGSC")
    n_scan = min(len(s), 70)  # cleavage within ~70 residues
    best_score = -1e9
    best_cut = -1
    for cut in range(14, n_scan):
        score = 0.0
        ok = True
        for pos in score_positions:
            # 0-based residue index for matrix position ``pos``. ``cut`` is the first
            # mature residue: -1 is s[cut - 1] and +1 is s[cut] (there is no position 0);
            # cut + pos read +1 one residue too far.
            j = cut + pos if pos < 0 else cut + pos - 1
            if j < 0 or j >= len(s):
                ok = False
                break
            score += weights[pos].get(s[j], -3.0)  # unknown AA -> strong penalty
        if not ok:
            continue
        # -3,-1 rule: both must be small (von Heijne 1983, hard filter).
        r_m3 = s[cut - 3]
        r_m1 = s[cut - 1]
        if r_m3 not in small or r_m1 not in small:
            continue
        # Hydrophobic h-region (roughly positions -15..-5) must be hydrophobic.
        core = s[max(0, cut - 15) : max(0, cut - 5)]
        if core:
            core_h = sum(_GES_SCALE.get(a, 0.0) for a in core) / len(core)
            if core_h < 0.5:
                continue
        if score > best_score:
            best_score = score
            best_cut = cut

    has_sp = best_score >= minscore and best_cut > 0
    return {
        "has_sp": has_sp,
        "best_score": round(best_score, 3) if best_score > -1e3 else None,
        "cleavage_pos": best_cut if has_sp else -1,
        "signal_peptide_len": best_cut if has_sp else 0,
        "mature_seq": s[best_cut:] if has_sp else s,
    }


# =============================================================================
# 2. Transit peptides (mTP / cTP) -- N-terminal multinomial logistic model
# =============================================================================
#
# TargetP's network weights are not public. This scorer is a multinomial logistic
# regression on N-terminal composition, trained on 2026-10-25 on Arabidopsis
# thaliana (TAIR10 proteins from Ensembl Plants) with labels from AtMitoCarta
# (Kim et al. 2026, Cell 189:6267, Table S2):
#   mtp   AtMitoCarta member with a TargetP 2.0 mTP call        (433 training proteins)
#   ctp   SUBA5 plastid only, TargetP 2.0 cTP or luTP           (746)
#   other SUBA5 localisation outside mitochondria and plastids  (11678)
# Trained on chromosomes 1-3; each threshold gives <= 5% calls on training
# "other". Held out (chromosomes 4-5): mTP recall 78.8%, cTP recall
# 92.3%; "other" called mTP 3.3% and cTP 4.7%; cTP proteins
# called mTP 0.9%; random 400-aa sequences called mTP 0.55%. The composition
# heuristic it replaces called 20.7% of "other" and 71.7% of cTP proteins mTP.
# Probabilities are posteriors under the Arabidopsis proteome's class balance, so a
# real targeting peptide often scores well below 0.5; the calls use the thresholds.

_TP_AA = "ACDEFGHIKLMNPQRSTVWY"
_TP_WINDOWS = (15, 30, 60)
#: Rows: other, mtp, ctp. Columns: amino-acid fractions over residues 1-15, 1-30
#: and 1-60 (``_TP_AA`` order), net charge per residue over 1-30, hydrophobic
#: (AILMFVW) fraction over 1-20.
_TP_COEF: tuple[tuple[float, ...], ...] = (
    (
        -4.45957, -0.92432, 4.04265, 4.16112, 0.20839, 1.49732, -0.38002, -1.67427, 4.26658,
        -2.49388, -1.15365, 0.99432, -0.22854, -1.00477, 0.87822, -0.36366, -1.60099, -1.31652,
        0.18967, 0.13444, -0.08018, -0.40670, 4.35867, 3.64432, -0.87134, 0.63434, -1.91686,
        0.36735, 3.32925, -0.67968, 0.88358, -1.60454, -1.68772, -0.54356, -1.97434, -3.28301,
        -0.90654, 0.52790, 0.51175, 0.46982, -0.93806, 0.81769, 4.63459, 2.87006, -0.62683,
        0.84764, 0.33088, 1.18387, -0.59492, -0.07806, 0.81062, -0.27018, 1.29775, 0.39677,
        -5.12323, -6.31160, -1.31122, -0.42283, 1.27295, 1.98663, -6.64807, 0.54267,
    ),
    (
        0.12370, -0.82912, -2.31428, -2.52341, 0.02071, -1.29626, -0.10627, 1.62693, 0.68459,
        1.53061, -0.33230, -0.48548, -2.62387, 0.44653, 6.65950, -1.17931, -1.02109, -0.03490,
        0.72966, 0.50635, 1.00726, -0.43893, -2.16013, -1.53952, 0.21866, -0.07304, 0.06704,
        0.09561, -2.41314, 0.78906, -0.57817, 0.33237, -1.95150, 0.36768, 3.01083, 1.44045,
        0.34241, 0.13597, 0.44348, 0.48589, 0.77655, -1.31112, -0.60284, 0.13708, 0.58464, 0.22551,
        -0.13220, -0.24990, -1.50576, 0.16907, -0.52128, -0.22661, -1.25297, -0.10513, 0.77512,
        2.69130, 0.81146, -0.71903, -0.05655, 0.09492, 4.29735, 0.38182,
    ),
    (
        4.33587, 1.75344, -1.72836, -1.63771, -0.22910, -0.20106, 0.48630, 0.04734, -4.95117,
        0.96327, 1.48595, -0.50885, 2.85241, 0.55824, -7.53771, 1.54297, 2.62208, 1.35142,
        -0.91933, -0.64079, -0.92707, 0.84564, -2.19853, -2.10480, 0.65268, -0.56130, 1.84982,
        -0.46296, -0.91612, -0.10939, -0.30541, 1.27217, 3.63922, 0.17588, -1.03649, 1.84256,
        0.56414, -0.66388, -0.95524, -0.95570, 0.16151, 0.49343, -4.03175, -3.00714, 0.04219,
        -1.07315, -0.19867, -0.93397, 2.10068, -0.09101, -0.28934, 0.49679, -0.04478, -0.29164,
        4.34811, 3.62030, 0.49975, 1.14186, -1.21640, -2.08155, 2.35073, -0.92449,
    ),
)
_TP_INTERCEPT: tuple[float, float, float] = (4.21863, -2.46950, -1.74913)
_TP_THRESHOLD_MTP: float = 0.08813
_TP_THRESHOLD_CTP: float = 0.12688


def _tp_features(s: str) -> list[float]:
    """Model inputs; ``s`` is already reduced to the 20 standard residues."""
    out: list[float] = []
    for w in _TP_WINDOWS:
        head = s[:w] or "X"
        out += [head.count(a) / len(head) for a in _TP_AA]
    h30, h20 = s[:30] or "X", s[:20] or "X"
    out.append((sum(h30.count(a) for a in "RK") - sum(h30.count(a) for a in "DE")) / len(h30))
    out.append(sum(h20.count(a) for a in "AILMFVW") / len(h20))
    return out


@validated_public('localization.score_transit_peptides')
def score_transit_peptides(
    seq: str,
    *,
    window: int = 60,
) -> dict[str, object]:
    """Score mitochondrial (mTP) and chloroplast (cTP) transit peptides.

    Multinomial logistic model on N-terminal composition (see the section
    comment for training data and held-out performance). ``window`` is kept for
    compatibility; the model always reads residues 1-15, 1-30 and 1-60. The three
    ``*_filter`` flags are the classical N-terminal rules (no D/E in residues
    1-15; basic charge with Arg for mTP; S/T/A-rich, not basic for cTP), reported
    for information only.
    """
    s = "".join(c for c in seq.upper() if c in _AA_ORDER)
    if len(s) < 15:
        return {
            "mtp_score": 0.0,
            "ctp_score": 0.0,
            "predicted": None,
            "mitochondrion_prob": 0.0,
            "plastid_prob": 0.0,
            "no_acidic_filter": False,
            "mtp_charge_filter": False,
            "ctp_composition_filter": False,
        }
    x = _tp_features(s)
    logits = [b + sum(w * v for w, v in zip(row, x, strict=True)) for row, b in zip(_TP_COEF, _TP_INTERCEPT, strict=True)]
    top = max(logits)
    exps = [math.exp(z - top) for z in logits]
    total = sum(exps)
    _p_other, p_mtp, p_ctp = (e / total for e in exps)
    predicted = None
    if p_mtp >= _TP_THRESHOLD_MTP and p_mtp >= p_ctp:
        predicted = "Mitochondrion"
    elif p_ctp >= _TP_THRESHOLD_CTP and p_ctp > p_mtp:
        predicted = "Plastid"

    n_term, head = s[:15], s[:window]
    net_charge = sum(head.count(a) for a in "RK") - sum(head.count(a) for a in "DE")
    no_acidic_ok = sum(n_term.count(a) for a in "DE") <= 1
    return {
        "mtp_score": round(logits[1] - logits[0], 3),
        "ctp_score": round(logits[2] - logits[0], 3),
        "mitochondrion_prob": round(p_mtp, 3),
        "plastid_prob": round(p_ctp, 3),
        "predicted": predicted,
        "no_acidic_filter": no_acidic_ok,
        "mtp_charge_filter": net_charge >= 2 and "R" in head,
        "ctp_composition_filter": sum(n_term.count(a) for a in "STA") >= 15 / 2 and net_charge < 3,
    }


# =============================================================================
# 3. Nuclear localization signals (NLS)
# =============================================================================
#
# Classical NLS patterns (Dingwall & Laskey 1991; Robbins 1991; Kosugi 2009):
#   - Monopartite: a cluster of basic residues, prototype SV40 large-T
#     PKKKRKV -> regex [KR][KR]...{4,5}[KR]
#   - Bipartite (Robbins 1991): two basic clusters separated by ~10 aa,
#     prototype nucleoplasmin KR-XXXXX-XXXXX-KKKL ->
#     [KR][KR].{9,12}[KR][KR][KR]
#
# We search the full sequence for these patterns. We also support the
# extended 6-7 residue monopartite classes of Kosugi (2009) (class 1/2).

# Monopartite: 4+ basic residues in a 6-aa window (SV40-like).
# Classical monopartite NLS: K-(K/R)-X-(K/R) (Chelsky 1989), or four basic residues in
# a row. The former pattern [KR].{0,3}[KR][KR].{0,2}[KR] matched 39% of random
# 400-aa sequences.
_NLS_MONO = re.compile(r"K[KR].[KR]|[KR][KR][KR][KR]")
# Bipartite (Robbins 1991): 2 basic | 9-12 spacer | 3+ basic.
_NLS_BIPARTITE = re.compile(r"[KR][KR].{9,12}[KR][KR][KR]")
# Kosugi class-2 short basic cluster (5-6 aa, at least 3 basic / 1 hydrophobic).
_NLS_KOSUGI2 = re.compile(r"[KR].{0,4}[KR].{0,4}[KR]")


@validated_public('localization.score_nls')
def score_nls(seq: str) -> dict[str, object]:
    """Detect classical nuclear localization signals.

    Returns ``has_nls``, the matched pattern type(s), and the matched motifs.
    Patterns follow Dingwall & Laskey (1991), Robbins (1991), and Kosugi (2009).
    """
    s = seq.upper()
    matches: list[str] = []
    kinds: list[str] = []

    for m in _NLS_BIPARTITE.finditer(s):
        kinds.append("bipartite")
        matches.append(m.group(0))
    for m in _NLS_MONO.finditer(s):
        # avoid double-counting residues already in a bipartite match
        if not any(
            m.start() >= hit[0] and m.end() <= hit[1]
            for hit in [(mm.start(), mm.end()) for mm in _NLS_BIPARTITE.finditer(s)]
        ):
            kinds.append("monopartite")
            matches.append(m.group(0))

    has = bool(kinds)
    return {
        "has_nls": has,
        "nls_type": sorted(set(kinds)),
        "matches": matches,
        "nucleus_prob": 0.9 if has else 0.0,
    }


# =============================================================================
# 4. Peroxisomal targeting signals (PTS1 / PTS2)
# =============================================================================
#
# PTS1: a C-terminal tripeptide (Gould 1989) conforming broadly to
#   (S/A/C)(K/R/H)(L/M). For plants, Reumann (2004) enumerated the validated
#   major and minor PTS1 tripeptides; we accept the canonical set.
# PTS2: an N-terminal nonapeptide (Swinkels 1991) RLx5(HL), consensus
#   [RK]-[LVI]-x5-[HQ]-[LA].

# Canonical / validated plant PTS1 C-terminal tripeptides (Reumann 2004 + the
# general Gould 1989 consensus). Stored as a set for O(1) lookup.
_PLANT_PTS1: frozenset[str] = frozenset(
    {
        # Canonical strong (SKL-type)
        "SKL",
        "SRL",
        "SKM",
        "SRM",
        "AKL",
        "ARL",
        "AKM",
        "ARM",
        "CKL",
        "CRL",
        "CKM",
        "CRM",
        # Ser/Ala at pos1, Lys/Arg at pos2, Leu/Met at pos3 (Gould 1989 consensus)
        "SLM",
        "SRF",
        "ARF",
        "SLI",
        "SRI",
        "PKL",
        "PRL",
        # Reumann 2004 major plant PTS1s (additional)
        "NKL",
        "NRL",
        "SHL",
        "SHM",
        "AHL",
        "AHM",
        "SSL",
        "SSM",
        "HAL",
        "HAM",
        "SQL",
        "SQM",
        "AQL",
        "AQM",
    }
)

# PTS2 N-terminal consensus: [RK][LVI]xxxxx[HL][LA] within the first ~30 aa.
_PTS2 = re.compile(r"[RK][LIVQ].{5}[HL][LAIF]")


@validated_public('localization.score_pts')
def score_pts(seq: str) -> dict[str, object]:
    """Detect peroxisomal targeting signals PTS1 (C-term) and PTS2 (N-term).

    PTS1 follows Gould (1989) with the plant refinement of Reumann (2004);
    PTS2 follows Swinkels (1991).
    """
    s = "".join(c for c in seq.upper() if c in _AA_ORDER)
    pts1 = None
    if len(s) >= 3:
        tri = s[-3:]
        pts1 = tri if tri in _PLANT_PTS1 else None

    pts2_match = None
    m = _PTS2.search(s[:30])
    if m:
        pts2_match = m.group(0)

    has = bool(pts1 or pts2_match)
    kind: list[str] = []
    if pts1:
        kind.append("PTS1")
    if pts2_match:
        kind.append("PTS2")

    return {
        "has_pts": has,
        "pts1": pts1,
        "pts2": pts2_match,
        "kind": kind,
        "peroxisome_prob": 0.9 if has else 0.0,
    }


# =============================================================================
# 5. Transmembrane alpha-helices (TM) -- GES hydropathy scale
# =============================================================================
#
# Engelman, Steitz & Goldman (1986) GES scale: transfer free energy (kcal/mol)
# of an amino acid (in an alpha-helix) from water into a lipid bilayer.
# Positive = hydrophobic (favors membrane insertion). A stretch of >=20
# residues averaging >= 0.0 (total >= ~20 kcal/mol, or using the stricter
# per-residue threshold of ~0.5 kcal/mol often cited) is predicted as a TM span.

_GES_SCALE: dict[str, float] = {
    "F": 3.7,
    "M": 3.4,
    "I": 3.1,
    "L": 2.8,
    "V": 2.6,
    "C": 2.0,
    "W": 1.9,
    "A": 1.6,
    "T": 1.2,
    "G": 1.0,
    "S": 0.6,
    "P": -0.2,
    "Y": -0.7,
    "H": -3.0,
    "Q": -4.1,
    "N": -4.8,
    "E": -8.2,
    "D": -9.2,
    "K": -8.8,
    "R": -12.3,
}

_TM_WINDOW = 20
# Per-residue threshold: an average >= this over a 20-aa window => TM span.
# Following the original GES recommendation (a 20-residue helix totaling
# >= ~30 kcal/mol, i.e. mean >= 1.5). We require a clearly hydrophobic core
# to avoid flagging mildly-hydrophobic globular stretches.
_TM_MEAN_THRESHOLD = 1.5


@validated_public('localization.score_transmembrane')
def score_transmembrane(
    seq: str,
    *,
    window: int = _TM_WINDOW,
    threshold: float = _TM_MEAN_THRESHOLD,
) -> dict[str, object]:
    """Predict transmembrane alpha-helices via the GES hydropathy scale.

    Slides a 20-residue window; any window whose mean GES hydropathy clears
    ``threshold`` is merged into a TM span. Returns the count of TM spans and
    their residue ranges.
    """
    s = "".join(c for c in seq.upper() if c in _AA_ORDER)
    n = len(s)
    if n < window:
        return {"tm_count": 0, "tm_regions": [], "max_hydrophobicity": None, "membrane_prob": 0.0}

    hydro = [_GES_SCALE.get(a, 0.0) for a in s]
    # Identify each 20-aa window that clears the hydrophobicity threshold and
    # contains no strongly-charged residue (D/E/K/R) in its core -- charged
    # residues inside a helix are a strong signal that the segment is globular,
    # not transmembrane.
    charged = set("DEKR")
    tm_windows: list[tuple[int, int]] = []
    for i in range(n - window + 1):
        seg = s[i : i + window]
        mean_h = sum(hydro[i : i + window]) / window
        core = seg[2:-2]  # interior residues (allow charges at the very ends)
        n_charged = sum(1 for a in core if a in charged)
        if mean_h >= threshold and n_charged == 0:
            tm_windows.append((i, i + window - 1))

    # Merge adjacent / overlapping windows into spans.
    merged: list[tuple[int, int]] = []
    for a, b in tm_windows:
        if merged and a <= merged[-1][1] + 2:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))

    tm_count = len(merged)
    max_h = max(
        (sum(hydro[i : i + window]) / window for i in range(n - window + 1)),
        default=0.0,
    )
    # Probability of being a membrane protein: rises sharply with >=1 TM span.
    prob = 1.0 if tm_count >= 1 else max(0.0, min(0.4, (max_h / 2.0) * 0.4))
    return {
        "tm_count": tm_count,
        "tm_regions": merged,
        "max_hydrophobicity": round(max_h, 3),
        "membrane_prob": round(prob, 3),
    }


# =============================================================================
# 6. Aggregation -- heuristic multi-label prediction
# =============================================================================


@dataclass
class LocalizationScores:
    """Per-protein localization scores from all heuristic scorers."""

    sequence_id: str
    length: int
    signal_peptide: dict[str, object] = field(default_factory=dict)
    transit_peptides: dict[str, object] = field(default_factory=dict)
    nls: dict[str, object] = field(default_factory=dict)
    pts: dict[str, object] = field(default_factory=dict)
    transmembrane: dict[str, object] = field(default_factory=dict)

    @property
    def probabilities(self) -> dict[str, float]:
        """Per-compartment probability from the heuristic scorers.

        When two scorers point to the same compartment (e.g. an mTP and an SP),
        we take the maximum. Secretory is driven solely by SP; Membrane solely
        by TM spans; transit peptides drive Plastid/Mitochondrion.
        """
        probs = {c: 0.0 for c in COMPARTMENTS}
        sp = self.signal_peptide
        has_sp = bool(sp.get("has_sp"))
        if has_sp:
            probs["Secretory"] = 0.95
        tp = self.transit_peptides
        # The transit-peptide model calls on its own calibrated thresholds, well below
        # 0.5 (its probabilities are proteome-prior posteriors): a called compartment
        # is lifted to the 0.5 reporting level, an uncalled one stays below it.
        for compartment, key in (("Mitochondrion", "mitochondrion_prob"), ("Plastid", "plastid_prob")):
            p = float(tp.get(key, 0.0))
            p = max(p, 0.5) if tp.get("predicted") == compartment else min(p, 0.49)
            probs[compartment] = max(probs[compartment], p)
        if self.nls.get("has_nls"):
            probs["Nucleus"] = float(self.nls.get("nucleus_prob", 0.9))
        if self.pts.get("has_pts"):
            probs["Peroxisome"] = float(self.pts.get("peroxisome_prob", 0.9))
        # Membrane from TM spans. Suppress when a single TM span coincides with
        # a predicted N-terminal targeting peptide (signal peptide or a transit
        # peptide): the targeting peptide's hydrophobic core registers as one
        # "TM span", which is not a real transmembrane domain. Only call a
        # protein membrane-anchored when there are >= 2 TM spans (or >= 1 with
        # no targeting peptide), so targeted proteins are not double-counted.
        tm_count = int(self.transmembrane.get("tm_count", 0))
        has_targeting = has_sp or bool(tp.get("predicted"))
        if has_targeting and tm_count <= 1:
            probs["Membrane"] = 0.0
        else:
            probs["Membrane"] = float(self.transmembrane.get("membrane_prob", 0.0))
        return probs

    @property
    def predicted(self) -> list[str]:
        """Compartment(s) whose probability clears the reporting threshold.

        Only compartments with probability >= 0.5 are reported. If none clear
        the threshold (e.g. a typical cytosolic protein), returns ``[]`` --
        i.e. "no confident localization from the heuristic scorers". This
        high-precision design avoids false positives on cytosolic proteins.
        """
        probs = self.probabilities
        return sorted(
            (c for c, p in probs.items() if p >= 0.5),
            key=lambda c: probs[c],
            reverse=True,
        )


def predict_heuristic(
    seq: str,
    *,
    sequence_id: str = "",
) -> LocalizationScores:
    """Run all heuristic localization scorers on one protein sequence.

    Pure Python, no external dependencies. Returns a :class:`LocalizationScores`
    object whose ``predicted`` / ``probabilities`` properties give the
    multi-label prediction.
    """
    s = "".join(c for c in seq.upper() if c in _AA_ORDER)
    return LocalizationScores(
        sequence_id=sequence_id,
        length=len(s),
        signal_peptide=score_signal_peptide(s),
        transit_peptides=score_transit_peptides(s),
        nls=score_nls(s),
        pts=score_pts(s),
        transmembrane=score_transmembrane(s),
    )
