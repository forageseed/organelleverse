"""Anticodon / isotype naming for tRNAs found by the CM engine.

Given a coordinate-exact mature tRNA sequence, the CYK traceback maps the CM's
consensus columns to sequence positions; reading the three anticodon consensus
columns yields the anticodon, hence the amino acid and gene name (``trnX-YYY``).
This is how Infernal/tRNAscan read the anticodon, and is reliable where a purely
structural heuristic mis-ranks the anticodon loop.
"""

from __future__ import annotations

from ..trna_core.anticodon import isotypes_for_anticodon, normalize_anticodon
from .cyk import cyk_trace
from .models import CovarianceModel

# Fallback anticodon consensus columns for the packaged ``plant_mito_trna.cm``.
# Normally the columns are auto-calibrated per model (see below), so a model
# rebuild needs no manual update; this is only used if calibration fails.
DEFAULT_ANTICODON_COLUMNS = (81, 82, 83)

# Reference tRNAs (anticodon, mature sequence) of known identity, used to
# auto-calibrate which consensus columns hold the anticodon for a given model.
_CALIBRATION_REFS = (
    ("GTC", "GGGATTGTAGTTCAATCGGTCAGAGCACCGCCCTGTCAAGGCGGAAGCTGCGGGTTCGAGCCCCGTCAGTCCCG"),
    ("TTC", "GTCCCTTTCGTCCAGTGGTTAGGACATCGTCTTTTCATGTCGAAGACACGGGTTCGATTCCCGTAAGGGATA"),
    ("GAA", "GTTCAGGTAGCTCAGCTGGTTAGAGCAAAGGACTGAAAATCCTTGTGTCAGTGGTTCGAATCCACTTCTAAGCG"),
    ("GTG", "GCGGATGTAGCCAAGTGGATCAAGGCAGTGGATTGTGAATCCACCATGCGCGGGTTCAATTCCCGTCGTTCGCC"),
)

# Per-model cache of calibrated anticodon columns, keyed by id(model). Each entry keeps the model
# itself and is only used when it is that same object: CPython reuses the id of a collected object,
# so an id alone can hand one model's columns to a different model (a plant-mito CM freed, a plastid
# CM allocated at the same address) and the anticodon is then read from the wrong columns.
_CALIBRATION_CACHE: dict[int, tuple[CovarianceModel, tuple[int, int, int]]] = {}


def _calibrate_anticodon_columns(model: CovarianceModel) -> tuple[int, int, int]:
    """Derive the 3 anticodon consensus columns by aligning reference tRNAs.

    The anticodon occupies the same consensus columns in every tRNA, so the
    column triple that reads each reference's known anticodon is the anticodon
    position. Returns ``DEFAULT_ANTICODON_COLUMNS`` if no consistent triple is
    found. Cached per model.
    """
    from .cyk import cyk_trace

    key = id(model)
    cached = _CALIBRATION_CACHE.get(key)
    if cached is not None and cached[0] is model:
        return cached[1]

    traces = []
    common: set[int] | None = None
    for anticodon, seq in _CALIBRATION_REFS:
        _, align = cyk_trace(model, seq)
        col2pos = dict(align)
        traces.append((anticodon, seq, col2pos))
        cols = set(col2pos)
        common = cols if common is None else (common & cols)

    result = DEFAULT_ANTICODON_COLUMNS
    if common:
        ordered = sorted(common)
        for i in range(len(ordered) - 2):
            triple = (ordered[i], ordered[i + 1], ordered[i + 2])
            if all(
                "".join(seq[col2pos[c] - 1] for c in triple) == anticodon
                for anticodon, seq, col2pos in traces
            ):
                result = triple
                break

    _CALIBRATION_CACHE[key] = (model, result)
    return result


_AA3_TO_1 = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "fMet": "fM",
    "SeC": "U",
    "Sup": "*",
}


def gene_name(amino_acid: str, anticodon: str) -> str:
    """Format ``trnX(abc)`` from an amino-acid letter and an anticodon."""
    aa = _AA3_TO_1.get(amino_acid, amino_acid)
    letter = "fM" if aa == "fM" else aa[0].upper() if aa else "X"
    return f"trn{letter}({anticodon.lower().replace('u', 't')})"


def name_trna(
    model: CovarianceModel,
    mature_seq: str,
    *,
    organelle: str = "mitochondrion",
    anticodon_columns: tuple[int, int, int] | None = None,
) -> tuple[str, str, str] | None:
    """Return ``(anticodon, amino_acid_letter, gene_name)`` for a mature tRNA.

    ``None`` if the anticodon columns are not aligned (e.g. a poorly-scoring or
    atypical tRNA) or the anticodon has no known isotype. The anticodon columns
    are auto-calibrated for ``model`` unless supplied explicitly.
    """
    if anticodon_columns is None:
        anticodon_columns = _calibrate_anticodon_columns(model)
    seq = mature_seq.upper().replace("U", "T")
    _, align = cyk_trace(model, seq)
    if not align:
        return None
    col2pos = dict(align)
    if not all(c in col2pos for c in anticodon_columns):
        return None
    anticodon = "".join(seq[col2pos[c] - 1] for c in anticodon_columns)
    if len(anticodon) != 3 or set(anticodon) - set("ACGT"):
        return None
    anticodon = normalize_anticodon(anticodon)
    isotypes = isotypes_for_anticodon(anticodon, organelle)
    if not isotypes:
        return None
    aa = isotypes[0]
    return anticodon.upper().replace("U", "T"), aa, gene_name(aa, anticodon)
