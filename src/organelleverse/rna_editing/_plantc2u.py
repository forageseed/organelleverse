"""PlantC2U backend — Python port of the plastid C-to-U editing predictor.

A faithful Python reimplementation of the R/CNN pipeline from
Xu C. et al. (2024) "PlantC2U: Deep learning of cross-species sequence
landscapes predict plastid C-to-U RNA editing in plants",
*Journal of Experimental Botany* 75(8):2266.

The original tool is R (keras + tensorflow). This module ports the three
preprocessing functions (``Extract_seq``, ``EncodingSeq``,
``convStringToMatrix``) to NumPy and runs the same packaged pre-trained CNN
(``Choose_flank_90_ratio_1.hdf5``) via TensorFlow/Keras from Python.

Key porting notes
-----------------
* Window: 90 nt of flanking sequence on each side of a cytidine, with the
  central C removed → 180 nt input window.
* One-hot encoding: A=(1,0,0,0,0) C=(0,1,0,0,0) G=(0,0,1,0,0)
  T=(0,0,0,1,0) N=(0,0,0,0,1) — 5 channels, matches the trained input shape
  ``(None, 180, 5)``.
* Output: the model emits a 2-column softmax. **Column 1 (0-based) is the
  editing probability** — the R pipeline trains with ``to_categorical(label,
  2)`` where positive sites carry label 1, and the authors' shipped
  application reads ``classes[,2]`` (1-based). Verified on the authors' own
  held-out set (``Choose_flank_90_ratio_1.Rdata``): column 1 reproduces
  accuracy 0.9975; column 0 is P(not-edited) and silently inverts every
  prediction. (An earlier port note claimed column 0 was correct "because
  ~6% of cytidines pass p>=0.9" — that reading was itself the inverted
  minority, not a validation.)
* Genome-wide calibration caveat: the training negatives were sampled from
  k-mer clusters whose GC content (0.465) differs from the positives (0.329),
  so on an AT-rich plastid CDS the model calls most cytidines positive at
  p >= 0.5. For genome-wide scans use a high threshold (>= 0.99) and
  report the calibration honestly.

Self-contained except for the optional ``tensorflow`` dependency.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = [
    "PLANTC2U_FLANK",
    "PLANTC2U_WINDOW",
    "plantc2u_model_path",
    "change_n",
    "extract_window",
    "encode_matrix",
    "score_cytidines",
]

# Model hyper-parameters (fixed by the packaged checkpoint).
PLANTC2U_FLANK = 90
PLANTC2U_WINDOW = PLANTC2U_FLANK * 2  # 180 nt (central cytidine dropped)

_NT2IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def plantc2u_model_path() -> Path:
    """Absolute path to the packaged PlantC2U HDF5 checkpoint."""
    return Path(__file__).resolve().parent / "data" / "plantc2u_flank90.hdf5"


# ---------------------------------------------------------------------------
# preprocessing — faithful ports of the R functions
# ---------------------------------------------------------------------------


def change_n(seq: str) -> str:
    """Replace every non-ACGT base with ``N`` (uppercase)."""
    return "".join(c if c in "ACGT" else "N" for c in seq.upper())


def extract_window(
    sequence: str, position_1based: int, strand: int, flank: int = PLANTC2U_FLANK
) -> str:
    """Extract the 2*flank window around a cytidine, dropping the central C.

    ``position_1based`` is the 1-based index of the cytidine within
    ``sequence``. ``strand`` follows the PlantC2U convention: ``1`` for the
    plus strand, ``0`` for the minus strand (complement + reverse).
    Out-of-bounds positions are padded with ``N``.
    """
    pos = position_1based
    L = len(sequence)
    start = pos - flank  # may be < 1
    end = pos + flank
    left_pad = "N" * (1 - start) if start < 1 else ""
    right_pad = "N" * (end - L) if end > L else ""
    s = max(1, start)
    e = min(L, end)
    window = left_pad + sequence[s - 1 : e] + right_pad
    # drop the central cytidine (1-based index flank+1 within `window`)
    final = window[:flank] + window[flank + 1 :]
    if strand == 0:
        final = final.translate(_COMP)[::-1]
    return final


def encode_matrix(seqs: list[str], seq_len: int = PLANTC2U_WINDOW) -> np.ndarray:
    """One-hot encode windows into the ``(N, seq_len, 5)`` tensor the CNN expects."""
    x = np.zeros((len(seqs), seq_len, 5), dtype="float32")
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:seq_len]):
            x[i, j, _NT2IDX.get(c, 4)] = 1.0
    return x


# ---------------------------------------------------------------------------
# model loading + scoring
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_model():
    """Lazy-load the TensorFlow Keras CNN (cached for the process)."""
    import tensorflow as tf  # noqa: F401  (import side-effect: registers tf)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    return tf.keras.models.load_model(str(plantc2u_model_path()), compile=False)


def score_cytidines(windows: list[str]) -> np.ndarray:
    """Score a list of 180-nt windows; return editing probability for each.

    Returns a 1-D ``np.ndarray`` of length ``len(windows)`` with the editing
    probability (softmax column 1, as read by the authors' own R app;
    see module docstring).
    """
    if not windows:
        return np.zeros(0, dtype="float32")
    model = _load_model()
    x = encode_matrix(windows)
    preds = model.predict(x, verbose=0)
    # column 1 = P(edited): positive sites were label 1 under to_categorical
    return preds[:, 1]
