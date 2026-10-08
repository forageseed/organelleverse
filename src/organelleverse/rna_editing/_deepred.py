"""Deepred-Mt backend — self-contained Python port of the upstream package.

A faithful reimplementation of the mitochondrial C-to-U editing predictor
pipeline from Edera A. A., Small I., Milone D. H., Sanchez-Puerta M. V.
(2021) "Deepred-Mt: Deep representation learning for predicting C-to-U RNA
editing in plant mitochondria", *Computers in Biology and Medicine*.

The upstream tool ships as a pip package (``deepredmt``). This module ports the
two functions that the OrganelleVerse backend actually needs —
``extract_wins_from_fasta`` and ``predict_from_fasta`` — to plain NumPy/TensorFlow
and loads the packaged pre-trained SavedModel, so OrganelleVerse has **no
runtime dependency on the upstream ``deepredmt`` package**.

The pre-trained model checkpoint (``210520.tf``, a TensorFlow SavedModel) is
packaged under ``rna_editing/data/`` and redistributed under the upstream MIT
license (see ``LICENSE`` third-party notices).

Porting notes
-------------
* Window: 41 nt = 20 nt left flank + central cytidine (kept) + 20 nt right
  flank. Out-of-bounds positions are padded with ``N``. The central C is
  **retained** (unlike PlantC2U, which drops it).
* Encoding: ``_NT2ID = {A:0, C:1, G:2, T:3, U:3}``; ``N`` (and any unknown
  base) maps to a 4-channel all-zero one-hot. Annotated editing sites ``E``/``e``
  are treated as cytidines (``C``).
* The model is an autoencoder + classifier returning a 3-tuple
  ``(x_rec, y_pred, edext)``; ``y_pred`` (index 1) is the editing probability.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = [
    "DEEPRED_WINDOW",
    "deepredmt_model_path",
    "extract_windows",
    "score_cytidines",
]

# Hyper-parameters fixed by the packaged checkpoint.
DEEPRED_FLANK = 20
DEEPRED_WINDOW = DEEPRED_FLANK * 2 + 1  # 41 nt (central cytidine retained)

# Nucleotide → integer index (one-hot depth = 4).
_NT2ID = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def deepredmt_model_path() -> Path:
    """Absolute path to the packaged Deepred-Mt SavedModel directory."""
    return Path(__file__).resolve().parent / "data" / "deepredmt_210520.tf"


# ---------------------------------------------------------------------------
# preprocessing — port of deepredmt.data_handler.extract_wins_from_fasta
# ---------------------------------------------------------------------------


def _read_fasta(fin: str | Path) -> dict[str, str]:
    """Read a FASTA file into ``{name: sequence}`` (port of upstream)."""
    seqs: dict[str, str] = {}
    name = None
    for line in Path(fin).read_text().splitlines():
        line = line.rstrip()
        if line.startswith(">"):
            name = line[1:]
            seqs[name] = ""
        elif name is not None:
            seqs[name] += line
    return seqs


def extract_windows(fin: str | Path) -> dict[str, str]:
    """Extract 41-nt windows centred on every cytidine (port of upstream).

    Returns ``{key: window}`` where ``key = "{seqname}!{1-based-pos}"`` and
    ``window`` is the 41-character nucleotide window (left flank + central C +
    right flank, padded with ``N``).
    """
    flank = DEEPRED_FLANK
    seqs = _read_fasta(fin)
    wins: dict[str, str] = {}
    for seqname, seq in seqs.items():
        seq = seq.upper()
        for i, t in enumerate(seq):
            if t in ("C", "E"):  # cytidine or annotated editing site
                start = max(0, i - flank)
                lwin = seq[start:i]
                rwin = seq[i + 1 : i + 1 + flank]
                lpad = "N" * (flank - len(lwin))
                rpad = "N" * (flank - len(rwin))
                window = lpad + lwin + t + rwin + rpad
                key = f"{seqname}!{i + 1}"  # 1-based position
                wins[key] = window
    return wins


def _encode_windows(wins: dict[str, str]) -> np.ndarray:
    """One-hot encode windows into the ``(N, 41, 4)`` tensor the CNN expects.

    ``E``/``e`` are treated as ``C`` (editing site ≡ cytidine). Unknown bases
    (``N`` and any non-ACGT) become an all-zero 4-vector.
    """
    nt2id = dict(_NT2ID)
    nt2id["E"] = nt2id["C"]
    nt2id["e"] = nt2id["C"]
    n = len(wins)
    x = np.zeros((n, DEEPRED_WINDOW, 4), dtype="float32")
    for i, w in enumerate(wins.values()):
        for j, base in enumerate(w):
            idx = nt2id.get(base)
            if idx is not None:
                x[i, j, idx] = 1.0
    return x


# ---------------------------------------------------------------------------
# model loading + scoring
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_model():
    """Lazy-load the TensorFlow SavedModel (cached for the process)."""
    import tensorflow as tf  # noqa: F401

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    return tf.keras.models.load_model(str(deepredmt_model_path()), compile=False)


def score_cytidines(windows: dict[str, str]) -> tuple[list[str], np.ndarray]:
    """Score every window; return ``(keys, editing_probabilities)``.

    ``keys`` preserves the input order (``"{seqname}!{1-based-pos}"``) and the
    returned probabilities are aligned to it.
    """
    if not windows:
        return [], np.zeros(0, dtype="float32")
    model = _load_model()
    keys = list(windows.keys())
    x = _encode_windows(windows)
    outputs = model.predict(x, verbose=0)
    # the model returns (x_rec, y_pred, edext); y_pred (index 1) is P(edited)
    y_pred = outputs[1] if isinstance(outputs, (list, tuple)) else outputs
    return keys, np.asarray(y_pred).reshape(-1)
