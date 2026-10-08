"""Package-data loading for OrganelleVerse-authored tRNA model parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources


@dataclass(frozen=True)
class TRNAModelParameters:
    version: str
    organelle: str
    provenance: str
    min_structure_score: float
    min_total_score: float
    stem_weights: dict[str, float]
    critical_stems: tuple[str, ...]
    critical_stem_bonus: float
    anticodon_match_score: float
    anticodon_mismatch_penalty: float
    no_structure_penalty: float
    low_complexity_penalty: float


@cache
def load_trna_model(organelle: str = "mitochondrion") -> TRNAModelParameters:
    if organelle != "mitochondrion":
        raise ValueError(f"Unsupported tRNA model organelle: {organelle!r}")
    path = resources.files("organelleverse.annotation.data").joinpath(
        "mitochondrion/trna_native/model_v1.json"
    )
    payload = json.loads(path.read_text())
    return TRNAModelParameters(
        version=payload["version"],
        organelle=payload["organelle"],
        provenance=payload["provenance"],
        min_structure_score=float(payload["min_structure_score"]),
        min_total_score=float(payload["min_total_score"]),
        stem_weights={key: float(value) for key, value in payload["stem_weights"].items()},
        critical_stems=tuple(payload["critical_stems"]),
        critical_stem_bonus=float(payload["critical_stem_bonus"]),
        anticodon_match_score=float(payload["anticodon_match_score"]),
        anticodon_mismatch_penalty=float(payload["anticodon_mismatch_penalty"]),
        no_structure_penalty=float(payload["no_structure_penalty"]),
        low_complexity_penalty=float(payload["low_complexity_penalty"]),
    )
