"""The anticodon-column calibration cache must not hand one model's columns to another."""

from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.annotation.cmsearch import naming
from organelleverse.annotation.cmsearch.model import parse_cm

_DATA = Path(naming.__file__).resolve().parent.parent / "data" / "models" / "trna"
_MODELS = [p for p in (_DATA / "plant_mito_trna.cm", _DATA / "trna.cm") if p.exists()]


@pytest.mark.skipif(len(_MODELS) < 1, reason="no packaged tRNA CM")
def test_a_stale_entry_under_a_recycled_id_is_not_used():
    model = parse_cm(_MODELS[0])
    other = parse_cm(_MODELS[-1])
    naming._CALIBRATION_CACHE.pop(id(model), None)
    # what a freed model leaves behind when a new model is allocated at the same address
    naming._CALIBRATION_CACHE[id(model)] = (other, (1, 2, 3))

    columns = naming._calibrate_anticodon_columns(model)

    assert columns != (1, 2, 3)
    assert naming._CALIBRATION_CACHE[id(model)][0] is model


@pytest.mark.skipif(len(_MODELS) < 1, reason="no packaged tRNA CM")
def test_the_same_model_is_calibrated_once():
    model = parse_cm(_MODELS[0])
    naming._CALIBRATION_CACHE.pop(id(model), None)

    first = naming._calibrate_anticodon_columns(model)
    entry = naming._CALIBRATION_CACHE[id(model)]
    second = naming._calibrate_anticodon_columns(model)

    assert first == second
    assert naming._CALIBRATION_CACHE[id(model)] is entry
