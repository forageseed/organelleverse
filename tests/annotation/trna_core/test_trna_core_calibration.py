from __future__ import annotations

from organelleverse.annotation.trna_core.calibration import load_trna_model


def test_trna_model_json_loads_from_package_data_without_local_paths():
    params = load_trna_model("mitochondrion")

    assert params.version == "mitochondrion_cleanroom_v1"
    assert params.min_structure_score > 0
    forbidden = ["/" + "home/", "data" + "16t", "ji" + "azc"]
    assert not any(pattern in params.provenance for pattern in forbidden)
