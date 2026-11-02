"""A plastome transfer that places nothing says why."""

from organelleverse.annotation.plastome.pipeline import _no_features_message


def test_no_hits_points_at_the_input_or_the_threshold():
    message = _no_features_message(
        "Ephedra_equisetina_chloroplast.gb",
        queries=80,
        hits=0,
        min_identity=70.0,
        qcoverage="(0.2, 1.0)",
        target_length=153395,
    )
    assert "Ephedra_equisetina_chloroplast.gb" in message
    assert "80 reference gene queries" in message and "153,395 bp" in message
    assert ">= 70% identity" in message
    assert "nothing matched" in message and "not a plastid genome" in message


def test_hits_without_placement_say_so():
    message = _no_features_message(
        "ref.gb", queries=80, hits=12, min_identity=70.0, qcoverage="(0.2, 1.0)", target_length=1000
    )
    assert "12 alignment hits" in message and "none could be placed" in message
    assert "nothing matched" not in message
