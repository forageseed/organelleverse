import pytest

from organelleverse.selection import translate_cds


@pytest.mark.parametrize(
    ("strip_stop", "expected"), [(True, ("MA", "MX")), (False, ("MA*W", "MX*"))]
)
def test_translation_returns_named_protein_records(tmp_path, strip_stop, expected):
    cds = tmp_path / "cds.fa"
    cds.write_text(">first\natggcctgatgg\n>second\nAUGNNNUAA\n")
    result = translate_cds(cds, strip_stop=strip_stop)
    assert result.status == "ok"
    assert result.metrics["n_sequences"] == 2
    assert [(r["name"], r["seq"]) for r in result.metrics["records"]] == [
        ("first", expected[0]), ("second", expected[1])
    ]
    assert result.artifacts == ()
