import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.selection.models import _prepare_tree


@pytest.mark.parametrize("use_map", [False, True])
def test_codeml_marks_exact_tips_only(tmp_path, use_map):
    tree = "((A:0.1,A2:0.2)A:0.3,(Zea:0.4,Zea_mays:0.5):0.6,B:0.7);"
    text = _prepare_tree(
        tree,
        None if use_map else ["A", "Zea"],
        tmp_path / "tree.nwk",
        label_map={"A": "#1", "Zea": "#2"} if use_map else None,
    )
    assert "A #1:0.1" in text
    assert f"Zea #{2 if use_map else 1}:0.4" in text
    assert "A2:0.2" in text and "Zea_mays:0.5" in text
    assert ")A:0.3" in text
    assert text.count("#") == 2


def test_codeml_missing_foreground_fails_at_boundary(tmp_path):
    with pytest.raises(OrganelleInputError, match="not found"):
        _prepare_tree("(A,A2,B);", ["absent"], tmp_path / "tree.nwk")


def test_codeml_preserves_existing_clade_marks(tmp_path):
    text = _prepare_tree("((A,A2) $2:0.3,Zea,B);", ["Zea"], tmp_path / "tree.nwk")
    assert ") $2:0.3" in text
    assert "Zea #1" in text
