"""Regression fixture is a real PanTools export from generated sequences."""

from pathlib import Path

import pytest

from organelleverse._bio import read_fasta
from organelleverse.pangenome._pantools_export import export_to_gfa
from organelleverse.pangenome.graph import path_sequences


def test_property_export_preserves_degenerate_copies_repeats_and_orientation(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "pantools435"
    expected = dict([*read_fasta(fixture / "a.fa"), *read_fasta(fixture / "b.fa")])
    output = tmp_path / "graph.gfa"
    result = export_to_gfa(fixture / "nodes.txt", fixture / "edges.txt", expected, output)
    assert result["node_count"] == 7 and result["path_count"] == 4
    assert path_sequences(output) == expected
    degenerate = [
        node for node in result["node_identity"].values() if node["label"] == "degenerate"
    ]
    assert len(degenerate) == 2
    assert degenerate[0]["sequence"] == degenerate[1]["sequence"]
    assert degenerate[0]["address"] != degenerate[1]["address"]
    wrong = dict(expected)
    name = next(iter(wrong))
    wrong[name] = wrong[name][::-1]
    with pytest.raises(ValueError, match="do not exactly spell"):
        export_to_gfa(fixture / "nodes.txt", fixture / "edges.txt", wrong, tmp_path / "bad.gfa")
