"""Regression cases captured from the real 28-plastome LSD2 runs."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from Bio import Phylo

from organelleverse.phylogeny import Calibration, dating

FIXTURE = Path(__file__).resolve().parents[1] / "data/dating/degeneracy"


def calibrations(group):
    return [
        Calibration.model_validate(c)
        for c in json.loads((FIXTURE / f"{group}-calibrations.json").read_text())
    ]


@pytest.mark.parametrize("group", ["original", "internal"])
def test_real_zero_branches_and_boundaries(group):
    parsed = dating.parse_timetree(FIXTURE / f"{group}.nex", calibrations=calibrations(group))
    diagnostics = parsed["degeneracy"]
    # Independent direct inspection of raw LSD2 branches, not annotated dates.
    tree = Phylo.read(FIXTURE / f"{group}.nex", "nexus")
    zero = [n for n in tree.get_nonterminals() if n is not tree.root and n.branch_length == 0]
    assert len(zero) == len(diagnostics["zero_time_branches"]) == 10
    assert diagnostics["tolerance_ma"] == 0.0005
    assert {"lsd2_zero_time_branches", "lsd2_calibration_boundary"} <= set(
        parsed["dating_warnings"]
    )
    assert {(r["node_id"], r["boundary"]) for r in diagnostics["calibration_boundary_hits"]} >= {
        ("node_0", "max")
    }
    if group == "internal":
        grass = tree.common_ancestor(["Anomochloa_marantoidea", "Oryza_sativa_temperate_japonic"])
        clade = sorted(t.name for t in grass.get_terminals())
        affected = next(r for r in diagnostics["affected_nodes"] if r["descendant_taxa"] == clade)
        assert affected["age_ma"] == 65.0
        assert not affected["calibrated"]
        assert affected["node_id"] in diagnostics["collapsed_uncalibrated_node_ids"]
        assert affected["collapsed_with_calibrated_node_ids"]
        # This spans multiple zero edges, rather than only an immediate parent.
        bep = tree.common_ancestor(
            ["Oryza_sativa_temperate_japonic", "Zea_mays_subsp__parviglumis"]
        )
        assert len(tree.get_path(bep)) - len(tree.get_path(grass)) == 2


def test_no_degeneracy_and_boundary_only_and_near_zero(tmp_path):
    path = tmp_path / "tree.nex"
    path.write_text(
        "#NEXUS\nbegin trees; tree 1 = ((A[&date=0]:5,B[&date=0]:5)[&date=-5]:5,C[&date=0]:10)[&date=-10]; end;"
    )
    parsed = dating.parse_timetree(
        path, calibrations=[Calibration(root=True, min_age_ma=9, max_age_ma=11)]
    )
    assert parsed["dating_warnings"] == []
    assert parsed["degeneracy"]["affected_nodes"] == []
    bounded = dating.parse_timetree(path, calibrations=[Calibration(root=True, min_age_ma=10)])
    assert bounded["dating_warnings"] == ["lsd2_calibration_boundary"]
    path.write_text(
        path.read_text()
        .replace(":5", ":10")
        .replace("]:10,C", "]:0.00001,C")
        .replace("date=-5", "date=-10")
    )
    near = dating.parse_timetree(path, calibrations=[Calibration(root=True, min_age_ma=10)])
    assert near["degeneracy"]["collapsed_uncalibrated_node_ids"] == ["node_1"]
    assert near["degeneracy"]["zero_time_branches"][0]["length_ma"] == 0.00001
    with pytest.raises(dating.OrganelleExecutionError, match="missing calibration MRCA taxa"):
        dating.parse_timetree(path, calibrations=[Calibration(mrca=["A", "missing"], min_age_ma=1)])


@pytest.mark.parametrize("group", ["original", "internal"])
def test_result_status_warning_with_captured_real_output(group, tmp_path, monkeypatch):
    raw = (FIXTURE / f"{group}.nex").read_text()
    tree = Phylo.read(io.StringIO(raw), "nexus")
    input_tree = tmp_path / "input.nwk"
    Phylo.write(tree, input_tree, "newick")
    alignment = tmp_path / "aln.fa"
    alignment.write_text("".join(f">{n.name}\nACGT\n" for n in tree.get_terminals()))
    monkeypatch.setattr(dating, "resolve_iqtree", lambda _: "/bin/iqtree2")

    def execute(argv, *, cwd, **kwargs):
        (cwd / "dating.timetree.nex").write_text(raw)
        return SimpleNamespace(stdout="IQ-TREE version 2.4.0\n", stderr="")

    monkeypatch.setattr(dating, "run_external", execute)
    result = dating.date_tree(
        input_tree, alignment, calibrations(group), output_dir=tmp_path / "out"
    )
    assert result.status == "warning"
    assert "lsd2_zero_time_branches" in result.flags
    assert len(result.metrics["degeneracy"]["zero_time_branches"]) == 10
    if group == "internal":
        assert "lsd2_uncalibrated_node_collapsed" in result.flags
