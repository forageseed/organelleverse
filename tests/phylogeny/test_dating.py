"""Dating contracts and failures, with captured real IQ-TREE/LSD2 output."""

import pytest

from organelleverse.core.errors import OrganelleDependencyError, OrganelleInputError
from organelleverse.phylogeny import Calibration, dating


@pytest.fixture
def inputs(tmp_path):
    tree = tmp_path / "tree.nwk"
    tree.write_text("((A:0.1,B:0.2):0.1,(C:0.2,D:0.3):0.2);\n")
    aln = tmp_path / "aln.fa"
    aln.write_text(">A\nACGT\n>B\nACGA\n>C\nTCGA\n>D\nTCGT\n")
    return tree, aln


def test_plan_age_direction_root_and_managed_semantics(inputs, tmp_path):
    result = dating.date_tree(
        *inputs,
        [
            Calibration(root=True, min_age_ma=90, max_age_ma=120),
            Calibration(mrca=["A", "B"], min_age_ma=70),
        ],
        output_dir=tmp_path / "out",
        dry_run=True,
    )
    argv = list(result.metrics["argv"])
    assert argv[argv.index("--date-root") + 1] == "b(-120,-90)"
    assert argv[argv.index("--date-tip") + 1] == "0"
    assert "--date" in argv
    assert result.status == "warning" and "node_ages" not in result.metrics
    assert not (tmp_path / "out").exists()
    assert dating._date_bound(Calibration(mrca=["A", "B"], min_age_ma=70)) == "u(-70)"
    assert dating._date_bound(Calibration(root=True, max_age_ma=100)) == "l(-100)"
    assert dating._date_bound(Calibration(root=True, min_age_ma=100, max_age_ma=100)) == "-100"


@pytest.mark.parametrize(
    "calibrations",
    [
        [],
        [{"mrca": ["A", "missing"], "min_age_ma": 1}],
        [{"mrca": ["A", "A"], "min_age_ma": 1}],
        [{"root": True, "min_age_ma": 10, "max_age_ma": 2}],
        [{"root": True, "min_age_ma": -1}],
        [{"root": True, "min_age_ma": float("nan")}],
        [{"root": True}],
        [{"root": True, "max_age_ma": 10}],
        [{"root": True, "mrca": ["A", "B"], "min_age_ma": 1}],
        [{"root": True, "min_age_ma": 1, "distribution": "lognormal"}],
        [{"root": True, "min_age_ma": 1, "distribution": "uniform"}],
        [{"root": True, "min_age_ma": 1}, {"mrca": ["A", "C"], "min_age_ma": 2}],
        [{"root": True, "min_age_ma": 1, "max_age_ma": 5}, {"mrca": ["A", "B"], "min_age_ma": 6}],
    ],
)
def test_invalid_calibrations_fail_at_boundary(inputs, tmp_path, calibrations):
    with pytest.raises(OrganelleInputError):
        dating.date_tree(*inputs, calibrations, output_dir=tmp_path / "out", dry_run=True)
    assert not (tmp_path / "out").exists()


def test_unrooted_requires_outgroup_and_keeps_monophyletic_split(inputs, tmp_path):
    tree, aln = inputs
    tree.write_text("(A:0.1,B:0.2,(C:0.2,D:0.3):0.2);")
    with pytest.raises(OrganelleInputError, match="explicit outgroup"):
        dating.date_tree(
            tree,
            aln,
            [Calibration(root=True, min_age_ma=1)],
            output_dir=tmp_path / "out",
            dry_run=True,
        )
    result = dating.date_tree(
        tree,
        aln,
        [Calibration(root=True, min_age_ma=1)],
        outgroup=["C", "D"],
        output_dir=tmp_path / "out",
        dry_run=True,
    )
    assert set(result.metrics["outgroup"]) in ({"C", "D"}, {"A", "B"})
    with pytest.raises(OrganelleInputError, match="monophyletic"):
        dating.date_tree(
            tree,
            aln,
            [Calibration(root=True, min_age_ma=1)],
            outgroup=["A", "C"],
            output_dir=tmp_path / "out",
            dry_run=True,
        )


def test_mismatched_alignment_and_missing_binary(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(dating, "resolve_iqtree", lambda _: None)
    with pytest.raises(OrganelleDependencyError, match="micromamba"):
        dating.date_tree(
            *inputs, [Calibration(root=True, min_age_ma=1)], output_dir=tmp_path / "out"
        )
    inputs[1].write_text(">A\nACGT\n>B\nACGT\n>C\nACGT\n>X\nACGT\n")
    with pytest.raises(OrganelleInputError, match="same unique"):
        dating.date_tree(
            *inputs,
            [Calibration(root=True, min_age_ma=1)],
            output_dir=tmp_path / "out",
            dry_run=True,
        )


def test_outgroup_can_span_arbitrary_newick_origin(inputs, tmp_path):
    tree, aln = inputs
    tree.write_text("(A:0.1,B:0.2,(C:0.2,D:0.3):0.2);")
    result = dating.date_tree(
        tree,
        aln,
        [
            Calibration(root=True, min_age_ma=100),
            Calibration(mrca=["A", "B"], min_age_ma=10, max_age_ma=50),
        ],
        outgroup=["A", "B"],
        output_dir=tmp_path / "out",
        dry_run=True,
    )
    assert set(result.metrics["outgroup"]) in ({"A", "B"}, {"C", "D"})
    assert (
        dating._date_bound(Calibration(root=True, min_age_ma=10, max_age_ma=50), date_file=True)
        == "-50:-10"
    )
    assert dating._date_bound(Calibration(root=True, min_age_ma=10), date_file=True) == "NA:-10"


def test_real_output_dates_intervals_units_and_independent_branch_sums():
    import io
    import json

    from Bio import Phylo

    from tests._paths import PROJECT_ROOT

    root = PROJECT_ROOT / "tests/data/dating"
    result = dating.parse_timetree(root / "poales.timetree.nex", require_ci=True)
    json.dumps(result, allow_nan=False)
    assert len(result["node_ages"]) == 27
    crown = next(
        r
        for r in result["node_ages"]
        if r["descendant_taxa"] == ["Anomochloa_marantoidea", "Streptochaeta_spicata"]
    )
    assert crown["age_ma"] == 33.9173
    assert crown["ci_lower_ma"] == 26.3251 and crown["ci_upper_ma"] == 40.0016
    tree = Phylo.read(io.StringIO(result["timetree_newick"]), "newick")
    for row in result["node_ages"]:
        node = tree.common_ancestor(row["descendant_taxa"])
        for leaf in node.get_terminals():
            # LSD2 prints six significant digits; rounding accumulates along a path.
            assert tree.distance(node, leaf) == pytest.approx(row["age_ma"], abs=0.001)
    converted = dating.parse_timetree(root / "poales.timetree.nwk", substitution_rate=0.000343478)
    for a, b in zip(result["node_ages"], converted["node_ages"], strict=True):
        assert a["descendant_taxa"] == b["descendant_taxa"]
        assert a["age_ma"] == pytest.approx(b["age_ma"], abs=0.001)
        assert b["ci_lower_ma"] is None and b["ci_upper_ma"] is None
    with pytest.raises(dating.OrganelleExecutionError, match="positive rate"):
        dating.parse_timetree(root / "poales.timetree.nwk")
    with pytest.raises(dating.OrganelleExecutionError, match="confidence intervals"):
        dating.parse_timetree(
            root / "poales.timetree.nwk", substitution_rate=0.000343478, require_ci=True
        )


def test_missing_dates_and_missing_ci_fail(tmp_path):
    p = tmp_path / "bad.nex"
    p.write_text('#NEXUS\nbegin trees; tree 1 = (A[&date=0]:10,B[&date=0]:10)[&date="-10"]; end;')
    assert dating.parse_timetree(p)["node_ages"][0]["age_ma"] == 10
    with pytest.raises(dating.OrganelleExecutionError, match="confidence intervals"):
        dating.parse_timetree(p, require_ci=True)
    p.write_text("#NEXUS\nbegin trees; tree 1 = (A:10,B:10); end;")
    with pytest.raises(dating.OrganelleExecutionError, match="date annotation"):
        dating.parse_timetree(p)


def test_zero_exit_without_dating_is_failure_and_preserves_existing_output(
    inputs, tmp_path, monkeypatch
):
    from subprocess import CompletedProcess

    monkeypatch.setattr(dating, "resolve_iqtree", lambda _: "iqtree3")
    monkeypatch.setattr(
        dating,
        "run_external",
        lambda *a, **kw: CompletedProcess(a, 0, "IQ-TREE version 3.0.1\n", ""),
    )
    output = tmp_path / "out"
    with pytest.raises(dating.OrganelleExecutionError, match="no time tree"):
        dating.date_tree(*inputs, [Calibration(root=True, min_age_ma=100)], output_dir=output)
    (output / "dating.timetree.nex").write_text("old output")
    with pytest.raises(FileExistsError):
        dating.date_tree(*inputs, [Calibration(root=True, min_age_ma=100)], output_dir=output)
    assert (output / "dating.timetree.nex").read_text() == "old output"


def test_no_ci_plan_and_calibration_precision(inputs, tmp_path):
    result = dating.date_tree(
        *inputs,
        [Calibration(root=True, min_age_ma=90.8908, max_age_ma=123.4522)],
        output_dir=tmp_path / "out",
        ci_replicates=0,
        dry_run=True,
    )
    assert "--date-ci" not in result.metrics["argv"]
    bound = dating._date_bound(Calibration(root=True, min_age_ma=90.8908, max_age_ma=123.4522))
    low, high = map(float, bound[2:-1].split(","))
    assert low == -123.4522 and high == -90.8908
