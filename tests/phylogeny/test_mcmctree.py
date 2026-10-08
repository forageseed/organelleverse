"""PAML contracts, real output mapping and independently calculated diagnostics."""

from __future__ import annotations

import io
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from Bio import Phylo

from organelleverse.core.errors import (
    OrganelleDependencyError,
    OrganelleExecutionError,
    OrganelleInputError,
)
from organelleverse.phylogeny import Calibration, MCMCTreeOptions, dating, mcmctree

FIXTURE = Path(__file__).resolve().parents[1] / "data/dating/mcmctree"


@pytest.fixture
def inputs(tmp_path):
    tree = tmp_path / "tree.nwk"
    tree.write_text("((A:0.1,B:0.2):0.1,(C:0.2,D:0.3):0.2);")
    aln = tmp_path / "aln.fa"
    aln.write_text(">A\nACGT\n>B\nACGA\n>C\nTCGA\n>D\nTCGT\n")
    return tree, aln


def test_soft_bounds_and_plan(inputs, tmp_path):
    rows = [
        Calibration(root=True, min_age_ma=90, max_age_ma=120),
        Calibration(mrca=["A", "B"], min_age_ma=70),
        Calibration(mrca=["C", "D"], max_age_ma=80),
    ]
    assert [mcmctree._bound(c) for c in rows] == [
        "B(0.90000000000000002,1.2)",
        "L(0.69999999999999996)",
        "U(0.80000000000000004)",
    ]
    result = dating.date_tree(
        *inputs,
        rows,
        output_dir=tmp_path / "out",
        backend="mcmctree",
        mcmctree_options=MCMCTreeOptions(clock=3),
        dry_run=True,
    )
    assert result.status == "warning"
    assert result.metrics["options"]["clock"] == 3
    assert result.metrics["time_unit_ma"] == 100
    assert result.metrics["chain_seeds"] == (42, 44)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "kwargs,rows,match",
    [
        ({"backend": "other"}, [Calibration(root=True, min_age_ma=1)], "backend"),
        ({}, [Calibration(root=True, min_age_ma=1)], "upper age bound"),
        ({}, [Calibration(root=True, min_age_ma=1, max_age_ma=1)], "fixed ages"),
        ({"partition_nexus": "scheme.nex"}, None, "Cannot read partition"),
        ({"model": "HKY"}, None, "GTR"),
        ({"mcmctree_options": {"clock": 1}}, None, "options"),
        ({"mcmctree_options": {"nsample": 3}}, None, "options"),
        ({"mcmctree_options": {"rgene_gamma": [2, 0]}}, None, "positive"),
        ({"seed": 0}, None, "seed"),
    ],
)
def test_invalid_inputs(inputs, tmp_path, kwargs, rows, match):
    parameters = dict(backend="mcmctree", dry_run=True)
    parameters.update(kwargs)
    with pytest.raises(OrganelleInputError, match=match):
        dating.date_tree(
            *inputs,
            rows or [Calibration(root=True, min_age_ma=90, max_age_ma=120)],
            output_dir=tmp_path / "out",
            **parameters,
        )
    assert not (tmp_path / "out").exists()


def test_missing_dependency_and_zero_exit_without_bv(inputs, tmp_path, monkeypatch):
    rows = [Calibration(root=True, min_age_ma=90, max_age_ma=120)]
    monkeypatch.setattr(mcmctree.shutil, "which", lambda *a, **k: None)
    with pytest.raises(OrganelleDependencyError, match="micromamba"):
        dating.date_tree(*inputs, rows, output_dir=tmp_path / "out", backend="mcmctree")
    monkeypatch.setattr(mcmctree.shutil, "which", lambda *a, **k: "/bin/paml")
    monkeypatch.setattr(
        mcmctree, "run_external", lambda *a, **k: SimpleNamespace(stdout="", stderr="")
    )
    with pytest.raises(OrganelleExecutionError, match=r"out\.BV"):
        dating.date_tree(*inputs, rows, output_dir=tmp_path / "out", backend="mcmctree")
    (tmp_path / "out" / "keep").write_text("user output")
    with pytest.raises(FileExistsError):
        dating.date_tree(*inputs, rows, output_dir=tmp_path / "out", backend="mcmctree")
    assert (tmp_path / "out" / "keep").read_text() == "user output"


def test_real_traces_match_paml_numbered_clades_and_means():
    tree = Phylo.read(io.StringIO((FIXTURE / "tree.tre").read_text().splitlines()[1]), "newick")
    result = mcmctree.summarize_chains(
        tree, [FIXTURE / "chain1.tsv", FIXTURE / "chain2.tsv"], 100, 2
    )
    assert len(result["node_ages"]) == 27
    assert not result["convergence"]["passed"]
    for i in (1, 2):
        text = (FIXTURE / f"chain{i}-summary.txt").read_text()
        numbered = Phylo.read(io.StringIO(text.splitlines()[1]), "newick")
        table = {
            m[0]: float(m[1]) * 100 for m in re.findall(r"^(t_n\d+)\s+([0-9.]+)\s+\(", text, re.M)
        }
        raw = np.loadtxt(FIXTURE / f"chain{i}.tsv", skiprows=1)
        columns = (FIXTURE / f"chain{i}.tsv").read_text().splitlines()[0].split()
        for node in numbered.get_nonterminals():
            name = f"t_n{int(node.confidence)}"
            clade = sorted(t.name.split("_", 1)[1] for t in node.get_terminals())
            row = next(r for r in result["node_ages"] if r["node_id"] == name)
            assert row["descendant_taxa"] == clade
            d = next(d for d in result["convergence"]["parameters"] if d["parameter"] == name)
            reference = (table[name] * 101 - raw[0, columns.index(name)] * 100) / 100
            assert abs(d["chain_means"][i - 1] - reference) <= 0.006
            assert 0 < row["hpd_lower_ma"] <= row["hpd_upper_ma"]
    dated = Phylo.read(io.StringIO(result["timetree_newick"]), "newick")
    for row in result["node_ages"]:
        node = dated.common_ancestor(row["descendant_taxa"])
        for tip in node.get_terminals():
            assert dated.distance(node, tip) == pytest.approx(row["age_ma"])


def test_ess_against_direct_autocovariance_and_known_ar1():
    rng = np.random.default_rng(2026)
    noise = rng.normal(size=100000)
    trace = np.empty(len(noise))
    trace[0] = noise[0]
    for i in range(1, len(trace)):
        trace[i] = 0.8 * trace[i - 1] + noise[i]
    ess = mcmctree.effective_sample_size(trace)
    assert ess / len(trace) == pytest.approx((1 - 0.8) / (1 + 0.8), rel=0.15)
    short = trace[:1000] - trace[:1000].mean()
    cov = [np.dot(short[: len(short) - lag], short[lag:]) / len(short) for lag in range(len(short))]
    total, previous = 0.0, float("inf")
    for i in range(0, len(cov) - 1, 2):
        pair = (cov[i] + cov[i + 1]) / cov[0]
        if pair <= 0:
            break
        previous = min(previous, pair)
        total += previous
    assert mcmctree.effective_sample_size(short) == pytest.approx(len(short) / (2 * total - 1))
    assert mcmctree.effective_sample_size(np.full(100, 0.1)) == 0


def test_diagnostics_detect_shifted_and_stuck_chains_and_hpd():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(2, 20000))
    assert mcmctree._split_rhat([a, b]) < 1.01
    assert mcmctree._split_rhat([a, b + 1]) > 1.01
    assert mcmctree._split_rhat([np.ones(100), np.ones(100)]) is None
    assert mcmctree._hpd(np.arange(100)) == (0, 94)


def test_truncated_trace_rejected(tmp_path):
    path = tmp_path / "trace.txt"
    path.write_text("Gen t_n5\n1 1.0\n")
    with pytest.raises(OrganelleExecutionError, match="incomplete"):
        mcmctree._read_trace(path, 100, 2)


@pytest.mark.parametrize("partitioned", [False, True])
def test_execution_contract_two_chains_warning_and_original_names(
    inputs, tmp_path, monkeypatch, partitioned
):
    monkeypatch.setattr(mcmctree.shutil, "which", lambda *a, **k: "/bin/paml")
    calls = []

    def execute(argv, *, cwd, **kwargs):
        ctl = (cwd / "run.ctl").read_text()
        calls.append((argv, ctl))
        assert "clock = 3" in ctl
        assert "GTR" not in ctl and "model = 7" in ctl
        assert "BDparas = 1 1 0.1 c" in ctl
        assert f"ndata = {2 if partitioned else 1}" in ctl
        assert "rgene_gamma = 2.0 20.0 1 0" in ctl
        if "usedata = 3" in ctl:
            (cwd / "out.BV").write_text("real likelihood placeholder supplied by test")
        else:
            assert (cwd / "in.BV").read_text() == "real likelihood placeholder supplied by test"
            (cwd / "mcmc.txt").write_text(
                "Gen t_n5 t_n6 t_n7 mu sigma2 lnL\n"
                + "".join(
                    f"{i} {1 + i / 10000} {0.5 + i / 10000} {0.6 + i / 10000} 0.1 0.2 -3\n"
                    for i in range(1, 101)
                )
            )
        return SimpleNamespace(stdout="MCMCTREE in paml version test\n", stderr="")

    monkeypatch.setattr(mcmctree, "run_external", execute)
    scheme = tmp_path / "scheme.nex"
    scheme.write_text("#NEXUS\nbegin sets; charset a = 1-2; charset b = 3-4; end;")
    result = dating.date_tree(
        *inputs,
        [Calibration(root=True, min_age_ma=90, max_age_ma=120)],
        output_dir=tmp_path / "out",
        backend="mcmctree",
        partition_nexus=scheme if partitioned else None,
        mcmctree_options=MCMCTreeOptions(clock=3, nsample=100, sampfreq=1),
    )
    assert result.status == "warning" and "mcmc_not_converged" in result.flags
    assert len(calls) == 3
    assert "seed = 42" in calls[1][1] and "seed = 44" in calls[2][1]
    assert set(result.metrics["taxa"]) == {"A", "B", "C", "D"}
    assert result.metrics["node_ages"][0]["age_ma"] == pytest.approx(100.505)
