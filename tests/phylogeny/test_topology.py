from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from organelleverse.phylogeny import topology


def _prefix(argv) -> str:
    flag = "--prefix" if "--prefix" in argv else "-pre"
    return argv[argv.index(flag) + 1]


@pytest.fixture(autouse=True)
def _managed_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    alignment = tmp_path / "alignment.fa"
    alignment.write_text(
        ">a\nACGTACGTACGTACGTACGT\n"
        ">b\nACGTACGTACGTACGTACGT\n"
        ">c\nTGCATGCATGCATGCATGCA\n"
        ">d\nTGCATGCATGCATGCATGCA\n"
    )
    trees = tmp_path / "candidate.tre"
    trees.write_text("(a,b,(c,d));\n(a,c,(b,d));\n")
    return alignment, trees


def test_topology_test_runs_iqtree_and_returns_user_tree_results(tmp_path, monkeypatch):
    alignment, trees = _inputs(tmp_path)
    monkeypatch.setattr(topology.shutil, "which", lambda name: "/usr/bin/iqtree3" if name == "iqtree3" else None)
    seen: list[str] = []

    def fake_run(argv, **kwargs):
        seen.extend(argv)
        report = Path(f"{_prefix(argv)}.iqtree")
        report.write_text(
            "USER TREES\n"
            "Tree logL deltaL bp-RELL p-KH p-SH c-ELW p-AU\n"
            "1 -100.0 0.0 0.7 + 0.8 + 0.9 + 0.6 + 0.95 +\n"
            "2 -102.0 2.0 0.3 - 0.2 - 0.1 - 0.4 - 0.05 -\n\n"
        )

    monkeypatch.setattr(topology, "run_external", fake_run)
    result = topology.test_topologies(alignment, trees, model="GTR+G", seed=17, threads=2)

    assert result.status == "ok"
    assert result.metrics["tested_tree_count"] == 2
    assert result.metrics["results"][1]["p-AU"] == 0.05
    assert seen[seen.index("--test") + 1] == "10000"
    assert "--test-au" in seen and "-n" in seen and seen[seen.index("-n") + 1] == "0"
    assert seen[seen.index("-m") + 1] == "GTR+G"
    assert result.artifacts[0].uri.endswith("topology_test.iqtree")


def test_topology_test_rejects_fewer_than_iqtree_minimum_replicates(tmp_path, monkeypatch):
    alignment, trees = _inputs(tmp_path)
    monkeypatch.setattr(topology.shutil, "which", lambda name: "/usr/bin/iqtree3" if name == "iqtree3" else None)
    result = topology.test_topologies(alignment, trees, replicates=999)
    assert result.status == "failed"
    assert result.errors[0].code == "phylogeny.test_topologies.invalid_parameters"
    assert not (tmp_path / "home").exists()  # rejected before any run dir


def test_topology_test_fails_if_report_has_no_user_tree_table(tmp_path, monkeypatch):
    alignment, trees = _inputs(tmp_path)
    monkeypatch.setattr(topology.shutil, "which", lambda name: "/usr/bin/iqtree3" if name == "iqtree3" else None)

    def fake_run(argv, **kwargs):
        Path(f"{_prefix(argv)}.iqtree").write_text("IQ-TREE report without tests\n")

    monkeypatch.setattr(topology, "run_external", fake_run)
    result = topology.test_topologies(alignment, trees)
    assert result.status == "failed"
    assert result.errors[0].code == "phylogeny.test_topologies.report_invalid"


@pytest.mark.skipif(shutil.which("iqtree3") is None, reason="IQ-TREE 3 is not installed")
def test_topology_test_with_real_iqtree3(tmp_path):
    alignment, trees = _inputs(tmp_path)
    result = topology.test_topologies(alignment, trees, threads=2)
    assert result.status == "ok"
    assert result.metrics["candidate_tree_count"] == 2
    assert result.metrics["tested_tree_count"] == 2
    assert result.metrics["results"][0]["logL"] > result.metrics["results"][1]["logL"]
    assert result.metrics["results"][0]["p-AU"] > result.metrics["results"][1]["p-AU"]
