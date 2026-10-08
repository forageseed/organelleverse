from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.phylogeny import astral


@pytest.fixture(autouse=True)
def _managed_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))


def test_run_astral_executes_and_returns_species_tree_artifact(tmp_path: Path, monkeypatch):
    gene_trees = tmp_path / "genes.tre"
    gene_trees.write_text("((a,b),c);\n((a,c),b);\n")
    jar = tmp_path / "astral.jar"
    jar.touch()

    def run(argv, **kwargs):
        assert argv[-2:] == ["-t", "2"]
        Path(argv[argv.index("-o") + 1]).write_text("((a,b),c)[q1=0.8];\n")
        return SimpleNamespace(stdout="Final quartet score: 0.83\n")

    monkeypatch.setattr(astral, "run_external", run)
    result = astral.run_astral(gene_trees, astral_jar=jar)

    assert result.status == "ok"
    assert result.metrics["n_gene_trees"] == 2
    assert result.metrics["astral_stdout"] == "Final quartet score: 0.83\n"
    assert result.artifacts[0].format == "newick"


def test_run_astral_reports_missing_jar_without_running(tmp_path, monkeypatch):
    genes = tmp_path / "genes.tre"
    genes.write_text("((a,b),c);\n")
    monkeypatch.delenv("ASTRAL_JAR", raising=False)
    result = astral.run_astral(genes, astral_jar=tmp_path / "missing.jar")
    assert result.status == "failed"
    assert result.errors[0].code == "phylogeny.astral.jar_missing"


def test_run_astral_preserves_stderr_tail_on_tool_failure(tmp_path, monkeypatch):
    genes = tmp_path / "genes.tre"
    genes.write_text("((a,b),c);\n")
    jar = tmp_path / "astral.jar"
    jar.touch()

    def run(argv, **kwargs):
        raise OrganelleExecutionError(
            code="phylogeny.astral.execution_failed",
            message="ASTRAL exited with status 1",
            details={"stderr_tail": "first\ninvalid gene tree"},
        )

    monkeypatch.setattr(astral, "run_external", run)
    result = astral.run_astral(genes, astral_jar=jar)
    assert result.status == "failed"
    assert "invalid gene tree" in result.summary_text
    assert result.errors[0].details["stderr_tail"] == "first\ninvalid gene tree"


def test_run_astral_rejects_empty_gene_tree_input(tmp_path):
    genes = tmp_path / "empty.tre"
    genes.write_text("  \n")
    result = astral.run_astral(genes, astral_jar=tmp_path / "astral.jar")
    assert result.status == "failed"
    assert result.errors[0].code == "phylogeny.astral.input_empty"


def test_astral_capability_bundle_parses_and_exposes_managed_output():
    from organelleverse.capabilities.parser import parse_capability_bundle

    bundle_file = (
        Path(__file__).parents[2]
        / "src"
        / "organelleverse"
        / "capabilities"
        / "phylogeny-run-astral"
        / "capability.toml"
    )
    parsed = parse_capability_bundle(bundle_file)
    assert parsed.capability.id == "phylogeny.run_astral"
    assert [parameter.name for parameter in parsed.contract.binding.parameters] == ["gene_trees", "astral_jar", "java"]
