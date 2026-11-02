"""Workbench runners for the mitogenome structure / transfer / comparative-model operations."""

from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleDependencyError
from organelleverse.operations.analysis_runners import ANALYSIS_TOOL_RUNNERS, _write_tsv

NEW_TOOLS = ("mito_graph_complexity", "mito_dispersed_repeats", "mito_intron_classes",
             "mito_mtpt_grading", "phylo_pglmm", "phylo_compare_trees")


def _run(tool: str, params: dict[str, object], out: Path) -> dict:
    return ANALYSIS_TOOL_RUNNERS[tool]({k: str(v) for k, v in params.items()}, output_dir=out, log=lambda _: None)


def test_every_new_tool_has_a_runner() -> None:
    assert set(NEW_TOOLS) <= set(ANALYSIS_TOOL_RUNNERS)


def test_nested_values_are_written_as_json(tmp_path) -> None:
    from organelleverse.core.frozen import freeze_json

    path = _write_tsv(tmp_path / "rows.tsv", [freeze_json({"tier": "ambiguous", "hit": {"identity": 99.5},
                                                             "positions": [1, 5]})])
    row = next(csv.DictReader(path.open(), delimiter="\t"))
    assert json.loads(row["hit"]) == {"identity": 99.5} and json.loads(row["positions"]) == [1, 5]
    assert row["tier"] == "ambiguous"


def test_graph_complexity_runner_keeps_the_mitochondrial_components(tmp_path) -> None:
    rng = random.Random(3)

    def dna(n: int) -> str:
        return "".join(rng.choice("ACGT") for _ in range(n))

    m1, m2, plastid = dna(3000), dna(2000), dna(4000)
    gfa = tmp_path / "raw.gfa"
    gfa.write_text("\n".join([f"S\tm1\t{m1}", f"S\tm2\t{m2}", f"S\tp1\t{plastid}",
                              "L\tm1\t+\tm2\t+\t0M", "L\tm2\t+\tm1\t+\t0M", "L\tp1\t+\tp1\t+\t0M"]) + "\n")
    mito = tmp_path / "mito.fa"
    mito.write_text(">c\n" + m1 + m2 + "\n")
    result = _run("mito_graph_complexity", {"input": gfa, "sequences": mito}, tmp_path / "out")
    assert result["summary_text"].startswith("complex graph: 2 segments")
    assert Path(result["output_files"]["components"]).is_file()


def _tree_and_traits(tmp_path: Path) -> tuple[Path, Path]:
    tree = tmp_path / "tree.nwk"
    tree.write_text("((a:1,b:1):1,((c:1,d:1):0.5,(e:1,f:1):0.5):0.5);\n")
    table = tmp_path / "traits.tsv"
    rows = ["species\tsize\tintrons", "a\t100\t10", "b\t120\t12", "c\t300\t20",
            "d\t320\t22", "e\t500\t30", "f\t520\t31"]
    table.write_text("\n".join(rows) + "\n")
    return tree, table


def test_pglmm_runner_parses_form_values(tmp_path) -> None:
    tree, table = _tree_and_traits(tmp_path)
    result = _run("phylo_pglmm", {"tree": tree, "table": table, "response": "size", "predictors": "introns",
                                  "standardize": "introns", "log_response": True, "iterations": 3000,
                                  "burnin": 500, "thin": 5}, tmp_path / "out")
    assert "PGLMM of size on 6 species" in result["summary_text"]
    rows = list(csv.DictReader(Path(result["output_files"]["fixed_effects"]).open(), delimiter="\t"))
    assert [row["term"] for row in rows] == ["(Intercept)", "introns"]
    with pytest.raises(ValueError):
        _run("phylo_pglmm", {"tree": tree, "table": table}, tmp_path / "out2")


def test_compare_trees_runner(tmp_path) -> None:
    tree, _ = _tree_and_traits(tmp_path)
    result = _run("phylo_compare_trees", {"tree_a": tree, "tree_b": tree, "prune_to_shared": False},
                  tmp_path / "out")
    assert "similarity 1.000" in result["summary_text"]


@pytest.mark.skipif(shutil.which("cmsearch") is not None, reason="checks the missing-Infernal message")
def test_intron_runner_names_the_missing_dependency(tmp_path) -> None:
    fasta = tmp_path / "m.fa"
    fasta.write_text(">m\nACGT\n")
    with pytest.raises(OrganelleDependencyError):
        _run("mito_intron_classes", {"input": fasta}, tmp_path / "out")
