"""Opt-in real sequence-phylogeny gate; requires the installed RAxML-NG tool."""

import json
import shutil

import numpy as np
import pytest

from organelleverse.pangenome.sequence_phylogeny import RaxmlOptions, run_raxml_ng

pytestmark = pytest.mark.integration

def test_actual_raxml_ng_search_and_bootstrap_produce_verified_tree(tmp_path):
    executable = shutil.which("raxml-ng")
    assert executable is not None, "Real phylogeny gate requires raxml-ng"
    rng = np.random.default_rng(291)
    ancestor = rng.choice(list("ACGT"), 800)
    records = []
    for index in range(4):
        sequence = ancestor.copy()
        positions = rng.choice(800, 60 + index * 15, replace=False)
        sequence[positions] = rng.choice(list("ACGT"), len(positions))
        records.append(("taxon" + str(index), "".join(sequence)))
    alignment = tmp_path / "aligned.fa"
    alignment.write_text("".join(f">{label}\n{sequence}\n" for label, sequence in records))
    result = run_raxml_ng(
        alignment,
        tmp_path / "tree",
        executable=executable,
        options=RaxmlOptions(
            model="GTR+G",
            bootstrap_replicates=10,
            seed=73,
            threads=1,
            parsimony_starts=1,
            random_starts=1,
        ),
    )
    assert result["method"] == "RAxML-NG maximum likelihood"
    assert result["parameters"]["bootstrap_replicates"] == 10
    assert result["alignment_sites"] == 800
    assert result["command"]["returncode"] == 0
    assert result["resource_usage"]["wall_seconds"] > 0
    assert len(result["paths"]) == 4
    assert "RAxML-NG" in result["tool_version"]
    (tmp_path / "real-raxml-evidence.json").write_text(json.dumps(result, indent=2))
