"""Verify the dated-tree capability through the real discovery/binding path."""

import importlib.metadata

import pytest

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)


@pytest.mark.parametrize("backend", ["iqtree_lsd2", "mcmctree"])
def test_dating_bundle_admission_schema_and_plan(tmp_path, monkeypatch, backend):
    home = tmp_path / "home"
    project = tmp_path / "project"
    (project / ".organelleverse" / "capabilities").mkdir(parents=True)
    home.mkdir()
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kwargs: ())
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(project)
    discovered = discover_capabilities()
    record = verify_capability(
        "phylogeny.date_tree",
        store=VerificationStore(home / "verifications"),
        environment=LocalVerificationEnvironment(discovered),
    )
    assert {"output_dir", "iqtree_bin", "mcmctree_bin", "executor"}.isdisjoint(
        record.parameter_schema["properties"]
    )
    assert "MCMCTreeOptions" in record.parameter_schema["$defs"]
    assert "Calibration" in record.parameter_schema["$defs"]
    admitted = discover_capabilities()
    assert admitted.describe("phylogeny.date_tree").status is CapabilityStatus.ADMITTED
    tree = project / "tree.nwk"
    tree.write_text("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")
    aln = project / "aln.fa"
    aln.write_text(">A\nACGT\n>B\nACGA\n>C\nTCGA\n>D\nTCGT\n")
    partitions = project / "partitions.nex"
    partitions.write_text("#NEXUS\nbegin sets; charset gene1 = 1-2; charset gene2 = 3-4; end;")
    result = (
        admitted.binding_source()
        .resolve("phylogeny.date_tree")
        .invoke(
            None,
            {
                "tree_newick": str(tree),
                "alignment_fasta": str(aln),
                "calibrations": [{"root": True, "min_age_ma": 100, "max_age_ma": 120}],
                "backend": backend,
                "partition_nexus": str(partitions),
                "mcmctree_options": {"clock": 3, "rgene_gamma": [2.0, 20.0]}
                if backend == "mcmctree"
                else None,
                "dry_run": True,
            },
        )
    )
    assert result.operation_id == "phylogeny.date_tree"
    assert result.status == "warning" and result.metrics["planned"]
    if backend == "mcmctree":
        assert result.metrics["n_partitions"] == 2
    assert not (tmp_path / "cache" / "runs").exists()
