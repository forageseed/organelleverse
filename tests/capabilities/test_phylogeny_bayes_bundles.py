"""Hand-written phylogeny bundles: partition scheme, MrBayes, tree comparison.

Unlike the generator-restored ``phylogeny-*`` bundles, these five manifests
are authored directly (like ``pangenome.recommend_parameters``). Each is
discovered from the real core channel, verified, admitted and invoked through
its real binding; tool-backed ones are invoked with ``dry_run=True`` so the
test never depends on IQ-TREE/MrBayes being installed.
"""

from __future__ import annotations

import importlib.metadata
import random
import shutil
from pathlib import Path

import pytest

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.result import OrganelleResult
from tests._paths import PROJECT_ROOT

_IDS = (
    "phylogeny.build_partitioned_supermatrix",
    "phylogeny.select_partition_scheme",
    "phylogeny.run_mrbayes",
    "phylogeny.compare_trees",
    "phylogeny.iqtree_to_mrbayes_model",
)


def _admit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    store = VerificationStore(home / "verifications")
    records = {}
    for capability_id in _IDS:
        assert discovered.describe(capability_id).status is CapabilityStatus.REJECTED
        records[capability_id] = verify_capability(
            capability_id, store=store, environment=LocalVerificationEnvironment(discovered)
        )
    admitted = discover_capabilities()
    for capability_id in _IDS:
        entry = admitted.describe(capability_id)
        assert entry.status is CapabilityStatus.ADMITTED, entry.diagnostic
        assert entry.origins[0].channel == "core"
    return admitted, records


def _alignment(tmp_path: Path) -> tuple[Path, Path]:
    rng = random.Random(5)
    aln = tmp_path / "aln.fa"
    aln.write_text(
        "".join(f">t{i}\n{''.join(rng.choice('ACGT') for _ in range(30))}\n" for i in range(5))
    )
    parts = tmp_path / "parts.nex"
    parts.write_text("#nexus\nbegin sets;\n charset a = 1-15;\n charset b = 16-30;\nend;\n")
    return aln, parts


def test_bundles_admit_and_expose_no_output_or_binary_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, records = _admit(tmp_path, monkeypatch)
    for capability_id, record in records.items():
        properties = set(record.parameter_schema["properties"])
        assert {"output_dir", "mb_bin", "iqtree_bin", "executor"}.isdisjoint(properties), (
            capability_id
        )
    assert {"ngen", "nruns", "nchains", "burninfrac", "asdsf_threshold", "min_ess"} <= set(
        records["phylogeny.run_mrbayes"].parameter_schema["properties"]
    )
    assert {"rcluster_fast", "rcluster_max", "compare_codon_positions"} <= set(
        records["phylogeny.select_partition_scheme"].parameter_schema["properties"]
    )


def test_model_mapping_and_tree_comparison_invoke_for_real(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted, _ = _admit(tmp_path, monkeypatch)
    source = admitted.binding_source()
    mapped = source.resolve("phylogeny.iqtree_to_mrbayes_model").invoke(
        None, {"model": "HKY+F+I+G4"}
    )
    assert isinstance(mapped, OrganelleResult) and mapped.status == "ok"
    assert mapped.metrics["mrbayes_model"]["nst"] == 2
    assert mapped.metrics["mrbayes_model"]["rates"] == "invgamma"

    tree_a = tmp_path / "a.nwk"
    tree_b = tmp_path / "b.nwk"
    tree_a.write_text("((A,B),(C,D),(E,F));\n")
    tree_b.write_text("((A,C),(B,D),(E,F));\n")
    compared = source.resolve("phylogeny.compare_trees").invoke(
        None, {"tree_a": str(tree_a), "tree_b": str(tree_b)}
    )
    assert compared.operation_id == "phylogeny.compare_trees"
    assert compared.metrics["rf"] == 4 and compared.metrics["max_rf"] == 6


def test_tool_bundles_plan_without_creating_managed_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted, _ = _admit(tmp_path, monkeypatch)
    source = admitted.binding_source()
    aln, parts = _alignment(tmp_path)
    scheme = source.resolve("phylogeny.select_partition_scheme").invoke(
        None, {"alignment_fasta": str(aln), "partition_nexus": str(parts), "dry_run": True}
    )
    assert scheme.operation_id == "phylogeny.select_partition_scheme"
    assert "MFP+MERGE" in list(scheme.metrics["argv"])
    bayes = source.resolve("phylogeny.run_mrbayes").invoke(
        None,
        {"alignment_fasta": str(aln), "scheme_nexus": str(parts), "dry_run": True, "ngen": 5000},
    )
    assert bayes.operation_id == "phylogeny.run_mrbayes"
    assert bayes.metrics["planned"] is True
    assert not (tmp_path / "cache" / "runs").exists()


def test_supermatrix_bundle_reads_genbank_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted, _ = _admit(tmp_path, monkeypatch)
    fixture = PROJECT_ROOT / "tests" / "data" / "cp.gbk"
    paths = []
    for index in range(4):
        path = tmp_path / f"g{index}.gbk"
        shutil.copy(fixture, path)
        paths.append(str(path))
    result = (
        admitted.binding_source()
        .resolve("phylogeny.build_partitioned_supermatrix")
        .invoke(
            None,
            {
                "genbank_paths": paths,
                "taxon_names": ["a", "b", "c", "d"],
                "min_codons": 1,
                "codon_positions": False,
            },
        )
    )
    assert isinstance(result, OrganelleResult)
    assert result.operation_id == "phylogeny.build_partitioned_supermatrix"
    assert result.status == "ok", result.summary_text
    assert result.metrics["n_taxa"] == 4 and result.metrics["n_genes"] >= 1
