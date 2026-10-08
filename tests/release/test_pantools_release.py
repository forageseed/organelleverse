"""Actual database build/export, independent reconstruction and managed reuse."""

import json
import random
import shutil

import pytest
from Bio.Seq import Seq

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.pangenome.graph import path_sequences
from organelleverse.pangenome.service import build_graph, require_graph_result

pytestmark = pytest.mark.integration


def test_real_pantools_preserves_repeated_reverse_and_multiple_molecule_paths(
    tmp_path, monkeypatch
):
    assert shutil.which("pantools"), "Actual PanTools 4.3.5 and Java 11 are required"
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    rng = random.Random(20260908)
    repeat = "".join(rng.choice("ACGT") for _ in range(700))
    gap = "".join(rng.choice("ACGT") for _ in range(500))
    first = repeat + gap + repeat + "N" * 30 + gap
    reverse = str(Seq(first).reverse_complement())
    genomes, expected = [], {}
    for sample, sequence in (("a", first), ("b", reverse)):
        path = tmp_path / f"{sample}.fa"
        path.write_text(f">main\n{sequence}\n>extra\n{gap}\n")
        genomes.append(
            OrganelleGenome(
                organelle="mitochondrion",
                sequence=ArtifactRef.from_path(path, kind="sequence", format="fasta"),
                metadata=OrganelleMetadata(accession=sample),
            )
        )
        expected.update({f"{sample}#1#1": sequence, f"{sample}#1#2": gap})
    result = build_graph(genomes, method="pantools", k=21, threads=2, pantools_memory_mb=2048)
    assert result.status == "ok", result
    assert path_sequences(require_graph_result(result, backend="pantools").resolve()) == expected
    assert "4.3.5" in result.provenance.software_versions["pantools"]
    assert result.metrics["path_sequence_verified"] is True
    evidence = next(
        a for a in result.artifacts if a.uri.endswith("pantools-export-validation.json")
    )
    assert json.loads(evidence.resolve().read_text())["path_count"] == 4
    cached = build_graph(genomes, method="pantools", k=21, threads=2, pantools_memory_mb=2048)
    assert result.object_id == cached.object_id


def test_real_pantools_complete_workflow_retains_overlap_paths_and_reports(tmp_path, monkeypatch):
    from organelleverse.pangenome import workflow, workflow_store
    from organelleverse.pangenome.pav_store import load_pav_metadata
    from organelleverse.pangenome.workflow_models import WorkflowRequest

    assert shutil.which("pantools"), "Actual PanTools 4.3.5 and Java 11 are required"
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    data = tmp_path / "dataset"
    data.mkdir()
    rng = random.Random(827)
    repeat = "".join(rng.choice("ACGT") for _ in range(800))
    spacer = "".join(rng.choice("ACGT") for _ in range(600))
    sequences = {
        "a": repeat + spacer + repeat,
        "b": str(Seq(repeat + spacer + repeat).reverse_complement()),
    }
    for sample, sequence in sequences.items():
        (data / f"{sample}.fa").write_text(f">mitochondrion\n{sequence}\n>minicircle\n{spacer}\n")
    created = workflow.create_run(
        WorkflowRequest(
            dataset_path=str(data),
            backend="pantools",
            k=21,
            pantools_memory_mb=2048,
            threads=2,
            generate_overview=False,
            formats=("svg",),
            bootstrap_replicates=10,
        )
    )
    finished = workflow.execute_run(created["run_id"])
    assert finished["status"] == "succeeded", finished
    directory = workflow_store.directory_for(created["run_id"])
    spelled = path_sequences(workflow.graph_path(created["run_id"]))
    assert spelled == {
        **{f"{sample}#1#1": seq for sample, seq in sequences.items()},
        **{f"{sample}#1#2": spacer for sample in sequences},
    }
    assert len(load_pav_metadata(directory).samples) == 2
    assert (directory / "tree.json").is_file()
    assert (directory / "report" / "node_frequency.tsv").is_file()
    assert workflow.execute_run(created["run_id"])["status"] == "succeeded"
