from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.assembly.backends import AssemblyProfile, PreparedBackendResources
from organelleverse.assembly.backends.base import AdapterContext
from organelleverse.assembly.backends.ptgaul import PtgaulAdapter
from organelleverse.assembly.contracts import (
    AssemblyInputPayload,
    AssemblyRequest,
    PtgaulParameters,
)
from organelleverse.assembly.environments import PreparedEnvironment, PreparedExecutable
from organelleverse.assembly.routing import AssemblyRoute
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleExecutionError, OrganelleInputError
from organelleverse.core.frozen import FrozenMap


def _artifact(path: Path, role: str, kind: str, fmt: str, media: str) -> ArtifactRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture")
    return ArtifactRef.from_path(path, kind=kind, format=fmt, media_type=media)


def _context(tmp_path: Path, *, organelle: str = "plastid") -> AdapterContext:
    reads = _artifact(tmp_path / "reads" / "sample.fastq.gz", "long", "long_read", "fastq.gz", "application/gzip")
    reference = _artifact(tmp_path / "refs" / "related.fa", "reference", "reference_genome", "fasta", "text/x-fasta")
    data = OrganelleData.model_validate({
        "modality": "sequencing_reads",
        "artifacts": {"long_reads": reads, "reference": reference},
        "payload": {
            "contract_version": "organelleverse.assembly-input.v1",
            "long_libraries": [{
                "technology": "ont",
                "quality_state": "raw",
                "reads_artifact": "long_reads",
            }],
            "auxiliary": {"reference_fasta_artifact": "reference"},
        },
    })
    payload = AssemblyInputPayload.model_validate(dict(data.payload))
    params = PtgaulParameters(genome_size=155000, coverage=40, minimum_read_length=2500)
    request = AssemblyRequest(
        data=data, organelle=organelle, method="ptgaul", backend_parameters=params, threads=5
    )
    prefix = tmp_path / "env"
    executable_names = (
        "ptGAUL.sh", "combine_gfa.py", "minimap2", "seqkit",
        "assembly-stats", "seqtk", "flye",
    )
    executables = []
    for name in executable_names:
        path = prefix / "bin" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)
        executables.append(PreparedExecutable(name=name, path=path))
    environment = PreparedEnvironment(
        backend_id="ptgaul",
        carrier="conda",
        platform="linux-64",
        digest="sha256:" + "c" * 64,
        prefix=prefix,
        executables=tuple(executables),
        version="ptGAUL 1.0.5",
    )
    return AdapterContext(
        request=request,
        payload=payload,
        route=AssemblyRoute(
            requested_method="ptgaul",
            selected_backend="ptgaul",
            profile=AssemblyProfile.ONT_RAW,
            rule_id="explicit.backend",
            compatible_candidates=("ptgaul",),
        ),
        environment=environment,
        profile=None,
        resources=PreparedBackendResources(),
        workspace=tmp_path / "workspace",
        input_artifacts=FrozenMap.from_items({
            "long_reads": reads,
            "reference": reference,
        }),
        effective_backend_parameters=FrozenMap.from_json(params.model_dump(mode="json")),
    )


def _output_paths(context: AdapterContext, number_edges: int) -> tuple[Path, Path]:
    output = context.workspace / "backend" / "ptgaul" / "output"
    result = output / "result_2500"
    flye = result / "flye_cpONT"
    final = result / "ptGAUL_final_assembly"
    flye.mkdir(parents=True, exist_ok=True)
    final.mkdir(parents=True, exist_ok=True)
    graph = flye / "assembly_graph.gfa"
    graph.write_text("H\tVN:Z:1.0\n" + "".join(f"S\te{i}\tACGT\n" for i in range(number_edges)))
    return graph, final


def _fasta(path: Path, name: str, sequence: str) -> Path:
    path.write_text(f">{name}\n{sequence}\n")
    return path


def test_ptgaul_command_stages_reference_and_uses_documented_options(tmp_path: Path) -> None:
    context = _context(tmp_path)
    command = PtgaulAdapter().build_command(context)
    assert command.stable_argv == (
        "ptGAUL.sh", "-r", "role://artifact/reference",
        "-l", "role://artifact/long_reads", "-t", "5", "-g", "155000",
        "-c", "40", "-f", "2500", "-o", "role://workspace/backend/ptgaul/output",
    )
    assert command.resolved_argv[0].endswith("ptGAUL.sh")
    assert (context.workspace / "input" / "sample.fastq.gz").read_text() == "fixture"
    assert (context.workspace / "input" / "reference.fasta").is_file()


def test_ptgaul_one_edge_collects_documented_final_fasta_and_graph(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _graph, final = _output_paths(context, 1)
    fasta = _fasta(final / "final_assembly.fasta", "plastome", "ACGT")
    raw = PtgaulAdapter().collect_outputs(context)
    assert raw.primary_sequence_role == "assembly_fasta"
    assert raw.primary_graph_role == "assembly_graph"
    assert raw.require("assembly_fasta").path == fasta
    normalized = PtgaulAdapter().normalize(context, raw, tmp_path / "normalized")
    assert normalized.alternate_sequence_roles == ()
    assert normalized.total_bases == 4
    assert normalized.primary_graph.path.read_text().startswith("H\tVN:Z:1.0\n")


def test_ptgaul_three_edges_preserves_both_candidates(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _graph, final = _output_paths(context, 3)
    path1 = _fasta(final / "path1.fasta", "path1", "AACCGG")
    path2 = _fasta(final / "path2.fasta", "path2", "TTGGCC")
    raw = PtgaulAdapter().collect_outputs(context)
    assert raw.primary_sequence_role == "path1_candidate"
    assert raw.require("path1_candidate").path == path1
    assert raw.require("path2_candidate").path == path2
    normalized = PtgaulAdapter().normalize(context, raw, tmp_path / "normalized")
    assert normalized.alternate_sequence_roles == ("alternate_fasta_2",)
    assert (tmp_path / "normalized" / "assembly.fasta").read_text().endswith("AACCGG\n")
    assert (tmp_path / "normalized" / "alternate_2.fasta").read_text().endswith("TTGGCC\n")


@pytest.mark.parametrize("edge_count", [0, 2, 4])
def test_ptgaul_rejects_unexpected_edge_counts(tmp_path: Path, edge_count: int) -> None:
    context = _context(tmp_path)
    _output_paths(context, edge_count)
    with pytest.raises(OrganelleExecutionError) as raised:
        PtgaulAdapter().collect_outputs(context)
    assert raised.value.code == "assembly.output_incomplete"
    assert raised.value.details["edge_count"] == edge_count


def test_ptgaul_rejects_mitochondrial_mode(tmp_path: Path) -> None:
    context = _context(tmp_path, organelle="mitochondrion")
    with pytest.raises(OrganelleInputError, match="plastid-only"):
        PtgaulAdapter().build_command(context)


def test_ptgaul_fails_when_one_of_three_edge_candidates_is_missing(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _graph, final = _output_paths(context, 3)
    _fasta(final / "path1.fasta", "path1", "AACCGG")
    with pytest.raises(OrganelleExecutionError, match="both documented path FASTAs"):
        PtgaulAdapter().collect_outputs(context)
