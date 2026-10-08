"""Real PanTools database construction and verified property-to-GFA export."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from .._bio import read_fasta, write_fasta
from ..core.artifacts import ArtifactRef
from ..core.result import OrganelleResult
from ._contract import (
    OPERATION_VERSION,
    input_artifact_hashes,
    input_object_ids,
    make_provenance,
    result_scope,
    utc_now,
)
from ._pantools_export import export_to_gfa
from ._runner import run_command
from .project import PangenomeProject


def build_pantools(genomes, directory, *, executable, k, threads, memory_mb=4096):
    """Use the audited PanTools 4.3.5 export schema and retain raw evidence.

    Every separate molecule must be at least k bases: the released native
    exporter fails on shorter records. GFA retains the genuine k-1 overlaps;
    a consumer requiring a blunt graph must report that representation limit.
    """
    if not 6 <= k <= 255 or threads < 1 or memory_mb < 128:
        raise ValueError("PanTools requires k in [6,255], positive threads and heap >=128 MiB")
    started = utc_now()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = PangenomeProject.from_genomes(genomes).materialize(directory / "pantools-input")
    sequences = dict(read_fasta(Path(manifest.fasta_path)))
    short = [name for name, sequence in sequences.items() if len(sequence) < k]
    if short:
        raise ValueError(f"PanTools 4.3.5 cannot export molecules shorter than k={k}: {short}")
    samples = directory / "pantools-samples"
    samples.mkdir()
    sample_files = []
    for sample in manifest.samples:
        path = samples / f"{sample.sample}.fa"
        write_fasta(path, [(name, sequences[name]) for name in sample.headers])
        sample_files.append(path)
    listing = directory / "genomes.txt"
    listing.write_text("".join(str(path) + "\n" for path in sample_files))
    commands = []

    def execute(arguments, name):
        record = run_command([executable, *arguments], cwd=directory, stdout_path=directory / name)
        commands.append(asdict(record))
        (directory / (name + ".stderr")).write_text(record.stderr)
        (directory / "pantools-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if not record.ok:
            raise RuntimeError(
                f"PanTools {name} failed with exit {record.returncode}: {record.stderr[-2000:]}"
            )
        return record

    execute(["--version"], "version.txt")
    version = (directory / "version.txt").read_text().strip()
    if not re.search(r"\b4\.3\.5\b", version):
        raise ValueError(
            f"PanTools property export schema is verified for 4.3.5; found {version!r}"
        )
    database = directory / "pantools-db"
    build = execute(
        [
            "-Xms128m",
            f"-Xmx{memory_mb}m",
            "build_pangenome",
            f"--threads={threads}",
            f"--kmer-size={k}",
            "--cache-size=10000",
            "--num-buckets=4",
            "--num-db-writer-threads=1",
            str(database),
            str(listing),
        ],
        "build.stdout",
    )
    nodes, edges, anchors = (directory / name for name in ("nodes.csv", "edges.csv", "anchors.csv"))
    export = execute(
        [
            "-Xms128m",
            f"-Xmx{memory_mb}m",
            "export_pangenome",
            f"--node-properties-file={nodes}",
            f"--relationship-properties-file={edges}",
            f"--sequence-node-anchors-file={anchors}",
            str(database),
        ],
        "export.stdout",
    )
    graph = directory / "pangenome.gfa"
    validation = export_to_gfa(nodes, edges, sequences, graph)
    validation_path = directory / "pantools-export-validation.json"
    validation_path.write_text(json.dumps(validation, indent=2) + "\n")
    identities = directory / "pantools-sample-identities.json"
    identities.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample": sample.sample,
                        "molecules": list(sample.headers),
                        "input_file": str(path),
                    }
                    for sample, path in zip(manifest.samples, sample_files, strict=True)
                ]
            },
            indent=2,
        )
        + "\n"
    )
    parameters = {
        "method": "pantools",
        "k": k,
        "threads": threads,
        "pantools_memory_mb": memory_mb,
        "export_schema": "4.3.5",
    }
    provenance = make_provenance(
        operation_id="pangenome.build_graph",
        parameters=parameters,
        input_object_ids=input_object_ids(genomes),
        input_artifact_hashes=input_artifact_hashes(genomes),
        requested_backend="pantools",
        actual_backend="pantools",
        attempted_backends=("pantools",),
        argv=build.argv,
        started_at=started,
        finished_at=utc_now(),
    ).model_copy(update={"software_versions": {"pantools": version}})
    resources = [build.resource_usage, export.resource_usage]
    return OrganelleResult(
        operation_id="pangenome.build_graph",
        operation_version=OPERATION_VERSION,
        scope=result_scope(genomes),
        status="ok",
        flags=("graph_built",),
        summary_text=f"PanTools 4.3.5 graph: {validation['node_count']} nodes, {validation['path_count']} exactly reconstructed molecule paths.",
        metrics={
            "method": "pantools",
            "genome_count": len(genomes),
            "gfa_segments": validation["node_count"],
            "gfa_edges": validation["edge_count"],
            "molecule_paths": validation["path_count"],
            "path_sequence_verified": True,
            "commands": commands,
            "command_resources": resources,
            "representation": "GFA1 with exact k-1 overlaps",
            "resource_usage": {
                "wall_seconds": sum(r.wall_seconds or 0 for r in (build, export)),
                "cpu_seconds": sum(r.cpu_seconds for r in (build, export))
                if all(r.cpu_seconds is not None for r in (build, export))
                else None,
                "peak_memory_bytes": max(r.peak_memory_bytes for r in (build, export))
                if all(r.peak_memory_bytes is not None for r in (build, export))
                else None,
                "memory_scope": "maximum OS child RSS across sequential build/export commands; not aggregate process-tree memory",
            },
        },
        artifacts=tuple(
            ArtifactRef.from_path(
                path,
                kind="pangenome_graph" if path == graph else "pangenome_build_evidence",
                format=path.suffix.lstrip("."),
            )
            for path in [
                graph,
                nodes,
                edges,
                anchors,
                validation_path,
                identities,
                directory / "pantools-commands.json",
                directory / "version.txt",
                directory / "build.stdout",
                directory / "build.stdout.stderr",
                directory / "export.stdout",
                directory / "export.stdout.stderr",
            ]
        ),
        provenance=provenance,
    )
