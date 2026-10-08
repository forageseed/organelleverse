"""Resumable pangenome analysis on real GFA artifacts in the L6 run store."""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import tempfile
import time
from pathlib import Path

from ..core.artifacts import ArtifactRef
from ..core.result import OrganelleResult
from ..plugin_experiments.store import _lock_file_descriptor, _unlock_file_descriptor
from ..runtime import managed_run_path, publish_staged_result
from . import workflow_store as store
from ._contract import make_provenance, parameters_hash, utc_now
from .stage_worker import StageFailure
from .stage_worker import execute_stage as _execute_stage
from .workflow_inputs import load_genomes, prepare_inputs, validate_project_paths
from .workflow_models import WorkflowRequest


def create_run(request: WorkflowRequest) -> dict:
    return store.public_view(store.create(request))


def execute_run(run_id: str) -> dict:
    """Run or resume; verify all completed checkpoints before reusing any output."""
    directory = store.directory_for(run_id)
    lock_path = directory.parent / f".{run_id}.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        _lock_file_descriptor(descriptor)
        return _execute_locked(run_id)
    finally:
        _unlock_file_descriptor(descriptor)
        os.close(descriptor)


def _execute_locked(run_id: str) -> dict:
    directory = store.directory_for(run_id)
    record = store.load(run_id)
    if record["status"] == "succeeded":
        try:
            for stage in record["stages"]:
                store.verify(directory, stage["outputs"])
            result = _read(directory / "result.json")
            for artifact in result["artifacts"]:
                _verified_artifact(artifact)
        except (ValueError, OSError) as error:
            # Published artifact bytes, including workflow.json, are immutable.
            record.update(status="failed", error=str(error))
        return store.public_view(record)
    if record.get("workflow_version") != store.WORKFLOW_VERSION:
        record.update(
            status="failed",
            error=(
                "This unfinished workflow uses an older checkpoint format. "
                "Start a new analysis to create columnar PAV and isolated stage records."
            ),
        )
        store.save(directory, record)
        return store.public_view(record)
    request = WorkflowRequest.model_validate(record["request"])
    request_hash = parameters_hash(request.model_dump(mode="json"))
    record.update(status="running", error=None)
    store.save(directory, record)
    active = None
    try:
        for stage in record["stages"]:
            if stage["status"] in {"completed", "skipped"}:
                if stage["workflow_parameters_hash"] != request_hash:
                    raise ValueError(
                        "Workflow parameters changed after a completed checkpoint; start a new analysis"
                    )
                store.verify(directory, stage["outputs"])
        for stage in record["stages"]:
            if stage["status"] in {"completed", "skipped"}:
                continue
            if _consume_pause(directory, record):
                return store.public_view(record)
            active = stage
            record["stage"] = stage["name"]
            stage.update(status="running", error=None, workflow_parameters_hash=request_hash)
            record["logs"].append(f"{utc_now().isoformat()} {stage['name']} started")
            store.save(directory, record)
            started = time.perf_counter()
            thread_started = time.thread_time()
            resources = None
            try:
                outcome = _execute_stage(stage["name"], directory, request)
                outputs, note = outcome.outputs, outcome.note
                resources = outcome.resources
            except StageFailure as error:
                resources = error.resources
                raise
            finally:
                stage["wall_seconds"] = round(time.perf_counter() - started, 6)
                stage["python_thread_cpu_seconds"] = round(time.thread_time() - thread_started, 6)
                stage["cpu_scope"] = (
                    "current Python worker thread; excludes subprocesses and other threads"
                )
                stage["peak_memory_bytes"] = None
                stage["memory_measurement"] = "unavailable for Python stage"
                if resources is not None:
                    stage.update(resources)
                    stage["memory_measurement"] = resources["measurement"]
            if stage["name"] == "graph" and (directory / "build.json").exists():
                measured = (
                    _read(directory / "build.json")["result"]
                    .get("metrics", {})
                    .get("resource_usage")
                )
                if measured is not None:
                    stage["external_command_resources"] = [measured]
            stage.update(
                status="skipped" if note else "completed",
                note=note,
                outputs=store.capture(directory, outputs),
            )
            if stage["name"] == "statistics":
                stats = _read(directory / "statistics.json")
                pav = (
                    _read(directory / "pav-summary.json")
                    if (directory / "pav-summary.json").exists()
                    else {}
                )
                record["summary"] = {
                    "segments": stats["nodes"],
                    "links": stats["edges"],
                    "paths": stats["paths"],
                    "samples": pav.get("sample_count", 0),
                    "total_bp": stats["total_bp"],
                    "core_threshold": request.core_threshold,
                    "cloud_threshold": request.cloud_threshold,
                }
            record["logs"].append(
                f"{utc_now().isoformat()} {stage['name']} {stage['status']}"
                + (f": {note}" if note else "")
            )
            store.save(directory, record)
        if _consume_pause(directory, record):
            return store.public_view(record)
        record.update(status="succeeded", stage="complete")
        record["files"] = _public_files(directory)
        store.save(directory, record)
        _publish(directory, record, request)
    except Exception as error:
        record.update(status="failed", error=str(error))
        if active:
            active.update(status="failed", error=str(error))
        record["logs"].append(f"{utc_now().isoformat()} failed: {error}")
        if directory.exists():
            store.save(directory, record)
    return store.public_view(record)


def _consume_pause(directory: Path, record: dict) -> bool:
    marker = store.pause_path(record["run_id"])
    if not marker.exists():
        return False
    marker.unlink(missing_ok=True)
    record.update(status="paused", error=None)
    record["logs"].append(f"{utc_now().isoformat()} paused at stage boundary")
    store.save(directory, record)
    return True


def _execute_stage_body(
    name: str, directory: Path, request: WorkflowRequest
) -> tuple[list[Path], str | None]:
    if name == "prepare":
        return prepare_inputs(request, directory), None
    if name == "repeat":
        if not request.recommend_parameters:
            return [], "Parameter recommendation and repeat analysis were not requested."
        from .tuning_service import recommend_parameters

        result = recommend_parameters(
            load_genomes(directory, request.organelle),
            threads=request.threads,
            run_repeatmasker=request.run_repeatmasker,
            species=request.repeatmasker_species if request.run_repeatmasker else "",
        )
        if result.status not in {"ok", "warning"}:
            raise ValueError("Parameter recommendation did not produce a successful result")
        references = [
            _verified_artifact(artifact.model_dump(mode="json")) for artifact in result.artifacts
        ]
        evidence = _write(
            directory / "recommendation.json",
            {
                "recommendation_result": result.model_dump(mode="json"),
                "source": "managed pangenome.recommend_parameters; Mash/RepeatMasker baseline prior with Poales plastid segment floor",
                "adopted": False,
                "adoption_policy": (
                    "adopt on successful graph construction"
                    if request.auto_adopt_recommendation
                    else "recommendation only; graph uses explicitly requested parameters"
                ),
                "graph_parameters": {
                    "identity": request.identity,
                    "segment_length": request.segment_length,
                },
                "repeatmasker_requested": request.run_repeatmasker,
            },
        )
        return [evidence, *references], None
    if name == "graph":
        from .graph import load_gfa

        if request.backend != "existing":
            from .service import build_graph, require_graph_result

            recommendation = None
            if request.auto_adopt_recommendation:
                recommendation = _read(directory / "recommendation.json")["recommendation_result"][
                    "metrics"
                ]["recommendation"]
            result = build_graph(
                load_genomes(directory, request.organelle),
                method=request.backend,
                k=request.k,
                pantools_memory_mb=request.pantools_memory_mb,
                threads=request.threads,
                segment_length=request.segment_length,
                identity=request.identity,
                recommendation=recommendation,
                auto_adopt_recommendation=request.auto_adopt_recommendation,
            )
            graph = require_graph_result(result, backend=request.backend)
            if request.auto_adopt_recommendation:
                evidence = _read(directory / "recommendation.json")
                evidence["adopted"] = True
                evidence["graph_parameters"] = {
                    key: recommendation[key] for key in ("identity", "segment_length")
                }
                _write(directory / "parameter-adoption.json", evidence)
            _write(
                directory / "build.json",
                {
                    "result": result.model_dump(mode="json"),
                    "graph_ref": graph.model_dump(mode="json"),
                },
            )
        source_graph = _graph_path(directory)
        load_gfa(source_graph)
        reference = _write(
            directory / "graph-reference.json",
            {
                "graph_ref": ArtifactRef.from_path(
                    source_graph, kind="pangenome_graph", format="gfa"
                ).model_dump(mode="json"),
            },
        )
        return [
            source_graph,
            reference,
            *([directory / "build.json"] if (directory / "build.json").exists() else []),
            *([directory / "parameter-adoption.json"] if request.auto_adopt_recommendation else []),
        ], None
    if name == "validate":
        from .graph import load_gfa

        source_graph = _graph_path(directory)
        graph = load_gfa(source_graph)
        validation = {
            "status": "passed",
            "format": "GFA1",
            "checks": ["managed artifact unchanged", "S/L/P/W references and path transitions"],
            "nodes": len(graph.segments),
            "links": len(graph.links),
            "paths": len(graph.paths),
            "sequence_consensus": "not inferred; exact sequence export validated separately",
        }
        if request.backend in {"pggb", "pantools"}:
            validation["input_path_reconstruction"] = validate_project_paths(
                source_graph, _read(directory / "inputs.json")
            )
        else:
            validation["input_path_reconstruction"] = {
                "status": "not_applicable",
                "reason": "Imported graph has no source molecule dataset"
                if request.backend == "existing"
                else "minigraph construction does not record every input as a complete embedded path",
            }
        return [_write(directory / "validation.json", validation)], None
    if name == "statistics":
        from .graph import graph_statistics, path_sequences
        from .pav_store import pav_summary, write_node_pav
        from .superbubbles import superbubbles

        stats = graph_statistics(_graph_path(directory))
        outputs = [
            _write(directory / "statistics.json", stats),
            _write(directory / "superbubbles.json", superbubbles(_graph_path(directory))),
        ]
        if stats["paths"]:
            metadata = write_node_pav(
                _graph_path(directory),
                directory,
                cloud_threshold=request.cloud_threshold,
                core_threshold=request.core_threshold,
            )
            outputs.extend(
                [
                    directory / "pav-metadata.json",
                    *[ref.resolve(directory) for ref in metadata.tables.values()],
                    *[ref.resolve(directory) for ref in metadata.tsvs.values()],
                    _write(directory / "pav-summary.json", pav_summary(directory)),
                ]
            )
            if len(metadata.samples) >= 2:
                from .similarity import graph_shared_windows

                try:
                    shared = graph_shared_windows(
                        _graph_path(directory), bin_size=request.window_bp
                    )
                except ValueError as error:
                    shared = {"status": "unavailable", "reason": str(error)}
                outputs.append(_write(directory / "shared-windows.json", shared))

            # Sequence reconstruction is an optional, explicitly diagnosed export.
            try:
                sequences = path_sequences(_graph_path(directory))
            except ValueError as error:
                outputs.append(
                    _write(
                        directory / "sequence-export.json",
                        {"status": "unavailable", "reason": str(error)},
                    )
                )
            else:
                fasta = directory / "paths.fasta"
                fasta.write_text("".join(f">{name}\n{seq}\n" for name, seq in sequences.items()))
                reference = ArtifactRef.from_path(fasta, kind="sequence", format="fasta")
                snapshots = _read(directory / "inputs.json")["snapshots"]
                identical = next(
                    (
                        item
                        for item in snapshots
                        if item["sha256"] == reference.sha256
                        and item["size_bytes"] == reference.size_bytes
                    ),
                    None,
                )
                evidence = {"status": "completed", "paths": len(sequences)}
                if identical is not None:
                    # Exact reconstruction can equal an input alignment. Retain
                    # its captured reference instead of claiming newly made bytes.
                    reused = _verified_artifact(identical)
                    fasta.unlink()
                    evidence["source_ref"] = identical
                    outputs.append(reused)
                else:
                    outputs.append(fasta)
                outputs.append(_write(directory / "sequence-export.json", evidence))

        return outputs, None
    if name == "overview":
        from ..core.errors import OrganelleInputError
        from .overview import render_odgi_overview

        if not request.generate_overview:
            return [], "Global graph overview was disabled in the submitted parameters."
        if not shutil.which("odgi"):
            return [], "ODGI is not installed; global graph overview is unavailable."
        # Fresh temporary outputs make a failed stage safely repeatable. Commands
        # retain their historical paths; artifact references are output-relative.
        with tempfile.TemporaryDirectory(prefix="organelleverse-overview-") as work:
            try:
                created = render_odgi_overview(_graph_path(directory), work, request.threads)
            except OrganelleInputError as error:
                if error.code not in {
                    "pangenome.overview_requires_p_paths",
                    "pangenome.overview_overlaps",
                    "pangenome.conversion_missing_sequence",
                }:
                    raise
                return [], str(error)
            target = directory / "overview"
            target.mkdir(exist_ok=True)
            outputs = []
            for source in created:
                destination = target / source.name
                shutil.move(str(source), destination)
                outputs.append(destination)
        return outputs, None
    if name == "phylogeny":
        from .workflow_science import phylogeny_stage

        return phylogeny_stage(_graph_path(directory), directory, request)
    if name == "annotation":
        from .workflow_science import annotation_stage

        return annotation_stage(_graph_path(directory), directory, request)
    if name == "report":
        from .benchmark_report import render_stage_benchmark
        from .stored_report import write_stored_report
        from .visualization import render_graph_summary

        outputs = []
        if (directory / "pav-metadata.json").exists():
            outputs.extend(
                write_stored_report(
                    directory,
                    directory / "report",
                    _optional(directory / "tree.json"),
                    _optional(directory / "annotation.json"),
                    formats=request.formats,
                )
            )
        outputs.extend(
            render_graph_summary(
                _read(directory / "statistics.json"),
                directory / "report",
                formats=request.formats,
            )
        )
        shared = _optional(directory / "shared-windows.json")
        if shared is not None and shared.get("status") != "unavailable":
            import csv

            from .similarity import render_shared_window_batches

            report_directory = directory / "report"
            with (report_directory / "figure_sources.tsv").open("a", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                for panels in render_shared_window_batches(
                    shared, report_directory, formats=request.formats, workers=request.threads
                ):
                    outputs.extend(Path(path) for path in panels.values())
                    table = Path(panels["table"]).relative_to(report_directory).as_posix()
                    for key, path in panels.items():
                        if key != "table":
                            writer.writerow(
                                [Path(path).relative_to(report_directory).as_posix(), table]
                            )
        outputs.extend(
            render_stage_benchmark(
                _read(directory / "workflow.json"), directory / "report", formats=request.formats
            )
        )
        outputs = list(dict.fromkeys(outputs))
        outputs.append(
            _write(
                directory / "analysis-notes.json",
                {
                    "frequency_unit": "sample",
                    "cloud_threshold": request.cloud_threshold,
                    "core_threshold": request.core_threshold,
                    "tree_interpretation": (
                        "RAxML-NG substitution-model maximum likelihood from supplied alignment"
                        if request.tree_mode == "msa"
                        else "Not requested"
                        if request.tree_mode == "none"
                        else "UPGMA similarity of graph-node presence; not a substitution-model phylogeny"
                    ),
                    "node_segmentation": "Node counts and bootstrap resampling units depend on graph segmentation.",
                    "sequence_policy": "Separate molecules retained; no artificial concatenation or circularization.",
                    "parameter_policy": (
                        "Verified recommendation adopted; plastid PGGB minimum 5000 bp."
                        if request.auto_adopt_recommendation
                        else "Requested graph parameters retained with plastid PGGB minimum 5000 bp; recommendations require explicit adoption."
                    ),
                    "graph_paths_available": (directory / "pav-metadata.json").exists(),
                },
            )
        )
        return outputs, None
    raise ValueError(f"unknown workflow stage: {name}")


def _publish(directory: Path, record: dict, request: WorkflowRequest) -> None:
    if (directory / "pav-metadata.json").exists():
        from .pav_store import verify_pav_store

        verify_pav_store(directory)
    sources = _read(directory / "inputs.json")["sources"]
    snapshots = _read(directory / "inputs.json")["snapshots"]
    graph_ref = ArtifactRef.from_path(_graph_path(directory), kind="pangenome_graph", format="gfa")
    input_hashes = tuple(
        dict.fromkeys(
            [
                *(source["sha256"] for source in [*sources, *snapshots]),
                graph_ref.sha256,
            ]
        )
    )
    artifacts = []
    for file in sorted(directory.rglob("*")):
        if not file.is_file() or file.name.endswith(".tmp") or file.name == "result.json":
            continue
        kind = (
            "pangenome_graph_reference"
            if file.name == "graph-reference.json"
            else "pangenome_graph"
            if file.name == "graph.gfa"
            else "pangenome_output"
        )
        if file.name == "annotation.json":
            kind = "pangenome_annotation"
        ref = ArtifactRef.from_path(
            file,
            kind=kind,
            format=file.suffix.lstrip("."),
            media_type=mimetypes.guess_type(file)[0] or "application/octet-stream",
        )
        artifacts.append(ref.model_copy(update={"uri": str(file.relative_to(directory))}))
    result = OrganelleResult(
        operation_id=store.OPERATION,
        operation_version=record["workflow_version"],
        scope=request.organelle,
        status="warning" if not (directory / "pav-metadata.json").exists() else "ok",
        summary_text="Pangenome graph analysis with verified stage checkpoints and source tables.",
        metrics=record["summary"] or {},
        artifacts=tuple(artifacts),
        provenance=make_provenance(
            operation_id=store.OPERATION,
            parameters=request.model_dump(),
            input_object_ids=tuple(
                dict.fromkeys(
                    [
                        *(
                            ArtifactRef.model_validate(source).object_id
                            for source in [*sources, *snapshots]
                        ),
                        graph_ref.object_id,
                    ]
                )
            ),
            input_artifact_hashes=input_hashes,
            requested_backend="organelleverse",
            actual_backend="organelleverse",
            attempted_backends=("organelleverse",),
        ).model_copy(update={"operation_version": record["workflow_version"]}),
    )
    destination = managed_run_path(store.OPERATION, record["run_id"])
    published = publish_staged_result(
        result, directory, destination, trusted_input_hashes=frozenset(input_hashes)
    )
    _write(destination / "result.json", published.model_dump(mode="json"))


def _public_files(directory: Path) -> list[dict]:
    return [
        {"name": str(p.relative_to(directory)), "kind": p.suffix.lstrip(".")}
        for p in sorted(directory.rglob("*"))
        if p.is_file()
        and p.name not in {"workflow.json", "workflow.json.tmp"}
        and "inputs" not in p.relative_to(directory).parts
    ] + [{"name": "result.json", "kind": "json"}]


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return path


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _optional(path: Path) -> dict | None:
    return _read(path) if path.exists() else None


def _verified_artifact(value: dict) -> Path:
    artifact = ArtifactRef.model_validate(value)
    path = artifact.resolve()
    current = ArtifactRef.from_path(path, kind=artifact.kind, format=artifact.format)
    if current.sha256 != artifact.sha256 or current.size_bytes != artifact.size_bytes:
        raise ValueError(f"checkpoint artifact changed: {path.name}")
    return path


def _graph_path(directory: Path) -> Path:
    build = directory / "build.json"
    reference = (
        _read(build)["graph_ref"] if build.exists() else _read(directory / "inputs.json")["graph"]
    )
    return _verified_artifact(reference)


def graph_path(run_id: str) -> Path:
    """Resolve the verified managed graph; imported graphs remain input snapshots."""
    return _graph_path(store.directory_for(run_id))
