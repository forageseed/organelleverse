"""Workflow checkpoints inside the existing L6 managed run store."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from ..core.artifacts import ArtifactRef
from ..runtime import create_staged_run, managed_run_path, managed_runs_root, staged_run_path
from ._contract import utc_now
from .workflow_models import WorkflowRequest

OPERATION = "pangenome.workflow"
WORKFLOW_VERSION = "2.1"
STAGES = (
    "prepare",
    "repeat",
    "graph",
    "validate",
    "statistics",
    "overview",
    "phylogeny",
    "annotation",
    "report",
)


def create(request: WorkflowRequest) -> dict:
    run_id = uuid4().hex
    directory = create_staged_run(OPERATION, run_id)
    now = utc_now().isoformat()
    record = {
        "run_id": run_id,
        "workflow_version": WORKFLOW_VERSION,
        "status": "queued",
        "stage": "prepare",
        "created_at": now,
        "updated_at": now,
        "request": request.model_dump(),
        "error": None,
        "logs": [],
        "summary": None,
        "files": [],
        "stages": [{"name": name, "status": "pending", "outputs": []} for name in STAGES],
    }
    save(directory, record)
    return record


def pause_path(run_id: str) -> Path:
    """Stage-boundary pause intent, outside immutable artifact directories."""
    return managed_run_path(OPERATION, run_id).parent / f".{run_id}.pause"


def directory_for(run_id: str) -> Path:
    completed = managed_run_path(OPERATION, run_id)
    return completed if completed.exists() else staged_run_path(OPERATION, run_id)


def load(run_id: str) -> dict:
    return read_json(run_id, "workflow.json")


def read_json(run_id: str, name: str) -> dict:
    """Read across the one-way atomic staging-to-publication directory rename."""
    try:
        content = (staged_run_path(OPERATION, run_id) / name).read_text()
    except FileNotFoundError:
        content = (managed_run_path(OPERATION, run_id) / name).read_text()
    return json.loads(content)


def save(directory: Path, record: dict) -> None:
    record["updated_at"] = utc_now().isoformat()
    temporary = directory / "workflow.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, directory / "workflow.json")


def list_runs() -> list[dict]:
    root = managed_runs_root() / OPERATION
    return sorted(
        (
            load(run_id)
            for run_id in {
                p.parent.name.removeprefix(".staging-") for p in root.glob("*/workflow.json")
            }
        ),
        key=lambda r: r["created_at"],
        reverse=True,
    )


def capture(directory: Path, paths: list[Path]) -> list[dict]:
    outputs = []
    for path in paths:
        ref = ArtifactRef.from_path(path, kind="pangenome_output", format=path.suffix.lstrip("."))
        outputs.append(
            {
                "name": str(path.relative_to(directory))
                if path.is_relative_to(directory)
                else str(path),
                "sha256": ref.sha256,
                "size_bytes": ref.size_bytes,
            }
        )
    return outputs


def verify(directory: Path, outputs: list[dict]) -> None:
    for output in outputs:
        path = directory / output["name"]
        if not path.is_file():
            raise ValueError(f"checkpoint artifact missing: {output['name']}")
        current = ArtifactRef.from_path(
            path, kind="pangenome_output", format=path.suffix.lstrip(".")
        )
        if current.sha256 != output["sha256"] or current.size_bytes != output["size_bytes"]:
            raise ValueError(f"checkpoint artifact changed: {output['name']}")


def public_view(record: dict) -> dict:
    public = {key: value for key, value in record.items() if key != "request"}
    public["pause_requested"] = pause_path(record["run_id"]).exists()
    source = record.get("request", {}).get("dataset_path") or record.get("request", {}).get(
        "gfa_path"
    )
    if source:
        public["label"] = Path(source).name
    public["parameters"] = {
        key: value
        for key, value in record.get("request", {}).items()
        if key
        in {
            "backend",
            "organelle",
            "threads",
            "segment_length",
            "identity",
            "cloud_threshold",
            "core_threshold",
            "bootstrap_replicates",
            "seed",
            "formats",
            "generate_overview",
            "window_bp",
            "recommend_parameters",
            "auto_adopt_recommendation",
            "run_repeatmasker",
            "repeatmasker_species",
            "k",
            "pantools_memory_mb",
            "normalization",
            "molecule_topologies",
            "gene_synonyms",
            "expected_genes",
            "annotation_reference_paths",
            "annotation_untangle",
            "tree_mode",
            "bootstrap_method",
            "adaptive_min_replicates",
            "adaptive_max_replicates",
            "adaptive_batch_size",
            "adaptive_convergence_threshold",
            "msa_format",
            "msa_seed",
            "msa_taxon_mapping",
            "msa_taxon_unit",
            "raxml_model",
            "tree_parsimony_starts",
            "tree_random_starts",
        }
    }
    return public


def dag_runs():
    """Project durable native workflow and embedded producer receipts read-only.

    The Result DAG shows recorded provenance; opening/using an artifact still
    performs the existing checkpoint verification at that operation boundary.
    """
    from ..core.result import OrganelleResult
    from ..results.dag import NativeRun

    runs = []
    for record in list_runs():
        run_id = record["run_id"]
        result = None
        status = record["status"]
        if status == "succeeded":
            try:
                result = OrganelleResult.model_validate(read_json(run_id, "result.json"))
            except FileNotFoundError:
                # The final receipt is written after the atomic publication
                # rename; the producer remains visible during that interval.
                status = "publishing"
        runs.append(NativeRun(run_id, OPERATION, status, record["created_at"], result))
        names = {stage["name"] for stage in record["stages"] if stage["status"] == "completed"}
        for stage, filename, key in (
            ("graph", "build.json", "result"),
            ("repeat", "recommendation.json", "recommendation_result"),
        ):
            path = directory_for(run_id) / filename
            if stage in names and path.exists():
                producer = OrganelleResult.model_validate(read_json(run_id, filename)[key])
                runs.append(
                    NativeRun(
                        producer.object_id, producer.operation_id, producer.status, result=producer
                    )
                )
    # Conversion receipts share the native Result/ArtifactRef contract. Their
    # consumed graph digest links them to the existing workflow or builder.
    for receipt in sorted((managed_runs_root() / "pangenome.convert_graph").glob("*/result.json")):
        if receipt.parent.name.startswith(".staging-"):
            continue
        conversion = OrganelleResult.model_validate_json(receipt.read_text())
        if conversion.operation_id != "pangenome.convert_graph":
            raise ValueError("Native conversion receipt has an unexpected operation")
        runs.append(
            NativeRun(
                conversion.object_id, conversion.operation_id, conversion.status, result=conversion
            )
        )
    return tuple(runs)
