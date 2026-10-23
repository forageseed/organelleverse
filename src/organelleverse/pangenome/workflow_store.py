"""Workflow checkpoints inside the existing L6 managed run store."""

from __future__ import annotations

import json
import os
import time
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
    def read() -> str:
        try:
            return (staged_run_path(OPERATION, run_id) / name).read_text()
        except FileNotFoundError:
            return (managed_run_path(OPERATION, run_id) / name).read_text()

    return json.loads(_retry_sharing_violation(read))


def save(directory: Path, record: dict) -> None:
    record["updated_at"] = utc_now().isoformat()
    temporary = directory / "workflow.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    _retry_sharing_violation(lambda: os.replace(temporary, directory / "workflow.json"))


def _retry_sharing_violation(action, attempts: int = 40, delay: float = 0.025):
    """Windows refuses to replace a file another handle is reading (WinError 5/32).

    Status readers poll workflow.json while the worker checkpoints it; such a
    collision lasts milliseconds, so retry briefly instead of failing the run.
    """
    for attempt in range(attempts):
        try:
            return action()
        except PermissionError:
            if os.name != "nt" or attempt == attempts - 1:
                raise
            time.sleep(delay)


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
                "name": path.relative_to(directory).as_posix()
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
