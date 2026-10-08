"""Agent-callable analysis of an existing verified pangenome graph Result."""

from __future__ import annotations

import json
from typing import Literal

from ..core.artifacts import ArtifactRef
from ..core.errors import OrganelleInputError
from ..core.result import ErrorDetail, OrganelleResult
from . import workflow, workflow_store
from ._contract import make_provenance
from .graph_operations import _graph_input
from .graph_selection import verified_artifact
from .workflow_models import WorkflowRequest


def analyze_graph(
    result: OrganelleResult,
    *,
    cloud_threshold: float = 0.05,
    core_threshold: float = 1.0,
    bootstrap_replicates: int = 100,
    seed: int = 0,
    threads: int = 4,
    tree_mode: Literal["pav", "msa", "none"] = "pav",
    bootstrap_method: Literal["felsenstein", "adaptive_pav"] = "felsenstein",
    adaptive_min_replicates: int = 1000,
    adaptive_max_replicates: int = 10000,
    adaptive_batch_size: int = 100,
    adaptive_convergence_threshold: float = 0.99,
    alignment_artifact_id: str | None = None,
    msa_taxon_mapping: dict[str, str] | None = None,
    msa_taxon_unit: Literal["sample", "path"] = "sample",
    msa_seed: int = 1,
    raxml_model: str = "GTR+G",
    tree_parsimony_starts: int = 10,
    tree_random_starts: int = 10,
    annotation_artifact_id: str | None = None,
    gene_synonyms: dict[str, str] | None = None,
    expected_genes: list[str] | None = None,
    annotation_reference_paths: list[str] | None = None,
    annotation_untangle: Literal["exact", "odgi"] = "exact",
    window_bp: int = 10000,
    generate_overview: bool = True,
    formats: list[Literal["svg", "pdf", "png"]] | None = None,
) -> OrganelleResult:
    """Compute real graph statistics, sample PAV, similarity tree and report.

    Accepts a canonical graph result, never a caller-supplied output directory.
    Alignment and annotation selectors are opaque ArtifactRef object IDs already present
    in the input Result, verified before snapshotting; they are not free paths.
    The desktop workbench invokes the same stage implementation. Graphs without
    sample paths yield statistics and an explicit warning, not invented PAV.
    """
    graph = _graph_input(result, {"gfa"})
    alignment = _selected_artifact(result, alignment_artifact_id, {"fasta", "maf"})
    annotations = _selected_artifact(result, annotation_artifact_id, {"gff", "gff3", "bed"})
    request = WorkflowRequest(
        backend="existing",
        gfa_path=str(graph.resolve()),
        organelle=result.scope,
        cloud_threshold=cloud_threshold,
        core_threshold=core_threshold,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        threads=threads,
        tree_mode=tree_mode,
        bootstrap_method=bootstrap_method,
        adaptive_min_replicates=adaptive_min_replicates,
        adaptive_max_replicates=adaptive_max_replicates,
        adaptive_batch_size=adaptive_batch_size,
        adaptive_convergence_threshold=adaptive_convergence_threshold,
        msa_path=str(verified_artifact(alignment)) if alignment else None,
        msa_format=alignment.format if alignment else "fasta",
        msa_taxon_mapping=msa_taxon_mapping or {},
        msa_taxon_unit=msa_taxon_unit,
        msa_seed=msa_seed,
        raxml_model=raxml_model,
        tree_parsimony_starts=tree_parsimony_starts,
        tree_random_starts=tree_random_starts,
        annotations_path=str(verified_artifact(annotations)) if annotations else None,
        gene_synonyms=gene_synonyms or {},
        expected_genes=tuple(expected_genes or ()),
        annotation_reference_paths=tuple(annotation_reference_paths or ()),
        annotation_untangle=annotation_untangle,
        window_bp=window_bp,
        generate_overview=generate_overview,
        formats=tuple(formats) if formats is not None else ("svg", "pdf", "png"),
    )
    run = workflow.create_run(request)
    finished = workflow.execute_run(run["run_id"])
    if finished["status"] != "succeeded":
        message = finished.get("error") or (
            f"Analysis {finished['status']}; resume run {finished['run_id']} to complete."
        )
        return OrganelleResult(
            operation_id="pangenome.analyze_graph",
            operation_version="1.0",
            scope=result.scope,
            status="failed",
            summary_text=message,
            errors=(ErrorDetail(code="pangenome.analysis_failed", message=message),),
        )
    path = workflow_store.directory_for(run["run_id"]) / "result.json"
    analyzed = OrganelleResult.model_validate(json.loads(path.read_text()))
    return analyzed.model_copy(
        update={
            "operation_id": "pangenome.analyze_graph",
            "provenance": make_provenance(
                operation_id="pangenome.analyze_graph",
                parameters={
                    **{
                        key: value
                        for key, value in request.model_dump(mode="json").items()
                        if key not in {"dataset_path", "gfa_path", "msa_path", "annotations_path"}
                    },
                    "alignment_artifact_id": alignment_artifact_id,
                    "annotation_artifact_id": annotation_artifact_id,
                },
                input_object_ids=(result.object_id,),
                input_artifact_hashes=tuple(
                    dict.fromkeys(
                        [
                            *analyzed.provenance.input_artifact_hashes,
                            *[
                                a.sha256
                                for a in result.artifacts
                                if a.kind == "pangenome_graph_reference"
                            ],
                        ]
                    )
                ),
                requested_backend="organelleverse",
                actual_backend="organelleverse",
                attempted_backends=("organelleverse",),
            ).model_copy(update={"operation_version": "1.0"}),
        }
    )


def _selected_artifact(
    result: OrganelleResult, artifact_id: str | None, formats: set[str]
) -> ArtifactRef | None:
    if artifact_id is None:
        return None
    selected = [artifact for artifact in result.artifacts if artifact.object_id == artifact_id]
    if len(selected) != 1 or selected[0].format not in formats:
        raise OrganelleInputError(
            code="pangenome.analysis_artifact_required",
            message=f"Select exactly one existing Result artifact in {sorted(formats)} by its recorded artifact_id",
        )
    verified_artifact(selected[0])
    return selected[0]
