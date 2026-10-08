"""Managed pangenome graph service for Registry and Agent callers.

The public contract accepts scientific inputs and parameters only.  Filesystem
placement is derived from their content identity and remains inside the L6
managed run store.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Literal, cast

from ..core.artifacts import ArtifactRef
from ..core.errors import OrganelleInputError
from ..core.genome import OrganelleGenome
from ..core.result import OrganelleResult
from ..quality_control.static import validate_gfa
from ..runtime import (
    create_staged_run,
    managed_run_path,
    publish_staged_result,
    verify_staged_artifacts,
)
from ._contract import OPERATION_VERSION as _OPERATION_VERSION
from ._contract import parameters_hash
from .pangenome import build_graph as _core_build_graph
from .project import PangenomeProject
from .tuning import (
    SEGMENT_POLICY_SOURCE,
    Recommendation,
    protected_segment_length,
    recommendation_segment_estimate,
    sequence_input_hashes,
    validate_recommendation,
    write_adoption_record,
)

_OPERATION_ID = "pangenome.build_graph"


def build_graph(
    genomes: list[OrganelleGenome],
    *,
    method: Literal["minigraph", "pggb", "pantools"] = "minigraph",
    k: int = 31,
    pantools_memory_mb: int = 4096,
    threads: int = 8,
    n_haplotypes: int | None = None,
    segment_length: int = 5000,
    identity: float = 90.0,
    recommendation: Recommendation | None = None,
    auto_adopt_recommendation: bool = False,
    reference_index: int = 0,
) -> OrganelleResult:
    """Build a graph in a deterministic managed workspace.

    No caller-provided destination or executable callback crosses this public
    boundary.  The legacy Python builder remains available separately for
    direct, explicitly placed workflows.
    With ``auto_adopt_recommendation=True``, use identity and segment length
    from a supplied, verified ``recommend_parameters`` payload. By default
    parameters remain explicit and any supplied recommendation must match.
    Plastid PGGB segment lengths have an empirical 5000 bp minimum.
    """

    if method not in {"minigraph", "pggb", "pantools"}:
        raise OrganelleInputError(
            code="pangenome.unsupported_method",
            message="managed pangenome builds support minigraph, pggb or pantools",
            details={"method": method},
        )
    project = PangenomeProject.from_genomes(genomes)
    if auto_adopt_recommendation:
        if method != "pggb" or recommendation is None:
            raise OrganelleInputError(
                code="pangenome.recommendation_required",
                message="Automatic adoption requires PGGB and a supplied recommendation",
            )
        # Validate the closed evidence payload before adopting any of its values.
        validated = validate_recommendation(
            recommendation,
            input_hashes=_source_hashes(project),
            identity=cast(float, recommendation.get("identity")),
            segment_length=cast(int, recommendation.get("segment_length")),
        )
        identity = cast(float, validated["identity"])
        segment_length = cast(int, validated["segment_length"])
    invalid = {
        "k": k <= 0 or (method == "pantools" and not 6 <= k <= 255),
        "pantools_memory_mb": pantools_memory_mb < 128,
        "threads": threads <= 0,
        "segment_length": segment_length <= 0,
        "identity": not 0 < identity <= 100,
        "reference_index": not 0 <= reference_index < len(project.samples),
        "n_haplotypes": n_haplotypes is not None and n_haplotypes <= 0,
    }
    rejected = sorted(name for name, failed in invalid.items() if failed)
    if rejected:
        raise OrganelleInputError(
            code="pangenome.invalid_parameter",
            message="managed pangenome build parameters are outside supported ranges",
            details={"parameters": rejected},
        )
    project.verify_sources()
    adopted_recommendation = (
        validate_recommendation(
            recommendation,
            input_hashes=_source_hashes(project),
            identity=identity,
            segment_length=segment_length,
        )
        if recommendation is not None
        else None
    )
    requested_segment_length = segment_length
    if method == "pggb":
        segment_length = protected_segment_length(genomes, segment_length)
        if adopted_recommendation is not None and segment_length != requested_segment_length:
            raise OrganelleInputError(
                code="pangenome.recommendation_below_plastid_floor",
                message="Recommendation is below the plastid 5000 bp minimum; regenerate it",
            )
    parameters: dict[str, Any] = {
        "method": method,
        "k": k,
        **({"pantools_memory_mb": pantools_memory_mb} if method == "pantools" else {}),
        "threads": threads,
        "n_haplotypes": n_haplotypes,
        "segment_length": segment_length,
        "identity": identity,
        "reference_index": reference_index,
        "requested_segment_length": requested_segment_length,
        "auto_adopt_recommendation": auto_adopt_recommendation,
    }
    if adopted_recommendation is not None:
        parameters["recommendation_digest"] = adopted_recommendation["recommendation_digest"]
    digest = _run_digest(project, parameters)
    run_id = f"sha256-{digest}"
    completed = managed_run_path(_OPERATION_ID, run_id)
    if completed.exists():
        return _load_reusable_result(
            completed,
            run_digest=digest,
            parameters=parameters,
            source_hashes=_source_hashes(project),
        )

    staging = create_staged_run(_OPERATION_ID, run_id)
    workspace = staging / "workspace"
    try:
        workspace.mkdir()
        staged_genomes = _snapshot_genomes(project, workspace / "inputs")
        result = _core_build_graph(
            staged_genomes,
            output_dir=workspace,
            method=method,
            k=k,
            pantools_memory_mb=pantools_memory_mb,
            threads=threads,
            n_haplotypes=n_haplotypes,
            segment_length=segment_length,
            identity=identity,
            reference_index=reference_index,
        )
        if result.status == "failed":
            return result.model_copy(
                update={
                    "flags": tuple(flag for flag in result.flags if flag != "graph_built"),
                    "artifacts": tuple(
                        artifact
                        for artifact in result.artifacts
                        if artifact.kind not in {"pangenome_graph", "pangenome_parameter_adoption"}
                    ),
                }
            )
        require_graph_result(result, backend=method)
        staged_result = _stage_result(
            result,
            staging,
            workspace,
            run_digest=digest,
            parameters=parameters,
            source_hashes=_source_hashes(project),
            recommendation=adopted_recommendation,
        )
        published = publish_staged_result(
            staged_result,
            staging,
            completed,
            trusted_input_hashes=frozenset(_source_hashes(project)),
        )
        assert isinstance(published, OrganelleResult)
        return published
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _run_digest(project: PangenomeProject, parameters: dict[str, Any]) -> str:
    payload = {
        "operation_id": _OPERATION_ID,
        "operation_version": _OPERATION_VERSION,
        "output_contract_version": 2,
        "parameters": parameters,
        "samples": [
            {
                "name": sample.name,
                "object_id": sample.genome.object_id,
                "sequence_sha256": sample.genome.sequence.sha256,
            }
            for sample in project.samples
            if sample.genome.sequence is not None
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_hashes(project: PangenomeProject) -> tuple[str, ...]:
    return sequence_input_hashes([sample.genome for sample in project.samples])


def _snapshot_genomes(project: PangenomeProject, target: Path) -> list[OrganelleGenome]:
    """Copy verified source bytes once and execute only immutable snapshots."""
    target.mkdir(parents=True)
    snapshots: list[OrganelleGenome] = []
    for index, sample in enumerate(project.samples, start=1):
        source = sample.genome.sequence
        assert source is not None
        destination = target / f"{index:04d}-{sample.name}.fa"
        digest = hashlib.sha256()
        size = 0
        try:
            with source.resolve().open("rb") as reader, destination.open("xb") as writer:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                    writer.write(chunk)
        except OSError as error:
            raise OrganelleInputError(
                code="pangenome.source_snapshot_failed",
                message="pangenome source could not be copied into managed staging",
                details={"uri": source.uri, "reason": str(error)},
            ) from error
        if digest.hexdigest() != source.sha256 or size != source.size_bytes:
            raise OrganelleInputError(
                code="pangenome.source_digest_mismatch",
                message="pangenome source bytes changed while entering managed staging",
                details={"uri": source.uri, "expected_sha256": source.sha256},
            )
        destination.chmod(0o444)
        staged = ArtifactRef.from_path(
            destination,
            kind=source.kind,
            format=source.format,
            media_type=source.media_type,
        )
        snapshots.append(sample.genome.model_copy(update={"sequence": staged}))
    return snapshots


def require_graph_result(
    result: OrganelleResult,
    *,
    backend: str,
    artifact_root: str | Path | None = None,
) -> ArtifactRef:
    """Require one rehashed, structurally valid built graph from ``backend``."""

    def reject(reason: str, **details: Any) -> None:
        raise OrganelleInputError(
            code="pangenome.invalid_builder_result",
            message=f"pangenome builder result is not publishable: {reason}",
            details={"reason": reason, **details},
        )

    if result.operation_id != _OPERATION_ID:
        reject("wrong operation", operation_id=result.operation_id)
    if result.status != "ok" or result.errors:
        reject("status/errors do not describe success", status=result.status)
    if "graph_built" not in result.flags or "graph_planned" in result.flags:
        reject("graph_built flag is required and graph_planned is forbidden")
    provenance = result.provenance
    if provenance is None or provenance.actual_backend != backend:
        reject(
            "actual backend provenance does not match request",
            actual_backend=None if provenance is None else provenance.actual_backend,
            expected_backend=backend,
        )
    graphs = [artifact for artifact in result.artifacts if artifact.kind == "pangenome_graph"]
    if len(graphs) != 1:
        reject("exactly one pangenome_graph artifact is required", count=len(graphs))
    graph = graphs[0]
    if graph.format != "gfa":
        reject("pangenome graph format must be gfa", format=graph.format)
    declared_path = Path(graph.uri).expanduser()
    root = None if artifact_root is None else Path(artifact_root).expanduser()
    graph_path = (
        declared_path if declared_path.is_absolute() else (root / declared_path if root else None)
    )
    if graph_path is None:
        reject("relative graph artifact requires an artifact root", uri=graph.uri)
    try:
        current = ArtifactRef.from_path(
            graph_path,
            kind=graph.kind,
            format=graph.format,
            media_type=graph.media_type,
        )
        if current.sha256 != graph.sha256 or current.size_bytes != graph.size_bytes:
            reject("graph artifact digest does not match disk", uri=graph.uri)
        from .graph import load_gfa

        load_gfa(graph_path)
        summary = validate_gfa(graph_path)
        if summary.segment_count < 1:
            reject("graph contains no segments", uri=graph.uri)
    except OrganelleInputError as error:
        if error.code == "pangenome.invalid_builder_result":
            raise
        reject("graph artifact failed structural validation", uri=graph.uri, cause=error.code)
    except (OSError, UnicodeError, ValueError) as error:
        reject("graph artifact could not be read or validated", uri=graph.uri, cause=str(error))
    return graph


def _stage_result(
    result: OrganelleResult,
    staging: Path,
    workspace: Path,
    *,
    run_digest: str,
    parameters: dict[str, Any],
    source_hashes: tuple[str, ...],
    recommendation: Recommendation | None,
) -> OrganelleResult:
    artifact_dir = staging / "artifacts"
    artifact_dir.mkdir()
    relocated: list[ArtifactRef] = []
    for index, artifact in enumerate(result.artifacts, start=1):
        source = Path(artifact.uri).expanduser().resolve()
        try:
            source.relative_to(workspace.resolve())
        except ValueError as error:
            raise OrganelleInputError(
                code="pangenome.artifact_outside_workspace",
                message="pangenome builder declared an artifact outside its managed workspace",
                details={"uri": artifact.uri},
            ) from error
        if not source.is_file():
            raise OrganelleInputError(
                code="pangenome.artifact_not_file",
                message="pangenome graph artifact must be a regular file",
                details={"uri": artifact.uri},
            )
        target_name = f"{index:02d}-{source.name}"
        target = artifact_dir / target_name
        shutil.copy2(source, target)
        copied = ArtifactRef.from_path(
            target,
            kind=artifact.kind,
            format=artifact.format,
            media_type=artifact.media_type,
        )
        if copied.sha256 != artifact.sha256 or copied.size_bytes != artifact.size_bytes:
            raise OrganelleInputError(
                code="pangenome.artifact_digest_mismatch",
                message="pangenome graph artifact changed while entering managed storage",
                details={"uri": artifact.uri},
            )
        relocated.append(copied.model_copy(update={"uri": f"artifacts/{target_name}"}))
    shutil.rmtree(workspace)
    managed_metrics: dict[str, Any] = dict(result.metrics)
    if parameters["method"] == "pggb":
        # The core builder receives the protected value. Preserve the public
        # caller's proposal rather than its already-protected staging argument.
        managed_metrics["requested_segment_length"] = parameters["requested_segment_length"]
        managed_metrics["segment_length"] = parameters["segment_length"]
    managed_metrics["parameter_policy"] = {
        "requested_segment_length": parameters["requested_segment_length"],
        "segment_length": parameters["segment_length"],
        "identity": parameters["identity"],
        "auto_adopt_recommendation": parameters["auto_adopt_recommendation"],
        "recommendation_digest": parameters.get("recommendation_digest"),
        "derived_segment_length": recommendation_segment_estimate(recommendation)
        if recommendation is not None
        else None,
        "plastid_floor_source": SEGMENT_POLICY_SOURCE,
    }
    managed_metrics.pop("output_path", None)
    base_result = result.model_copy(
        update={
            "artifacts": tuple(relocated),
            "metrics": managed_metrics,
            "provenance": result.provenance.model_copy(
                update={"parameters_hash": parameters_hash(parameters)}
            )
            if result.provenance is not None
            else None,
        }
    )
    if recommendation is not None:
        graph = next(
            artifact for artifact in base_result.artifacts if artifact.kind == "pangenome_graph"
        )
        adoption = write_adoption_record(
            staging / "adoption.json",
            recommendation=recommendation,
            graph=graph,
        ).model_copy(update={"uri": "adoption.json"})
        base_result = base_result.model_copy(
            update={"artifacts": (*base_result.artifacts, adoption)}
        )
    record_path = staging / "run-record.json"
    record_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "operation_id": _OPERATION_ID,
                "operation_version": _OPERATION_VERSION,
                "run_digest": run_digest,
                "parameters": parameters,
                "source_hashes": list(source_hashes),
                "result": base_result.model_dump(mode="json"),
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    record_artifact = ArtifactRef.from_path(
        record_path,
        kind="pangenome_run_record",
        format="json",
        media_type="application/json",
    ).model_copy(update={"uri": "run-record.json"})
    return base_result.model_copy(update={"artifacts": (*base_result.artifacts, record_artifact)})


def _load_reusable_result(
    completed: Path,
    *,
    run_digest: str,
    parameters: dict[str, Any],
    source_hashes: tuple[str, ...],
) -> OrganelleResult:
    if completed.is_symlink() or not completed.is_dir():
        raise OrganelleInputError(
            code="pangenome.managed_run_conflict",
            message="existing managed pangenome run must be a real directory",
            details={"path": str(completed)},
        )
    result_path = completed / "run-record.json"
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "operation_id": _OPERATION_ID,
            "operation_version": _OPERATION_VERSION,
            "run_digest": run_digest,
            "parameters": parameters,
            "source_hashes": list(source_hashes),
        }
        if not isinstance(payload, dict) or any(
            payload.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("run record identity does not match the request")
        result = OrganelleResult.model_validate(payload["result"])
        if result.operation_id != _OPERATION_ID:
            raise ValueError("run record operation identity does not match")
        require_graph_result(
            result,
            backend=str(parameters["method"]),
            artifact_root=completed,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise OrganelleInputError(
            code="pangenome.managed_run_conflict",
            message="existing managed pangenome run is incomplete or invalid",
            details={"path": str(completed), "reason": str(error)},
        ) from error
    record_artifact = ArtifactRef.from_path(
        result_path,
        kind="pangenome_run_record",
        format="json",
        media_type="application/json",
    ).model_copy(update={"uri": "run-record.json"})
    staged_shape = result.model_copy(update={"artifacts": (*result.artifacts, record_artifact)})
    verify_staged_artifacts(
        staged_shape,
        completed,
        trusted_input_hashes=frozenset(source_hashes),
    )
    return staged_shape.model_copy(
        update={
            "artifacts": tuple(
                artifact.model_copy(update={"uri": str(completed / artifact.uri)})
                for artifact in staged_shape.artifacts
            )
        }
    )


__all__ = ["build_graph", "require_graph_result"]
