"""Released PMAT2 ``graphBuild`` continuation operation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from organelleverse.assembly.api import PmatGraphEnvironmentHint, pmat_graph_build
from organelleverse.assembly.backends.base import BackendOutput
from organelleverse.assembly.backends.pmat_graph import PmatGraphAdapter, PmatGraphContext
from organelleverse.assembly.backends.runtime import pmat_runtime
from organelleverse.assembly.continuation import (
    DirectoryManifest,
    materialize_directory_manifest,
)
from organelleverse.assembly.data_contract import validate_pmat_graph_build_data
from organelleverse.assembly.environment_capabilities import runtime_capabilities
from organelleverse.assembly.environment_contracts import (
    BackendVersionSelector,
    EnvironmentHint,
    EnvironmentSource,
)
from organelleverse.assembly.environment_resolver import EnvironmentResolver
from organelleverse.assembly.environment_versions import (
    environment_spec_for_version,
    resolve_backend_version,
)
from organelleverse.assembly.environments import EnvironmentManager, PreparedEnvironment
from organelleverse.assembly.execution import CommandOutcome, CommandRunner
from organelleverse.assembly.manifests import (
    AssemblyComponentIdentity,
    AssemblyEnvironmentIdentity,
    AssemblyRunManifest,
    AssemblyRunObservations,
    AssemblyRunParameters,
    AssemblyStageOutcome,
    ManifestArtifact,
)
from organelleverse.assembly.service import (
    execution_environment,
    package_version,
    published_artifact,
    remove_workspace,
    reverify_all_input_artifacts,
    temporary_sibling,
    verify_published_artifacts,
    write_manifest_evidence,
)
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import ErrorDetail, OrganelleResult
from organelleverse.core.serialization import load_result, save_contract
from organelleverse.operations.spec import StrictSpecModel
from organelleverse.runtime import managed_run_path, publish_run

__all__ = [
    "PmatGraphBuildParameters",
    "PmatGraphEnvironmentHint",
    "execute_pmat_graph_build",
    "pmat_graph_build",
]

_OPERATION_ID = "assembly.pmat_graph_build"
# Invalidate graphBuild results cached before the managed orientation correction.
_OPERATION_VERSION = "1.1"


class PmatGraphBuildParameters(StrictSpecModel):
    """Every public non-path PMAT2 v2.1.5 graphBuild option."""

    organelle: Literal["mitochondrion", "plastid"]
    taxon_group: Literal["plant", "animal", "fungi"] = "plant"
    depth: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    seeds: tuple[int, ...] | None = None
    threads: int = Field(default=8, ge=1, le=256)

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is not None and (
            not value or any(seed < 1 for seed in value) or len(set(value)) != len(value)
        ):
            raise ValueError("seeds must be unique positive contig IDs")
        return value


class _PmatGraphRequest(StrictSpecModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
        arbitrary_types_allowed=True,
    )

    data: OrganelleData
    parameters: PmatGraphBuildParameters
    timeout_seconds: int | None = Field(default=None, ge=1)
    environment_source: EnvironmentSource = "auto"
    backend_version: BackendVersionSelector = "tested"
    environment_hint: EnvironmentHint | None = None

    @property
    def semantic_hash(self) -> str:
        payload = {
            "operation_id": _OPERATION_ID,
            "operation_version": _OPERATION_VERSION,
            "input_data_id": self.data.object_id,
            "input_artifacts": {
                role: artifact.sha256 for role, artifact in sorted(self.data.artifacts.items())
            },
            "parameters": self.parameters.model_dump(mode="json"),
            "environment_source": self.environment_source,
            "backend_version": self.backend_version,
            "environment_hint": (
                self.environment_hint.model_dump(mode="json")
                if self.environment_hint is not None
                else None
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def execute_pmat_graph_build(
    data: OrganelleData,
    *,
    organelle: Literal["mitochondrion", "plastid"],
    taxon_group: Literal["plant", "animal", "fungi"] = "plant",
    depth: float | None = None,
    seeds: tuple[int, ...] | None = None,
    threads: int = 8,
    timeout_seconds: int | None = None,
    environment_source: EnvironmentSource = "auto",
    backend_version: BackendVersionSelector = "tested",
    environment_hint: EnvironmentHint | None = None,
    environment_manager: EnvironmentManager | None = None,
    environment_resolver: EnvironmentResolver | None = None,
    runner: CommandRunner | None = None,
    adapter: PmatGraphAdapter | None = None,
) -> OrganelleResult:
    payload = validate_pmat_graph_build_data(data)
    parameters = PmatGraphBuildParameters(
        organelle=organelle,
        taxon_group=taxon_group,
        depth=depth,
        seeds=seeds,
        threads=threads,
    )
    request = _PmatGraphRequest(
        data=data,
        parameters=parameters,
        timeout_seconds=timeout_seconds,
        environment_source=environment_source,
        backend_version=backend_version,
        environment_hint=environment_hint,
    )
    reverify_all_input_artifacts(data)
    destination = managed_run_path(_OPERATION_ID, f"sha256-{request.semantic_hash}")
    if destination.exists():
        return _load_reusable(destination, request)

    manager = environment_manager or EnvironmentManager()
    runtime = pmat_runtime()
    environment: PreparedEnvironment
    if environment_manager is None or environment_resolver is not None:
        effective = {"correction_task": "skip", "correction_software": "nextdenovo"}
        resolved_version = resolve_backend_version("pmat", backend_version)
        capabilities = runtime_capabilities(runtime, effective, resolved_version)
        capabilities = capabilities.model_copy(
            update={
                "items": tuple(
                    item for item in capabilities.items if item.role in {"pmat", "blastn"}
                )
            }
        )
        resolution = (environment_resolver or EnvironmentResolver()).resolve(
            request=request,
            capability_contract=capabilities,
            resolved_version=resolved_version,
            effective_parameters=effective,
        )
        if resolution.selected_provider is not None:
            environment = manager.prepare_provider(resolution.selected_provider)
        else:
            environment_spec = environment_spec_for_version(
                runtime.environment_spec,
                resolved_version,
            )
            environment = manager.prepare(environment_spec, policy="ensure")
            if type(manager) is EnvironmentManager:
                environment = manager.register_prepared_provider(
                    environment,
                    capabilities,
                    requested_source=environment_source,
                    resolved_version=resolved_version.version,
                )
    else:
        environment = manager.prepare(runtime.environment_spec, policy="ensure")

    workspace = temporary_sibling(destination)
    active_adapter = adapter or PmatGraphAdapter()
    outcome: CommandOutcome | None = None
    try:
        subsample_manifest = _read_manifest(data, payload.subsample_manifest_artifact)
        assembly_manifest = _read_manifest(data, payload.assembly_result_manifest_artifact)
        subsample_dir = workspace / "subsample"
        assembly_result_dir = workspace / "assembly_result"
        materialize_directory_manifest(subsample_manifest, data.artifacts, subsample_dir)
        materialize_directory_manifest(assembly_manifest, data.artifacts, assembly_result_dir)
        context = PmatGraphContext(
            workspace=workspace,
            environment=environment,
            subsample_dir=subsample_dir,
            assembly_result_dir=assembly_result_dir,
            organelle=organelle,
            taxon_group=taxon_group,
            depth=depth,
            seeds=seeds,
            threads=threads,
        )
        command = active_adapter.build_command(context)
        outcome = (runner or CommandRunner()).run(
            command.resolved_argv,
            stage="execute_backend",
            cwd=workspace,
            timeout_seconds=timeout_seconds or 86400,
            stdout_path=workspace / "stdout.log",
            stderr_path=workspace / "stderr.log",
            env=execution_environment(environment),
        )
        if not outcome.started:
            raise OrganelleExecutionError(
                code="assembly.environment_unavailable",
                message="PMAT graphBuild could not start",
            )
        if outcome.termination != "exit" or outcome.returncode != 0:
            result = _failed_result(
                request,
                environment,
                command.stable_argv,
                outcome,
                workspace,
                destination,
                code="assembly.execution_failed",
                message="PMAT graphBuild execution failed",
                failed_stage="execute_backend",
            )
            return _publish(result, outcome, workspace, destination)
        try:
            raw = active_adapter.collect_outputs(context)
        except OrganelleExecutionError as error:
            result = _failed_result(
                request,
                environment,
                command.stable_argv,
                outcome,
                workspace,
                destination,
                code=error.code,
                message=error.message,
                failed_stage="collect_outputs",
            )
            return _publish(result, outcome, workspace, destination)
        try:
            normalized = active_adapter.normalize(context, raw, workspace / "normalized")
        except OrganelleExecutionError as error:
            result = _failed_result(
                request,
                environment,
                command.stable_argv,
                outcome,
                workspace,
                destination,
                code=error.code,
                message=error.message,
                failed_stage="normalize_outputs",
            )
            return _publish(result, outcome, workspace, destination)
        result = _success_result(
            request,
            environment,
            command.stable_argv,
            outcome,
            normalized.outputs,
            normalized.primary_sequence_role,
            workspace,
            destination,
        )
        return _publish(result, outcome, workspace, destination)
    except BaseException:
        remove_workspace(workspace)
        raise


def _read_manifest(data: OrganelleData, role: str) -> DirectoryManifest:
    return DirectoryManifest.model_validate_json(Path(data.artifacts[role].uri).read_bytes())


def _manifest_parameters(request: _PmatGraphRequest) -> AssemblyRunParameters:
    return AssemblyRunParameters(
        threads=request.parameters.threads,
        timeout_seconds=request.timeout_seconds,
        environment_source=request.environment_source,
        backend_version=request.backend_version,
        environment_hint=request.environment_hint,
        backend_parameters=FrozenMap(request.parameters.model_dump(mode="json")),
    )


def _manifest_inputs(data: OrganelleData) -> tuple[ManifestArtifact, ...]:
    return tuple(
        ManifestArtifact(role=role, artifact=artifact)
        for role, artifact in sorted(data.artifacts.items())
    )


def _stages(*, failed_stage: str | None = None) -> tuple[AssemblyStageOutcome, ...]:
    names = (
        "validate_input",
        "resolve_environment",
        "prepare_workspace",
        "execute_backend",
        "collect_outputs",
        "normalize_outputs",
        "validate_outputs",
        "write_run_manifest",
        "write_result",
        "publish_output",
    )
    output: list[AssemblyStageOutcome] = []
    for name in names:
        if failed_stage is not None and name == failed_stage:
            output.append(AssemblyStageOutcome(stage=name, status="failed"))
            break
        output.append(AssemblyStageOutcome(stage=name, status="ok"))
    return tuple(output)


def _base_manifest(
    request: _PmatGraphRequest,
    environment: PreparedEnvironment,
    stable_argv: tuple[str, ...],
    outcome: CommandOutcome,
    *,
    stages: tuple[AssemblyStageOutcome, ...],
    outputs: tuple[ManifestArtifact, ...],
    primary_sequence_role: str = "assembly_fasta",
) -> AssemblyRunManifest:
    adjusted = tuple(
        AssemblyStageOutcome(
            stage=stage.stage,
            status=stage.status,
            process_started=stage.stage == "execute_backend",
            exit_code=outcome.returncode if stage.stage == "execute_backend" else None,
            termination=outcome.termination if stage.stage == "execute_backend" else None,
            output_roles=stage.output_roles,
        )
        for stage in stages
    )
    return AssemblyRunManifest(
        operation_id=_OPERATION_ID,
        input_data_id=request.data.object_id,
        input_artifacts=_manifest_inputs(request.data),
        organelle=request.parameters.organelle,
        parameters=_manifest_parameters(request),
        requested_method="pmat",
        selected_backend="pmat",
        route_reason_code="explicit.pmat_graph_build",
        environment=AssemblyEnvironmentIdentity(
            carrier=environment.carrier,
            digest=environment.digest,
            platform=environment.platform,
        ),
        components=(
            AssemblyComponentIdentity(
                category="software", name="pmat", version=environment.version
            ),
        ),
        primary_sequence_role=primary_sequence_role,
        stable_argv=stable_argv,
        stages=adjusted,
        process_exit_code=outcome.returncode,
        outputs=outputs,
    )


def _provenance(
    request: _PmatGraphRequest,
    environment: PreparedEnvironment,
    stable_argv: tuple[str, ...],
    outcome: CommandOutcome,
    run_manifest_id: str,
) -> ResultProvenance:
    return ResultProvenance(
        operation_id=_OPERATION_ID,
        operation_version=_OPERATION_VERSION,
        package_version=package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        input_object_ids=(request.data.object_id,),
        input_artifact_hashes=tuple(
            item.sha256
            for item in sorted(request.data.artifacts.values(), key=lambda item: item.sha256)
        ),
        parameters_hash=request.semantic_hash,
        requested_backend="pmat",
        actual_backend="pmat",
        attempted_backends=("pmat",),
        software_versions=FrozenMap(
            {
                "organelleverse": package_version(),
                "pmat": environment.software_version or environment.version,
            }
        ),
        argv=stable_argv,
        started_at=outcome.started_at,
        finished_at=outcome.finished_at,
        duration_seconds=outcome.duration_seconds,
        run_manifest_id=run_manifest_id,
    )


def _success_result(
    request: _PmatGraphRequest,
    environment: PreparedEnvironment,
    stable_argv: tuple[str, ...],
    outcome: CommandOutcome,
    outputs: tuple[BackendOutput, ...],
    primary_sequence_role: str,
    workspace: Path,
    destination: Path,
) -> OrganelleResult:
    manifest_outputs = tuple(
        ManifestArtifact(
            role=output.role,
            artifact=published_artifact(
                output.path,
                workspace=workspace,
                destination=destination,
                kind="assembly_output",
                format=output.format,
                media_type=output.media_type,
            ),
        )
        for output in outputs
    )
    manifest = _base_manifest(
        request,
        environment,
        stable_argv,
        outcome,
        stages=_stages(),
        outputs=manifest_outputs,
        primary_sequence_role=primary_sequence_role,
    )
    canonical, record = write_manifest_evidence(
        manifest, workspace=workspace, destination=destination
    )
    artifacts = (canonical, record, *(item.artifact for item in manifest_outputs))
    return OrganelleResult(
        operation_id=_OPERATION_ID,
        operation_version=_OPERATION_VERSION,
        scope=request.parameters.organelle,
        status="ok",
        artifacts=artifacts,
        provenance=_provenance(
            request, environment, stable_argv, outcome, manifest.run_manifest_id
        ),
    )


def _failed_result(
    request: _PmatGraphRequest,
    environment: PreparedEnvironment,
    stable_argv: tuple[str, ...],
    outcome: CommandOutcome,
    workspace: Path,
    destination: Path,
    *,
    code: str,
    message: str,
    failed_stage: str,
) -> OrganelleResult:
    manifest = _base_manifest(
        request,
        environment,
        stable_argv,
        outcome,
        stages=_stages(failed_stage=failed_stage),
        outputs=(),
    )
    canonical, record = write_manifest_evidence(
        manifest, workspace=workspace, destination=destination
    )
    return OrganelleResult(
        operation_id=_OPERATION_ID,
        operation_version=_OPERATION_VERSION,
        scope=request.parameters.organelle,
        status="failed",
        artifacts=(canonical, record),
        provenance=_provenance(
            request, environment, stable_argv, outcome, manifest.run_manifest_id
        ),
        errors=(
            ErrorDetail(
                code=code,
                message=message,
                suggested_action=FrozenMap({"failed_stage": failed_stage}),
            ),
        ),
    )


def _publish(
    result: OrganelleResult,
    outcome: CommandOutcome,
    workspace: Path,
    destination: Path,
) -> OrganelleResult:
    observations = AssemblyRunObservations(
        run_manifest_id=result.provenance.run_manifest_id or "" if result.provenance else "",
        started_at=outcome.started_at,
        finished_at=outcome.finished_at,
        duration_seconds=outcome.duration_seconds,
        workspace_path=str(workspace),
        output_path=str(destination),
        resolved_argv=outcome.argv,
        stdout_path=outcome.stdout_path,
        stderr_path=outcome.stderr_path,
    )
    observations_path = workspace / "observations.json"
    observations_path.write_text(
        json.dumps(observations.model_dump(mode="json"), sort_keys=True) + "\n"
    )
    evidence = list(result.artifacts)
    evidence.append(
        published_artifact(
            observations_path,
            workspace=workspace,
            destination=destination,
            kind="assembly_run_observations",
            format="json",
            media_type="application/json",
        )
    )
    for path_text, kind in (
        (outcome.stdout_path, "assembly_stdout_log"),
        (outcome.stderr_path, "assembly_stderr_log"),
    ):
        path = Path(path_text)
        if path.is_file():
            evidence.append(
                published_artifact(
                    path,
                    workspace=workspace,
                    destination=destination,
                    kind=kind,
                    format="log",
                    media_type="text/plain",
                )
            )
    result = result.model_copy(update={"artifacts": tuple(evidence)})
    save_contract(result, workspace / "result.json")
    try:
        publish_run(workspace, destination)
    except OrganelleExecutionError as error:
        raise OrganelleExecutionError(
            code="assembly.publication_failed",
            message="atomic PMAT graphBuild publication failed",
            details={"destination": str(destination), "reason": error.message},
        ) from error
    return result


def _load_reusable(destination: Path, request: _PmatGraphRequest) -> OrganelleResult:
    try:
        result = load_result(destination / "result.json")
        if result.operation_id != _OPERATION_ID or result.status != "ok":
            raise ValueError("existing result is not a successful PMAT graphBuild run")
        if result.provenance is None or result.provenance.parameters_hash != request.semantic_hash:
            raise ValueError("existing result parameters do not match")
        if result.provenance.input_object_ids != (request.data.object_id,):
            raise ValueError("existing result input does not match")
        verify_published_artifacts(destination, result.artifacts)
        record = AssemblyRunManifest.model_validate_json(
            (destination / "assembly_run_record.json").read_bytes()
        )
        if record.operation_id != _OPERATION_ID or record.input_data_id != request.data.object_id:
            raise ValueError("existing run manifest does not match")
        return result
    except Exception as error:
        raise OrganelleExecutionError(
            code="assembly.destination_conflict",
            message=f"existing PMAT graphBuild destination is not reusable: {error}",
            details={"destination": str(destination)},
        ) from error
