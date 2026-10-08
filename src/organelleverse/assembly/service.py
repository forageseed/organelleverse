"""Internal fixed-stage Oatk assembly execution service.

This is the only execution entry point for the Oatk HiFi slice. It runs the 14
fixed stages, enforces pre-start exception vs post-start failed-Result boundaries,
writes evidence (run manifest -> Genome -> Result), and atomically publishes the
output directory. Exact completed destinations are reused without re-running Oatk.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, Literal, cast

from organelleverse.assembly.backends.base import (
    AdapterContext,
    AssemblyAdapter,
    AssemblyCommand,
    ExpectedBackendResources,
    NormalizedAssemblyOutputs,
    PreparedBackendResources,
)
from organelleverse.assembly.backends.runtime import RUNTIMES, BackendRuntime
from organelleverse.assembly.contracts import (
    AssemblyInputPayload,
    AssemblyMethod,
    AssemblyRequest,
    effective_backend_parameters,
)
from organelleverse.assembly.data_contract import validate_assembly_data
from organelleverse.assembly.environment_capabilities import runtime_capabilities
from organelleverse.assembly.environment_resolver import EnvironmentResolver
from organelleverse.assembly.environment_versions import (
    environment_spec_for_version,
    resolve_backend_version,
)
from organelleverse.assembly.environments import (
    EnvironmentManager,
    PreparedEnvironment,
    PreparedProfile,
)
from organelleverse.assembly.execution import (
    CommandOutcome,
    CommandRunner,
    managed_prefix_environment,
)
from organelleverse.assembly.manifests import (
    AssemblyComponentIdentity as ManifestComponentIdentity,
)
from organelleverse.assembly.manifests import (
    AssemblyEnvironmentIdentity,
    AssemblyRunManifest,
    AssemblyRunObservations,
    AssemblyRunParameters,
    AssemblyStageOutcome,
    ManifestArtifact,
    RoutingEvidence,
)
from organelleverse.assembly.routing import (
    AssemblyRoute,
    RoutingContext,
    prepare_routing_context,
    resolve_backend,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import LineageRecord, OrganelleData
from organelleverse.core.errors import (
    OrganelleContractError,
    OrganelleDependencyError,
    OrganelleExecutionError,
    OrganelleInputError,
)
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import ErrorDetail, OrganelleResult
from organelleverse.core.serialization import load_result, save_contract
from organelleverse.runtime import managed_run_path, publish_run

__all__ = ["execute_assembly"]

_OPERATION_ID = "assembly.assemble"
_OPERATION_VERSION = "1.0"

# The fixed stage names, in execution order. Context preparation sits between
# common input validation and backend routing so coverage routing can consult
# streaming statistics and raise any missing-input signal before side effects.
_ALL_STAGES = (
    "validate_input",
    "prepare_routing_context",
    "route_backend",
    "resolve_parameters",
    "resolve_environment",
    "resolve_resources",
    "prepare_workspace",
    "execute_backend",
    "collect_outputs",
    "normalize_outputs",
    "validate_outputs",
    "write_run_manifest",
    "write_primary_genome",
    "write_result",
    "publish_output",
)


def execute_assembly(
    request: AssemblyRequest,
    *,
    environment_manager: EnvironmentManager | None = None,
    runner: CommandRunner | None = None,
    adapter: AssemblyAdapter | None = None,
    environment_resolver: EnvironmentResolver | None = None,
) -> OrganelleResult:
    """Run the fixed-stage assembly and return a published OrganelleResult."""
    completed: list[str] = []

    # validate_input: the common, backend-independent contract is checked first.
    payload = validate_assembly_data(request.data)
    completed.append("validate_input")

    # prepare_routing_context: streaming statistics and traceable evidence,
    # assembled before routing. May raise OrganelleNeedsInput when the released
    # runtime makes a coverage decision that needs a genome size. Read-only: no
    # workspace, environment, or backend is touched here.
    released_backend_ids = tuple(RUNTIMES)
    routing_context = prepare_routing_context(
        request,
        released_backend_ids=released_backend_ids,
    )
    completed.append("prepare_routing_context")

    # route_backend: ordered declarative rules over the released runtime set.
    route = resolve_backend(
        request.data,
        organelle=request.organelle,
        method=request.method,
        context=routing_context,
        released_backend_ids=released_backend_ids,
    )
    try:
        runtime = RUNTIMES.require(route.selected_backend)
    except OrganelleContractError as exc:
        raise OrganelleContractError(
            code="contract.operation_not_released",
            message=f"assembly backend {route.selected_backend} is not implemented",
        ) from exc
    selected_backend = runtime.backend_id
    payload = runtime.data_validator(request.data)
    completed.append("route_backend")

    # resolve_parameters
    effective = effective_backend_parameters(
        selected_backend,
        request.backend_parameters,
        payload=payload,
        organelle=request.organelle,
        taxon_group=request.taxon_group,
    )
    resolved_request_hash = request.resolved_semantic_hash(selected_backend)
    completed.append("resolve_parameters")

    routing_evidence = _routing_evidence(route, routing_context)
    native = runtime.environment_manager_factory
    manager = environment_manager or (native() if native is not None else EnvironmentManager())
    # a native runtime has no provider to resolve: its manager names the binary itself
    resolve_providers = (
        environment_manager is None and native is None
    ) or environment_resolver is not None
    active_runner = runner or CommandRunner()
    active_adapter = adapter if adapter is not None else runtime.adapter_factory()
    if active_adapter.backend_id != runtime.backend_id:
        raise OrganelleContractError(
            code="contract.assembly_backend_runtime_mismatch",
            message="injected adapter does not match the selected runtime",
            details={
                "selected_backend": runtime.backend_id,
                "adapter_backend": active_adapter.backend_id,
            },
        )
    destination = managed_run_path(_OPERATION_ID, f"sha256-{resolved_request_hash}")

    # Re-verify every declared input artifact hash BEFORE the reuse check, so a
    # changed input at the same path cannot silently reuse a prior result.
    _reverify_all_input_artifacts(request.data)
    provider_environment: PreparedEnvironment | None = None
    capability_contract = None
    resolved_version = None
    environment_spec = runtime.environment_spec
    if resolve_providers:
        resolved_version = resolve_backend_version(selected_backend, request.backend_version)
        capability_contract = runtime_capabilities(runtime, effective, resolved_version)
        resolution = (environment_resolver or EnvironmentResolver()).resolve(
            request=request,
            capability_contract=capability_contract,
            resolved_version=resolved_version,
            effective_parameters=effective,
        )
        if resolution.selected_provider is not None:
            provider_environment = manager.prepare_provider(resolution.selected_provider)
            expected_environment_digest: str | None = provider_environment.digest
        else:
            environment_spec = environment_spec_for_version(environment_spec, resolved_version)
            expected_environment_digest = None
    else:
        expected_environment_digest = manager.expected_environment_digest(environment_spec)
    expected_resources = runtime.resource_provider.expected(
        manager, environment_spec, request, payload
    )

    # Reuse check: if destination holds an exact matching result, return it.
    reusable = None
    if expected_environment_digest is not None:
        reusable = _load_reusable_result(
            destination,
            request=request,
            resolved_request_hash=resolved_request_hash,
            selected_backend=selected_backend,
            expected_route_reason_code=route.rule_id,
            expected_environment_digest=expected_environment_digest,
            expected_resources=expected_resources,
        )
    if reusable is not None:
        return reusable

    workspace = _temporary_sibling(destination)
    outcome: CommandOutcome | None = None
    command: AssemblyCommand | None = None
    context: AdapterContext | None = None
    environment: PreparedEnvironment | None = None
    profile: PreparedProfile | None = None
    prepared_resources: PreparedBackendResources | None = None
    extra_stages: list[str] = []
    try:
        # resolve_environment
        environment = provider_environment or manager.prepare(
            environment_spec,
            policy="require"
            if request.environment_source == "managed" and not resolve_providers
            else "ensure",
        )
        if (
            resolve_providers
            and provider_environment is None
            and capability_contract is not None
            and resolved_version is not None
            and type(manager) is EnvironmentManager
        ):
            environment = manager.register_prepared_provider(
                environment,
                capability_contract,
                requested_source=request.environment_source,
                resolved_version=resolved_version.version,
            )
        if (
            expected_environment_digest is not None
            and environment.digest != expected_environment_digest
        ):
            raise RuntimeError("prepared environment identity differs from its static identity")
        extra_stages.append("resolve_environment")

        # resolve_resources
        prepared_resources = runtime.resource_provider.prepare(
            manager,
            environment_spec,
            request,
            payload,
            environment,
        )
        if not _resources_match(prepared_resources, expected_resources):
            raise RuntimeError("prepared resource identity differs from its expected identity")
        extra_stages.append("resolve_resources")
        profile = _profile_from_resources(prepared_resources, request.organelle)

        # prepare_workspace
        context = _adapter_context(
            request, payload, route, environment, prepared_resources, workspace, effective
        )
        active_adapter.preflight(context)
        # The backend's own directory creation is not recursive: it can create the
        # final path component of its output prefix but fails with "No such file
        # or directory" if the parent is also missing. Pre-create the backend
        # execution directory the adapter's build_command() targets.
        (workspace / "backend" / runtime.backend_id).mkdir(parents=True, exist_ok=True)
        extra_stages.append("prepare_workspace")

        command = active_adapter.build_command(context)
        # execute_backend
        outcome = active_runner.run(
            command.resolved_argv,
            stage="execute_backend",
            cwd=workspace,
            timeout_seconds=request.timeout_seconds or 86400,
            stdout_path=workspace / "stdout.log",
            stderr_path=workspace / "stderr.log",
            env=_execution_environment(environment),
        )
        extra_stages.append("execute_backend")
        if not outcome.started:
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message=f"{runtime.backend_id} could not start in the verified environment",
                details=outcome.model_dump(mode="json"),
            )
        if outcome.termination != "exit" or outcome.returncode != 0:
            failed_stages = _stage_list(
                completed + extra_stages + ["execute_backend"], "execute_backend"
            )
            result = _build_failed_result(
                context,
                destination,
                command,
                outcome,
                resolved_request_hash,
                environment,
                runtime,
                prepared_resources,
                failed_stages,
                error_code="assembly.execution_failed",
                error_message=f"{runtime.backend_id} exited with termination={outcome.termination} "
                f"returncode={outcome.returncode}",
                routing_evidence=routing_evidence,
            )
            return _publish(
                workspace,
                destination,
                result,
                context,
                command,
                outcome,
                environment,
                profile,
                resolved_request_hash,
                completed + extra_stages,
            )

        raw = active_adapter.collect_outputs(context)
        extra_stages.append("collect_outputs")
        normalized = active_adapter.normalize(context, raw, workspace / "normalized")
        extra_stages.append("normalize_outputs")
        extra_stages.append("validate_outputs")

        result = _build_success_result(
            context,
            destination,
            command,
            outcome,
            normalized,
            resolved_request_hash,
            environment,
            runtime,
            prepared_resources,
            completed + extra_stages,
            routing_evidence=routing_evidence,
        )
        return _publish(
            workspace,
            destination,
            result,
            context,
            command,
            outcome,
            environment,
            profile,
            resolved_request_hash,
            completed + extra_stages,
        )
    except OrganelleExecutionError as error:
        if error.code in {"assembly.destination_conflict", "assembly.publication_failed"}:
            _remove_workspace(workspace)
            raise
        if outcome is not None and outcome.started and context is not None:
            # Determine which stage the error occurred in based on how far execution progressed.
            all_done = completed + extra_stages
            if "collect_outputs" not in all_done:
                failed_stage = "collect_outputs"
            elif "normalize_outputs" not in all_done:
                failed_stage = "normalize_outputs"
            elif "validate_outputs" not in all_done:
                failed_stage = "validate_outputs"
            else:
                failed_stage = "execute_backend"
            failed_stages = _stage_list(
                all_done + (["execute_backend"] if "execute_backend" not in all_done else []),
                "execute_backend",
            )
            if all(stage.stage != failed_stage for stage in failed_stages):
                failed_stages.append(AssemblyStageOutcome(stage=failed_stage, status="failed"))
            result = _build_failed_result(
                context,
                destination,
                command,
                outcome,
                resolved_request_hash,
                environment,
                runtime,
                prepared_resources,
                failed_stages,
                error_code=error.code,
                error_message=error.message,
                failed_stage=failed_stage,
                error_details=error.details,
                routing_evidence=routing_evidence,
            )
            return _publish(
                workspace,
                destination,
                result,
                context,
                command,
                outcome,
                environment,
                profile,
                resolved_request_hash,
                completed + extra_stages,
            )
        _remove_workspace(workspace)
        raise
    except BaseException:
        _remove_workspace(workspace)
        raise


# ---------------------------------------------------------------------------
# Input verification
# ---------------------------------------------------------------------------


def _reverify_all_input_artifacts(data: OrganelleData) -> None:
    """Re-hash every artifact on the data object before reuse check or execution."""
    for artifact in data.artifacts.values():
        _reverify(artifact)


def _adapter_context(
    request: AssemblyRequest,
    payload: AssemblyInputPayload,
    route: AssemblyRoute,
    environment: PreparedEnvironment,
    prepared_resources: PreparedBackendResources,
    workspace: Path,
    effective: dict[str, object],
) -> AdapterContext:
    # The common contract already rejects unused roles; retain every verified role
    # so backend adapters can consume declared auxiliary artifacts without a second
    # path channel.
    real_artifacts = {
        role: _reverify(artifact) for role, artifact in request.data.artifacts.items()
    }
    return AdapterContext(
        request=request,
        payload=payload,
        route=route,
        environment=environment,
        profile=_profile_from_resources(prepared_resources, request.organelle),
        resources=prepared_resources,
        workspace=workspace,
        input_artifacts=FrozenMap.from_items(real_artifacts),
        effective_backend_parameters=FrozenMap(effective),
    )


def _routing_evidence(
    route: AssemblyRoute,
    routing_context: RoutingContext,
) -> RoutingEvidence:
    """Project the routing context into the stable manifest evidence record."""
    total_bases = routing_context.total_bases
    genome_size_bp = routing_context.genome_size_bp
    multiplier = routing_context.threshold_multiplier
    threshold_numerator: int | None = None
    threshold_denominator: int | None = None
    coverage_display: str | None = None
    # RoutingEvidence derives the display from bases and genome size whatever the rule: an
    # explicitly chosen backend given a genome size (PMAT's -g) has no multiplier but still
    # must carry it, or the evidence record is refused and the run never starts.
    if total_bases is not None and genome_size_bp is not None:
        coverage_display = f"{total_bases / genome_size_bp:.2f}x"
        if multiplier is not None:
            threshold_numerator = total_bases
            threshold_denominator = multiplier * genome_size_bp
    accession: str | None = None
    evidence = routing_context.genome_size_evidence
    if evidence is not None and evidence.assembly_accession is not None:
        accession = evidence.assembly_accession
    return RoutingEvidence(
        rule_id=route.rule_id,
        taxon_group=routing_context.taxon_group,
        total_bases=total_bases,
        genome_size_bp=genome_size_bp,
        threshold_multiplier=multiplier,
        threshold_numerator=threshold_numerator,
        threshold_denominator=threshold_denominator,
        coverage_display=coverage_display,
        genome_size_accession=accession,
    )


def _profile_from_resources(
    resources: PreparedBackendResources,
    organelle: Literal["mitochondrion", "plastid"],
) -> PreparedProfile | None:
    """Reconstruct a PreparedProfile from the provider's prepared resources."""
    hmm_artifact = resources.artifacts.get("hmm_profiles")
    if hmm_artifact is None:
        return None
    profile_components = tuple(
        component for component in resources.components if component.category == "profile"
    )
    if len(profile_components) != 1:
        return None
    component = profile_components[0]
    source: Literal["managed", "user_supplied"] = (
        "user_supplied" if resources.model_hashes.get("hmm_profile") is not None else "managed"
    )
    return PreparedProfile(
        source=source,
        target=organelle,
        path=Path(hmm_artifact.uri),
        artifact=hmm_artifact,
        component=component,
    )


def _resources_match(
    prepared: PreparedBackendResources,
    expected: ExpectedBackendResources,
) -> bool:
    return (
        prepared.components == expected.components
        and prepared.model_hashes == expected.model_hashes
        and prepared.database_hashes == expected.database_hashes
    )


def _reverify(artifact: ArtifactRef) -> ArtifactRef:
    """Re-hash the artifact at its URI and verify it matches the declared sha256."""
    path = Path(artifact.uri)
    if not path.is_file():
        raise OrganelleExecutionError(
            code="assembly.output_incomplete",
            message=f"declared input artifact is missing: {artifact.uri}",
            details={"uri": artifact.uri},
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != artifact.sha256:
        raise OrganelleExecutionError(
            code="assembly.output_incomplete",
            message="declared input artifact hash no longer matches its content",
            details={
                "uri": artifact.uri,
                "declared_sha256": artifact.sha256,
                "actual_sha256": actual,
            },
        )
    return artifact


def _execution_environment(environment: PreparedEnvironment) -> dict[str, str]:
    return managed_prefix_environment(environment.prefix)


def _temporary_sibling(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    )


def _remove_workspace(workspace: Path) -> None:
    shutil.rmtree(workspace, ignore_errors=True)


def _stage_list(names: list[str], started_stage: str) -> list[AssemblyStageOutcome]:
    """Build the list of AssemblyStageOutcome from completed stage names."""
    outcomes: list[AssemblyStageOutcome] = []
    for name in _ALL_STAGES:
        if name not in names:
            break
        if name == started_stage:
            outcomes.append(
                AssemblyStageOutcome(
                    stage=name, status="ok", process_started=True, exit_code=0, termination="exit"
                )
            )
        else:
            outcomes.append(AssemblyStageOutcome(stage=name, status="ok"))
    return outcomes


def _package_version() -> str:
    try:
        return _pkg_version("organelleverse")
    except Exception:
        return "0.0.0"


def _provenance(
    *,
    resolved_request_hash: str,
    request: AssemblyRequest,
    command: AssemblyCommand | None,
    environment: PreparedEnvironment | None,
    runtime: BackendRuntime,
    prepared_resources: PreparedBackendResources | None,
    run_manifest_id: str | None,
    started_at: datetime,
    finished_at: datetime,
) -> ResultProvenance:
    requested_backend = request.method if request.method != "auto" else ""
    backend_id = runtime.backend_id
    return ResultProvenance(
        operation_id=_OPERATION_ID,
        operation_version=_OPERATION_VERSION,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        input_object_ids=(request.data.object_id,),
        input_artifact_hashes=tuple(
            a.sha256 for a in sorted(request.data.artifacts.values(), key=lambda x: x.sha256)
        ),
        parameters_hash=resolved_request_hash,
        requested_backend=requested_backend,
        actual_backend=backend_id,
        attempted_backends=(backend_id,),
        software_versions=FrozenMap(
            {
                "organelleverse": _package_version(),
                backend_id: (environment.software_version or environment.version)
                if environment
                else "",
            }
        ),
        model_hashes=prepared_resources.model_hashes
        if prepared_resources is not None
        else FrozenMap(),
        database_hashes=prepared_resources.database_hashes
        if prepared_resources is not None
        else FrozenMap(),
        argv=command.stable_argv if command else (),
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=(finished_at - started_at).total_seconds(),
        run_manifest_id=run_manifest_id,
    )


def _manifest_input_artifacts(
    data_artifacts: FrozenMap[ArtifactRef],
    prepared_resources: PreparedBackendResources | None,
) -> tuple[ManifestArtifact, ...]:
    """Every artifact role referenced by stable_argv must appear here, including
    the resolved backend resources — for Oatk this is the HMM profile, whether it
    came from the managed OatkDB or a user-supplied override. request.data.artifacts
    alone omits the managed case, since the caller never declared an hmm_profiles
    artifact."""
    combined: dict[str, ArtifactRef] = dict(data_artifacts.items())
    if prepared_resources is not None:
        for role, artifact in prepared_resources.artifacts.items():
            combined[role] = artifact
    return tuple(
        ManifestArtifact(role=role, artifact=artifact)
        for role, artifact in sorted(combined.items())
    )


def _build_run_manifest(
    request: AssemblyRequest,
    environment: PreparedEnvironment,
    runtime: BackendRuntime,
    prepared_resources: PreparedBackendResources | None,
    command: AssemblyCommand,
    stages: list[AssemblyStageOutcome],
    outcome: CommandOutcome,
    normalized: NormalizedAssemblyOutputs | None,
    workspace: Path,
    destination: Path,
    route_reason_code: str,
    payload: AssemblyInputPayload,
    routing_evidence: RoutingEvidence | None = None,
) -> AssemblyRunManifest:
    backend_id = runtime.backend_id
    effective = effective_backend_parameters(
        backend_id,
        request.backend_parameters,
        payload=payload,
        organelle=request.organelle,
        taxon_group=request.taxon_group,
    )
    input_artifacts = _manifest_input_artifacts(request.data.artifacts, prepared_resources)
    output_artifacts: list[ManifestArtifact] = []
    primary_sequence_role = "assembly_fasta"
    if normalized is not None:
        primary_sequence_role = normalized.primary_sequence_role
        for output in normalized.outputs:
            artifact = _published_artifact(
                output.path,
                workspace=workspace,
                destination=destination,
                kind=output.role,
                format=output.format,
                media_type=output.media_type,
            )
            output_artifacts.append(ManifestArtifact(role=output.role, artifact=artifact))
    return AssemblyRunManifest(
        input_data_id=request.data.object_id,
        input_artifacts=input_artifacts,
        input_payload=payload,
        organelle=request.organelle,
        parameters=AssemblyRunParameters(
            threads=request.threads,
            memory_gb=request.memory_gb,
            timeout_seconds=request.timeout_seconds,
            environment_source=request.environment_source,
            backend_version=request.backend_version,
            environment_hint=request.environment_hint,
            backend_parameters=cast("Any", effective),
        ),
        requested_method=request.method,
        selected_backend=backend_id,
        route_reason_code=route_reason_code,
        routing_evidence=routing_evidence,
        environment=AssemblyEnvironmentIdentity(
            carrier=environment.carrier,
            digest=environment.digest,
            platform=environment.platform,
        ),
        components=(
            ManifestComponentIdentity(
                category="software",
                name=backend_id,
                version=environment.version,
            ),
            *(prepared_resources.components if prepared_resources is not None else ()),
        ),
        primary_sequence_role=primary_sequence_role,
        stable_argv=command.stable_argv,
        stages=tuple(stages),
        process_exit_code=outcome.returncode,
        outputs=tuple(output_artifacts),
    )


def _write_manifest_evidence(
    manifest: AssemblyRunManifest,
    *,
    workspace: Path,
    destination: Path,
) -> tuple[ArtifactRef, ArtifactRef]:
    """Write the path-independent canonical manifest plus its loadable record."""
    manifest_path = workspace / "assembly_run_manifest.json"
    manifest_path.write_bytes(manifest.canonical_bytes())
    manifest_artifact = _published_artifact(
        manifest_path,
        workspace=workspace,
        destination=destination,
        kind="assembly_run_manifest",
        format="json",
        media_type="application/json",
    )
    if manifest_artifact != manifest.as_artifact(manifest_artifact.uri):
        raise RuntimeError("canonical run manifest artifact identity is inconsistent")

    record_path = workspace / "assembly_run_record.json"
    record_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    record_artifact = _published_artifact(
        record_path,
        workspace=workspace,
        destination=destination,
        kind="assembly_run_record",
        format="json",
        media_type="application/json",
    )
    return manifest_artifact, record_artifact


def _build_success_result(
    context: AdapterContext,
    destination: Path,
    command: AssemblyCommand,
    outcome: CommandOutcome,
    normalized: NormalizedAssemblyOutputs,
    resolved_request_hash: str,
    environment: PreparedEnvironment,
    runtime: BackendRuntime,
    prepared_resources: PreparedBackendResources,
    completed_stages: list[str],
    routing_evidence: RoutingEvidence | None = None,
) -> OrganelleResult:
    started_at = outcome.started_at
    finished_at = outcome.finished_at
    stages = _stage_list(
        [
            *completed_stages,
            "write_run_manifest",
            "write_primary_genome",
            "write_result",
            "publish_output",
        ],
        "execute_backend",
    )
    manifest = _build_run_manifest(
        context.request,
        environment,
        runtime,
        prepared_resources,
        command,
        stages,
        outcome,
        normalized,
        context.workspace,
        destination,
        context.route.rule_id,
        context.payload,
        routing_evidence,
    )
    manifest_artifact, record_artifact = _write_manifest_evidence(
        manifest,
        workspace=context.workspace,
        destination=destination,
    )

    # Build Genome from the normalized primary sequence role.
    primary_sequence = normalized.primary_sequence
    sequence_artifact = _published_artifact(
        primary_sequence.path,
        workspace=context.workspace,
        destination=destination,
        kind="sequence",
        format=primary_sequence.format,
        media_type=primary_sequence.media_type,
    )
    genome = OrganelleGenome(
        organelle=context.request.organelle,
        sequence=sequence_artifact,
        lineage=(
            LineageRecord(
                parent_object_ids=(context.request.data.object_id,),
                operation_id=_OPERATION_ID,
                operation_version=_OPERATION_VERSION,
                parameters_hash=resolved_request_hash,
            ),
        ),
        source_manifests=(manifest_artifact,),
    )
    genome_path = context.workspace / "primary_genome.json"
    save_contract(genome, genome_path)
    genome_artifact = _published_artifact(
        genome_path,
        workspace=context.workspace,
        destination=destination,
        kind="primary_genome_manifest",
        format="json",
        media_type="application/json",
    )

    provenance = _provenance(
        resolved_request_hash=resolved_request_hash,
        request=context.request,
        command=command,
        environment=environment,
        runtime=runtime,
        prepared_resources=prepared_resources,
        run_manifest_id=manifest.run_manifest_id,
        started_at=started_at,
        finished_at=finished_at,
    )

    artifacts: list[ArtifactRef] = [manifest_artifact, record_artifact, genome_artifact]
    for output in normalized.outputs:
        artifacts.append(
            _published_artifact(
                output.path,
                workspace=context.workspace,
                destination=destination,
                kind="assembly_output",
                format=output.format,
                media_type=output.media_type,
            )
        )

    result = OrganelleResult(
        operation_id=_OPERATION_ID,
        operation_version=_OPERATION_VERSION,
        scope=context.request.organelle,
        status="ok",
        artifacts=tuple(artifacts),
        provenance=provenance,
    )
    # copy genome manifest and result into workspace root for publication
    save_contract(result, context.workspace / "result.json")
    return result


def _build_failed_result(
    context: AdapterContext,
    destination: Path,
    command: AssemblyCommand | None,
    outcome: CommandOutcome,
    resolved_request_hash: str,
    environment: PreparedEnvironment | None,
    runtime: BackendRuntime,
    prepared_resources: PreparedBackendResources | None,
    stages: list[AssemblyStageOutcome],
    *,
    error_code: str,
    error_message: str,
    failed_stage: str = "execute_backend",
    error_details: Any = None,
    routing_evidence: RoutingEvidence | None = None,
) -> OrganelleResult:
    started_at = outcome.started_at
    finished_at = outcome.finished_at
    backend_id = runtime.backend_id
    # Mark the correct stage as failed based on where the error occurred.
    failed_stages: list[AssemblyStageOutcome] = []
    for s in stages:
        if s.stage == failed_stage:
            if s.stage == "execute_backend":
                failed_stages.append(
                    AssemblyStageOutcome(
                        stage="execute_backend",
                        status="failed",
                        process_started=True,
                        exit_code=outcome.returncode,
                        termination=outcome.termination,
                    )
                )
            else:
                failed_stages.append(AssemblyStageOutcome(stage=s.stage, status="failed"))
        else:
            failed_stages.append(s)

    manifest = AssemblyRunManifest(
        input_data_id=context.request.data.object_id,
        input_artifacts=_manifest_input_artifacts(
            context.request.data.artifacts, prepared_resources
        ),
        input_payload=context.payload,
        organelle=context.request.organelle,
        parameters=AssemblyRunParameters(
            threads=context.request.threads,
            memory_gb=context.request.memory_gb,
            timeout_seconds=context.request.timeout_seconds,
            environment_source=context.request.environment_source,
            backend_version=context.request.backend_version,
            environment_hint=context.request.environment_hint,
            backend_parameters=cast(
                "Any",
                effective_backend_parameters(
                    backend_id,
                    context.request.backend_parameters,
                    payload=context.payload,
                    organelle=context.request.organelle,
                    taxon_group=context.request.taxon_group,
                ),
            ),
        ),
        requested_method=context.request.method,
        selected_backend=backend_id,
        route_reason_code=context.route.rule_id,
        routing_evidence=routing_evidence,
        environment=AssemblyEnvironmentIdentity(
            carrier=environment.carrier if environment else "conda",
            digest=environment.digest if environment else "sha256:" + "0" * 64,
            platform=environment.platform if environment else "unknown",
        ),
        components=(
            ManifestComponentIdentity(
                category="software",
                name=backend_id,
                version=environment.version if environment else "",
            ),
            *(prepared_resources.components if prepared_resources is not None else ()),
        ),
        stable_argv=command.stable_argv if command else (),
        stages=tuple(failed_stages),
        process_exit_code=outcome.returncode,
        outputs=(),
    )
    manifest_artifact, record_artifact = _write_manifest_evidence(
        manifest,
        workspace=context.workspace,
        destination=destination,
    )

    provenance = _provenance(
        resolved_request_hash=resolved_request_hash,
        request=context.request,
        command=command,
        environment=environment,
        runtime=runtime,
        prepared_resources=prepared_resources,
        run_manifest_id=manifest.run_manifest_id,
        started_at=started_at,
        finished_at=finished_at,
    )

    return OrganelleResult(
        operation_id=_OPERATION_ID,
        operation_version=_OPERATION_VERSION,
        scope=context.request.organelle,
        status="failed",
        artifacts=(manifest_artifact, record_artifact),
        provenance=provenance,
        errors=(
            ErrorDetail(
                code=error_code,
                message=error_message,
                details=FrozenMap(error_details) if error_details is not None else FrozenMap(),
                suggested_action=FrozenMap({"failed_stage": failed_stage}),
            ),
        ),
    )


def _publish(
    workspace: Path,
    destination: Path,
    result: OrganelleResult,
    context: AdapterContext | None,
    command: AssemblyCommand | None,
    outcome: CommandOutcome,
    environment: PreparedEnvironment | None,
    profile: PreparedProfile | None,
    resolved_request_hash: str,
    completed_stages: list[str],
) -> OrganelleResult:
    # Write observations
    observations = AssemblyRunObservations(
        run_manifest_id=(result.provenance.run_manifest_id or "") if result.provenance else "",
        started_at=outcome.started_at,
        finished_at=outcome.finished_at,
        duration_seconds=outcome.duration_seconds,
        workspace_path=str(workspace),
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
        _published_artifact(
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
                _published_artifact(
                    path,
                    workspace=workspace,
                    destination=destination,
                    kind=kind,
                    format="log",
                    media_type="text/plain",
                )
            )
    result = result.model_copy(update={"artifacts": tuple(evidence)})

    # Rewrite result.json with the final published evidence set.
    result_path = workspace / "result.json"
    save_contract(result, result_path)
    try:
        publish_run(workspace, destination)
    except OrganelleInputError as error:
        raise OrganelleExecutionError(
            code="assembly.destination_conflict",
            message="assembly destination appeared before atomic publication",
            details={"destination": str(destination), "reason": error.message},
        ) from error
    except OrganelleExecutionError as error:
        raise OrganelleExecutionError(
            code="assembly.publication_failed",
            message="atomic assembly publication failed",
            details={
                "destination": str(destination),
                "reason": error.message,
            },
        ) from error
    return result


def _require_manifest_matches_request(
    manifest: AssemblyRunManifest,
    *,
    request: AssemblyRequest,
    selected_backend: AssemblyMethod,
    expected_route_reason_code: str,
    payload: AssemblyInputPayload,
    expected_resources: ExpectedBackendResources,
) -> None:
    expected_parameters = AssemblyRunParameters(
        threads=request.threads,
        memory_gb=request.memory_gb,
        timeout_seconds=request.timeout_seconds,
        environment_source=request.environment_source,
        backend_version=request.backend_version,
        environment_hint=request.environment_hint,
        backend_parameters=cast(
            "Any",
            effective_backend_parameters(
                selected_backend,
                request.backend_parameters,
                payload=payload,
                organelle=request.organelle,
                taxon_group=request.taxon_group,
            ),
        ),
    )
    if manifest.input_data_id != request.data.object_id:
        raise ValueError("input data identity mismatch")
    if manifest.organelle != request.organelle:
        raise ValueError("organelle mismatch")
    if manifest.parameters != expected_parameters:
        raise ValueError("run manifest parameters do not match the current request")
    if manifest.requested_method != request.method:
        raise ValueError("requested method mismatch")
    if manifest.selected_backend != selected_backend:
        raise ValueError("selected_backend mismatch")
    if manifest.route_reason_code != expected_route_reason_code:
        raise ValueError("route reason mismatch")

    manifest_inputs = {item.role: item.artifact for item in manifest.input_artifacts}
    expected_roles = set(request.data.artifacts) | set(expected_resources.artifact_roles)
    if set(manifest_inputs) != expected_roles:
        raise ValueError("run manifest input roles do not match the current request")
    for role, artifact in request.data.artifacts.items():
        if role in expected_resources.artifact_roles:
            continue
        if not _artifact_semantic_identity_match(artifact, manifest_inputs[role]):
            raise ValueError(f"run manifest input {role!r} does not match the current request")


def _require_provenance_matches_request(
    result: OrganelleResult,
    *,
    request: AssemblyRequest,
    manifest: AssemblyRunManifest,
    resolved_request_hash: str,
) -> None:
    provenance = result.provenance
    if provenance is None:
        raise ValueError("missing provenance")
    expected_requested_backend = request.method if request.method != "auto" else ""
    expected_input_hashes = tuple(
        artifact.sha256
        for artifact in sorted(request.data.artifacts.values(), key=lambda item: item.sha256)
    )
    if (
        result.operation_id != manifest.operation_id
        or result.operation_version != _OPERATION_VERSION
    ):
        raise ValueError("result operation identity does not match the run manifest")
    if result.scope != request.organelle:
        raise ValueError("result scope does not match the current request")
    if provenance.operation_id != manifest.operation_id:
        raise ValueError("provenance operation does not match the run manifest")
    if provenance.operation_version != _OPERATION_VERSION:
        raise ValueError("provenance operation version mismatch")
    if provenance.input_object_ids != (request.data.object_id,):
        raise ValueError("provenance input object does not match the current request")
    if provenance.input_artifact_hashes != expected_input_hashes:
        raise ValueError("provenance input artifacts do not match the current request")
    if provenance.parameters_hash != resolved_request_hash:
        raise ValueError("provenance parameters do not match the current request")
    if provenance.requested_backend != expected_requested_backend:
        raise ValueError("provenance requested backend mismatch")
    if provenance.actual_backend != manifest.selected_backend:
        raise ValueError("provenance actual backend mismatch")
    if provenance.attempted_backends != (manifest.selected_backend,):
        raise ValueError("provenance attempted backends mismatch")
    if provenance.argv != manifest.stable_argv:
        raise ValueError("provenance argv does not match the run manifest")


def _load_reusable_result(
    destination: Path,
    *,
    request: AssemblyRequest,
    resolved_request_hash: str,
    selected_backend: AssemblyMethod,
    expected_route_reason_code: str,
    expected_environment_digest: str,
    expected_resources: ExpectedBackendResources,
) -> OrganelleResult | None:
    if not destination.exists():
        return None
    try:
        result_path = destination / "result.json"
        manifest_path = destination / "assembly_run_manifest.json"
        record_path = destination / "assembly_run_record.json"
        observations_path = destination / "observations.json"
        genome_path = destination / "primary_genome.json"
        stdout_path = destination / "stdout.log"
        stderr_path = destination / "stderr.log"
        required_paths = (
            result_path,
            manifest_path,
            record_path,
            observations_path,
            genome_path,
            stdout_path,
            stderr_path,
        )
        if any(not path.is_file() for path in required_paths):
            # A FAILED prior run legitimately lacks the publication set
            # (no primary genome, no result.json). It must not block a retry
            # of the same request: a scientific failure is a real outcome,
            # but "same failed destination forever" turns one bad read set
            # (or one transient backend error) into a permanent wedge. The
            # stale failed tree is archived aside so atomic publication has
            # a clear destination, exactly as if the run had never happened.
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("process_exit_code") not in (0, None):
                import shutil
                import time as _time

                archived = destination.with_name(
                    f"{destination.name}.failed-{int(_time.time() * 1000)}"
                )
                shutil.move(str(destination), str(archived))
                return None  # prior run failed: allow a fresh attempt
            raise ValueError("published evidence set is incomplete")
        result = load_result(result_path)
        manifest = AssemblyRunManifest.model_validate_json(record_path.read_bytes())
        if manifest_path.read_bytes() != manifest.canonical_bytes():
            raise ValueError("canonical run manifest bytes do not match the loadable record")
        if result.status != "ok":
            raise ValueError("destination result is not successful")
        if result.provenance is None:
            raise ValueError("missing provenance")
        if result.provenance.parameters_hash != resolved_request_hash:
            raise ValueError("resolved request hash mismatch")
        payload = validate_assembly_data(request.data)
        _require_manifest_matches_request(
            manifest,
            request=request,
            selected_backend=selected_backend,
            expected_route_reason_code=expected_route_reason_code,
            payload=payload,
            expected_resources=expected_resources,
        )
        if manifest.environment.digest != expected_environment_digest:
            raise ValueError("environment digest mismatch")
        profile_components = tuple(
            component for component in manifest.components if component.category == "profile"
        )
        if profile_components != expected_resources.components:
            raise ValueError("profile identity mismatch")
        if result.provenance.run_manifest_id != manifest.run_manifest_id:
            raise ValueError("run manifest id mismatch")
        _require_provenance_matches_request(
            result,
            request=request,
            manifest=manifest,
            resolved_request_hash=resolved_request_hash,
        )

        required_result_kinds = (
            "assembly_run_manifest",
            "assembly_run_record",
            "primary_genome_manifest",
            "assembly_run_observations",
            "assembly_stdout_log",
            "assembly_stderr_log",
        )
        evidence_by_kind = _unique_result_evidence(result, required_result_kinds)

        # Verify every published output artifact's SHA256 still matches its file.
        _verify_published_artifacts(destination, result.artifacts)

        canonical_artifact = evidence_by_kind["assembly_run_manifest"]
        expected_canonical_artifact = manifest.as_artifact(str(manifest_path))
        if not (
            _artifact_content_and_location_match(
                canonical_artifact,
                expected_canonical_artifact,
            )
            and _artifact_semantic_identity_match(
                canonical_artifact,
                expected_canonical_artifact,
            )
        ):
            raise ValueError("result does not reference the canonical run manifest")
        _require_exact_artifact(
            evidence_by_kind["assembly_run_record"],
            record_path,
            kind="assembly_run_record",
            format="json",
            media_type="application/json",
        )
        _require_exact_artifact(
            evidence_by_kind["primary_genome_manifest"],
            genome_path,
            kind="primary_genome_manifest",
            format="json",
            media_type="application/json",
        )
        _require_exact_artifact(
            evidence_by_kind["assembly_run_observations"],
            observations_path,
            kind="assembly_run_observations",
            format="json",
            media_type="application/json",
        )
        _require_exact_artifact(
            evidence_by_kind["assembly_stdout_log"],
            stdout_path,
            kind="assembly_stdout_log",
            format="log",
            media_type="text/plain",
        )
        _require_exact_artifact(
            evidence_by_kind["assembly_stderr_log"],
            stderr_path,
            kind="assembly_stderr_log",
            format="log",
            media_type="text/plain",
        )

        observations = AssemblyRunObservations.model_validate_json(observations_path.read_bytes())
        if observations.run_manifest_id != manifest.run_manifest_id:
            raise ValueError("observations reference a different run manifest")

        # Verify the run manifest's output artifacts also match.
        for manifest_art in manifest.outputs:
            _verify_single_artifact(destination, manifest_art.artifact)

        # Verify the primary genome can be loaded and its source manifest hash matches.
        from organelleverse.core.serialization import load_genome

        genome = load_genome(genome_path)
        if genome.sequence is None:
            raise ValueError("primary genome sequence is missing")
        _verify_single_artifact(destination, genome.sequence)
        if len(genome.source_manifests) != 1 or not _artifact_content_and_location_match(
            genome.source_manifests[0],
            canonical_artifact,
        ):
            raise ValueError("primary genome does not reference the canonical run manifest")
        expected_lineage = (
            LineageRecord(
                parent_object_ids=(request.data.object_id,),
                operation_id=_OPERATION_ID,
                operation_version=_OPERATION_VERSION,
                parameters_hash=resolved_request_hash,
            ),
        )
        if genome.organelle != request.organelle or genome.lineage != expected_lineage:
            raise ValueError("primary genome lineage does not match the current request")
        primary_role = manifest.primary_sequence_role
        if primary_role is None:
            raise ValueError("released assembly manifest must declare a primary sequence role")
        if not _artifact_content_and_location_match(
            genome.sequence,
            _manifest_output(manifest, primary_role),
        ):
            raise ValueError(f"primary genome sequence does not match manifest {primary_role}")

        result_outputs = tuple(
            artifact for artifact in result.artifacts if artifact.kind == "assembly_output"
        )
        if len(result_outputs) != len(manifest.outputs):
            raise ValueError("result and manifest output counts differ")
        unmatched_outputs = list(result_outputs)
        for output in manifest.outputs:
            match_indexes = tuple(
                index
                for index, artifact in enumerate(unmatched_outputs)
                if _artifact_content_and_location_match(artifact, output.artifact)
            )
            if len(match_indexes) != 1:
                raise ValueError(f"manifest output {output.role!r} is not uniquely bound in result")
            unmatched_outputs.pop(match_indexes[0])
        if unmatched_outputs:
            raise ValueError("result contains outputs not bound by the run manifest")

        expected_model_hashes = dict(expected_resources.model_hashes.items())
        expected_database_hashes = dict(expected_resources.database_hashes.items())
        if expected_model_hashes:
            if result.provenance.model_hashes != FrozenMap(expected_model_hashes):
                raise ValueError("user profile provenance mismatch")
            if result.provenance.database_hashes:
                raise ValueError("user profile is incorrectly recorded as managed database")
        elif result.provenance.database_hashes != FrozenMap(expected_database_hashes):
            raise ValueError("managed profile provenance mismatch")

        return result
    except Exception as error:
        raise OrganelleExecutionError(
            code="assembly.destination_conflict",
            message=f"existing destination is not an exact reusable result: {error}",
            details={"destination": str(destination), "reason": str(error)},
        ) from error


def _unique_result_evidence(
    result: OrganelleResult,
    required_kinds: tuple[str, ...],
) -> dict[str, ArtifactRef]:
    evidence: dict[str, ArtifactRef] = {}
    for kind in required_kinds:
        matches = tuple(artifact for artifact in result.artifacts if artifact.kind == kind)
        if len(matches) != 1:
            raise ValueError(f"result requires exactly one {kind!r} artifact")
        evidence[kind] = matches[0]
    return evidence


def _require_exact_artifact(
    artifact: ArtifactRef,
    path: Path,
    *,
    kind: str,
    format: str,
    media_type: str,
) -> None:
    expected = ArtifactRef.from_path(path, kind=kind, format=format, media_type=media_type)
    if not (
        _artifact_content_and_location_match(artifact, expected)
        and _artifact_semantic_identity_match(artifact, expected)
    ):
        raise ValueError(f"{kind} ArtifactRef does not match its canonical published file")


def _manifest_output(manifest: AssemblyRunManifest, role: str) -> ArtifactRef:
    matches = tuple(item.artifact for item in manifest.outputs if item.role == role)
    if len(matches) != 1:
        raise ValueError(f"run manifest requires exactly one {role!r} output")
    return matches[0]


def _artifact_content_and_location_match(first: ArtifactRef, second: ArtifactRef) -> bool:
    return (
        Path(first.uri).resolve(strict=False) == Path(second.uri).resolve(strict=False)
        and first.sha256 == second.sha256
        and first.size_bytes == second.size_bytes
        and first.format == second.format
    )


def _artifact_semantic_identity_match(first: ArtifactRef, second: ArtifactRef) -> bool:
    return (
        first.kind == second.kind
        and first.format == second.format
        and first.media_type == second.media_type
        and first.sha256 == second.sha256
        and first.size_bytes == second.size_bytes
    )


def _verify_published_artifacts(destination: Path, artifacts: tuple[ArtifactRef, ...]) -> None:
    """Verify every result artifact's SHA256 matches its file content on disk."""
    for artifact in artifacts:
        _verify_single_artifact(destination, artifact)


def _verify_single_artifact(destination: Path, artifact: ArtifactRef) -> None:
    """Verify one published artifact without allowing paths outside the result."""
    destination = destination.resolve(strict=False)
    path = Path(artifact.uri)
    if not path.is_absolute():
        path = destination / path
    path = path.resolve(strict=False)
    if not path.is_relative_to(destination):
        raise ValueError(f"published artifact escapes destination: {artifact.uri}")
    if not path.is_file():
        raise ValueError(f"published artifact file missing: {artifact.uri}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != artifact.sha256:
        raise ValueError(
            f"published artifact hash mismatch for {artifact.uri}: "
            f"declared={artifact.sha256} actual={actual}"
        )


def _published_artifact(
    source: Path,
    *,
    workspace: Path,
    destination: Path,
    kind: str,
    format: str,
    media_type: str = "application/octet-stream",
) -> ArtifactRef:
    """Hash a workspace file while publishing an absolute destination URI."""
    try:
        relative = source.resolve(strict=True).relative_to(workspace.resolve(strict=True))
    except ValueError as error:
        raise RuntimeError(f"published artifact is outside the workspace: {source}") from error
    artifact = ArtifactRef.from_path(
        source,
        kind=kind,
        format=format,
        media_type=media_type,
    )
    published_uri = destination.expanduser().resolve(strict=False) / relative
    return artifact.model_copy(update={"uri": str(published_uri)})


# Package-internal primitives shared by continuation operations. Keeping these
# aliases here avoids duplicating the assembly service's evidence boundary while
# leaving the public ``organelleverse.assembly`` facade unchanged. Atomic
# publication itself now lives in ``organelleverse.runtime.publish_run``.
execution_environment = _execution_environment
package_version = _package_version
published_artifact = _published_artifact
remove_workspace = _remove_workspace
reverify_all_input_artifacts = _reverify_all_input_artifacts
temporary_sibling = _temporary_sibling
verify_published_artifacts = _verify_published_artifacts
write_manifest_evidence = _write_manifest_evidence
