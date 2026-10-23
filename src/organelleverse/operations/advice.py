"""Live, read-only advice attached to an agent response, outside result identity.

Candidates describe contract compatibility. Readiness additionally checks the
converter, files, metadata, target parameters, dependencies and granted effects
on the calling host. It does not rank scientific methods or execute a backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, cast

from pydantic import Field, JsonValue, TypeAdapter, ValidationError

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleError, OrganelleInputError
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OrganelleResult
from organelleverse.core.serialization import load_genome

from .dependencies import DependencyReport
from .spec import CoreKind, OperationSpec, SideEffect, StrictSpecModel

if TYPE_CHECKING:
    from .registry import OperationRegistry


class AdviceIssue(StrictSpecModel):
    code: str
    message: str


class InputConversion(StrictSpecModel):
    """An existing public reader and the exact artifact it uses."""

    callable: Literal["ov.io.read_assembly"]
    artifact_id: str
    artifact_uri: str
    required_metadata: tuple[str, ...] = ("species",)
    required_side_effects: tuple[SideEffect, ...] = (SideEffect.READ_FILES,)


class NextOperationAdvice(StrictSpecModel):
    operation_id: str
    reason_code: str
    ready: bool
    input_artifact_kind: str = ""
    input_conversion: InputConversion | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    missing_metadata: tuple[str, ...] = ()
    missing_parameters: tuple[str, ...] = ()
    dependencies: DependencyReport | None = None
    required_side_effects: tuple[SideEffect, ...] = ()
    missing_side_effects: tuple[SideEffect, ...] = ()
    issues: tuple[AdviceIssue, ...] = ()


class AgentAdvice(StrictSpecModel):
    schema_version: Literal["organelleverse.agent-advice.v1"] = "organelleverse.agent-advice.v1"
    execution_entrypoint: Literal["organelleverse.operations.adapters.invoke_next_json"] = (
        "organelleverse.operations.adapters.invoke_next_json"
    )
    result_id: str
    scope: str
    status: str
    summary_text: str
    limitations: tuple[AdviceIssue, ...] = ()
    next_operations: tuple[NextOperationAdvice, ...] = ()


_FLAG_LIMITATIONS = {
    "missing_core_genes": "Core genes were not recovered; this alone does not establish gene loss.",
    "cds_candidates_rejected": "Invalid CDS candidates were excluded from the annotation.",
    "sequence_only_orf_candidates": (
        "Sequence-only ORFs do not establish expression, RNA editing, splicing or CMS causality."
    ),
    "multi_sequence_output": "The assembly contains multiple sequences; topology needs separate evidence.",
}


def build_agent_advice(
    registry: OperationRegistry,
    result: OrganelleResult,
    *,
    metadata: Mapping[str, object] | None = None,
    parameters: Mapping[str, Mapping[str, object]] | None = None,
    granted_side_effects: tuple[str, ...] = (),
) -> AgentAdvice:
    """Discover strict candidates and explain why each can or cannot run here."""
    carried = {artifact.kind for artifact in result.artifacts}
    supplied = parameters or {}
    candidates: list[NextOperationAdvice] = []
    for target in registry.list():
        if target.operation_id == result.operation_id or result.scope not in target.organelle_types:
            continue
        for kind in target.result_artifact_inputs:
            if kind in carried:
                candidates.append(
                    _inspect_candidate(
                        registry,
                        result,
                        target,
                        kind,
                        metadata or {},
                        supplied.get(target.operation_id, {}),
                        granted_side_effects,
                    )
                )
    return AgentAdvice(
        result_id=result.object_id,
        scope=result.scope,
        status=result.status,
        summary_text=result.summary(),
        limitations=tuple(
            AdviceIssue(code=flag, message=_FLAG_LIMITATIONS[flag])
            for flag in result.flags
            if flag in _FLAG_LIMITATIONS
        ),
        next_operations=tuple(candidates),
    )


def _inspect_candidate(
    registry: OperationRegistry,
    result: OrganelleResult,
    target: OperationSpec,
    kind: str,
    metadata: Mapping[str, object],
    parameters: Mapping[str, object],
    grants: tuple[str, ...],
) -> NextOperationAdvice:
    issues: list[AdviceIssue] = []
    conversion: InputConversion | None = None
    missing_metadata: tuple[str, ...] = ()
    if result.status == "failed":
        issues.append(
            AdviceIssue(code="source.failed", message="A failed result cannot feed this operation.")
        )
    artifacts = [artifact for artifact in result.artifacts if artifact.kind == kind]
    if len(artifacts) != 1:
        issues.append(
            AdviceIssue(
                code="input.ambiguous_artifact",
                message=f"Expected one {kind}; found {len(artifacts)}.",
            )
        )
    supported = kind == "primary_genome_manifest" and target.input_kind is CoreKind.GENOME
    if not supported:
        issues.append(
            AdviceIssue(
                code="input.converter_unavailable",
                message="No public input converter is implemented for this declared artifact kind.",
            )
        )
    elif len(artifacts) == 1:
        artifact = artifacts[0]
        conversion = InputConversion(
            callable="ov.io.read_assembly",
            artifact_id=artifact.object_id,
            artifact_uri=artifact.uri,
        )
        species = metadata.get("species")
        if not isinstance(species, str) or not species.strip():
            missing_metadata = ("species",)
            issues.append(
                AdviceIssue(
                    code="input.species_required",
                    message="Provide species to ov.io.read_assembly before annotation.",
                )
            )
        if not result.operation_id.startswith("assembly."):
            issues.append(
                AdviceIssue(
                    code="input.not_an_assembly_result",
                    message="ov.io.read_assembly requires an assembly result.",
                )
            )
        if target.input_sequence:
            issues.append(
                AdviceIssue(
                    code="input.sequence_required",
                    message="This target needs a collection; the converter produces one Genome.",
                )
            )

    required = set(target.side_effects)
    if supported:
        required.add(SideEffect.READ_FILES)
    missing_effects = tuple(sorted(effect for effect in required if effect.value not in grants))
    if missing_effects:
        issues.append(
            AdviceIssue(
                code="permission.required",
                message="Grant the listed side effects before preparing input or executing the target.",
            )
        )
    # File inspection itself requires read permission. Never read a result's paths
    # merely because the caller was allowed to execute a different operation.
    if (
        conversion is not None
        and SideEffect.READ_FILES.value in grants
        and result.status != "failed"
    ):
        try:
            _validate_assembly_input(result, artifacts[0])
            if not missing_metadata:
                _convert_assembly(result, metadata)
        except (OrganelleError, OSError, ValueError) as error:
            code = (
                error.code
                if isinstance(error, OrganelleError)
                else "input.metadata_invalid"
                if isinstance(error, ValidationError)
                else "input.artifact_invalid"
            )
            issues.append(
                AdviceIssue(
                    code=code,
                    message="The input files or metadata do not satisfy the conversion contract.",
                )
            )

    missing_parameters: tuple[str, ...] = ()
    parameter_values: dict[str, JsonValue] = {}
    dependency_report: DependencyReport | None = None
    try:
        from jsonschema import Draft202012Validator

        schema = registry.parameter_schema(target.operation_id)
        properties = cast(dict[str, dict[str, JsonValue]], schema.get("properties", {}))
        parameter_values = {
            name: prop["default"] for name, prop in properties.items() if "default" in prop
        }
        try:
            supplied = TypeAdapter(dict[str, JsonValue]).validate_python(dict(parameters))
            parameter_values.update(supplied)
        except ValidationError:
            issues.append(
                AdviceIssue(
                    code="parameter.invalid", message="Target parameters must be JSON values."
                )
            )
        missing_parameters = tuple(
            sorted(
                name
                for name in cast(list[str], schema.get("required", []))
                if name not in parameter_values
            )
        )
        for error in Draft202012Validator(schema).iter_errors(parameter_values):  # pyright: ignore[reportUnknownMemberType]
            issues.append(
                AdviceIssue(
                    code="parameter.missing"
                    if error.validator == "required"
                    else "parameter.invalid",
                    message=error.message,
                )
            )
        dependency_report = registry.check_dependencies(target.operation_id)
        if not dependency_report.ready:
            issues.append(
                AdviceIssue(
                    code="dependency.not_ready",
                    message="Required dependencies are missing, incompatible or unverified on this host.",
                )
            )
    except OrganelleError as error:
        issues.append(AdviceIssue(code=error.code, message=error.message))
    return NextOperationAdvice(
        operation_id=target.operation_id,
        reason_code="result_artifact_feeds_input",
        ready=not issues,
        input_artifact_kind=kind,
        input_conversion=conversion,
        parameters=parameter_values,
        missing_metadata=missing_metadata,
        missing_parameters=missing_parameters,
        dependencies=dependency_report,
        required_side_effects=tuple(sorted(required)),
        missing_side_effects=missing_effects,
        issues=tuple(issues),
    )


def _validate_assembly_input(result: OrganelleResult, artifact: ArtifactRef) -> None:
    """Check existing artifact receipts and the Genome/FASTA input contract."""
    from organelleverse.io_verify import read_verified_artifact

    if artifact.format != "json":
        raise OrganelleInputError(
            code="input.manifest_format", message="The primary Genome manifest must be JSON."
        )
    observed = ArtifactRef.from_path(artifact.resolve(), kind=artifact.kind, format=artifact.format)
    if (observed.sha256, observed.size_bytes) != (artifact.sha256, artifact.size_bytes):
        raise OrganelleInputError(
            code="input.artifact_changed",
            message="The primary Genome manifest changed since publication.",
        )
    genome = load_genome(artifact.resolve())
    if genome.organelle != result.scope:
        raise OrganelleInputError(
            code="input.organelle_mismatch", message="Genome organelle and result scope disagree."
        )
    if genome.sequence is None or genome.sequence.format != "fasta":
        raise OrganelleInputError(
            code="input.sequence_required", message="The Genome needs a FASTA sequence artifact."
        )
    sequence = read_verified_artifact(
        genome.sequence.resolve(), kind=genome.sequence.kind, format="fasta"
    )
    if (sequence.sha256, sequence.size_bytes) != (
        genome.sequence.sha256,
        genome.sequence.size_bytes,
    ):
        raise OrganelleInputError(
            code="input.sequence_artifact_changed",
            message="The Genome sequence changed since publication.",
        )


def _convert_assembly(result: OrganelleResult, metadata: Mapping[str, object]) -> OrganelleGenome:
    from organelleverse.core.genome import OrganelleMetadata
    from organelleverse.io import read_assembly

    supplied = OrganelleMetadata.model_validate(dict(metadata))
    # A species-only continuation must retain the metadata carried by the
    # published Genome, exactly as the public reader does without overrides.
    overrides = supplied if supplied.model_fields_set - {"species"} else None
    return read_assembly(result, species=supplied.species, metadata=overrides)


def prepare_next_input(
    registry: OperationRegistry,
    result: OrganelleResult,
    operation_id: str,
    *,
    metadata: Mapping[str, object],
    parameters: Mapping[str, object],
    granted_side_effects: tuple[str, ...],
) -> OrganelleGenome:
    """Recheck advice and use the named public reader; never execute a backend."""
    advice = build_agent_advice(
        registry,
        result,
        metadata=metadata,
        parameters={operation_id: parameters},
        granted_side_effects=granted_side_effects,
    )
    matches = [
        candidate for candidate in advice.next_operations if candidate.operation_id == operation_id
    ]
    if len(matches) != 1 or not matches[0].ready:
        raise OrganelleInputError(
            code="input.next_operation_not_ready",
            message="The requested next operation is not ready.",
            details=advice.model_dump(mode="json"),
        )
    return _convert_assembly(result, metadata)
