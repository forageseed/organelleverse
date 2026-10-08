"""Stable assembly run manifests and volatile execution observations."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Literal
from unicodedata import category

from pydantic import ConfigDict, Field, field_serializer, field_validator, model_validator

from organelleverse.assembly.environment_contracts import (
    BackendVersionSelector,
    EnvironmentHint,
    EnvironmentSource,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInternalError
from organelleverse.core.frozen import FrozenJson, FrozenMap, freeze_json, thaw_json
from organelleverse.operations.spec import StrictSpecModel

from .contracts import AssemblyInputPayload, AssemblyMethod, canonical_json_bytes

_ARTIFACT_ROLE_URI = re.compile(r"^role://artifact/(?P<role>[a-z][a-z0-9_]*)$")
_WORKSPACE_ROLE_URI = re.compile(r"^role://workspace/(?P<role>[a-z][a-z0-9_]*)$")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_NUMERIC_LITERAL = re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)$")

# Stable manifests use only these host-resolved workspace roles.
_STABLE_WORKSPACE_ROLES = frozenset(
    {"assembly", "assembly_result", "cache", "input", "output", "subsample", "temp"}
)


def _artifact_identity(artifact: ArtifactRef) -> dict[str, object]:
    return {
        "kind": artifact.kind,
        "format": artifact.format,
        "media_type": artifact.media_type,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
    }


def _is_host_path(value: str) -> bool:
    folded = value.casefold()
    return (
        value.startswith(("/", "\\"))
        or folded.startswith("file:")
        or _WINDOWS_DRIVE.match(value) is not None
        or value.startswith(("./", "../", ".\\", "..\\", "~/", "~\\"))
    )


def _is_word_literal(value: str) -> bool:
    return (
        bool(value)
        and any(category(character)[0] in {"L", "N"} for character in value)
        and all(category(character)[0] in {"L", "N"} or character in "_-+" for character in value)
    )


def _is_executable_basename(value: str) -> bool:
    if value in {".", ".."} or value.startswith(("-", "+")):
        return False
    return all(_is_word_literal(part) for part in value.split("."))


def _is_later_stable_token(value: str) -> bool:
    if _NUMERIC_LITERAL.fullmatch(value) is not None or _is_word_literal(value):
        return True
    if value.startswith("--") and "=" in value:
        key, attached = value[2:].split("=", 1)
        return _is_word_literal(key) and (
            _is_word_literal(attached) or _NUMERIC_LITERAL.fullmatch(attached) is not None
        )
    if value.startswith("--"):
        return _is_word_literal(value[2:])
    if value.startswith("-"):
        return _is_word_literal(value[1:])
    return False


class ManifestArtifact(StrictSpecModel):
    role: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    artifact: ArtifactRef

    @field_validator("artifact", mode="before")
    @classmethod
    def validate_serialized_artifact(cls, value: object) -> ArtifactRef:
        return ArtifactRef.model_validate(value)


class AssemblyEnvironmentIdentity(StrictSpecModel):
    carrier: Literal["conda", "container", "native"]
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    platform: str = Field(min_length=1)


class AssemblyComponentIdentity(StrictSpecModel):
    category: Literal["software", "source", "database", "profile"]
    name: str = Field(min_length=1)
    version: str = ""
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    locator: str = ""

    @model_validator(mode="after")
    def require_stable_identity(self) -> AssemblyComponentIdentity:
        if self.locator and _is_host_path(self.locator):
            raise ValueError("component requires a stable locator, not a local path")
        if not (self.version or self.sha256 or self.locator):
            raise ValueError("component requires a version, SHA256, or stable locator")
        return self


class AssemblyStageOutcome(StrictSpecModel):
    stage: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    status: Literal["ok", "failed", "skipped"]
    process_started: bool = False
    exit_code: int | None = None
    termination: Literal["exit", "signal", "timeout", "launch_error"] | None = None
    output_roles: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_process_state(self) -> AssemblyStageOutcome:
        if self.status == "skipped" and self.process_started:
            raise ValueError("a skipped stage cannot start a process")
        if not self.process_started:
            if self.termination is not None:
                raise ValueError("an unstarted process cannot record a termination")
            if self.exit_code is not None:
                raise ValueError("an unstarted process cannot record an exit code")
        if self.process_started and self.termination is None:
            raise ValueError("a started process requires a termination")
        if self.termination == "timeout" and self.exit_code is not None:
            raise ValueError("a timed-out process cannot record an exit code")
        if (
            self.process_started
            and self.termination in {"exit", "signal"}
            and self.exit_code is None
        ):
            raise ValueError("an exit/signal termination requires an exit code")
        return self


class AssemblyRunParameters(StrictSpecModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
        arbitrary_types_allowed=True,
    )

    threads: int = Field(ge=1, le=256)
    memory_gb: int | None = Field(default=None, ge=1)
    timeout_seconds: int | None = Field(default=None, ge=1)
    environment_source: EnvironmentSource = "auto"
    backend_version: BackendVersionSelector = "tested"
    environment_hint: EnvironmentHint | None = None
    backend_parameters: FrozenMap[FrozenJson] = Field(default_factory=FrozenMap)

    @field_validator("environment_hint", mode="before")
    @classmethod
    def _validate_serialized_hint(cls, value: object) -> EnvironmentHint | None:
        if value is None:
            return None
        return EnvironmentHint.model_validate(value)

    @field_validator("backend_parameters", mode="before")
    @classmethod
    def freeze_backend_parameters(cls, value: object) -> FrozenMap[FrozenJson]:
        try:
            frozen = freeze_json({} if value is None else value)
        except OrganelleInternalError as error:
            raise ValueError(str(error)) from error
        if not isinstance(frozen, FrozenMap):
            raise ValueError("backend_parameters must be a JSON object")
        return frozen

    @field_serializer("backend_parameters")
    def serialize_backend_parameters(self, value: FrozenMap[FrozenJson]) -> object:
        return thaw_json(value)


class RoutingEvidence(StrictSpecModel):
    """Stable, destination-free record of how one assembly was routed.

    The rule id, taxon group, integer coverage threshold, and traceable
    genome-size accession are part of the run identity; local paths never
    appear here.
    """

    rule_id: str = Field(pattern=r"^[a-z][a-z0-9_.]*$")
    taxon_group: Literal["plant", "animal", "fungi"]
    total_bases: int | None = Field(default=None, ge=0)
    genome_size_bp: int | None = Field(default=None, gt=0)
    threshold_multiplier: int | None = Field(default=None, ge=1)
    threshold_numerator: int | None = Field(default=None, ge=0)
    threshold_denominator: int | None = Field(default=None, gt=0)
    coverage_display: str | None = None
    genome_size_accession: str | None = None

    @model_validator(mode="after")
    def validate_derived_fields(self) -> RoutingEvidence:
        threshold_values = (
            self.threshold_multiplier,
            self.threshold_numerator,
            self.threshold_denominator,
        )
        coverage_rules = {
            "auto.mito_hifi_lte_3x_pmat",
            "auto.mito_hifi_gt_3x_oatk",
        }
        if self.rule_id in coverage_rules:
            if any(value is None for value in threshold_values):
                raise ValueError("coverage routing requires complete threshold evidence")
        elif any(value is not None for value in threshold_values):
            raise ValueError("non-coverage routing forbids threshold evidence")

        if any(value is not None for value in threshold_values):
            if self.total_bases is None or self.genome_size_bp is None:
                raise ValueError("threshold evidence requires total bases and genome size")
            assert self.threshold_multiplier is not None
            if self.threshold_numerator != self.total_bases:
                raise ValueError("threshold numerator must equal total bases")
            if self.threshold_denominator != self.threshold_multiplier * self.genome_size_bp:
                raise ValueError("threshold denominator must equal multiplier times genome size")

        if self.total_bases is not None and self.genome_size_bp is not None:
            expected_display = f"{self.total_bases / self.genome_size_bp:.2f}x"
            if self.coverage_display != expected_display:
                raise ValueError(
                    "coverage display must be derived from total bases and genome size"
                )
        elif self.coverage_display is not None:
            raise ValueError("coverage display requires total bases and genome size")
        if self.genome_size_accession is not None and self.genome_size_bp is None:
            raise ValueError("genome-size accession requires a genome size")
        return self


class AssemblyRunManifest(StrictSpecModel):
    schema_version: Literal["organelleverse.assembly-run.v1"] = "organelleverse.assembly-run.v1"
    operation_id: Literal[
        "assembly.assemble",
        "assembly.pmat_graph_build",
    ] = "assembly.assemble"
    contract_version: Literal["1.0"] = "1.0"
    input_data_id: str = Field(pattern=r"^data:sha256:[0-9a-f]{64}$")
    input_artifacts: tuple[ManifestArtifact, ...]
    input_payload: AssemblyInputPayload | None = None
    organelle: Literal["mitochondrion", "plastid"]
    parameters: AssemblyRunParameters
    requested_method: AssemblyMethod | Literal["auto"]
    selected_backend: AssemblyMethod
    route_reason_code: str = Field(pattern=r"^[a-z][a-z0-9_.]*$")
    routing_evidence: RoutingEvidence | None = None
    environment: AssemblyEnvironmentIdentity
    components: tuple[AssemblyComponentIdentity, ...]
    primary_sequence_role: str | None = Field(
        default="assembly_fasta",
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    stable_argv: tuple[str, ...]
    stages: tuple[AssemblyStageOutcome, ...]
    process_exit_code: int | None = None
    outputs: tuple[ManifestArtifact, ...]

    @field_validator("stable_argv")
    @classmethod
    def validate_stable_argv(cls, tokens: tuple[str, ...]) -> tuple[str, ...]:
        if not tokens:
            raise ValueError("stable argv must not be empty")
        if not _is_executable_basename(tokens[0]):
            raise ValueError("stable argv executable must be a path-free basename")
        for token in tokens[1:]:
            if not token:
                raise ValueError("stable argv tokens must not be empty")
            artifact_role = _ARTIFACT_ROLE_URI.fullmatch(token)
            if artifact_role is not None:
                continue
            workspace_role = _WORKSPACE_ROLE_URI.fullmatch(token)
            if workspace_role is not None:
                if workspace_role.group("role") not in _STABLE_WORKSPACE_ROLES:
                    raise ValueError("stable argv contains an unknown workspace role")
                continue
            if token.casefold().startswith("role://") or not _is_later_stable_token(token):
                raise ValueError("stable argv tokens must use a path-free canonical grammar")
        return tokens

    @model_validator(mode="after")
    def validate_semantic_links(self) -> AssemblyRunManifest:
        if self.primary_sequence_role is None:
            raise ValueError("primary_sequence_role must identify the primary genome sequence")
        if self.requested_method != "auto" and self.selected_backend != self.requested_method:
            raise ValueError("explicit assembly method forbids backend fallback")
        if (
            self.routing_evidence is not None
            and self.routing_evidence.rule_id != self.route_reason_code
        ):
            raise ValueError("routing evidence rule must match route reason code")

        input_roles = tuple(item.role for item in self.input_artifacts)
        if len(set(input_roles)) != len(input_roles):
            raise ValueError("input artifact roles must be unique")

        output_roles = tuple(item.role for item in self.outputs)
        if len(set(output_roles)) != len(output_roles):
            raise ValueError("output artifact roles must be unique")

        declared_artifact_roles = set(input_roles) | set(output_roles)
        argv_artifact_roles = {
            match.group("role")
            for token in self.stable_argv
            if (match := _ARTIFACT_ROLE_URI.fullmatch(token)) is not None
        }
        if argv_artifact_roles - declared_artifact_roles:
            raise ValueError("stable argv artifact roles must reference manifest artifacts")

        component_keys = tuple((item.category, item.name) for item in self.components)
        if len(set(component_keys)) != len(component_keys):
            raise ValueError("component category and name pairs must be unique")

        stage_names = tuple(item.stage for item in self.stages)
        if len(set(stage_names)) != len(stage_names):
            raise ValueError("stage names must be ordered and unique")

        available_outputs = set(output_roles)
        for stage in self.stages:
            if len(set(stage.output_roles)) != len(stage.output_roles):
                raise ValueError("stage output roles must be unique")
            unknown = set(stage.output_roles) - available_outputs
            if unknown:
                raise ValueError("stage output roles must reference manifest outputs")

        started_stages = tuple(stage for stage in self.stages if stage.process_started)
        if not started_stages:
            if self.process_exit_code is not None:
                raise ValueError("process exit code requires a started process")
        elif self.process_exit_code != started_stages[-1].exit_code:
            raise ValueError("process exit code must match the final started stage")
        try:
            canonical_json_bytes(self.semantic_payload())
        except (TypeError, ValueError) as error:
            raise ValueError(
                "assembly run manifest semantic identity must be canonical UTF-8 JSON"
            ) from error
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "contract_version": self.contract_version,
            "input_data_id": self.input_data_id,
            "input_artifacts": {
                item.role: _artifact_identity(item.artifact)
                for item in sorted(self.input_artifacts, key=lambda item: item.role)
            },
            **(
                {"input_payload": self.input_payload.model_dump(mode="json", exclude_none=True)}
                if self.input_payload is not None
                else {}
            ),
            "organelle": self.organelle,
            "parameters": self.parameters.model_dump(mode="json"),
            "requested_method": self.requested_method,
            "selected_backend": self.selected_backend,
            "route_reason_code": self.route_reason_code,
            "routing_evidence": (
                self.routing_evidence.model_dump(mode="json", exclude_none=True)
                if self.routing_evidence is not None
                else None
            ),
            "environment": self.environment.model_dump(mode="json"),
            "components": [
                item.model_dump(mode="json", exclude_none=True)
                for item in sorted(
                    self.components,
                    key=lambda item: (item.category, item.name),
                )
            ],
            "primary_sequence_role": self.primary_sequence_role,
            "stable_argv": list(self.stable_argv),
            "stages": [item.model_dump(mode="json") for item in self.stages],
            "process_exit_code": self.process_exit_code,
            "outputs": {
                item.role: _artifact_identity(item.artifact)
                for item in sorted(self.outputs, key=lambda item: item.role)
            },
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())

    @property
    def run_manifest_id(self) -> str:
        return f"assembly-run:sha256:{hashlib.sha256(self.canonical_bytes()).hexdigest()}"

    def as_artifact(self, uri: str) -> ArtifactRef:
        payload = self.canonical_bytes()
        return ArtifactRef(
            kind="assembly_run_manifest",
            uri=uri,
            format="json",
            media_type="application/json",
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            validated=True,
        )


def assembly_run_id_from_artifact(artifact: ArtifactRef) -> str:
    if (
        artifact.kind != "assembly_run_manifest"
        or artifact.format != "json"
        or artifact.media_type != "application/json"
    ):
        raise ValueError("artifact is not an assembly run manifest")
    return f"assembly-run:sha256:{artifact.sha256}"


class AssemblyRunObservations(StrictSpecModel):
    schema_version: Literal["organelleverse.assembly-observations.v1"] = (
        "organelleverse.assembly-observations.v1"
    )
    run_manifest_id: str = Field(pattern=r"^assembly-run:sha256:[0-9a-f]{64}$")
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    peak_cpu_percent: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    workspace_path: str = ""
    cache_path: str = ""
    output_path: str = ""
    resolved_argv: tuple[str, ...] = ()
    stdout_path: str = ""
    stderr_path: str = ""

    @model_validator(mode="after")
    def validate_timestamps(self) -> AssemblyRunObservations:
        if self.started_at is None or self.finished_at is None:
            return self
        started_is_aware = (
            self.started_at.tzinfo is not None and self.started_at.utcoffset() is not None
        )
        finished_is_aware = (
            self.finished_at.tzinfo is not None and self.finished_at.utcoffset() is not None
        )
        if started_is_aware != finished_is_aware:
            raise ValueError("started_at and finished_at must have matching timezone awareness")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot be earlier than started_at")
        return self
