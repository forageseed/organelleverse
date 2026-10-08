"""Adapter runtime value types and the minimal adapter protocol.

The adapter owns only backend-specific concerns: preflight, command construction,
output collection, and normalization. It does not create environments, start
processes, publish directories, or create core objects.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from pydantic import ConfigDict, Field, field_serializer, field_validator, model_validator

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.core.frozen import FrozenJson, FrozenMap, freeze_json, thaw_json
from organelleverse.operations.spec import StrictSpecModel


def _empty_artifact_map() -> FrozenMap[ArtifactRef]:
    return FrozenMap.from_items({})


if TYPE_CHECKING:
    from organelleverse.assembly.contracts import AssemblyInputPayload, AssemblyRequest
    from organelleverse.assembly.environments import PreparedEnvironment, PreparedProfile
    from organelleverse.assembly.manifests import AssemblyComponentIdentity
    from organelleverse.assembly.routing import AssemblyRoute

__all__ = [
    "AdapterContext",
    "AssemblyAdapter",
    "AssemblyCommand",
    "BackendOutput",
    "ExpectedBackendResources",
    "NormalizedAssemblyOutputs",
    "PreparedBackendResources",
    "RawAssemblyOutputs",
]


class AdapterContext(StrictSpecModel):
    """Frozen runtime context handed to an adapter for one assembly run."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
        arbitrary_types_allowed=True,
    )

    request: AssemblyRequest
    payload: AssemblyInputPayload
    route: AssemblyRoute
    environment: PreparedEnvironment
    profile: PreparedProfile | None = None
    resources: PreparedBackendResources
    workspace: Path
    input_artifacts: FrozenMap[ArtifactRef]
    effective_backend_parameters: FrozenMap[FrozenJson]

    @property
    def input_roles(self) -> tuple[str, ...]:
        return tuple(self.input_artifacts.keys())


class AssemblyCommand(StrictSpecModel):
    """Stable role-based argv plus host-resolved argv."""

    stable_argv: tuple[str, ...] = Field(min_length=1)
    resolved_argv: tuple[str, ...] = Field(min_length=1)


class BackendOutput(StrictSpecModel):
    """One backend output file mapped to a declared role."""

    role: str
    path: Path
    format: str
    media_type: str = "application/octet-stream"


class RawAssemblyOutputs(StrictSpecModel):
    """Raw backend output files mapped to declared roles."""

    outputs: tuple[BackendOutput, ...]
    primary_sequence_role: str
    primary_graph_role: str | None = None

    @model_validator(mode="after")
    def validate_roles(self) -> RawAssemblyOutputs:
        roles = tuple(item.role for item in self.outputs)
        if len(set(roles)) != len(roles):
            raise ValueError("raw assembly outputs contain duplicate roles")
        if self.primary_sequence_role not in roles:
            raise ValueError(
                f"primary_sequence_role {self.primary_sequence_role!r} is not present in outputs"
            )
        if self.primary_graph_role is not None and self.primary_graph_role not in roles:
            raise ValueError(
                f"primary_graph_role {self.primary_graph_role!r} is not present in outputs"
            )
        return self

    def require(self, role: str) -> BackendOutput:
        matches = tuple(item for item in self.outputs if item.role == role)
        if len(matches) != 1:
            raise OrganelleExecutionError(
                code="assembly.output_incomplete",
                message=f"raw assembly outputs have no unique role {role!r}",
                details={"role": role},
            )
        return matches[0]


class NormalizedAssemblyOutputs(RawAssemblyOutputs):
    """Backend-independent normalized output files and metrics."""

    alternate_sequence_roles: tuple[str, ...] = ()
    record_count: int = Field(gt=0)
    total_bases: int = Field(gt=0)
    gene_markers: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_alternate_roles(self) -> NormalizedAssemblyOutputs:
        roles = {item.role for item in self.outputs}
        for role in self.alternate_sequence_roles:
            if role not in roles:
                raise ValueError(f"alternate_sequence_role {role!r} is not present in outputs")
        return self

    @property
    def primary_sequence(self) -> BackendOutput:
        return self.require(self.primary_sequence_role)

    @property
    def primary_graph(self) -> BackendOutput | None:
        if self.primary_graph_role is None:
            return None
        return self.require(self.primary_graph_role)


class ExpectedBackendResources(StrictSpecModel):
    """Expected backend resources: components and provenance hashes."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
        arbitrary_types_allowed=True,
    )

    components: tuple[AssemblyComponentIdentity, ...] = ()
    artifact_roles: tuple[str, ...] = ()
    model_hashes: FrozenMap[FrozenJson] = Field(default_factory=FrozenMap)
    database_hashes: FrozenMap[FrozenJson] = Field(default_factory=FrozenMap)

    @field_validator("model_hashes", "database_hashes", mode="before")
    @classmethod
    def freeze_objects(cls, value: object) -> FrozenMap[FrozenJson]:
        try:
            frozen = freeze_json({} if value is None else value)
        except Exception as error:
            raise ValueError(str(error)) from error
        if not isinstance(frozen, FrozenMap):
            raise ValueError("resource hash field must be a JSON object")
        return frozen

    @field_serializer("model_hashes", "database_hashes")
    def serialize_objects(self, value: FrozenMap[FrozenJson]) -> object:
        return thaw_json(value)


class PreparedBackendResources(ExpectedBackendResources):
    """Prepared backend resources with role-addressed artifact references."""

    artifacts: FrozenMap[ArtifactRef] = Field(default_factory=_empty_artifact_map)

    @field_validator("artifacts", mode="before")
    @classmethod
    def freeze_artifacts(cls, value: object) -> FrozenMap[ArtifactRef]:
        if value is None:
            return _empty_artifact_map()
        if isinstance(value, FrozenMap):
            return cast(FrozenMap[ArtifactRef], value)
        if isinstance(value, Mapping):
            return FrozenMap.from_items(cast(Mapping[str, ArtifactRef], value))
        raise ValueError("artifacts must be a mapping from role to ArtifactRef")

    @field_serializer("artifacts")
    def serialize_artifacts(self, value: FrozenMap[ArtifactRef]) -> dict[str, object]:
        return {key: value[key].model_dump(mode="json") for key in value}


@runtime_checkable
class AssemblyAdapter(Protocol):
    """Minimal adapter protocol implemented by every assembly backend."""

    backend_id: str

    def preflight(self, context: AdapterContext) -> None: ...

    def build_command(self, context: AdapterContext) -> AssemblyCommand: ...

    def collect_outputs(self, context: AdapterContext) -> RawAssemblyOutputs: ...

    def normalize(
        self,
        context: AdapterContext,
        raw: RawAssemblyOutputs,
        output_dir: Path,
    ) -> NormalizedAssemblyOutputs: ...


# Resolve the forward-referenced contract types once their modules are importable.
# This is deferred to break the import cycle: routing -> backends package -> base.
def _rebuild() -> None:
    # Importing these names registers them in the module namespace so Pydantic can
    # resolve the forward references in AdapterContext.model_rebuild(). Each name is
    # bound to a module-level alias via globals() to make the registration explicit and
    # avoid pyright flagging the imports as unused.
    from organelleverse.assembly.contracts import (
        AssemblyInputPayload,
        AssemblyRequest,
    )
    from organelleverse.assembly.environments import (
        PreparedEnvironment,
        PreparedProfile,
    )
    from organelleverse.assembly.manifests import AssemblyComponentIdentity
    from organelleverse.assembly.routing import AssemblyRoute

    globals().update(
        {
            "AssemblyInputPayload": AssemblyInputPayload,
            "AssemblyRequest": AssemblyRequest,
            "PreparedEnvironment": PreparedEnvironment,
            "PreparedProfile": PreparedProfile,
            "AssemblyRoute": AssemblyRoute,
            "AssemblyComponentIdentity": AssemblyComponentIdentity,
        }
    )
    AdapterContext.model_rebuild()
    ExpectedBackendResources.model_rebuild()
    PreparedBackendResources.model_rebuild()


_rebuild()
