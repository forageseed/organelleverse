"""Slurm provider configuration types (spec §13).

Owner-configured, closed identities. Opaque IDs (``cluster_id``,
``workspace_id``, ``environment_id``) map to administrator-configured workspace,
managed environment, module set, partition, account, and QoS through
provider-owned mappings — never through model-supplied scheduler fields.
``allowed_resource_profiles`` is the closed set the caller may select from;
model content cannot supply raw resource values.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import Field, model_validator

from organelleverse.operations.spec import StrictSpecModel

__all__ = ["SlurmResources", "SlurmTargetConfig"]

_ID = r"^[a-z0-9][a-z0-9._-]{0,63}$"


class SlurmResources(StrictSpecModel):
    """Bounded resource request for one Slurm job (spec §13).

    Every field is bounded so a model cannot request unbounded compute. The
    optional partition/account/QoS ids must come from trusted owner config,
    not from model content.
    """

    cpus: int = Field(ge=1, le=1024)
    memory_mb: int = Field(ge=128, le=16_777_216)
    walltime_seconds: int = Field(ge=60, le=2_592_000)
    gpus: int = Field(default=0, ge=0, le=64)
    partition_id: str | None = Field(default=None, pattern=_ID)
    account_id: str | None = Field(default=None, pattern=_ID)
    qos_id: str | None = Field(default=None, pattern=_ID)


class SlurmTargetConfig(StrictSpecModel):
    """An owner-configured Slurm target and its closed resource profiles.

    ``allowed_resource_profiles`` is the exhaustive set of profiles a caller may
    select by id; the selection happens at the L5-01 admission boundary, not in
    model content. The model carries no scheduler path, script, module text, or
    credential.
    """

    target_id: str = Field(pattern=r"^slurm:[a-z0-9][a-z0-9._-]{0,127}$")
    cluster_id: str = Field(pattern=_ID)
    workspace_id: str = Field(pattern=_ID)
    environment_id: str = Field(pattern=_ID)
    allowed_resource_profiles: Mapping[str, SlurmResources]

    @model_validator(mode="after")
    def _target_and_profiles_are_closed(self) -> SlurmTargetConfig:
        if not self.allowed_resource_profiles:
            raise ValueError("at least one allowed_resource_profiles entry is required")
        # profile ids share the closed id alphabet
        for profile_id in self.allowed_resource_profiles:
            import re

            if re.fullmatch(_ID, profile_id) is None:
                raise ValueError(f"resource profile id is not a closed identifier: {profile_id!r}")
        return self
