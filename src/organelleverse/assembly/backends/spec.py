from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from organelleverse.operations.spec import StrictSpecModel

from ..profiles import AssemblyProfile

OrganelleType = Literal["mitochondrion", "plastid"]


class AuxiliaryRole(StrEnum):
    SEED_FASTA = "seed_fasta"
    REFERENCE_FASTA = "reference_fasta"
    REFERENCE_GENBANK = "reference_genbank"
    CHLOROPLAST_FASTA = "chloroplast_fasta"
    HMM_PROFILES = "hmm_profiles"
    CORRECTION_CONFIG = "correction_config"
    GENOME_SIZE = "genome_size"
    GENOME_RANGE = "genome_range"
    GENOME_SIZE_REPORT = "genome_size_report"
    CANU_EXECUTABLE = "canu_executable"
    NEXTDENOVO_EXECUTABLE = "nextdenovo_executable"


class EnvironmentCarrier(StrEnum):
    CONDA = "conda"
    CONTAINER = "container"
    NATIVE = "native"


class AssemblyEnvironmentRef(StrictSpecModel):
    carrier: EnvironmentCarrier
    contract_locator: str = Field(
        pattern=r"^organelleverse\.assembly\.environment_specs:[a-z][a-z0-9_]*$"
    )


class BackendInstallSpec(StrictSpecModel):
    tier: Literal["pip", "conda", "source"]
    pip: str | None = None
    conda: str | None = None
    source: str | None = None
    note: str = ""

    @model_validator(mode="after")
    def require_selected_tier(self) -> BackendInstallSpec:
        if getattr(self, self.tier) is None:
            raise ValueError("install tier requires a matching locator")
        return self


class AssemblyBackendSpec(StrictSpecModel):
    backend_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str = Field(min_length=1)
    organelles: tuple[OrganelleType, ...]
    profiles: tuple[AssemblyProfile, ...]
    required_auxiliary: tuple[AuxiliaryRole, ...] = ()
    mitochondrial_auxiliary: tuple[AuxiliaryRole, ...] = ()
    required_short_fields: tuple[Literal["read_length", "insert_size"], ...] = ()
    cli: str = Field(min_length=1)
    adapter_locator: str = Field(
        pattern=r"^organelleverse\.assembly\.backends\.[a-z0-9_]+:[A-Za-z_]\w*$"
    )
    output_contract: str = Field(pattern=r"^organelleverse\.assembly-output\.[a-z0-9_]+\.v1$")
    environment: AssemblyEnvironmentRef
    official_url: str = Field(pattern=r"^https://")
    fixture_ids: tuple[str, ...]
    install: BackendInstallSpec

    @model_validator(mode="after")
    def validate_unique_nonempty_capabilities(self) -> AssemblyBackendSpec:
        if not self.organelles or len(set(self.organelles)) != len(self.organelles):
            raise ValueError("backend organelles must be non-empty and unique")
        if not self.profiles or len(set(self.profiles)) != len(self.profiles):
            raise ValueError("backend profiles must be non-empty and unique")
        if len(set(self.required_auxiliary)) != len(self.required_auxiliary):
            raise ValueError("required auxiliary roles must be unique")
        return self
