"""Agent-facing sequencing-read data contracts for assembly operations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleInputError

from .continuation import DirectoryManifest
from .contracts import (
    AssemblyInputPayload,
    ContigContract,
    GenomeSizeEvidence,
    LongLibraryContract,
    QualityState,
    ShortLibraryContract,
    validate_assembly_data,
)

if TYPE_CHECKING:
    from organelleverse.operations.data_contracts import DataContract


class _OatkContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class _OatkLongLibrary(_OatkContractModel):
    technology: Literal["pacbio_hifi"]
    quality_state: Literal["ccs"]
    reads_artifact: str = Field(min_length=1)


class _OatkAuxiliary(_OatkContractModel):
    seed_fasta_artifact: None = None
    reference_fasta_artifact: None = None
    reference_genbank_artifact: None = None
    chloroplast_fasta_artifact: None = None
    hmm_profiles_artifact: str | None = None
    correction_config_artifact: None = None
    genome_size: int | None = Field(default=None, gt=0)
    genome_range: None = None
    genome_size_evidence: GenomeSizeEvidence | None = None
    genome_size_report_artifact: str | None = None
    canu_executable_artifact: None = None
    nextdenovo_executable_artifact: None = None


class _OatkAssemblyInputPayload(_OatkContractModel):
    contract_version: Literal["organelleverse.assembly-input.v1"] = (
        "organelleverse.assembly-input.v1"
    )
    short_libraries: tuple[ShortLibraryContract, ...] = Field(default=(), max_length=0)
    long_libraries: tuple[_OatkLongLibrary, ...] = Field(min_length=1)
    contig_inputs: tuple[ContigContract, ...] = Field(default=(), max_length=0)
    auxiliary: _OatkAuxiliary = _OatkAuxiliary()


class _HimtContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class _HimtLongLibrary(_HimtContractModel):
    technology: Literal["pacbio_hifi", "pacbio_clr", "ont"]
    quality_state: QualityState
    reads_artifact: str = Field(min_length=1)


class _HimtAuxiliary(_HimtContractModel):
    seed_fasta_artifact: None = None
    reference_fasta_artifact: None = None
    reference_genbank_artifact: None = None
    chloroplast_fasta_artifact: None = None
    hmm_profiles_artifact: None = None
    correction_config_artifact: None = None
    genome_size: None = None
    genome_range: None = None


class _HimtAssemblyInputPayload(_HimtContractModel):
    contract_version: Literal["organelleverse.assembly-input.v1"] = (
        "organelleverse.assembly-input.v1"
    )
    short_libraries: tuple[ShortLibraryContract, ...] = Field(default=(), max_length=0)
    long_libraries: tuple[_HimtLongLibrary, ...] = Field(min_length=1, max_length=1)
    contig_inputs: tuple[ContigContract, ...] = Field(default=(), max_length=0)
    auxiliary: _HimtAuxiliary = _HimtAuxiliary()


class _PmatContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class _PmatLongLibrary(_PmatContractModel):
    technology: Literal["pacbio_hifi", "pacbio_clr", "ont"]
    quality_state: QualityState
    reads_artifact: str = Field(min_length=1)


class _PmatAuxiliary(_PmatContractModel):
    seed_fasta_artifact: None = None
    reference_fasta_artifact: None = None
    reference_genbank_artifact: None = None
    chloroplast_fasta_artifact: None = None
    hmm_profiles_artifact: None = None
    correction_config_artifact: str | None = None
    genome_size: int | None = Field(default=None, gt=0)
    genome_range: None = None
    genome_size_evidence: GenomeSizeEvidence | None = None
    genome_size_report_artifact: str | None = None
    canu_executable_artifact: str | None = None
    nextdenovo_executable_artifact: str | None = None


class _PmatAssemblyInputPayload(_PmatContractModel):
    contract_version: Literal["organelleverse.assembly-input.v1"] = (
        "organelleverse.assembly-input.v1"
    )
    short_libraries: tuple[ShortLibraryContract, ...] = Field(default=(), max_length=0)
    long_libraries: tuple[_PmatLongLibrary, ...] = Field(min_length=1)
    contig_inputs: tuple[ContigContract, ...] = Field(default=(), max_length=0)
    auxiliary: _PmatAuxiliary = _PmatAuxiliary()


class _GetOrganelleContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class _GetOrganellePairedEndLibrary(_GetOrganelleContractModel):
    technology: Literal["illumina"]
    layout: Literal["paired_end"]
    read1_artifact: str = Field(min_length=1)
    read2_artifact: str = Field(min_length=1)
    read_length: int = Field(gt=0)
    insert_size: int | None = Field(default=None, gt=0)


class _GetOrganelleSingleEndLibrary(_GetOrganelleContractModel):
    technology: Literal["illumina"]
    layout: Literal["single_end"]
    read1_artifact: str = Field(min_length=1)
    read_length: int = Field(gt=0)
    insert_size: int | None = Field(default=None, gt=0)


_GetOrganelleShortLibrary = _GetOrganellePairedEndLibrary | _GetOrganelleSingleEndLibrary


class _GetOrganelleAuxiliary(_GetOrganelleContractModel):
    seed_fasta_artifact: str | None = None
    reference_fasta_artifact: None = None
    reference_genbank_artifact: None = None
    chloroplast_fasta_artifact: None = None
    hmm_profiles_artifact: None = None
    correction_config_artifact: None = None
    genome_size: None = None
    genome_range: None = None
    genome_size_evidence: None = None
    genome_size_report_artifact: None = None
    canu_executable_artifact: None = None
    nextdenovo_executable_artifact: None = None
    anti_seed_artifact: str | None = None
    label_genes_artifact: str | None = None
    exclude_genes_artifact: str | None = None


class _GetOrganelleAssemblyInputPayload(_GetOrganelleContractModel):
    contract_version: Literal["organelleverse.assembly-input.v1"] = (
        "organelleverse.assembly-input.v1"
    )
    short_libraries: tuple[_GetOrganelleShortLibrary, ...] = Field(min_length=1, max_length=1)
    long_libraries: tuple[LongLibraryContract, ...] = Field(default=(), max_length=0)
    contig_inputs: tuple[ContigContract, ...] = Field(default=(), max_length=0)
    auxiliary: _GetOrganelleAuxiliary = _GetOrganelleAuxiliary()


class _PtgaulContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class _PtgaulLongLibrary(_PtgaulContractModel):
    technology: Literal["ont"]
    quality_state: Literal["raw"]
    reads_artifact: str = Field(min_length=1)


class _PtgaulAuxiliary(_PtgaulContractModel):
    seed_fasta_artifact: None = None
    reference_fasta_artifact: str = Field(min_length=1)
    reference_genbank_artifact: None = None
    chloroplast_fasta_artifact: None = None
    hmm_profiles_artifact: None = None
    correction_config_artifact: None = None
    genome_size: None = None
    genome_range: None = None
    genome_size_evidence: None = None
    genome_size_report_artifact: None = None
    canu_executable_artifact: None = None
    nextdenovo_executable_artifact: None = None
    anti_seed_artifact: None = None
    label_genes_artifact: None = None
    exclude_genes_artifact: None = None


class _PtgaulAssemblyInputPayload(_PtgaulContractModel):
    contract_version: Literal["organelleverse.assembly-input.v1"] = "organelleverse.assembly-input.v1"
    short_libraries: tuple[ShortLibraryContract, ...] = Field(default=(), max_length=0)
    long_libraries: tuple[_PtgaulLongLibrary, ...] = Field(min_length=1, max_length=1)
    contig_inputs: tuple[ContigContract, ...] = Field(default=(), max_length=0)
    auxiliary: _PtgaulAuxiliary


class _TippoContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class _TippoLongLibrary(_TippoContractModel):
    technology: Literal["pacbio_hifi"]
    quality_state: Literal["ccs"]
    reads_artifact: str = Field(min_length=1)


class _TippoAuxiliary(_TippoContractModel):
    seed_fasta_artifact: None = None
    reference_fasta_artifact: None = None
    reference_genbank_artifact: None = None
    chloroplast_fasta_artifact: None = None
    hmm_profiles_artifact: None = None
    correction_config_artifact: None = None
    genome_size: None = None
    genome_range: None = None
    genome_size_evidence: None = None
    genome_size_report_artifact: None = None
    canu_executable_artifact: None = None
    nextdenovo_executable_artifact: None = None
    anti_seed_artifact: None = None
    label_genes_artifact: None = None
    exclude_genes_artifact: None = None


class _TippoAssemblyInputPayload(_TippoContractModel):
    contract_version: Literal["organelleverse.assembly-input.v1"] = (
        "organelleverse.assembly-input.v1"
    )
    short_libraries: tuple[ShortLibraryContract, ...] = Field(default=(), max_length=0)
    long_libraries: tuple[_TippoLongLibrary, ...] = Field(min_length=1, max_length=1)
    contig_inputs: tuple[ContigContract, ...] = Field(default=(), max_length=0)
    auxiliary: _TippoAuxiliary = _TippoAuxiliary()


class _NovoplastyContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class _NovoplastyPairedLibrary(_NovoplastyContractModel):
    technology: Literal["illumina"]
    layout: Literal["paired_end"]
    read1_artifact: str = Field(min_length=1)
    read2_artifact: str = Field(min_length=1)
    read_length: int = Field(gt=0)
    insert_size: int = Field(gt=0)


class _NovoplastyAuxiliary(_NovoplastyContractModel):
    seed_fasta_artifact: str | None = None
    reference_fasta_artifact: str | None = None
    chloroplast_fasta_artifact: str | None = None
    genome_range: tuple[int, int] | None = None
    reference_genbank_artifact: None = None
    hmm_profiles_artifact: None = None
    correction_config_artifact: None = None
    genome_size: None = None
    genome_size_evidence: None = None
    genome_size_report_artifact: None = None
    canu_executable_artifact: None = None
    nextdenovo_executable_artifact: None = None
    anti_seed_artifact: None = None
    label_genes_artifact: None = None
    exclude_genes_artifact: None = None


class _NovoplastyAssemblyInputPayload(_NovoplastyContractModel):
    contract_version: Literal["organelleverse.assembly-input.v1"] = (
        "organelleverse.assembly-input.v1"
    )
    short_libraries: tuple[_NovoplastyPairedLibrary, ...] = Field(min_length=1, max_length=1)
    long_libraries: tuple[LongLibraryContract, ...] = Field(default=(), max_length=0)
    contig_inputs: tuple[ContigContract, ...] = Field(default=(), max_length=0)
    auxiliary: _NovoplastyAuxiliary = _NovoplastyAuxiliary()


class _ReleasedAssemblyInputPayload(
    RootModel[
        _OatkAssemblyInputPayload
        | _HimtAssemblyInputPayload
        | _GetOrganelleAssemblyInputPayload
        | _PmatAssemblyInputPayload
        | _TippoAssemblyInputPayload
        | _NovoplastyAssemblyInputPayload
    ]
):
    """Root union used only for Agent-facing schema generation.

    The union is ``anyOf`` so valid payloads for any released backend are described
    by the schema without forcing a single branch selection.
    """


_HIMT_QUALITY_PAIRS: dict[str, set[str]] = {
    "pacbio_hifi": {"ccs"},
    "pacbio_clr": {"raw", "corrected"},
    "ont": {"raw", "corrected", "duplex", "hq"},
}

_PMAT_QUALITY_PAIRS = _HIMT_QUALITY_PAIRS


class PmatGraphBuildData(_PmatContractModel):
    """Closed continuation payload consumed by ``PMAT graphBuild``."""

    contract_version: Literal["organelleverse.pmat-graph-input.v1"] = (
        "organelleverse.pmat-graph-input.v1"
    )
    subsample_manifest_artifact: str = Field(min_length=1)
    assembly_result_manifest_artifact: str = Field(min_length=1)
    file_artifact_roles: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_roles(self) -> PmatGraphBuildData:
        if len(set(self.file_artifact_roles)) != len(self.file_artifact_roles):
            raise ValueError("PMAT graphBuild file artifact roles must be unique")
        return self


def assembly_data_json_schema() -> dict[str, object]:
    from organelleverse.operations.schemas import specialize_data_schema

    return specialize_data_schema(
        modality="sequencing_reads",
        payload_model=AssemblyInputPayload,
    )


def oatk_assembly_data_json_schema() -> dict[str, object]:
    """Return the closed Agent input Schema for the released Oatk slice."""
    from organelleverse.operations.schemas import specialize_data_schema

    return specialize_data_schema(
        modality="sequencing_reads",
        payload_model=_OatkAssemblyInputPayload,
    )


def himt_assembly_data_json_schema() -> dict[str, object]:
    """Return the closed Agent input Schema for the released HiMT slice."""
    from organelleverse.operations.schemas import specialize_data_schema

    return specialize_data_schema(
        modality="sequencing_reads",
        payload_model=_HimtAssemblyInputPayload,
    )


def getorganelle_assembly_data_json_schema() -> dict[str, object]:
    """Return the closed Agent input Schema for the released GetOrganelle slice."""
    from organelleverse.operations.schemas import specialize_data_schema

    return specialize_data_schema(
        modality="sequencing_reads",
        payload_model=_GetOrganelleAssemblyInputPayload,
    )


def pmat_assembly_data_json_schema() -> dict[str, object]:
    """Return the closed Agent input Schema for the candidate PMAT2 slice."""
    from organelleverse.operations.schemas import specialize_data_schema

    return specialize_data_schema(
        modality="sequencing_reads",
        payload_model=_PmatAssemblyInputPayload,
    )


def pmat_graph_input_json_schema() -> dict[str, object]:
    """Return the closed Agent schema for a PMAT2 continuation."""
    from organelleverse.operations.schemas import specialize_data_schema

    return specialize_data_schema(
        modality="pmat_graph_input",
        payload_model=PmatGraphBuildData,
    )


def ptgaul_assembly_data_json_schema() -> dict[str, object]:
    """Return the closed Agent input schema for reference-guided ptGAUL ONT assembly."""
    from organelleverse.operations.schemas import specialize_data_schema

    return specialize_data_schema(
        modality="sequencing_reads",
        payload_model=_PtgaulAssemblyInputPayload,
    )


def tippo_assembly_data_json_schema() -> dict[str, object]:
    """Return the closed Agent input schema for the TIPPo plastid HiFi slice."""
    from organelleverse.operations.schemas import specialize_data_schema

    return specialize_data_schema(
        modality="sequencing_reads",
        payload_model=_TippoAssemblyInputPayload,
    )


def released_assembly_data_json_schema() -> dict[str, object]:
    """Return the combined anyOf Agent input Schema for released assembly backends."""
    from organelleverse.operations.schemas import specialize_data_schema

    return specialize_data_schema(
        modality="sequencing_reads",
        payload_model=_ReleasedAssemblyInputPayload,
    )


def validate_oatk_assembly_data(data: OrganelleData) -> AssemblyInputPayload:
    """Validate that an assembly input can be consumed completely by Oatk."""
    payload = validate_assembly_data(data)
    common_routing_auxiliary = {
        "genome_size",
        "genome_size_evidence",
        "genome_size_report_artifact",
    }
    unsupported_auxiliary = sorted(
        name
        for name, value in payload.auxiliary.model_dump(mode="python").items()
        if name not in common_routing_auxiliary | {"hmm_profiles_artifact"} and value is not None
    )
    invalid_libraries = [
        {
            "technology": library.technology,
            "quality_state": library.quality_state,
        }
        for library in payload.long_libraries
        if library.technology != "pacbio_hifi" or library.quality_state != "ccs"
    ]
    if (
        payload.short_libraries
        or not payload.long_libraries
        or invalid_libraries
        or payload.contig_inputs
        or unsupported_auxiliary
    ):
        raise OrganelleInputError(
            code="assembly.unsupported_data_profile",
            message="released Oatk assembly accepts only pure PacBio HiFi CCS reads",
            details={
                "short_library_count": len(payload.short_libraries),
                "long_library_count": len(payload.long_libraries),
                "invalid_long_libraries": invalid_libraries,
                "contig_input_count": len(payload.contig_inputs),
                "unsupported_auxiliary": unsupported_auxiliary,
            },
        )
    return payload


def validate_himt_assembly_data(data: OrganelleData) -> AssemblyInputPayload:
    """Validate that an assembly input can be consumed completely by HiMT."""
    payload = validate_assembly_data(data)
    errors: list[dict[str, object]] = []
    if payload.short_libraries:
        errors.append({"reason": "short_reads_not_supported"})
    if payload.contig_inputs:
        errors.append({"reason": "contig_inputs_not_supported"})
    auxiliary_values = payload.auxiliary.model_dump(mode="python")
    non_empty_auxiliary = sorted(
        name for name, value in auxiliary_values.items() if value is not None
    )
    if non_empty_auxiliary:
        errors.append({"reason": "auxiliary_not_supported", "fields": non_empty_auxiliary})
    if len(payload.long_libraries) != 1:
        errors.append(
            {
                "reason": "exactly_one_long_library_required",
                "long_library_count": len(payload.long_libraries),
            }
        )
    else:
        library = payload.long_libraries[0]
        allowed = _HIMT_QUALITY_PAIRS.get(library.technology, set())
        if library.quality_state not in allowed:
            errors.append(
                {
                    "reason": "invalid_quality_pair",
                    "technology": library.technology,
                    "quality_state": library.quality_state,
                }
            )
    if errors:
        raise OrganelleInputError(
            code="assembly.unsupported_data_profile",
            message="released HiMT assembly accepts exactly one supported long-read library",
            details={"errors": errors},
        )
    return payload


def validate_pmat_assembly_data(data: OrganelleData) -> AssemblyInputPayload:
    """Validate PMAT2 long reads without inferring technology or quality."""
    payload = validate_assembly_data(data)
    errors: list[dict[str, object]] = []
    if payload.short_libraries:
        errors.append({"reason": "short_reads_not_supported"})
    if payload.contig_inputs:
        errors.append({"reason": "contig_inputs_not_supported"})
    allowed_auxiliary = {
        "correction_config_artifact",
        "genome_size",
        "genome_size_evidence",
        "genome_size_report_artifact",
        "canu_executable_artifact",
        "nextdenovo_executable_artifact",
    }
    unsupported_auxiliary = sorted(
        name
        for name, value in payload.auxiliary.model_dump(mode="python").items()
        if value is not None and name not in allowed_auxiliary
    )
    if unsupported_auxiliary:
        errors.append({"reason": "auxiliary_not_supported", "fields": unsupported_auxiliary})
    profiles = {(library.technology, library.quality_state) for library in payload.long_libraries}
    if not profiles:
        errors.append({"reason": "long_reads_required"})
    elif len(profiles) != 1:
        errors.append({"reason": "mixed_long_read_profiles", "profiles": sorted(profiles)})
    else:
        technology, quality_state = next(iter(profiles))
        if quality_state not in _PMAT_QUALITY_PAIRS.get(technology, set()):
            errors.append(
                {
                    "reason": "invalid_quality_pair",
                    "technology": technology,
                    "quality_state": quality_state,
                }
            )
    if errors:
        raise OrganelleInputError(
            code="assembly.unsupported_data_profile",
            message="PMAT2 accepts same-profile HiFi, CLR, or ONT long-read libraries",
            details={"errors": errors},
        )
    return payload


def validate_getorganelle_assembly_data(data: OrganelleData) -> AssemblyInputPayload:
    """Validate GetOrganelle short-read input without inferring layout or chemistry.

    GetOrganelle consumes exactly one Illumina short-read library, paired-end or
    single-end. Long reads, contig inputs, and every auxiliary role other than
    the four artifact-backed roles ``seed_fasta``, ``anti_seed``, ``label_genes``,
    and ``exclude_genes`` are rejected. Artifact-role and hash binding is enforced
    by :func:`validate_assembly_data` before this backend-specific gate runs.
    """
    payload = validate_assembly_data(data)
    errors: list[dict[str, object]] = []
    if payload.long_libraries:
        errors.append({"reason": "long_reads_not_supported"})
    if payload.contig_inputs:
        errors.append({"reason": "contig_inputs_not_supported"})
    if len(payload.short_libraries) == 0:
        errors.append({"reason": "short_reads_required"})
    elif len(payload.short_libraries) > 1:
        errors.append(
            {
                "reason": "too_many_short_libraries",
                "short_library_count": len(payload.short_libraries),
            }
        )
    allowed_auxiliary = {
        "seed_fasta_artifact",
        "anti_seed_artifact",
        "label_genes_artifact",
        "exclude_genes_artifact",
    }
    unsupported_auxiliary = sorted(
        name
        for name, value in payload.auxiliary.model_dump(mode="python").items()
        if value is not None and name not in allowed_auxiliary
    )
    if unsupported_auxiliary:
        errors.append({"reason": "auxiliary_not_supported", "fields": unsupported_auxiliary})
    if errors:
        raise OrganelleInputError(
            code="assembly.unsupported_data_profile",
            message="GetOrganelle accepts exactly one Illumina short-read library",
            details={"errors": errors},
        )
    return payload


def validate_pmat_graph_build_data(data: OrganelleData) -> PmatGraphBuildData:
    """Validate exact manifest/file bindings for a graphBuild continuation."""
    if data.modality != "pmat_graph_input":
        raise OrganelleInputError(
            code="assembly.invalid_continuation",
            message="PMAT graphBuild requires modality 'pmat_graph_input'",
        )
    try:
        payload = PmatGraphBuildData.model_validate(dict(data.payload))
        manifest_roles = (
            payload.subsample_manifest_artifact,
            payload.assembly_result_manifest_artifact,
        )
        if len(set(manifest_roles)) != 2:
            raise ValueError("continuation manifest roles must be distinct")
        manifests: list[DirectoryManifest] = []
        for role in manifest_roles:
            artifact = data.artifacts.get(role)
            if artifact is None:
                raise ValueError(f"continuation manifest artifact {role!r} is missing")
            path = Path(artifact.uri)
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"continuation manifest artifact {role!r} is unsafe")
            manifests.append(DirectoryManifest.model_validate_json(path.read_bytes()))
        if manifests[0].role != "pmat_subsample":
            raise ValueError("subsample manifest has the wrong directory role")
        if manifests[1].role != "pmat_assembly_result":
            raise ValueError("assembly-result manifest has the wrong directory role")
        referenced_roles = {item.artifact_role for manifest in manifests for item in manifest.files}
        declared_roles = set(payload.file_artifact_roles)
        if declared_roles != referenced_roles:
            raise ValueError("file roles must exactly match both directory manifests")
        expected_artifacts = declared_roles | set(manifest_roles)
        if set(data.artifacts) != expected_artifacts:
            raise ValueError("continuation artifacts must exactly match declared roles")
    except (OSError, ValueError) as error:
        raise OrganelleInputError(
            code="assembly.invalid_continuation",
            message=f"invalid PMAT graphBuild continuation: {error}",
        ) from error
    return payload


def validate_ptgaul_assembly_data(data: OrganelleData) -> AssemblyInputPayload:
    """Validate one raw ONT library and a required related plastome FASTA."""
    payload = validate_assembly_data(data)
    try:
        _PtgaulAssemblyInputPayload.model_validate(payload.model_dump(mode="python"))
    except Exception as error:
        raise OrganelleInputError(
            code="assembly.unsupported_data_profile",
            message="ptGAUL accepts one raw ONT library and requires a reference FASTA",
            details={"validation_error": str(error)},
        ) from error
    referenced = payload.auxiliary.reference_fasta_artifact
    if referenced is None or referenced not in data.artifacts:
        raise OrganelleInputError(
            code="assembly.missing_artifact_role",
            message="ptGAUL requires a reference_fasta_artifact role",
            details={"role": referenced},
        )
    return payload


def validate_tippo_assembly_data(data: OrganelleData) -> AssemblyInputPayload:
    """Validate exactly one pure PacBio HiFi CCS library for TIPPo."""
    payload = validate_assembly_data(data)
    invalid = [
        {"technology": item.technology, "quality_state": item.quality_state}
        for item in payload.long_libraries
        if item.technology != "pacbio_hifi" or item.quality_state != "ccs"
    ]
    unsupported_auxiliary = sorted(
        name
        for name, value in payload.auxiliary.model_dump(mode="python").items()
        if value is not None
    )
    if (
        len(payload.long_libraries) != 1
        or invalid
        or payload.short_libraries
        or payload.contig_inputs
        or unsupported_auxiliary
    ):
        raise OrganelleInputError(
            code="assembly.unsupported_data_profile",
            message="TIPPo accepts exactly one PacBio HiFi CCS library and no auxiliary inputs",
            details={
                "long_library_count": len(payload.long_libraries),
                "invalid_long_libraries": invalid,
                "short_library_count": len(payload.short_libraries),
                "contig_input_count": len(payload.contig_inputs),
                "unsupported_auxiliary": unsupported_auxiliary,
            },
        )
    return payload


def validate_ovasm_assembly_data(data: OrganelleData) -> AssemblyInputPayload:
    """One HiFi, raw ONT or raw CLR library; one Illumina library, single-end or paired-end;
    or raw ONT/CLR reads together with one Illumina library; at most a seed FASTA.

    Raw ONT/CLR reads are self-corrected by the pipeline, and corrected with the short
    reads' k-mers when an Illumina library comes with them (hybrid). Every file of a
    paired-end library is read; the pairing itself is not used.
    """
    payload = validate_assembly_data(data)
    errors: list[dict[str, object]] = []
    if payload.contig_inputs:
        errors.append({"reason": "contig_inputs_not_supported"})
    long_count, short_count = len(payload.long_libraries), len(payload.short_libraries)
    if long_count > 1 or short_count > 1 or long_count + short_count == 0:
        errors.append(
            {
                "reason": "unsupported_library_combination",
                "long_library_count": long_count,
                "short_library_count": short_count,
            }
        )
    elif long_count == 1 and short_count == 1 and payload.long_libraries[0].technology == (
        "pacbio_hifi"
    ):
        errors.append({"reason": "hybrid_needs_noisy_long_reads"})
    unsupported_long = [
        {"technology": item.technology, "quality_state": item.quality_state}
        for item in payload.long_libraries
        if (item.technology, item.quality_state)
        not in {("pacbio_hifi", "ccs"), ("ont", "raw"), ("pacbio_clr", "raw")}
    ]
    if unsupported_long:
        errors.append({"reason": "long_read_profile_not_supported", "libraries": unsupported_long})
    unsupported_auxiliary = sorted(
        name
        for name, value in payload.auxiliary.model_dump(mode="python").items()
        if value is not None and name != "seed_fasta_artifact"
    )
    if unsupported_auxiliary:
        errors.append({"reason": "auxiliary_not_supported", "fields": unsupported_auxiliary})
    if errors:
        raise OrganelleInputError(
            code="assembly.unsupported_data_profile",
            message=(
                "ovasm accepts one PacBio HiFi, raw ONT or raw PacBio CLR library, one "
                "single-end or paired-end Illumina library, or raw ONT/CLR reads with one "
                "Illumina library, and no auxiliary input other than seed_fasta"
            ),
            details={"errors": errors},
        )
    return payload


def validate_novoplasty_assembly_data(data: OrganelleData) -> AssemblyInputPayload:
    """One Illumina PE library plus documented sequence/configuration inputs."""
    payload = validate_assembly_data(data)
    allowed = {
        "seed_fasta_artifact",
        "reference_fasta_artifact",
        "chloroplast_fasta_artifact",
        "genome_range",
    }
    if (
        len(payload.short_libraries) != 1
        or payload.long_libraries
        or payload.contig_inputs
        or any(
            value is not None and name not in allowed
            for name, value in payload.auxiliary.model_dump().items()
        )
    ):
        raise OrganelleInputError(
            code="assembly.unsupported_data_profile",
            message="NOVOPlasty accepts one Illumina PE library with seed/reference/chloroplast FASTA and genome_range",
        )
    library = payload.short_libraries[0]
    if library.layout != "paired_end" or library.insert_size is None:
        raise OrganelleInputError(
            code="assembly.unsupported_data_profile",
            message="NOVOPlasty requires paired-end reads and an insert_size estimate",
        )
    return payload


def validate_released_assembly_data(data: OrganelleData) -> AssemblyInputPayload:
    """Validate that an assembly input is supported by at least one released backend."""
    # Generic modality/role errors are not backend-specific and must surface first.
    validate_assembly_data(data)
    errors: list[tuple[str, OrganelleInputError]] = []
    try:
        return validate_novoplasty_assembly_data(data)
    except OrganelleInputError as error:
        errors.append(("novoplasty", error))
    try:
        return validate_oatk_assembly_data(data)
    except OrganelleInputError as error:
        errors.append(("oatk", error))
    try:
        return validate_himt_assembly_data(data)
    except OrganelleInputError as error:
        errors.append(("himt", error))
    try:
        return validate_getorganelle_assembly_data(data)
    except OrganelleInputError as error:
        errors.append(("getorganelle", error))
    try:
        return validate_pmat_assembly_data(data)
    except OrganelleInputError as error:
        errors.append(("pmat", error))
    try:
        return validate_tippo_assembly_data(data)
    except OrganelleInputError as error:
        errors.append(("tippo", error))
    try:
        return validate_ptgaul_assembly_data(data)
    except OrganelleInputError as error:
        errors.append(("ptgaul", error))
    try:
        return validate_ovasm_assembly_data(data)
    except OrganelleInputError as error:
        errors.append(("ovasm", error))
    raise OrganelleInputError(
        code="assembly.unsupported_data_profile",
        message="released assembly accepts only Oatk, HiMT, GetOrganelle, PMAT, TIPPo, ptGAUL, or ovasm supported profiles",
        details={key: error.as_dict() for key, error in errors},
    )


def sequencing_reads_data_contract() -> DataContract:
    """Construct the generic assembly contract without import-time catalog coupling."""
    from organelleverse.operations.data_contracts import DataContract

    return DataContract(
        modality="sequencing_reads",
        schema_version="organelleverse.assembly-input.v1",
        validator=validate_assembly_data,
        schema_provider=assembly_data_json_schema,
    )


def oatk_sequencing_reads_data_contract() -> DataContract:
    """Construct the Oatk release contract without mutating global registries."""
    from organelleverse.operations.data_contracts import DataContract

    return DataContract(
        modality="sequencing_reads",
        schema_version="organelleverse.assembly-input.v1",
        validator=validate_oatk_assembly_data,
        schema_provider=oatk_assembly_data_json_schema,
    )


def himt_sequencing_reads_data_contract() -> DataContract:
    """Construct the HiMT release contract without mutating global registries."""
    from organelleverse.operations.data_contracts import DataContract

    return DataContract(
        modality="sequencing_reads",
        schema_version="organelleverse.assembly-input.v1",
        validator=validate_himt_assembly_data,
        schema_provider=himt_assembly_data_json_schema,
    )


def pmat_sequencing_reads_data_contract() -> DataContract:
    """Construct the released PMAT2 sequencing-read contract."""
    from organelleverse.operations.data_contracts import DataContract

    return DataContract(
        modality="sequencing_reads",
        schema_version="organelleverse.assembly-input.v1",
        validator=validate_pmat_assembly_data,
        schema_provider=pmat_assembly_data_json_schema,
    )


def pmat_graph_input_data_contract() -> DataContract:
    """Construct the released PMAT2 graphBuild continuation contract."""
    from organelleverse.operations.data_contracts import DataContract

    return DataContract(
        modality="pmat_graph_input",
        schema_version="organelleverse.pmat-graph-input.v1",
        validator=validate_pmat_graph_build_data,
        schema_provider=pmat_graph_input_json_schema,
    )


def released_assembly_sequencing_reads_data_contract() -> DataContract:
    """Construct the combined released assembly contract.

    The catalog registers exactly one contract per modality. The runtime validator
    accepts a payload if either backend-specific validator accepts it, preserving
    the most useful rejection details when both reject.
    """
    from organelleverse.operations.data_contracts import DataContract

    return DataContract(
        modality="sequencing_reads",
        schema_version="organelleverse.assembly-input.v1",
        validator=validate_released_assembly_data,
        schema_provider=released_assembly_data_json_schema,
    )


SEQUENCING_READS_DATA_CONTRACT = sequencing_reads_data_contract()
OATK_SEQUENCING_READS_DATA_CONTRACT = oatk_sequencing_reads_data_contract()
HIMT_SEQUENCING_READS_DATA_CONTRACT = himt_sequencing_reads_data_contract()
RELEASED_ASSEMBLY_SEQUENCING_READS_DATA_CONTRACT = (
    released_assembly_sequencing_reads_data_contract()
)
