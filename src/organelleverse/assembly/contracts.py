"""Immutable v1 contracts for structured assembly inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Set
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeAlias, cast, get_args

from annotated_types import Ge, Gt, Le
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from pydantic.main import IncEx

from organelleverse.assembly.environment_contracts import (
    BackendVersionSelector,
    EnvironmentHint,
    EnvironmentSource,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.frozen import thaw_json
from organelleverse.operations.parameters import OperationParameterModel

ArtifactSource: TypeAlias = str | Path | ArtifactRef
AbstractSetIntStr: TypeAlias = Set[int] | Set[str]
MappingIntStrAny: TypeAlias = Mapping[int, Any] | Mapping[str, Any]
ShortTechnology = Literal["illumina"]
ShortLayout = Literal["paired_end", "single_end"]
LongTechnology = Literal["pacbio_hifi", "pacbio_clr", "ont"]
QualityState = Literal["raw", "corrected", "duplex", "hq", "ccs"]
AssemblyMethod: TypeAlias = Literal[
    "oatk", "himt", "getorganelle", "pmat", "tippo", "ptgaul", "novoplasty", "ovasm"
]


def assembly_method_ids() -> tuple[AssemblyMethod, ...]:
    return cast(tuple[AssemblyMethod, ...], get_args(AssemblyMethod))


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class _FrozenInputModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    def _restore_dumped_models(
        self,
        values: dict[str, Any],
        updated_fields: set[str],
    ) -> None:
        for name in values.keys() - updated_fields:
            original = getattr(self, name)
            if isinstance(original, BaseModel):
                values[name] = type(original).model_validate(values[name])

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        values = self.model_dump(mode="python", round_trip=True)
        if deep:
            values = deepcopy(values)
        updated_fields: set[str] = set()
        if update is not None:
            values.update(update)
            updated_fields.update(update)
        self._restore_dumped_models(values, updated_fields)
        return type(self).model_validate(values)

    def copy(
        self,
        *,
        include: AbstractSetIntStr | MappingIntStrAny | None = None,
        exclude: AbstractSetIntStr | MappingIntStrAny | None = None,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        values = self.model_dump(
            mode="python",
            include=cast(IncEx | None, include),
            exclude=cast(IncEx | None, exclude),
            round_trip=True,
        )
        if deep:
            values = deepcopy(values)
        updated_fields: set[str] = set()
        if update is not None:
            values.update(update)
            updated_fields.update(update)
        self._restore_dumped_models(values, updated_fields)
        return type(self).model_validate(values)


class ShortReadLibrary(_FrozenInputModel):
    technology: ShortTechnology
    layout: ShortLayout
    read1: ArtifactSource
    read2: ArtifactSource | None = None
    read_length: int = Field(gt=0)
    insert_size: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_layout(self) -> ShortReadLibrary:
        if self.layout == "paired_end" and self.read2 is None:
            raise ValueError("paired_end requires read2")
        if self.layout == "single_end" and self.read2 is not None:
            raise ValueError("single_end forbids read2")
        return self


class LongReadLibrary(_FrozenInputModel):
    technology: LongTechnology
    quality_state: QualityState
    reads: ArtifactSource

    @model_validator(mode="after")
    def validate_quality_state(self) -> LongReadLibrary:
        allowed = {
            "pacbio_hifi": {"ccs"},
            "pacbio_clr": {"raw", "corrected"},
            "ont": {"raw", "corrected", "duplex", "hq"},
        }
        if self.quality_state not in allowed[self.technology]:
            if self.technology == "pacbio_hifi":
                raise ValueError("pacbio_hifi requires quality_state='ccs'")
            raise ValueError(
                f"{self.technology} does not support quality_state={self.quality_state!r}"
            )
        return self


class ContigInput(_FrozenInputModel):
    fasta: ArtifactSource


class GenomeSizeEvidence(_FrozenInputModel):
    """Traceable nuclear genome-size evidence bound to one assembly invocation.

    ``user`` evidence records a person-supplied number and must not carry NCBI
    report fields. ``ncbi_assembly`` evidence records a complete NCBI Datasets
    assembly-report selection and the content address of its canonical report
    artifact; the artifact linkage itself (role presence and SHA256 equality) is
    enforced by :func:`validate_assembly_data`, which sees both the evidence and
    the resolved artifact map.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source: Literal["user", "ncbi_assembly"]
    genome_size_bp: int = Field(gt=0)
    taxon_id: int | None = Field(default=None, gt=0)
    scientific_name: str | None = None
    assembly_accession: str | None = None
    assembly_level: str | None = None
    refseq_category: str | None = None
    report_uri: str | None = None
    report_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    report_artifact_role: str | None = None

    @model_validator(mode="after")
    def validate_source_consistency(self) -> GenomeSizeEvidence:
        report_fields = (self.report_uri, self.report_sha256, self.report_artifact_role)
        if self.source == "user":
            if any(field is not None for field in report_fields):
                raise ValueError("user genome-size evidence must not carry report fields")
            return self
        required_fields = {
            "taxon_id": self.taxon_id,
            "assembly_accession": self.assembly_accession,
            "assembly_level": self.assembly_level,
            "report_uri": self.report_uri,
            "report_sha256": self.report_sha256,
            "report_artifact_role": self.report_artifact_role,
        }
        missing = sorted(name for name, value in required_fields.items() if value is None)
        if missing:
            raise ValueError(
                "ncbi_assembly genome-size evidence requires non-null fields: " + ", ".join(missing)
            )
        return self


class AssemblyAuxiliary(_FrozenInputModel):
    seed_fasta: ArtifactSource | None = None
    reference_fasta: ArtifactSource | None = None
    reference_genbank: ArtifactSource | None = None
    chloroplast_fasta: ArtifactSource | None = None
    hmm_profiles: ArtifactSource | None = None
    correction_config: ArtifactSource | None = None
    genome_size: int | None = Field(default=None, gt=0)
    genome_range: tuple[int, int] | None = None
    genome_size_evidence: GenomeSizeEvidence | None = None
    genome_size_report: ArtifactSource | None = None
    canu_executable: ArtifactSource | None = None
    nextdenovo_executable: ArtifactSource | None = None
    anti_seed: ArtifactSource | None = None
    label_genes: ArtifactSource | None = None
    exclude_genes: ArtifactSource | None = None

    @model_validator(mode="after")
    def validate_genome_range(self) -> AssemblyAuxiliary:
        if self.genome_range is not None and self.genome_range[0] >= self.genome_range[1]:
            raise ValueError("genome_range minimum must be smaller than maximum")
        return self

    @model_validator(mode="after")
    def validate_genome_size_evidence(self) -> AssemblyAuxiliary:
        evidence = self.genome_size_evidence
        if evidence is None:
            return self
        if self.genome_size is None or self.genome_size != evidence.genome_size_bp:
            raise ValueError("genome_size must equal genome_size_evidence.genome_size_bp")
        if evidence.source == "user" and self.genome_size_report is not None:
            raise ValueError("user genome-size evidence must not reference a report artifact")
        if evidence.source == "ncbi_assembly" and self.genome_size_report is None:
            raise ValueError(
                "ncbi_assembly genome-size evidence requires the genome_size_report artifact"
            )
        return self


class ShortLibraryContract(_FrozenInputModel):
    technology: ShortTechnology
    layout: ShortLayout
    read1_artifact: str = Field(min_length=1)
    read2_artifact: str | None = None
    read_length: int = Field(gt=0)
    insert_size: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_layout(self) -> ShortLibraryContract:
        if self.layout == "paired_end" and self.read2_artifact is None:
            raise ValueError("paired_end requires read2_artifact")
        if self.layout == "single_end" and self.read2_artifact is not None:
            raise ValueError("single_end forbids read2_artifact")
        return self


class LongLibraryContract(_FrozenInputModel):
    technology: LongTechnology
    quality_state: QualityState
    reads_artifact: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_quality_state(self) -> LongLibraryContract:
        allowed = {
            "pacbio_hifi": {"ccs"},
            "pacbio_clr": {"raw", "corrected"},
            "ont": {"raw", "corrected", "duplex", "hq"},
        }
        if self.quality_state not in allowed[self.technology]:
            raise ValueError(
                f"{self.technology} does not support quality_state={self.quality_state!r}"
            )
        return self


class ContigContract(_FrozenInputModel):
    fasta_artifact: str = Field(min_length=1)


class AuxiliaryContract(_FrozenInputModel):
    seed_fasta_artifact: str | None = None
    reference_fasta_artifact: str | None = None
    reference_genbank_artifact: str | None = None
    chloroplast_fasta_artifact: str | None = None
    hmm_profiles_artifact: str | None = None
    correction_config_artifact: str | None = None
    genome_size: int | None = Field(default=None, gt=0)
    genome_range: tuple[int, int] | None = None
    genome_size_evidence: GenomeSizeEvidence | None = None
    genome_size_report_artifact: str | None = None
    canu_executable_artifact: str | None = None
    nextdenovo_executable_artifact: str | None = None
    anti_seed_artifact: str | None = None
    label_genes_artifact: str | None = None
    exclude_genes_artifact: str | None = None

    @model_validator(mode="after")
    def validate_genome_range(self) -> AuxiliaryContract:
        if self.genome_range is not None and self.genome_range[0] >= self.genome_range[1]:
            raise ValueError("genome_range minimum must be smaller than maximum")
        return self

    @model_validator(mode="after")
    def validate_genome_size_evidence(self) -> AuxiliaryContract:
        evidence = self.genome_size_evidence
        if evidence is None:
            return self
        if self.genome_size is None:
            raise ValueError("genome_size_evidence requires genome_size")
        if self.genome_size != evidence.genome_size_bp:
            raise ValueError("genome_size must equal genome_size_evidence.genome_size_bp")
        if evidence.source == "ncbi_assembly":
            if evidence.report_artifact_role != "genome_size_report":
                raise ValueError(
                    "ncbi_assembly genome-size evidence report_artifact_role must be "
                    "'genome_size_report'"
                )
            if self.genome_size_report_artifact is None:
                raise ValueError(
                    "ncbi_assembly genome-size evidence requires the "
                    "genome_size_report_artifact role"
                )
            if self.genome_size_report_artifact != evidence.report_artifact_role:
                raise ValueError(
                    "genome_size_report_artifact must equal "
                    "genome_size_evidence.report_artifact_role"
                )
        elif evidence.source == "user" and self.genome_size_report_artifact is not None:
            raise ValueError("user genome-size evidence must not reference a report artifact")
        return self


class AssemblyInputPayload(_FrozenInputModel):
    contract_version: Literal["organelleverse.assembly-input.v1"] = (
        "organelleverse.assembly-input.v1"
    )
    short_libraries: tuple[ShortLibraryContract, ...] = ()
    long_libraries: tuple[LongLibraryContract, ...] = ()
    contig_inputs: tuple[ContigContract, ...] = ()
    auxiliary: AuxiliaryContract = AuxiliaryContract()

    @model_validator(mode="after")
    def require_primary_input(self) -> AssemblyInputPayload:
        if not (self.short_libraries or self.long_libraries or self.contig_inputs):
            raise ValueError("assembly input requires reads or contigs")
        return self


def _input_error(code: str, message: str, **details: object) -> OrganelleInputError:
    return OrganelleInputError(code=code, message=message, details=details)


def _referenced_roles(payload: AssemblyInputPayload) -> set[str]:
    roles = {library.read1_artifact for library in payload.short_libraries}
    roles.update(
        library.read2_artifact
        for library in payload.short_libraries
        if library.read2_artifact is not None
    )
    roles.update(library.reads_artifact for library in payload.long_libraries)
    roles.update(item.fasta_artifact for item in payload.contig_inputs)
    roles.update(
        value
        for name, value in payload.auxiliary.model_dump(mode="python").items()
        if name.endswith("_artifact")
        and name != "genome_size_report_artifact"
        and isinstance(value, str)
    )
    # The genome-size report artifact is referenced only when NCBI evidence
    # points at it; a report artifact with no bound evidence is unused.
    evidence = payload.auxiliary.genome_size_evidence
    if evidence is not None and evidence.report_artifact_role is not None:
        roles.add(evidence.report_artifact_role)
    return roles


def validate_assembly_data(data: OrganelleData) -> AssemblyInputPayload:
    """Validate a v1 sequencing-read data object and its artifact roles."""
    if type(data) is not OrganelleData or data.modality != "sequencing_reads":
        raise _input_error(
            "assembly.unsupported_data_profile",
            "assembly requires sequencing_reads OrganelleData",
            provided_type=type(data).__name__,
            provided_modality=getattr(data, "modality", None),
        )
    try:
        payload = AssemblyInputPayload.model_validate(thaw_json(data.payload))
    except ValidationError as error:
        raise _input_error(
            "assembly.unsupported_data_profile",
            "sequencing-read payload is invalid",
            validation_errors=[
                {
                    "location": [str(item) for item in detail["loc"]],
                    "message": detail["msg"],
                    "type": detail["type"],
                }
                for detail in error.errors(include_url=False)
            ],
        ) from error
    referenced = _referenced_roles(payload)
    available = set(data.artifacts)
    missing = sorted(referenced - available)
    if missing:
        raise _input_error(
            "assembly.missing_artifact_role",
            "assembly payload references a missing artifact role",
            artifact_role=missing[0],
            artifact_roles=missing,
        )
    unused = sorted(available - referenced)
    if unused:
        raise _input_error(
            "assembly.unused_artifact_role",
            "assembly data contains unreferenced artifacts",
            artifact_role=unused[0],
            artifact_roles=unused,
        )
    evidence = payload.auxiliary.genome_size_evidence
    if (
        evidence is not None
        and evidence.source == "ncbi_assembly"
        and evidence.report_artifact_role is not None
    ):
        report_role = evidence.report_artifact_role
        report_artifact = data.artifacts.get(report_role)
        if report_artifact is not None and report_artifact.sha256 != evidence.report_sha256:
            raise _input_error(
                "assembly.genome_size_evidence_mismatch",
                "genome_size_report artifact sha256 does not match evidence report_sha256",
                artifact_role=report_role,
                expected_sha256=evidence.report_sha256,
                observed_sha256=report_artifact.sha256,
            )
    return payload


class OatkParameters(OperationParameterModel):
    """Strict Oatk backend parameters exposed to an Agent.

    Only the scientifically important ``minimum_kmer_coverage`` is a user-tunable
    field. The pinned Oatk 1.0 defaults (syncmer/k-mer size 1001 and minimum k-mer
    coverage 30) are recorded in the semantic manifest via
    :func:`effective_backend_parameters` rather than exposed as untested public knobs,
    so this model deliberately carries no ``kmer_size`` field.
    """

    backend: Literal["oatk"] = "oatk"
    minimum_kmer_coverage: Annotated[int, Ge(1)] = 30


class HimtParameters(OperationParameterModel):
    """Strict HiMT backend parameters exposed to an Agent."""

    backend: Literal["himt"] = "himt"
    species: Literal["plant", "animal"] = "plant"
    kmer_length: Annotated[int, Ge(1)] = 21
    head_number: Annotated[int, Ge(1)] = 4
    extract_parallel: Annotated[int, Ge(1)] = 2
    base_number: Annotated[int, Ge(3), Le(4)] = 3
    filter_depth: Annotated[int, Ge(0)] = 0
    filter_percentage: Annotated[float, Ge(0), Le(1)] = 0.3
    proportion: Annotated[float, Ge(0), Le(1)] = 0.0
    accuracy: Annotated[float, Ge(0), Le(1)] | None = None
    no_flye_meta: bool = False
    normalize_depth: int = 0


class PmatParameters(OperationParameterModel):
    """Complete non-path ``PMAT autoMito`` parameters for PMAT2 v2.1.5."""

    backend: Literal["pmat"] = "pmat"
    kmer_size: Annotated[int, Ge(1), Le(31)] = 31
    correction_task: Literal["auto", "run", "skip"] = "auto"
    correction_software: Literal["nextdenovo", "canu"] = "nextdenovo"
    subsample_factor: Annotated[float, Gt(0), Le(1)] = 1.0
    random_seed: Annotated[int, Ge(0)] = 6
    long_read_break_length: Annotated[int, Ge(100)] = 20_000
    minimum_overlap_identity: Annotated[int, Ge(1), Le(100)] = 90
    minimum_overlap_length: Annotated[int, Ge(1)] = 40
    keep_sequences_in_memory: bool = False


class GetOrganelleParameters(OperationParameterModel):
    """Complete non-path GetOrganelle parameters exposed to an Agent.

    Every field maps one GetOrganelle CLI option. A ``None`` value means the
    upstream option is omitted so GetOrganelle presets and defaults remain
    intact; a boolean ``False`` means the flag is absent rather than inverted.
    The model is closed (``extra="forbid"``), carries no raw argv channel, and
    exposes no dependency path field; cross-field constraints are enforced in
    :func:`effective_backend_parameters` because definition-time validators are
    forbidden on :class:`OperationParameterModel`.
    """

    backend: Literal["getorganelle"] = "getorganelle"
    max_reads: Annotated[int, Ge(1)] | None = None
    reduce_reads_for_coverage: Annotated[float, Gt(10)] | Literal["inf"] | None = None
    max_ignore_percent: Annotated[float, Ge(0), Le(1)] | None = None
    phred_offset: Annotated[int, Ge(33), Le(33)] | Annotated[int, Ge(64), Le(64)] | None = None
    min_quality_score: int | None = None
    output_prefix: str | None = None
    output_per_round: bool = False
    zip_files: bool = False
    keep_temp: bool = False
    fast: bool = False
    memory_save: bool = False
    memory_unlimited: bool = False
    word_size: Annotated[float, Gt(0)] | None = None
    pregroup_word_size: Annotated[float, Gt(0)] | None = None
    max_rounds: Annotated[int, Ge(2)] | Literal["inf"] | None = None
    max_words: Annotated[int, Ge(1)] | None = None
    jump_step: Annotated[int, Ge(1)] | None = None
    mesh_size: Annotated[int, Ge(1)] | None = None
    bowtie2_options: str | None = None
    larger_auto_word_size: bool = False
    target_genome_size: Annotated[int, Ge(1)] | None = None
    max_extending_length: Annotated[int, Ge(0)] | Literal["auto", "inf"] | None = None
    spades_kmers: tuple[Annotated[int, Ge(1)], ...] | None = None
    spades_options: str | None = None
    no_spades: bool = False
    ignore_kmer: Annotated[int, Ge(0)] | None = None
    disentangle_depth_factor: Annotated[float, Gt(0)] | None = None
    contamination_depth: Annotated[float, Gt(0)] | None = None
    contamination_similarity: Annotated[float, Ge(0), Le(1)] | None = None
    no_degenerate: bool = False
    degenerate_depth: Annotated[float, Gt(0)] | None = None
    degenerate_similarity: Annotated[float, Ge(0), Le(1)] | None = None
    disentangle_time_limit: Annotated[int, Ge(1)] | None = None
    expected_max_size: Annotated[int, Ge(1)] | None = None
    expected_min_size: Annotated[int, Ge(1)] | None = None
    reverse_lsc: bool = False
    max_paths: Annotated[int, Ge(1)] | None = None
    pregrouped_reads: Annotated[int, Ge(0)] | None = None
    index_in_memory: bool = False
    remove_duplicates: Annotated[int, Ge(0)] | None = None
    flush_step: Annotated[int, Ge(1)] | Literal["inf"] | None = None
    random_seed: int | None = None
    verbose: bool = False


class TippoParameters(OperationParameterModel):
    """TIPPo v2.4 parameters; method is fixed to the documented HiFi plastid slice."""

    backend: Literal["tippo"] = "tippo"


class PtgaulParameters(OperationParameterModel):
    """Documented ptGAUL parameters used by the plastid long-read pipeline."""

    backend: Literal["ptgaul"] = "ptgaul"
    genome_size: Annotated[int, Ge(1)] = 160_000
    coverage: Annotated[int, Ge(1)] = 50
    minimum_read_length: Annotated[int, Ge(1)] = 3_000


class NovoplastyParameters(OperationParameterModel):
    """NOVOPlasty 4.3.5 config defaults (upstream config.txt).

    Read length and insert size come from ShortReadLibrary. Seed/reference
    paths use AssemblyAuxiliary. Plant mitochondria require an explicit range,
    seed and chloroplast assembly; no universal plant mitochondrial size exists.
    """

    backend: Literal["novoplasty"] = "novoplasty"
    kmer_size: Annotated[int, Ge(1)] = 33
    genome_range: tuple[Annotated[int, Ge(1)], Annotated[int, Ge(1)]] | None = None
    insert_size_auto: bool = True
    use_quality_scores: bool = False
    extended_log: bool = False
    save_assembled_reads: bool = False


class OvasmParameters(OperationParameterModel):
    """Parameters of the native ovasm pipeline (the options of the desktop assembly tools).

    ``read_set``: ``whole_genome`` reads are recruited first; ``target_reads`` are organelle
    reads already (e.g. an earlier ``ovasm recruit``) and are assembled as given.
    ``seed_source``: the bundled seed database, the ``seed_fasta`` auxiliary input
    (``custom``), or no seeds at all (``discover``, HiFi only). ``also_discover`` adds the
    seed-free depth clusters called as the target to the seed-recruited reads (HiFi only).
    ``sample`` names the segments and PanSN paths of the unified graph (OV-GFA).
    """

    backend: Literal["ovasm"] = "ovasm"
    read_set: Literal["whole_genome", "target_reads"] = "whole_genome"
    seed_source: Literal["builtin", "custom", "discover"] = "builtin"
    also_discover: bool = False
    sample: str = "sample"


# The structured parameter union admitted by the assembly operation. Each backend
# exposes a closed ``OperationParameterModel`` subclass rather than a free-form
# mapping.
ReleasedAssemblyBackendParameters: TypeAlias = (
    OatkParameters
    | HimtParameters
    | GetOrganelleParameters
    | PmatParameters
    | TippoParameters
    | PtgaulParameters
    | NovoplastyParameters
    | OvasmParameters
)
AssemblyBackendParameters: TypeAlias = ReleasedAssemblyBackendParameters


def effective_pmat_parameters(
    request: AssemblyRequest,
    payload: AssemblyInputPayload,
    provided: PmatParameters | None,
) -> dict[str, object]:
    """Resolve PMAT2 policy into one closed, path-free parameter block."""
    return _effective_pmat_parameters(
        payload,
        provided,
        organelle=request.organelle,
        taxon_group=request.taxon_group,
    )


def _effective_pmat_parameters(
    payload: AssemblyInputPayload,
    provided: PmatParameters | None,
    *,
    organelle: Literal["mitochondrion", "plastid"],
    taxon_group: Literal["plant", "animal", "fungi"],
) -> dict[str, object]:
    if not payload.long_libraries:
        raise _input_error(
            "assembly.unsupported_data_profile",
            "PMAT requires at least one long-read library",
        )
    profiles = {(library.technology, library.quality_state) for library in payload.long_libraries}
    if len(profiles) != 1:
        raise _input_error(
            "assembly.unsupported_data_profile",
            "PMAT libraries must use one technology and quality profile",
        )
    technology, quality_state = next(iter(profiles))
    seqtype = {
        "pacbio_hifi": "hifi",
        "pacbio_clr": "clr",
        "ont": "ont",
    }[technology]
    parameters = provided if provided is not None else PmatParameters()
    correction_task = parameters.correction_task
    if correction_task == "auto":
        correction_task = "run" if quality_state == "raw" else "skip"
    if technology == "pacbio_hifi" and correction_task == "run":
        raise _input_error(
            "assembly.pmat_hifi_correction_unsupported",
            "PMAT2 does not error-correct HiFi input",
        )
    if organelle == "plastid" and taxon_group != "plant":
        raise _input_error(
            "assembly.pmat_non_plant_plastid_unsupported",
            "PMAT2 plastid assembly requires plant taxon mode",
        )
    effective = parameters.model_dump(mode="json")
    effective.update(
        {
            "correction_task": correction_task,
            "seqtype": seqtype,
            "target": "mt" if organelle == "mitochondrion" else "pt",
            "taxon": {"plant": 0, "animal": 1, "fungi": 2}[taxon_group],
            "genome_size": payload.auxiliary.genome_size,
        }
    )
    return effective


#: Canonical mapping from request (organelle, taxon_group) onto the exact
#: GetOrganelle ``-F`` target. This is the single source of truth shared by the
#: effective-parameter resolver, the resource provider, and the adapter.
GETORGANELLE_TARGETS: Mapping[tuple[str, str], str] = {
    ("plastid", "plant"): "embplant_pt",
    ("plastid", "animal"): "other_pt",
    ("plastid", "fungi"): "other_pt",
    ("mitochondrion", "plant"): "embplant_mt",
    ("mitochondrion", "animal"): "animal_mt",
    ("mitochondrion", "fungi"): "fungus_mt",
}


def _effective_getorganelle_parameters(
    provided: GetOrganelleParameters | None,
    *,
    organelle: Literal["mitochondrion", "plastid"],
    taxon_group: Literal["plant", "animal", "fungi"],
) -> dict[str, object]:
    """Resolve GetOrganelle policy into one closed, path-free parameter block.

    Cross-field constraints that :class:`OperationParameterModel` cannot express
    at definition time are enforced here: at most one memory preset, clean
    prefix/option strings, unique odd SPAdes kmers, and an ordered expected-size
    range. The organelle/taxon target maps exactly onto GetOrganelle's mode names.
    """
    parameters = provided if provided is not None else GetOrganelleParameters()

    preset_names = [
        name
        for name, flag in (
            ("fast", parameters.fast),
            ("memory_save", parameters.memory_save),
            ("memory_unlimited", parameters.memory_unlimited),
        )
        if flag
    ]
    if len(preset_names) > 1:
        raise _input_error(
            "assembly.getorganelle_conflicting_memory_preset",
            "GetOrganelle accepts at most one memory preset",
            presets=preset_names,
        )

    for name in ("output_prefix", "bowtie2_options", "spades_options"):
        value = getattr(parameters, name)
        if value is None:
            continue
        if value == "" or "\0" in value or "\n" in value:
            raise _input_error(
                "assembly.getorganelle_invalid_option_string",
                "GetOrganelle option strings must be non-empty and free of NUL/newline",
                field=name,
            )

    kmers = parameters.spades_kmers
    if kmers is not None:
        if any(kmer % 2 == 0 for kmer in kmers):
            raise _input_error(
                "assembly.getorganelle_invalid_spades_kmers",
                "GetOrganelle SPAdes kmers must be odd",
                reason="even",
                kmers=list(kmers),
            )
        if len(set(kmers)) != len(kmers):
            raise _input_error(
                "assembly.getorganelle_invalid_spades_kmers",
                "GetOrganelle SPAdes kmers must be unique",
                reason="duplicate",
                kmers=list(kmers),
            )

    if (
        parameters.expected_min_size is not None
        and parameters.expected_max_size is not None
        and parameters.expected_min_size > parameters.expected_max_size
    ):
        raise _input_error(
            "assembly.getorganelle_invalid_size_range",
            "GetOrganelle expected_min_size must not exceed expected_max_size",
            expected_min_size=parameters.expected_min_size,
            expected_max_size=parameters.expected_max_size,
        )

    target = GETORGANELLE_TARGETS[(organelle, taxon_group)]
    effective = parameters.model_dump(mode="json")
    effective["target"] = target
    return effective


def _ovasm_read_type(payload: AssemblyInputPayload) -> str:
    """The read type of the native pipeline for one validated ovasm input."""
    if payload.short_libraries and payload.long_libraries:
        return "hybrid"
    if payload.short_libraries:
        return "short_read"
    technology = payload.long_libraries[0].technology
    return {"pacbio_hifi": "hifi_only", "ont": "ont_only", "pacbio_clr": "clr_only"}[technology]


def _effective_ovasm_parameters(
    payload: AssemblyInputPayload,
    parameters: OvasmParameters,
) -> dict[str, object]:
    """Resolve the ovasm options; an option that would not be used is refused, not ignored."""
    read_type = _ovasm_read_type(payload)
    has_seed = payload.auxiliary.seed_fasta_artifact is not None
    whole_genome = parameters.read_set == "whole_genome"
    problem = None
    if not whole_genome and (
        has_seed or parameters.seed_source != "builtin" or parameters.also_discover
    ):
        problem = "target_reads are assembled as given: seeds and discovery do not apply"
    elif whole_genome and parameters.seed_source == "custom" and not has_seed:
        raise _input_error(
            "assembly.missing_auxiliary",
            "ovasm seed_source='custom' requires the seed_fasta auxiliary input",
            missing=["seed_fasta"],
        )
    elif has_seed and parameters.seed_source != "custom":
        problem = "seed_fasta is used only with seed_source='custom'"
    elif read_type != "hifi_only" and (
        parameters.seed_source == "discover" or parameters.also_discover
    ):
        problem = "seed-free discovery needs PacBio HiFi reads"
    elif parameters.also_discover and parameters.seed_source == "discover":
        problem = "seed_source='discover' already uses discovery; drop also_discover"
    elif not parameters.sample or any(
        not (char.isascii() and (char.isalnum() or char in "_-")) for char in parameters.sample
    ):
        # PanSN separates its fields with '#' and segment names put a '.' after the sample;
        # the name is also a token of the run manifest's stable argv
        problem = "sample must be letters, digits, '_' or '-'"
    if problem is not None:
        raise _input_error(
            "assembly.ovasm_invalid_options",
            f"ovasm: {problem}",
            read_type=read_type,
            read_set=parameters.read_set,
            seed_source=parameters.seed_source,
            also_discover=parameters.also_discover,
            seed_fasta=has_seed,
        )
    effective = parameters.model_dump(mode="json")
    effective["read_type"] = read_type
    return effective


def effective_backend_parameters(
    backend_id: AssemblyMethod,
    provided: AssemblyBackendParameters | None,
    *,
    payload: AssemblyInputPayload | None = None,
    organelle: Literal["mitochondrion", "plastid"] | None = None,
    taxon_group: Literal["plant", "animal", "fungi"] | None = None,
) -> dict[str, object]:
    """Return the closed, recorded effective parameter block for a selected backend.

    ``kmer_size`` is the pinned Oatk 1.0 default recorded for traceability; it is not
    a field of :class:`OatkParameters` and therefore cannot be tuned by an Agent.
    """
    if provided is not None and provided.backend != backend_id:
        raise ValueError("backend_parameters must match selected backend")
    if backend_id == "novoplasty":
        parameters = (
            provided if isinstance(provided, NovoplastyParameters) else NovoplastyParameters()
        )
        if payload is None or organelle is None:
            raise ValueError("NOVOPlasty parameters require an assembly payload and organelle")
        interval = parameters.genome_range or payload.auxiliary.genome_range
        if (
            parameters.genome_range
            and payload.auxiliary.genome_range
            and parameters.genome_range != payload.auxiliary.genome_range
        ):
            raise _input_error("assembly.novoplasty_invalid_range", "conflicting genome ranges")
        if interval is None:
            if organelle == "plastid":
                interval = (120_000, 200_000)
            elif taxon_group == "animal":
                interval = (12_000, 20_000)
            else:
                raise _input_error(
                    "assembly.novoplasty_range_required",
                    "provide the expected mitochondrial genome_range",
                )
        if interval[0] >= interval[1]:
            raise _input_error(
                "assembly.novoplasty_invalid_range",
                "genome_range minimum must be smaller than maximum",
            )
        if parameters.kmer_size >= payload.short_libraries[0].read_length:
            raise _input_error(
                "assembly.novoplasty_invalid_kmer", "k-mer must be shorter than the reads"
            )
        if organelle == "mitochondrion" and payload.auxiliary.seed_fasta_artifact is None:
            raise _input_error(
                "assembly.novoplasty_seed_required",
                "provide a mitochondrial seed_fasta; the bundled rbcL seed is plastid-only",
            )
        if (
            organelle == "mitochondrion"
            and taxon_group == "plant"
            and payload.auxiliary.chloroplast_fasta_artifact is None
        ):
            raise _input_error(
                "assembly.novoplasty_chloroplast_required",
                "mito_plant requires a chloroplast_fasta assembly",
            )
        effective = parameters.model_dump(mode="json")
        effective["genome_range"] = list(interval)
        effective["type"] = (
            "chloro"
            if organelle == "plastid"
            else ("mito_plant" if taxon_group == "plant" else "mito")
        )
        effective["seed_source"] = (
            payload.auxiliary.seed_fasta_artifact
            or "bundled:Arabidopsis_thaliana_chloroplast.gb:CDS:rbcL"
        )
        return effective
    if backend_id == "ovasm":
        parameters = provided if isinstance(provided, OvasmParameters) else OvasmParameters()
        if provided is not None and not isinstance(provided, OvasmParameters):
            raise ValueError("backend_parameters must match selected backend")
        if payload is None:
            raise ValueError("ovasm parameters require an assembly payload")
        return _effective_ovasm_parameters(payload, parameters)
    if backend_id == "tippo":
        parameters = provided if isinstance(provided, TippoParameters) else TippoParameters()
        if provided is not None and not isinstance(provided, TippoParameters):
            raise ValueError("backend_parameters must match selected backend")
        return parameters.model_dump(mode="json")
    if backend_id == "ptgaul":
        parameters = provided if isinstance(provided, PtgaulParameters) else PtgaulParameters()
        if provided is not None and not isinstance(provided, PtgaulParameters):
            raise ValueError("backend_parameters must match selected backend")
        return parameters.model_dump(mode="json")
    if backend_id == "oatk":
        parameters = provided if isinstance(provided, OatkParameters) else OatkParameters()
        payload_out: dict[str, object] = parameters.model_dump(mode="json")
        payload_out["kmer_size"] = 1001
        return payload_out
    if backend_id == "himt":
        if payload is None:
            raise ValueError("HiMT parameters require an assembly payload")
        if len(payload.long_libraries) != 1:
            raise ValueError("HiMT requires exactly one long-read library")
        parameters = provided if isinstance(provided, HimtParameters) else HimtParameters()
        effective = parameters.model_copy(update={"accuracy": parameters.accuracy})
        if effective.accuracy is None:
            quality = payload.long_libraries[0].quality_state
            resolved_accuracy: float = {
                "raw": 0.3,
                "corrected": 0.8,
                "ccs": 0.8,
                "hq": 0.8,
                "duplex": 0.8,
            }[quality]
            effective = effective.model_copy(update={"accuracy": resolved_accuracy})
        return effective.model_dump(mode="json")
    if backend_id == "pmat":
        if payload is None:
            raise ValueError("PMAT parameters require an assembly payload")
        if organelle is None or taxon_group is None:
            raise ValueError("PMAT parameters require organelle and taxon_group")
        parameters = provided if isinstance(provided, PmatParameters) else None
        if provided is not None and parameters is None:
            raise ValueError("backend_parameters must match selected backend")
        return _effective_pmat_parameters(
            payload,
            parameters,
            organelle=organelle,
            taxon_group=taxon_group,
        )
    if backend_id == "getorganelle":
        if organelle is None or taxon_group is None:
            raise ValueError("GetOrganelle parameters require organelle and taxon_group")
        parameters = provided if isinstance(provided, GetOrganelleParameters) else None
        if provided is not None and parameters is None:
            raise ValueError("backend_parameters must match selected backend")
        return _effective_getorganelle_parameters(
            parameters,
            organelle=organelle,
            taxon_group=taxon_group,
        )
    if provided is not None:
        raise ValueError("backend_parameters are not released for selected backend")
    return {}


class AssemblyRequest(_FrozenInputModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    data: OrganelleData
    organelle: Literal["mitochondrion", "plastid"]
    method: AssemblyMethod | Literal["auto"] = "auto"
    backend_parameters: AssemblyBackendParameters | None = None
    threads: int = Field(default=4, ge=1, le=256)
    memory_gb: int | None = Field(default=None, ge=1)
    timeout_seconds: int | None = Field(default=None, ge=1)
    taxon_group: Literal["plant", "animal", "fungi"] = "plant"
    environment_source: EnvironmentSource = "auto"
    backend_version: BackendVersionSelector = "tested"
    environment_hint: EnvironmentHint | None = None

    @field_validator("data", mode="before")
    @classmethod
    def validate_serialized_data(cls, value: object) -> OrganelleData:
        return OrganelleData.model_validate(value)

    @field_validator("environment_hint", mode="before")
    @classmethod
    def _validate_serialized_hint(cls, value: object) -> EnvironmentHint | None:
        if value is None:
            return None
        return EnvironmentHint.model_validate(value)

    @model_validator(mode="after")
    def validate_data_contract(self) -> AssemblyRequest:
        validate_assembly_data(self.data)
        if (
            self.method != "auto"
            and self.backend_parameters is not None
            and self.backend_parameters.backend != self.method
        ):
            raise ValueError(f"backend_parameters must match explicit method {self.method!r}")
        try:
            canonical_json_bytes(self.semantic_payload())
        except (TypeError, ValueError) as error:
            raise ValueError(
                "assembly request semantic identity must be canonical UTF-8 JSON"
            ) from error
        return self

    # ------------------------------------------------------------------
    # semantic identity
    # ------------------------------------------------------------------

    def semantic_payload(self) -> dict[str, object]:
        evidence = validate_assembly_data(self.data).auxiliary.genome_size_evidence
        payload: dict[str, object] = {
            "input_data_id": self.data.object_id,
            "input_artifacts": {
                role: {
                    "kind": artifact.kind,
                    "format": artifact.format,
                    "media_type": artifact.media_type,
                    "sha256": artifact.sha256,
                    "size_bytes": artifact.size_bytes,
                }
                for role, artifact in sorted(self.data.artifacts.items())
            },
            "organelle": self.organelle,
            "requested_method": self.method,
            "backend_parameters": (
                self.backend_parameters.model_dump(mode="json")
                if self.backend_parameters is not None
                else None
            ),
            "threads": self.threads,
            "memory_gb": self.memory_gb,
            "timeout_seconds": self.timeout_seconds,
            "taxon_group": self.taxon_group,
            "genome_size_evidence": (
                evidence.model_dump(mode="json") if evidence is not None else None
            ),
            "environment_source": self.environment_source,
            "backend_version": self.backend_version,
        }
        if self.environment_hint is not None:
            payload["environment_hint"] = self.environment_hint.model_dump(mode="json")
        return payload

    def resolved_semantic_payload(self, selected_backend: AssemblyMethod) -> dict[str, object]:
        payload = self.semantic_payload()
        payload["selected_backend"] = selected_backend
        payload["effective_backend_parameters"] = effective_backend_parameters(
            selected_backend,
            self.backend_parameters,
            payload=validate_assembly_data(self.data),
            organelle=self.organelle,
            taxon_group=self.taxon_group,
        )
        return payload

    def resolved_semantic_hash(self, selected_backend: AssemblyMethod) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.resolved_semantic_payload(selected_backend))
        ).hexdigest()

    @property
    def semantic_hash(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.semantic_payload())).hexdigest()
