"""Canonical reader for structured sequencing assembly inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from organelleverse.assembly.contracts import (
    AssemblyAuxiliary,
    AssemblyInputPayload,
    AuxiliaryContract,
    ContigContract,
    ContigInput,
    LongLibraryContract,
    LongReadLibrary,
    ShortLibraryContract,
    ShortReadLibrary,
    validate_assembly_data,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleInputError
from organelleverse.io_verify import VerifiedFormat, read_verified_artifact


def _format_for(source: str | Path | ArtifactRef, fallback: str) -> str:
    if isinstance(source, ArtifactRef):
        return source.format
    name = Path(source).name.lower()
    if name.endswith((".fastq", ".fq", ".fastq.gz", ".fq.gz")):
        return "fastq"
    if name.endswith((".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz")):
        return "fasta"
    if name.endswith((".gb", ".gbk", ".genbank", ".gb.gz")):
        return "genbank"
    return fallback


def _artifact(
    source: str | Path | ArtifactRef,
    *,
    kind: str,
    fallback_format: str,
) -> ArtifactRef:
    if isinstance(source, ArtifactRef):
        return source
    return ArtifactRef.from_path(
        source,
        kind=kind,
        format=_format_for(source, fallback_format),
    )


def _declared_format(source: str | Path, fallback: str) -> VerifiedFormat:
    """Resolve the format to verify against.

    A suffix is not evidence. A recognised suffix that contradicts the caller's
    declared format is an error; an unrecognised suffix neither selects nor
    rescues a format, so the declared value stands and the verifier decides.
    """
    suffixes = {
        "fastq": (".fastq", ".fq"),
        "fasta": (".fasta", ".fa", ".fna"),
        "genbank": (".gb", ".gbk", ".genbank"),
    }
    name = Path(source).name.lower()
    for candidate, endings in suffixes.items():
        if name.endswith(endings):
            if candidate != fallback:
                raise OrganelleInputError(
                    code=f"input.invalid_{fallback}",
                    message=(
                        f"Filename suffix declares {candidate} but "
                        f"{fallback} was requested: {Path(source).name}"
                    ),
                    details={"basename": Path(source).name},
                )
            return cast("VerifiedFormat", candidate)
    if fallback not in suffixes:
        raise OrganelleInputError(
            code="input.unsupported_artifact_format",
            message=f"Unsupported declared format: {fallback}",
            details={"declared_format": fallback},
        )
    return cast("VerifiedFormat", fallback)


def read_reads(
    *,
    short_libraries: tuple[ShortReadLibrary, ...] = (),
    long_libraries: tuple[LongReadLibrary, ...] = (),
    contig_inputs: tuple[ContigInput, ...] = (),
    auxiliary: AssemblyAuxiliary = AssemblyAuxiliary(),  # noqa: B008 - frozen Pydantic model
) -> OrganelleData:
    """Build content-addressed v1 data for explicitly described sequencing inputs."""
    artifacts: dict[str, ArtifactRef] = {}
    short_contracts: list[ShortLibraryContract] = []
    long_contracts: list[LongLibraryContract] = []
    contig_contracts: list[ContigContract] = []

    for index, library in enumerate(short_libraries):
        read1_role = f"short_{index}_read1"
        artifacts[read1_role] = _artifact(
            library.read1,
            kind="short_read",
            fallback_format="fastq",
        )
        read2_role: str | None = None
        if library.read2 is not None:
            read2_role = f"short_{index}_read2"
            artifacts[read2_role] = _artifact(
                library.read2,
                kind="short_read",
                fallback_format="fastq",
            )
        short_contracts.append(
            ShortLibraryContract(
                technology=library.technology,
                layout=library.layout,
                read1_artifact=read1_role,
                read2_artifact=read2_role,
                read_length=library.read_length,
                insert_size=library.insert_size,
            )
        )

    for index, library in enumerate(long_libraries):
        role = f"long_{index}_reads"
        artifacts[role] = _artifact(
            library.reads,
            kind="long_read",
            fallback_format="fastq",
        )
        long_contracts.append(
            LongLibraryContract(
                technology=library.technology,
                quality_state=library.quality_state,
                reads_artifact=role,
            )
        )

    for index, contigs in enumerate(contig_inputs):
        role = f"preassembled_contigs_{index}"
        artifacts[role] = _artifact(
            contigs.fasta,
            kind="assembly_contigs",
            fallback_format="fasta",
        )
        contig_contracts.append(ContigContract(fasta_artifact=role))

    auxiliary_sources = {
        "seed_fasta": (auxiliary.seed_fasta, "seed", "fasta"),
        "reference_fasta": (auxiliary.reference_fasta, "reference", "fasta"),
        "reference_genbank": (
            auxiliary.reference_genbank,
            "reference_annotation",
            "genbank",
        ),
        "chloroplast_fasta": (
            auxiliary.chloroplast_fasta,
            "chloroplast_reference",
            "fasta",
        ),
        "hmm_profiles": (auxiliary.hmm_profiles, "hmm_profiles", "hmm"),
        "correction_config": (
            auxiliary.correction_config,
            "correction_config",
            "text",
        ),
        "genome_size_report": (
            auxiliary.genome_size_report,
            "genome_size_report",
            "json",
        ),
        "canu_executable": (auxiliary.canu_executable, "executable", "executable"),
        "nextdenovo_executable": (
            auxiliary.nextdenovo_executable,
            "executable",
            "executable",
        ),
    }
    auxiliary_roles: dict[str, str | None] = {}
    for role, (source, kind, fallback_format) in auxiliary_sources.items():
        auxiliary_roles[f"{role}_artifact"] = role if source is not None else None
        if source is not None:
            artifacts[role] = _artifact(
                source,
                kind=kind,
                fallback_format=fallback_format,
            )

    payload = AssemblyInputPayload(
        short_libraries=tuple(short_contracts),
        long_libraries=tuple(long_contracts),
        contig_inputs=tuple(contig_contracts),
        auxiliary=AuxiliaryContract(
            seed_fasta_artifact=auxiliary_roles["seed_fasta_artifact"],
            reference_fasta_artifact=auxiliary_roles["reference_fasta_artifact"],
            reference_genbank_artifact=auxiliary_roles["reference_genbank_artifact"],
            chloroplast_fasta_artifact=auxiliary_roles["chloroplast_fasta_artifact"],
            hmm_profiles_artifact=auxiliary_roles["hmm_profiles_artifact"],
            correction_config_artifact=auxiliary_roles["correction_config_artifact"],
            genome_size=auxiliary.genome_size,
            genome_range=auxiliary.genome_range,
            genome_size_evidence=auxiliary.genome_size_evidence,
            genome_size_report_artifact=auxiliary_roles["genome_size_report_artifact"],
            canu_executable_artifact=auxiliary_roles["canu_executable_artifact"],
            nextdenovo_executable_artifact=auxiliary_roles["nextdenovo_executable_artifact"],
        ),
    )
    data = OrganelleData(
        modality="sequencing_reads",
        artifacts=artifacts,
        payload=payload.model_dump(mode="json"),
        dimensions={
            "short_libraries": len(short_libraries),
            "long_libraries": len(long_libraries),
            "contig_inputs": len(contig_inputs),
        },
        metadata={"reader": "ov.io.read_reads"},
    )
    validate_assembly_data(data)
    return data


def read_long_reads(
    reads: Path,
    *,
    technology: Literal["pacbio_hifi", "pacbio_clr", "ont"],
    quality_state: Literal["raw", "corrected", "duplex", "hq", "ccs"],
) -> OrganelleData:
    """Read one long-read library as released sequencing-reads data.

    Verification is streaming with bounded memory (a 30 GB gzipped HiFi
    library verifies at ~50 MB RSS), gzip is accepted transparently, and
    the recorded digest always describes the compressed bytes on disk; the
    size cap is the 64 GiB module default (call
    ``io_verify.read_verified_artifact`` directly for a different cap -
    this released surface stays a closed three-parameter schema).

    The long-read artifact is verified before entering ``read_reads`` so the
    released reader earns ``validated=True`` rather than asserting it. The built
    ``ArtifactRef`` is passed through ``LongReadLibrary.reads``; the public
    ``ov.io.read_reads`` path and its auxiliary formats are unchanged.
    """
    artifact = read_verified_artifact(
        reads,
        kind="long_read",
        format=_declared_format(reads, "fastq"),
    )
    return read_reads(
        long_libraries=(
            LongReadLibrary(
                technology=technology,
                quality_state=quality_state,
                reads=artifact,
            ),
        )
    )
