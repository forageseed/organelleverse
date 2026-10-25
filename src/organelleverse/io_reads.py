"""Canonical reader for structured sequencing assembly inputs."""

from __future__ import annotations

import gzip
import re
import statistics
from collections.abc import Sequence
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
        "read_origins": (auxiliary.read_origins, "read_origins", "json"),
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
            read_origins_artifact=auxiliary_roles["read_origins_artifact"],
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


_FASTQ_SUFFIXES = (".fastq", ".fq", ".fastq.gz", ".fq.gz")

# Short spellings accepted for ``technology=``. The value is (contract technology,
# default quality state); ``illumina`` has no long-read quality state.
_LONG_TECHNOLOGIES: dict[str, tuple[str, str]] = {
    "hifi": ("pacbio_hifi", "ccs"),
    "pacbio_hifi": ("pacbio_hifi", "ccs"),
    "clr": ("pacbio_clr", "raw"),
    "pacbio_clr": ("pacbio_clr", "raw"),
    "ont": ("ont", "raw"),
}
_SHORT_TECHNOLOGIES = {"illumina"}

# ``technology="auto"`` looks at the first records of the file only.
_SAMPLE_RECORDS = 1000
_SHORT_READ_MAX_MEDIAN = 1000
_PACBIO_MOVIE = re.compile(r"^m\d+[a-z]*_\d{6}_\d{6}")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_SUBREAD_RANGE = re.compile(r"^\d+_\d+$")


def _sample_reads(path: Path) -> tuple[list[str], list[int]]:
    """Names and lengths of the first records of a FASTQ/FASTA file (plain or gzip)."""
    if not path.is_file():
        raise OrganelleInputError(
            code="input.missing_artifact",
            message=f"Artifact does not exist: {path.name}",
            details={"basename": path.name},
        )
    with path.open("rb") as raw:
        gzipped = raw.read(2) == b"\x1f\x8b"
    opener = gzip.open if gzipped else open
    names: list[str] = []
    lengths: list[int] = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:  # type: ignore[operator]
        first = handle.readline()
        if first.startswith("@"):
            line = first
            while line and len(names) < _SAMPLE_RECORDS:
                names.append(line[1:].strip())
                lengths.append(len(handle.readline().strip()))
                handle.readline()
                handle.readline()
                line = handle.readline()
        elif first.startswith(">"):
            names.append(first[1:].strip())
            current: int | None = 0
            for line in handle:
                if line.startswith(">"):
                    lengths.append(current or 0)
                    if len(names) >= _SAMPLE_RECORDS:
                        current = None
                        break
                    names.append(line[1:].strip())
                    current = 0
                else:
                    current = (current or 0) + len(line.strip())
            if current is not None:
                lengths.append(current)
        else:
            raise OrganelleInputError(
                code="input.unknown_reads_format",
                message="Reads must be FASTQ or FASTA",
                details={"basename": path.name},
            )
    if not names:
        raise OrganelleInputError(
            code="input.empty_reads",
            message="Reads file has no records",
            details={"basename": path.name},
        )
    return names, lengths


def _name_technology(name: str) -> str | None:
    """Technology a read name proves, or None when the name proves nothing."""
    token = name.split()[0] if name.split() else name
    if "runid=" in name or _UUID.match(token):
        return "ont"
    if _PACBIO_MOVIE.match(token):
        parts = token.split("/")
        if "ccs" in parts[1:]:
            return "hifi"
        if parts and _SUBREAD_RANGE.match(parts[-1]):
            return "clr"
    return None


def _auto_technology(path: Path) -> str:
    """Decide the technology from evidence in the file, or refuse.

    Short reads: median length of the first records is at most 1000 bases.
    Long reads: every sampled read name identifies the same platform - an ONT run
    id or UUID, a PacBio ``/ccs`` name (HiFi), or a PacBio subread range (CLR). A
    read length or quality value alone is not evidence, so anything else raises
    and asks for an explicit ``technology=``.
    """
    names, lengths = _sample_reads(path)
    median_length = statistics.median(lengths)
    if median_length <= _SHORT_READ_MAX_MEDIAN:
        return "illumina"
    found = {_name_technology(name) for name in names}
    if len(found) == 1 and None not in found:
        (technology,) = found
        assert technology is not None
        return technology
    raise OrganelleInputError(
        code="input.technology_not_identifiable",
        message=(
            "technology='auto' could not identify the sequencing platform from the read "
            "names; pass technology='hifi', 'clr', 'ont' or 'illumina'"
        ),
        details={
            "basename": path.name,
            "sampled_reads": len(names),
            "median_length": int(median_length),
        },
    )


def _read_sequencing_file(
    paths: str | Path | Sequence[str | Path],
    *,
    technology: str,
    quality_state: str | None,
    read_length: int | None,
) -> OrganelleData:
    """``read("reads.fastq.gz", technology=...)``: wrap the file in a read library."""
    files = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    if not files:
        raise OrganelleInputError(
            code="input.read_source_required",
            message="at least one reads file is required",
        )
    selected = technology.lower()
    if selected == "auto":
        selected = _auto_technology(files[0])
    if selected in _SHORT_TECHNOLOGIES:
        if len(files) > 2:
            raise OrganelleInputError(
                code="input.too_many_read_files",
                message="Illumina reads take one file (single-end) or two (read 1, read 2)",
                details={"files": len(files)},
            )
        if quality_state is not None:
            raise OrganelleInputError(
                code="input.quality_state_not_applicable",
                message="quality_state applies to long reads only",
            )
        length = read_length
        if length is None:
            length = max(_sample_reads(files[0])[1])
        short = ShortReadLibrary(
            technology="illumina",
            layout="paired_end" if len(files) == 2 else "single_end",
            read1=files[0],
            read2=files[1] if len(files) == 2 else None,
            read_length=length,
        )
        return read_reads(short_libraries=(short,), auxiliary=AssemblyAuxiliary())
    if selected not in _LONG_TECHNOLOGIES:
        raise OrganelleInputError(
            code="input.invalid_technology",
            message="technology must be 'auto', 'hifi', 'clr', 'ont' or 'illumina'",
            details={"technology": technology},
        )
    if len(files) != 1:
        raise OrganelleInputError(
            code="input.too_many_read_files",
            message="Pass one long-read file per call; use long_libraries for several libraries",
            details={"files": len(files)},
        )
    if read_length is not None:
        raise OrganelleInputError(
            code="input.read_length_not_applicable",
            message="read_length applies to Illumina reads only",
        )
    contract_technology, default_quality = _LONG_TECHNOLOGIES[selected]
    long = LongReadLibrary(
        technology=contract_technology,  # type: ignore[arg-type]
        quality_state=quality_state or default_quality,  # type: ignore[arg-type]
        reads=files[0],
    )
    return read_reads(long_libraries=(long,), auxiliary=AssemblyAuxiliary())


def read(
    path: str | Path | Sequence[str | Path] | None = None,
    *,
    technology: str | None = None,
    quality_state: str | None = None,
    read_length: int | None = None,
    short_libraries: tuple[ShortReadLibrary, ...] = (),
    long_libraries: tuple[LongReadLibrary, ...] = (),
    contig_inputs: tuple[ContigInput, ...] = (),
    auxiliary: AssemblyAuxiliary | None = None,
) -> OrganelleData:
    """Read one sequencing file or one explicitly described sequencing dataset.

    ``read("reads.fastq.gz", technology="hifi")`` wraps a reads file; the technology is
    one of ``"hifi"``, ``"clr"``, ``"ont"``, ``"illumina"`` or ``"auto"`` (identify it
    from the read names, or refuse). Illumina takes one file or two (read 1, read 2).
    ``read(long_libraries=(...,))`` takes libraries built by hand.
    """
    has_libraries = bool(short_libraries or long_libraries or contig_inputs)
    has_read_input = has_libraries or auxiliary is not None
    if technology is not None:
        if path is None:
            raise OrganelleInputError(
                code="input.read_source_required",
                message="technology requires the path of a reads file",
            )
        if has_read_input:
            raise OrganelleInputError(
                code="input.reads_shorthand_conflict",
                message=(
                    "technology reads a sequencing file on its own; it cannot be combined "
                    "with libraries, format, organelle or genome metadata"
                ),
            )
        return _read_sequencing_file(
            path, technology=technology, quality_state=quality_state, read_length=read_length
        )
    if quality_state is not None or read_length is not None:
        raise OrganelleInputError(
            code="input.technology_required",
            message="quality_state and read_length require technology",
        )
    if path is not None:
        if not isinstance(path, (str, Path)):
            raise OrganelleInputError(
                code="input.invalid_read_path",
                message="Several paths are only accepted together with technology='illumina'",
            )
        name = Path(path).name
        if has_read_input and not name.lower().endswith(_FASTQ_SUFFIXES):
            raise OrganelleInputError(
                code="input.ambiguous_read",
                message="a sequence path cannot be mixed with sequencing libraries",
            )
        raise OrganelleInputError(
            code="input.technology_required",
            message=(
                "a reads file needs technology=: 'hifi', 'clr', 'ont', 'illumina' or 'auto'"
            ),
            details={"basename": name},
        )
    if not has_read_input:
        raise OrganelleInputError(
            code="input.read_source_required",
            message="reading sequencing input requires a reads path with technology= or libraries",
        )
    return read_reads(
        short_libraries=short_libraries,
        long_libraries=long_libraries,
        contig_inputs=contig_inputs,
        auxiliary=auxiliary if auxiliary is not None else AssemblyAuxiliary(),
    )
