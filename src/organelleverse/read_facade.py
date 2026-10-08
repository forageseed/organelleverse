"""Unified human-facing reader over the canonical input readers."""

from __future__ import annotations

import gzip
import re
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from .annotation.readers import ReaderOrganelle
from .assembly.contracts import (
    AssemblyAuxiliary,
    ContigInput,
    LongReadLibrary,
    ShortReadLibrary,
)
from .core.data import OrganelleData
from .core.errors import OrganelleInputError
from .core.genome import OrganelleGenome, OrganelleMetadata
from .io_genome import read_fasta_genome, read_genbank_genome
from .io_reads import read_reads

ReadFormat = Literal["auto", "fasta", "genbank"]

_FASTA_SUFFIXES = (".fa", ".fasta", ".fna")
_GENBANK_SUFFIXES = (".gb", ".gbk", ".genbank")

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


def _format_from_suffix(path: str | Path) -> Literal["fasta", "genbank"] | None:
    name = Path(path).name.lower()
    if name.endswith(_FASTA_SUFFIXES):
        return "fasta"
    if name.endswith(_GENBANK_SUFFIXES):
        return "genbank"
    return None


def _format_from_content(path: str | Path) -> Literal["fasta", "genbank"] | None:
    candidate = Path(path)
    if not candidate.is_file():
        raise OrganelleInputError(
            code="input.missing_artifact",
            message=f"Artifact does not exist: {candidate.name}",
            details={"basename": candidate.name},
        )
    with candidate.open("rb") as handle:
        prefix = handle.read(8192)
    try:
        text = prefix.decode("utf-8").lstrip()
    except UnicodeDecodeError:
        return None
    if text.startswith(">"):
        return "fasta"
    if text.startswith("LOCUS"):
        return "genbank"
    return None


def _sequence_format(path: str | Path, declared: ReadFormat) -> Literal["fasta", "genbank"]:
    inferred = _format_from_suffix(path)
    if declared == "auto":
        if inferred is None:
            inferred = _format_from_content(path)
        if inferred is None:
            raise OrganelleInputError(
                code="input.unknown_genome_format",
                message="Input is neither recognizable FASTA nor GenBank",
                details={"basename": Path(path).name},
            )
        return inferred
    if inferred is not None and inferred != declared:
        raise OrganelleInputError(
            code="input.sequence_format_conflict",
            message="Declared sequence format conflicts with the filename suffix",
            details={
                "basename": Path(path).name,
                "declared_format": declared,
                "suffix_format": inferred,
            },
        )
    return declared


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
    """``ov.read("reads.fastq.gz", technology=...)``: wrap the file in a read library."""
    files = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    if not files:
        raise OrganelleInputError(
            code="input.read_source_required",
            message="ov.read needs at least one reads file",
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
    format: ReadFormat = "auto",
    organelle: ReaderOrganelle | None = None,
    species: str = "",
    accession: str = "",
    genetic_code: int | None = None,
    assembly_type: str = "",
    plastid_type: str = "",
    source: str = "",
    short_libraries: tuple[ShortReadLibrary, ...] = (),
    long_libraries: tuple[LongReadLibrary, ...] = (),
    contig_inputs: tuple[ContigInput, ...] = (),
    auxiliary: AssemblyAuxiliary | None = None,
) -> OrganelleGenome | OrganelleData:
    """Read one sequence file or one explicitly described sequencing dataset.

    ``ov.read("genome.fasta", organelle="mitochondrion")`` reads a genome.
    ``ov.read("reads.fastq.gz", technology="hifi")`` reads sequencing reads; the
    technology is one of ``"hifi"``, ``"clr"``, ``"ont"``, ``"illumina"`` or
    ``"auto"`` (identify it from the read names, or refuse). Illumina takes one file
    or two (read 1, read 2).
    """

    has_libraries = bool(short_libraries or long_libraries or contig_inputs)
    has_read_input = has_libraries or auxiliary is not None
    if technology is not None:
        if path is None:
            raise OrganelleInputError(
                code="input.read_source_required",
                message="technology requires the path of a reads file",
            )
        if (
            has_read_input
            or format != "auto"
            or organelle is not None
            or genetic_code is not None
            or any((species, accession, assembly_type, plastid_type, source))
        ):
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
    if path is not None and not isinstance(path, (str, Path)):
        raise OrganelleInputError(
            code="input.invalid_read_path",
            message="Several paths are only accepted together with technology='illumina'",
        )
    if path is not None:
        if format not in {"auto", "fasta", "genbank"}:
            raise OrganelleInputError(
                code="input.invalid_read_format",
                message="format must be 'auto', 'fasta', or 'genbank'",
                details={"format": format},
            )
        if has_read_input:
            raise OrganelleInputError(
                code="input.ambiguous_read",
                message="ov.read cannot mix a sequence path with sequencing libraries",
            )
        if organelle is None:
            raise OrganelleInputError(
                code="input.organelle_required",
                message="organelle is required when reading FASTA or GenBank",
            )
        if organelle not in {"mito", "mitochondrion", "chloro", "plastid"}:
            raise OrganelleInputError(
                code="input.invalid_organelle",
                message="organelle must identify a mitochondrion or plastid",
                details={"organelle": organelle},
            )
        selected_format = _sequence_format(path, format)
        canonical_organelle = (
            "mitochondrion" if organelle in {"mito", "mitochondrion"} else "plastid"
        )
        # genetic_code is passed only when stated, so the reader can take it from
        # the GenBank file or the organelle's standard table instead.
        metadata = OrganelleMetadata(
            species=species,
            accession=accession,
            assembly_type=assembly_type,
            plastid_type=plastid_type,
            source=source,
            **({} if genetic_code is None else {"genetic_code": genetic_code}),
        )
        if selected_format == "fasta":
            return read_fasta_genome(
                Path(path),
                organelle=canonical_organelle,
                species=species,
                metadata=metadata,
            )
        return read_genbank_genome(
            Path(path),
            organelle=canonical_organelle,
            species=species,
            metadata=metadata,
        )

    if not has_read_input:
        raise OrganelleInputError(
            code="input.read_source_required",
            message="ov.read requires a FASTA/GenBank path or structured sequencing input",
        )
    if (
        format != "auto"
        or organelle is not None
        or any((species, accession, assembly_type, plastid_type, source))
        or genetic_code is not None
    ):
        raise OrganelleInputError(
            code="input.sequence_parameters_without_path",
            message="Sequence format, organelle, and genome metadata require a sequence path",
        )
    return read_reads(
        short_libraries=short_libraries,
        long_libraries=long_libraries,
        contig_inputs=contig_inputs,
        auxiliary=auxiliary if auxiliary is not None else AssemblyAuxiliary(),
    )


__all__ = ["ReadFormat", "read"]
