"""Released genome readers, one per file format.

Split rather than one reader with a format parameter, because the two produce
genomes with different downstream capability: annotation.annotate passes
genome.sequence straight through as input_fasta, so a GenBank path cannot serve
as a sequence artifact. FASTA feeds annotate; GenBank feeds extract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.core.result import OrganelleResult
from organelleverse.core.serialization import load_genome
from organelleverse.io_verify import read_verified_artifact

Organelle = Literal["mitochondrion", "plastid"]

# NCBI translation table used when neither the caller nor the file states one:
# table 11 for every plastid, table 1 for plant mitochondria.
_STANDARD_GENETIC_CODE: dict[str, int] = {"plastid": 11, "mitochondrion": 1}


def read_fasta_genome(
    path: Path,
    *,
    organelle: Organelle,
    species: str,
    metadata: OrganelleMetadata | None = None,
) -> OrganelleGenome:
    """Read one FASTA sequence file as a released organelle genome.

    Without a caller-stated genetic_code the organelle's standard NCBI table is
    used (plastid 11, mitochondrion 1).
    """
    resolved = _resolved_metadata(organelle, species, metadata)
    sequence = read_verified_artifact(path, kind="sequence", format="fasta")
    return OrganelleGenome(
        organelle=organelle,
        sequence=sequence,
        metadata=_reconciled(resolved, _asserted(metadata), organelle),
    )


def read_assembly_genome(
    result: OrganelleResult,
    *,
    species: str,
    metadata: OrganelleMetadata | None = None,
) -> OrganelleGenome:
    """The primary genome of an ``assemble`` result, ready for ``annotate``.

    An assembly result carries its sequence as a ``primary_genome_manifest`` artifact
    but not the species, which the annotation backends require; it is given here. With
    ``method="ovasm"`` the sequence holds every representative molecule (one FASTA
    record each, with its ``circular=`` flag), and annotation keeps one record per
    molecule.

    Without a caller-stated genetic_code the organelle's standard NCBI table is
    used (plastid 11, mitochondrion 1).
    """
    if not result.operation_id.startswith("assembly."):
        raise OrganelleInputError(
            code="input.not_an_assembly_result",
            message="read_assembly_genome needs the result of an assembly operation",
            details={"operation_id": result.operation_id},
        )
    if result.status == "failed":
        raise OrganelleInputError(
            code="input.failed_assembly_result",
            message="a failed assembly result has no genome to annotate",
            details={"status": result.status},
        )
    manifests = [a for a in result.artifacts if a.kind == "primary_genome_manifest"]
    if len(manifests) != 1:
        raise OrganelleInputError(
            code="input.assembly_genome_missing",
            message="the assembly result must carry exactly one primary genome manifest",
            details={"found": len(manifests)},
        )
    genome = load_genome(manifests[0].resolve())
    resolved = _resolved_metadata(genome.organelle, species, metadata or genome.metadata)
    # Assembly never states a genetic code, so only the caller's metadata can.
    return genome.evolve(metadata=_reconciled(resolved, _asserted(metadata), genome.organelle))


def read_genbank_genome(
    path: Path,
    *,
    organelle: Organelle,
    species: str,
    metadata: OrganelleMetadata | None = None,
) -> OrganelleGenome:
    """Read one GenBank annotation file as a released organelle genome.

    The accession (ACCESSION + VERSION of a single-record file) and the CDS
    /transl_table are taken from the file; a caller value that contradicts
    either is refused.
    """
    resolved = _resolved_metadata(organelle, species, metadata)
    annotation = read_verified_artifact(path, kind="annotation", format="genbank")
    file_accession, file_genetic_code = _genbank_identity(path)
    return OrganelleGenome(
        organelle=organelle,
        annotation=annotation,
        metadata=_reconciled(
            resolved,
            _asserted(metadata),
            organelle,
            file_accession=file_accession,
            file_genetic_code=file_genetic_code,
        ),
    )


def _asserted(metadata: OrganelleMetadata | None) -> frozenset[str]:
    """Metadata fields the caller set explicitly (a default is not an assertion)."""
    return frozenset(metadata.model_fields_set) if metadata is not None else frozenset()


def _genbank_identity(path: Path) -> tuple[str, int | None]:
    """Return the file's versioned accession and its single CDS translation table.

    A multi-record file names several sequences, so it yields no accession; a
    file without an ACCESSION line (e.g. a locally written record) yields none
    either, because its LOCUS name is not an accession.
    """
    from Bio import SeqIO  # type: ignore

    records = list(SeqIO.parse(str(path), "genbank"))
    accession = ""
    if len(records) == 1:
        annotations = records[0].annotations
        accessions = annotations.get("accessions") or []
        if accessions:
            version = annotations.get("sequence_version")
            accession = f"{accessions[0]}.{version}" if version else str(accessions[0])
    tables = {
        int(table)
        for record in records
        for feature in record.features
        if feature.type == "CDS"
        for table in feature.qualifiers.get("transl_table", [])
    }
    if len(tables) > 1:
        raise OrganelleInputError(
            code="input.mixed_genetic_codes",
            message="GenBank CDS features declare more than one /transl_table",
            details={"basename": path.name, "transl_tables": sorted(tables)},
        )
    return accession, next(iter(tables), None)


def _reconciled(
    resolved: OrganelleMetadata,
    asserted: frozenset[str],
    organelle: Organelle,
    *,
    file_accession: str = "",
    file_genetic_code: int | None = None,
) -> OrganelleMetadata:
    """Fill accession and genetic_code: caller value, else file, else standard table."""
    accession = resolved.accession.strip()
    if file_accession:
        if accession and accession not in {file_accession, file_accession.split(".")[0]}:
            raise OrganelleInputError(
                code="input.metadata_accession_conflict",
                message="metadata.accession contradicts the GenBank ACCESSION/VERSION",
                details={"metadata_accession": accession, "file_accession": file_accession},
            )
        accession = file_accession
    if "genetic_code" in asserted:
        genetic_code = resolved.genetic_code
        if file_genetic_code is not None and genetic_code != file_genetic_code:
            raise OrganelleInputError(
                code="input.metadata_genetic_code_conflict",
                message="metadata.genetic_code contradicts the GenBank CDS /transl_table",
                details={
                    "metadata_genetic_code": genetic_code,
                    "file_transl_table": file_genetic_code,
                },
            )
    elif file_genetic_code is not None:
        genetic_code = file_genetic_code
    else:
        genetic_code = _STANDARD_GENETIC_CODE[organelle]
    return resolved.evolve(accession=accession, genetic_code=genetic_code)


def _resolved_metadata(
    organelle: Organelle, species: str, metadata: OrganelleMetadata | None
) -> OrganelleMetadata:
    """Reconcile the required species with optional caller metadata.

    species is a required parameter because the released mitochondrial backend
    rejects a blank one, while OrganelleMetadata.species defaults to empty - a
    genome that would be type-valid, reported reachable by the audit, and
    rejected at invocation. Checked before any file is opened.

    A contradictory metadata.species is refused rather than silently overridden,
    and so is a plastid_type set on a non-plastid genome: the canonical form for
    a chloroplast is organelle=plastid with metadata.plastid_type=chloroplast, so
    plastid_type under mitochondrion asserts two incompatible things at once.
    Refusing one contradiction and accepting its twin would be arbitrary.
    Everything else in metadata is a caller assertion: object_id proves canonical
    consistency, not truth or provenance.
    """
    if not species.strip():
        raise OrganelleInputError(
            code="input.missing_species",
            message="species is required and must not be blank",
            details={"parameter": "species"},
        )
    base = metadata if metadata is not None else OrganelleMetadata()
    declared = base.species.strip()
    if declared and declared != species.strip():
        raise OrganelleInputError(
            code="input.metadata_species_conflict",
            message="metadata.species contradicts the species parameter",
            details={"parameter_species": species.strip(), "metadata_species": declared},
        )
    if base.plastid_type.strip() and organelle != "plastid":
        raise OrganelleInputError(
            code="input.metadata_organelle_conflict",
            message="metadata.plastid_type is only meaningful for a plastid genome",
            details={"organelle": organelle, "plastid_type": base.plastid_type.strip()},
        )
    return base.evolve(species=species.strip())


_FASTA_SUFFIXES = (".fa", ".fasta", ".fna")
_GENBANK_SUFFIXES = (".gb", ".gbk", ".genbank")

ReadFormat = Literal["auto", "fasta", "genbank"]


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


_ORGANELLE_SPELLINGS = {
    "mito": "mitochondrion",
    "mitochondrion": "mitochondrion",
    "chloro": "plastid",
    "plastid": "plastid",
}


def read_genome(
    path: str | Path,
    *,
    organelle: str | None = None,
    format: ReadFormat = "auto",
    species: str = "",
    accession: str = "",
    genetic_code: int | None = None,
    assembly_type: str = "",
    plastid_type: str = "",
    source: str = "",
) -> OrganelleGenome:
    """Read one FASTA or GenBank file as an organelle genome.

    ``organelle`` is ``"mito"``/``"mitochondrion"`` or ``"chloro"``/``"plastid"``. The
    format comes from the suffix or the file header unless ``format`` states it.
    """
    if format not in {"auto", "fasta", "genbank"}:
        raise OrganelleInputError(
            code="input.invalid_read_format",
            message="format must be 'auto', 'fasta', or 'genbank'",
            details={"format": format},
        )
    if organelle is None:
        raise OrganelleInputError(
            code="input.organelle_required",
            message="organelle is required when reading FASTA or GenBank",
        )
    if organelle not in _ORGANELLE_SPELLINGS:
        raise OrganelleInputError(
            code="input.invalid_organelle",
            message="organelle must identify a mitochondrion or plastid",
            details={"organelle": organelle},
        )
    selected_format = _sequence_format(path, format)
    canonical_organelle = _ORGANELLE_SPELLINGS[organelle]
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
    reader = read_fasta_genome if selected_format == "fasta" else read_genbank_genome
    return reader(
        Path(path),
        organelle=canonical_organelle,  # type: ignore[arg-type]
        species=species,
        metadata=metadata,
    )
