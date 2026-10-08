"""Agent-facing wrappers — the approved facade the registry registers.

The Python API and the agent API are not the same interface, and the contract is
what proves it. ``entrez_query(*, organelle, dest, transport=None, filters=None)``
cannot be registered: a READ operation must take its source positionally, every
parameter must carry a JSON-safe annotation, and ``Transport`` (a Protocol) and
``FetchFilters`` (a dataclass) are neither.

That rejection is the contract working, not the contract being awkward. An agent
calls these operations by sending JSON; a parameter it cannot express in JSON is
a parameter it cannot send. So the registered callables are thin wrappers with
flat, JSON-expressible signatures, and the rich Python API stays exactly as it is
for humans and notebooks.

``dedup`` needed the same treatment for a different reason: ``dedup_genomes``
returns ``list[GenomeCluster]``, and a TRANSFORM operation must return a core
object. The wrapper is what turns clusters back into ``OrganelleData``.
"""

from __future__ import annotations

from typing import Annotated, Any

from annotated_types import Ge, Le

from ..core.data import OrganelleData
from .dedup import dedup_genomes, unlinked_refseq_records
from .filters import FetchFilters
from .genes import fetch_genes
from .genome_size import genome_size_candidates
from .ncbi import entrez_query, fetch_accessions, refseq_snapshot
from .ngdc import fetch_ngdc
from .nuclear import fetch_nuclear_genome

__all__ = [
    "op_accession_list",
    "op_dedup_genomes",
    "op_entrez_query",
    "op_gene_records",
    "op_genome_size_candidates",
    "op_ngdc_gwh",
    "op_nuclear_assembly",
    "op_refseq_snapshot",
]


def op_refseq_snapshot(
    organelle: str,
    *,
    dest: str,
    release: int,
) -> OrganelleData:
    """Download a pinned RefSeq release snapshot for one organelle."""
    return refseq_snapshot(organelle=organelle, dest=dest, release=release)


def op_entrez_query(
    organelle: str,
    *,
    dest: str,
    taxon: str | None = None,
    complete_only: bool = True,
    min_length: int | None = None,
    max_length: int | None = None,
    refseq_only: bool = False,
    min_gene_count: int | None = None,
    max_ambiguous: int | None = None,
    max_records: int | None = None,
    genbank_metadata: bool = False,
    group_genomes: bool = True,
) -> OrganelleData:
    """Search live nuccore for organelle genomes, filtering before download."""
    filters = FetchFilters(
        min_length=min_length,
        max_length=max_length,
        refseq_only=refseq_only,
        min_gene_count=min_gene_count,
        max_ambiguous=max_ambiguous,
    )
    return entrez_query(
        organelle=organelle,
        dest=dest,
        taxon=taxon,
        complete_only=complete_only,
        filters=filters,
        max_records=max_records,
        genbank_metadata=genbank_metadata,
        group_genomes=group_genomes,
    )


def op_accession_list(
    organelle: str,
    *,
    dest: str,
    accessions: list[str],
    genbank_metadata: bool = False,
) -> OrganelleData:
    """Fetch an explicit accession list — how someone else's dataset is reproduced."""
    return fetch_accessions(
        accessions=accessions,
        dest=dest,
        organelle=organelle,
        genbank_metadata=genbank_metadata,
    )


def op_gene_records(
    gene: str,
    *,
    dest: str,
    taxon: str | None = None,
    scope: str = "plastid",
    max_records: int | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
) -> OrganelleData:
    """Fetch gene records. Fragments are the target here, not noise."""
    return fetch_genes(
        gene=gene,
        dest=dest,
        taxon=taxon,
        scope=scope,  # type: ignore[arg-type]
        max_records=max_records,
        min_length=min_length,
        max_length=max_length,
    )


def op_nuclear_assembly(
    taxon: str,
    *,
    dest: str,
    assembly_level: str | None = "chromosome",
    reference_only: bool = True,
    include: tuple[str, ...] = ("protein",),
    max_records: int = 20,
    download: bool = True,
) -> OrganelleData:
    """Fetch nuclear assemblies from NCBI Datasets — the other half of an ERC run."""
    return fetch_nuclear_genome(
        taxon=taxon,
        dest=dest,
        assembly_level=assembly_level,  # type: ignore[arg-type]
        reference_only=reference_only,
        include=include,
        max_records=max_records,
        download=download,
    )


def op_genome_size_candidates(
    scientific_name: str,
    *,
    max_candidates: Annotated[int, Ge(1), Le(50)] = 10,
) -> OrganelleData:
    """List ranked exact-species nuclear genome-size candidates from NCBI.

    ``transport`` is a Python-only injection point for hermetic tests and is not
    part of the Agent contract; the registered callable resolves the species,
    ranks exact-TaxID assemblies, and stores one canonical report artifact per
    candidate. A near relative is never substituted.
    """
    return genome_size_candidates(scientific_name, max_candidates=max_candidates)


def op_dedup_genomes(data: OrganelleData) -> OrganelleData:
    """Collapse records that describe the same genome, and pick a representative.

    A TRANSFORM operation must return a core object, so the clusters are written
    back into the container rather than handed out as a bare list. The original
    records are kept: dedup marks duplicates, it never deletes evidence.
    """
    records: list[dict[str, Any]] = list(data.payload.get("records", []) or [])
    clusters = dedup_genomes(records)

    payload = dict(data.payload)
    payload["clusters"] = [cluster.as_record() for cluster in clusters]
    payload["representatives"] = [str(cluster.representative["accession"]) for cluster in clusters]

    manifest = dict(payload.get("manifest", {}) or {})
    manifest["n_genomes_deduped"] = len(clusters)
    manifest["n_duplicates_removed"] = len(records) - len(clusters)
    # A RefSeq record whose curation link we could not parse. Silence here would
    # mean an unrecognised mirror and a genome counted twice.
    unlinked = unlinked_refseq_records(records)
    if unlinked:
        manifest["unlinked_refseq_records"] = unlinked
    payload["manifest"] = manifest

    return OrganelleData(
        modality=data.modality,
        artifacts=dict(data.artifacts),
        payload=payload,
    )


def op_ngdc_gwh(
    organelle: str,
    *,
    dest: str,
    taxon: str | None = None,
    formats: tuple[str, ...] = ("fasta",),
    with_metadata: bool = True,
    max_records: int | None = None,
) -> OrganelleData:
    """Fetch organelle genomes from NGDC/GWH — the ones NCBI does not have."""
    return fetch_ngdc(
        organelle=organelle,  # type: ignore[arg-type]
        dest=dest,
        taxon=taxon,
        formats=formats,
        with_metadata=with_metadata,
        max_records=max_records,
    )
