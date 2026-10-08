"""Fetch suite: deterministic retrieval of organelle records from NCBI.

Two channels, deliberately not one:

    ov.fetch.refseq_snapshot(organelle="plastid", dest="refseq/")   # deterministic
    ov.fetch.entrez_query(organelle="mitochondrion", dest="live/")  # live nuccore
    ov.fetch.fetch_accessions(accessions=[...], dest="set/")        # reproduce a dataset

Metadata is fetched and filtered before any sequence is downloaded, so records
that will be discarded never cross the wire (after gget virus, arXiv 2606.06749).
Searches run on NCBI's history server and are paged, so a whole-corpus pull is a
normal call rather than a special case.

Standard library only; ``urllib`` is imported lazily so that importing this
module never touches the network.
"""

from __future__ import annotations

# Complete the lightweight release catalog before importing fetch submodules.
# A cold import of any ``organelleverse.fetch.*`` path executes this package
# first; without this ordering, ``operations.catalog`` can re-enter a partially
# initialised ``fetch.operations`` module.
from .. import operations as _release_operations
from .cache import append_records, merge_with_baseline, read_baseline, read_jsonl, write_jsonl
from .dedup import GenomeCluster, dedup_genomes, quality_rank, select_representative
from .export import genbank_to_fasta, merge_metadata_csv, write_metadata_csv
from .filters import FetchFilters, apply_filters, count_ambiguous, is_refseq_accession
from .genbank_meta import enrich_with_genbank_metadata, parse_gbseq_xml
from .genes import build_gene_term, fetch_genes
from .genome_size import GenomeSizeCandidate, genome_size_candidates
from .gir import fetch_gir
from .imp import fetch_imp
from .manifest import FETCH_MANIFEST_SCHEMA_VERSION, accession_set_sha256, build_manifest
from .ncbi import (
    History,
    build_organelle_term,
    entrez_query,
    fetch_accessions,
    name_variants,
    refseq_snapshot,
)
from .ngdc import classify_folder, fetch_ngdc, gwh_assembly_metadata
from .nuclear import fetch_nuclear_genome, parse_assembly_reports
from .operations import (
    ENTREZ_QUERY_SPEC,
    FETCH_SPECS,
    REFSEQ_SNAPSHOT_SPEC,
    register_fetch_operations,
)
from .pgd import fetch_pgd
from .tair import fetch_tair

del _release_operations


__all__ = [
    "ENTREZ_QUERY_SPEC",
    "FETCH_MANIFEST_SCHEMA_VERSION",
    "FETCH_SPECS",
    "REFSEQ_SNAPSHOT_SPEC",
    "FetchFilters",
    "GenomeCluster",
    "GenomeSizeCandidate",
    "History",
    "accession_set_sha256",
    "append_records",
    "apply_filters",
    "build_gene_term",
    "build_manifest",
    "build_organelle_term",
    "classify_folder",
    "count_ambiguous",
    "dedup_genomes",
    "enrich_with_genbank_metadata",
    "entrez_query",
    "fetch_accessions",
    "fetch_genes",
    "fetch_gir",
    "fetch_imp",
    "fetch_ngdc",
    "fetch_nuclear_genome",
    "fetch_pgd",
    "fetch_tair",
    "genbank_to_fasta",
    "genome_size_candidates",
    "gwh_assembly_metadata",
    "is_refseq_accession",
    "merge_metadata_csv",
    "merge_with_baseline",
    "name_variants",
    "parse_assembly_reports",
    "parse_gbseq_xml",
    "quality_rank",
    "read_baseline",
    "read_jsonl",
    "refseq_snapshot",
    "register_fetch_operations",
    "select_representative",
    "write_jsonl",
    "write_metadata_csv",
]
