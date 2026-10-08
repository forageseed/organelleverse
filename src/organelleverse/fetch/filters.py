"""Metadata filters — the stage that decides what is worth downloading.

Every predicate here is a pure function over one metadata record, so a filter
can be applied before a single base of sequence crosses the wire (the gget-virus
staging rule).

Two tiers of metadata exist, and a filter must not silently pass a record whose
field it cannot see:

* **esummary** (cheap, always fetched): accession, length, dates, status.
* **GenBank XML** (``genbank_metadata=True``): gene/protein counts, ambiguous
  base counts, collection date, country, submitter.

Asking for a GenBank-tier filter without GenBank-tier metadata raises rather
than quietly returning everything — a filter that silently does nothing is how
a wrong dataset ends up in a paper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from ..core.errors import OrganelleParameterError

__all__ = [
    "GENBANK_TIER_FIELDS",
    "FetchFilters",
    "apply_filters",
    "apply_filters_to_units",
    "check_min_max",
    "is_refseq_accession",
    "looks_like_gene_record",
    "parse_partial_date",
]

# Fields that only exist once GenBank XML metadata has been fetched.
GENBANK_TIER_FIELDS = frozenset(
    {
        "gene_count",
        "protein_count",
        "ambiguous_count",
        "collection_date",
        "geo_location",
        "submitter_country",
        "submitter_institution",
        "submitter_name",
        "isolate",
        "isolation_source",
        "has_proteins",
        "annotated",
    }
)

# RefSeq curated records carry these prefixes; everything else is a primary
# GenBank submission. The pair is the same genome twice — the core redundancy
# problem for organelles.
_REFSEQ_PREFIXES = ("NC_", "NW_", "NZ_", "NG_", "AC_")

_AMBIGUOUS = re.compile(r"[^ACGTacgt]")

# A record can be titled "complete sequence" and still be a 1 kb spacer. For
# Viridiplantae chloroplasts, 236,172 of 237,010 such records (99.6%) are under
# 10 kb: genes, introns and intergenic spacers, not genomes. Counting one as a
# genome is the same class of error as counting one chromosome as a genome — the
# number is wrong and nothing complains.
_GENE_RECORD = re.compile(
    r"\b("
    r"gene|genes|gene\s+for|cds|"
    r"spacer|intergenic|intron|"
    r"rrna|trna|ribosomal\s+rna|transfer\s+rna|"
    r"barcode|region"
    r")\b",
    re.IGNORECASE,
)
_GENOME_WORD = re.compile(r"\b(genome|chromosome|plasmid)\b", re.IGNORECASE)


def looks_like_gene_record(title: str) -> bool:
    """True when a title describes a gene or spacer rather than a genome.

    A title that names a genome or a chromosome wins outright — a multipartite
    molecule is a genome part, however small (Begonia's chromosome 9 is 2,354 bp,
    and it must survive). Only records that speak of genes, introns or spacers
    and never of a genome are rejected.
    """
    text = title or ""
    if _GENOME_WORD.search(text):
        return False
    return bool(_GENE_RECORD.search(text))


def is_refseq_accession(accession: str) -> bool:
    return accession.upper().startswith(_REFSEQ_PREFIXES)


def parse_partial_date(value: str, *, field_name: str = "date") -> date:
    """Parse ``2024``, ``2024/06``, ``2024-06-11`` — NCBI mixes all three.

    A partial date resolves to its earliest day, so a range check on a
    year-only record behaves predictably.
    """
    raw = value.strip().replace("-", "/")
    if not raw:
        raise OrganelleParameterError(
            code="input.bad_date",
            message=f"{field_name} is empty",
            details={"field": field_name},
        )
    parts = raw.split("/")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return date(year, month, day)
    except (ValueError, IndexError) as exc:
        raise OrganelleParameterError(
            code="input.bad_date",
            message=f"{field_name} is not a date NCBI would emit: {value!r}",
            details={"field": field_name, "value": value},
        ) from exc


def check_min_max(minimum: Any, maximum: Any, *, name: str, is_date: bool = False) -> None:
    """Reject an empty interval up front rather than returning zero records."""
    if minimum is None or maximum is None:
        return
    lo = parse_partial_date(minimum, field_name=f"min_{name}") if is_date else minimum
    hi = parse_partial_date(maximum, field_name=f"max_{name}") if is_date else maximum
    if lo > hi:
        raise OrganelleParameterError(
            code="input.empty_interval",
            message=f"min_{name} ({minimum}) is greater than max_{name} ({maximum})",
            details={"min": str(minimum), "max": str(maximum), "filter": name},
        )


@dataclass(frozen=True)
class FetchFilters:
    """Every filter gget virus offers that has an organelle meaning, plus ours.

    Size, in bp — plastomes run ~120-160 kb, plant mitogenomes ~200-700 kb, so
    length is the strongest cheap identity/quality signal available.
    """

    # -- esummary tier (free) ------------------------------------------------
    exclude_gene_records: bool = True
    min_length: int | None = None
    max_length: int | None = None
    min_release_date: str | None = None
    max_release_date: str | None = None
    refseq_only: bool = False
    source_database: str | None = None  # "refseq" | "genbank"
    live_only: bool = True  # drop suppressed/replaced records

    # -- GenBank tier (needs genbank_metadata=True) --------------------------
    min_gene_count: int | None = None
    max_gene_count: int | None = None
    min_protein_count: int | None = None
    max_protein_count: int | None = None
    max_ambiguous: int | None = None
    annotated_only: bool = False
    has_proteins: bool = False
    min_collection_date: str | None = None
    max_collection_date: str | None = None
    geo_location: str | None = None
    submitter_country: str | None = None
    submitter_institution: str | None = None
    submitter_name: str | None = None
    isolate: str | None = None
    isolation_source: str | None = None

    def __post_init__(self) -> None:
        check_min_max(self.min_length, self.max_length, name="length")
        check_min_max(self.min_gene_count, self.max_gene_count, name="gene_count")
        check_min_max(self.min_protein_count, self.max_protein_count, name="protein_count")
        check_min_max(
            self.min_release_date, self.max_release_date, name="release_date", is_date=True
        )
        check_min_max(
            self.min_collection_date,
            self.max_collection_date,
            name="collection_date",
            is_date=True,
        )
        if self.source_database and self.source_database not in ("refseq", "genbank"):
            raise OrganelleParameterError(
                code="input.bad_source_database",
                message=f"source_database must be 'refseq' or 'genbank', got {self.source_database!r}",
                details={"source_database": self.source_database},
            )

    def requires_genbank_metadata(self) -> bool:
        """True when any GenBank-tier filter is set."""
        return bool(
            self.min_gene_count is not None
            or self.max_gene_count is not None
            or self.min_protein_count is not None
            or self.max_protein_count is not None
            or self.max_ambiguous is not None
            or self.annotated_only
            or self.has_proteins
            or self.min_collection_date
            or self.max_collection_date
            or self.geo_location
            or self.submitter_country
            or self.submitter_institution
            or self.submitter_name
            or self.isolate
            or self.isolation_source
        )

    # Filters that are on by default. They are hygiene, not scope: reporting
    # them in the manifest would bury the filters the caller actually chose.
    _DEFAULT_ON = ("live_only", "exclude_gene_records")

    def active(self) -> dict[str, Any]:
        """The filters actually in force — goes into the manifest scope."""
        return {
            name: value
            for name, value in self.__dict__.items()
            if value not in (None, False) and not (name in self._DEFAULT_ON and value is True)
        }


def _matches_text(record_value: Any, wanted: str) -> bool:
    """Case-insensitive substring match, the way NCBI's own web filters behave."""
    return wanted.strip().lower() in str(record_value or "").lower()


def count_ambiguous(sequence: str) -> int:
    """Non-ACGT bases. Organelle assemblies routinely carry N-gaps."""
    return len(_AMBIGUOUS.findall(sequence))


def _passes(record: dict[str, Any], filters: FetchFilters) -> bool:
    # A gene or spacer record is not a genome, in either organelle. Reject it
    # before any other predicate, so it can never be counted, grouped, or
    # reported as one.
    if filters.exclude_gene_records and looks_like_gene_record(str(record.get("title", ""))):
        return False

    length = int(record.get("length") or 0)
    if filters.min_length is not None and length < filters.min_length:
        return False
    if filters.max_length is not None and length > filters.max_length:
        return False

    accession = str(record.get("accession", ""))
    is_refseq = is_refseq_accession(accession)
    if filters.refseq_only and not is_refseq:
        return False
    if filters.source_database == "refseq" and not is_refseq:
        return False
    if filters.source_database == "genbank" and is_refseq:
        return False

    if filters.live_only:
        status = str(record.get("status", "live")).lower()
        if status and status != "live":
            return False

    if filters.min_release_date or filters.max_release_date:
        raw = record.get("create_date") or record.get("update_date") or ""
        if not raw:
            return False
        released = parse_partial_date(str(raw), field_name="release_date")
        if filters.min_release_date and released < parse_partial_date(
            filters.min_release_date, field_name="min_release_date"
        ):
            return False
        if filters.max_release_date and released > parse_partial_date(
            filters.max_release_date, field_name="max_release_date"
        ):
            return False

    # -- GenBank tier --------------------------------------------------------
    if (
        filters.min_gene_count is not None
        and int(record.get("gene_count") or 0) < filters.min_gene_count
    ):
        return False
    if (
        filters.max_gene_count is not None
        and int(record.get("gene_count") or 0) > filters.max_gene_count
    ):
        return False
    if (
        filters.min_protein_count is not None
        and int(record.get("protein_count") or 0) < filters.min_protein_count
    ):
        return False
    if (
        filters.max_protein_count is not None
        and int(record.get("protein_count") or 0) > filters.max_protein_count
    ):
        return False
    if (
        filters.max_ambiguous is not None
        and int(record.get("ambiguous_count") or 0) > filters.max_ambiguous
    ):
        return False
    if filters.annotated_only and not record.get("annotated"):
        return False
    if filters.has_proteins and int(record.get("protein_count") or 0) <= 0:
        return False

    if filters.min_collection_date or filters.max_collection_date:
        raw = record.get("collection_date") or ""
        if not raw:
            return False
        collected = parse_partial_date(str(raw), field_name="collection_date")
        if filters.min_collection_date and collected < parse_partial_date(
            filters.min_collection_date, field_name="min_collection_date"
        ):
            return False
        if filters.max_collection_date and collected > parse_partial_date(
            filters.max_collection_date, field_name="max_collection_date"
        ):
            return False

    for wanted, key in (
        (filters.geo_location, "geo_location"),
        (filters.submitter_country, "submitter_country"),
        (filters.submitter_institution, "submitter_institution"),
        (filters.submitter_name, "submitter_name"),
        (filters.isolate, "isolate"),
        (filters.isolation_source, "isolation_source"),
    ):
        if wanted and not _matches_text(record.get(key), wanted):
            return False

    return True


def apply_filters_to_units(
    units: list[Any],
    filters: FetchFilters,
    *,
    have_genbank_metadata: bool = False,
) -> list[Any]:
    """Filter whole genomes, not molecules.

    A length floor of 200 kb is sane for a plant mitogenome and lethal to the
    44 kb third chromosome of the cucumber one. Size, gene count and protein
    count are therefore evaluated against the genome's totals; every other
    predicate is evaluated against its first molecule, where the organism and
    submission metadata live.
    """
    kept: list[Any] = []
    for unit in units:
        summary = dict(unit.molecules[0])
        summary["length"] = unit.total_length
        summary["gene_count"] = unit.gene_count
        summary["protein_count"] = unit.protein_count
        if apply_filters([summary], filters, have_genbank_metadata=have_genbank_metadata):
            kept.append(unit)
    return kept


def apply_filters(
    records: list[dict[str, Any]],
    filters: FetchFilters,
    *,
    have_genbank_metadata: bool = False,
) -> list[dict[str, Any]]:
    """Filter metadata records. Never silently ignores a filter it cannot apply.

    If a GenBank-tier filter is set but GenBank metadata was not fetched, this
    raises. Returning every record in that case would look like "the filter
    matched everything" — the failure mode that puts a wrong dataset in a paper.
    """
    if filters.requires_genbank_metadata() and not have_genbank_metadata:
        raise OrganelleParameterError(
            code="input.genbank_metadata_required",
            message=(
                "filters on gene/protein count, ambiguity, collection date, geography "
                "or submitter need GenBank metadata; pass genbank_metadata=True"
            ),
            details={"filters": sorted(filters.active())},
            suggested_action={"set": "genbank_metadata=True"},
        )
    return [record for record in records if _passes(record, filters)]
