"""The fetch manifest: provenance today, the benchmark substrate tomorrow.

``accession_set_sha256`` is the determinism primitive. It identifies a *set* of
records, so it is invariant to the order NCBI happened to return them in. A
pinned RefSeq release plus a query should always produce the same digest — that
equality is the whole benchmark, and unlike an annotation accuracy metric it has
no definitional ambiguity to argue about.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from ..core.errors import OrganelleParameterError

__all__ = [
    "FETCH_MANIFEST_SCHEMA_VERSION",
    "accession_set_sha256",
    "build_manifest",
]

FETCH_MANIFEST_SCHEMA_VERSION = "organelleverse.fetch.manifest.v1"


def accession_set_sha256(accessions: Sequence[str]) -> str:
    """SHA-256 over the sorted, de-duplicated accession set.

    Order-independent and duplicate-insensitive by construction: the digest
    identifies which records were retrieved, not how they were listed.
    """
    unique = sorted({a.strip() for a in accessions if a and a.strip()})
    if not unique:
        raise OrganelleParameterError(
            code="input.empty_accession_set",
            message="cannot hash an empty accession set",
            suggested_action={"check": "widen the query, or handle the zero-hit case"},
        )
    digest = hashlib.sha256()
    for accession in unique:
        digest.update(accession.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(
    *,
    source: Literal[
        "refseq_release",
        "entrez_query",
        "accession_list",
        "gene_records",
        "nuclear_assembly",
        "ngdc_gwh",
        "gir",
        "imp",
        "pgd",
        "tair",
    ],
    organelle: str,
    accessions: Sequence[str],
    scope: dict[str, Any],
    query: str = "",
    refseq_release: int | None = None,
    fetched_at: str | None = None,
    hits_before_filter: int | None = None,
    examined: int | None = None,
    names_tried: Sequence[str] | None = None,
    baseline_skipped: int = 0,
    missing: Sequence[str] | None = None,
    genomes: Sequence[dict[str, Any]] | None = None,
    incomplete_genomes: Sequence[dict[str, Any]] | None = None,
    sources_tried: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest payload carried on the returned ``OrganelleData``.

    ``scope`` is the scientific boundary of the claim (organelle, taxon,
    completeness, every active filter). A validation claim must always be stated
    against this scope — never as a bare "fetch is validated".

    The audit trail of the staging step is three numbers, and conflating them
    lies:

    * ``hits_before_filter`` — what the query matched in the database.
    * ``examined`` — how many of those we actually pulled metadata for
      (``max_records`` may have capped it far below the hit count).
    * ``n_records`` — how many survived the filters.

    ``n_filtered_out`` is therefore ``examined - n_records - baseline_skipped``,
    **not** ``hits_before_filter - n_records``. Reporting the latter would claim
    a filter rejected thousands of records it never looked at. ``truncated``
    says outright that the hit list was cut short.
    """
    ordered = sorted({a.strip() for a in accessions if a and a.strip()})
    manifest: dict[str, Any] = {
        "schema_version": FETCH_MANIFEST_SCHEMA_VERSION,
        "source": source,
        "organelle": organelle,
        "query": query,
        "scope": dict(scope),
        "n_records": len(ordered),
        "accessions": ordered,
        "accession_set_sha256": accession_set_sha256(ordered) if ordered else "",
        "fetched_at": fetched_at or datetime.now(UTC).isoformat(),
        "organelleverse_version": _organelleverse_version(),
    }
    if refseq_release is not None:
        manifest["refseq_release"] = refseq_release
    if hits_before_filter is not None:
        manifest["hits_before_filter"] = hits_before_filter
        looked_at = hits_before_filter if examined is None else examined
        manifest["examined"] = looked_at
        manifest["n_filtered_out"] = max(0, looked_at - len(ordered) - baseline_skipped)
        # Say it plainly when the hit list was cut short — a silent cap reads as
        # "this is everything there is".
        manifest["truncated"] = looked_at < hits_before_filter
    if names_tried and len(list(names_tried)) > 1:
        manifest["names_tried"] = list(names_tried)
    if baseline_skipped:
        manifest["baseline_skipped"] = baseline_skipped
    if missing:
        manifest["missing"] = list(missing)
    if genomes is not None:
        # n_records counts accessions; n_genomes counts genomes. A multipartite
        # mitogenome is several accessions and ONE genome, and reporting the
        # former as a sample size inflates every statistic downstream.
        manifest["n_genomes"] = len(genomes)
        manifest["multipartite_genomes"] = sum(1 for g in genomes if g.get("multipartite"))
    if incomplete_genomes:
        # A gap in the chromosome numbering means a molecule of this genome
        # exists and we do not have it. Never let that pass silently.
        manifest["incomplete_genomes"] = [
            {
                "unit_id": g["unit_id"],
                "organism": g.get("organism", ""),
                "have": g.get("accessions", []),
                "missing_chromosomes": g.get("missing_chromosomes", []),
            }
            for g in incomplete_genomes
        ]
    if sources_tried:
        manifest["sources_tried"] = list(sources_tried)
    return manifest


def _organelleverse_version() -> str:
    from .. import __version__

    return __version__
