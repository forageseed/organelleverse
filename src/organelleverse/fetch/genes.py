"""Fetch gene records — the fragments that are *noise* when you want genomes.

``entrez_query`` drops gene and spacer records by default, because 77% of plant
mitochondrial records and ~97% of chloroplast records are fragments, and one
counted as a genome is a wrong sample size nobody notices.

But sometimes the fragment **is** the target: a barcode study wants *matK*, a
phylogeny wants *cox1*, an ERC run wants the nuclear proteins. So the same
machinery runs with the fragment filter inverted, and grouping switched off —
gene records are not molecules of a genome and must never be grouped into one.

Entrez indexes gene symbols in the ``[Gene]`` field:

    Oryza[Organism] AND matK[Gene] AND chloroplast[filter]   ->   530 records
    Oryza[Organism] AND cox1[Gene] AND mitochondrion[filter] ->    28 records

``organelle=None`` drops the organelle filter and searches nuclear genes too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from ..core.data import OrganelleData
from ..core.errors import OrganelleParameterError
from ._http import Transport, default_transport
from .filters import FetchFilters, apply_filters

__all__ = ["build_gene_term", "fetch_genes"]

GeneScope = Literal["mitochondrion", "plastid", "nuclear"]

_SCOPE_FILTER = {
    "mitochondrion": "mitochondrion[filter]",
    "plastid": "chloroplast[filter]",
    # Nuclear genes carry no organelle filter; excluding the organelles is what
    # makes the search nuclear.
    "nuclear": "NOT mitochondrion[filter] NOT chloroplast[filter]",
}


def build_gene_term(
    *,
    gene: str,
    taxon: str | None = None,
    scope: GeneScope = "plastid",
) -> str:
    """Build the Entrez term for a gene record search."""
    if not gene or not gene.strip():
        raise OrganelleParameterError(
            code="input.missing_gene",
            message="fetch_genes() needs a gene symbol, e.g. 'matK' or 'cox1'",
        )
    if scope not in _SCOPE_FILTER:
        raise OrganelleParameterError(
            code="input.unknown_scope",
            message=f"scope must be one of {sorted(_SCOPE_FILTER)}, got {scope!r}",
            details={"scope": scope},
        )

    parts: list[str] = []
    if taxon:
        parts.append(f"{taxon}[Organism]")
    parts.append(f"{gene.strip()}[Gene]")

    term = " AND ".join(parts)
    scope_clause = _SCOPE_FILTER[scope]
    # The nuclear clause is a NOT, which cannot be joined with AND.
    if scope_clause.startswith("NOT"):
        return f"{term} {scope_clause}"
    return f"{term} AND {scope_clause}"


def fetch_genes(
    *,
    gene: str,
    dest: str | Path,
    taxon: str | None = None,
    scope: GeneScope = "plastid",
    max_records: int | None = None,
    page_size: int = 500,
    min_length: int | None = None,
    max_length: int | None = None,
    transport: Transport | None = None,
) -> OrganelleData:
    """Fetch gene records for one gene symbol.

    Unlike :func:`entrez_query` this keeps fragments (they are the point) and
    does **not** group records into genomes — a *matK* record is not a molecule
    of anything.
    """
    from .manifest import build_manifest
    from .ncbi import (  # local import keeps the module dependency-light
        _as_data,
        _efetch_accessions,
        _esearch_history,
        _esummary_paged,
    )

    http = transport or default_transport()
    term = build_gene_term(gene=gene, taxon=taxon, scope=scope)
    history = _esearch_history(http, term=term)

    scope_record: dict[str, Any] = {
        "gene": gene,
        "taxon": taxon or "",
        "scope": scope,
        "min_length": min_length,
        "max_length": max_length,
    }

    def empty() -> OrganelleData:
        return _as_data(
            manifest=build_manifest(
                source="gene_records",
                organelle=scope if scope != "nuclear" else "nuclear",
                accessions=[],
                scope=scope_record,
                query=term,
                hits_before_filter=history.count,
                examined=0,
            ),
            artifact=None,
            # A gene record is not an organelle record. The spec declares
            # `gene_records`, and the object has to agree with the contract or
            # the contract is decoration.
            modality="gene_records",
        )

    if history.count == 0:
        return empty()

    wanted = history.count if max_records is None else min(max_records, history.count)
    records = _esummary_paged(http, history=history, limit=wanted, page_size=page_size)
    examined = len(records)

    # Keep the fragments. Length bounds still apply — a "gene" record spanning
    # 150 kb is a genome that mentions a gene, not a gene.
    records = apply_filters(
        records,
        FetchFilters(
            exclude_gene_records=False,
            min_length=min_length,
            max_length=max_length,
        ),
    )
    accessions = sorted({str(r["accession"]) for r in records})
    if not accessions:
        return empty()

    path = _efetch_accessions(
        http,
        accessions=accessions,
        dest=Path(dest),
        stem=f"{gene}_{scope}",
        page_size=page_size,
    )
    return _as_data(
        manifest=build_manifest(
            source="gene_records",
            organelle=scope if scope != "nuclear" else "nuclear",
            accessions=accessions,
            scope=scope_record,
            query=term,
            hits_before_filter=history.count,
            examined=examined,
        ),
        artifact=path,
        records=records,
        modality="gene_records",
    )
